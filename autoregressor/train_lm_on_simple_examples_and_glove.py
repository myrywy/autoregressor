import os
import re
import argparse
import datetime
from pathlib import Path
from itertools import count, repeat, islice
from collections import namedtuple
import contextlib

import tensorflow as tf
import numpy as np
from pytest import approx

from utils import without
from generalized_vocabulary import SpecialUnit
from vocabularies_preprocessing.glove300d import Glove300
from corpora_preprocessing.simple_examples import SimpleExamplesCorpus, DatasetType
from lm_input_data_pipeline import LmInputDataPipeline
from lstm_lm import get_autoregressor_model_fn
from hparams import hparams

tf.logging.set_verbosity(tf.logging.DEBUG)


try:
    if not os.environ["CUDA_VISIBLE_DEVICES"]:
        USE_GPU = False
    else:
        USE_GPU = True
except KeyError:
    USE_GPU = True


TEST_SERIALIZATION = False

class InitializeVocabularyHook(tf.train.SessionRunHook):
    def __init__(self, vocabulary):
        self._vocabulary = vocabulary

    def after_create_session(self, session, coord):
        self._vocabulary.initialize_embeddings_in_graph(
            tf.get_default_graph(), session)


class FullLogHook(tf.train.SessionRunHook):
    def before_run(self, run_context):
        fetches = run_context.original_args.fetches
        feed_dict = run_context.original_args.feed_dict
        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        options.report_tensor_allocations_upon_oom = True
        return tf.train.SessionRunArgs(fetches, feed_dict, options)


def read_dataset_from_dir(data_dir, only_named_subset, embedding_size):
    """
    Note: side-effect - this function creates tensors in default Graph
    """
    files_paths = list_dataset_files_in_directory(data_dir, only_named_subset)
    files_paths = tf.convert_to_tensor(files_paths, dtype=tf.string)
    dataset = read_dataset_from_files(files_paths, embedding_size)
    return dataset


def list_dataset_files_in_directory(data_dir, only_named_subset=None):
    data_dir = Path(data_dir)
    data_files = [file_path for file_path in data_dir.iterdir() if re.match(r"\w+\.\d+\.tfrecords$",file_path.name)]
    data_files = sorted(data_files, key=lambda p: p.name)
    if only_named_subset is not None:
        data_files = [file_path for file_path in data_files if file_path.name.split(".")[0] == only_named_subset]
    data_files = [str(file_path) for file_path in data_files]
    return data_files


def read_dataset_from_files(input_paths, embedding_size):
    """
    Note: side-effect - may create tensors in default Graph
    """
    input_paths = tf.convert_to_tensor(input_paths)

    feature_description = {
        'inputs': tf.FixedLenSequenceFeature([embedding_size], tf.float32, allow_missing=True, default_value=0.0),
        'length': tf.FixedLenFeature([], tf.int64, default_value=0),
        'targets': tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True, default_value=0),
    }

    def parse_tf_record(example_proto):
        return tf.parse_single_example(example_proto, feature_description)

    def restore_structure(example):
        features = {"inputs": example["inputs"], "length": example["length"]}
        labels = {"targets": example["targets"]}
        return features, labels

    dataset = tf.data.TFRecordDataset(input_paths)
    return dataset.map(parse_tf_record).map(restore_structure)


def prepare_training_dataset(ouput_path):
    """This will transform input corpus into language model training examples with embeddings vectors as inputs and save it to disk.
    Expect HUGE dataset in terms of occupied space."""
    if TEST_SERIALIZATION:
        test_examples = []
    ouput_path = Path(ouput_path)
    glove = Glove300()

    def create_input():
        simple_examples = SimpleExamplesCorpus()
        train_data = simple_examples.get_tokens_dataset(DatasetType.TRAIN)
        input_pipe = LmInputDataPipeline(glove, None)
        return input_pipe.load_data(train_data)
    dataset = create_input()

    def make_tf_record_example(features, labels) -> tf.train.SequenceExample:
        feature_inputs = tf.train.Feature(float_list=tf.train.FloatList(value=features["inputs"].reshape(-1)))
        feature_length = tf.train.Feature(int64_list=tf.train.Int64List(value=[features["length"]]))
        feature_targets = tf.train.Feature(int64_list=tf.train.Int64List(value=labels["targets"]))
        feature_dict = {"inputs": feature_inputs, "length": feature_length, "targets": feature_targets}
        features = tf.train.Features(feature=feature_dict)
        example = tf.train.Example(features=features)
        return example
    
    def max_length_condition(max_length):
        def check_length(features, labels):
            return tf.less_equal(features["length"], max_length)
        return check_length
    dataset = dataset.filter(max_length_condition(40))

    it = dataset.make_initializable_iterator()
    next = it.get_next()

    EXAMPLES_PER_FILE = 2000

    with tf.Session() as sess:
        sess.run(tf.tables_initializer())
        glove.initialize_embeddings_in_graph(tf.get_default_graph(), sess)
        sess.run(it.initializer)
        for i in count(1):
            dataset_filename = str(ouput_path/"train.{:0=10}.tfrecords".format(i))
            writer = tf.python_io.TFRecordWriter(dataset_filename)
            try:
                for _ in range(EXAMPLES_PER_FILE):
                    features, labels = sess.run(next)
                    if TEST_SERIALIZATION:
                        test_examples.append((features, labels))
                    example = make_tf_record_example(features, labels)
                    writer.write(example.SerializeToString())
            except tf.errors.OutOfRangeError:
                break
            writer.close()

    if TEST_SERIALIZATION:
        embedding_size = LmInputDataPipeline(glove, None)._vocab_generalized.vector_size()
        records_dataset = read_dataset_from_files([dataset_filename], embedding_size=embedding_size)
        it = records_dataset.make_initializable_iterator()
        next_record = it.get_next()
        with tf.Session() as sess:
            sess.run(it.initializer)
            for expected_features, expected_labels in test_examples:
                actual_features, actual_labels = sess.run(next_record)
                assert (actual_features["inputs"] == expected_features["inputs"]).all()
                assert (actual_features["length"] == expected_features["length"]).all()
                assert actual_labels["targets"] == approx(expected_labels["targets"])
        

def train_lm_on_simple_examples_with_glove(model_dir):
    glove = Glove300()

    def create_input():
        simple_examples = SimpleExamplesCorpus()
        train_data = simple_examples.get_tokens_dataset(
            DatasetType.TRAIN).repeat().shuffle(1000, seed=0)
        input_pipe = LmInputDataPipeline(glove, 8)
        return input_pipe.load_data(train_data)

    def model_function(features, labels, mode, params):
        input_pipe = LmInputDataPipeline(glove)
        vocab_size = glove.vocab_size()
        id_to_embeding_fn = input_pipe.get_id_to_embedding_mapping()
        with tf.device(device_assignment_function):
            concrete_model_fn = get_autoregressor_model_fn(
                vocab_size, id_to_embeding_fn)
            estimator_spec = concrete_model_fn(features, labels, mode, params)
        training_hooks = [InitializeVocabularyHook(glove)]
        estimator_spec_with_hooks = tf.estimator.EstimatorSpec(
            mode=estimator_spec.mode,
            loss=estimator_spec.loss,
            train_op=estimator_spec.train_op,
            eval_metric_ops=estimator_spec.eval_metric_ops,
            predictions=estimator_spec.predictions,
            training_hooks=training_hooks
        )
        return estimator_spec_with_hooks

    params = {"learning_rate": 0.05, "number_of_alternatives": 1}
    estimator = tf.estimator.Estimator(
        model_function, params=params, model_dir=model_dir)
    t1 = datetime.datetime.now()
    estimator.train(create_input, max_steps=4)
    t2 = datetime.datetime.now()
    print("start:", t1)
    print("stop:", t2)
    print("duration:", t2-t1)


def device_assignment_function(node):
    print(node.name)
    print(node.type)
    if not USE_GPU:
        print("USE_GPU == False")
        print("assigned:", "/device:CPU:0")
        return "/device:CPU:0"
    if node.name.find("rnn/while/rnn/predict_next/predict_next/multi_rnn_cell/cell_3/cell_3/pseudo_rnn_cell/xw_plus_b/MatMul") != -1:
        return "/device:GPU:0"
    if node.name.split("/")[-1] == "transpose":
        for input in node.inputs:
            print("input:", input)
            if input.shape == tf.TensorShape(None):
                continue
            for size in input.shape:
                if size > 2000:
                    print("input too large")
                    print("assigned:", "/device:CPU:0")
                    return "/device:CPU:0"

    for node_output in node.outputs:
        print("ouput:", node_output)
        if node_output.shape == tf.TensorShape(None):
            continue
        for size in node_output.shape:
            if size > 2000:
                print("ouput too large")
                print("assigned:", "/device:CPU:0")
                return "/device:CPU:0"
    print("assigned:", "/device:GPU:0")
    return "/device:GPU:0"



CREATE_RTEST_INPUT = False
def train_lm_on_cached_simple_examples_with_glove(data_dir, model_dir, hparams):
    glove = Glove300(dry_run=False)
    BATCH_SIZE = 5

    def create_input():
        input_pipe = LmInputDataPipeline(glove, 5)
        embedding_size = LmInputDataPipeline(glove, None)._vocab_generalized.vector_size()
        train_data = read_dataset_from_dir(data_dir, DatasetType.TRAIN, embedding_size)
        train_data = train_data.repeat().shuffle(1000, seed=0)
        train_data = input_pipe.padded_batch(train_data, BATCH_SIZE)
        return train_data

    def model_function(features, labels, mode, params):
        input_pipe = LmInputDataPipeline(glove)
        vocab_size = glove.vocab_size()
        embedding_size = input_pipe._vocab_generalized.vector_size()
        id_to_embeding_fn = input_pipe.get_id_to_embedding_mapping() if mode == tf.estimator.ModeKeys.PREDICT else lambda x: tf.zeros((tf.shape(x), embedding_size), tf.float32)
        with tf.device(device_assignment_function) if hparams.size_based_device_assignment else without:
            concrete_model_fn = get_autoregressor_model_fn(
                    vocab_size, id_to_embeding_fn, time_major_optimization=True, hparams=hparams)
            estimator_spec = concrete_model_fn(features, labels, mode, params)
        if hparams.write_target_text_to_summary:
            words_shape = tf.shape(labels["targets"])
            to_vocab_id = input_pipe._vocab_generalized.generalized_id_to_vocab_id()
            to_word = glove.id_to_word_op()
            flat_targets = tf.reshape(labels["targets"], shape=[-1])
            flat_targets_words = to_word(to_vocab_id(flat_targets))
            targets_words = tf.reshape(flat_targets_words, shape=words_shape)
            tf.summary.text("targets_words", targets_words)
        training_hooks = []
        if mode == tf.estimator.ModeKeys.PREDICT:
            training_hooks.append(InitializeVocabularyHook(glove))
        if hparams.profiler:
            training_hooks.append(tf.train.ProfilerHook(output_dir=model_dir, save_secs=30, show_memory=True))
            training_hooks.append(FullLogHook())
        estimator_spec_with_hooks = tf.estimator.EstimatorSpec(
            mode=estimator_spec.mode,
            loss=estimator_spec.loss,
            train_op=estimator_spec.train_op,
            eval_metric_ops=estimator_spec.eval_metric_ops,
            predictions=estimator_spec.predictions,
            training_hooks=training_hooks
        )
        return estimator_spec_with_hooks

    params = {"learning_rate": hparams.learning_rate, "number_of_alternatives": 1}
    if CREATE_RTEST_INPUT:
        dataset = create_input()
        it = dataset.make_initializable_iterator()
        next_example = it.get_next()
        with tf.Session() as sess:
            sess.run(tf.tables_initializer())
            sess.run(tf.global_variables_initializer())
            sess.run(it.initializer)
            expected = []
            for i in range(3000):
                expected.append(sess.run(next_example))
            with open("retest_expected.pickle", "wb") as rtest_expected:
                import pickle
                pickle.dump(expected, rtest_expected)
        return
    #config=tf.estimator.RunConfig(session_config=tf.ConfigProto(log_device_placement=False))
    #config=tf.estimator.RunConfig(session_config=tf.ConfigProto())
    config=tf.estimator.RunConfig()
    estimator = tf.estimator.Estimator(
        model_function, params=params, model_dir=model_dir, config=config)
    t1 = datetime.datetime.now()
    estimator.train(create_input, max_steps=hparams.max_training_steps)
    t2 = datetime.datetime.now()
    print("start:", t1)
    print("stop:", t2)
    print("duration:", t2-t1)



def eval_lm_on_cached_simple_examples_with_glove(data_dir, model_dir, subset, hparams, take_first_n=20):
    glove = Glove300(dry_run=True)
    BATCH_SIZE = 5

    def create_input():
        input_pipe = LmInputDataPipeline(glove, 5)
        embedding_size = LmInputDataPipeline(glove, None)._vocab_generalized.vector_size()
        train_data = read_dataset_from_dir(data_dir, subset, embedding_size)
        if take_first_n is not None:
            train_data = train_data.take(take_first_n)
        train_data = input_pipe.padded_batch(train_data, BATCH_SIZE)
        return train_data

    def model_function(features, labels, mode, params):
        input_pipe = LmInputDataPipeline(glove)
        vocab_size = glove.vocab_size()
        embedding_size = input_pipe._vocab_generalized.vector_size()
        id_to_embeding_fn = input_pipe.get_id_to_embedding_mapping() if mode == tf.estimator.ModeKeys.PREDICT else lambda x: tf.zeros((tf.shape(x), embedding_size), tf.float32)
        #with tf.device(device_assignment_function) if hparams.size_based_device_assignment else without:
        with tf.device("/device:CPU:0"):
            concrete_model_fn = get_autoregressor_model_fn(
                    vocab_size, id_to_embeding_fn, time_major_optimization=True, predict_as_pure_lm=True, hparams=hparams)
            estimator_spec = concrete_model_fn(features, labels, mode, params)
        training_hooks = []
        predictions = estimator_spec.predictions
        if mode == tf.estimator.ModeKeys.PREDICT:
            training_hooks.append(InitializeVocabularyHook(glove))

            predicted_ids = predictions["predicted_word_id"]
            words_shape = tf.shape(predicted_ids)
            to_vocab_id = input_pipe._vocab_generalized.generalized_id_to_vocab_id()
            to_word = glove.id_to_word_op()
            predicted_ids = tf.reshape(predicted_ids, shape=[-1])
            predicted_words = to_word(to_vocab_id(predicted_ids))
            predicted_words = tf.reshape(predicted_words, shape=words_shape)
            predictions["predicted_word"] = predicted_words
        if hparams.profiler:
            training_hooks.append(tf.train.ProfilerHook(output_dir=model_dir, save_secs=30, show_memory=True))
            training_hooks.append(FullLogHook())
        estimator_spec_with_hooks = tf.estimator.EstimatorSpec(
            mode=estimator_spec.mode,
            loss=estimator_spec.loss,
            train_op=estimator_spec.train_op,
            eval_metric_ops=estimator_spec.eval_metric_ops,
            predictions=estimator_spec.predictions,
            training_hooks=training_hooks
        )
        return estimator_spec_with_hooks

    params = {"learning_rate": hparams.learning_rate, "number_of_alternatives": 1}
    #config=tf.estimator.RunConfig(session_config=tf.ConfigProto(log_device_placement=False))
    #config=tf.estimator.RunConfig(session_config=tf.ConfigProto())
    config=tf.estimator.RunConfig()
    estimator = tf.estimator.Estimator(
        model_function, params=params, model_dir=model_dir, config=config)
    t1 = datetime.datetime.now()
    predictions = estimator.predict(create_input)
    t2 = datetime.datetime.now()
    predictions = islice(predictions, take_first_n)
    for prediction in predictions:
        print(prediction)
    print("start:", t1)
    print("stop:", t2)
    print("duration:", t2-t1)



def disambiguation_with_glove(input_sentence, model_dir, hparams):
    """Input sentence is a list of lists of possible words at a given position"""
    glove = Glove300()
    BATCH_SIZE = 1


    def create_input():
        def data_gen():
            yield ({"inputs": np.array(
                            [[
                                LmInputDataPipeline(glove, None)._vocab_generalized.get_special_unit_id(SpecialUnit.START_OF_SEQUENCE)
                            ]],
                            dtype=np.int32
                            ), 
                    "length": len(input_sentence)}, 
                    np.array([0])
                    )
        
        
        data = tf.data.Dataset.from_generator(data_gen, 
            output_types=({"inputs": tf.int32, "length": tf.int32},tf.int32),
            output_shapes=({"inputs": (1,1,), "length": ()},(1,)))
        return data

    def model_function(features, labels, mode, params):
        input_pipe = LmInputDataPipeline(glove)
        vocab_size = glove.vocab_size()
        embedding_size = input_pipe._vocab_generalized.vector_size()
        id_to_embeding_fn = input_pipe.get_id_to_embedding_mapping() if mode == tf.estimator.ModeKeys.PREDICT else lambda x: tf.zeros((tf.shape(x), embedding_size), tf.float32)
        #with tf.device(device_assignment_function) if hparams.size_based_device_assignment else without:
        with tf.device("/device:CPU:0"):
            concrete_model_fn = get_autoregressor_model_fn(
                    vocab_size, 
                    id_to_embeding_fn, 
                    time_major_optimization=True, 
                    predict_as_pure_lm=False, 
                    mask_allowables=input_sentence,
                    hparams=hparams)
            estimator_spec = concrete_model_fn(features, labels, mode, params)
        training_hooks = []
        

        
        to_restore = tf.contrib.framework.get_variables_to_restore()
        predictions = estimator_spec.predictions
        if mode == tf.estimator.ModeKeys.PREDICT:
            training_hooks.append(InitializeVocabularyHook(glove))


            predicted_ids = tf.cast(predictions["paths"], dtype=tf.int64)
            words_shape = tf.shape(predicted_ids)
            to_vocab_id = input_pipe._vocab_generalized.generalized_id_to_vocab_id()
            to_word = glove.id_to_word_op()
            predicted_ids = tf.reshape(predicted_ids, shape=[-1])
            predicted_words = to_word(to_vocab_id(predicted_ids))
            predicted_words = tf.reshape(predicted_words, shape=words_shape)
            predictions["predicted_words"] = predicted_words
        if hparams.profiler:
            training_hooks.append(tf.train.ProfilerHook(output_dir=model_dir, save_secs=30, show_memory=True))
            training_hooks.append(FullLogHook())
        estimator_spec_with_hooks = tf.estimator.EstimatorSpec(
            mode=estimator_spec.mode,
            loss=estimator_spec.loss,
            train_op=estimator_spec.train_op,
            eval_metric_ops=estimator_spec.eval_metric_ops,
            predictions=estimator_spec.predictions,
            training_hooks=training_hooks
        )
        return estimator_spec_with_hooks

    params = {"learning_rate": hparams.learning_rate, "number_of_alternatives": 5}
    #config=tf.estimator.RunConfig(session_config=tf.ConfigProto(log_device_placement=False))
    #config=tf.estimator.RunConfig(session_config=tf.ConfigProto())
    config=tf.estimator.RunConfig()
    estimator = tf.estimator.Estimator(
        model_function, params=params, model_dir=model_dir, config=config)
    t1 = datetime.datetime.now()
    predictions = estimator.predict(create_input)
    t2 = datetime.datetime.now()
    predictions = islice(predictions, 1)
    for prediction in predictions:
        print(prediction)
    print("start:", t1)
    print("stop:", t2)
    print("duration:", t2-t1)
    return prediction


def load_pseudowords_corpus(input_file_name, take_first_n=1):
    sentences = []
    with open(input_file_name) as f:
        input_it = f if take_first_n is None else islice(f, take_first_n)
        for sentence in input_it:
            sentences.append(sentence.strip().split(" "))
    inputs, references = [], []
    for annotated_sentence in sentences:
        input, reference = [], []
        for annotated_word in annotated_sentence:
            word, annotation = annotated_word.split("|")
            input.append(word)
            reference.append(annotation)
        inputs.append(input)
        references.append(reference)
    return inputs, references

WordDisambiguationResult = namedtuple("WordDisambiguationResult", ["input", "hypothesis", "reference"])
SentenceDisambiguationResults = namedtuple("SentenceDisambiguationResults", ["words", "total", "ambiguous", "correct", "correct_ambiguous_only"])

def disambiguation_check(input_file_name, model_dir, hparams):
    inputs, references = load_pseudowords_corpus(input_file_name)
    allowables = disambiguation_preprocessing(inputs)
    predictions = [disambiguation_with_glove(allowables[0], model_dir, hparams)]
    results = []
    for prediction, input, reference, allowable in zip(predictions,inputs,references,allowables):
        result_sentence = []
        correct_in_sentence = 0
        correct_ambiguous_in_sentence = 0
        total_in_sentence = len(input)
        ambiguous_in_sentence = 0
        for predicted_meaning, input_word, reference_meaning, possible_meanings_ids in zip(prediction["predicted_words"][0][1:], input, reference, allowable):
            predicted_meaning = predicted_meaning.decode()
            r = WordDisambiguationResult(input_word, predicted_meaning, reference_meaning)
            result_sentence.append(r)
            was_ambiguous = len(possible_meanings_ids) != 1
            if was_ambiguous:
                ambiguous_in_sentence += 1
            if predicted_meaning == reference_meaning:
                correct_in_sentence += 1
                if was_ambiguous:
                    correct_ambiguous_in_sentence += 1
        sentence_results = SentenceDisambiguationResults(result_sentence, total_in_sentence, ambiguous_in_sentence, correct_in_sentence, correct_ambiguous_in_sentence)
        results.append(sentence_results)
    return results


def print_disambiguation_results(disambituation_results):
    OKGREEN = '\033[92m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    for sentence in disambituation_results:
        print()
        tokens = []
        for word in sentence.words:
            ok = "==" if word.hypothesis == word.reference else "!="
            if word.hypothesis == word.reference:
                if "^" in word.input:
                    color = OKGREEN
                else:
                    color = ""
            else:
                color = FAIL
            tokens.append("{}{}|H:{}{}R:{}{}".format(color, word.input, word.hypothesis, ok, word.reference,ENDC))
        print("\n".join(tokens))
        print()
        print("total: {}\ncorrect rate: {}\ncorrect disambiguations rate: {}".format(sentence.total, sentence.correct/sentence.total, sentence.correct_ambiguous_only/sentence.ambiguous))


def disambiguation_preprocessing(inputs):
    glove = Glove300()
    input_pipe = LmInputDataPipeline(glove)
    t_words = tf.placeholder(dtype=tf.string)
    t_vocab_ids = glove.word_to_id_op()(t_words)
    t_genralized_ids = input_pipe._vocab_generalized.vocab_id_to_generalized_id()(t_vocab_ids)

    meanings_all = set()
    for sentence in inputs:
        for word in sentence:
            for meaning in word.split("^"):
                meanings_all.add(meaning)
    meanings_all = list(meanings_all)

    with tf.Session() as sess:
        sess.run(tf.tables_initializer())
        sess.run(tf.global_variables_initializer())
        glove.after_session_created_hook_fn(sess)
        ids_all = sess.run(t_genralized_ids, feed_dict={t_words: meanings_all})

    mapping = {meaning: id for meaning, id in zip(meanings_all, ids_all)}

    sentences_as_ids = []
    for sentence in inputs:
        sentence_as_ids = []
        for word in sentence:
            allowables = []
            for meaning in word.split("^"):
                allowables.append(mapping[meaning])
            sentence_as_ids.append(allowables)
        sentences_as_ids.append(sentence_as_ids)
    
    return sentences_as_ids
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode")
    parser.add_argument("model_dir")
    parser.add_argument("--cached_dataset_dir")
    parser.add_argument('--hparams', type=str,
                    help='Comma separated list of "name=value" pairs.')
    args = parser.parse_args()
    if args.mode == "train":
        if args.hparams:
            hparams.parse(args.hparams)
        
        if args.cached_dataset_dir is None:
            train_lm_on_simple_examples_with_glove(args.model_dir)
        else:
            with open(Path(args.model_dir)/"hparams.json", "wt") as params_file:
                print(hparams.to_json(), file=params_file)
            train_lm_on_cached_simple_examples_with_glove(args.cached_dataset_dir, args.model_dir, hparams)
    elif args.mode == "prepare":
        prepare_training_dataset(args.model_dir)
    elif args.mode == "predict":
        if args.hparams:
            hparams.parse(args.hparams)
        eval_lm_on_cached_simple_examples_with_glove(args.cached_dataset_dir, args.model_dir, "train", hparams)
    elif args.mode == "disambiguate":
        if args.hparams:
            hparams.parse(args.hparams)
        #sentences_allowables = disambiguation_preprocessing("./data/simple_examples/check/pseudowords-0.8_0.2-20190106/ptb.check.txt")
        #print(sentences_allowables)
        #predictions = disambiguation_with_glove(sentences_allowables[0], args.model_dir, hparams)
        #print(predictions)
        disambiguation_results = disambiguation_check(
            "./data/simple_examples/check/pseudowords-0.8_0.2-20190106/ptb.check.txt", 
            args.model_dir, 
            hparams)
        print_disambiguation_results(disambiguation_results)
    else:
        print("wrong mode:", args.mode)
