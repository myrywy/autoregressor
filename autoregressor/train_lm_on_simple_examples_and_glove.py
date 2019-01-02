import os
import re
import argparse
import datetime
from pathlib import Path
from itertools import count

import tensorflow as tf
from pytest import approx

from vocabularies_preprocessing.glove300d import Glove300
from corpora_preprocessing.simple_examples import SimpleExamplesCorpus, DatasetType
from lm_input_data_pipeline import LmInputDataPipeline
from test_lstm_model.lstm_lm import get_autoregressor_model_fn
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


def train_lm_on_cached_simple_examples_with_glove(data_dir, model_dir, hparams):
    glove = Glove300(dry_run=True)
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
        #with tf.device(device_assignment_function):
        concrete_model_fn = get_autoregressor_model_fn(
                vocab_size, id_to_embeding_fn, time_major_optimization=True, hparams=hparams)        #
        estimator_spec = concrete_model_fn(features, labels, mode, params)          #
        training_hooks = []
        if mode == tf.estimator.ModeKeys.PREDICT:
            training_hooks.append(InitializeVocabularyHook(glove))
        #training_hooks.append(tf.train.ProfilerHook(output_dir=model_dir, save_secs=30, show_memory=True))
        #training_hooks.append(FullLogHook())
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
    estimator.train(create_input, max_steps=hparams.max_training_steps)
    t2 = datetime.datetime.now()
    print("start:", t1)
    print("stop:", t2)
    print("duration:", t2-t1)


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
    else:
        print("wrong mode:", args.mode)
