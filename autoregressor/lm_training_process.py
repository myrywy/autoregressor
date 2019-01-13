import logging

import tensorflow as tf

from lstm_lm import PredictNext
from utils import maybe_inject_hparams

class LanguageModel:
    FLOAT_TYPE = tf.float32
    def __init__(self, features, labels, mode, vocabulary_generalized, hparams):
        self.hparams = hparams
        
        self.time_major_optimization = None
        self.mask_padding_cost = None
        self.dynamic_rnn_swap_memory = None
        self.rnn_num_units = None
        self.rnn_num_layers = None
        self.rnn_last_layer_num_units = None
        self.learning_rate = None
        self.predict_top_k = None
        self.words_as_text_preview = None
        
        maybe_inject_hparams(
            self, 
            hparams, 
            [
                "time_major_optimization", 
                "mask_padding_cost", 
                "dynamic_rnn_swap_memory",
                "rnn_num_units",
                "rnn_num_layers",
                "rnn_last_layer_num_units",
                "learning_rate",
                "predict_top_k",
                "words_as_text_preview"
                ]
            )
        
        # TODO: this should probably be factored out of this class
        self.vocabulary_generalized = vocabulary_generalized
        self.vocab_size = vocabulary_generalized.vocab_size() - 3 # THIS -3 is for regression test only; there was bug in previous version so...

        self.mode = mode

        self.inputs, self.targets, self.lengths = self.unpack_nested_example(features, labels)
        self.inputs, self.targets = self.maybe_transpose_batch_time(self.inputs, self.targets)

        self.graph_build = False
    
    def unpack_nested_example(self, features, labels):
        inputs = features["inputs"]
        lengths = features["length"]
        targets = labels["targets"] if labels is not None else None
        return inputs, targets, lengths

    def build_graph(self):
        self.logits, _ = self.unrolled_rnn(self.inputs, self.lengths)
        self.probabilities = self.probabilities_fn(self.logits)
        self.predictions_ids = self.make_predictions(self.logits)
        tf.summary.tensor_summary("top_k_predictions_ids", self.predictions_ids)
        if self.words_as_text_preview:
            self.predictions_tokens = self.predictions_ids_to_tokens(self.predictions_ids)
            tf.summary.text("top_k_predictions", self.predictions_tokens)
        if self.mode != tf.estimator.ModeKeys.PREDICT:
            self.loss = self.loss_fn(self.targets, self.logits, self.lengths)
            self.train_op = self.optimize(self.loss)
            self.position_of_true_word = self.score_of_true_word_fn(self.logits, self.targets)
            tf.summary.tensor_summary("position_of_true_word", self.position_of_true_word)
            self.mean_position_of_true_word = tf.reduce_mean(self.position_of_true_word)
            tf.summary.scalar("mean_position_of_true_word", self.position_of_true_word)
            tf.summary.scalar("batch_perplexity", self.perplexity_from_loss(self.loss))
            self.set_metrics()
        self.graph_build = True

    def perplexity_from_loss(self, loss):
        return tf.exp(loss)

    def set_metrics(self):
        self.position_of_true_word_metric = tf.metrics.mean(self.position_of_true_word)
        sequence_mask = self.cost_mask(self.lengths, self.max_length(), self.time_major_optimization)
        cross_entropy = self.cross_entropy_fn(self.targets, self.logits, self.lengths)
        self.log_perplexity_metric, self.log_perplexity_metric_update_op = tf.metrics.mean(cross_entropy, weights=sequence_mask)
        self.perplexity_metric = tf.exp(self.log_perplexity_metric)
        self.metrics = {
            "position_of_true_word": self.position_of_true_word_metric,
            "log_perplexity": (self.log_perplexity_metric, self.log_perplexity_metric_update_op),
            "perplexity": (self.perplexity_metric, None),
        }        

    def score_of_true_word_fn(self, logits, targets):
        flatten_logits = tf.reshape(logits, (-1,tf.shape(logits)[-1]))
        flatten_targets = tf.reshape(targets, (-1,))
        flatten_logits_of_true_words = tf.map_fn((lambda x: tf.gather(x[0], x[1])), [flatten_logits, flatten_targets], dtype=logits.dtype)
        logits_of_true_words = tf.reshape(flatten_logits_of_true_words, tf.shape(targets))
        
        is_score_higher_eq_than_true_one = tf.greater_equal(logits, tf.expand_dims(logits_of_true_words,-1))
        n_score_higher_eq_than_true_one = tf.reduce_sum(tf.cast(is_score_higher_eq_than_true_one, tf.int8), axis=-1)

        return n_score_higher_eq_than_true_one - 1 # -1 comes from the fact that true index has score equal to true index so it counts to higher-equal

    def probabilities_fn(self, logits):
        return tf.nn.softmax(logits=logits)

    def make_predictions(self, logits):
        top_predicted_words_ids = tf.nn.top_k(logits, self.predict_top_k)
        return top_predicted_words_ids.indices

    def predictions_ids_to_tokens(self, top_predicted_words_ids):
        if self.vocabulary_generalized is None:
            raise ValueError("vocabulary_generalized is None but predictions_ids_to_tokens called")
        original_shape = tf.shape(top_predicted_words_ids)
        flatten_top_predicted_words_ids = tf.reshape(top_predicted_words_ids, (-1,))
        flatten_top_predicted_words = self.vocabulary_generalized.generalized_id_to_token()(flatten_top_predicted_words_ids)
        top_predicted_words = tf.reshape(flatten_top_predicted_words, shape=original_shape)
        return top_predicted_words

    def loss_fn(self, targets, logits, lengths):
        cross_entropy = self.cross_entropy_fn(targets, logits, lengths)
        cross_entropy_sum = tf.reduce_sum(cross_entropy)

        mask = self.cost_mask(lengths, self.max_length(), self.time_major_optimization)
        mask_sum = tf.reduce_sum(mask)
        
        mean_of_non_masked_values = cross_entropy_sum / mask_sum
        return mean_of_non_masked_values

    def cross_entropy_fn(self, targets, logits, lengths):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets,
                                                                        logits=logits)
        if self.mask_padding_cost:
            mask = self.cost_mask(lengths, self.max_length(), self.time_major_optimization)
            cross_entropy = cross_entropy * mask
        return cross_entropy

    def cost_mask(self, lengths, max_length, time_major):
        mask = tf.sequence_mask(lengths, max_length, self.FLOAT_TYPE)
        if time_major:
            mask = tf.transpose(mask, (1, 0))
        return mask

    def maybe_transpose_batch_time(self, inputs, targets):
        if self.time_major_optimization:
            inputs = tf.transpose(inputs, (1, 0, 2))
            targets = tf.transpose(targets, (1, 0))
        return inputs, targets

    def unrolled_rnn(self, inputs, lengths):
        logits, state = tf.nn.dynamic_rnn(self.cell(), inputs,
                                            sequence_length=lengths,
                                            dtype=self.FLOAT_TYPE,
                                            time_major=self.time_major_optimization,
                                            swap_memory=self.dynamic_rnn_swap_memory)
        return logits, state

    def cell(self):
        cell = PredictNext(
            self.rnn_num_units, self.rnn_num_layers, self.vocab_size, last_layer_num_units=self.rnn_last_layer_num_units)
        return cell

    def optimize(self, loss):
        optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate)

        train_op = optimizer.minimize(
            loss=loss, global_step=tf.train.get_global_step())
        
        return train_op

    def max_length(self):
        time_dim = 0 if self.time_major_optimization else 1
        return tf.shape(self.targets)[time_dim]

    def estimator_spec(self):
        if not self.graph_build:
            self.build_graph()
        predictions = {"probabilities": self.probabilities, "predictions_ids": self.predictions_ids}
        if self.words_as_text_preview:
            predictions["predictions_tokens"] = self.predictions_tokens
        if self.mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                self.mode,
                predictions=predictions,
            )
        else:
            return tf.estimator.EstimatorSpec(
                self.mode,
                predictions=predictions,
                loss=self.loss,
                train_op=self.train_op,
                eval_metric_ops=self.metrics,
            )


class LanguageModelCallable:
    def __init__(self, vocabulary_generalized, hparams):
        self.hparams = hparams
        self.vocabulary_generalized = vocabulary_generalized

    def __call__(self, features, labels, mode, params):
        model = LanguageModel(features, labels, mode, self.vocabulary_generalized, self.hparams)
        return model.estimator_spec()




from itertools import islice
import datetime
import pickle

from vocabularies_preprocessing.glove300d import Glove300
from corpora_preprocessing.simple_examples import SimpleExamplesCorpus, DatasetType
from lm_input_data_pipeline import LmInputDataPipeline
from train_lm_on_simple_examples_and_glove import read_dataset_from_dir 
from generalized_vocabulary import GeneralizedVocabulary, SpecialUnit



def eval_lm_on_cached_simple_examples_with_glove_check(data_dir, model_dir, subset, hparams, take_first_n=20):
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

    '''def model_function(features, labels, mode, params):
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
    '''
    params = {"learning_rate": hparams.learning_rate, "number_of_alternatives": 1}
    #config=tf.estimator.RunConfig(session_config=tf.ConfigProto(log_device_placement=False))
    #config=tf.estimator.RunConfig(session_config=tf.ConfigProto())

    specials = [SpecialUnit.OUT_OF_VOCABULARY, SpecialUnit.START_OF_SEQUENCE, SpecialUnit.END_OF_SEQUENCE]
    generalized = GeneralizedVocabulary(glove, specials)
    
    with tf.device("/device:CPU:0"):
        model = LanguageModelCallable(generalized, hparams)
    config=tf.estimator.RunConfig()

    with tf.device("/device:CPU:0"):
        estimator = tf.estimator.Estimator(
            model, params=params, model_dir=model_dir, config=config)
    t1 = datetime.datetime.now()
    with tf.device("/device:CPU:0"):
        predictions = estimator.predict(create_input)
    t2 = datetime.datetime.now()
    predictions = islice(predictions, take_first_n)
    with open("rtest_expected.pickle", "rb") as expected_file:
        expected_predictions = pickle.load(expected_file)
    for prediction, expected in zip(predictions, expected_predictions):
        print(prediction)
        assert (prediction["predictions_ids"][:,0]==expected["predicted_word_id"]).all()
        assert (prediction["predictions_tokens"][:,0]==expected["predicted_word"]).all()
        assert (expected["probabilities"] == expected["probabilities"]).all()
    print("start:", t1)
    print("stop:", t2)
    print("duration:", t2-t1)