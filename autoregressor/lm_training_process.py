import logging
import datetime

import tensorflow as tf

from lstm_lm import PredictNext
from utils import maybe_inject_hparams, without


from tensorflow.contrib.cudnn_rnn import CudnnLSTM 

logger = logging.getLogger(__name__)

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
        self.size_based_device_assignment = None
        self.device = None
        self.use_cudnn_rnn = None
        
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
                "words_as_text_preview",
                "size_based_device_assignment",
                "device",
                "use_cudnn_rnn"
                ]
            )
        
        # TODO: this should probably be factored out of this class
        self.vocabulary_generalized = vocabulary_generalized
        self.vocab_size = vocabulary_generalized.vocab_size() if vocabulary_generalized is not None else None # THIS -3 is for regression test only; there was bug in previous version so...

        self.mode = mode

        self.inputs, self.targets, self.lengths = self.unpack_nested_example(features, labels)

        with self.get_device_context_manager():
            self.inputs, self.targets = self.maybe_transpose_batch_time(self.inputs, self.targets)

        self.graph_build = False
    
    def unpack_nested_example(self, features, labels):
        inputs = features["inputs"]
        lengths = features["length"]
        targets = labels["targets"] if labels is not None else None
        return inputs, targets, lengths

    def build_graph(self):
        with self.get_device_context_manager():
            self.logits, _ = self.unrolled_rnn(self.inputs, self.lengths)
            self.probabilities = self.probabilities_fn(self.logits)
            self.predictions_ids = self.make_predictions(self.logits)
            if self.words_as_text_preview:
                self.predictions_tokens = self.predictions_ids_to_tokens(self.predictions_ids)
            if self.time_major_optimization:
                self.probabilities = tf.transpose(self.probabilities, (1,0,2))
                self.predictions_ids = tf.transpose(self.predictions_ids, (1,0,2))
                if self.words_as_text_preview:
                    self.predictions_tokens = tf.transpose(self.predictions_tokens, (1,0,2))
            self.create_summaries_predicted()
            if self.mode != tf.estimator.ModeKeys.PREDICT:
                self.loss = self.loss_fn(self.targets, self.logits, self.lengths)
                self.train_op = self.optimize(self.loss)
                self.position_of_true_word = self.score_of_true_word_fn(self.logits, self.targets)
                tf.summary.text("position_of_true_word", tf.as_string(self.position_of_true_word))
                self.mean_position_of_true_word = tf.reduce_mean(
                        tf.to_float(
                            self.position_of_true_word
                        )
                    )
                tf.summary.scalar("mean_position_of_true_word", self.mean_position_of_true_word)
                tf.summary.scalar("batch_perplexity", self.perplexity_from_loss(self.loss))
                self.set_metrics()
            self.graph_build = True

    def create_summaries_predicted(self):
        for i in range(self.predict_top_k):
            tf.summary.tensor_summary("{}-th_predictions_ids".format(i+1), self.predictions_ids[:,:,i])
        
        if self.words_as_text_preview:
            for i in range(self.predict_top_k):
                tf.summary.text("{}-th_predictions".format(i+1), self.predictions_tokens[:,:,i])


    def get_device_context_manager(self):
        if self.size_based_device_assignment:
            raise NotImplementedError
        if self.device == "GPU":
            return tf.device("/device:GPU:0")
        if self.device == "CPU":
            return tf.device("/device:CPU:0")
        if self.device == "" or self.device is None:
            return without
        raise RuntimeError("Invalid device parameter '{}'".format(self.device))

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
        }        

    def score_of_true_word_fn(self, logits, targets):
        flatten_logits = tf.reshape(logits, (-1,tf.shape(logits)[-1]))
        flatten_targets = tf.reshape(targets, (-1,))
        flatten_logits_of_true_words = tf.map_fn((lambda x: tf.gather(x[0], x[1])), [flatten_logits, flatten_targets], dtype=logits.dtype)
        logits_of_true_words = tf.reshape(flatten_logits_of_true_words, tf.shape(targets))
        

        # Beware! This may not look as the most neat way to count values meeting a condition along a dimension but:
        # Size of "distribution over vocabulory" dimension may be huge and that takes a lot of memory to be comuted at once
        # (btw. it may cause overflow if one uses little int type as int16).
        # On the other hand resulting tensor is relatively small so computation of boolean values and reduction  
        # is performed in parts.
        def distribution_to_correct_guess_score(logits_local, logits_of_true_words_local):
            is_score_higher_eq_than_true_one = tf.greater_equal(logits_local, tf.expand_dims(logits_of_true_words_local,-1))
            return tf.reduce_sum(tf.cast(is_score_higher_eq_than_true_one, tf.int64), axis=-1, name="count_meeting_condition_reduction")

        if self.device != "GPU":
            max_prallel_reductions = 5

            n_score_higher_eq_than_true_one = tf.map_fn(
                lambda x: distribution_to_correct_guess_score(x[0], x[1]), 
                (logits, logits_of_true_words), 
                parallel_iterations=max_prallel_reductions, 
                dtype=tf.int64,
                name="mapped_n_score_higher_eq_than_true_one",
                back_prop=False,
                swap_memory=False,
                infer_shape=True)

        else:
            n_score_higher_eq_than_true_one = distribution_to_correct_guess_score(logits, logits_of_true_words)

        n_score_higher_eq_than_true_one = tf.identity(n_score_higher_eq_than_true_one, name="mapped_n_score_higher_eq_than_true_one")
        

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
            if targets is not None:
                targets = tf.transpose(targets, (1, 0))
        return inputs, targets

    def unrolled_rnn(self, inputs, lengths):
        if not self.use_cudnn_rnn:
            logits, state = tf.nn.dynamic_rnn(self.cell(), inputs,
                                                sequence_length=lengths,
                                                dtype=self.FLOAT_TYPE,
                                                time_major=self.time_major_optimization,
                                                swap_memory=self.dynamic_rnn_swap_memory)
        else:
            rnn = CudnnLSTM(self.rnn_num_layers, self.rnn_num_units)
            from layers_utils import AffineProjectionLayer
            proj = AffineProjectionLayer(self.rnn_num_units, self.vocab_size, self.FLOAT_TYPE)
            out, state = rnn(inputs)
            logits = proj(out)
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
                predictions=predictions if self.mode == tf.estimator.ModeKeys.EVAL else None,
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
from generalized_vocabulary import GeneralizedVocabulary, SpecialUnit, LMGeneralizedVocabulary
from lm_training_data import LanguageModelTrainingData
from vocabulary_factory import get_vocabulary


def eval_lm_on_cached_simple_examples_with_glove_check(data_dir, model_dir, subset, hparams, take_first_n=20):
    vocabulary = get_vocabulary(hparams.vocabulary_name)
    
    data = LanguageModelTrainingData(
        vocabulary_name=hparams.vocabulary_name, 
        corpus_name=hparams.corpus_name, 
        cached_data_dir=data_dir, 
        batch_size=hparams.batch_size, 
        shuffle_examples_buffer_size=None, 
        hparams=hparams)
    
    def create_input():
        return data.load_training_data()

    generalized = LMGeneralizedVocabulary(vocabulary)
    
    with tf.device("/device:CPU:0"):
        model = LanguageModelCallable(generalized, hparams)
        
        config=tf.estimator.RunConfig()

        estimator = tf.estimator.Estimator(
            model, model_dir=model_dir, config=config)
        predictions = estimator.predict(create_input)
    predictions = islice(predictions, take_first_n)
    return predictions

def train_and_eval(data_dir, model_dir, hparams):
    vocabulary = get_vocabulary(hparams.vocabulary_name)

    data = LanguageModelTrainingData(
        vocabulary_name=hparams.vocabulary_name, 
        corpus_name=hparams.corpus_name, 
        cached_data_dir=data_dir, 
        batch_size=hparams.batch_size, 
        shuffle_examples_buffer_size=hparams.shuffle_examples_buffer_size, 
        hparams=hparams)

    def create_input():
        return data.load_training_data()

    generalized = LMGeneralizedVocabulary(vocabulary)

    config = tf.estimator.RunConfig(
            save_summary_steps=500,
            save_checkpoints_secs=30*60,
            #save_checkpoints_steps=2,
            session_config=None,
            keep_checkpoint_max=10,
            keep_checkpoint_every_n_hours=10000,
            log_step_count_steps=500,
        )
    model = LanguageModelCallable(generalized, hparams)
    estimator = tf.estimator.Estimator(model, model_dir=model_dir, config=config)

    t1 = datetime.datetime.now()
    estimator.train(create_input, max_steps=hparams.max_training_steps)
    t2 = datetime.datetime.now()

    logger.info("start: {}".format(t1))
    logger.info("stop: {}".format(t2))
    logger.info("duration: {}".format(t2-t1))


if __name__ == "__main__":
    import os
    import argparse
    from utils import now_time_stirng
    from hparams import hparams

    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("cached_dataset_dir")
    parser.add_argument('--hparams', type=str,
                    help='Comma separated list of "name=value" pairs.')
    args = parser.parse_args()

    tf_log = logging.getLogger('tensorflow')
    tf_log.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    fh = logging.FileHandler(os.path.join(args.model_dir, 'tensorflow.log'))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    tf_log.addHandler(fh)

    if args.hparams:
        hparams.parse(args.hparams)

    with open(os.path.join(args.model_dir, 'hparams_{}.log'.format(now_time_stirng())), "wt") as hparams_log:
        hparams_log.write(hparams.to_json())
    logger.info("Running with parameters: {}".format(hparams.to_json()))
    train_and_eval(args.cached_dataset_dir, args.model_dir, hparams)
