import tensorflow as tf

import utils
from layers_utils import AffineProjectionPseudoCell
from autoregression_with_alternatives import AutoregressionWithAlternativePaths

NUM_UNITS = 30
NUM_LAYERS = 3
VOCAB_SIZE = 2196027 # glove vocab size +10 (which is reserved for special id)


class PredictNext(tf.nn.rnn_cell.RNNCell):
    def __init__(self, n_units, n_layers, probability_distribution_size, dtype=tf.float32, **kwargs):
        super(PredictNext, self).__init__(dtype=dtype, **kwargs)
        self.probability_distribution_size = probability_distribution_size
        self.n_units = n_units
        self.n_layers = n_layers

        # alternatively this could be done via projection_num atgument of last LSTM cell
        projection = AffineProjectionPseudoCell(self.n_units, self.probability_distribution_size, self.dtype)

        self.rnn_cell = tf.nn.rnn_cell.MultiRNNCell(
            [self._create_single_cell() for _ in range(self.n_layers)] + [projection],
            state_is_tuple=True)
        
    
    def call(self, input, state):
        """
        Args:
            input: Tensor of shape [batch_size, max_sequence_length, embeddings_size]
        """
        output, new_state = self.rnn_cell(input, state)
        assert utils.nested_tuple_apply(state, lambda x: x.dtype) == utils.nested_tuple_apply(new_state, lambda x: x.dtype)
        return output, new_state

    def _create_single_cell(self):
        return tf.nn.rnn_cell.LSTMCell(self.n_units, state_is_tuple=True)

    @property
    def state_size(self):
        return self.rnn_cell.state_size

    @property
    def output_size(self):
        return self.rnn_cell.output_size

    def zero_state(self, *a, **k):
        return self.rnn_cell.zero_state(*a, **k)
        

def language_model_input_dataset(raw_dataset, id_to_embedding_mapping):
    """`raw_dataset` should be non-batched, non-padded dataset of sequences of ids.
    Note:
        This function _doesn't_ add start or end marks
    Args:
        `id_to_embedding_mapping`: function that maps batch of ids to batch of corresponding embedding vectors"""
    def prepare_sequence_example(sequence):
        """
        Args:
            sequence: 1-D vector of word ids
        """
        length = tf.shape(sequence)[0]
        features, labels = {}, {}
        inputs_raw = sequence[:-1]
        features["length"] = length - 1
        features["inputs"] = id_to_embedding_mapping(inputs_raw)
        labels["targets"] = sequence[1:]
        return (features, labels)
    return raw_dataset.map(prepare_sequence_example)


def get_language_model_fn(vocab_size):
    def language_model_fn(features, labels, mode, params):
        # Args:
        #
        # features: This is the x-arg from the input_fn.
        # labels:   Its not used at all.
        # mode:     Either TRAIN, EVAL, or PREDICT
        # params:   User-defined hyper-parameters, currently `learning_rate` only.
        sentence, length = features["inputs"], features["length"]
        if not mode == tf.estimator.ModeKeys.PREDICT:
            targets = labels["targets"]

        predictor = PredictNext(NUM_UNITS, NUM_LAYERS, vocab_size)
        logits, state = tf.nn.dynamic_rnn(predictor, sentence, sequence_length=length, dtype=tf.float32)

        probabilities = tf.nn.softmax(logits=logits)

        if mode == tf.estimator.ModeKeys.PREDICT:
            spec = tf.estimator.EstimatorSpec(mode=mode, predictions=probabilities)
        else:
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets,
                                                                        logits=logits)

            loss = tf.reduce_mean(cross_entropy)
            optimizer = tf.train.AdamOptimizer(learning_rate=params["learning_rate"])

            # Get the TensorFlow op for doing a single optimization step.
            train_op = optimizer.minimize(
                loss=loss, global_step=tf.train.get_global_step())

            predicted_word = tf.argmax(probabilities, 2)

            metrics = {
                    #"cross_entropy": cross_entropy,
                    "accuracy": tf.metrics.accuracy(targets, predicted_word)
                }

            # Wrap all of this in an EstimatorSpec.
            spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op,
                eval_metric_ops=metrics
                )
            
        return spec
    return language_model_fn



def get_autoregressor_model_fn(vocab_size, id_to_embedding_mapping):
    def autoregressor_model_fn(features, labels, mode, params):
        # Args:
        #
        # features: This is the x-arg from the input_fn.
        # labels:   Its not used at all.
        # mode:     Either TRAIN, EVAL, or PREDICT
        # params:   User-defined hyper-parameters, currently `learning_rate` only.

        predictor = PredictNext(NUM_UNITS, NUM_LAYERS, vocab_size)

        if mode == tf.estimator.ModeKeys.PREDICT:
            initial_inputs = features["inputs"] # expect tensor of a shape [batch_size] with first elemetns
            length = features["length"] # expect tensor of a shape [batch_size] with number of elements to generate
            length = tf.squeeze(length)
            if len(length.shape) != 0:
                print("warning, multiple output sequence length for one batch not supported")
                length = length[0]
            autoregressor = AutoregressionWithAlternativePaths(
                conditional_probability_model=predictor,
                number_of_alternatives=params["number_of_alternatives"],
                number_of_elements_to_generate=length,
                index_in_probability_distribution_to_id_mapping=tf.identity,
                id_to_embedding_mapping=id_to_embedding_mapping,
                conditional_probability_model_initial_state=None,
                probability_masking_layer=None)
            paths, paths_probabilities = autoregressor.call_with_initial_sequence(initial_inputs)
            spec = tf.estimator.EstimatorSpec(
                mode=mode, 
                predictions={"paths": paths, "paths_probabilities": paths_probabilities})
        else:
            sentence, length = features["inputs"], features["length"]
            targets = labels["targets"]
            logits, state = tf.nn.dynamic_rnn(predictor, sentence, sequence_length=length, dtype=tf.float32)
            probabilities = tf.nn.softmax(logits=logits)
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets,
                                                                        logits=logits)

            loss = tf.reduce_mean(cross_entropy)
            optimizer = tf.train.AdamOptimizer(learning_rate=params["learning_rate"])

            # Get the TensorFlow op for doing a single optimization step.
            train_op = optimizer.minimize(
                loss=loss, global_step=tf.train.get_global_step())

            predicted_word = tf.argmax(probabilities, 2)

            metrics = {
                    #"cross_entropy": cross_entropy,
                    "accuracy": tf.metrics.accuracy(targets, predicted_word)
                }

            # Wrap all of this in an EstimatorSpec.
            spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op,
                eval_metric_ops=metrics
                )
            
        return spec
    return autoregressor_model_fn
