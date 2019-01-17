import tensorflow as tf

import utils
from layers_utils import AffineProjectionPseudoCell
from autoregression_with_alternatives import AutoregressionWithAlternativePaths
from element_probablity_mask import ElementProbabilityMasking

NUM_UNITS = 50
NUM_LAYERS = 3


class PredictNext(tf.nn.rnn_cell.RNNCell):
    rnn_type_names = {
        None: tf.nn.rnn_cell.LSTMCell,
        "lstm": tf.nn.rnn_cell.LSTMCell,
        "lstm_block_cell": tf.contrib.rnn.LSTMBlockCell,
        "CudnnCompatibleLSTMCell": tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell
    }

    def __init__(self, n_units, n_layers, probability_distribution_size, dtype=tf.float32, last_layer_num_units=None, rnn_cell_type=None, **kwargs):
        super(PredictNext, self).__init__(dtype=dtype, **kwargs)
        self.probability_distribution_size = probability_distribution_size
        self.n_units = n_units
        self.n_layers = n_layers
        self.last_layer_n_units = last_layer_num_units
        self.rnn_cell_type = PredictNext.rnn_type_names[rnn_cell_type]

        # alternatively this could be done via projection_num atgument of last LSTM cell
        
        if self.last_layer_n_units is None:
            projection = AffineProjectionPseudoCell(
                self.n_units, self.probability_distribution_size, self.dtype)
            self.rnn_cell = tf.nn.rnn_cell.MultiRNNCell(
                [self._create_single_cell()
                 for _ in range(self.n_layers)] + [projection],
                state_is_tuple=True)
        else:
            projection = AffineProjectionPseudoCell(
                self.last_layer_n_units, self.probability_distribution_size, self.dtype)
            self.rnn_cell = tf.nn.rnn_cell.MultiRNNCell(
                [self._create_single_cell() for _ in range(self.n_layers-1)] +
                [self._create_last_cell()] +
                [projection],
                state_is_tuple=True)

    def call(self, input, state):
        """
        Args:
            input: Tensor of shape [batch_size, max_sequence_length, embeddings_size]
        """
        output, new_state = self.rnn_cell(input, state)
        assert utils.nested_tuple_apply(
            state, lambda x: x.dtype) == utils.nested_tuple_apply(new_state, lambda x: x.dtype)
        return output, new_state

    def _create_single_cell(self):
        return self.rnn_cell_type(self.n_units)

    def _create_last_cell(self):
        return self.rnn_cell_type(self.last_layer_n_units)

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
        logits, state = tf.nn.dynamic_rnn(
            predictor, sentence, sequence_length=length, dtype=tf.float32)

        probabilities = tf.nn.softmax(logits=logits)

        if mode == tf.estimator.ModeKeys.PREDICT:
            spec = tf.estimator.EstimatorSpec(
                mode=mode, predictions=probabilities)
        else:
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets,
                                                                           logits=logits)

            loss = tf.reduce_mean(cross_entropy)
            optimizer = tf.train.AdamOptimizer(
                learning_rate=params["learning_rate"])

            # Get the TensorFlow op for doing a single optimization step.
            train_op = optimizer.minimize(
                loss=loss, global_step=tf.train.get_global_step())

            predicted_word = tf.argmax(probabilities, 2)

            metrics = {
                # "cross_entropy": cross_entropy,
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


def get_autoregressor_model_fn(
        vocab_size,
        id_to_embedding_mapping,
        mask_allowables=None,
        time_major_optimization=False,
        predict_as_pure_lm=False,
        hparams=None):
    def autoregressor_model_fn(features, labels, mode, params):
        # Args:
        #
        # features: This is the x-arg from the input_fn.
        # labels:   Its not used at all.
        # mode:     Either TRAIN, EVAL, or PREDICT
        # params:   User-defined hyper-parameters, currently `learning_rate` only.

        if hparams is not None:
            num_units = hparams.rnn_num_units
            num_layers = hparams.rnn_num_layers
            last_layer_num_units = hparams.rnn_last_layer_num_units
            mask_padding_cost = hparams.mask_padding_cost
            predictor = PredictNext(num_units, num_layers, vocab_size,
                                    last_layer_num_units=last_layer_num_units, rnn_cell_type=hparams.rnn_layer)
        else:
            num_units = NUM_UNITS
            num_layers = NUM_LAYERS
            last_layer_num_units = None
            mask_padding_cost = False
            predictor = PredictNext(
                num_units, num_layers, vocab_size, last_layer_num_units=last_layer_num_units)

        if mode == tf.estimator.ModeKeys.PREDICT and not predict_as_pure_lm:
            # expect tensor of a shape [batch_size] with first elemetns
            initial_inputs = features["inputs"]
            # expect tensor of a shape [batch_size] with number of elements to generate
            length = features["length"]
            length = tf.squeeze(length)
            if len(length.shape) != 0:
                print(
                    "warning, multiple output sequence length for one batch not supported")
                length = length[0]
            if mask_allowables is not None:
                mask = ElementProbabilityMasking(
                    mask_allowables,
                    vocab_size,
                    0,
                    vocab_size,
                    tf.identity)
            else:
                mask = None
            autoregressor = AutoregressionWithAlternativePaths(
                conditional_probability_model=predictor,
                number_of_alternatives=params["number_of_alternatives"],
                number_of_elements_to_generate=length,
                index_in_probability_distribution_to_id_mapping=tf.identity,
                id_to_embedding_mapping=id_to_embedding_mapping,
                conditional_probability_model_initial_state=None,
                probability_masking_layer=mask,
                probability_aggregation_op=tf.add)
            paths, paths_probabilities = autoregressor.call_with_initial_sequence(
                initial_inputs)
            spec = tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={"paths": paths, "paths_probabilities": paths_probabilities})
        elif mode == tf.estimator.ModeKeys.PREDICT and predict_as_pure_lm:
            sentence, length = features["inputs"], features["length"]
            if time_major_optimization:
                sentence = tf.transpose(sentence, (1, 0, 2))
            logits, state = tf.nn.dynamic_rnn(predictor, sentence,
                                              sequence_length=length,
                                              dtype=tf.float32,
                                              time_major=time_major_optimization,
                                              swap_memory=True)
            probabilities = tf.nn.softmax(logits=logits)
            if time_major_optimization:
                probabilities = tf.transpose(probabilities, (1, 0, 2))
            predicted_word_id = tf.argmax(probabilities, 2)
            spec = tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={
                    "predicted_word_id": predicted_word_id,
                    "probabilities": probabilities
                })

        else:
            sentence, length = features["inputs"], features["length"]
            targets = labels["targets"]
            cost_mask = tf.sequence_mask(length, tf.shape(targets)[1], tf.float32)
            if time_major_optimization:
                sentence = tf.transpose(sentence, (1, 0, 2))
                targets = tf.transpose(targets, (1, 0))
                cost_mask = tf.transpose(cost_mask, (1, 0))
            logits, state = tf.nn.dynamic_rnn(predictor, sentence,
                                              sequence_length=length,
                                              dtype=tf.float32,
                                              time_major=time_major_optimization,
                                              swap_memory=True)

            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets,
                                                                           logits=logits)

            if mask_padding_cost == True:
                cross_entropy = cross_entropy * cost_mask

            loss = tf.reduce_mean(cross_entropy)
            optimizer = tf.train.AdamOptimizer(
                learning_rate=params["learning_rate"])

            # Get the TensorFlow op for doing a single optimization step.
            train_op = optimizer.minimize(
                loss=loss, global_step=tf.train.get_global_step())

            probabilities = tf.nn.softmax(logits=logits)
            predicted_word = tf.argmax(probabilities, 2)

            metrics = {
                # "cross_entropy": cross_entropy,
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
