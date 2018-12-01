import tensorflow as tf

from layers_utils import AffineProjectionPseudoCell

NUM_UNITS = 300
NUM_LAYERS = 3
VOCAB_SIZE = 2196027 # glove vocab size +10 (which is reserved for special id)


class PredictNext(tf.keras.layers.Layer):
    def __init__(self, n_units, n_layers, probability_distribution_size, dtype=tf.float32, **kwargs):
        super(PredictNext, self).__init__(**kwargs)
        self.probability_distribution_size = probability_distribution_size
        self.dtype = dtype
        self.n_units = n_units
        self.n_layers = n_layers
    
    def call(self, input, sequence_length=None):
        """
        Args:
            input: Tensor of shape [batch_size, max_sequence_length, embeddings_size]
        """
        projection = AffineProjectionPseudoCell(self.n_units, self.probability_distribution_size, self.dtype)

        rnn_cell = tf.nn.rnn_cell.MultiRNNCell(
            [self._create_single_cell() for _ in range(self.n_layers)] + [projection]
            )
        
        # TODO: Custom initial state?
        logits = tf.nn.dynamic_rnn(rnn_cell, input, sequence_length=sequence_length)

        return logits

    def _create_single_cell(self):
        return tf.nn.rnn_cell.LSTMCell(self.n_units)
        

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


def language_model_fn(features, labels, mode, params):
    # Args:
    #
    # features: This is the x-arg from the input_fn.
    # labels:   Its not used at all.
    # mode:     Either TRAIN, EVAL, or PREDICT
    # params:   User-defined hyper-parameters, currently `learning_rate` only.
    sentence, length = features["inputs"], features["length"]
    targets = labels["targets"]

    predictor = PredictNext(NUM_UNITS, NUM_LAYERS, VOCAB_SIZE)
    logits = predictor(sentence, length)
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

        metrics = {
                "cross_entropy": cross_entropy,
            }

        # Wrap all of this in an EstimatorSpec.
        spec = tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=metrics)
        
    return spec