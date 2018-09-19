import tensorflow as tf


def naive_lookup_op(keys, values, first_dim_is_batch=False):
    """op that for given key vecor returns corresponding vector from values

    keys - tensor, at least one dimensional,
    velues - tensor such that velues[i] wil be return if a tensor equal to keys[i] is passed to the op,

    If a key that is equal to no entry in passed, nobody knows what happens ;D"""
    def single_element_lookup_op(entry):
        with tf.name_scope("naive_lookup_single"):
            eq = tf.equal(keys, entry)
            if len(eq.shape) > 1:
                for dim in range(1, len(eq.shape)):
                    eq = tf.reduce_all(eq, axis=1, name="dim_{}_reduce_all".format(dim))
            index = tf.argmax(tf.cast(eq, tf.int32))
            return values[index]
    if first_dim_is_batch:
        return lambda batch: tf.map_fn(single_element_lookup_op, batch, dtype=values.dtype)
    else:
        return single_element_lookup_op


def mock_model(hisory_to_probability_mapping, first_dim_is_batch=False):
    keys = [*hisory_to_probability_mapping.keys()]
    values = tf.constant([hisory_to_probability_mapping[key] for key in keys])
    keys = tf.constant(keys)

    def mock_model_fn(input, state):
        history_length, history = state

        # THIS IS DIRTY HACK to make tensorflow believe that history is a vector of size <batch size>
        #if first_dim_is_batch:
        #    print(history_length)
        #    print(history)
        #    history_length_vector = history_length
        #    history_length = history_length[0]

        if first_dim_is_batch:
            history_now = tf.concat((history[:,0:history_length], tf.expand_dims(input,1), history[:,history_length + 1:]), axis=1, name="concat_past_future")
        else:
            history_now = tf.concat((history[0:history_length], [input], history[history_length + 1:]), axis=0, name="concat_past_future")
        
        # THIS IS DIRTY HACK to make tensorflow believe that history is a vector of size <batch size>
        #if first_dim_is_batch:
        #    history_length = history_length_vector

        new_state = (history_length + 1, history_now)
        output = naive_lookup_op(keys, values, first_dim_is_batch=first_dim_is_batch)(history_now)
        return output, new_state
    
    return mock_model_fn

class MockModelLayer(tf.nn.rnn_cell.RNNCell):
    def __init__(self, hisory_to_probability_mapping, history_entry_dims=(), first_dim_is_batch=True):
        super(MockModelLayer, self).__init__()
        self.history_size = max(len(history) for history in hisory_to_probability_mapping.keys())
        self.output_distribution_size = max(len(dist_size) for dist_size in hisory_to_probability_mapping.values())
        assert self.history_size == \
            min(len(history) for history in hisory_to_probability_mapping.keys()), \
            "History vectors must be the same size in every examle" 
        assert self.output_distribution_size == \
            min(len(dist_size) for dist_size in hisory_to_probability_mapping.values()), \
            "Distribution size must be the same size in every examle"
        self._history_entry_dims = history_entry_dims
        self.layer_function = mock_model(hisory_to_probability_mapping, first_dim_is_batch=first_dim_is_batch)


    #def compute_output_shape(self, input_shape):
    #    return (input_shape[0], self.output_distribution_size)

    @property
    def state_size(self):
        return (tf.TensorShape(()), tf.TensorShape([self.history_size]))

    @property
    def output_size(self):
        return self.output_distribution_size

    def zero_state(self, batch_size, dtype):
        return (tf.constant(0), tf.zeros(shape=(batch_size, self.history_size, *self._history_entry_dims), dtype=dtype))

    def call(self, input, state):
        return self.layer_function(input, state)