import tensorflow as tf

def rnn_cell_extension(layer):
    """Decorator wrapping regular layer (that takes just one input as call argument) into rnn cell.
    Layer of returned type transforms input into output calling wrapped layer but returns state with no modification."""
    class PseudoRnnCell(tf.nn.rnn_cell.RNNCell):
        def __init__(self, *a, **k):
            super(PseudoRnnCell, self).__init__()
            self.wrapped_layer = layer(*a, **k)
            try:
                self.input_spec = self.wrapped_layer.input_spec
            except AttributeError:
                pass

        def call(self, input, state):
            """ output <- wrapped_layer(input); new_state <- state"""
            return self.wrapped_layer.call(input), state

        def zero_state(self, *a, **k):
            return ()
            
        @property
        def output_size(self):
            return self.wrapped_layer.output_size
        
        @property
        def state_size(self):
            return ()

    return PseudoRnnCell


class AffineProjectionLayer(tf.keras.layers.Layer):
    """Layer projecting vectors of size `input_size` to vectors `output_size` by learnable affine transformation."""
    def __init__(self, input_size, output_size, dtype, w_initializer=None, b_initializer=None):
        super(AffineProjectionLayer, self).__init__()
        self._input_size = input_size
        self._output_size = output_size
        self._dtype = dtype
        self.w = self.add_variable("w", (input_size, output_size), dtype, trainable=True, initializer=w_initializer)
        self.b = self.add_variable("b", (output_size), dtype, trainable=True, initializer=b_initializer)

    def call(self, input):
        return tf.nn.xw_plus_b(input, self.w, self.b)

    @property
    def output_size(self):
        return self._output_size


@rnn_cell_extension
class AffineProjectionPseudoCell(AffineProjectionLayer):
    pass
