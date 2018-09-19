from collections import namedtuple
import tensorflow as tf


AutoregressionState = namedtuple("AutoregressionState", ["step", "paths", "path_probabilities", "probability_model_states"])

class AutoregressionWithAlternativePaths(tf.keras.layers.Layer):
    def __init__(self,
            number_of_alternatives,
            conditional_probability_model,
            output_sequence_length):
        self.output_sequence_length = output_sequence_length
        self.number_of_alternatives = number_of_alternatives
        self.conditional_probability_model,

    def compute_output_shape(self, input_shape):
        return (self.number_of_alternatives, self.output_sequence_length)


class AutoregressionWithAlternativePathsStep(tf.nn.rnn_cell.RNNCell):
    """W pierwszej wersji zakładamy, że prawdopodobieństwa są wyrażone w sposób absolutny [0;1] i w takiej postaci przewiduje je model.
    Ponadto zakładamy, że model działa na batchach, kolejne "wiersze" tensora stanu to stany dla kolejnych sekwencji wejściowych.
    Model prawdopodobieństwa musi też przyjmować wartość wejściową w danym kroku i stan z poprzedniego kroku, 
    a zwracać rozkład prawdopodobieństwa i nowy stan."""
    def __init__(self,
            number_of_alternatives,
            conditional_probability_model,
            max_output_sequence_length):
        self.number_of_alternatives = number_of_alternatives
        self.conditional_probability_model = conditional_probability_model
        self.max_output_sequence_length = max_output_sequence_length

    def compute_output_shape(self, input_shape):
        return (self.number_of_alternatives,)


    def zero_state(self, batch_size=1, dtype=tf.int32):
        assert batch_size == 1, "Batch size MUST be equal to 1"
        assert dtype == tf.int64 or dtype == tf.int32 or dtype == tf.int16, "Batch size MUST be equal to 1"
        # TODO: Mogą być potrzebne standardowe argumenty batch_size i dtype, -
        return AutoregressionState(
            tf.zeros((self.number_of_alternatives, self.max_output_sequence_length), dtype=dtype),
            tf.zeros((self.number_of_alternatives,), dtype=tf.float32),
            self.conditional_probability_model.zero_state())
        

    def call(self, input, state: AutoregressionState):
        conditional_probability, new_probability_model_states = self.conditional_probability_model(
            state.paths[:, state.step], 
            state.probabiblity_model_states)
        temp_path_probabilities = conditional_probability * state.path_probabilities # TODO: to zależy od reprezentacji prawdopodobieństwa, przy bardziej praktycznej logitowej reprezentacji to powinien być raczej plus
        p_values, (path_index, element_index) = self._top_k_from_2d_tensor(temp_path_probabilities, self.number_of_alternatives)
        new_paths = state.paths[path_index]
        new_paths[state.step+1] = element_index
        new_probabilities = state.path_probabilities[path_index] + p_values
        new_state = AutoregressionState(state.step + 1, new_paths, new_probabilities, new_probability_model_states)
        output = new_probabilities
        return output, new_state

    @staticmethod
    def _top_k_from_2d_tensor(tensor2d, k):
        """Find top k values of 2D tensor. Return values themselves and vectors their first and second indices."""
        flat_tensor = tf.reshape(tensor2d, (-1,))
        top_values, top_indices = tf.nn.top_k(flat_tensor, k)
        top_index1 = top_indices // tf.shape(tensor2d)[1]
        top_index2 = top_indices % tf.shape(tensor2d)[1]
        return top_values, (top_index1, top_index2)
