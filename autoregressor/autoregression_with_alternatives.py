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
            max_output_sequence_length,
            probability_model_initial_input=None,
            probability_distribution_starting_from_id_zero=True
            ):
        self.number_of_alternatives = number_of_alternatives
        self.conditional_probability_model = conditional_probability_model
        self.max_output_sequence_length = max_output_sequence_length
        self.probability_distribution_starting_from_id_zero = probability_distribution_starting_from_id_zero
        self.probability_model_initial_input = probability_model_initial_input

    def compute_output_shape(self, input_shape):
        return (self.number_of_alternatives,)

    def zero_state(self, batch_size=1, dtype=tf.int32):
        assert batch_size == 1, "Batch size MUST be equal to 1"
        assert dtype == tf.int64 or dtype == tf.int32 or dtype == tf.int16, "Batch size MUST be equal to 1"
        initial_paths = tf.zeros((self.number_of_alternatives, self.max_output_sequence_length), dtype=dtype)
        if self.probability_model_initial_input is not None:
            initial_paths = tf.concat(
                    (
                        (
                            tf.ones((self.number_of_alternatives, 1)) * self.probability_model_initial_input
                        ), 
                    initial_paths
                    )
                )
        return AutoregressionState(
            tf.zeros((self.number_of_alternatives,),dtype=tf.int32),
            initial_paths,
            tf.ones((self.number_of_alternatives,), dtype=tf.float32),  # TODO: Ones here comes from absolute probabilities [0,1]
            self.conditional_probability_model.zero_state(self.number_of_alternatives, dtype))

    def call(self, input, state: AutoregressionState):
        conditional_probability, new_probability_model_states = self._compute_next_step_probability(state.step, state.paths, state.probability_model_states)
        temp_path_probabilities = tf.transpose(tf.transpose(conditional_probability) * state.path_probabilities) # TODO: to zależy od reprezentacji prawdopodobieństwa, przy bardziej praktycznej logitowej reprezentacji to powinien być raczej plus
        p_values, (path_index, element_index) = self._top_k_from_2d_tensor(temp_path_probabilities, self.number_of_alternatives)
        new_paths = state.paths# tf.gather(state.paths, path_index)
        # TODO: rozwiązać to jakimiś konfigurowalnymi mapowaniami
        if self.probability_distribution_starting_from_id_zero:
            offset = 0
        else:
            offset = 1
        new_paths = tf.concat(
            (
                new_paths[:, :state.step[0]], 
                tf.expand_dims(element_index+offset,1), 
                new_paths[:, state.step[0]+1:]
                ),
            axis=1)
        #new_paths = tf.concat((new_paths, tf.expand_dims(element_index,1)),axis=1)
        
        new_probabilities = tf.gather(state.path_probabilities, path_index) * p_values # See above
        new_state = AutoregressionState(state.step + 1, new_paths, new_probabilities, new_probability_model_states)
        output = new_probabilities
        return output, new_state

    def _compute_next_step_probability(self, step, paths, model_states):
        previuos_step_output = tf.gather_nd(paths, 
                tf.concat(
                    (
                        tf.expand_dims(tf.range(0, tf.shape(step)[0], name="range_path_number"), axis=1, name="expand_path_number"),
                        tf.expand_dims(step, axis=1, name="expand_step")
                        ), 
                    axis=1,
                    name="concat_indices_to_gather")
            )
        return self.conditional_probability_model(
            previuos_step_output, 
            model_states)

    @staticmethod
    def _top_k_from_2d_tensor(tensor2d, k):
        """Find top k values of 2D tensor. Return values themselves and vectors their first and second indices."""
        flat_tensor = tf.reshape(tensor2d, (-1,))
        top_values, top_indices = tf.nn.top_k(flat_tensor, k)
        top_index1 = top_indices // tf.shape(tensor2d)[1]
        top_index2 = top_indices % tf.shape(tensor2d)[1]
        return top_values, (top_index1, top_index2)
