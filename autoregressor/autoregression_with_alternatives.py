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
    a zwracać rozkład prawdopodobieństwa i nowy stan.
    probability_model_initial_input - scalar that is inserted as a first element passed to cond. prob. model
    """
    def __init__(self,
            number_of_alternatives,
            conditional_probability_model,
            max_output_sequence_length,
            probability_model_initial_input=None,
            index_in_probability_distribution_to_element_id_mapping=lambda x: tf.expand_dims(x, 1),
            ):
        self.number_of_alternatives = number_of_alternatives
        self.conditional_probability_model = conditional_probability_model
        self.max_output_sequence_length = max_output_sequence_length
        self.index_in_probability_distribution_to_element_id_mapping = index_in_probability_distribution_to_element_id_mapping
        self.probability_model_initial_input = tf.identity(probability_model_initial_input)

    def compute_output_shape(self, input_shape):
        return (self.number_of_alternatives,)

    def zero_state(self, batch_size=1, dtype=tf.int32):
        assert batch_size == 1, "Batch size MUST be equal to 1"
        assert dtype == tf.int64 or dtype == tf.int32 or dtype == tf.int16, "Batch size MUST be equal to 1"

        return AutoregressionState(
            self._get_initial_step(),
            self._get_initial_paths(dtype),
            tf.ones((self.number_of_alternatives,), dtype=tf.float32),  # TODO: Ones here comes from absolute probabilities [0,1]
            self.conditional_probability_model.zero_state(self.number_of_alternatives, dtype))

    def _get_initial_step(self):
        if self.probability_model_initial_input not in {None, False}:
            return tf.ones((self.number_of_alternatives,),dtype=tf.int32)
        else:
            return tf.zeros((self.number_of_alternatives,),dtype=tf.int32)

    def _get_initial_paths(self, dtype):
        initial_paths = tf.zeros((self.number_of_alternatives, self.max_output_sequence_length), dtype=dtype)
        if self.probability_model_initial_input not in {None, False}:
            initial_paths = tf.concat(
                    (
                        tf.tile([[self.probability_model_initial_input]], 
                                [self.number_of_alternatives,1]),
                        initial_paths
                    ),
                    axis=1
                )
        initial_paths = tf.expand_dims(initial_paths, 2)
        return initial_paths

    def call(self, input, state: AutoregressionState):
        conditional_probability, new_probability_model_states = self._compute_next_step_probability(state.step-1, state.paths, state.probability_model_states)
        temp_path_probabilities = tf.transpose(tf.transpose(conditional_probability) * state.path_probabilities) # TODO: to zależy od reprezentacji prawdopodobieństwa, przy bardziej praktycznej logitowej reprezentacji to powinien być raczej plus
        p_values, (path_index, element_index) = self._top_k_from_2d_tensor(temp_path_probabilities, self.number_of_alternatives)

        new_paths = tf.gather(state.paths, path_index)
        if type(new_probability_model_states) is tuple:
            new_probability_model_states = self._recurrent_gather(new_probability_model_states, path_index)
        else:
            new_probability_model_states = tf.gather(new_probability_model_states, path_index)

        next_element_ids = self.index_in_probability_distribution_to_element_id_mapping(element_index)
        new_paths = tf.concat(
            (
                new_paths[:, :state.step[0]], 
                tf.expand_dims(next_element_ids,1), 
                new_paths[:, state.step[0]+1:]
                ),
            axis=1)
        #new_paths = tf.concat((new_paths, tf.expand_dims(element_index,1)),axis=1)
        
        new_probabilities = p_values # tf.gather(state.path_probabilities, path_index) * p_values # See above
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

    def _recurrent_gather(self, t, indices):
        if type(t) is tuple:
            return tuple(self._recurrent_gather(e, indices) for e in t)
        elif type(t) is tf.Tensor:
            if len(t.shape) == 0:
                return t
            else:
                return tf.gather(t, indices)
        else:
            raise ValueError("Probability model state type not supported by _recurrent_gather")

    @staticmethod
    def _top_k_from_2d_tensor(tensor2d, k):
        """Find top k values of 2D tensor. Return values themselves and vectors their first and second indices."""
        flat_tensor = tf.reshape(tensor2d, (-1,))
        top_values, top_indices = tf.nn.top_k(flat_tensor, k)
        top_index1 = top_indices // tf.shape(tensor2d)[1]
        top_index2 = top_indices % tf.shape(tensor2d)[1]
        return top_values, (top_index1, top_index2)
