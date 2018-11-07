from collections import namedtuple
import tensorflow as tf


AutoregressionState = namedtuple("AutoregressionState", ["step", "paths", "path_probabilities", "probability_model_states"])


class AutoregressionInitializer:
    """
    Class that creates n one-element paths from initial input (e.g. id of special <START> element). 
    
    Initial input should be one sequence of element IDs per batch (possibly one-element sequence per batch).
    Probability distribution predicted by the conditional probability model after consuming all initial input elements is used to chose n most probable next elements.
    n output paths are created by appending initial input sequence with each of these n elements.
    
    Input dims: 
        b x i: (b - batch size - each element is an initial input element to conditional probability model)
    Output dims:
        b x n x (i+1): (b - batch size; n - number of alternative paths)

    Args:
        conditional_probability_model (tf.nn.rnn_cell.RNNCell): Layer object that takes input and state and returns output and new state with output interpreted as a probability distribution of being the next element over elements from some finate vocabulary. 
        number_of_ouput_paths (int): how many alternative paths (per batch element) will be created.
        index_in_probability_distribution_to_element_id_mapping: layer or tensorflow op that takes index in probability distribution produced by conditional_probability_model and returns id of corresponding element of the vocabulary.
    """
    def __init__(self, conditional_probability_model, number_of_ouput_paths, index_in_probability_distribution_to_element_id_mapping):
        self.conditional_probability_model = conditional_probability_model
        self.number_of_ouput_paths = number_of_ouput_paths, 
        self.index_in_probability_distribution_to_element_id_mapping = index_in_probability_distribution_to_element_id_mapping

    def call(self, input):
        """
        Args:
            input: tensor of dimemsions [batch, initial_sequence_length] and integer type (tf.int32).
        
        Returns:
            paths: tensor of dimemsions [batch, number_of_ouput_paths, initial_sequence_length+1] and type the same as input.
            paths_probabilites: tensor of dimensions [batch, number_of_ouput_paths] and type the same as conditional probability  
        """
        raise NotImplementedError

class AutoregressionExtender:
    """
    Class that takes n paths and generates additional k elements of this paths.
    Input dims: 
        b x n x l: (b - batch size; n - number of alternative paths; l - number of elements in each of paths)
    Output dims:
        b x n x (l+k): (b - batch size; n - number of alternative paths; k - number of newly generated elements in each of paths)

    Args:
        autoregression_step (tf.nn.rnn_cell.RNNCell): recursive layer that takes no meaningful input and AutoregressionState as state and appends one element to paths in the state at each call; it must be able to be called at least number_of_elements_to_generate.
        number_of_elements_to_generate (int): numbers of elements that will be appended to path == number of calls of autoregression_step
    """
    def __init__(self, autoregression_step, number_of_elements_to_generate):
        self.autoregression_step = autoregression_step
        self.number_of_elements_to_generate = number_of_elements_to_generate

    def call(self, paths, probabilites):
    """
    Args:
        paths (tf.Tensor): paths that are to be extended, dimensions - [batch, path, element]
        paths_probabilites (tf.Tensor): probabilites of paths that are to be extended, dimensions - [batch, path]
    """
    raise NotImplementedError

class AutoregressionBroadcaster:
    """
    Class that performs takes tensor of n paths, re-writes it to vector of m paths and generates additional k elements of this paths.
    Input dims: 
        b x n x l: (b - batch size; n - number of alternative paths; l - number of elements in each of paths)
    Output dims:
        b x m x l: (b - batch size; n - number of alternative paths; k - number of newly generated elements in each of paths)

    If m == n then input == output.
    If m > n then first n paths of output are equal to corresponding paths of input and remaining m-n paths are filled with zeros and have probability of 0.0.
    If m < n then m paths of output are equal to corresponding first m paths of input. It is therefore important for input to be sorted by probability.
    
    Args:
        output_alternative_paths_number (int): how much alternative paths will be included in output paths and corresponding paths probabilities
    """
    def __init__(self, output_alternative_paths_number):
        self.output_alternative_paths_number = output_alternative_paths_number

    def call(self, paths, paths_probabilites):
        """Choses self.output_alternative_paths_number alternative paths from input, fills with zero if there is not enough paths in input. 
        
        It is assumed that the first dimension is batch index, the second is alternative path index and the third is element-in-path index
        Args:
            paths (tf.Tensor): paths that are to be broadcasted, dimensions - [batch, path, element]
            paths_probabilites (tf.Tensor): probabilites of paths that are to be broadcasted, dimensions - [batch, path]
        
        Returns:
            broadcasted_paths: Tensor of dimensions [batch size, self.output_alternative_paths_number, number of elements] and type the same as paths
            broadcasted_paths_probabilites: Tensor of dimensions [batch size, self.output_alternative_paths_number] and type the same as paths_probabilites
        """
        raise NotImplementedError


class AutoregressionWithAlternativePaths(tf.keras.layers.Layer):
    def __init__(self,
            autoregression_step_builder,
            number_of_alternatives,
            output_sequence_length):
            """
            Args:
                autoregression_step_builder (Function[(int, int),tf.nn.rnn_cell.RNNCell]): callable -> 
            """
        self.output_sequence_length = output_sequence_length
        self.number_of_alternatives = number_of_alternatives
        self.conditional_probability_model = conditional_probability_model

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
            index_in_probability_distribution_to_element_id_mapping=tf.identity,
            id_to_embedding_mapping=lambda x: tf.expand_dims(x,1),
            probability_masking_layer=None,
            ):
        """
        Args:
            max_output_sequence_length: maximum number of elements that are to be generated
        """
        super(AutoregressionWithAlternativePathsStep, self).__init__()
        self.number_of_alternatives = number_of_alternatives
        self.conditional_probability_model = conditional_probability_model
        self.max_output_sequence_length = max_output_sequence_length
        self.index_in_probability_distribution_to_element_id_mapping = index_in_probability_distribution_to_element_id_mapping
        self.probability_model_initial_input = tf.identity(probability_model_initial_input)
        self.id_to_embedding_mapping = id_to_embedding_mapping
        self.probability_masking_layer = probability_masking_layer
        self.conditional_probability_model_feeder = ConditionalProbabilityModelFeeder(conditional_probability_model, id_to_embedding_mapping)
        self.next_element_generator = NextElementGenerator(self.conditional_probability_model_feeder)

    @property
    def state_size(self):
        return AutoregressionState(
                tf.TensorShape(1), 
                tf.TensorShape((self.number_of_alternatives, self.max_output_sequence_length)),
                tf.TensorShape((self.number_of_alternatives,)),
                self.conditional_probability_model.state_size
                )

    @property
    def output_size(self):
        return tf.TensorShape((self.number_of_alternatives,))

    def zero_state(self, batch_size=1, dtype=tf.int32, initial_paths=None, initial_probablities=None):
        # TODO: jakieś bardziej eleganckie rozwiązanie z dtype
        #assert dtype == tf.int64 or dtype == tf.int32 or dtype == tf.int16, "Batch size MUST be equal to 1"
        dtype=tf.int32

        return AutoregressionState(
            self._get_initial_step(batch_size),
            initial_paths if initial_paths is not None else self._get_initial_paths(dtype, batch_size),
            initial_probablities if initial_probablities is not None else self._get_initial_probabilites(batch_size), 
            self._recurrent_replicate_for_batch(self.conditional_probability_model.zero_state(self.number_of_alternatives, dtype),batch_size)
            )

    def _get_initial_step(self, batch_size):
        if self.probability_model_initial_input not in {None, False}:
            return tf.ones((batch_size, self.number_of_alternatives,),dtype=tf.int32)
        else:
            return tf.zeros((batch_size, self.number_of_alternatives,),dtype=tf.int32)

    def _get_initial_paths(self, dtype, batch_size):
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
        initial_paths = tf.gather([initial_paths], tf.zeros((batch_size,), dtype=tf.int32))
        return initial_paths
    
    def _get_initial_probabilites(self, batch_size):
        return tf.ones((batch_size, self.number_of_alternatives,), dtype=tf.float32)  # TODO: Ones here comes from absolute probabilities [0,1]

    def call(self, input, state: AutoregressionState):
        # This firs call is just for dtype inference
        example_result = self._call_with_one_batch_element(input[0], self._recurrent_gather(state,0))
        dtype = self._recurrent_apply(example_result, lambda t: t.dtype)
        dtype = (dtype[0], AutoregressionState(*dtype[1]))
        return tf.map_fn(
            lambda input_state: self._call_with_one_batch_element(input_state[0], input_state[1]),
            (input, state),
            infer_shape=True,
            dtype=dtype
            )

    def _extend_sequence(self, input_sequences, input_probabilities, step, probability_model_state):
        """This function takes N sequences with probabilites assigned to them and produces M output sequences and their probabilites according to 
        a conditional probability model. Each of output sequences is one of inputs sequences with an exactly one element appended at the end. 
        Output sequences are sorted with respect to their probabelity."""
        previuos_step_output = self._get_ids_in_step(input_sequences, step-1)
        '''previuos_step_output = tf.gather_nd(input_sequences, 
                tf.concat(
                    (
                        tf.expand_dims(tf.range(0, tf.shape(step-1)[0], name="range_path_number"), axis=1, name="expand_path_number"),
                        tf.expand_dims(step-1, axis=1, name="expand_step")
                        ), 
                    axis=1,
                    name="concat_indices_to_gather")
            )'''
        temp_path_probabilities, new_probability_model_states = self.next_element_generator.call(
            previous_step_output=previuos_step_output, 
            input_probabilities=input_probabilities, 
            probability_model_state=probability_model_state)

        if self.probability_masking_layer is not None:
            masked_probabilities = self.probability_masking_layer(temp_path_probabilities, step=step)
        else:
            masked_probabilities = temp_path_probabilities
        probability_values, (sequence_index, element_index) = self._top_k_from_2d_tensor(masked_probabilities, self.number_of_alternatives)

        new_paths = tf.gather(input_sequences, sequence_index)
        if isinstance(new_probability_model_states, tuple):
            new_probability_model_states = self._recurrent_gather(new_probability_model_states, sequence_index)
        else:
            new_probability_model_states = tf.gather(new_probability_model_states, sequence_index)

        next_element_ids = self.index_in_probability_distribution_to_element_id_mapping(element_index)
        new_paths = self._insert(new_paths, next_element_ids, step)
        return probability_values, new_paths, new_probability_model_states

    def _call_with_one_batch_element(self, input, state: AutoregressionState):
        """Jeśli initial step nie jest None to step zaczyna się liczyć od 1, nie od 0. """
        state = AutoregressionState(*state)
        assert len(state.paths.shape) == 2
        
        #new_paths = tf.concat((new_paths, tf.expand_dims(element_index,1)),axis=1)zx
        
        new_probabilities, new_paths, new_probability_model_states = self._extend_sequence(state.paths, state.path_probabilities, state.step, state.probability_model_states) # tf.gather(state.path_probabilities, path_index) * p_values # See above
        new_step = tf.add(state.step, 1, name="increment_step")
        new_state = AutoregressionState(new_step, new_paths, new_probabilities, new_probability_model_states)
        output = tf.identity(new_probabilities, name="regressor_step_output")
        return output, new_state

    def _get_ids_in_step(self, input_sequences, step):
        return tf.gather_nd(input_sequences, 
                tf.concat(
                    (
                        tf.expand_dims(
                            tf.range(
                                0, 
                                tf.shape(step)[0], 
                                name="range_path_number"), 
                            axis=1, 
                            name="expand_path_number"),
                        tf.expand_dims(
                            step, 
                            axis=1, 
                            name="expand_step")
                        ), 
                    axis=1,
                    name="concat_indices_to_gather")
            )

    def _recurrent_gather(self, t, indices):
        if isinstance(t, tuple):
            return tuple(self._recurrent_gather(e, indices) for e in t)
        elif isinstance(t, tf.Tensor):
            if len(t.shape) == 0:
                return t
            else:
                return tf.gather(t, indices)
        else:
            raise ValueError("Probability model state type not supported by _recurrent_gather: {}".format(type(t)))

    def _recurrent_expand_dims(self, t, axis):
        if isinstance(t, tuple):
            return tuple(self._recurrent_expand_dims(e, axis) for e in t)
        elif isinstance(t, tf.Tensor):
            return tf.expand_dims(t, axis)
        else:
            raise ValueError("Probability model state type not supported by _recurrent_epand_dims: {}".format(type(t)))
    
    def _recurrent_replicate_for_batch(self, t, batch_size):
        def replicate(tensor):
            return tf.gather([tensor], tf.zeros((batch_size), tf.int32))
        r = self._recurrent_apply(t, replicate)
        return r

    def _recurrent_apply(self, t, fn, *a, **k):
        if isinstance(t, tuple):
            return tuple(self._recurrent_apply(e, fn, *a, **k) for e in t)
        else:
            return fn(t, *a, **k)

    @staticmethod
    def _top_k_from_2d_tensor(tensor2d, k):
        """Find top k values of 2D tensor. Return values themselves and vectors their first and second indices.
        If two elements are equal, the lower-row has priority, if they are in the same row, lower index has priority. """
        flat_tensor = tf.reshape(tensor2d, (-1,))
        top_values, top_indices = tf.nn.top_k(flat_tensor, k)
        top_index1 = top_indices // tf.shape(tensor2d)[1]
        top_index2 = top_indices % tf.shape(tensor2d)[1]
        return top_values, (top_index1, top_index2)

    @staticmethod
    def _insert(batch, values, index):
        """Insert elements from values as index-th elements of corresponding tensors from batch.
        Example:
            tensor = [[1,2,3], [4,5,6]]
            values = [8,9]
            index = [1,1]
            
            returns [[1,8,3], [4,9,6]]
            
        WARNING:
            This is a work in progress, current implementation supports only index vector that has all values the same."""
        assert len(values.shape) == len(batch.shape) - 1, "Values rank thas not match vector in which they are to be inserted"
        assert values.shape[0].is_compatible_with(batch.shape[0]), "Batch size different than number of values to insert"
        assert index.shape[0].is_compatible_with(batch.shape[0]), "Batch size different than number of indices"
        assert values.shape[1:].is_compatible_with(batch.shape[2:]), "Dimensionality of values doesnt match dimensionality of batch elements"

        return tf.concat(
            (
                batch[:, :index[0]], 
                tf.expand_dims(values,1), 
                batch[:, index[0]+1:]
                ),
            axis=1)


class NextElementGenerator:
    def __init__(
        self,
        conditional_probability_model_feeder,
        probability_aggregation_op=tf.multiply, # to zależy od reprezentacji prawdopodobieństwa, przy bardziej praktycznej logitowej reprezentacji to powinien być raczej plus
    ):
        self.conditional_probability_model_feeder = conditional_probability_model_feeder
        self.probability_aggregation_op = probability_aggregation_op

    def call(self, previous_step_output, input_probabilities, probability_model_state):
        conditional_probability, new_probability_model_states = self.conditional_probability_model_feeder.call(previous_step_output, probability_model_state)
        aggregated_continuation_probabilities = tf.transpose(
                self.probability_aggregation_op(
                    tf.transpose(conditional_probability), input_probabilities, name="next_step_probability_aggregation"
                    )
                )
        return aggregated_continuation_probabilities, new_probability_model_states


class ConditionalProbabilityModelFeeder:
    def __init__(
        self,
        conditional_probability_model,
        id_to_embedding_mapping
        ):
        self.conditional_probability_model = conditional_probability_model
        self.id_to_embedding_mapping = id_to_embedding_mapping
    
    def call(self, previous_step_output, model_state):
        previuos_step_embeddings = self.id_to_embedding_mapping(previous_step_output)
        return self.conditional_probability_model(
            previuos_step_embeddings, 
            model_state)
