"""
The aim of this test is to check if different modules behave well when there is only one element of vocabulary (one real element and START element = 0) 

Vocabulary
name    id  vector
<START>   0 [1,2,3]
A   1   [4,5,6]
"""
import tensorflow as tf
import numpy as np
import pytest
from pytest import approx
from autoregression_with_alternatives import *
from element_probablity_mask import ElementProbabilityMasking


class OneElementConditionalProbabilityModel(tf.nn.rnn_cell.RNNCell):
    DISTRIBUTION_SIZE = 1
    def __init__(self):
        super(OneElementConditionalProbabilityModel, self).__init__()

    def call(self, input, state):
        output = tf.ones(
            shape=(
                tf.shape(input)[0], # batch
                self.DISTRIBUTION_SIZE # size of probability distribution
                )
            )
        new_state = state + 1
        return output, new_state
    
    def zero_state(self, batch_size, dtype=tf.float32):
        return tf.zeros((batch_size,self.DISTRIBUTION_SIZE), dtype=dtype)
    
    @property
    def output_size(self):
        return tf.TensorShape((self.DISTRIBUTION_SIZE,))

    @property
    def state_size(self):
        return tf.TensorShape((1,))


def prob_dist_index_to_id(index):
    return index + 1

def element_id_to_index_in_probability_distribution(id):
    return id - 1

def id_to_vector(ids):
    assert len(ids.shape) == 1, "Id must be scalar"
    embeddings = tf.constant([[1,2,3],[4,5,6]])
    return tf.nn.embedding_lookup(embeddings, ids-1)


@pytest.mark.parametrize("batch_size", [1,2,3])
def test_OneElementConditionalProbabilityModel(batch_size):
    model = OneElementConditionalProbabilityModel()
    input = tf.ones((batch_size,), dtype=tf.int32)
    zero_state = model.zero_state(batch_size)
    output, state = model.call(input, zero_state)
    with tf.Session() as sess:
        r_zero_state, r_output, r_state = sess.run((zero_state, output, state))

    assert r_zero_state == approx(np.array([[0]]*batch_size))
    assert r_output == approx(np.array([[1]]*batch_size))
    assert r_state == approx(np.array([[1]]*batch_size))


@pytest.mark.parametrize("alternatives, output_length, batch_size",
    [
        (1,7,3),
        (2,7,3),
        (10,7,3),
        (33,7,3),
    ]
)
def test_step_call(alternatives, output_length, batch_size):
    regresor = AutoregressionWithAlternativePathsStep(
        conditional_probability_model=OneElementConditionalProbabilityModel(),
        number_of_alternatives=alternatives,
        max_output_sequence_length=output_length,
        probability_model_initial_input=0,
        index_in_probability_distribution_to_element_id_mapping=prob_dist_index_to_id,
        id_to_embedding_mapping=id_to_vector,
        probability_masking_layer=None,
        )
    mock_input = tf.ones(batch_size)
    zero_state = regresor.zero_state(batch_size)
    t_outputs = []
    t_states = [zero_state]
    for _ in range(output_length):
        t_o, t_s = regresor.call(mock_input, t_states[-1])
        t_outputs.append(t_o)
        t_states.append(t_s)

    with tf.Session() as sess:
        r_outputs, r_states = sess.run((t_outputs, t_states))
    
    for output in r_outputs:
        assert output == approx(np.array([[1.0]*alternatives]*batch_size))
    
    for i, state in enumerate(r_states):
        assert state.paths == approx(
                np.array(
                    [
                        [
                            [0] + [1]*i + [0]*(output_length-i)
                        ] * alternatives
                    ] * batch_size
                )
            )

        assert state.path_probabilities == approx(np.array([[1.0]*alternatives]*batch_size))
        assert state.step == approx(np.array([[i+1]*alternatives]*batch_size))

    