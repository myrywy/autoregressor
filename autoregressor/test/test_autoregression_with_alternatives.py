import numpy as np
import pytest
from pytest import approx
import mock_prob_model
from mock_prob_model import MockModelLayer
from element_probablity_mask import ElementProbabilityMasking
from autoregression_with_alternatives import *
from tensorflow.python import debug as tf_debug


@pytest.fixture
def probabilities():
    probabilities = {
        (0,0,0):[0.5, 0.5, 0.0], # początek historii

        (1,0,0):[0.6, 0.4, 0.0], # a
        (2,0,0):[0.0, 0.0, 1.0], # b
        
        (1,1,0):[0.6, 0.4, 0.0], # aa
        (1,2,0):[0.95, 0.05, 0.0], # ab
        (2,1,0):[0.0, 0.0, 1.0], # ba
        (2,2,0):[0.0, 0.0, 1.0], # bb
        }

    probabilities = {tuple((i,) for i in k):v for k, v in probabilities.items()}
    return probabilities


@pytest.fixture
def probabilities_with_start_element():
    """Czyli z elementem -1 na poczatku kazdej historii """
    probabilities = {
        (-1,0,0,0):[0.5, 0.5, 0.0], # początek historii

        (-1,1,0,0):[0.6, 0.4, 0.0], # a
        (-1,2,0,0):[0.0, 0.0, 1.0], # b
        
        (-1,1,1,0):[0.6, 0.4, 0.0], # aa
        (-1,1,2,0):[0.95, 0.05, 0.0], # ab
        (-1,2,1,0):[0.0, 0.0, 1.0], # ba
        (-1,2,2,0):[0.0, 0.0, 1.0], # bb
        }

    probabilities = {tuple((i,) for i in k):v for k, v in probabilities.items()}
    return probabilities


@pytest.fixture
def probabilities_with_start_element_no_third():
    """Czyli z elementem -1 na poczatku kazdej historii """
    probabilities = {
        (-1,0,0,0):[0.5, 0.5, 0.0], # początek historii

        (-1,1,0,0):[0.6, 0.4, 0.0], # a
        (-1,2,0,0):[0.0, 0.0, 0.0], # b
        
        (-1,1,1,0):[0.6, 0.4, 0.0], # aa
        (-1,1,2,0):[0.95, 0.05, 0.0], # ab
        (-1,2,1,0):[0.0, 0.0, 0.0], # ba
        (-1,2,2,0):[0.0, 0.0, 0.0], # bb
        }

    probabilities = {tuple((i,) for i in k):v for k, v in probabilities.items()}
    return probabilities

@pytest.mark.parametrize("batch_size", [1,2])
def test_autoregressor_with_dynamic_rnn(probabilities_with_start_element_no_third, batch_size):
    seq_len = 3
    model = MockModelLayer(probabilities_with_start_element_no_third, history_entry_dims=(1,))
    regresor = AutoregressionWithAlternativePathsStep(
        2, 
        model, 
        seq_len, 
        probability_model_initial_input=-1,
        index_in_probability_distribution_to_element_id_mapping=lambda x: x+1)
    inputs = tf.zeros((batch_size, seq_len, 1), tf.float32) # this values actually doesn't matter, only shape is important
    output, states = tf.nn.dynamic_rnn(regresor, inputs, sequence_length=[seq_len]*batch_size, dtype=tf.float32)
    with tf.Session() as sess:
        r_output = sess.run(output)
    assert r_output == approx(
            np.array(
                [
                    [[0.5, 0.5], [0.3, 0.2], [0.19, 0.18]],
                ]*batch_size
            )
        )


@pytest.mark.parametrize(
    "batch_size, allowed, expected", 
    [
        # all allowed - results should be the same as with no mask
        (
            1,
            [[],[],[]],
            [[0.5, 0.5], [0.3, 0.2], [0.19, 0.18]]
            ),
        (
            2,
            [[],[],[]],
            [[0.5, 0.5], [0.3, 0.2], [0.19, 0.18]]
            ),
        # only one element allowed
        (
            2,
            [[1],[1],[1]], 
            [[0.5, 0.5], [0.5*0.6, 0.5*0.6], [0.5*0.6*0.6, 0.5*0.6*0.6]] # jeśli nie zrobi się różnych ścieżek początkowych to zawsze wyjdą takie same wszystkie
            ),
    ]
    )
def test_autoregressor_with_mask_with_dynamic_rnn(probabilities_with_start_element_no_third, batch_size, allowed, expected):
    seq_len = 3
    DISTRIBUTION_SIZE = 3
    masking = ElementProbabilityMasking(allowed, DISTRIBUTION_SIZE, 1, 3, lambda x: x-1)
    model = MockModelLayer(probabilities_with_start_element_no_third, history_entry_dims=(1,))
    regresor = AutoregressionWithAlternativePathsStep(
        2, 
        model, 
        seq_len, 
        probability_model_initial_input=-1,
        index_in_probability_distribution_to_element_id_mapping=lambda x: x+1,
        probability_masking_layer=masking)
    inputs = tf.zeros((batch_size, seq_len, 1), tf.float32) # this values actually doesn't matter, only shape is important
    output, states = tf.nn.dynamic_rnn(regresor, inputs, sequence_length=[seq_len]*batch_size, dtype=tf.float32)
    with tf.Session() as sess:
        r_output,r_states = sess.run((output, states))
    assert r_output == approx(
            np.array(
                [
                expected,
                ]*batch_size
            )
        )


def test_step_call(probabilities_with_start_element_no_third):
    model = MockModelLayer(probabilities_with_start_element_no_third, history_entry_dims=(1,))
    regresor = AutoregressionWithAlternativePathsStep(
        2, 
        model, 
        3, 
        probability_model_initial_input=-1,
        index_in_probability_distribution_to_element_id_mapping=lambda x: x+1)
    zero_state = regresor.zero_state(1, tf.int32)
    input = tf.zeros((1, 1), tf.int32) # [one batch, one sequence element]
    output1, state1 = regresor.call(input, zero_state)
    output2, state2 = regresor.call(input, state1)
    output3, state3 = regresor.call(input, state2)

    with tf.Session() as sess:
        r_zero, r_s1, r_s2, r_s3, r_o1, r_o2, r_o3 = sess.run((zero_state, state1, state2, state3, output1, output2, output3))

    assert r_s1.path_probabilities == approx(np.array([[0.5, 0.5]]))
    assert r_s2.path_probabilities == approx(np.array([[0.3, 0.2]])) # [0.5*0.6, 0.5*0.4]
    assert r_s3.path_probabilities == approx(np.array([[0.19, 0.18]]))


def test_step_call_with_identity_mask(probabilities_with_start_element_no_third):
    model = MockModelLayer(probabilities_with_start_element_no_third, history_entry_dims=(1,))
    regresor = AutoregressionWithAlternativePathsStep(
        2, 
        model, 
        3, 
        probability_model_initial_input=-1,
        index_in_probability_distribution_to_element_id_mapping=lambda x: x+1,
        probability_masking_layer=lambda x, step: x)
    zero_state = regresor.zero_state(1, tf.int32)
    input = tf.zeros((1, 1), tf.int32) # [one batch, one sequence element]
    output1, state1 = regresor.call(input, zero_state)
    output2, state2 = regresor.call(input, state1)
    output3, state3 = regresor.call(input, state2)

    with tf.Session() as sess:
        r_zero, r_s1, r_s2, r_s3, r_o1, r_o2, r_o3 = sess.run((zero_state, state1, state2, state3, output1, output2, output3))

    assert r_s1.path_probabilities == approx(np.array([[0.5, 0.5]]))
    assert r_s2.path_probabilities == approx(np.array([[0.3, 0.2]])) # [0.5*0.6, 0.5*0.4]
    assert r_s3.path_probabilities == approx(np.array([[0.19, 0.18]]))


def test_step_on_one_path(probabilities_with_start_element):
    model = MockModelLayer(probabilities_with_start_element, history_entry_dims=(1,))
    regresor = AutoregressionWithAlternativePathsStep(
        1, 
        model, 
        3, 
        probability_model_initial_input=-1,
        index_in_probability_distribution_to_element_id_mapping=lambda x: x+1
        )
    zero_state = regresor.zero_state(1, tf.int32)

    zero_state_for_one_batch_element = AutoregressionState(*regresor._recurrent_apply(zero_state, lambda t: tf.gather(t, 0)))

    previuos_step_output = regresor._get_ids_in_step(zero_state_for_one_batch_element.paths, zero_state_for_one_batch_element.step-1)

    conditional_probability, new_probability_model_states = regresor.conditional_probability_model_feeder.call(previuos_step_output, zero_state_for_one_batch_element.probability_model_states)

    input = tf.zeros((1, 1), tf.int32)
    output1, state1 = regresor.call(input, zero_state)
    output2, state2 = regresor.call(input, state1)
    output3, state3 = regresor.call(input, state2)

    with tf.Session() as sess:
        r_cp, r_nsp = sess.run((conditional_probability, new_probability_model_states))
        r_zero, r_s1, r_s2, r_s3, r_o1, r_o2, r_o3 = sess.run((zero_state, state1, state2, state3, output1, output2, output3))

    assert r_s3.path_probabilities == approx(np.array([[0.18]]))


@pytest.mark.parametrize("target_step,path,expected_prob_dist", [
    #(0, [0,0,0], [0.5, 0.5, 0.0]),
    # w path poniżej są właściwie embeddingsy
    (1, [[1],[0],[0]], [0.6, 0.4, 0.0]), # a W tym teście chcemy przewidzieć 2-gie słowo (o indeksie 0 bo indeksujemy od 0), wiemy, że pierwsze słowo (indeks=0) to "1". Oczekiwany rozkład prawdopodobieństwa 2-go słowa po słowniku to P(a_2=1, a_2=2, a_2=3)[0.6, 0.4, 0.0]
    (1, [[2],[0],[0]], [0.0, 0.0, 1.0]), # b

    (2, [[1],[1],[0]], [0.6, 0.4, 0.0]), # aa
    (2, [[1],[2],[0]], [0.95, 0.05, 0.0]), # ab
    (2, [[2],[1],[0]], [0.0, 0.0, 1.0]), # ba
    (2, [[2],[2],[0]], [0.0, 0.0, 1.0]), # bb
])
def test__compute_next_step_probability(probabilities, target_step, path, expected_prob_dist):
    
    # To jest jednoczesnie numer słowa, który będzie feedowany do modelu prawd. war. Ma to sens bo jak chcemy przewidzieć
    # n-te słowo to musimy podać n-1-sze przy założeniu, że resza informacji o kontekście jest przechowywana w stanie modelu pr. war.
    current_autoregressor_step = tf.constant([target_step], name="init_step") 
    # To jest to co (udajemy, że) do tej pory wygenerował autoregresor.

    paths = tf.constant([path], name="init_path")[:,:,0] # dodanie wymiaru batcha i redukcja jednoelementowych embeddingsów do skalarnych id-ków

    # MockModel history 
    # Jak chemy poznać n-te słowo
    history_length = tf.constant(target_step-1, name="init_mock_history_length")
    history = [*path]
    history[target_step-1] = [0]
    history = tf.constant([history], name="init_mock_history")
    model_state = (history_length, history)

    model = MockModelLayer(probabilities, first_dim_is_batch=True)
    regresor = AutoregressionWithAlternativePathsStep(1, model, 3, False)
    previuos_step_output = regresor._get_ids_in_step(paths, current_autoregressor_step-1)
    next_step_probability_dist, _ = regresor.conditional_probability_model_feeder.call(previuos_step_output, model_state)
    with tf.Session() as sess:
        r_prob_dist = sess.run(next_step_probability_dist)
    assert r_prob_dist == approx(np.array([expected_prob_dist]))


def test__top_k_from_2d_tensor():
    c1 = tf.constant([[0.1, 0.2],
                      [0.3, 0.4]])
    c2 = tf.constant([[0.1, 0.2],
                      [0.5, 0.4]])
    c3 = tf.constant([[0.2, 3.2],
                      [0.4, 0.34],
                      [5.5, 2.4],
                      [0.5, 8.4]])
    c4 = tf.constant([[1.1, 0.2],
                      [0.5, 0.4]])
    top_k = AutoregressionWithAlternativePathsStep._top_k_from_2d_tensor
    t1 =  top_k(c1, 1)
    exp1 = np.array([0.4]), (np.array([1]), np.array([1]))
    t2 =  top_k(c2, 1)
    exp2 = np.array([0.5]), (np.array([1]), np.array([0]))
    t3 =  top_k(c3, 1)
    exp3 = np.array([8.4]), (np.array([3]), np.array([1]))
    t4 =  top_k(c4, 1)
    exp4 = np.array([1.1]), (np.array([0]), np.array([0]))
    t3_1 =  top_k(c3, 3)
    exp3_1 = np.array([8.4, 5.5, 3.2]), (np.array([3, 2, 0]), np.array([1, 0, 1]))
    with tf.Session() as sess:
        r1, r2, r3, r4, r3_1 = sess.run((t1, t2, t3, t4, t3_1))

    assert exp1[1] == r1[1]
    assert exp2[1] == r2[1]
    assert exp3[1] == r3[1]
    assert exp4[1] == r4[1]
    np.testing.assert_array_almost_equal(exp3_1[1][0], r3_1[1][0])
    np.testing.assert_array_almost_equal(exp3_1[1][1], r3_1[1][1])

    np.testing.assert_array_almost_equal(exp1[0], r1[0])
    np.testing.assert_array_almost_equal(exp2[0], r2[0])
    np.testing.assert_array_almost_equal(exp3[0], r3[0])
    np.testing.assert_array_almost_equal(exp4[0], r4[0])
    np.testing.assert_array_almost_equal(exp3_1[0], r3_1[0])


@pytest.mark.parametrize(
    "batch, values, indices, expected",
    [
        (
            [[1,2,3], [4,5,6]],
            [8,9],
            [0,0],
            [[8,2,3], [9,5,6]]
        ),
        (
            [[1,2,3], [4,5,6]],
            [8,9],
            [1,1],
            [[1,8,3], [4,9,6]]
        ),
        (
            [[1,2,3], [4,5,6]],
            [8,9],
            [2,2],
            [[1,2,8], [4,5,9]]
        ),
        (
            [[[1,7],[2,8],[3,9]], [[4,10],[5,11],[6,12]]],
            [[8,13],[9,14]],
            [0,0],
            [[[8,13],[2,8],[3,9]], [[9,14],[5,11],[6,12]]]
        ),
        (
            [[[1,7],[2,8],[3,9]], [[4,10],[5,11],[6,12]]],
            [[8,13],[9,14]],
            [1,1],
            [[[1,7],[8,13],[3,9]], [[4,10],[9,14],[6,12]]]
        ),
        (
            [[[1,7],[2,8],[3,9]], [[4,10],[5,11],[6,12]]],
            [[8,13],[9,14]],
            [2,2],
            [[[1,7],[2,8],[8,13]], [[4,10],[5,11],[9,14]]]
        )
        # Okej, okej, tu powinny być jeszcze testy na indices postaci innej niż [i, i, i,... i], ale ta funkcja na razie nie jest potrzebna 
    ]
)
def test__insert(batch, values, indices, expected):
    t_batch = tf.constant(batch)
    t_values = tf.constant(values)
    t_indices = tf.constant(indices)
    expected = np.array(expected)

    new_batch = AutoregressionWithAlternativePathsStep._insert(t_batch, t_values, t_indices)

    with tf.Session() as sess:
        result = sess.run(new_batch)

    assert result == approx(np.array(expected))

@pytest.mark.parametrize("input, state, expected_output, expected_state",
[
    (
        tf.constant([ # batch
            [ # time
                [ 1,], [ 2,], [ 3,]
            ],
            [ # time
                [ 4,], [ 5,], [ 6,]
            ]
        ]),
        tf.constant([
            [1],
            [2],
        ]),
        np.array([ # batch
            [ # time
                [ 1, 1, 1], [ 2, 2, 2], [ 3, 3, 3]
            ],
            [ # time
                [ 4, 4, 4], [ 5, 5, 5], [ 6, 6, 6]
            ]
        ]),
        np.array([
            [4],
            [5],
        ])
    )
])
def test_ConditionalProbabilityModelFeeder(input, state, expected_output, expected_state):
    """This test is intended to assert only that ConditionalProbabilityModelFeeder has features needed to use it with tf.nn.dynamic_rnn function."""
    class MockRnn(tf.nn.rnn_cell.RNNCell):
        def __init__(self):
            super(MockRnn, self).__init__()

        def call(self, input, state):
            return input, state + 1

        @property
        def output_size(self):
            return 3

        @property
        def state_size(self):
            return 2

        def zero_state(self, batch_size, dtype):
            return tf.zeros((batch_size,) + self.state_size, dtype=dtype)

    model = MockRnn()
    id_to_embeddings = lambda t: tf.concat([t, t, t], axis=1)
    feeder = ConditionalProbabilityModelFeeder(model, id_to_embeddings)
    t_output, t_state = tf.nn.dynamic_rnn(
        feeder,
        input,
        initial_state=state,
    )
    with tf.Session() as sess:
        r_output, r_state = sess.run((t_output, t_state))
    assert r_output == approx(expected_output)
    assert r_state == approx(expected_state)


@pytest.mark.parametrize("paths, paths_probabilities, states, zero_state, number_of_output_paths, expected_paths, expected_paths_probabilities, expected_states",
    [
        ( # case: output is smaller
            [ # batch level
                [ # paths level
                    [1,2,3,4],
                    [5,6,7,8],
                    [9,10,11,12],
                ],
                [ # paths level
                    [13,14,15,16],
                    [17,18,19,20],
                    [21,22,23,24],
                ],
            ],
            [ # batch level
                [ # paths level
                    0.1,
                    0.2,
                    0.3,
                ],
                [ # paths level
                    0.4,
                    0.5,
                    0.6,
                ],
            ],
            [ # batch level
                [ # paths level
                    [11,11,11], 
                    [12,12,12],
                    [13,13,13],
                ],
                [ # paths level
                    [21,21,21], 
                    [22,22,22],
                    [23,23,23],
                ]
            ],
            None,
            2,
            [ # batch level
                [ # paths level
                    [1,2,3,4],
                    [5,6,7,8],
                ],
                [ # paths level
                    [13,14,15,16],
                    [17,18,19,20],
                ],
            ],
            [ # batch level
                [ # paths level
                    0.1,
                    0.2,
                ],
                [ # paths level
                    0.4,
                    0.5,
                ],
            ],
            [ # batch level
                [ # paths level
                    [11,11,11], 
                    [12,12,12],
                ],
                [ # paths level
                    [21,21,21], 
                    [22,22,22],
                ]
            ],
        ),
        ( # case: the same input and output size, nothing changes
            [ # batch level
                [ # paths level
                    [1,2,3,4],
                    [5,6,7,8],
                    [9,10,11,12],
                ],
                [ # paths level
                    [13,14,15,16],
                    [17,18,19,20],
                    [21,22,23,24],
                ],
            ],
            [ # batch level
                [ # paths level
                    0.1,
                    0.2,
                    0.3,
                ],
                [ # paths level
                    0.4,
                    0.5,
                    0.6,
                ],
            ],
            [ # batch level
                [ # paths level
                    [11,11,11], 
                    [12,12,12],
                    [13,13,13],
                ],
                [ # paths level
                    [21,21,21], 
                    [22,22,22],
                    [23,23,23],
                ]
            ],
            None,
            3,
            [ # batch level
                [ # paths level
                    [1,2,3,4],
                    [5,6,7,8],
                    [9,10,11,12],
                ],
                [ # paths level
                    [13,14,15,16],
                    [17,18,19,20],
                    [21,22,23,24],
                ],
            ],
            [ # batch level
                [ # paths level
                    0.1,
                    0.2,
                    0.3,
                ],
                [ # paths level
                    0.4,
                    0.5,
                    0.6,
                ],
            ],
            [ # batch level
                [ # paths level
                    [11,11,11], 
                    [12,12,12],
                    [13,13,13],
                ],
                [ # paths level
                    [21,21,21], 
                    [22,22,22],
                    [23,23,23],
                ]
            ],
        ),
        ( # case: output is larger
            [ # batch level
                [ # paths level
                    [1,2,3,4],
                    [5,6,7,8],
                    [9,10,11,12],
                ],
                [ # paths level
                    [13,14,15,16],
                    [17,18,19,20],
                    [21,22,23,24],
                ],
            ],
            [ # batch level
                [ # paths level
                    0.1,
                    0.2,
                    0.3,
                ],
                [ # paths level
                    0.4,
                    0.5,
                    0.6,
                ],
            ],
            [ # batch level
                [ # paths level
                    [11,11,11], 
                    [12,12,12],
                    [13,13,13],
                ],
                [ # paths level
                    [21,21,21], 
                    [22,22,22],
                    [23,23,23],
                ]
            ],
            [0,0,0],
            4,
            [ # batch level
                [ # paths level
                    [1,2,3,4],
                    [5,6,7,8],
                    [9,10,11,12],
                    [0,0,0,0],
                ],
                [ # paths level
                    [13,14,15,16],
                    [17,18,19,20],
                    [21,22,23,24],
                    [0,0,0,0],
                ],
            ],
            [ # batch level
                [ # paths level
                    0.1,
                    0.2,
                    0.3,
                    0.0,
                ],
                [ # paths level
                    0.4,
                    0.5,
                    0.6,
                    0.0,
                ],
            ],
            [ # batch level
                [ # paths level
                    [11,11,11], 
                    [12,12,12],
                    [13,13,13],
                    [0,0,0],
                ],
                [ # paths level
                    [21,21,21], 
                    [22,22,22],
                    [23,23,23],
                    [0,0,0],
                ]
            ],
        ),
    ]
)
def test_AutoregressionBroadcaster(paths, paths_probabilities, states, zero_state, number_of_output_paths, expected_paths, expected_paths_probabilities, expected_states):
    broadcaster = AutoregressionBroadcaster(number_of_output_paths, zero_state=zero_state)
    t_paths = tf.constant(paths)
    t_paths_probabilities = tf.constant(paths_probabilities)
    t_states = tf.constant(states)
    output_paths, output_paths_probabilities, output_states = broadcaster.call(t_paths, t_paths_probabilities, t_states)
    with tf.Session() as sess:
        result_paths, result_paths_probabilites, result_states = sess.run((output_paths, output_paths_probabilities, output_states))
    assert result_paths == approx(np.array(expected_paths))
    assert result_paths_probabilites == approx(np.array(expected_paths_probabilities))
    assert expected_states == approx(np.array(result_states))


@pytest.mark.parametrize("paths, paths_probabilities, states, zero_state, number_of_output_paths, expected_paths, expected_paths_probabilities, expected_states",
    [
        ( # case: output is larger
            [ # batch level
                [ # paths level
                    [1,2,3,4],
                    [5,6,7,8],
                    [9,10,11,12],
                ],
                [ # paths level
                    [13,14,15,16],
                    [17,18,19,20],
                    [21,22,23,24],
                ],
            ],
            [ # batch level
                [ # paths level
                    0.1,
                    0.2,
                    0.3,
                ],
                [ # paths level
                    0.4,
                    0.5,
                    0.6,
                ],
            ],
            [ # batch level
                [ # paths level
                    [11,11,11], 
                    [12,12,12],
                    [13,13,13],
                ],
                [ # paths level
                    [21,21,21], 
                    [22,22,22],
                    [23,23,23],
                ]
            ],
            tf.constant([0,0,0]),
            4,
            [ # batch level
                [ # paths level
                    [1,2,3,4],
                    [5,6,7,8],
                    [9,10,11,12],
                    [0,0,0,0],
                ],
                [ # paths level
                    [13,14,15,16],
                    [17,18,19,20],
                    [21,22,23,24],
                    [0,0,0,0],
                ],
            ],
            [ # batch level
                [ # paths level
                    0.1,
                    0.2,
                    0.3,
                    0.0,
                ],
                [ # paths level
                    0.4,
                    0.5,
                    0.6,
                    0.0,
                ],
            ],
            [ # batch level
                [ # paths level
                    [11,11,11], 
                    [12,12,12],
                    [13,13,13],
                    [0,0,0],
                ],
                [ # paths level
                    [21,21,21], 
                    [22,22,22],
                    [23,23,23],
                    [0,0,0],
                ]
            ],
        ),
        ( # case: output is larger
            [ # batch level
                [ # paths level
                    [1,2,3,4],
                    [5,6,7,8],
                    [9,10,11,12],
                ],
                [ # paths level
                    [13,14,15,16],
                    [17,18,19,20],
                    [21,22,23,24],
                ],
            ],
            [ # batch level
                [ # paths level
                    0.1,
                    0.2,
                    0.3,
                ],
                [ # paths level
                    0.4,
                    0.5,
                    0.6,
                ],
            ],
            [ # batch level
                [ # paths level
                    [11,11,11], 
                    [12,12,12],
                    [13,13,13],
                ],
                [ # paths level
                    [21,21,21], 
                    [22,22,22],
                    [23,23,23],
                ]
            ],
            tf.constant([1,2,3]),
            4,
            [ # batch level
                [ # paths level
                    [1,2,3,4],
                    [5,6,7,8],
                    [9,10,11,12],
                    [0,0,0,0],
                ],
                [ # paths level
                    [13,14,15,16],
                    [17,18,19,20],
                    [21,22,23,24],
                    [0,0,0,0],
                ],
            ],
            [ # batch level
                [ # paths level
                    0.1,
                    0.2,
                    0.3,
                    0.0,
                ],
                [ # paths level
                    0.4,
                    0.5,
                    0.6,
                    0.0,
                ],
            ],
            [ # batch level
                [ # paths level
                    [11,11,11], 
                    [12,12,12],
                    [13,13,13],
                    [1,2,3],
                ],
                [ # paths level
                    [21,21,21], 
                    [22,22,22],
                    [23,23,23],
                    [1,2,3],
                ]
            ],
        ),
        ( # case: output is larger
            [ # batch level
                [ # paths level
                    [1,2,3,4],
                    [5,6,7,8],
                    [9,10,11,12],
                ],
                [ # paths level
                    [13,14,15,16],
                    [17,18,19,20],
                    [21,22,23,24],
                ],
            ],
            [ # batch level
                [ # paths level
                    0.1,
                    0.2,
                    0.3,
                ],
                [ # paths level
                    0.4,
                    0.5,
                    0.6,
                ],
            ],
            (
                tf.constant(
                        [ # batch level
                            [ # path level
                                [101,102,103],
                                [201,202,203],
                                [301,302,303],
                            ],
                            [ # path level
                                [111,112,113],
                                [211,212,213],
                                [311,312,313],
                            ],
                        ]
                    ),
                (
                    tf.constant(
                            [ # batch level
                                [ # path level
                                    [104,105,106],
                                    [204,205,206],
                                    [304,305,306],
                                ],
                                [ # path level
                                    [114,115,116],
                                    [214,215,216],
                                    [314,315,316],
                                ],
                            ]
                        ),
                    tf.constant(
                            [ # batch level
                                [ # path level
                                    [107,108,109],
                                    [207,208,209],
                                    [307,308,309],
                                ],
                                [ # path level
                                    [117,118,119],
                                    [217,218,219],
                                    [317,318,319],
                                ],
                            ]
                        )
                )
            ),
            (tf.constant([1,2,3]),(tf.constant([4,5,6]),tf.constant([7,8,9]))),
            4,
            [ # batch level
                [ # paths level
                    [1,2,3,4],
                    [5,6,7,8],
                    [9,10,11,12],
                    [0,0,0,0],
                ],
                [ # paths level
                    [13,14,15,16],
                    [17,18,19,20],
                    [21,22,23,24],
                    [0,0,0,0],
                ],
            ],
            [ # batch level
                [ # paths level
                    0.1,
                    0.2,
                    0.3,
                    0.0,
                ],
                [ # paths level
                    0.4,
                    0.5,
                    0.6,
                    0.0,
                ],
            ],
            (
                (
                        [ # batch level
                            [ # path level
                                [101,102,103],
                                [201,202,203],
                                [301,302,303],
                                [1,2,3],
                            ],
                            [ # path level
                                [111,112,113],
                                [211,212,213],
                                [311,312,313],
                                [1,2,3],
                            ],
                        ]
                    ),
                (
                    (
                            [ # batch level
                                [ # path level
                                    [104,105,106],
                                    [204,205,206],
                                    [304,305,306],
                                    [4,5,6],
                                ],
                                [ # path level
                                    [114,115,116],
                                    [214,215,216],
                                    [314,315,316],
                                    [4,5,6],
                                ],
                            ]
                        ),
                    (
                            [ # batch level
                                [ # path level
                                    [107,108,109],
                                    [207,208,209],
                                    [307,308,309],
                                    [7,8,9],
                                ],
                                [ # path level
                                    [117,118,119],
                                    [217,218,219],
                                    [317,318,319],
                                    [7,8,9],
                                ],
                            ]
                        )
                )
            ),
        ),
    ]
)
def test_AutoregressionBroadcaster_with_explicitly_specified_zero_state(paths, paths_probabilities, states, zero_state, number_of_output_paths, expected_paths, expected_paths_probabilities, expected_states):
    def maybe_convert_to_constant_tensor(t):
        if isinstance(t, tf.Tensor):
            return t
        return tf.constant(t)
    broadcaster = AutoregressionBroadcaster(number_of_output_paths, zero_state=zero_state)
    t_paths = tf.constant(paths)
    t_paths_probabilities = tf.constant(paths_probabilities)
    t_states = nested_tuple_apply(states, maybe_convert_to_constant_tensor)
    output_paths, output_paths_probabilities, output_states = broadcaster.call(t_paths, t_paths_probabilities, t_states)
    with tf.Session() as sess:
        result_paths, result_paths_probabilites, result_states = sess.run((output_paths, output_paths_probabilities, output_states))
    assert result_paths == approx(np.array(expected_paths))
    assert result_paths_probabilites == approx(np.array(expected_paths_probabilities))
    state_ok = parallel_nested_tuples_apply([expected_states, result_states], lambda a, b: (a == b).all())
    if isinstance(state_ok, tuple):
        assert state_ok[0]
        assert state_ok[1][0]
        assert state_ok[1][1]
    else:
        assert state_ok



@pytest.mark.parametrize("input, number_of_output_paths, expected_paths, expected_paths_probabilities",
    [
        (
            [ # batch level
                [1], # one-element input sequence
                [2], # one-element input sequence
            ],
            3,
            [ # batch level
                [ # paths level
                    [1, 1], # path
                    [1, 2],
                    [1, 3],
                ],
                [ # paths level
                    [2, 3], # path
                    [2, 1],
                    [2, 2],
                ],
            ],
            [ # batch level
                [ # paths level
                    0.6, # path probability
                    0.4,
                    0.0
                ],
                [ # paths level
                    1.0, # path probability
                    0.0,
                    0.0
                ]
            ],
        ),
        (
            [ # batch level
                [1], # one-element input sequence
            ],
            3,
            [ # batch level
                [ # paths level
                    [1, 1], # path
                    [1, 2],
                    [1, 3],
                ],
            ],
            [ # batch level
                [ # paths level
                    0.6, # path probability
                    0.4,
                    0.0
                ],
            ],
        ),
        (
            [ # batch level
                [2], # one-element input sequence
            ],
            3,
            [ # batch level
                [ # paths level
                    [2, 3], # path
                    [2, 1],
                    [2, 2],
                ],
            ],
            [ # batch level
                [ # paths level
                    1.0, # path probability
                    0.0,
                    0.0
                ]
            ],
        ),
        (
            [ # batch level
                [1], # one-element input sequence
                [2], # one-element input sequence
            ],
            1,
            [ # batch level
                [ # paths level
                    [1, 1], # path
                ],
                [ # paths level
                    [2, 3], # path
                ],
            ],
            [ # batch level
                [ # paths level
                    0.6, # path probability
                ],
                [ # paths level
                    1.0, # path probability
                ]
            ],
        ),
        (
            [ # batch level
                [1,1], # two-element input sequence
                [1,2], # two-element input sequence
            ],
            3,
            [ # batch level
                [ # paths level
                    [1, 1, 1], # path
                    [1, 1, 2],
                    [1, 1, 3],
                ],
                [ # paths level
                    [1, 2, 1], # path
                    [1, 2, 2],
                    [1, 2, 3],
                ],
            ],
            [ # batch level
                [ # paths level
                    0.6, # path probability
                    0.4,
                    0.0
                ],
                [ # paths level
                    0.95, # path probability
                    0.05,
                    0.0
                ]
            ],
        ),
    ]
)
def test_AutoregressionInitializer(probabilities, input, number_of_output_paths, expected_paths, expected_paths_probabilities):
    conditional_probability_model = MockModelLayer(probabilities, first_dim_is_batch=True, step_redundant=True, history_entry_dims=(1,))
    paths_initializer = AutoregressionInitializer(conditional_probability_model, number_of_output_paths, lambda i: i+1, lambda id: tf.expand_dims(id, 1), state_dtype=tf.int32)
    t_input = tf.constant(input)
    paths, paths_probabilities, states = paths_initializer.call(t_input)
    with tf.Session() as sess:
        result_paths, result_paths_probabilities, states = sess.run((paths, paths_probabilities, states))
    assert result_paths == approx(np.array(expected_paths))
    assert result_paths_probabilities == approx(np.array(expected_paths_probabilities))
    


@pytest.mark.parametrize("input, number_of_output_paths, expected_paths, expected_paths_probabilities, expected_states",
    [
        (
            [ # batch level
                [1], # one-element input sequence
                [2], # one-element input sequence
            ],
            3,
            [ # batch level
                [ # paths level
                    [1, 1], # path
                    [1, 2],
                    [1, 3],
                ],
                [ # paths level
                    [2, 3], # path
                    [2, 1],
                    [2, 2],
                ],
            ],
            [ # batch level
                [ # paths level
                    0.6, # path probability
                    0.4,
                    0.0
                ],
                [ # paths level
                    1.0, # path probability
                    0.0,
                    0.0
                ]
            ],
            (
                [
                    [
                        1,
                        1,
                        1,
                    ],
                    [
                        1,
                        1,
                        1,
                    ]
                ],
                [ # batch level
                    [
                        [1, 0, 0],
                        [1, 0, 0],
                        [1, 0, 0],
                    ],
                    [
                        [2, 0, 0],
                        [2, 0, 0],
                        [2, 0, 0],
                    ],
                ]
            )
        ),
        (
            [ # batch level
                [1], # one-element input sequence
            ],
            3,
            [ # batch level
                [ # paths level
                    [1, 1], # path
                    [1, 2],
                    [1, 3],
                ],
            ],
            [ # batch level
                [ # paths level
                    0.6, # path probability
                    0.4,
                    0.0
                ],
            ],
            (
                [
                    [
                        1,
                        1,
                        1,
                    ]
                ],
                [ # batch level
                    [
                        [1, 0, 0],
                        [1, 0, 0],
                        [1, 0, 0],
                    ]
                ]
            )
        ),
        (
            [ # batch level
                [2], # one-element input sequence
            ],
            3,
            [ # batch level
                [ # paths level
                    [2, 3], # path
                    [2, 1],
                    [2, 2],
                ],
            ],
            [ # batch level
                [ # paths level
                    1.0, # path probability
                    0.0,
                    0.0
                ]
            ],
            (
                [
                    [
                        1,
                        1,
                        1,
                    ]
                ],
                [ # batch level
                    [
                        [2, 0, 0],
                        [2, 0, 0],
                        [2, 0, 0],
                    ]
                ]
            )
        ),
        (
            [ # batch level
                [1], # one-element input sequence
                [2], # one-element input sequence
            ],
            1,
            [ # batch level
                [ # paths level
                    [1, 1], # path
                ],
                [ # paths level
                    [2, 3], # path
                ],
            ],
            [ # batch level
                [ # paths level
                    0.6, # path probability
                ],
                [ # paths level
                    1.0, # path probability
                ]
            ],
            (
                [
                    [
                        1,
                    ],
                    [
                        1,
                    ],
                ],
                [ # batch level
                    [
                        [1, 0, 0],
                    ],
                    [
                        [2, 0, 0],
                    ]
                ]
            )
        ),
        (
            [ # batch level
                [1,1], # two-element input sequence
                [1,2], # two-element input sequence
            ],
            3,
            [ # batch level
                [ # paths level
                    [1, 1, 1], # path
                    [1, 1, 2],
                    [1, 1, 3],
                ],
                [ # paths level
                    [1, 2, 1], # path
                    [1, 2, 2],
                    [1, 2, 3],
                ],
            ],
            [ # batch level
                [ # paths level
                    0.6, # path probability
                    0.4,
                    0.0
                ],
                [ # paths level
                    0.95, # path probability
                    0.05,
                    0.0
                ]
            ],
            (
                [
                    [
                        2,
                        2,
                        2,
                    ],
                    [
                        2,
                        2,
                        2,
                    ]
                ],
                [ # batch level
                    [
                        [1, 1, 0],
                        [1, 1, 0],
                        [1, 1, 0],
                    ],
                    [
                        [1, 2, 0],
                        [1, 2, 0],
                        [1, 2, 0],
                    ]
                ]
            )
        ),
    ]
)
def test_AutoregressionInitializer_with_explicit_zero_state(probabilities, input, number_of_output_paths, expected_paths, expected_paths_probabilities, expected_states):
    """Expected states are lacking one (last) dimension - 1 insted of [1] so they are expanded in the test"""
    expected_state_history = np.expand_dims(np.array(expected_states[1]), 3)
    expected_states = (np.array(expected_states[0]), expected_state_history)
    history_entry_dims = (1,)
    conditional_probability_model = MockModelLayer(probabilities, first_dim_is_batch=True, step_redundant=True, history_entry_dims=(1,))
    paths_initializer = AutoregressionInitializer(conditional_probability_model, number_of_output_paths, lambda i: i+1, lambda id: tf.expand_dims(id, 1))
    t_input = tf.constant(input)
    batch_size = len(input)
    history_size = 3 
    zero_state = (
        tf.zeros(
            shape=(batch_size,), 
            dtype=tf.int32
            ), 
        tf.zeros(
            shape=(batch_size, history_size, *history_entry_dims), 
            dtype=tf.int32
            )
        )
    paths, paths_probabilities, states = paths_initializer.call(t_input, conditional_probability_model_initial_state=zero_state)
    with tf.Session() as sess:
        result_paths, result_paths_probabilities, result_states = sess.run((paths, paths_probabilities, states))
    assert result_paths == approx(np.array(expected_paths))
    assert result_paths_probabilities == approx(np.array(expected_paths_probabilities))
    assert result_states[0] == approx(np.array(expected_states[0]))
    assert result_states[1] == approx(np.array(expected_states[1]))



@pytest.mark.parametrize("input_sequence, input_probabilities, input_states, output_sequence, output_pobabilities, output_states, steps", 
    [
        ( # case 1 - one-element batch, one patch
            # INPUT
            # input sequence
            [
                [ # 1-st batch element
                    [ # 1-st (and only) path
                        -1, # the only element of path
                    ],
                ],
            ],
            # input probabilities
            [
                [ # 1. batch element
                    1.0, # 1-st (and olny) path probability
                ],
            ],
            # input states (using MockProbabilityModel with redundant step argument)
            (
                [ # step tensor
                    [
                        0, # step for the first batch element of MockProbablityModel = 1-st patch of 1-st batch element for Extender
                    ]
                ],
                [
                    [
                        [ # history tensor for the first patch
                            [0],[0],[0],[0],
                        ],
                    ]
                ],
            ),
            # OUTPUT
            # output sequence
            [
                [ # 1-st batch element
                    [ # 1-st (and only) path
                        -1, 1, # a path extended with one element (of id 1 - it has the same conditional prob. as 2 so 1 is chosen as the first in prob-dist vector)
                    ],
                ],
            ],
            # output probabilities
            [
                [ # 1. batch element
                    0.5, # 1-st (and olny) path probability
                ],
            ],
            # output states (using MockProbabilityModel with redundant step argument)
            (
                [
                    [ # step tensor
                        1, # step for the first batch element of MockProbablityModel = 1-st patch of 1-st batch element for Extender
                    ]
                ],
                [
                    [
                        [ # history tensor for the first patch
                            [-1],[0],[0],[0],
                        ],
                    ]
                ],
            ),
            # STEPS
            1,
        ),
        ( # case 2 - one-element batch, two patches but one is zero-filled with zero probability (like an output from AutoregressorInitializer could be)
            # INPUT
            # input sequence
            [
                [ # 1-st batch element
                    [ # 1-st path
                        -1, 1, 
                    ],
                    [ # 2-nd path
                        0, 0,
                    ],
                ],
            ],
            # input probabilities
            [
                [ # 1. batch element
                    0.5, # 1-st path probability
                    0.0,
                ],
            ],
            # input states (using MockProbabilityModel with redundant step argument)
            (
                [ # step tensor
                    [
                        1, # step for the first batch element of MockProbablityModel = 1-st patch of 1-st batch element for Extender
                        1,
                    ]
                ],
                [
                    [ # first batch element
                        [ # history tensor for the first patch
                            [-1],[0],[0],[0],
                        ],
                        [
                            [-1],[0],[0],[0], # this element of mock probability model state doesn't really match the input patch because the patch is zero-filled and there is no such history i probability model
                        ],
                    ]
                ],
            ),
            # OUTPUT
            # output sequence
            [
                [ # 1-st batch element
                    [ # 1-st path
                        -1, 1, 1, # a path extended with one element
                    ],
                    [
                        -1, 1, 2,
                    ],
                ],
            ],
            # output probabilities
            [
                [ # 1. batch element
                    0.5 * 0.6, # 1-st (and olny) path probability
                    0.5 * 0.4
                ],
            ],
            # output states (using MockProbabilityModel with redundant step argument)
            (
                [ # step tensor
                    [
                        2, # step for the first batch element of MockProbablityModel = 1-st patch of 1-st batch element for Extender
                        2,
                    ]
                ],
                [
                    [ # first batch element
                        [
                            [-1],[1],[0],[0], # history tensor for the first patch
                        ],
                        [
                            [-1],[1],[0],[0],
                        ]
                    ],
                ],
            ),
            # STEPS
            1,
        ),
        ( # case 3 - one-element batch, three patches but two are zero-filled with zero probability (like an output from AutoregressorInitializer could be)
            # INPUT
            # input sequence
            [
                [ # 1-st batch element
                    [ # 1-st path
                        -1, 1,
                    ],
                    [
                        0, 0, 
                    ],
                    [
                        0, 0, 
                    ],
                ],
            ],
            # input probabilities
            [
                [ # 1. batch element
                    0.5, # 1-st path probability
                    0.0,
                    0.0,
                ],
            ],
            # input states (using MockProbabilityModel with redundant step argument)
            (
                [ # step tensor
                    [ # for the first sequence in batch
                        1, # step for the first batch element of MockProbablityModel = 1-st patch of 1-st batch element for Extender
                        1,
                        1,
                    ]
                ],
                [
                    [     
                        [ # history tensor for the first patch
                            [-1],[0],[0],[0],
                        ],
                        [
                            [-1],[0],[0],[0], # this element of mock probability model state doesn't really match the input patch because the patch is zero-filled and there is no such history i probability model
                        ],
                        [
                            [-1],[0],[0],[0], # this element of mock probability model state doesn't really match the input patch because the patch is zero-filled and there is no such history i probability model
                        ],
                    ]
                ],
            ),
            # OUTPUT
            # output sequence
            [
                [ # 1-st batch element
                    [ # 1-st path
                        -1, 1, 1, # a path extended with one element
                    ],
                    [
                        -1, 1, 2,
                    ],
                    [
                        -1, 1, 3,
                    ],
                ],
            ],
            # output probabilities
            [
                [ # 1. batch element
                    0.5 * 0.6, # 1-st (and olny) path probability
                    0.5 * 0.4,
                    0.5 * 0.0,
                ],
            ],
            # output states (using MockProbabilityModel with redundant step argument)
            (
                [ # step tensor
                    [
                        2, # step for the first batch element of MockProbablityModel = 1-st patch of 1-st batch element for Extender
                        2,
                        2,
                    ]
                ],
                [
                    [
                        [ # history tensor for the first patch
                            [-1],[1],[0],[0],
                        ],
                        [
                            [-1],[1],[0],[0],
                        ],
                        [
                            [-1],[1],[0],[0],
                        ],
                    ]
                ],
            ),
            # STEPS
            1,
        ),
        ( # case 4 - cont. of 2-nd
            # INPUT
            # input sequence
            [
                [ # 1-st batch element
                    [ # 1-st path
                        -1, 1, 1, 
                    ],
                    [
                        -1, 1, 2,
                    ],
                ],
            ],
            # input probabilities
            [
                [ # 1. batch element
                    0.6, # 1-st path probability
                    0.4
                ],
            ],
            # input states (using MockProbabilityModel with redundant step argument)
            (
                [ # step tensor
                    [
                        2, # step for the first batch element of MockProbablityModel = 1-st patch of 1-st batch element for Extender
                        2,
                    ]
                ],
                [
                    [
                        [ 
                            [-1],[1],[0],[0], # history tensor for the first patch
                        ],
                        [
                            [-1],[1],[0],[0],
                        ],
                    ]
                ],
            ),
            # OUTPUT
            # output sequence
            [
                [ # 1-st batch element
                    [ # 1-st path
                        -1, 1, 2, 1, # a path extended with one element
                    ],
                    [
                        -1, 1, 1, 1,
                    ],
                ],
            ],
            # output probabilities
            [
                [ # 1. batch element
                    0.38, # 1-st path probability
                    0.36
                ],
            ],
            # output states (using MockProbabilityModel with redundant step argument)
            (
                [ # step tensor
                    [
                        3, # step for the first batch element of MockProbablityModel = 1-st patch of 1-st batch element for Extender
                        3,
                    ]
                ],
                [
                    [
                        [ 
                            [-1],[1],[2],[0], # history tensor for the first patch
                        ],
                        [
                            [-1],[1],[1],[0],
                        ],
                    ]
                ],
            ),
            # STEPS
            1,
        ),
        ( # case 5 - like 4-th but with 3 paths, additional third path is zero-filled with zero probability (like after broadcasting)
            # INPUT
            # input sequence
            [
                [ # 1-st batch element
                    [ # 1-st path
                        -1, 1, 1, 
                    ],
                    [
                        -1, 1, 2,
                    ],
                    [
                        0, 0, 0,
                    ],
                ],
            ],
            # input probabilities
            [
                [ # 1. batch element
                    0.6, # 1-st path probability
                    0.4,
                    0.0,
                ],
            ],
            # input states (using MockProbabilityModel with redundant step argument)
            (
                [ # step tensor
                    [
                        2, # step for the first batch element of MockProbablityModel = 1-st patch of 1-st batch element for Extender
                        2,
                        2,
                    ]
                ],
                [
                    [
                        [ 
                            [-1],[1],[0],[0], # history tensor for the first patch
                        ],
                        [
                            [-1],[1],[0],[0],
                        ],
                        [
                            [-1],[0],[0],[0],
                        ],
                    ]
                ],
            ),
            # OUTPUT
            # output sequence
            [
                [ # 1-st batch element
                    [ # 1-st path
                        -1, 1, 2, 1, # a path extended with one element
                    ],
                    [
                        -1, 1, 1, 1,
                    ],
                    [
                        -1, 1, 1, 2,
                    ],
                ],
            ],
            # output probabilities
            [
                [ # 1. batch element
                    0.38, # 1-st path probability
                    0.36,
                    0.24,
                ],
            ],
            # output states (using MockProbabilityModel with redundant step argument)
            (
                [ # step tensor
                    [
                        3, # step for the first batch element of MockProbablityModel = 1-st patch of 1-st batch element for Extender
                        3,
                        3,
                    ]
                ],
                [
                    [
                        [ 
                            [-1],[1],[2],[0], # history tensor for the first patch
                        ],
                        [
                            [-1],[1],[1],[0],
                        ],
                        [
                            [-1],[1],[1],[0],
                        ],
                    ]
                ],
            ),
            # STEPS
            1,
        ),
        ( # case 6 - like 4-th but with 4 paths, additional third and fourth path is zero-filled with zero probability (like after broadcasting)
            # INPUT
            # input sequence
            [
                [ # 1-st batch element
                    [ # 1-st path
                        -1, 1, 1, 
                    ],
                    [
                        -1, 1, 2,
                    ],
                    [
                        0, 0, 0,
                    ],
                    [
                        0, 0, 0,
                    ],
                ],
            ],
            # input probabilities
            [
                [ # 1. batch element
                    0.6, # 1-st path probability
                    0.4,
                    0.0,
                    0.0,
                ],
            ],
            # input states (using MockProbabilityModel with redundant step argument)
            (
                [ # step tensor
                    [
                        2, # step for the first batch element of MockProbablityModel = 1-st patch of 1-st batch element for Extender
                        2,
                        2,
                        2,
                    ]
                ],
                [
                    [
                        [ 
                            [-1],[1],[0],[0], # history tensor for the first patch
                        ],
                        [
                            [-1],[1],[0],[0],
                        ],
                        [
                            [-1],[0],[0],[0],
                        ],
                        [
                            [-1],[0],[0],[0],
                        ],
                    ]
                ],
            ),
            # OUTPUT
            # output sequence
            [
                [ # 1-st batch element
                    [ # 1-st path
                        -1, 1, 2, 1, # a path extended with one element
                    ],
                    [
                        -1, 1, 1, 1,
                    ],
                    [
                        -1, 1, 1, 2,
                    ],
                    [
                        -1, 1, 2, 2,
                    ],
                ],
            ],
            # output probabilities
            [
                [ # 1. batch element
                    0.38, # 1-st path probability
                    0.36,
                    0.24,
                    0.02,
                ],
            ],
            # output states (using MockProbabilityModel with redundant step argument)
            (
                [ # step tensor
                    [
                        3, # step for the first batch element of MockProbablityModel = 1-st patch of 1-st batch element for Extender
                        3,
                        3,
                        3,
                    ]
                ],
                [
                    [
                        [ 
                            [-1],[1],[2],[0], # history tensor for the first patch
                        ],
                        [
                            [-1],[1],[1],[0],
                        ],
                        [
                            [-1],[1],[1],[0],
                        ],
                        [
                            [-1],[1],[2],[0],
                        ],
                    ]
                ],
            ),
            # STEPS
            1,
        ),
        ( # case 7 - cases 2, 4 at once
            # INPUT
            # input sequence
            [
                [ # 1-st batch element
                    [ # 1-st path
                        -1, 1, 
                    ],
                    [ # 2-nd path
                        0, 0,
                    ],
                ],
            ],
            # input probabilities
            [
                [ # 1. batch element
                    0.5, # 1-st path probability
                    0.0,
                ],
            ],
            # input states (using MockProbabilityModel with redundant step argument)
            (
                [ # step tensor
                    [
                        1, # step for the first batch element of MockProbablityModel = 1-st patch of 1-st batch element for Extender
                        1,
                    ]
                ],
                [
                    [
                        [ # history tensor for the first patch
                            [-1],[0],[0],[0],
                        ],
                        [
                            [-1],[0],[0],[0], # this element of mock probability model state doesn't really match the input patch because the patch is zero-filled and there is no such history i probability model
                        ],
                    ]
                ],
            ),
            # OUTPUT
            # output sequence
            [
                [ # 1-st batch element
                    [ # 1-st path
                        -1, 1, 2, 1, # a path extended with one element
                    ],
                    [
                        -1, 1, 1, 1,
                    ],
                ],
            ],
            # output probabilities
            [
                [ # 1. batch element
                    0.19, # 1-st path probability
                    0.18
                ],
            ],
            # output states (using MockProbabilityModel with redundant step argument)
            (
                [ # step tensor
                    [
                        3, # step for the first batch element of MockProbablityModel = 1-st patch of 1-st batch element for Extender
                        3,
                    ]
                ],
                [
                    [
                        [ 
                            [-1],[1],[2],[0], # history tensor for the first patch
                        ],
                        [
                            [-1],[1],[1],[0],
                        ],
                    ]
                ],
            ),
            # NUMBER OF STEPS TO MAKE (how much elents to produce)
            2,
        ),
        ( # case 8 - cases 2 and 4 in one batch
            # INPUT
            # input sequence
            [
                [ # 1-st batch element
                    [ # 1-st path
                        -1, 1, 
                    ],
                    [ # 2-nd path
                        0, 0,
                    ],
                ],
                [ # 2-nd batch element
                    [ # 1-st path
                        1, 1, # this is kind of inconsistent with the input state of mock cond. prob. model state, initiaj "-1" was removed to match dims
                    ],
                    [
                        1, 2,
                    ],
                ],
            ],
            # input probabilities
            [
                [ # 1. batch element
                    0.5, # 1-st path probability
                    0.0,
                ],
                [ # 2. batch element
                    0.6, # 1-st path probability
                    0.4
                ],
            ],
            # input states (using MockProbabilityModel with redundant step argument)
            (
                [ # step tensor
                    [
                        1, # step for the first batch element of MockProbablityModel = 1-st patch of 1-st batch element for Extender
                        1,
                    ],
                    [
                        2,
                        2,
                    ]
                ],
                [
                    [
                        [ # history tensor for the first patch
                            [-1],[0],[0],[0],
                        ],
                        [
                            [-1],[0],[0],[0], # this element of mock probability model state doesn't really match the input patch because the patch is zero-filled and there is no such history i probability model
                        ],
                    ],
                    [
                        [ 
                            [-1],[1],[0],[0], # history tensor for the first patch
                        ],
                        [
                            [-1],[1],[0],[0],
                        ]
                    ]
                ],
            ),
            # OUTPUT
            # output sequence
            [
                [ # 1-st batch element
                    [ # 1-st path
                        -1, 1, 1, # a path extended with one element
                    ],
                    [
                        -1, 1, 2,
                    ],
                ],
                [ # 2-nd batch element
                    [ # 1-st path
                        1, 2, 1, # a path extended with one element
                    ],
                    [
                        1, 1, 1,
                    ],
                ],
            ],
            # output probabilities
            [
                [ # 1. batch element
                    0.5 * 0.6, # 1-st (and olny) path probability
                    0.5 * 0.4
                ],
                [ # 2. batch element
                    0.38, # 1-st path probability
                    0.36
                ],
            ],
            # output states (using MockProbabilityModel with redundant step argument)
            (
                [ # step tensor
                    [
                        2, # step for the first batch element of MockProbablityModel = 1-st patch of 1-st batch element for Extender
                        2,
                    ],
                    [
                        3,
                        3,
                    ]
                ],
                [
                    [
                        [ 
                            [-1],[1],[0],[0], # history tensor for the first patch
                        ],
                        [
                            [-1],[1],[0],[0],
                        ],
                    ],
                    [
                        [ 
                            [-1],[1],[2],[0], # history tensor for the first patch
                        ],
                        [
                            [-1],[1],[1],[0],
                        ],
                    ]
                ],
            ),
            # STEPS
            1,
        ),
    ]
)
def test_AutoregressionExtender(probabilities_with_start_element_no_third, input_sequence, input_probabilities, input_states, output_sequence, output_pobabilities, output_states, steps):
    model = MockModelLayer(probabilities_with_start_element_no_third, first_dim_is_batch=True, step_redundant=True, history_entry_dims=(1,))
    #nested_tuple_apply(input_states, repeat_in_ith_dimension, 1, len(input_sequence[0]))
    #steps_to_make = len(input_sequence[0][0])
    numbers_of_alternatives = len(input_sequence[0])
    regresor_step = AutoregressionWithAlternativePathsStep(
        numbers_of_alternatives, 
        model, 
        steps, 
        probability_model_initial_input=-1,
        index_in_probability_distribution_to_element_id_mapping=lambda x: x+1)
    
    extender = AutoregressionExtender(regresor_step, steps)

    paths, paths_probabilities = tf.constant(input_sequence), tf.constant(input_probabilities)
    states = tuple(tf.constant(t) for t in input_states)
    new_paths, new_path_probabilities, new_states = extender.call(paths, paths_probabilities, states)

    with tf.Session() as sess:
        r_paths, r_probabilities, r_states  = sess.run((new_paths, new_path_probabilities, new_states))
    assert r_paths == approx(np.array(output_sequence))
    assert r_probabilities == approx(np.array(output_pobabilities))
    assert r_states[0] == approx(np.array(output_states[0]))
    assert r_states[1] == approx(np.array(output_states[1]))


@pytest.mark.parametrize("input, expected_paths, expected_paths_probabilities, steps", 
    [
        (
            [ -1, -1],
            # OUTPUT
            # output sequence
            [
                [ # 1-st batch element
                    [ # 1-st path
                        -1, 1, 2, 1, # a path extended with one element
                    ],
                    [
                        -1, 1, 1, 1,
                    ],
                ],
                [ # 2-nd batch element
                    [ # 1-st path
                        -1, 1, 2, 1, # a path extended with one element
                    ],
                    [
                        -1, 1, 1, 1,
                    ],
                ],
            ],
            # output probabilities
            [
                [ # 1. batch element
                    0.19, # 1-st path probability
                    0.18
                ],
                [ # 2. batch element
                    0.19, # 1-st path probability
                    0.18
                ],
            ],
            # Number of steps to take
            3
        ),
        (
            [ -1, -1],
            # OUTPUT
            # output sequence
            [
                [ # 1-st batch element
                    [ # 1-st path
                        -1, 1, # a path extended with one element
                    ],
                    [
                        -1, 2,
                    ],
                ],
                [ # 2-nd batch element
                    [ # 1-st path
                        -1, 1, # a path extended with one element
                    ],
                    [
                        -1, 2,
                    ],
                ],
            ],
            # output probabilities
            [
                [ # 1. batch element
                    0.5, # 1-st path probability
                    0.5
                ],
                [ # 2. batch element
                    0.5, # 1-st path probability
                    0.5
                ],
            ],
            # Number of steps to take
            1
        ),
    ]
)
def test_AutoregressionWithAlternativePaths_no_mask(probabilities_with_start_element_no_third, input, expected_paths, expected_paths_probabilities, steps):
    conditional_probability_model = MockModelLayer(probabilities_with_start_element_no_third, first_dim_is_batch=True, step_redundant=True, history_entry_dims=(1,))
    initial_state = conditional_probability_model.zero_state(2, tf.int32)
    autoregressor = AutoregressionWithAlternativePaths(
        conditional_probability_model,
        len(expected_paths[0]),
        steps,
        index_in_probability_distribution_to_id_mapping=lambda x: x+1,
        id_to_embedding_mapping=lambda id: tf.expand_dims(id, 1),
        conditional_probability_model_initial_state=initial_state
        )
    t_input = tf.constant(input)
    paths, paths_probabilities = autoregressor.call(t_input)
    with tf.Session() as sess:
        r_paths, r_paths_probabilities = sess.run((paths, paths_probabilities))
    assert r_paths == approx(np.array(expected_paths))
    assert r_paths_probabilities == approx(np.array(expected_paths_probabilities))
