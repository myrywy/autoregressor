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

    assert r_s1.path_probabilities == approx([[0.5, 0.5]])
    assert r_s2.path_probabilities == approx([[0.3, 0.2]]) # [0.5*0.6, 0.5*0.4]
    assert r_s3.path_probabilities == approx([[0.19, 0.18]])


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

    assert r_s1.path_probabilities == approx([[0.5, 0.5]])
    assert r_s2.path_probabilities == approx([[0.3, 0.2]]) # [0.5*0.6, 0.5*0.4]
    assert r_s3.path_probabilities == approx([[0.19, 0.18]])


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

    conditional_probability, new_probability_model_states = regresor._compute_next_step_probability(zero_state_for_one_batch_element.step, zero_state_for_one_batch_element.paths, zero_state_for_one_batch_element.probability_model_states)

    input = tf.zeros((1, 1), tf.int32)
    output1, state1 = regresor.call(input, zero_state)
    output2, state2 = regresor.call(input, state1)
    output3, state3 = regresor.call(input, state2)

    with tf.Session() as sess:
        r_cp, r_nsp = sess.run((conditional_probability, new_probability_model_states))
        r_zero, r_s1, r_s2, r_s3, r_o1, r_o2, r_o3 = sess.run((zero_state, state1, state2, state3, output1, output2, output3))

    assert r_s3.path_probabilities == approx([0.18])


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
    current_autoregressor_step = tf.constant([target_step-1], name="init_step") 
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
    next_step_probability_dist, _ = regresor._compute_next_step_probability(current_autoregressor_step, paths, model_state)
    with tf.Session() as sess:
        r_prob_dist = sess.run(next_step_probability_dist)
    assert r_prob_dist == approx([expected_prob_dist])


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

    assert result == approx(expected)
