import numpy as np
import pytest
from pytest import approx
import mock_prob_model
from mock_prob_model import MockModelLayer
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

@pytest.mark.skipif()
def test_step_call(probabilities):
    model = MockModelLayer(probabilities)
    regresor = AutoregressionWithAlternativePathsStep(2, model, 3)
    zero_state = regresor.zero_state(1, tf.int32)
    input = tf.zeros(1, tf.int32)
    output1, state1 = regresor.call(input, zero_state)
    output2, state2 = regresor.call(input, state1)
    output3, state3 = regresor.call(input, state2)

    with tf.Session() as sess:
        r_zero, r_s1, r_s2, r_s3, r_o1, r_o2, r_o3 = sess.run((zero_state, state1, state2, state3, output1, output2, output3))

    print(r_s3)
    assert r_s3.path_probabilities == approx([0.19, 0.18])


@pytest.mark.skipif()
def test_step_on_one_path(probabilities):
    model = MockModelLayer(probabilities)
    regresor = AutoregressionWithAlternativePathsStep(1, model, 3, False)
    zero_state = regresor.zero_state(1, tf.int32)
    input = tf.zeros(1, tf.int32)
    output1, state1 = regresor.call(input, zero_state)
    output2, state2 = regresor.call(input, state1)
    output3, state3 = regresor.call(input, state2)

    with tf.Session() as sess:
        r_zero, r_s1, r_s2, r_s3, r_o1, r_o2, r_o3 = sess.run((zero_state, state1, state2, state3, output1, output2, output3))

    print(r_s3)
    assert r_s3.path_probabilities == approx([0.18])


@pytest.mark.parametrize("target_step,path,expected_prob_dist", [
    #(0, [0,0,0], [0.5, 0.5, 0.0]),
    
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

    paths = tf.constant([path], name="init_path") 

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