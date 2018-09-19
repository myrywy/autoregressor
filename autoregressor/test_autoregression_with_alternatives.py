import numpy as np
from mock_prob_model import mock_model
from autoregression_with_alternatives import *

def test_step_call():
    probabilities = {
        (0,0,0):[0.5, 0.5, 0.0], # początek historii

        (1,0,0):[0.6, 0.4, 0.0], # a
        (2,0,0):[0.0, 0.0, 1.0], # b
        
        (1,1,0):[0.6, 0.4, 0.0], # aa
        (1,2,0):[0.95, 0.05, 0.0], # ab
        (2,1,0):[0.0, 0.0, 1.0], # ba
        (2,2,0):[0.0, 0.0, 1.0], # bb
        }

    model = mock_model(probabilities)
    regresor = AutoregressionWithAlternativePathsStep(2, model, 3)
    zero_state = regresor.zero_state()
    input = tf.zeros(1, tf.int32)
    output1, state1 = regresor.call(input, zero_state)
    output2, state2 = regresor.call(input, state1)
    output3, state3 = regresor.call(input, state2)

    with tf.Session() as sess:
        r_zero, r_s1, r_s2, r_s3, r_o1, r_o2, r_o3 = sess.run((zero_state, state1, state2, state3, output1, output2, output3))
    print(state3)


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