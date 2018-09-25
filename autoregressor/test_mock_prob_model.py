from mock_prob_model import *
import pytest
from pytest import approx

# P (a) = 0.5,
# P (a|a) = 0.6, P (b|a) = 0.4, P (a|aa) = 0.6, P (b|aa) = 0.4, P (a|ab) = 0.95,
# P (b|ab) = 0.05

def test_mock_model():
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

    initial_state = (tf.constant(0), tf.constant([0,0,0]))
    with tf.Session() as sess:
        pr_estimation, new_state = sess.run(model(tf.constant(1), initial_state))

    assert pr_estimation == approx([0.6,0.4,0.0])
    assert new_state == (1, approx([1,0,0]))


    initial_state = (tf.constant(0), tf.constant([0,0,0]))
    with tf.Session() as sess:
        pr_estimation, new_state = sess.run(model(tf.constant(2), initial_state))

    assert pr_estimation == approx([0.0,0.0,1.0])
    assert new_state == (1, approx([2,0,0]))


    initial_state = (tf.constant(1), tf.constant([1,0,0]))
    with tf.Session() as sess:
        pr_estimation, new_state = sess.run(model(tf.constant(1), initial_state))

    assert pr_estimation == approx([0.6,0.4,0.0])
    assert new_state == (2, approx([1,1,0]))


    initial_state = (tf.constant(1), tf.constant([1,0,0]))
    with tf.Session() as sess:
        pr_estimation, new_state = sess.run(model(tf.constant(2), initial_state))

    assert pr_estimation == approx([0.95,0.05,0.0])
    assert new_state == (2, approx([1,2,0]))


def test_mock_model_first_dim_batch():
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

    model = mock_model(probabilities, first_dim_is_batch=True)

    initial_state = (tf.constant(0), tf.constant([[[0],[0],[0]]]))
    with tf.Session() as sess:
        pr_estimation, new_state = sess.run(model(tf.constant([[1]]), initial_state))

    assert pr_estimation == approx([[0.6,0.4,0.0]])
    assert new_state == (1, approx([[[1],[0],[0]]]))


    initial_state = (tf.constant(0), tf.constant([[[0],[0],[0]]]))
    with tf.Session() as sess:
        pr_estimation, new_state = sess.run(model(tf.constant([[2]]), initial_state))

    assert pr_estimation == approx([[0.0,0.0,1.0]])
    assert new_state == (1, approx([[[2],[0],[0]]]))


    initial_state = (tf.constant(1), tf.constant([[[1],[0],[0]]]))
    with tf.Session() as sess:
        pr_estimation, new_state = sess.run(model(tf.constant([[1]]), initial_state))

    assert pr_estimation == approx([[0.6,0.4,0.0]])
    assert new_state == (2, approx([[[1],[1],[0]]]))


    initial_state = (tf.constant(1), tf.constant([[[1],[0],[0]]]))
    with tf.Session() as sess:
        pr_estimation, new_state = sess.run(model(tf.constant([[2]]), initial_state))

    assert pr_estimation == approx([[0.95,0.05,0.0]])
    assert new_state == (2, approx([[[1],[2],[0]]]))


def test_mock_model_layer_step():
    probabilities = {
        (0,0,0):[0.5, 0.5, 0.0], # początek historii

        (1,0,0):[0.6, 0.4, 0.0], # a
        (2,0,0):[0.0, 0.0, 1.0], # b
        
        (1,1,0):[0.6, 0.4, 0.0], # aa
        (1,2,0):[0.95, 0.05, 0.0], # ab
        (2,1,0):[0.0, 0.0, 1.0], # ba
        (2,2,0):[0.0, 0.0, 1.0], # bb
        }

    model = MockModelLayer(probabilities, first_dim_is_batch=False)

    initial_state = (tf.constant(0), tf.constant([0,0,0]))
    with tf.Session() as sess:
        pr_estimation, new_state = sess.run(model(tf.constant(1), initial_state))

    assert pr_estimation == approx([0.6,0.4,0.0])
    assert new_state == (1, approx([1,0,0]))


    initial_state = (tf.constant(0), tf.constant([0,0,0]))
    with tf.Session() as sess:
        pr_estimation, new_state = sess.run(model(tf.constant(2), initial_state))

    assert pr_estimation == approx([0.0,0.0,1.0])
    assert new_state == (1, approx([2,0,0]))


    initial_state = (tf.constant(1), tf.constant([1,0,0]))
    with tf.Session() as sess:
        pr_estimation, new_state = sess.run(model(tf.constant(1), initial_state))

    assert pr_estimation == approx([0.6,0.4,0.0])
    assert new_state == (2, approx([1,1,0]))


    initial_state = (tf.constant(1), tf.constant([1,0,0]))
    with tf.Session() as sess:
        pr_estimation, new_state = sess.run(model(tf.constant(2), initial_state))

    assert pr_estimation == approx([0.95,0.05,0.0])
    assert new_state == (2, approx([1,2,0]))


def test_mock_model_layer_step_deeper_input():
    probabilities = {
        ((0,0),(0,0),(0,0)):[0.5, 0.5, 0.0], # początek historii

        ((1,1),(0,0),(0,0)):[0.6, 0.4, 0.0], # a
        ((2,2),(0,0),(0,0)):[0.0, 0.0, 1.0], # b
        
        ((1,1),(1,1),(0,0)):[0.6, 0.4, 0.0], # aa
        ((1,1),(2,2),(0,0)):[0.95, 0.05, 0.0], # ab
        ((2,2),(1,1),(0,0)):[0.0, 0.0, 1.0], # ba
        ((2,2),(2,2),(0,0)):[0.0, 0.0, 1.0], # bb
        }

    model = MockModelLayer(probabilities, first_dim_is_batch=False)

    initial_state = (tf.constant(0), tf.constant([(0,0),(0,0),(0,0)]))
    with tf.Session() as sess:
        pr_estimation, new_state = sess.run(model(tf.constant([1,1]), initial_state))

    assert pr_estimation == approx([0.6,0.4,0.0])
    assert new_state == (1, approx([[1,1],[0,0],[0,0]]))


    initial_state = (tf.constant(0), tf.constant([(0,0),(0,0),(0,0)]))
    with tf.Session() as sess:
        pr_estimation, new_state = sess.run(model(tf.constant([2,2]), initial_state))

    assert pr_estimation == approx([0.0,0.0,1.0])
    assert new_state == (1, approx([[2,2],[0,0],[0,0]]))


    initial_state = (tf.constant(1), tf.constant([(1,1),(0,0),(0,0)]))
    with tf.Session() as sess:
        pr_estimation, new_state = sess.run(model(tf.constant([1,1]), initial_state))

    assert pr_estimation == approx([0.6,0.4,0.0])
    assert new_state == (2, approx([[1,1],[1,1],[0,0]]))


    initial_state = (tf.constant(1), tf.constant([(1,1),(0,0),(0,0)]))
    with tf.Session() as sess:
        pr_estimation, new_state = sess.run(model(tf.constant([2,2]), initial_state))

    assert pr_estimation == approx([0.95,0.05,0.0])
    assert new_state == (2, approx([[1,1],[2,2],[0,0]]))



def test_mock_model_layer_as_rnn():
    probabilities = {
        (0,0,0):[0.5, 0.5, 0.0], # początek historii

        (1,0,0):[0.6, 0.4, 0.0], # a
        (2,0,0):[0.0, 0.0, 1.0], # b
        
        (1,1,0):[0.6, 0.4, 0.0], # aa
        (1,2,0):[0.95, 0.05, 0.0], # ab
        (2,1,0):[0.0, 0.0, 1.0], # ba
        (2,2,0):[0.0, 0.0, 1.0], # bb
        }
    def expand_dim(t):
        return tuple((e,) for e in t)
    probabilities = {expand_dim(k):v for k, v in probabilities.items()}

    model = MockModelLayer(probabilities, history_entry_dims=(1,))
    inputs = tf.constant(
        [   # batch
            [[1],[1]]
        ]
    )
    initial_state = (tf.constant(0, tf.int32), tf.constant([[[0],[0],[0]]],dtype=tf.int32))
    outputs, final_state = tf.nn.dynamic_rnn(model, inputs, sequence_length=[2], initial_state=initial_state, dtype=tf.float32)
    with tf.Session() as sess:
        r_outputs, r_final_state = sess.run((outputs, final_state))

    print("r_outputs")
    print(r_outputs)
    print("r_final_state")
    print(r_final_state)
    assert r_outputs == approx([[[0.6, 0.4, 0.0],[0.6, 0.4, 0.0]]])
    assert r_final_state == (2, approx([[[1],[1],[0]]]))


def test_naive_lookup_op():
    k1 = [9,8,7]
    k2 = [[1,2,3],[5,6,7]]
    d1_keys = tf.constant(k1)
    d2_keys = tf.constant(k2)
    vals_1 = tf.constant([0.3,0.2,0.3])
    vals_2 = tf.constant([[0.3,1.1],[0.2,1.2],[0.3,1.3]])

    vals_3 = tf.constant([0.3,0.2])
    vals_4 = tf.constant([[0.3,1.1],[0.2,1.2]])

    l_1_1 = naive_lookup_op(d1_keys, vals_1)
    l_1_2 = naive_lookup_op(d1_keys, vals_2)
    
    l_2_3 = naive_lookup_op(d2_keys, vals_3)
    l_2_4 = naive_lookup_op(d2_keys, vals_4)

    e_1_1 = [0.3,0.2,0.3]
    e_1_2 = [[0.3,1.1],[0.2,1.2],[0.3,1.3]]

    e_2_3 = [0.3,0.2]
    e_2_4 = [[0.3,1.1],[0.2,1.2]]

    with tf.Session() as sess:
        for i in range(len(e_1_1)):
            assert sess.run(l_1_1(tf.constant(k1[i]))) == approx(e_1_1[i])

        for i in range(len(e_1_2)):
            assert sess.run(l_1_2(tf.constant(k1[i]))) == approx(e_1_2[i])

        for i in range(len(e_2_3)):
            assert sess.run(l_2_3(tf.constant(k2[i]))) == approx(e_2_3[i])

    with tf.Session() as sess:
        for i in range(len(e_2_4)):
            assert sess.run(l_2_4(tf.constant(k2[i]))) == approx(e_2_4[i])



def test_naive_lookup_op_batched():
    k1 = [9,8,7]
    k2 = [[1,2,3],[5,6,7]]
    d1_keys = tf.constant(k1)
    d2_keys = tf.constant(k2)
    vals_1 = tf.constant([0.3,0.2,0.3])
    vals_2 = tf.constant([[0.3,1.1],[0.2,1.2],[0.3,1.3]])

    vals_3 = tf.constant([0.3,0.2])
    vals_4 = tf.constant([[0.3,1.1],[0.2,1.2]])

    l_1_1 = naive_lookup_op(d1_keys, vals_1, first_dim_is_batch=True)
    l_1_2 = naive_lookup_op(d1_keys, vals_2, first_dim_is_batch=True)
    
    l_2_3 = naive_lookup_op(d2_keys, vals_3, first_dim_is_batch=True)
    l_2_4 = naive_lookup_op(d2_keys, vals_4, first_dim_is_batch=True)

    e_1_1 = [0.3,0.2,0.3]
    e_1_2 = [[0.3,1.1],[0.2,1.2],[0.3,1.3]]

    e_2_3 = [0.3,0.2]
    e_2_4 = [[0.3,1.1],[0.2,1.2]]

    with tf.Session() as sess:
        for i in range(len(e_1_1)):
            assert sess.run(l_1_1(tf.constant([k1[i]]))) == approx([e_1_1[i]])

        for i in range(len(e_1_2)):
            assert sess.run(l_1_2(tf.constant([k1[i]]))) == approx([e_1_2[i]])

        for i in range(len(e_2_3)):
            assert sess.run(l_2_3(tf.constant([k2[i]]))) == approx([e_2_3[i]])

    with tf.Session() as sess:
        for i in range(len(e_2_4)):
            assert sess.run(l_2_4(tf.constant([k2[i]]))) == approx([e_2_4[i]])