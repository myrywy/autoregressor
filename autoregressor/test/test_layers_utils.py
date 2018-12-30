import tensorflow as tf
import numpy as np

from pytest import approx

from layers_utils import rnn_cell_extension, AffineProjectionLayer, AffineProjectionPseudoCell

def test_rnn_cell_extension():
    @rnn_cell_extension
    class TestLayer(tf.keras.layers.Layer):
        def __init__(self, a, b, c=None):
            self.a = a
            self.b = b
            self.c = c
        
        def call(self, input):
            return self.a * input + self.b

        @property
        def output_size(self):
            return self.b.shape[0]
    
    a = tf.constant(2)
    b = tf.constant([1,1,1])
    c = -1
    input = np.array(
        [
            #first step - one input tensor for TestLayer
            [
                [1,2,3],
                [4,5,6],
                [7,8,9],
                [10,11,12],
            ],
            #second step - one input tensor for TestLayer
            [
                [-1,-2,-3],
                [-4,-5,-6],
                [-7,-8,-9],
                [-10,-11,-12],
            ],
        ],
        dtype=np.int32
    )
    input = np.transpose(input, (1,0,2))

    expected_output = np.array(
        [
            #first step
            [
                [3,5,7],
                [9,11,13],
                [15,17,19],
                [21,23,25]
            ],
            #second step
            [
                [-1,-3,-5],
                [-7,-9,-11],
                [-13,-15,-17],
                [-19,-21,-23]
            ],
        ],
        dtype=np.int32
    )

    test_layer = TestLayer(a, b, c= c)

    output, final_state = tf.nn.dynamic_rnn(test_layer, input, dtype=tf.int32)
    output = tf.transpose(output, (1,0,2))

    with tf.Session() as sess:
        r_output, r_final_state = sess.run((output, final_state))

    assert r_output == approx(expected_output)
    assert r_final_state == ()
    assert test_layer.wrapped_layer.c == c


def test_rnn_cell_extension_with_multi_rnn_cell():
    @rnn_cell_extension
    class TestLayer(tf.keras.layers.Layer):
        def __init__(self, a, b, c=None):
            self.a = a
            self.b = b
            self.c = c
        
        def call(self, input):
            return self.a * input + self.b

        @property
        def output_size(self):
            return self.b.shape[0]
    
    a_1 = tf.constant(2)
    b_1 = tf.constant([0,0,0])
    a_2 = tf.constant(1)
    b_2 = tf.constant([1,1,1])
    input = np.array(
        [
            #first step - one input tensor for TestLayer
            [
                [1,2,3],
                [4,5,6],
                [7,8,9],
                [10,11,12],
            ],
            #second step - one input tensor for TestLayer
            [
                [-1,-2,-3],
                [-4,-5,-6],
                [-7,-8,-9],
                [-10,-11,-12],
            ],
        ],
        dtype=np.int32
    )
    input = np.transpose(input, (1,0,2))

    expected_output = np.array(
        [
            #first step
            [
                [3,5,7],
                [9,11,13],
                [15,17,19],
                [21,23,25]
            ],
            #second step
            [
                [-1,-3,-5],
                [-7,-9,-11],
                [-13,-15,-17],
                [-19,-21,-23]
            ],
        ],
        dtype=np.int32
    )

    test_layer_1 = TestLayer(a_1, b_1)
    test_layer_2 = TestLayer(a_2, b_2)

    cell = tf.nn.rnn_cell.MultiRNNCell([test_layer_1, test_layer_2])

    output, final_state = tf.nn.dynamic_rnn(cell, input, dtype=tf.int32)
    output = tf.transpose(output, (1,0,2))

    with tf.Session() as sess:
        r_output, r_final_state = sess.run((output, final_state))

    assert r_output == approx(expected_output)
    assert r_final_state == ((), ())


def test_AffineProjectionLayer():
    input = np.array(
        [
            [1,2,3],
            [4,5,6],
            [7,8,9],
            [10,11,12],
            [-1,-2,-3],
            [-4,-5,-6],
            [-7,-8,-9],
            [-10,-11,-12],
        ],
        dtype=np.int32
    )

    expected_output = np.array(
        [
            [3,5,7],
            [9,11,13],
            [15,17,19],
            [21,23,25],
            [-1,-3,-5],
            [-7,-9,-11],
            [-13,-15,-17],
            [-19,-21,-23],
        ],
        dtype=np.int32
    )
    w_initializer=tf.constant_initializer(
        [2,0,0,
         0,2,0,
         0,0,2]
        )
    b_initializer=tf.constant_initializer([1,1,1])
    test_layer = AffineProjectionLayer(3, 3, tf.int32, w_initializer=w_initializer, b_initializer=b_initializer)

    output = test_layer(input)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        r_output = sess.run(output)

    assert r_output == approx(expected_output)


def test_AffineProjectionPseudoCell():
    input = np.array(
        [
            [
                [1,2,3],
                [4,5,6],
                [7,8,9],
                [10,11,12],
            ],
            [
                [-1,-2,-3],
                [-4,-5,-6],
                [-7,-8,-9],
                [-10,-11,-12],
            ]
        ],
        dtype=np.int32
    )

    expected_output = np.array(
        [
            [
                [3,5,7],
                [9,11,13],
                [15,17,19],
                [21,23,25],
            ],
            [
                [-1,-3,-5],
                [-7,-9,-11],
                [-13,-15,-17],
                [-19,-21,-23],
            ]
        ],
        dtype=np.int32
    )
    w_initializer=tf.constant_initializer(
        [2,0,0,
         0,2,0,
         0,0,2]
        )
    b_initializer=tf.constant_initializer([1,1,1])
    test_layer = AffineProjectionPseudoCell(3, 3, tf.int32, w_initializer=w_initializer, b_initializer=b_initializer)

    output, final_state = tf.nn.dynamic_rnn(test_layer, input, dtype=tf.int32)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        r_output, r_final_state = sess.run((output, final_state))

    assert r_output == approx(expected_output)
    assert r_final_state == ()


def test_AffineProjectionPseudoCell_learnt():
    input = np.array(
        [
            [
                [1,2,3],
                [4,5,6],
                [7,8,9],
                [10,11,12],
            ],
            [
                [-1,-2,-3],
                [-4,-5,-6],
                [-7,-8,-9],
                [-10,-11,-12],
            ]
        ],
        dtype=np.float32
    )

    expected_output = np.array(
        [
            [
                [3,5,7],
                [9,11,13],
                [15,17,19],
                [21,23,25],
            ],
            [
                [-1,-3,-5],
                [-7,-9,-11],
                [-13,-15,-17],
                [-19,-21,-23],
            ]
        ],
        dtype=np.float32
    )
    w_initializer=tf.truncated_normal_initializer(0, 0.3, 0)
    b_initializer=tf.truncated_normal_initializer(0, 0.3, 0)
    test_layer = AffineProjectionPseudoCell(3, 3, tf.float32, w_initializer=w_initializer, b_initializer=b_initializer)

    output, final_state = tf.nn.dynamic_rnn(test_layer, input, dtype=tf.float32)

    loss = (output - tf.constant(expected_output))*(output - tf.constant(expected_output))
    optimizer_1 = tf.train.AdamOptimizer(learning_rate=0.1)
    train_op_1 = optimizer_1.minimize(
        loss=loss, 
        global_step=tf.train.get_global_step()
        ) 
    optimizer_2 = tf.train.AdamOptimizer(learning_rate=0.01)
    train_op_2 = optimizer_2.minimize(
        loss=loss, 
        global_step=tf.train.get_global_step()
        ) 

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for _ in range(600):
            sess.run(train_op_1)
        for _ in range(400):
            sess.run(train_op_2)
        r_output, r_final_state = sess.run((output, final_state))

    assert r_output == approx(expected_output)
    assert r_final_state == ()