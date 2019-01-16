import tensorflow as tf
import numpy as np

import pytest
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


@pytest.mark.parametrize("input, expected_output",
    [
        (
            np.array(
                [
                    [
                        [1,2,3],
                        [4,5,6],
                        [7,8,9],
                        [10,11,12],
                        [-1,-2,-3],
                        [-4,-5,-6],
                        [-7,-8,-9],
                        [-10,-11,-12],
                    ]
                ],
                dtype=np.int32
            ),
            np.array(
                [
                    [
                        [3,5,7],
                        [9,11,13],
                        [15,17,19],
                        [21,23,25],
                        [-1,-3,-5],
                        [-7,-9,-11],
                        [-13,-15,-17],
                        [-19,-21,-23],
                    ]
                ],
                dtype=np.int32
            )
        ),
        (
            np.array(
                [
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
                ],
                dtype=np.int32
            ),
            np.array(
                [
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
                ],
                dtype=np.int32
            )
        ),
        (
            np.array(
                [
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
                    [
                        [-1,-2,-3],
                        [-4,-5,-6],
                        [-7,-8,-9],
                        [-10,-11,-12],
                        [1,2,3],
                        [4,5,6],
                        [7,8,9],
                        [10,11,12],
                    ],
                ],
                dtype=np.float32
            ),
            np.array(
                [
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
                    [
                        [-1,-3,-5],
                        [-7,-9,-11],
                        [-13,-15,-17],
                        [-19,-21,-23],
                        [3,5,7],
                        [9,11,13],
                        [15,17,19],
                        [21,23,25],
                    ],
                ],
                dtype=np.float32
            )
        ),
    ]
)
def test_AffineProjectionLayer__higher_dim(input, expected_output):
    w_initializer=tf.constant_initializer(
        [2,0,0,
         0,2,0,
         0,0,2]
        )
    b_initializer=tf.constant_initializer([1,1,1])
    test_layer = AffineProjectionLayer(3, 3, tf.float32, w_initializer=w_initializer, b_initializer=b_initializer)

    output = test_layer(input)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        r_output = sess.run(output)

    assert r_output == approx(expected_output)



@pytest.mark.parametrize("input, expected_output",
    [
        (
            np.array(
                    [
                        [
                            [0.29572036, 0.213786  , 0.61282877, 0.26853551],
                            [0.04626981, 0.18281538, 0.37672273, 0.7598044 ],
                            [0.20735884, 0.27059426, 0.77088035, 0.16054048],
                            [0.86332122, 0.31865709, 0.06168338, 0.05183509],
                            [0.04390807, 0.1714842 , 0.6549949 , 0.44832607],
                        ],
                        [
                            [0.85000961, 0.52946506, 0.70820974, 0.59385658],
                            [0.83306494, 0.67819428, 0.88849439, 0.90106441],
                            [0.93006509, 0.51304136, 0.68790109, 0.30967898],
                            [0.91711842, 0.19691457, 0.97621936, 0.52497913],
                            [0.6720289 , 0.35742026, 0.79034631, 0.41658605],
                        ],
                        [
                            [0.84588703, 0.46057988, 0.9833772 , 0.3433403 ],
                            [0.02122789, 0.80089545, 0.88373024, 0.51811923],
                            [0.39006566, 0.19844857, 0.79809054, 0.67962612],
                            [0.45545568, 0.83677274, 0.36098646, 0.43998848],
                            [0.37643843, 0.88617167, 0.27792202, 0.23696645]
                        ],
                    ],
                    dtype=np.float32
                ),
            np.array(
                    [
                        [
                            [1.54678728, 1.75332976, 1.439962  ],
                            [1.70408111, 1.61878979, 1.34570108],
                            [1.49624524, 1.74937444, 1.45762503],
                            [1.4288187 , 1.73959515, 1.33232109],
                            [1.56669748, 1.64380139, 1.39890144],
                        ],
                        [
                            [2.32582618, 2.7097728 , 1.90907529],
                            [2.73580073, 3.10216493, 2.1258729 ],
                            [2.10037849, 2.57822557, 1.83741305],
                            [2.35437574, 2.84101075, 2.0410583 ],
                            [2.05028784, 2.42556313, 1.7921276 ],
                        ],
                        [
                            [2.22469824, 2.75023218, 1.97470614],
                            [1.92107416, 2.08554609, 1.60802673],
                            [2.07536495, 2.27046299, 1.7331012 ],
                            [1.84306321, 2.03105942, 1.4923397 ],
                            [1.57868333, 1.76560281, 1.34084286],
                        ],
                    ],
                    dtype=np.float32
                ),
            ),
    ]
)
def test_AffineProjectionLayer__higher_dim__nonsquare(input, expected_output):
    w_initializer=tf.constant_initializer(
        np.array(
                [
                    [0.61403865, 0.98941508, 0.48814617],
                    [0.28687176, 0.35473289, 0.10019905],
                    [0.54283428, 0.8517719 , 0.58251629],
                    [0.91094139, 0.66025953, 0.3422693 ]
                ],
                dtype=np.float32
            )
        )
    b_initializer=tf.constant_initializer(
        np.array(
                [0.7265898 , 0.68560919, 0.82529188],
                dtype=np.float32
            )
        )
    test_layer = AffineProjectionLayer(4, 3, tf.float32, w_initializer=w_initializer, b_initializer=b_initializer)

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