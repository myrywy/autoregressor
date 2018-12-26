import pytest
from pytest import approx
import tensorflow as tf
import numpy as np
from element_probablity_mask import ElementProbabilityMasking


@pytest.fixture
def allowed():
    return [
            [0,1,2],    # 0
            [7,8,9],    # 1
            [0,9],      # 2
            [0,0],      # 3
            [0],        # 4
            [5],        # 5
            [9],        # 6
            [],         # 7
            [4],        # 8
        ]
        

@pytest.fixture
def expected_mask():
    return np.array(
        [  # 0 1 2 3 4 5 6 7 8 9
            [1,1,1,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,1,1,1],
            [1,0,0,0,0,0,0,0,0,1],
            [1,0,0,0,0,0,0,0,0,0],
            [1,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,1,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,1],
            [1,1,1,1,1,1,1,1,1,1],
            [0,0,0,0,1,0,0,0,0,0],
        ]
    )
BATCH_SIZE = 9


@pytest.mark.parametrize("steps,mask_slice", 
    [   # note: step is counted from 1.
        (tf.range(BATCH_SIZE) + 1, np.arange(BATCH_SIZE)),
        (tf.range(BATCH_SIZE-1, -1, -1) + 1, np.arange(BATCH_SIZE-1, -1, -1)),
        (tf.zeros((BATCH_SIZE,), dtype=tf.int64) + 1, np.zeros((BATCH_SIZE,), dtype=np.int64)),
        (tf.ones((BATCH_SIZE,), dtype=tf.int64) + 1, np.ones((BATCH_SIZE,), dtype=np.int64)),
        (2*tf.ones((BATCH_SIZE,), dtype=tf.int64) + 1, 2*np.ones((BATCH_SIZE,), dtype=np.int64)),
        (3*tf.ones((BATCH_SIZE,), dtype=tf.int64) + 1, 3*np.ones((BATCH_SIZE,), dtype=np.int64)),
        (4*tf.ones((BATCH_SIZE,), dtype=tf.int64) + 1, 4*np.ones((BATCH_SIZE,), dtype=np.int64)),
        (5*tf.ones((BATCH_SIZE,), dtype=tf.int64) + 1, 5*np.ones((BATCH_SIZE,), dtype=np.int64)),
        (6*tf.ones((BATCH_SIZE,), dtype=tf.int64) + 1, 6*np.ones((BATCH_SIZE,), dtype=np.int64)),
        (7*tf.ones((BATCH_SIZE,), dtype=tf.int64) + 1, 7*np.ones((BATCH_SIZE,), dtype=np.int64)),
        (8*tf.ones((BATCH_SIZE,), dtype=tf.int64) + 1, 8*np.ones((BATCH_SIZE,), dtype=np.int64)),
    ]
)
def test_call(allowed, expected_mask, steps, mask_slice):
    DISTRIBUTION_SIZE = 10
    masking = ElementProbabilityMasking(allowed, DISTRIBUTION_SIZE, 0, 10, tf.identity)
    np.random.seed(0)
    probabilities = np.random.rand(len(expected_mask), DISTRIBUTION_SIZE)
    t_probabilities = tf.constant(probabilities, dtype=tf.float32)
    masked = masking.call(t_probabilities, steps)
    with tf.Session() as sess:
        r_masked = sess.run(masked)
    expected_masked = probabilities * np.array(expected_mask[mask_slice])
    assert expected_masked == approx(r_masked)


input_distribution_int = tf.constant(
    [
        [1,-2,3],
        [4,-5,6],
        [7,-8,9]
    ],
    dtype=tf.int32
)

input_distribution_float = tf.constant(
    [
        [0.1,-0.2,0.3],
        [0.4,-0.5,0.6],
        [0.7,-0.8,0.9]
    ],
    dtype=tf.float32
)


@pytest.mark.parametrize("input, allowed_ids, masked_value, step, expected_output",
    [
        (input_distribution_float, [[],[],[]], 0, [1,1,1],
            np.array(
                [
                    [0.1,-0.2,0.3],
                    [0.4,-0.5,0.6],
                    [0.7,-0.8,0.9]
                ],
                dtype=np.float32
            )
        ), 
        (input_distribution_int, [[],[],[]], 0, [1,1,1],
            np.array(
                [
                    [1,-2,3],
                    [4,-5,6],
                    [7,-8,9]
                ],
                dtype=np.int32
            )
        ), 
        (input_distribution_float, [[],[1],[0,2]], 16, [2,2,2],
            np.array(
                [
                    [0.1,16,0.3],
                    [0.4,16,0.6],
                    [0.7,16,0.9]
                ],
                dtype=np.float32
            )
        ), 
        (input_distribution_int, [[],[1],[0,2]], 16, [2,2,2],
            np.array(
                [
                    [1,16,3],
                    [4,16,6],
                    [7,16,9]
                ],
                dtype=np.int32
            )
        ), 
        (input_distribution_float, [[],[1],[0,2]], "min", [0,0,0],
            np.array(
                [
                    [0.1,-0.2,0.3],
                    [0.4,-0.5,0.6],
                    [0.7,-0.8,0.9]
                ],
                dtype=np.float32
            )
        ), 
        (input_distribution_int, [[],[1],[0,2]], "min", [0,0,0],
            np.array(
                [
                    [1,-2,3],
                    [4,-5,6],
                    [7,-8,9]
                ],
                dtype=np.int32
            )
        ), 
        (input_distribution_float, [[],[1],[0,2]], "min", [2,2,2],
            np.array(
                [
                    [0.1,-0.8,0.3],
                    [0.4,-0.8,0.6],
                    [0.7,-0.8,0.9]
                ],
                dtype=np.float32
            )
        ), 
        (input_distribution_int, [[],[1],[0,2]], "min", [2,2,2],
            np.array(
                [
                    [1,-8,3],
                    [4,-8,6],
                    [7,-8,9]
                ],
                dtype=np.int32
            )
        ), 
        (input_distribution_float, [[],[1],[0,2]], "min", [1,1,1],
            np.array(
                [
                    [-0.8,-0.2,-0.8],
                    [-0.8,-0.5,-0.8],
                    [-0.8,-0.8,-0.8]
                ],
                dtype=np.float32
            )
        ), 
        (input_distribution_int, [[],[1],[0,2]], "min", [1,1,1],
            np.array(
                [
                    [-8,-2,-8],
                    [-8,-5,-8],
                    [-8,-8,-8]
                ],
                dtype=np.int32
            )
        ), 


        (input_distribution_float, [[],[1],[0,2]], "<min", [0,0,0],
            np.array(
                [
                    [0.1,-0.2,0.3],
                    [0.4,-0.5,0.6],
                    [0.7,-0.8,0.9]
                ],
                dtype=np.float32
            )
        ), 
        (input_distribution_int, [[],[1],[0,2]], "<min", [0,0,0],
            np.array(
                [
                    [1,-2,3],
                    [4,-5,6],
                    [7,-8,9]
                ],
                dtype=np.int32
            )
        ), 
        (input_distribution_float, [[],[1],[0,2]], "<min", [2,2,2],
            np.array(
                [
                    [0.1,-1.8,0.3],
                    [0.4,-1.8,0.6],
                    [0.7,-1.8,0.9]
                ],
                dtype=np.float32
            )
        ), 
        (input_distribution_int, [[],[1],[0,2]], "<min", [2,2,2],
            np.array(
                [
                    [1,-9,3],
                    [4,-9,6],
                    [7,-9,9]
                ],
                dtype=np.int32
            )
        ), 
        (input_distribution_float, [[],[1],[0,2]], "<min", [1,1,1],
            np.array(
                [
                    [-1.8,-0.2,-1.8],
                    [-1.8,-0.5,-1.8],
                    [-1.8,-0.8,-1.8]
                ],
                dtype=np.float32
            )
        ), 
        (input_distribution_int, [[],[1],[0,2]], "<min", [1,1,1],
            np.array(
                [
                    [-9,-2,-9],
                    [-9,-5,-9],
                    [-9,-8,-9]
                ],
                dtype=np.int32
            )
        ), 
    ]
)
def test_call_non_default_values(input, allowed_ids, masked_value, step, expected_output):
    DISTRIBUTION_SIZE = 3
    t_step = tf.convert_to_tensor(step) + 1
    mask = ElementProbabilityMasking(
        allowed_ids, 
        DISTRIBUTION_SIZE, 
        0, 
        DISTRIBUTION_SIZE, 
        tf.identity, 
        masked_value=masked_value)
    t_output = mask.call(input, t_step)
    with tf.Session() as sess:
        r_output = sess.run(t_output)
    assert r_output == approx(expected_output)
