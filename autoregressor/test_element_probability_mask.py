import pytest
from pytest import approx
import tensorflow as tf
import numpy as np
from element_probablity_mask import ElementProbabilityMasking


@pytest.mark.parametrize("allowed,expected_mask", [
    (
        [
            [0,1,2],    # 0
            [7,8,9],    # 1
            [0,9],      # 2
            [0,0],      # 3
            [0],        # 4
            [5],        # 5
            [9],        # 6
            [],         # 7
            [4],        # 8
        ],
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
])
def test_call(allowed, expected_mask):
    DISTRIBUTION_SIZE = 10
    masking = ElementProbabilityMasking(allowed, DISTRIBUTION_SIZE, tf.identity)
    np.random.seed(0)
    probabilities = np.random.rand(len(expected_mask), DISTRIBUTION_SIZE)
    t_probabilities = tf.constant(probabilities, dtype=tf.float32)
    step = tf.range(len(expected_mask))
    masked = masking.call(t_probabilities, step)
    with tf.Session() as sess:
        r_masked = sess.run(masked)
    expected_masked = probabilities * np.array(expected_mask)
    assert expected_masked == approx(r_masked)
