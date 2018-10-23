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


BATCH_SIZE = len(expected_mask())

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
    masking = ElementProbabilityMasking(allowed, DISTRIBUTION_SIZE, tf.identity)
    np.random.seed(0)
    probabilities = np.random.rand(len(expected_mask), DISTRIBUTION_SIZE)
    t_probabilities = tf.constant(probabilities, dtype=tf.float32)
    masked = masking.call(t_probabilities, steps)
    with tf.Session() as sess:
        r_masked = sess.run(masked)
    expected_masked = probabilities * np.array(expected_mask[mask_slice])
    assert expected_masked == approx(r_masked)
