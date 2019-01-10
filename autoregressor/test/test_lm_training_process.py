import tensorflow as tf 
from tensorflow.contrib.training import HParams
import numpy as np

import pytest
from pytest import approx

from lm_training_process import LanguageModel

@pytest.fixture
def hparams():
    return HParams(
        learning_rate=0.1,
        rnn_num_units=100,
        rnn_num_layers=6,
        rnn_last_layer_num_units=50, # None to set the same as in previous layers
        max_training_steps=25,
        rnn_layer="lstm_block_cell",
        profiler=False,
        size_based_device_assignment=False,
        batch_size=5,
        shuffle_examples_buffer_size=1000,
        shuffle_examples_seed=0,
        cached_data_dir=None,
        write_target_text_to_summary=False,
        mask_padding_cost=True,
        dynamic_rnn_swap_memory=True,
        predict_top_k=5,
        words_as_text_preview=True,
        time_major_optimization=True,
    )


@pytest.mark.parametrize("time_major_optimization, inputs, targets, length, expected_inputs, expected_targets",
    [
        (
            True,
            # input
            np.array([
                [
                    [0.1, 0.2, 0.3],
                    [1.1, 1.2, 1.3],
                    [2.1, 2.2, 2.3],
                ],
                [
                    [3.1, 3.2, 3.3],
                    [4.1, 4.2, 4.3],
                    [5.1, 5.2, 5.3],
                ],
            ]),
            np.array([
                [0,1,2],
                [3,4,5],
            ]),
            np.array([3, 3]),
            # expected output
            np.array([
                [
                    [0.1, 0.2, 0.3],
                    [3.1, 3.2, 3.3],
                ],
                [
                    [1.1, 1.2, 1.3],
                    [4.1, 4.2, 4.3],
                ],
                [
                    [2.1, 2.2, 2.3],
                    [5.1, 5.2, 5.3],
                ],
            ]),
            np.array([
                [0,3,],
                [1,4,],
                [2,5,],
            ]),
        ),
        (
            False,
            # input
            np.array([
                [
                    [0.1, 0.2, 0.3],
                    [1.1, 1.2, 1.3],
                    [2.1, 2.2, 2.3],
                ],
                [
                    [3.1, 3.2, 3.3],
                    [4.1, 4.2, 4.3],
                    [5.1, 5.2, 5.3],
                ],
            ]),
            np.array([
                [0,1,2],
                [3,4,5],
            ]),
            np.array([3, 3]),
            # expected output
            np.array([
                [
                    [0.1, 0.2, 0.3],
                    [1.1, 1.2, 1.3],
                    [2.1, 2.2, 2.3],
                ],
                [
                    [3.1, 3.2, 3.3],
                    [4.1, 4.2, 4.3],
                    [5.1, 5.2, 5.3],
                ],
            ]),
            np.array([
                [0,1,2],
                [3,4,5],
            ]),
        ),
    ]
)
def test_maybe_transpose_batch_time(hparams, time_major_optimization, inputs, targets, length, expected_inputs, expected_targets):
    hparams.set_hparam("time_major_optimization", time_major_optimization)
    lm = LanguageModel({"inputs": inputs, "length": length}, {"targets": targets}, tf.estimator.ModeKeys.TRAIN, hparams)
    inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
    targets = tf.convert_to_tensor(targets, dtype=tf.int32)
    t_inputs, t_targets = lm.maybe_transpose_batch_time(inputs, targets)
    with tf.Session() as sess:
        r_inputs, r_targets = sess.run((t_inputs, t_targets))
    
    assert r_inputs == approx(expected_inputs)
    assert (r_targets == expected_targets).all()
