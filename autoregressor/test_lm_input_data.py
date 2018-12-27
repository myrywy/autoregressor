from data_pipeline import LmInputData
from vocabularies_preprocessing.mock_vocabulary import MockVocab

import tensorflow as tf
import numpy as np

import pytest
from pytest import approx

from itertools import islice


def test_in_out():
    def input_generator():
        yield ["a", "b", "c"]
        yield ["c", "b"]
        yield ["a", "b", "c", "d"]
        yield ["a"]
        yield ["d"]

    expected_output = [
        (
            {
                "inputs": np.array(
                    [
                        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.5, 2.5, 3.5],
                        [0.0, 0.0, 0.0, 4.5, 5.5, 6.5],
                        [0.0, 0.0, 0.0, 7.5, 8.5, 9.5]
                    ],
                    dtype=np.float32
                ),
                "length": np.array(4, dtype=np.int32),
            },
            {"targets": np.array([4, 5, 6, 2], dtype=np.int32)},
        ),
        (
            {
                "inputs": np.array(
                    [
                        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 7.5, 8.5, 9.5],
                        [0.0, 0.0, 0.0, 4.5, 5.5, 6.5],
                    ],
                    dtype=np.float32
                ),
                "length": np.array(3, dtype=np.int32),
            },
            {"targets": np.array([6, 5, 2], dtype=np.int32)},
        ),
        (
            {
                "inputs": np.array(
                    [
                        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.5, 2.5, 3.5],
                        [0.0, 0.0, 0.0, 4.5, 5.5, 6.5],
                        [0.0, 0.0, 0.0, 7.5, 8.5, 9.5],
                        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    ],
                    dtype=np.float32
                ),
                "length": np.array(5, dtype=np.int32),
            },
            {"targets": np.array([4, 5, 6, 7, 2], dtype=np.int32)},
        ),
        (
            {
                "inputs": np.array(
                    [
                        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.5, 2.5, 3.5],
                    ],
                    dtype=np.float32
                ),
                "length": np.array(2, dtype=np.int32),
            },
            {"targets": np.array([4, 2], dtype=np.int32)},
        ),
        (
            {
                "inputs": np.array(
                    [
                        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    ],
                    dtype=np.float32
                ),
                "length": np.array(2, dtype=np.int32),
            },
            {"targets": np.array([7, 2], dtype=np.int32)},
        ),
    ]

    input_dataset = tf.data.Dataset.from_generator(
        input_generator,
        output_types=tf.string
    )

    vocab = MockVocab()
    input_pipeline = LmInputData(vocab)
    input_data = input_pipeline.transform_dataset(input_dataset)

    it = input_data.make_initializable_iterator()
    example = it.get_next()

    with tf.Session() as sess:
        sess.run(tf.tables_initializer())
        sess.run(it.initializer)
        #sess.run(tf.global_variables_initializer())
        for _, expected in enumerate(expected_output):
            actual = sess.run(example)
            assert actual[0]["inputs"] == approx(expected[0]["inputs"])
            assert actual[0]["length"] == approx(expected[0]["length"])
            assert actual[1]["targets"] == approx(expected[1]["targets"])


def test_in_out_with_estimator():
    def input_generator():
        yield ["a", "b", "c"]
        yield ["c", "b"]
        yield ["a", "b", "c", "d"]
        yield ["a"]
        yield ["d"]

    expected_output = [
        (
            {
                "inputs": np.array(
                    [
                        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.5, 2.5, 3.5],
                        [0.0, 0.0, 0.0, 4.5, 5.5, 6.5],
                        [0.0, 0.0, 0.0, 7.5, 8.5, 9.5]
                    ],
                    dtype=np.float32
                ),
                "length": np.array(4, dtype=np.int32),
            },
            {"targets": np.array([4, 5, 6, 2], dtype=np.int32)},
        ),
        (
            {
                "inputs": np.array(
                    [
                        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 7.5, 8.5, 9.5],
                        [0.0, 0.0, 0.0, 4.5, 5.5, 6.5],
                    ],
                    dtype=np.float32
                ),
                "length": np.array(3, dtype=np.int32),
            },
            {"targets": np.array([6, 5, 2], dtype=np.int32)},
        ),
        (
            {
                "inputs": np.array(
                    [
                        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.5, 2.5, 3.5],
                        [0.0, 0.0, 0.0, 4.5, 5.5, 6.5],
                        [0.0, 0.0, 0.0, 7.5, 8.5, 9.5],
                        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    ],
                    dtype=np.float32
                ),
                "length": np.array(5, dtype=np.int32),
            },
            {"targets": np.array([4, 5, 6, 7, 2], dtype=np.int32)},
        ),
        (
            {
                "inputs": np.array(
                    [
                        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.5, 2.5, 3.5],
                    ],
                    dtype=np.float32
                ),
                "length": np.array(2, dtype=np.int32),
            },
            {"targets": np.array([4, 2], dtype=np.int32)},
        ),
        (
            {
                "inputs": np.array(
                    [
                        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    ],
                    dtype=np.float32
                ),
                "length": np.array(2, dtype=np.int32),
            },
            {"targets": np.array([7, 2], dtype=np.int32)},
        ),
    ]


    def input_fn():
        vocab = MockVocab()
        input_pipeline = LmInputData(vocab)
        input_dataset = tf.data.Dataset.from_generator(
            input_generator,
            output_types=tf.string
        )
        return input_pipeline.transform_dataset(input_dataset)

    def mock_model_fn(features, labels, mode, params):
        if mode == tf.estimator.ModeKeys.PREDICT:
            spec = tf.estimator.EstimatorSpec(mode=mode, predictions=features)
        else:
            spec = tf.estimator.EstimatorSpec(mode=mode,
                predictions=features,
                train_op=tf.train.get_global_step().assign_add(1), # without it "training" would last forever regardles for number of steps
                loss=tf.constant(0))
        return spec

    estimator = tf.estimator.Estimator(mock_model_fn)
    r = estimator.train(input_fn, max_steps=1)
    r = estimator.predict(input_fn)
    output = islice(r, 7)

    for actual, expected in zip(output, expected_output):
        assert actual[0]["inputs"] == approx(expected[0]["inputs"])
        assert actual[0]["length"] == approx(expected[0]["length"])
        assert actual[1]["targets"] == approx(expected[1]["targets"])
