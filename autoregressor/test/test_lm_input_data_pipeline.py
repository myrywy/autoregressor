from lm_input_data_pipeline import LmInputDataPipeline
from vocabularies_preprocessing.mock_vocabulary import MockVocab
from generalized_vocabulary import GeneralizedVocabulary
from lstm_lm import get_autoregressor_model_fn
from config import TEST_TMP_DIR

import tensorflow as tf
import numpy as np

import pytest
from pytest import approx

from itertools import islice
import shutil

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
    input_pipeline = LmInputDataPipeline(vocab)
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
        input_pipeline = LmInputDataPipeline(vocab)
        input_dataset = tf.data.Dataset.from_generator(
            input_generator,
            output_types=tf.string
        )
        return input_pipeline.load_data(input_dataset)

    def mock_model_fn(features, labels, mode, params):
        if mode == tf.estimator.ModeKeys.PREDICT:
            # Flattens features and labels into one dict (estimator's requirements)
            output = features.copy()
            spec = tf.estimator.EstimatorSpec(mode=mode, predictions=output)
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
        actual_length = int(actual["length"])
        assert actual["inputs"][:actual_length] == approx(expected[0]["inputs"][:actual_length])
        assert (actual["inputs"][actual_length:] == np.zeros_like(actual["inputs"][actual_length:])).all()
        assert actual["length"] == approx(expected[0]["length"])
        #assert actual["targets"] == approx(expected[1]["targets"])


def test_in_lm_learning():
    try:
        shutil.rmtree(TEST_TMP_DIR)
    except FileNotFoundError:
        pass
    def input_generator():
        yield ["c", "b", "a"]
        yield ["a", "b", "c", "d"]
        yield ["d"]

    def input_fn():
        vocab = MockVocab()
        input_pipeline = LmInputDataPipeline(vocab, batch_size=3)
        input_dataset = tf.data.Dataset.from_generator(
            input_generator,
            output_types=tf.string
        )
        return input_pipeline.load_data(input_dataset).repeat()
    
    def predict_input_fn():
        def input():
            yield {"inputs": [[1, 4], [1, 6]], "length": [4, 4]}, [0,0]
        return tf.data.Dataset.from_generator(input, 
            ({"inputs": tf.int32, "length": tf.int32}, tf.int32),
            ({"inputs": [2,2], "length": [2]}, [2]))

    expected_predictions = [
        [1, 4, 5, 6, 7, 2],
        [1, 6, 5, 4, 2]
    ]

    def model_fn(features, labels, mode, params):
        vocab_copy = MockVocab()
        input_pipeline_copy = LmInputDataPipeline(vocab_copy)
        return get_autoregressor_model_fn(vocab_size, input_pipeline_copy.get_id_to_embedding_mapping())(features, labels, mode, params)

    vocab_size = 8

    params = {"learning_rate": 0.05, "number_of_alternatives": 1}
    estimator = tf.estimator.Estimator(model_fn, params=params, model_dir=TEST_TMP_DIR)
    r = estimator.train(input_fn, max_steps=100)
    r = estimator.predict(predict_input_fn)
    output = [*islice(r, 2)]

    for actual, expected in zip(output, expected_predictions):
        relevant_length = len(expected)
        assert actual["paths"][0][:relevant_length] == approx(np.array(expected[:relevant_length]))



def test_in_lm_learning_with_batching_afterwards():
    try:
        shutil.rmtree(TEST_TMP_DIR)
    except FileNotFoundError:
        pass
    def input_generator():
        yield ["c", "b", "a"]
        yield ["a", "b", "c", "d"]
        yield ["d"]

    def input_fn():
        vocab = MockVocab()
        input_pipeline = LmInputDataPipeline(vocab, batch_size=None)
        input_dataset = tf.data.Dataset.from_generator(
            input_generator,
            output_types=tf.string
        )
        corpus = input_pipeline.load_data(input_dataset).repeat()
        corpus = input_pipeline.padded_batch(corpus, 3)
        return corpus
    
    def predict_input_fn():
        def input():
            yield {"inputs": [[1, 4], [1, 6]], "length": [4, 4]}, [0,0]
        return tf.data.Dataset.from_generator(input, 
            ({"inputs": tf.int32, "length": tf.int32}, tf.int32),
            ({"inputs": [2,2], "length": [2]}, [2]))

    expected_predictions = [
        [1, 4, 5, 6, 7, 2],
        [1, 6, 5, 4, 2]
    ]

    def model_fn(features, labels, mode, params):
        vocab_copy = MockVocab()
        input_pipeline_copy = LmInputDataPipeline(vocab_copy)
        return get_autoregressor_model_fn(vocab_size, input_pipeline_copy.get_id_to_embedding_mapping())(features, labels, mode, params)

    vocab_size = 8

    params = {"learning_rate": 0.05, "number_of_alternatives": 1}
    estimator = tf.estimator.Estimator(model_fn, params=params, model_dir=TEST_TMP_DIR)
    r = estimator.train(input_fn, max_steps=100)
    r = estimator.predict(predict_input_fn)
    output = [*islice(r, 2)]

    for actual, expected in zip(output, expected_predictions):
        relevant_length = len(expected)
        assert actual["paths"][0][:relevant_length] == approx(np.array(expected[:relevant_length]))



def test_load_with_batching():
    def input_generator():
        yield ["a", "b", "c"]
        yield ["c", "b"]

    expected_output = [
        (
            {
                "inputs": np.array(
                    [
                        [
                            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.5, 2.5, 3.5],
                            [0.0, 0.0, 0.0, 4.5, 5.5, 6.5],
                            [0.0, 0.0, 0.0, 7.5, 8.5, 9.5]
                        ],
                        [
                            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 7.5, 8.5, 9.5],
                            [0.0, 0.0, 0.0, 4.5, 5.5, 6.5],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                        ],
                    ],
                    dtype=np.float32
                ),
                "length": np.array([4,3], dtype=np.int32),
            },
            {"targets": np.array([[4, 5, 6, 2], [6, 5, 2, 0]], dtype=np.int32)},
        ),
    ]

    input_dataset = tf.data.Dataset.from_generator(
        input_generator,
        output_types=tf.string
    )

    vocab = MockVocab()
    input_pipeline = LmInputDataPipeline(vocab, batch_size=2)
    input_data = input_pipeline.load_data(input_dataset)

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


def test_load_no_batching():
    def input_generator():
        yield ["a", "b", "c"]
        yield ["c", "b"]

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
    ]

    input_dataset = tf.data.Dataset.from_generator(
        input_generator,
        output_types=tf.string
    )

    vocab = MockVocab()
    input_pipeline = LmInputDataPipeline(vocab, batch_size=None)
    input_data = input_pipeline.load_data(input_dataset)

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


def test_padded_batch_called_externally():
    def input_generator():
        yield ["a", "b", "c"]
        yield ["c", "b"]

    expected_output = [
        (
            {
                "inputs": np.array(
                    [
                        [
                            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.5, 2.5, 3.5],
                            [0.0, 0.0, 0.0, 4.5, 5.5, 6.5],
                            [0.0, 0.0, 0.0, 7.5, 8.5, 9.5]
                        ],
                        [
                            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 7.5, 8.5, 9.5],
                            [0.0, 0.0, 0.0, 4.5, 5.5, 6.5],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                        ],
                    ],
                    dtype=np.float32
                ),
                "length": np.array([4,3], dtype=np.int32),
            },
            {"targets": np.array([[4, 5, 6, 2], [6, 5, 2, 0]], dtype=np.int32)},
        ),
    ] * 2

    input_dataset = tf.data.Dataset.from_generator(
        input_generator,
        output_types=tf.string
    )

    vocab = MockVocab()
    input_pipeline = LmInputDataPipeline(vocab, batch_size=None)
    input_data = input_pipeline.load_data(input_dataset)
    input_data = input_data.repeat(2)
    input_data = input_pipeline.padded_batch(input_data,2)

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
