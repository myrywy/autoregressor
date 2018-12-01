"""
VOCABULARY:
id  |   vector
1   |   [1,0,0] (start mark)       
2   |   [0,0,1] (stop mark)
5   |   [1,2,3]
6   |   [1,-2,3]
7   |   [-1,2,-3]
"""
import tensorflow as tf
import numpy as np
import itertools
import pytest
from pytest import approx

from lstm_lm import language_model_fn, language_model_input_dataset

SEED = 0

@pytest.fixture
def embedding_lookup_fn():
    return lambda t: tf.nn.embedding_lookup(
        [
            [[0,0,0]], # .
            [[1,0,0]], # 1 (start mark)       
            [[0,0,1]], # 2 (stop mark)
            [[0,0,0]], # .
            [[0,0,0]], # .
            [[1,2,3]], # 5 
            [[1,-2,3]],    # 6
            [[-1,2,-3]],   # 7
        ],
        t
    )

@pytest.fixture
def input_data_minimal():
    examples_raw = [
        [5,6,7],
        [7,6,5],
        [6, 6, 7, 5]
    ]
    examples = [np.array([1]+ex+[2]) for ex in examples_raw]

    def examples_generator():
        for example in examples:
            yield example

    input_data = tf.data.Dataset.from_generator(examples_generator, tf.int32)
    return input_data

@pytest.fixture
def input_data_raw(input_data_minimal):
    return input_data_minimal.repeat(500).shuffle(60, seed=SEED)

@pytest.fixture
def input_data(input_data_raw, embedding_lookup_fn):
    return language_model_input_dataset(input_data_raw, embedding_lookup_fn).padded_batch(20, tf.Dimension(None))

def test_language_model_input_dataset(input_data_minimal, embedding_lookup_fn):
    dataset = language_model_input_dataset(input_data_minimal, embedding_lookup_fn)
    iterator = dataset.make_one_shot_iterator()
    next_example = iterator.get_next()
    expected_features = [
        {"inputs": [[1,0,0], [1,2,3], [1,-2,3],[-1,2,-3]], "length": 4},
        {"inputs": [[1,0,0], [-1,2,-3], [1,-2,3],[1,2,3]], "length": 4},
        {"inputs": [[1,0,0], [1,-2,3], [1,-2,3], [-1,2,-3], [1,2,3]], "length": 5},
    ]
    expected_labels = [
        {"targets": [5,6,7,2], },
        {"targets": [7,6,5,2], },
        {"targets": [6,6,7,5,2], },
    ]
    expected = zip(expected_features, expected_labels)
    
    with tf.Session() as sess:
        for expected_feature, expected_label in expected:
            r_features, r_labels = sess.run(next_example)
            assert r_features["inputs"] == approx(expected_feature["inputs"])
            assert r_features["length"] == approx(expected_feature["length"])
            assert r_labels["targets"] == approx(expected_label["targets"])

def test_language_model_fn(input_data, input_data_minimal, embedding_lookup_fn):
    estimator = tf.estimator.Estimator(language_model_fn)
    estimator.train(lambda: input_data, steps=1000)
    def predict_input():
        return language_model_input_dataset(input_data_minimal, embedding_lookup_fn).padded_batch(3, tf.Dimension(None))
    predictions = estimator.predict(predict_input)
    predictions = itertools.islice(predictions, 3)
    assert False
