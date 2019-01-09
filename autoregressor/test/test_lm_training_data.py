import pickle

import tensorflow as tf

import pytest
from pytest import approx

from lm_training_data import LanguageModelTrainingData


def test_regression_load_training_data():
    td = LanguageModelTrainingData("glove300", "simple_examples", "data/simple_examples/cached_glove300/train_trimmed_to_40_tokens/", 5)
    dataset = td.load_training_data()
    it = dataset.make_initializable_iterator()
    next_example = it.get_next()
    with tf.Session() as sess:
        sess.run(tf.tables_initializer())
        sess.run(tf.global_variables_initializer())
        sess.run(it.initializer)
        actual = []
        for i in range(3000):
            actual.append(sess.run(next_example))
    with open("retest_expected.pickle", "rb") as rtest_expected:
        import pickle
        expected = pickle.load(rtest_expected)

    for (actual_features, actual_labels), (expected_features, expected_labels) in zip(actual, expected):
        assert actual_features["inputs"] == approx(expected_features["inputs"])
        assert (actual_features["length"] == expected_features["length"]).all()
        assert (actual_labels["targets"] == expected_labels["targets"]).all()