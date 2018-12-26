from corpora_preprocessing.simple_examples import text_dataset_to_token_ids, token_ids_to_text_dataset
from vocabularies_preprocessing.glove300d import get_words_to_id_op, get_id_to_word_op

import tensorflow as tf
import numpy as np
from pytest import approx
from itertools import islice

s1 = " no it was n't black monday "
s2 = " but while the new york stock exchange did n't fall apart friday as the dow jones industrial average plunged N points most of it in the final hour it barely managed to stay this side of chaos "
s3 = " some circuit breakers installed after the october N crash failed their first test traders say unable to cool the selling panic in both stocks and futures "

def get_test_dataset():
    def make_example():
        yield s1
        yield s2
        yield s3
    return tf.data.Dataset.from_generator(make_example, tf.string)


def test_words_to_id_glove():
    dataset = get_test_dataset()
    dataset = text_dataset_to_token_ids(dataset, get_words_to_id_op)
    it = dataset.make_initializable_iterator()
    next_element = it.get_next()
    with tf.Session() as sess:
        sess.run(tf.tables_initializer())
        sess.run(it.initializer)
        l1 = sess.run(next_element)
        l2 = sess.run(next_element)
        l3 = sess.run(next_element)
    assert l1 == approx(np.array([96, 21, 30, 40, 536, 21594]))
    assert l2 == approx(np.array([42, 212, 2, 94, 8707, 1126, 2616, 127, 40, 1213, 3195, 15014, 28, 2, 70717, 23749, 3660, 1149, 29571, 1630, 504, 122, 5, 21, 7, 2, 1009, 1193, 21, 5333, 2470, 4, 767, 27, 401, 5, 10436]))
    assert l3 == approx(np.array([85, 3972, 35185, 2545, 149, 2, 28428, 1630, 4575, 2377, 58, 106, 804, 12445, 216, 3644, 4, 970, 2, 1929, 8618, 7, 227, 5881, 3, 14521]))


def test_words_to_id_glove_estimator():
    def get_input():
        dataset = get_test_dataset()
        dataset = text_dataset_to_token_ids(dataset, get_words_to_id_op)
        return dataset.batch(1)

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
    r = estimator.train(get_input, max_steps=1)
    r = estimator.predict(get_input)
    l1, l2, l3 = islice(r, 3)
    assert l1 == approx(np.array([96, 21, 30, 40, 536, 21594]))
    assert l2 == approx(np.array([42, 212, 2, 94, 8707, 1126, 2616, 127, 40, 1213, 3195, 15014, 28, 2, 70717, 23749, 3660, 1149, 29571, 1630, 504, 122, 5, 21, 7, 2, 1009, 1193, 21, 5333, 2470, 4, 767, 27, 401, 5, 10436]))
    assert l3 == approx(np.array([85, 3972, 35185, 2545, 149, 2, 28428, 1630, 4575, 2377, 58, 106, 804, 12445, 216, 3644, 4, 970, 2, 1929, 8618, 7, 227, 5881, 3, 14521]))

def test_two_way_words_id_transformation_glove_estimator():
    def get_input():
        dataset = get_test_dataset()
        dataset = text_dataset_to_token_ids(dataset, get_words_to_id_op)
        dataset = token_ids_to_text_dataset(dataset, get_id_to_word_op)
        return dataset.batch(1)

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
    r = estimator.train(get_input, max_steps=1)
    r = estimator.predict(get_input)
    l1, l2, l3 = islice(r, 3)
    assert (l1 == s1.encode().split()).all()
    assert (l2 == s2.encode().split()).all()
    assert (l3 == s3.encode().split()).all()
