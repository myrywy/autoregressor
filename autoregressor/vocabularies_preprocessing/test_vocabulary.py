from vocabularies_preprocessing.mock_vocabulary import MockVocab
from generalized_vocabulary import SpecialUnit

import tensorflow as tf
import numpy as np
import pytest
from pytest import approx

@pytest.mark.parametrize("id, default, expected_output",
    [
        ([1,2,3], 0, [[1.5,2.5,3.5],[4.5,5.5,6.5],[7.5,8.5,9.5]]),
        ([1,2,3,5], 0, [[1.5,2.5,3.5],[4.5,5.5,6.5],[7.5,8.5,9.5],[0.0,0.0,0.0]]),
        ([1], 0, [[1.5,2.5,3.5]]),
        ([1,2,3], 1.2, [[1.5,2.5,3.5],[4.5,5.5,6.5],[7.5,8.5,9.5]]),
        ([1,2,3,5], 1.2, [[1.5,2.5,3.5],[4.5,5.5,6.5],[7.5,8.5,9.5],[1.2,1.2,1.2]]),
        ([1], 1.2, [[1.5,2.5,3.5]]),
        ([1,2,3], [1.2,2.2,3.2], [[1.5,2.5,3.5],[4.5,5.5,6.5],[7.5,8.5,9.5]]),
        ([1,2,3,5], [1.2,2.2,3.2], [[1.5,2.5,3.5],[4.5,5.5,6.5],[7.5,8.5,9.5],[1.2,2.2,3.2]]),
        ([1], [1.2,2.2,3.2], [[1.5,2.5,3.5]]),
    ]
)
def test_id_to_vecor_or_default(id, default, expected_output):
    vocab = MockVocab()
    t_vector = vocab.id_to_vecor_or_default(id, default=default)
    with tf.Session() as sess:
        sess.run(tf.tables_initializer())
        r_vector = sess.run(t_vector)
    assert r_vector == approx(expected_output)


@pytest.mark.parametrize("vocab_type, default",
    [
        (MockVocab, 0.0),
        (MockVocab, 1.1),
        (MockVocab, [1.1,2.1,3.1]),
    ]
)
def test_subclass_non_id_with_id_to_vecor_or_default(vocab_type, default):
    """This test checks if subclass functions get_non_id_integer, get_valid_id_example, vector_size works well together with base's id_to_vecor_or_default.
    It is intended to be used with each subclass."""
    vocab = vocab_type()
    not_id = vocab.get_non_id_integer()
    valid_id = vocab.get_valid_id_example()
    vector_size = vocab.vector_size()
    expected_output_for_non_id = np.array(default) if isinstance(default, list) else np.array([default]*vector_size)
    test_input_ids = tf.constant([not_id, valid_id])
    t_embedding_or_default = vocab.id_to_vecor_or_default(test_input_ids, default=default)
    t_valid_vector = vocab.id_to_vector_op(tf.constant([valid_id]))
    with tf.Session() as sess:
        r_embedding_or_default = sess.run(t_embedding_or_default)
        r_valid_vector = sess.run(t_valid_vector)
    assert r_embedding_or_default[0] == approx(expected_output_for_non_id)
    assert r_embedding_or_default[1] == approx(r_valid_vector[0])
