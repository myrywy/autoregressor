from vocabularies_preprocessing.mock_vocabulary import MockVocab
from generalized_vocabulary import SpecialUnit

import tensorflow as tf
import numpy as np
import pytest

@pytest.mark.parametrize("word, expected_output",
    [
        (["a", "b", "c", "d"], [1,2,3,4]),
        (["a"], [1]),
        (["d"], [4]),
    ]
)
def test_word_to_id_op(word, expected_output):
    vocab = MockVocab()
    t_id = vocab.word_to_id_op(word)
    with tf.Session() as sess:
        sess.run(tf.tables_initializer())
        r_id = sess.run(t_id)
    assert (r_id == expected_output).all()


@pytest.mark.parametrize("id, expected_output",
    [
        ([1,2,3], ["a","b","c"]),
        ([1,2,3,4], ["a","b","c","UNK"]),
        ([1], ["a"]),
    ]
)
def test_id_to_word_op(id, expected_output):
    vocab = MockVocab()
    t_word = vocab.id_to_word_op(id)
    with tf.Session() as sess:
        sess.run(tf.tables_initializer())
        r_word = sess.run(t_word)
    word = [word.decode() for word in r_word]
    assert (word == expected_output)


@pytest.mark.parametrize("id, expected_output",
    [
        ([1,2,3], [[1.5,2.5,3.5],[4.5,5.5,6.5],[7.5,8.5,9.5]]),
        ([1,2,3,4], [[1.5,2.5,3.5],[4.5,5.5,6.5],[7.5,8.5,9.5],[0.0,0.0,0.0]]),
        ([1], [[1.5,2.5,3.5]]),
    ]
)
def test_id_to_vector_op(id, expected_output):
    vocab = MockVocab()
    t_vector = vocab.id_to_vector_op(id)
    with tf.Session() as sess:
        sess.run(tf.tables_initializer())
        r_vector = sess.run(t_vector)
    assert (r_vector == np.array(expected_output)).all()


@pytest.mark.parametrize("special_unit_name, expected_output",
    [
        (SpecialUnit.OUT_OF_VOCABULARY, 4),
        (SpecialUnit.END_OF_SEQUENCE, None),
        (SpecialUnit.START_OF_SEQUENCE, None),
        ("", None),
        ("foo", None),
    ]
)
def test_special_unit_to_id(special_unit_name, expected_output):
    vocab = MockVocab()
    assert vocab.special_unit_to_id(special_unit_name) == expected_output

