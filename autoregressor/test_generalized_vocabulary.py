from generalized_vocabulary import GeneralizedVocabulary, SpecialUnit
from vocabularies_preprocessing.mock_vocabulary import MockVocab

import tensorflow as tf
import pytest
from pytest import approx

@pytest.mark.parametrize("specials",
[
    [SpecialUnit.OUT_OF_VOCABULARY, SpecialUnit.START_OF_SEQUENCE, SpecialUnit.END_OF_SEQUENCE],
    [SpecialUnit.OUT_OF_VOCABULARY],
    [SpecialUnit.OUT_OF_VOCABULARY, SpecialUnit.START_OF_SEQUENCE],
    [SpecialUnit.START_OF_SEQUENCE, SpecialUnit.END_OF_SEQUENCE],
    [SpecialUnit.END_OF_SEQUENCE],
]
)
def test_get_special_unit_id__complete_uniq(specials):
    vocab = MockVocab()
    generalized = GeneralizedVocabulary(vocab, specials)
    ids = set()
    for special_unit_name in specials:
        id = generalized.get_special_unit_id(special_unit_name)
        assert isinstance(id, int)
        ids.add(id)
    assert len(specials) == len(ids)


def test_get_special_unit_id__use_already_supported_ids():
    specials = [SpecialUnit.OUT_OF_VOCABULARY, SpecialUnit.START_OF_SEQUENCE, SpecialUnit.END_OF_SEQUENCE]
    vocab = MockVocab()
    generalized = GeneralizedVocabulary(vocab, specials)
    
    gen_oov_id = generalized.get_special_unit_id(SpecialUnit.OUT_OF_VOCABULARY)
    oov_vocab_id = vocab.special_unit_to_id(SpecialUnit.OUT_OF_VOCABULARY)
    t_oov_id_from_vocab_via_generalized = generalized.vocab_id_to_generalized_id([oov_vocab_id])
    with tf.Session() as sess:
        r_oov_id_from_vocab_via_generalized = sess.run(t_oov_id_from_vocab_via_generalized)
    assert r_oov_id_from_vocab_via_generalized[0] == gen_oov_id


def test_get_special_unit_id__non_supported_ids_get_non_id():
    specials = [SpecialUnit.OUT_OF_VOCABULARY, SpecialUnit.START_OF_SEQUENCE, SpecialUnit.END_OF_SEQUENCE]
    vocab = MockVocab()
    non_id = vocab.get_non_id_integer()
    generalized = GeneralizedVocabulary(vocab, specials)
    
    gen_start_id = generalized.get_special_unit_id(SpecialUnit.START_OF_SEQUENCE)
    gen_end_id = generalized.get_special_unit_id(SpecialUnit.END_OF_SEQUENCE)
    t_vocab_ids = generalized.generalized_id_to_vocab_id([gen_start_id, gen_end_id])
    with tf.Session() as sess:
        r_vocab_ids = sess.run(t_vocab_ids)
    assert r_vocab_ids[0] == non_id
    assert r_vocab_ids[1] == non_id


@pytest.mark.parametrize("generalized_id, expected_encoded, special_units",
    [
        ([1], [[1,0,0]], [SpecialUnit.OUT_OF_VOCABULARY, SpecialUnit.START_OF_SEQUENCE, SpecialUnit.END_OF_SEQUENCE]),
        ([2], [[0,1,0]], [SpecialUnit.OUT_OF_VOCABULARY, SpecialUnit.START_OF_SEQUENCE, SpecialUnit.END_OF_SEQUENCE]),
        ([5], [[0,0,0]], [SpecialUnit.OUT_OF_VOCABULARY, SpecialUnit.START_OF_SEQUENCE, SpecialUnit.END_OF_SEQUENCE]),
        ([2,1], [[0,1,0],[1,0,0]], [SpecialUnit.OUT_OF_VOCABULARY, SpecialUnit.START_OF_SEQUENCE, SpecialUnit.END_OF_SEQUENCE]),
        ([2,1,0], [[0,1,0],[1,0,0],[0,0,0]], [SpecialUnit.OUT_OF_VOCABULARY, SpecialUnit.START_OF_SEQUENCE, SpecialUnit.END_OF_SEQUENCE]),
        ([1], [[1,0]], [SpecialUnit.START_OF_SEQUENCE, SpecialUnit.END_OF_SEQUENCE]),
        ([2], [[0,1]], [SpecialUnit.START_OF_SEQUENCE, SpecialUnit.END_OF_SEQUENCE]),
        ([3], [[0,0]], [SpecialUnit.START_OF_SEQUENCE, SpecialUnit.END_OF_SEQUENCE]),
        ([1], [[1]], [SpecialUnit.START_OF_SEQUENCE]),
        ([2], [[0]], [SpecialUnit.START_OF_SEQUENCE]),
    ]
)
def test_encoded_features(generalized_id, expected_encoded, special_units):
    vocab = MockVocab()
    generalized = GeneralizedVocabulary(vocab, special_units)

    generalized_id = tf.convert_to_tensor(generalized_id)

    t_encoded = generalized.encoded_features(generalized_id)

    with tf.Session() as sess:
        sess.run(tf.tables_initializer())
        r_encoded = sess.run(t_encoded)

    assert r_encoded == approx(expected_encoded)
    

