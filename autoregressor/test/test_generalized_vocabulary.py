from generalized_vocabulary import GeneralizedVocabulary, SpecialUnit
from vocabularies_preprocessing.mock_vocabulary import MockVocab

import tensorflow as tf
import numpy as np
import pytest
from pytest import approx

@pytest.fixture()
def special_units_all():
    return [SpecialUnit.START_OF_SEQUENCE,
            SpecialUnit.END_OF_SEQUENCE,
            SpecialUnit.OUT_OF_VOCABULARY]


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


def test_generalized_id_to_token():
    specials = [SpecialUnit.OUT_OF_VOCABULARY, SpecialUnit.START_OF_SEQUENCE, SpecialUnit.END_OF_SEQUENCE]
    vocab = MockVocab()
    generalized = GeneralizedVocabulary(vocab, specials)
    
    ids = [0,1,2,4,5,6]
    expected = [b"<<ZERO>>", b"<<START_OF_SEQUENCE>>", b"<<END_OF_SEQUENCE>>", b"a", b"b", b"c"]

    tokens = generalized.generalized_id_to_token()(ids)

    with tf.Session() as sess:
        sess.run(tf.tables_initializer())
        r_tokens = sess.run(tokens)

    assert (r_tokens == expected).all()



def test_get_special_unit_id__use_already_supported_ids():
    specials = [SpecialUnit.OUT_OF_VOCABULARY, SpecialUnit.START_OF_SEQUENCE, SpecialUnit.END_OF_SEQUENCE]
    vocab = MockVocab()
    generalized = GeneralizedVocabulary(vocab, specials)

    gen_oov_id = generalized.get_special_unit_id(SpecialUnit.OUT_OF_VOCABULARY)
    oov_vocab_id = vocab.special_unit_to_id(SpecialUnit.OUT_OF_VOCABULARY)
    t_oov_id_from_vocab_via_generalized = generalized.vocab_id_to_generalized_id()([oov_vocab_id])
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
    t_vocab_ids = generalized.generalized_id_to_vocab_id()([gen_start_id, gen_end_id])
    with tf.Session() as sess:
        r_vocab_ids = sess.run(t_vocab_ids)
    assert r_vocab_ids[0] == non_id
    assert r_vocab_ids[1] == non_id


@pytest.mark.parametrize("generalized_id, expected_encoded, special_units",
    [
        ([1], [[1,0,0]], [SpecialUnit.START_OF_SEQUENCE, SpecialUnit.END_OF_SEQUENCE, SpecialUnit.OUT_OF_VOCABULARY]),
        ([2], [[0,1,0]], [SpecialUnit.START_OF_SEQUENCE, SpecialUnit.END_OF_SEQUENCE, SpecialUnit.OUT_OF_VOCABULARY]),
        ([5], [[0,0,0]], [SpecialUnit.START_OF_SEQUENCE, SpecialUnit.END_OF_SEQUENCE, SpecialUnit.OUT_OF_VOCABULARY]),
        ([2,1], [[0,1,0],[1,0,0]], [SpecialUnit.START_OF_SEQUENCE, SpecialUnit.END_OF_SEQUENCE, SpecialUnit.OUT_OF_VOCABULARY]),
        ([2,1,0], [[0,1,0],[1,0,0],[0,0,0]], [SpecialUnit.START_OF_SEQUENCE, SpecialUnit.END_OF_SEQUENCE, SpecialUnit.OUT_OF_VOCABULARY]),
        ([1], [[1,0]], [SpecialUnit.START_OF_SEQUENCE, SpecialUnit.END_OF_SEQUENCE]),
        ([2], [[0,1]], [SpecialUnit.START_OF_SEQUENCE, SpecialUnit.END_OF_SEQUENCE]),
        ([3], [[0,0]], [SpecialUnit.START_OF_SEQUENCE, SpecialUnit.END_OF_SEQUENCE]),
        ([1], [[1]], [SpecialUnit.START_OF_SEQUENCE]),
        ([2], [[0]], [SpecialUnit.START_OF_SEQUENCE]),
    ]
)
def test_encode_features_op(generalized_id, expected_encoded, special_units):
    vocab = MockVocab()
    generalized = GeneralizedVocabulary(vocab, special_units)

    generalized_id = tf.convert_to_tensor(generalized_id)

    t_encoded = generalized.encode_features_op()(generalized_id)

    with tf.Session() as sess:
        sess.run(tf.tables_initializer())
        r_encoded = sess.run(t_encoded)

    assert r_encoded == approx(np.array(expected_encoded))

@pytest.mark.parametrize("special_unit_to_encode, supported_special_units, expected_encoded",
    [
        (
            SpecialUnit.OUT_OF_VOCABULARY,
            [SpecialUnit.OUT_OF_VOCABULARY, SpecialUnit.START_OF_SEQUENCE, SpecialUnit.END_OF_SEQUENCE],
            [1,0,0]
        ),
        (
            SpecialUnit.START_OF_SEQUENCE,
            [SpecialUnit.OUT_OF_VOCABULARY, SpecialUnit.START_OF_SEQUENCE, SpecialUnit.END_OF_SEQUENCE],
            [0,1,0]
        ),
        (
            SpecialUnit.END_OF_SEQUENCE,
            [SpecialUnit.OUT_OF_VOCABULARY, SpecialUnit.START_OF_SEQUENCE, SpecialUnit.END_OF_SEQUENCE],
            [0,0,1]
        ),
        (
            SpecialUnit.OUT_OF_VOCABULARY,
            [SpecialUnit.START_OF_SEQUENCE, SpecialUnit.OUT_OF_VOCABULARY, SpecialUnit.END_OF_SEQUENCE],
            [0,1,0]
        ),
        (
            SpecialUnit.OUT_OF_VOCABULARY,
            [SpecialUnit.START_OF_SEQUENCE, SpecialUnit.END_OF_SEQUENCE, SpecialUnit.OUT_OF_VOCABULARY],
            [0,0,1]
        ),
        (
            SpecialUnit.OUT_OF_VOCABULARY,
            [SpecialUnit.START_OF_SEQUENCE, SpecialUnit.OUT_OF_VOCABULARY],
            [0,1]
        ),
        (
            SpecialUnit.OUT_OF_VOCABULARY,
            [SpecialUnit.OUT_OF_VOCABULARY, SpecialUnit.START_OF_SEQUENCE],
            [1,0]
        ),
    ]
)
def test_encode_features_op_by_special_units_names(special_unit_to_encode,
                                                 supported_special_units,
                                                 expected_encoded):
    vocab = MockVocab()
    generalized = GeneralizedVocabulary(vocab, supported_special_units)

    generalized_id = generalized.get_special_unit_id(special_unit_to_encode)
    generalized_id = tf.convert_to_tensor([generalized_id])

    t_encoded = generalized.encode_features_op()(generalized_id)

    with tf.Session() as sess:
        sess.run(tf.tables_initializer())
        r_encoded = sess.run(t_encoded)

    assert r_encoded == approx(np.array([expected_encoded]))


@pytest.mark.parametrize("generalized_id, expected_vocab_id",
    [
        ([1,2,4,5,6], [5,5,1,2,3]),
        ([1,2], [5,5]),
        ([1], [5]),
        ([4,5,6], [1,2,3]),
        ([6], [3]),
        ([3], [0]),
    ]
)
def test_generalized_id_to_vocab_id(generalized_id, expected_vocab_id):
    vocab = MockVocab()

    special_units = [SpecialUnit.START_OF_SEQUENCE,
                     SpecialUnit.END_OF_SEQUENCE,
                     SpecialUnit.OUT_OF_VOCABULARY]

    generalized_vocab = GeneralizedVocabulary(vocab, special_units)
    t_generalized_id = tf.convert_to_tensor(generalized_id)
    t_vocab_id = generalized_vocab.generalized_id_to_vocab_id()(t_generalized_id)

    with tf.Session() as sess:
        r_vocab_id = sess.run(t_vocab_id)

    assert expected_vocab_id == approx(r_vocab_id)


@pytest.mark.parametrize("vocab_id, expected_generalized_id",
    [
        ([1,2,3,4], [4,5,6,7]),
        ([1], [4]),
        ([4], [7]),
        ([1,4], [4,7]),
    ]
)
def test_vocab_id_to_generalized_id(vocab_id, expected_generalized_id):
    vocab = MockVocab()
    special_units = [SpecialUnit.START_OF_SEQUENCE,
                     SpecialUnit.END_OF_SEQUENCE,
                     SpecialUnit.OUT_OF_VOCABULARY]
    generalized_vocab = GeneralizedVocabulary(vocab, special_units)

    t_vocab_id = tf.convert_to_tensor(vocab_id)
    t_generalized_id = generalized_vocab.vocab_id_to_generalized_id()(t_vocab_id)

    with tf.Session() as sess:
        r_generalized_id = sess.run(t_generalized_id)

    assert expected_generalized_id == approx(r_generalized_id)


@pytest.mark.parametrize("vocab_type, generalized_id_original",
    [
        (MockVocab, [4,5,6,7]),
        (MockVocab, [4]),
        (MockVocab, [7]),
    ]
)
def test_generalized_to_vocab_to_generalized_idempotent(vocab_type, generalized_id_original):
    vocab = vocab_type()
    special_units  = [SpecialUnit.START_OF_SEQUENCE,
                     SpecialUnit.END_OF_SEQUENCE,
                     SpecialUnit.OUT_OF_VOCABULARY]
    generalized_vocab = GeneralizedVocabulary(vocab, special_units)

    t_generalized_id_original = tf.convert_to_tensor(generalized_id_original)
    t_vocab_id = generalized_vocab.generalized_id_to_vocab_id()(t_generalized_id_original)
    t_generalized_id_restored = generalized_vocab.vocab_id_to_generalized_id()(t_vocab_id)

    with tf.Session() as sess:
        r_generalized_id_restored = sess.run(t_generalized_id_restored)

    assert np.array(generalized_id_original) == approx(r_generalized_id_restored)


@pytest.mark.parametrize("vocab_type, unsupported_special_unit_names",
    [
        (MockVocab, [SpecialUnit.START_OF_SEQUENCE,
                         SpecialUnit.END_OF_SEQUENCE]),
    ]
)
def test_generalized_id_to_vocab_id_on_usupported_specials(vocab_type, unsupported_special_unit_names):
    vocab = vocab_type()
    generalized_vocab = GeneralizedVocabulary(vocab, unsupported_special_unit_names)

    generalized_id = [ generalized_vocab.get_special_unit_id(unit_name) \
                        for unit_name in unsupported_special_unit_names]

    t_generalized_id = tf.convert_to_tensor(generalized_id)
    t_vocab_id = generalized_vocab.generalized_id_to_vocab_id()(t_generalized_id)

    expected_output = [vocab.get_non_id_integer()] * len(unsupported_special_unit_names)

    with tf.Session() as sess:
        r_vocab_id = sess.run(t_vocab_id)

    assert np.array(expected_output) == approx(r_vocab_id)


@pytest.mark.parametrize("vocab_type, vocab_id_original",
    [
        (MockVocab, [1,2,3,4]),
        (MockVocab, [1]),
        (MockVocab, [4]),
    ]
)
def test_vocab_to_generalized_to_vocab_idempotent(vocab_type, vocab_id_original):
    vocab = vocab_type()
    special_units  = [SpecialUnit.START_OF_SEQUENCE,
                     SpecialUnit.END_OF_SEQUENCE,
                     SpecialUnit.OUT_OF_VOCABULARY]
    generalized_vocab = GeneralizedVocabulary(vocab, special_units)

    t_vocab_id_original = tf.convert_to_tensor(vocab_id_original)
    t_generalized_id = generalized_vocab.vocab_id_to_generalized_id()(t_vocab_id_original)
    t_vocab_id_restored = generalized_vocab.generalized_id_to_vocab_id()(t_generalized_id)

    with tf.Session() as sess:
        r_vocab_id_restored = sess.run(t_vocab_id_restored)

    assert np.array(vocab_id_original) == approx(r_vocab_id_restored)


@pytest.mark.parametrize("vocab_type, generalized_id",
    [
        (MockVocab, [4,5,6,7]),
        (MockVocab, [4]),
        (MockVocab, [7]),
    ]
)
def test_generalized_id_to_extended_vector__composition(special_units_all, vocab_type, generalized_id):
    vocab = vocab_type()
    generalized_vocab = GeneralizedVocabulary(vocab, special_units_all)

    t_vector = generalized_vocab.generalized_id_to_extended_vector()(generalized_id)
    t_features = generalized_vocab.encode_features_op()(generalized_id)
    t_embedding = vocab.id_to_vector_op()(generalized_vocab.generalized_id_to_vocab_id()(generalized_id))

    with tf.Session() as sess:
        sess.run(tf.tables_initializer())
        r_vector, r_features, r_embedding = sess.run((t_vector, t_features, t_embedding))

    assert r_vector == approx(np.concatenate((r_features, r_embedding), axis=1))



@pytest.mark.parametrize("vocab_type, generalized_id, expected_output",
    [
        (MockVocab, [1,2,4,5,6,7], [
            [1.0,0,0,0,0,0],
            [0.0,1,0,0,0,0],
            [0.0,0,0,1.5,2.5,3.5],
            [0.0,0,0,4.5,5.5,6.5],
            [0.0,0,0,7.5,8.5,9.5],
            [0.0,0,1,0,0,0],
        ]),
        (MockVocab, [1], [[1.0,0,0,0,0,0]]),
        (MockVocab, [4], [[0.0,0,0,1.5,2.5,3.5]]),
        (MockVocab, [7], [[0.0,0,1,0,0,0]]),
    ]
)
def test_generalized_id_to_extended_vector(special_units_all, vocab_type, generalized_id, expected_output):
    vocab = vocab_type()
    generalized_vocab = GeneralizedVocabulary(vocab, special_units_all)

    t_vector = generalized_vocab.generalized_id_to_extended_vector()(generalized_id)

    with tf.Session() as sess:
        sess.run(tf.tables_initializer())
        r_vector = sess.run(t_vector)

    assert r_vector == approx(np.array(expected_output))
