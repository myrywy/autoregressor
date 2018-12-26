import tensorflow as tf

class SpecialUnit:
    OUT_OF_VOCABULARY = "OUT_OF_VOCABULARY"
    START_OF_SEQUENCE = "START_OF_SEQUENCE"
    END_OF_SEQUENCE = "END_OF_SEQUENCE"

class GeneralizedVocabulary:
    """Creates generalizations of identifiers and vectors from vocabulary to include special units.

    Args:
        vocabulary (Vocabulary): vocabulary to generalize
        special_elements (list): list of names of special units from SpecialUnit to be included in generalized vocabulary"""
    def __init__(self, vocabulary, special_elements):
        self._vocab = vocabulary
        self._special_elements_order = tuple(special_elements)
        self._special_elements, self._offset = self._initialize_special_elements_ids(special_elements)

    def get_special_unit_id(self, special_unit_name):
        return self._special_elements[special_unit_name]

    def _initialize_special_elements_ids(self, special_elements):
        unsupported_special_elements = 0
        special_elements_to_generalized_ids = {}
        special_elements_to_vocab_ids = {}
        for special_unit_name in special_elements:
            id = self._vocab.special_unit_to_id(special_unit_name)
            if id is None:
                unsupported_special_elements += 1
                id = unsupported_special_elements
                special_elements_to_generalized_ids[special_unit_name] = id
            else:
                special_elements_to_vocab_ids[special_unit_name] = id

        offset = unsupported_special_elements + 1
        for special_unit_name, vocab_id in special_elements_to_vocab_ids.items():
            special_elements_to_generalized_ids[special_unit_name] = vocab_id + offset
        return special_elements_to_generalized_ids, offset

    def generalized_id_to_vocab_id(self):
        def op(id_generalized):
            id_generalized = tf.convert_to_tensor(id_generalized)
            is_added_special = tf.less(id_generalized, self._offset)
            broadcasted_not_in_vocab = tf.ones_like(id_generalized) * self._vocab.get_non_id_integer()
            id_vocab = tf.where(is_added_special, broadcasted_not_in_vocab, id_generalized - self._offset)
            return id_vocab
        return op

    def vocab_id_to_generalized_id(self):
        def op(id_vocab):
            id_vocab = tf.convert_to_tensor(id_vocab)
            return id_vocab + self._offset
        return op

    def generalized_id_to_extended_vector(self):
        id_to_vecor_or_default_op = self._vocab.id_to_vecor_or_default_op()
        generalized_id_to_vocab_id = self.generalized_id_to_vocab_id()
        encode_features_op = self.encode_features_op()

        def op(id_generalized):
            id_generalized = tf.convert_to_tensor(id_generalized)
            features_vector = encode_features_op(id_generalized)
            id_vocab = generalized_id_to_vocab_id(id_generalized)
            embedding_vector = id_to_vecor_or_default_op(id_vocab)
            return tf.concat((features_vector, embedding_vector), axis=1)
        return op

    def encode_features_op(self):
        special_element_ids = [self._special_elements[special_element_name] for special_element_name in
                               self._special_elements_order]
        feature_number = [*range(len(special_element_ids))]
        number_of_features = len(feature_number)
        special_element_ids = tf.constant(special_element_ids, dtype=tf.int64)
        feature_number = tf.constant(feature_number, dtype=tf.int64)
        key_value = tf.contrib.lookup.KeyValueTensorInitializer(special_element_ids, feature_number, key_dtype=tf.int64,
                                                                value_dtype=tf.int64)
        table = tf.contrib.lookup.HashTable(key_value, default_value=-1)

        def op(id_generalized):
            if isinstance(id_generalized, tf.Tensor):
                id_generalized = tf.cast(id_generalized, tf.int64)
            id_generalized = tf.convert_to_tensor(id_generalized, dtype=tf.int64)
            feature_number = table.lookup(id_generalized)
            return tf.one_hot(feature_number, depth=number_of_features)
        return op
