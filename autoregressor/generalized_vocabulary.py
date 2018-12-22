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
        self._special_elements, self._offset = self._initialize_special_elements_ids(special_elements)

    def get_special_unit_id(self, special_unit_name):
        return self._special_elements[special_unit_name]

    def _initialize_special_elements_ids(self, special_elements):
        unsupported_special_elements = 0
        special_elements_to_generalized_ids = {}
        for special_unit_name in special_elements:
            id = self._vocab.special_unit_to_id(special_unit_name)
            if id is None:
                unsupported_special_elements += 1
                id = unsupported_special_elements
            special_elements_to_generalized_ids[special_unit_name] = id
        return special_elements_to_generalized_ids, unsupported_special_elements + 1

    def generalized_id_to_vocab_id(self, id_generalized):
        is_added_special = tf.less(id_generalized, self._offset)
        broadcasted_not_in_vocab = tf.ones_like(id_generalized) * self._vocab.get_non_id_integer()
        id_vocab = tf.where(is_added_special, broadcasted_not_in_vocab, id_generalized - self._offset)
        return id_vocab

    def vocab_id_to_generalized_id(self, id_vocab):
        return id_vocab + self._offset

    def generalized_id_to_extended_vector(self, id_generalized):
        features_vector = self.encoded_features(id_generalized)
        id_vocab = self.generalized_id_to_vocab_id(id_generalized)
        embedding_vector = self._vocab.id_to_vecor_or_default(id_vocab)
        return tf.concat((features_vector, embedding_vector), axis=1)
        
    def encoded_features(self, id_generalized):
        special_element_ids = [*sorted(self._special_elements.values)]
        feature_number = [*range(len(special_element_ids))]
        number_of_features = len(feature_number)
        special_element_ids = tf.constant(special_element_ids, dtype=tf.int64)
        feature_number = tf.constant(feature_number, dtype=tf.int64)
        key_value = tf.contrib.lookup.KeyValueTensorInitializer(special_element_ids, feature_number, key_dtype=tf.int64, value_dtype=tf.int64)
        table = tf.contrib.lookup.HashTable(key_value, default_value=-1)
        feature_number = table.lookup(id_generalized)
        one_hot_encoded = tf.one_hot(feature_number, depth=number_of_features)
        raise one_hot_encoded