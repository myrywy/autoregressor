"""These are for testing only"""
import tensorflow as tf
from vocabularies_preprocessing.vocabulary import Vocabulary
from generalized_vocabulary import SpecialUnit

class MockVocab(Vocabulary):
    def __init__(self):
        self.tokens = ["a", "b", "c"]
        self.vectors = [ [1.5,2.5,3.5], [4.5,5.5,6.5], [7.5,8.5,9.5], [0.0,0.0,0.0] ]
        self.FIRST_ID = 1
        self.UNKNOWN_WORD_ID = len(self.tokens) + self.FIRST_ID

    def word_to_id_op(self):
        lookup_table = tf.contrib.lookup.index_table_from_tensor(self.tokens,
                                                                 default_value=self.UNKNOWN_WORD_ID - self.FIRST_ID,
                                                                 name="MockVocab_word_to_id_lookup")

        def op(word):
            word = tf.convert_to_tensor(word, dtype=tf.string)
            return lookup_table.lookup(word)+self.FIRST_ID

        return op

    def id_to_word_op(self):
        lookup_table = tf.contrib.lookup.index_to_string_table_from_tensor(self.tokens, name="MockVocab_id_to_word_lookup")

        def op(id):
            if isinstance(id, tf.Tensor) and id.dtype == tf.int32:
                id = tf.cast(id, tf.int64)
            id = tf.convert_to_tensor(id, dtype=tf.int64)
            return lookup_table.lookup(id-self.FIRST_ID)

        return op

    def id_to_vector_op(self):
        def op(id):
            if isinstance(id, tf.Tensor) and id.dtype == tf.int32:
                id = tf.cast(id, tf.int64)
            id = tf.convert_to_tensor(id, dtype=tf.int64)
            return tf.squeeze(tf.nn.embedding_lookup(tf.expand_dims(self.vectors, 1), id-self.FIRST_ID), axis=1)
        return op

    def special_unit_to_id(self, special_unit_name):
        if special_unit_name == SpecialUnit.OUT_OF_VOCABULARY:
            return self.UNKNOWN_WORD_ID
        else:
            return None

    def get_non_id_integer(self):
        return self.UNKNOWN_WORD_ID + 1 # i.e. greater than biggest valid id
        
    def get_valid_id_example(self):
        return self.FIRST_ID

    def vector_size(self):
        return 3

    def vocab_size(self):
        return 4
    
    def ids_range(self):
        return self.FIRST_ID, self.FIRST_ID + self.vocab_size() - 1

    
