from abc import ABC, abstractmethod

import tensorflow as tf

class Vocabulary(ABC):
    """
    Vocabulary should not instantiate any TF ops during instantiation,
    each time functions should create new ops to make it possible to to use
    them inside Estimator i.e. with a new computational graph each time
    train() or predict() is called.

    It is assumed that ID assigned to a word must not be negative
    """
    VECTOR_TYPE = tf.float32

    @abstractmethod
    def word_to_id_op(self, word):
        """Converts words/tokens to integer identifiers used by this vocabulary.

        Args:
            word (tf.Tensor): 1D tensor of strings

        Returns:
            tf.Tensor: 1D tensor of integer type
        """
        pass

    @abstractmethod
    def id_to_word_op(self, id):
        """Converts integer identifiers used by this vocabulary to words/tokens.
        TODO: describe <Unknown> identifier

        Args:
            id (tf.Tensor): 1D tensor of of integer type

        Returns:
            tf.Tensor: 1D tensor strings
        """
        pass

    @abstractmethod
    def id_to_vector_op(self, id):
        """Converts integer identifiers used by this vocabulary to embeddings vectors.

        Args:
            id (tf.Tensor): 1D tensor of strings

        Returns:
            tf.Tensor: 2D tensor of float type
        """
        pass

    def id_to_vecor_or_default(self, id, default=0):
        """Converts integer identifiers used by this vocabulary to embeddings vectors unless an id is equal to value returned by get_non_id_integer - then it's converted to default.
        default is broadcasted to vector_size so it should be either scalar or vector of size equalt to vector_size.

        Args:
            id (tf.Tensor): 1D tensor of strings

        Returns:
            tf.Tensor: 2D tensor of float type
        """
        default_vector = self._maybe_tile_to_vector_size(default)
        a_valid_id = self.get_valid_id_example()
        use_default = tf.equal(id, self.get_non_id_integer())
        valid_ids = tf.where(use_default, a_valid_id, id)
        valid_vectors = self.id_to_vector_op(valid_ids)
        return tf.where(use_default, default_vector, valid_vectors)

    def _maybe_tile_to_vector_size(self, vector_or_scalar):
        t = tf.convert_to_tensor(vector_or_scalar, dtype=Vocabulary.VECTOR_TYPE)
        vector_size = self.vector_size()
        if len(t.shape) == 0:
            t = tf.tile([t], [vector_size])
        assert t.shape.is_compatible_with([vector_size]), "Incompatible vector size"
        
    @abstractmethod
    def special_unit_to_id(self, special_unit):
        """Returns ids of vocabulary elements that have very special meaning like out of vocabulary word, start/end of sequence etc.
        List of types of "elements of special meaning" is global and defined in generalized_vocabulary.SpecialUnit.
        If vocabulary does not have special vector for a given element then the function should return None
        
        Args:
            special_unit (str): global identifier of type of special unit
        
        Returns:
            int or None: identifier of special unit in this vocabulary if supported or None 
        """
        pass

    @abstractmethod
    def get_non_id_integer(self):
        """Returns integer that is guaranteed not to be identifier recognized by this vocabulary.
        Should return the same number each call. For example, if a vocabulary contains 10000 words and assings them 
        ids from a range [0:9999] then the function could return for example -1 or 10000 (but it should be decided when object is
        created or when class is defined and does'n change between calls). 
        Note: if vocabulary has special id for out of vocabulary element, say 0, and assigns ids 1..10000 to "real" words, 
        then the function cannot return 0 because it still counts as id recognised by this vocabulary.
        
        Returns:
            int: a value that is not an id recognized by this vocabulary.
        """
        pass

    @abstractmethod
    def get_valid_id_example(self):
        """Returns integer that is guaranteed to be valid identifier of some word in vocabulary. It may be used where a placeholder for 
        an id is needed but it must be convertible to vector.
        
        Returns:
            int: a value that is an id recognized by this vocabulary.
        """
        pass

    @abstractmethod
    def vector_size(self):
        """Returns size of embedding vectors in the vocabulary.

        Returns:
            int or tf.Tensor: If tensor then it should be a scalar.
        """
        pass


