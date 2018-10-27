from typing import List
import tensorflow as tf

class ElementProbabilityMasking(tf.keras.layers.Layer):
    def __init__(
            self, 
            allowed: List[List[int]], 
            probability_distribution_size: int,
            element_id_to_index_in_probability_distribution_mapping=tf.identity,
            ):
        """
        This class is work in progress.

        Semantyka jest taka: allowed zawiera kolejne listy z dozwolony id-ki ze słownika dla danego elementu tj.
        allowed_values[0] <- lista możliwych elementów na pozycji zerowej, wszystkie elementy, których nie ma na tej liści
        będą miały zerowe prawdopodobieństwo. Wyjątek: jeśli lista jest pusta to oznacza, że wszytkie elementy są możliwe
        probability_distribution_size - szerokość rozkładu prawdopodobieństwa 
        """
        super(ElementProbabilityMasking, self).__init__()
        self.element_id_to_index_in_probability_distribution_mapping = element_id_to_index_in_probability_distribution_mapping
        indices_step = []
        indices_id = []
        for step, allowed_ids in enumerate(allowed):
            if not allowed_ids:
                allowed_ids = [*range(probability_distribution_size)]
            # remove duplicates and sort
            allowed_ids = [*sorted({*allowed_ids})]
            indices_id.extend(identifier for identifier in allowed_ids)
            indices_step.extend(step for _ in allowed_ids)
        
        values = tf.ones((len(indices_id)), dtype=tf.float32)
        indices_id = tf.constant(indices_id, dtype=tf.int64)
        indices_step = tf.constant(indices_step, dtype=tf.int64)
        indices_position = element_id_to_index_in_probability_distribution_mapping(indices_id)
        indices = tf.concat(
            (
                tf.expand_dims(indices_step, 1), 
                tf.expand_dims(indices_position, 1)
                ), 
                axis=1)
        dense_shape = tf.constant([len(allowed), probability_distribution_size], dtype=tf.int64)
        mask = tf.SparseTensor(indices, values, dense_shape)
        self.mask = tf.sparse_tensor_to_dense(mask)
        

    def call(self, probabilites, step=None):
        """
        probabilites - tensor [batch_size, vocabulary_size], probabilities[i] - probability distribution over vocabulary in i-th batch element
            (probabilities[i][j] is a probability of j-th 'word' from a vocabulary in i-th batch element)
        step - number of element in sequence (sentence) that is being predicted, assuming that first predicted is 1"""
        
        mask_at_step = tf.gather(self.mask, step - 1, name="gather_mask_for_steps")
        return probabilites * mask_at_step
        