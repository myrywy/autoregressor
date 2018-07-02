import tensorflow as tf 
import numpy as np
from collections import namedtuple, defaultdict

# Layer poszukiwanie najtanszej sciezki
# input: 
# 1) funkcja generująca następne możliwe stany następne
# 2) funkcja przypisująca koszt do stanu

# w przykładzie zdaniowym:
# kolejne stany to będą kolejne pary (zakres tokenów; znaczenie)

# Model jezyka przypisuje prawdopodobienstwo elementom ciagu
# Generator interpretacji podaje kolejne mozliwe elementy ciagu na podstawie znaczen poszczegolnych slow
# z kazdej interpretacji na etapie slowa n powstaje m interpretacji uwzgledniajacych nastepne slowo lub fraze
# gdzie m to liczba znaczen tego slowa lub frazy

# Zeby wiedziec ktorego slowa lub frazy uzyc w nastepnym kroku trzeba znac pozycje i wariant dzielenia na frazy
# np.

InterpretationSearchState = namedtuple("InterpretationSearchState",
        [
            "tokenization_variants", 
            "len_of_tokenization_variants",
            "position_in_tokenization_variants",
            "index_of_tokenization_variant", 
            "interpretations", 
            "interpretations_probabilities", 
            "lm_state"
        ])

class BestInterpretationFinder:
    def __init__(self, 
                 language_model,
                 variants_to_follow):
        self.language_model = language_model
        self.variants_to_follow = variants_to_follow

    def call(self, inputs, state):
        # wszystkie wymiary w komentarzach ponizej podano z pominieciem pierwszego (batch)
        # zakładamy na razie, że zawsze batch size = 1, a wewnętrznie w language modelu
        tokenization_variants, \
        position_in_tokenization_variants, \
        index_of_tokenization_variant, \
        interpretations, \
        interpretations_probabilities, \
        lm_state, \
        index, \
        tokens_meanings_lookup = state
        # tokenisation_variants (<variants_to_follow>, <sequence_length>) - identyfikatory slow lub fraz, np. 
        # "He is in the white house" można podzielić na ["he"(1), "is"(2), "in"(3), "the"(4), "white"(5), "house"(6)] 
        # lub ["he"(1), "is"(2), "in"(3), "the white house"(7)] więc tokenisation_variants miałoby postac
        # [[1,2,3,4,5,6],
        #  [1,2,3,7,0,0]]
        # kazdemu z tych identyfikatorow odpowiada cała lista możliwych znaczeń (lub <nieznane znaczenie>)
        # 
        # Wartości w index_of_tokenization_variant wskazują na wiersze w tokenization_variants, które odpowiadają kolejnym 
        # interpretacjom w interpretations, jesli jakas interpretacja wypada bo inna ma dwa 
        # bardziej prawdopodobne rozwiniecia to trzeba zmienic wiersz w index_of_tokenization_variant
        # na taki sam jak wiersz odpowiadajacy tej interpretacji ktora byla lepsza
        #
        # interpretations (<variants_to_follow>, <sequence_length>) - <variants_to_follow> wierszy, każdy wiersz początkowo powinien 
        # zawierać n (np. 2) identyfikatorów znaczeń i będzie uzupełniany o kolejne na podstawie najbardziej prawdopodobynch znaczeń tokenów
        #
        # tokens_meanings_lookup (<liczba roznych tokenow>, <maksymalna liczba znaczen slowa>)
        #
        # interpretations_probabelities (<liczba interpretacji>,) - dla każdej interpretacji przechowuje logit prawdopodobieństwa tej interpretacji
        next_word_probabilities, new_lm_states = self.language_model.call(interpretations, lm_state)
        position_in_tokenization_variants = position_in_tokenization_variants + 1

        next_word_probabilities = self._mask_senses_matching_words(
            next_word_probabilities, 
            tokenization_variants, 
            index_of_tokenization_variant, 
            position_in_tokenization_variants,
            tokens_meanings_lookup) 
        
        # produce array of probabilites of dims - (interpretation, next word), and values being probabilites of interpratation if continued by next word
        new_interpretations_probabilities = next_word_probabilities + tf.expand_dims(interpretations_probabelities, 1)

        new_top_interpretations_probabilites, (top_interpretations, top_meanings) = \
            self._top_k_from_2d_tensor(new_interpretations_probabilities, self.variants_to_follow)

        new_interpretations = tf.gather(interpretations, top_interpretations)

        # tu jest ważne założenie, że id znaczenia to jego indeks w warstwie wyjściowej Modelu Języka
        # i jeszcze kolejne zalożenie, że wektor wyjściowy za każdym razem rośnie o 1
        new_interpretations = tf.concat((new_interpretations, [top_meanings]), axis=1) 
        new_position_in_tokenization_variants = tf.gather(position_in_tokenization_variants, top_interpretations)
        new_index_of_tokenization_variant = tf.gather(index_of_tokenization_variant, top_interpretations)

        # brakuje warunku stopu
        new_state = tokenization_variants, \
                    new_position_in_tokenization_variants, \
                    new_index_of_tokenization_variant, \
                    new_interpretations, \
                    new_top_interpretations_probabilites, \
                    new_lm_states, \
                    index+1, \
                    tokens_meanings_lookup\

        # TODO: dodać warunek stopu
        # sprawdzić czy maskowanie jest w odpowiednim miejscu
        # i inkrementacja indeksów


        # wydaje mi się, że w tym forze jakoś działam jak na tensorach zawierających wszystkie interpretacje batche itd i cały ten for po 
        # interpretacjach jest bez sensu
        #

        # Tu jest jakiś taki błąd w koncepcji z tym tokenization variatns bo jeśli te wektory miałyby się dynamicznie zmieniać 
        # to musiałby być w stanie, a nie na wejściu, a jeśli miałby być na wejściu to potrzebujemy jeszcze wektora wskaźników
        # mówiących o tym, którego wariantu tokenizacji używa dana interpretacja

    @staticmethod
    def _mask_senses_matching_words(
            interpretations_probabilities, 
            tokenization_variants, 
            index_of_tokenization_variant, 
            position_in_tokenization_variants,
            tokens_meanings_lookup):
        """The assumption is that probability of a meaning that doesn't match the word is 0. I restrict possible meanings
        with the meanings that according to a dictionary are valid meanings of the word."""
        def _get_elements_from_rows(data, indices):
            indeces = tf.range(0, tf.shape(indices)[0])*data.shape[1] + indices
            return tf.gather(tf.reshape(data, [-1]), indeces)
        # TODO: Consider if zeroeth probability (so probability of unknow word) shouldn't be masked (zeroed) additionally.
        # No it shouldn't cause it would mean that every meaning's probability is 0 so we don't know what to put next in interpr
        tf.zeros(shape=tf.shape(interpretations_probabilities), dtype=interpretations_probabilities.dtype)
        tokenization_variants = tf.gather(tokenization_variants, index_of_tokenization_variant)
        current_tokens = _get_elements_from_rows(tokenization_variants, position_in_tokenization_variants)
        lookup_dense = tf.sparse_to_dense(
            sparse_indices=tokens_meanings_lookup.indices,
            output_shape=tokens_meanings_lookup.dense_shape,
            sparse_values=tokens_meanings_lookup.values)
        mask = tf.cast(tf.gather(lookup_dense, current_tokens), tf.float32)
        return interpretations_probabilities * mask
        
    def _get_token_meanings_mask(self, tokens):
        indices = [[0, 0]] # unknown token, unknown meaning
        indices.extend(
            [
                [i+1, meaning_id] 
                    for i, token in enumerate(tokens) 
                        for meaning_id in self._get_possible_meanings_ids(token)
            ]
        )
        mask_shape = tf.constant([len(tokens)+1, self.meanings_count+1], dtype=tf.int64)
        values = tf.ones((len(indices),), dtype=tf.int64)
        indices = tf.constant(indices, dtype=tf.int64)
        tokens_meanings_lookup = tf.SparseTensor(
            indices=indices,
            values=values,
            dense_shape=mask_shape,
        )
        return tokens_meanings_lookup
    
    @staticmethod
    def _top_k_from_2d_tensor(tensor2d, k):
        flat_tensor = tf.reshape(tensor2d, (-1,))
        top_values, top_indices = tf.nn.top_k(flat_tensor, k)
        top_index1 = top_indices // tf.shape(tensor2d)[1]
        top_index2 = top_indices % tf.shape(tensor2d)[1]
        return top_values, (top_index1, top_index2)

    def find_best_interpretation(
        self,
        tokenization_variants, 
        len_of_tokenization_variants,
        index_of_tokenization_variant,
        interpretations
    ):
        def condition(state):
            return tf.logical_not(tf.reduce_all(self._interpretations_done(state)))
        final_state = tf.while_loop(
            cond=condition,
            body=self.step,
            loop_vars=initial_state)

    def _interpretations_done(self, state):
        state = InterpretationSearchState(state)
        return state.position_in_tokenization_variants > state.len_of_tokenization_variants

    def step(self, state, tokens_meanings_lookup):
        # wszystkie wymiary w komentarzach ponizej podano z pominieciem pierwszego (batch)
        # zakładamy na razie, że zawsze batch size = 1, a wewnętrznie w language modelu
        state = InterpretationSearchState(state)
        # tokenisation_variants (<variants_to_follow>, <sequence_length>) - identyfikatory slow lub fraz, np. 
        # "He is in the white house" można podzielić na ["he"(1), "is"(2), "in"(3), "the"(4), "white"(5), "house"(6)] 
        # lub ["he"(1), "is"(2), "in"(3), "the white house"(7)] więc tokenisation_variants miałoby postac
        # [[1,2,3,4,5,6],
        #  [1,2,3,7,0,0]]
        # kazdemu z tych identyfikatorow odpowiada cała lista możliwych znaczeń (lub <nieznane znaczenie>)
        # 
        # Wartości w index_of_tokenization_variant wskazują na wiersze w tokenization_variants, które odpowiadają kolejnym 
        # interpretacjom w interpretations, jesli jakas interpretacja wypada bo inna ma dwa 
        # bardziej prawdopodobne rozwiniecia to trzeba zmienic wiersz w index_of_tokenization_variant
        # na taki sam jak wiersz odpowiadajacy tej interpretacji ktora byla lepsza
        #
        # interpretations (<variants_to_follow>, <sequence_length>) - <variants_to_follow> wierszy, każdy wiersz początkowo powinien 
        # zawierać n (np. 2) identyfikatorów znaczeń i będzie uzupełniany o kolejne na podstawie najbardziej prawdopodobynch znaczeń tokenów
        #
        # tokens_meanings_lookup (<liczba roznych tokenow>, <maksymalna liczba znaczen slowa>)
        #
        # interpretations_probabelities (<liczba interpretacji>,) - dla każdej interpretacji przechowuje logit prawdopodobieństwa tej interpretacji
        next_word_probabilities, new_lm_states = self.language_model.call(state.interpretations, state.lm_state)
        position_in_tokenization_variants = state.position_in_tokenization_variants + 1

        
        # produce array of probabilites of dims - (interpretation, next word), and values being probabilites of interpratation if continued by next word
        # uses broadcasting to add current intepretation probability to the probability of every word in vocab being next word in this interpretation 
        new_interpretations_probabilities = next_word_probabilities + tf.expand_dims(state.interpretations_probabilities, 1)

        next_word_probabilities = self._mask_senses_matching_words(
            new_interpretations_probabilities, 
            state.tokenization_variants, 
            state.index_of_tokenization_variant, 
            state.position_in_tokenization_variants,
            tokens_meanings_lookup) 

        # TODO: Gdzieś trzeba wstawić 0 jako rozwiniecie interpretacji, jeśli prawdopodobieństwa wszystkich słów są zerem 
        # bo nie odnaleziono zadnego mozliwego znaczenia slowa.

        new_top_interpretations_probabilites, (top_interpretations, top_meanings) = \
            self._top_k_from_2d_tensor(new_interpretations_probabilities, self.variants_to_follow)

        new_interpretations = tf.gather(state.interpretations, top_interpretations)

        # tu jest ważne założenie, że id znaczenia to jego indeks w warstwie wyjściowej Modelu Języka
        # i jeszcze kolejne zalożenie, że wektor wyjściowy za każdym razem rośnie o 1
        new_interpretations = tf.concat((new_interpretations, [top_meanings]), axis=1) 
        new_len_of_tokenization_variants = tf.gather(len_of_tokenization_variants, top_interpretations)
        new_position_in_tokenization_variants = tf.gather(position_in_tokenization_variants, top_interpretations)
        new_index_of_tokenization_variant = tf.gather(index_of_tokenization_variant, top_interpretations)

        new_state = tokenization_variants, \
                    new_len_of_tokenization_variants, \
                    new_position_in_tokenization_variants, \
                    new_index_of_tokenization_variant, \
                    new_interpretations, \
                    new_top_interpretations_probabilites, \
                    new_lm_states
        return new_state








def test__get_token_meanings_mask():
    tokens = list("abcdef")
    tokens_senses = [
        [1,2],
        [3],
        [],
        [4],
        [5,6],
        [7,8,9]
    ]
    senses_dict = defaultdict(list, zip(tokens, tokens_senses))
    class S:
        def _get_possible_meanings_ids(self, token):
            return senses_dict[token]
        meanings_count = 9
    mask = BestInterpretationFinder._get_token_meanings_mask(S(), tokens)
    with tf.Session() as sess:
        r = sess.run(mask)
    np.testing.assert_equal(r.dense_shape, np.array([ 7, 10]))
    np.testing.assert_equal(r.values, np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))
    np.testing.assert_equal(r.indices, np.array(
        [[0, 0],
         [1, 1],
         [1, 2],
         [2, 3],
         [4, 4],
         [5, 5],
         [5, 6],
         [6, 7],
         [6, 8],
         [6, 9]]
         ))
    return r


def test__mask_senses_matching_words():
    interpretations_probabilities = tf.constant([
        [0, 0.3, 0.3, 0.4, 0.3, 0.3, 0.4, 0.3, 0.3, 0.4],
        [0, 0.3, 0.3, 0.4, 0.3, 0.3, 0.4, 0.3, 0.3, 0.4],
        [0, 0.3, 0.3, 0.4, 0.3, 0.3, 0.4, 0.3, 0.3, 0.4],
    ])
    tokenization_variants = tf.constant([
        [1,2,3,4,5,0,0],
        [1,2,6,0,0,0,0],
        [1,2,4,4,5,7,8],
    ])
    index_of_tokenization_variant = tf.constant([
        0,
        1,
        2,
    ])
    position_in_tokenization_variants = tf.constant([
        2,
        2,
        2,
    ])
    # Przy takich stałych jak powyżej mamy 4 możliwe znaczenia w słowniku (licząc 0 - nieznane znaczenie)
    # i 8 różych tokenów (zera w tokenization_variants to tylko padding)
    '''tf.constant([
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 1, 1],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 1, 0, 1],
        [0, 1, 0, 0],
        [0, 1, 1, 1],
    ])'''
    tokens = list("abcdef")
    tokens_senses = [
        [1,2],
        [3],
        [],
        [4],
        [5,6],
        [7,8,9]
    ]
    senses_dict = defaultdict(list, zip(tokens, tokens_senses))
    class S:
        def _get_possible_meanings_ids(self, token):
            return senses_dict[token]
        meanings_count = 9
    tokens_meanings_lookup = BestInterpretationFinder._get_token_meanings_mask(S(), tokens)
    mask = BestInterpretationFinder._mask_senses_matching_words(
            interpretations_probabilities, 
            tokenization_variants, 
            index_of_tokenization_variant, 
            position_in_tokenization_variants,
            tokens_meanings_lookup)
    with tf.Session() as sess:
        print(sess.run(mask))


def test__top_k_from_2d_tensor():
    c1 = tf.constant([[0.1, 0.2],
                      [0.3, 0.4]])
    c2 = tf.constant([[0.1, 0.2],
                      [0.5, 0.4]])
    c3 = tf.constant([[0.2, 3.2],
                      [0.4, 0.34],
                      [5.5, 2.4],
                      [0.5, 8.4]])
    c4 = tf.constant([[1.1, 0.2],
                      [0.5, 0.4]])
    top_k = BestInterpretationFinder._top_k_from_2d_tensor
    t1 =  top_k(c1, 1)
    exp1 = np.array([0.4]), (np.array([1]), np.array([1]))
    t2 =  top_k(c2, 1)
    exp2 = np.array([0.5]), (np.array([1]), np.array([0]))
    t3 =  top_k(c3, 1)
    exp3 = np.array([8.4]), (np.array([3]), np.array([1]))
    t4 =  top_k(c4, 1)
    exp4 = np.array([1.1]), (np.array([0]), np.array([0]))
    t3_1 =  top_k(c3, 3)
    exp3_1 = np.array([8.4, 5.5, 3.2]), (np.array([3, 2, 0]), np.array([1, 0, 1]))
    with tf.Session() as sess:
        r1, r2, r3, r4, r3_1 = sess.run((t1, t2, t3, t4, t3_1))

    assert exp1[1] == r1[1]
    assert exp2[1] == r2[1]
    assert exp3[1] == r3[1]
    assert exp4[1] == r4[1]
    np.testing.assert_array_almost_equal(exp3_1[1][0], r3_1[1][0])
    np.testing.assert_array_almost_equal(exp3_1[1][1], r3_1[1][1])

    np.testing.assert_array_almost_equal(exp1[0], r1[0])
    np.testing.assert_array_almost_equal(exp2[0], r2[0])
    np.testing.assert_array_almost_equal(exp3[0], r3[0])
    np.testing.assert_array_almost_equal(exp4[0], r4[0])
    np.testing.assert_array_almost_equal(exp3_1[0], r3_1[0])
