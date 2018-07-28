from find_cheapest_path import *
from pytest import approx


for name in [name for name in globals() if name[:5] == "test_"]:
    del globals()[name]






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


def test__interpretations_done():
    len_of_tokenization_variants = tf.constant([3,4,5])
    index_of_tokenization_variant = tf.constant([0,1,1,2])
    test_cases_position_in_tokenization_variants = [
        tf.constant([1,2,2,3]),
        tf.constant([3,4,4,5]),
        tf.constant([3,3,4,5]),
        tf.constant([0,3,4,5]),
        tf.constant([0,3,4,4]),
        tf.constant([0,3,2,4]),
        tf.constant([0,4,2,4]),
        tf.constant([3,4,2,4]),
        tf.constant([1,4,2,5]),
    ]

    expected_results = np.array([
        [0, 0, 0, 0],
        [1, 1, 1, 1],
        [1, 0, 1, 1],
        [0, 0, 1, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 1, 0, 1],
    ], dtype=np.bool)

    test_cases_state = \
    [
        InterpretationSearchState(
            "tokenization_variants", 
            len_of_tokenization_variants,
            position_in_tokenization_variants,
            index_of_tokenization_variant, 
            "interpretations", 
            "interpretations_probabilities", 
            "lm_state"
        ) for position_in_tokenization_variants in test_cases_position_in_tokenization_variants
    ]

    test_cases_computed_done = [BestInterpretationFinder._interpretations_done(case) for case in test_cases_state]

    with tf.Session() as sess:
        results = sess.run(test_cases_computed_done)

    np.testing.assert_equal(np.array(results), expected_results)


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




def test__step_2_unknown():
    # końcepcja jest taka:
    # zapisać kilka sekwencji w postaci numerów tokenów 
    # i prawdopodobieństwa odpowiadające kolejnym znaczeniom i mappingi ze znaczenia na kolejne znaczenie
    # a potem przerabiać nr-y tokenów na one hot embeddingsy i w call zaimplementować ojejku, to się robi skomplikowane
    no_of_meanings = 5
    class MockRnn(tf.nn.rnn_cell.RNNCell):
        def __init__(self, number_of_meanings, size_of_input_embedding):
            self.number_of_meanings = number_of_meanings
            self.size_of_input_embedding = size_of_input_embedding

        def call(self, input, state):
            output = tf.one_hot(state, self.size_of_input_embedding)
            return output, state+1

    mock_language_model = MockRnn(no_of_meanings, no_of_meanings+1)
    lm_state = tf.constant([3, 5,]) # w takiej konfiguracji tylko pierwsza interpretacja zyska na prawdopodobienstwie
    # bo w przypadku drugiej prawdopodobieństwo wszystkich znaczeń uprawdopodabnianiych przez tokeny jest zerowe

    interpretations_probabilities = tf.constant([
        [0.6],
        [0.2],
    ])
    tokenization_variants = tf.constant([
        [1,2,3],
    ])
    index_of_tokenization_variant = tf.constant([
        0,
        0,
    ])
    position_in_tokenization_variants = tf.constant([
        1,
        1,
    ])
    len_of_tokenization_variants = tf.constant([
        3, 
        3,
    ])
    tokens = list("abc")
    tokens_senses = [
        [1,2],
        [],
        [5],
    ]

    interpretations = tf.constant(
        [
            [1, ],#0, 0,],
            [2, ],#0, 0,],
        ]
    )

    senses_dict = defaultdict(list, zip(tokens, tokens_senses))
    class S:
        def _get_possible_meanings_ids(self, token):
            return senses_dict[token]
        meanings_count = no_of_meanings
    tokens_meanings_lookup = BestInterpretationFinder._get_token_meanings_mask(S(), tokens)


    finder = BestInterpretationFinder(
                 mock_language_model,
                 variants_to_follow=2
                 )
    def _get_possible_meanings_ids(self, token):
            return senses_dict[token]
    type(finder)._get_possible_meanings_ids = _get_possible_meanings_ids
    finder.meanings_count = no_of_meanings
    tokens_meanings_lookup = finder._get_token_meanings_mask(tokens)
    
    current_state = InterpretationSearchState(
            tokenization_variants, 
            len_of_tokenization_variants,
            position_in_tokenization_variants,
            index_of_tokenization_variant, 
            interpretations, 
            interpretations_probabilities, 
            lm_state
        )

    expected_index_of_tokenization_variant = np.array([
        0,
        0,
    ])
    expected_position_in_tokenization_variants = np.array([
        2,
        2,
    ])
    expected_interpretations = np.array([
        [1, 0],
        [2, 0],
    ])
    expected_interpretations_probabilities = np.array([
        [0.6],
        [0.2],
    ], dtype=np.float32)

    with tf.Session() as sess:
        new_state = sess.run(finder.step(current_state, tokens_meanings_lookup))

    print(new_state)

    np.testing.assert_equal(expected_index_of_tokenization_variant, new_state.index_of_tokenization_variant, "tokenization variant mismatch")
    np.testing.assert_equal(expected_position_in_tokenization_variants, new_state.position_in_tokenization_variants, "position in tokenization mismatch")

    #np.testing.assert_equal(expected_interpretations, new_state.interpretations, "interpretations mismatch")
    np.testing.assert_almost_equal(expected_interpretations_probabilities, new_state.interpretations_probabilities, 7, "interpretations propabilities mismatch")

