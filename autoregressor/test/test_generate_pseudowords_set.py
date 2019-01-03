from generate_pseudowords_set import *

import pytest
from pytest import approx

import numpy as np

from collections import Counter
from itertools import product

@pytest.mark.parametrize("input, expected_output",
    [
        ([1, 1, 1], [1, 1 ,1]),
        ([1.0,0.5,0.5], [1,1,0]),
        ([1.0,0.2,0.8], [1,1,0]),
        ([2.0,0.2,0.8], [2,1,0]),
        ([2.5,0.2,0.3], [3,0,0]),
        ([0.2,0.3,2.5], [1,0,2]),
        ([0.3,0.2,2.5], [1,0,2]),
        ([0.3,2.5,0.2], [1,2,0]),
    ]
)
def test_round_to_int_preserving_total(input, expected_output):
    actual_output = round_to_int_preserving_total(input)
    assert (np.array(expected_output) == np.array(actual_output)).all()
    assert sum(expected_output) == approx(sum(input))


@pytest.mark.parametrize("words_ambiguity_proportions",
    [
        [1],
        [0.5,0.5],
        [0.8,0.2],
        [0.8,0.1,0.1],
        [0.7,0.2,0.1],
        [0.1, 0.9]
    ]
)
def test_generate_pseudowords_set(words_ambiguity_proportions):
    uniq_pseudowords = set()
    words = [a+b for a, b in product([chr(i) for i in range(ord("a"),ord("z")+1)], [chr(i) for i in range(ord("A"),ord("Z")+1)])]
    to_pseudowords_mapping = PseudoWordsVocabulary.generate_pseudowords_set(words, words_ambiguity_proportions)
    words_in_pseudowords = set()
    for pseudoword in to_pseudowords_mapping.values():
        uniq_pseudowords.add(tuple(pseudoword))
        for word in pseudoword:
            assert word in word
            words_in_pseudowords.add(word)
    
    for word in to_pseudowords_mapping:
        assert word in to_pseudowords_mapping[word]
        assert word in words
    
    for word in words:
        assert word in words_in_pseudowords
        assert word in to_pseudowords_mapping

    actual_counts = Counter(len(pseudoword) for pseudoword in uniq_pseudowords)
    assert len(actual_counts) == len(words_ambiguity_proportions)
    actual_counts = [actual_counts[i] for i in range(1, len(actual_counts)+1)]
    assert np.array(actual_counts)/len(uniq_pseudowords) == approx(np.array(words_ambiguity_proportions), 0.05)