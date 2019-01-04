from itertools import chain
from pathlib import Path
import datetime
import logging
import argparse

import numpy as np


def as_versioned_resource(output_directory_root, input_components, params):
    output_root = Path(output_directory_root)
    resource_name = "pseudowords-{}-{}".format("_".join(map(str, params)), datetime.date.today().strftime("%Y%m%d"))
    output_dir = (output_root/resource_name)
    output_dir.mkdir(parents=True)
    with open(output_dir/"inputs.txt", "wt") as f:
        for input in input_components:
            print(input, file=f)
    with open(output_dir/"params.txt", "wt") as f:
        for param in params:
            print(param, file=f)
    return output_dir



def prepare_pseudowords_experiment_data(input_file_paths, output_directory, words_ambiguity_proportions):
    """Ouput files will be named the same as input files => output direcory should not contain input files."""
    output_directory = Path(output_directory)
    corpora = [get_corpus(input_file_path) for input_file_path in input_file_paths]
    output_corpora, pseudo_voc = transform_corpora_with_common_pseudowords_set(corpora, words_ambiguity_proportions)
    pseudo_voc.to_file(output_directory/"pseudo-words-vocabulary.txt")
    for input_path, corpus in zip(input_file_paths, output_corpora):
        name =  Path(input_path).name
        output_path = output_directory/name
        save_corpus(output_path, corpus)


def transform_corpora_with_common_pseudowords_set(corpora, words_ambiguity_proportions):
    tokens_all = flatten(chain(*corpora))
    pseudo_voc = PseudoWordsVocabulary.from_words(tokens_all, words_ambiguity_proportions)
    return [pseudo_voc.transform_corpus(corpus) for corpus in corpora], pseudo_voc


class PseudoWordsVocabulary:
    INTERNAL_SEPARATOR = "^"
    ANNOTATION_SEPARATOR = "|"

    def __init__(self):
        self._mapping = None
    
    @staticmethod
    def from_words(original_words, words_ambiguity_proportions):
        """Create new pseudo-words vocabulary, see `generate_pseudowords_set` for more details"""
        new = PseudoWordsVocabulary()
        new._mapping = PseudoWordsVocabulary.generate_pseudowords_set(original_words, words_ambiguity_proportions)
        return new

    @staticmethod
    def from_file(self, file_path):
        mapping = {}
        with open(file_path) as f:
            for pseudoword in f:
                pseudoword = pseudoword.strip()
                original_words = pseudoword.split(PseudoWordsVocabulary.INTERNAL_SEPARATOR)
                for word in original_words:
                    mapping[word] = original_words
        new = PseudoWordsVocabulary()
        new._mapping = mapping
    
    def to_file(self, file_path):
        separator = PseudoWordsVocabulary.INTERNAL_SEPARATOR
        pseudowords_all = [separator.join(words) for words in self._mapping.values()]
        pseudowords_all = set(pseudowords_all)
        pseudowords_all = sorted(
            pseudowords_all, 
            key=lambda pseudoword: (-pseudoword.count(separator), pseudoword)
            )
        with open(file_path, "wt") as output_file:
            for pseudoword in pseudowords_all:
                print(pseudoword, file=output_file)

    def transform_corpus(self, corpus, annotate=True):
        """
        Args:
            corpus (Iterable[Iterable[str]]): inner Iterables represent sentences, str - words
        """
        for sentence in corpus:
            new_sentnce = []
            for word in sentence:
                new_word = self.INTERNAL_SEPARATOR.join(self._mapping[word])
                if annotate:
                    new_word = new_word + self.ANNOTATION_SEPARATOR + word
                new_sentnce.append(new_word)
            yield new_sentnce

    @staticmethod
    def generate_pseudowords_set(original_words, words_ambiguity_proportions):
        """
        Args:
            original_words (Iterable[str]): words from which pseudo-words will be generated
            words_ambiguity_proportions (List[float]): list of percentages of pseudowords of given number of meanings in the output set
                Should sum up to 1. For example with words_ambiguity_proportions = [0.6, 0.3, 0.1] 60% of pseudo-words will have one meaning, 
                30% will have two meanings and 10% will have 3 three meanings.
        Returns:
            dict: dictionary which keys are words from `original_words` and values are list of words that were merged into one pseudoword
        """
        words = set(original_words)
        n_uniq_words = len(words)
        denominator = sum(c_i * i for i, c_i in enumerate(words_ambiguity_proportions, 1))
        counts_of_pseudowords = np.array(words_ambiguity_proportions) * n_uniq_words / denominator

        counts_rounded = round_to_int_preserving_total(counts_of_pseudowords)

        # It is preventing leaks of a word when counts_rounded sets of given cardinalities doesn't divide original set.
        if sum(counts_rounded * np.arange(1, len(words_ambiguity_proportions)+1)) < n_uniq_words:
            remainder = n_uniq_words - sum(counts_rounded * np.arange(1, len(words_ambiguity_proportions)+1))
            counts_rounded[0] += remainder
            logging.warn("Added {} one-meaning pseudowords.".format(remainder))

        words_bag = words.copy()
        pseudowords = []
        for n_meanings, count in enumerate(counts_rounded, 1):
            pseudowords.append([])
            for _ in range(count):
                pseudoword_meanings = []
                for _ in range(n_meanings):
                    random_meaning = list(words_bag)[np.random.randint(0, len(words_bag))]
                    pseudoword_meanings.append(random_meaning)
                    words_bag.remove(random_meaning)
                pseudowords[-1].append(pseudoword_meanings)

        words_to_pseudowords_mapping = {}

        for n_meanings_pseudowords in pseudowords:
            for pseudoword in n_meanings_pseudowords:
                for word in pseudoword:
                    words_to_pseudowords_mapping[word] = pseudoword
        
        return words_to_pseudowords_mapping


def round_to_int_preserving_total(numbers_before_rounding):
    """Rounds number in such a way that their sum should remain the same (if their sum is integer from the beginning).

    Args:
        numbers_before_rounding (Iterable[float]): flaots that are to be rounded to ints. Should all be >= 0.
    Returns:
        Iterable[int]: rounded numbers from numbers_before_rounding
    """
    desired_total = np.round(sum(numbers_before_rounding))
    counts = [int(c) for c in numbers_before_rounding]
    remainder = desired_total - sum(counts)
    while remainder > 0:
        remainder = desired_total - sum(counts)
        underrepresented = [c < p for c, p in zip(counts, numbers_before_rounding)]
        for i, is_under in enumerate(underrepresented):
            if is_under:
                counts[i] += 1
                remainder -= 1
            if remainder == 0:
                break
    return counts


# TODO: Following functions maybe should be integrated in corpora classes
def get_corpus(file):
    with open(file) as f:
        sentences = []
        for line in f:
            sentences.append(line.strip().split())
    return sentences

def flatten(corpus):
    return (word for sentence in corpus for word in sentence)

def save_corpus(path, corpus):
    with open(path, "wt") as file:
        for sentence in corpus:
            print(" ".join(sentence), file=file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    parser.add_argument("--input-files", nargs="*")
    parser.add_argument("--ratios", nargs="*", type=float)

    args = parser.parse_args()

    outpur_dir = as_versioned_resource(args.output_dir, args.input_files, args.ratios)
    prepare_pseudowords_experiment_data(args.input_files, outpur_dir, args.ratios)