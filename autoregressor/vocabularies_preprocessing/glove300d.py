from pathlib import Path
import logging

import numpy as np
import tensorflow as tf

from config import VOCABULARIES_BASE_DIR

VECORS_FILE_NAME = "vectors.npy"
INDEX_FILE_NAME = "index.txt"
EXCLUDED_ELEMENT_TEMPLATE = "<EXCLUDED_VOCABULARY_ELEMENT_{}>"
UNKNOWN_ELEMENT_MARKER = "<UNKNOWN>"


def get_words_to_id_op():
    index_path = str(get_base_dir()/INDEX_FILE_NAME)
    index = tf.contrib.lookup.index_table_from_file(
        vocabulary_file=index_path,
        default_value=-1,
    )
    return index.lookup


def get_id_to_word_op():
    index_path = str(get_base_dir()/INDEX_FILE_NAME)
    table = tf.contrib.lookup.index_to_string_table_from_file(
        vocabulary_file=index_path, default_value=UNKNOWN_ELEMENT_MARKER)
    return table.lookup


def maybe_prepare_glove300_vocabulary():
    base_dir = get_base_dir()
    already_done = True
    already_done &= (base_dir/INDEX_FILE_NAME).is_file()
    already_done &= (base_dir/VECORS_FILE_NAME).is_file()
    if not already_done:
        prepare_glove300_vocabulary()


def prepare_glove300_vocabulary():
    base_dir = get_base_dir()
    raw_file_name = base_dir/"glove.840B.300d.txt"
    make_index_and_vectors_cache(raw_file_name, base_dir)


def get_base_dir():
    return VOCABULARIES_BASE_DIR/"glove300"


def make_index_and_vectors_cache(glove_file_name, cache_dir):
    glove_file_name, cache_dir = Path(glove_file_name), Path(cache_dir)
    words, vectors = read_vecotors_from_file(glove_file_name)
    
    index_path = cache_dir/INDEX_FILE_NAME
    save_index(words, index_path)

    vectors_path = cache_dir/VECORS_FILE_NAME
    save_vectors(vectors, vectors_path)


def save_index(words, index_path):
    with open(index_path, "wt") as index_file:
        for word in words:
            print(word, file=index_file)


def save_vectors(vectors, vectors_path):
    np.save(vectors_path, vectors)


def read_vecotors_from_file(file_name):
    """Reads file with glove vectors
    Repeated entries are replaced with special uniq string to prevent error when loockup is created. """
    words_set = set()
    words = []
    embeddings = []
    number_of_excluded_entries = 0
    with open(file_name) as f:
        for line in f:
            elements = line.split(" ")
            word = elements[0]
            if word in words_set:
                number_of_excluded_entries += 1
                logging.warning("Found repeated element, excuding (this is {}-th excluded element): {}".format(number_of_excluded_entries, word))
                word = EXCLUDED_ELEMENT_TEMPLATE.format(number_of_excluded_entries)
            words_set.add(word)
            words.append(word)
            vector = np.array([float(coeff) for coeff in elements[1:]])
            embeddings.append(vector)
    return words, np.stack(embeddings, 0)
