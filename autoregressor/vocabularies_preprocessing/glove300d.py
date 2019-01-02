from pathlib import Path
import logging
from functools import lru_cache
from collections import defaultdict

import numpy as np
import tensorflow as tf

from config import VOCABULARIES_BASE_DIR
from vocabularies_preprocessing.vocabulary import Vocabulary
from generalized_vocabulary import SpecialUnit

VECORS_FILE_NAME = "vectors.npy"
INDEX_FILE_NAME = "index.txt"
EXCLUDED_ELEMENT_TEMPLATE = "<EXCLUDED_VOCABULARY_ELEMENT_{}>"
UNKNOWN_ELEMENT_MARKER = "<UNKNOWN>"


def get_default_base_dir():
    return VOCABULARIES_BASE_DIR/"glove300"


class Glove300(Vocabulary):
    def __init__(self, base_dir=get_default_base_dir(), dry_run=False):
        super(Glove300, self).__init__()
        self._base_dir = Path(base_dir)
        self._index_file_path = self._base_dir/INDEX_FILE_NAME
        self._vectors_file_path = self._base_dir/VECORS_FILE_NAME
        self._embedding_assigns = defaultdict(list)
        self._embedding_variables = {}
        self._dry_run = dry_run

    def initialize_embeddings_in_graph(self, graph, session):
        print("initialize_embeddings_in_graph")
        print("graph:", graph)
        print("session", session)
        vectors_path = str(self._vectors_file_path)
        vectors = np.load(vectors_path)
        for assign_op, placeholder in self._embedding_assigns[graph]:
            print("assign_op:", assign_op)
            print("placeholder:", placeholder)
            session.run(assign_op, feed_dict={placeholder: vectors})

    def word_to_id_op(self):
        index_path = str(self._index_file_path)
        index = tf.contrib.lookup.index_table_from_file(
            vocabulary_file=index_path,
            default_value=-1,
        )
        def op(word):
            word = tf.convert_to_tensor(word, dtype=tf.string)
            return index.lookup(word)
        return op

    def id_to_word_op(self):
        index_path = str(self._index_file_path)
        table = tf.contrib.lookup.index_to_string_table_from_file(
            vocabulary_file=index_path, default_value=UNKNOWN_ELEMENT_MARKER)
        return table.lookup

    def id_to_vector_op(self):
        print("id_to_vector_op")
        print("graph:", tf.get_default_graph())
        embeddings = self._get_embeddings_variable(tf.get_default_graph())
        def op(id):
            with tf.device("/device:CPU:0"):
                return tf.nn.embedding_lookup(embeddings, id)
        return op
    
    def _get_embeddings_variable(self, graph):
        if self._dry_run:
            return None
        try:
            return self._embedding_variables[graph]
        except KeyError:
            with tf.device("/device:CPU:0"):
                embeddings = tf.Variable(tf.constant(0.0, shape=[self.vocab_size(), self.vector_size()]),
                    trainable=False, name="glove_embeddings")
                embedding_placeholder = tf.placeholder(tf.float32, [self.vocab_size(), self.vector_size()])
                self._embedding_assigns[tf.get_default_graph()].append((embeddings.assign(embedding_placeholder), embedding_placeholder))
                self._embedding_variables[graph] = embeddings
                return embeddings
        

    def special_unit_to_id(self, special_unit_name):
        return None 
        
    def get_non_id_integer(self):
        return 2196020

    def get_valid_id_example(self):
        return 1
    
    def vector_size(self):
        return 300

    @lru_cache()
    def vocab_size(self):
        index_path = str(self._index_file_path)
        with open(index_path) as index_file:
            return len([*index_file])


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
