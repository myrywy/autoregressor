from vocabularies_preprocessing.glove300d import Glove300
from vocabularies_preprocessing.mock_vocabulary import MockVocab


VOCABULARY_TYPES = {
    "glove300": Glove300,
    "mock_vocab": MockVocab,
}

def get_vocabulary(name):
    return VOCABULARY_TYPES[name]()

