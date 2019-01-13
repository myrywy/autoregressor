from corpora_preprocessing.simple_examples import SimpleExamplesCorpus

CORPUS_TYPES = {
    "simple_examples": SimpleExamplesCorpus
}

def get_corpus(name):
    return CORPUS_TYPES[name]()