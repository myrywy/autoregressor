import argparse
from corpora_preprocessing import simple_examples
from vocabularies_preprocessing import glove300d

def maybe_download_all():
    simple_examples.maybe_get_corpus()

def maybe_prepare_embeddings():
    glove300d.maybe_prepare_glove300_vocabulary()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode")
    parser.add_argument("corpus")
    args = parser.parse_args()
    if args.mode == "maybe_download" and args.corpus == "all":
        maybe_download_all()
    if args.mode == "maybe_prepare_embeddings" and args.corpus == "all":
        maybe_prepare_embeddings()