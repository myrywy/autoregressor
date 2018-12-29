import os
import urllib.request
from pathlib import Path
from subprocess import run
from tempfile import TemporaryDirectory
import tensorflow as tf 

def get_corpus():
    base_data_dir = get_create_base_data_dir()
    download_path = base_data_dir / "simple-examples.tgz"
    extracted_data_dir = get_corpus_base_dir() / "raw"

    extracted_data_dir.mkdir(parents=True)

    download(download_path)
    extract_needed(download_path, extracted_data_dir)

    download_path.unlink()


def maybe_get_corpus():
    base_dir = get_corpus_base_dir()
    corpus_fetched = True
    corpus_fetched &= (base_dir/"raw"/"ptb.test.txt").is_file()
    corpus_fetched &= (base_dir/"raw"/"ptb.train.txt").is_file()
    corpus_fetched &= (base_dir/"raw"/"ptb.valid.txt").is_file()
    if not corpus_fetched:
        get_corpus()


def get_corpus_base_dir():
    return get_create_base_data_dir() / "simple_examples"


def get_create_base_data_dir():
    code_root_dir = Path(__file__).parent.parent
    data_dir = code_root_dir/"data"
    if not data_dir.is_dir():
        data_dir.mkdir(parents=True)
    return data_dir


def download(download_path):
    response = urllib.request.urlopen("http://www.fit.vutbr.cz/%7Eimikolov/rnnlm/simple-examples.tgz")
    data = response.read()
    with open(download_path, "wb") as f:
        f.write(data)


def extract_needed(targz_path, output_path):
    """Extract files that are necessary from tar.gz file under targz_path and place in output_path"""
    output_path = Path(output_path)
    with TemporaryDirectory() as tmpdir:
        run(["tar", "-zxvf", targz_path, "-C", tmpdir])
        tmp_data_dir = Path(tmpdir) / "./simple-examples/data"
        (tmp_data_dir/"ptb.test.txt").replace(output_path/"ptb.test.txt")
        (tmp_data_dir/"ptb.train.txt").replace(output_path/"ptb.train.txt")
        (tmp_data_dir/"ptb.valid.txt").replace(output_path/"ptb.valid.txt")


def preprocess_file_for_standard_language_model_training(path, get_word_to_ids_fn):
    """
    Args:
        get_word_to_ids_fn: a function that takes tensor of strings and returns 
            tensor of their ids according to embeddings lookup
    """
    dataset = tf.data.TextLineDataset(path)
    dataset = text_dataset_to_token_ids(dataset, get_word_to_ids_fn)
    return dataset

def text_dataset_to_token_ids(dataset, get_word_to_ids_fn):
    dataset = dataset.map(lambda string: tf.string_split([string]).values)
    lookup = get_word_to_ids_fn()
    dataset = dataset.map(lambda strings: lookup(strings))
    return dataset

def token_ids_to_text_dataset(dataset, get_id_to_words_fn):
    lookup = get_id_to_words_fn()
    dataset = dataset.map(lambda strings: lookup(strings))
    return dataset

class DatasetType:
    TRAIN = "train"
    VALID = "valid"
    TEST = "test"


class SimpleExamplesCorpus:
    def __init__(self, root_path=get_corpus_base_dir()):
            self._root_path = Path(root_path)

    def get_tokens_dataset(self, subset):
        file_path = str(self._get_file_path(subset))
        dataset = tf.data.TextLineDataset([file_path])
        dataset = dataset.map(lambda string: tf.string_split([string]).values)
        return dataset

    def _get_file_path(self, subset):
        if subset == DatasetType.TRAIN:
            return self._root_path/"raw"/"ptb.train.txt"
        if subset == DatasetType.VALID:
            return self._root_path/"raw"/"ptb.valid.txt"
        if subset == DatasetType.TEST:
            return self._root_path/"raw"/"ptb.test.txt"
        else:
            raise ValueError("Invalid corpus subset type: {}".format(subset))


    
