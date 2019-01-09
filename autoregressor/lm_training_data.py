import tensorflow as tf 


from utils import without, maybe_inject_hparams
from vocabularies_preprocessing.glove300d import Glove300
from corpora_preprocessing.simple_examples import SimpleExamplesCorpus, DatasetType
from lm_input_data_pipeline import LmInputDataPipeline
from lm_dataset_io import CachedLmInputDataPipeline
from lstm_lm import get_autoregressor_model_fn
from lm_dataset_io import read_dataset_from_dir, save_dataset_to_files
from hparams import hparams

VOCABULARY_TYPES = {
    "glove300": Glove300
}

CORPUS_TYPES = {
    "simple_examples": SimpleExamplesCorpus
}

class LanguageModelTrainingData:
    def __init__(self, vocabulary_name, corpus_name, cached_data_dir, batch_size=None, shuffle_examples_buffer_size=None, hparams=hparams):
        self.vocabulary, self.corpus = VOCABULARY_TYPES[vocabulary_name](), CORPUS_TYPES[corpus_name]()
        self.batch_size = batch_size
        self.shuffle_examples_buffer_size = shuffle_examples_buffer_size
        self.cached_data_dir = cached_data_dir
        self.shuffle_examples_seed = None
        if hparams is not None: 
            maybe_inject_hparams(self, hparams, ["batch_size", "shuffle_examples_buffer_size", "shuffle_examples_seed", "cached_data_dir", "shuffle_examples_seed"])
        self.input_pipe = CachedLmInputDataPipeline(self.vocabulary, self.cached_data_dir, None, hparams=hparams)
        self.embedding_size = self.input_pipe._vocab_generalized.vector_size()

    def prepare_data(self):
        for subset in (DatasetType.TRAIN, DatasetType.VALID, DatasetType.TEST):
            token_based_dataset = self.corpus.get_tokens_dataset(subset)
            self.input_pipe.save_data(token_based_dataset, subset)

    def load_cached_data(self, subset):
        return self.input_pipe.load_cached_data(subset)

    def load_training_data(self):
        dataset = self.load_cached_data(DatasetType.TRAIN)
        dataset = dataset.repeat().shuffle(self.shuffle_examples_buffer_size, seed=self.shuffle_examples_seed)
        dataset = self.input_pipe.padded_batch(dataset, self.batch_size)
        return dataset


