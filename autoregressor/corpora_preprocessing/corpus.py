from abc import ABC, abstractmethod

class DatasetType:
    TRAIN = "train"
    VALID = "valid"
    TEST = "test"

class Corpus:
    @abstractmethod
    def get_tokens_dataset(self, subset):
        """Returns tf.data.Dataset object yielding sentences from subset of corpus as tensors of a shape [sentence_length] and type tf.string.
        Subset should be one of options from DatasetType."""
