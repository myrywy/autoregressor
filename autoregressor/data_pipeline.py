import tensorflow as tf
from generalized_vocabulary import GeneralizedVocabulary, SpecialUnit

class DataPipeline:
    def __init__(self):
        self._transformations = []


    def add_unit_transformation(
            self,
            transformation_funtion,
            *keys
            ):
        """Adds transformation of some unit in example to pipeline. Access unit in example by chain of keys used as example[keys[0]][keys[1]]...[keys[n]]

        Warning:
            If example yielded by dataset iterator has a form of one-element tuple then it will be reduced to this
            element - without enclosing tuple. This is because of the way Dataset.map function works.

        Args:
            transformation_funtion (Function[(tf.Tensor,), tf.Tensor]): function that takes tensor and return tensor, 
            keys (Iterable): keys used one by one to access element that is to be transformed in example
        
        Returns:
            None
        """
        def get_unit(example):
            for key in keys:
                example = example[key]
            return example

        def set_unit(unit, example, keys=keys):
            if not keys:
                return unit
            current_level = example
            key = keys[0]
            if isinstance(example, tuple):
                current_level = list(current_level)
                current_level[key] = set_unit(unit, current_level[key], keys=keys[1:])
                current_level = tuple(current_level)
            elif isinstance(example, dict):
                current_level = current_level.copy()
                current_level[key] = set_unit(unit, current_level[key], keys=keys[1:])
            else:
                raise ValueError("Unsupported container. Only dict and tuple allowed.")
            return current_level

        self.add_generic_unit_transformation_in_nested_structure(transformation_funtion, get_unit, set_unit)

    def add_generic_unit_transformation_in_nested_structure(
            self,
            transformation_funtion, 
            get_unit_from_example_function, 
            set_unit_in_example_function,
            ):
        """Adds transformation of some unit in example to pipeline.
        
        Args:
            transformation_funtion (Function[(tf.Tensor,), tf.Tensor]): function that takes tensor and return tensor, 
            get_unit_from_example_function: function that takes tuple of elements in dataset's example (for exaple tuple(features, labels)) and returns element that is to be transformed, 
            set_unit_in_example_function: function of signature: (unit, example) -> example that replace some element in example (that is some tuple of possibly nested dicts) with unit

        Returns:
            None
        """
        def example_transformation(*args):
            if len(args) == 1:
                args = args[0]
            else:
                args = args
            unit = get_unit_from_example_function(args)
            result = transformation_funtion(unit)
            args = set_unit_in_example_function(result, args)
            return args

        self._transformations.append(example_transformation)
    
    def add_structural_transformation(
            self,
            transformation_function,
            ):
        """Adds transformation of example to data pipeline

        Args:
            transformation_function: function that modifies dataset's example
        """
        self._transformations.append(transformation_function)

    def transform_dataset(self, dataset):
        """Sequentially applies added transformations to dataset.

        Args:
            dataset (tf.data.Dataset): a dataset which elements are to be transformed

        Returns:
            tf.data.Dataset: transformed dataset
        """
        for transformation in self._transformations:
            dataset = dataset.map(transformation)
        return dataset


class LmInputData(DataPipeline):
    def __init__(self, vocab, batch_size=20):
        """Creates preprocessing pipeline that converts token-based dataset into a dataset suitable for LanguageModel training.
        Input dataset examples should be 1D string tensor representing sentence (each element of such tensor is one word/token).

        Args:
            vocab (Vocabulary): vocabulary that will be used to convert tokens into 
            batch_size (int or None): size of batch created by load_data or None - then no batching will be performed
        """
        super(LmInputData, self).__init__()
        self.batch_size = batch_size
        self._vocab_generalized = vocab_generalized = GeneralizedVocabulary(
                vocab,
                [
                    SpecialUnit.START_OF_SEQUENCE,
                    SpecialUnit.END_OF_SEQUENCE,
                    SpecialUnit.OUT_OF_VOCABULARY
                ]
            )
        self.add_unit_transformation(vocab.word_to_id_op())
        self.add_unit_transformation(vocab_generalized.vocab_id_to_generalized_id())
        self.add_structural_transformation(self.make_input_target_example)
        self.add_unit_transformation(vocab_generalized.generalized_id_to_extended_vector(), 0, "inputs")

    def load_data(self, corpus):
        """Transforms dataset of string tensors into dataset with (features, labels) pair of a following structure:
        If self.batch_size is None
            features:
                {"inputs": <tensor size (sentence_length, embedding_vector_size) dtype float32>, "length": <tensor scalar of type int>}
            labels:
                {"targets": <tensor size (sentence_length,) dtype int32>}
        If self.batch_size is not None
            features:
                {"inputs": <tensor size (self.batch_size, sentence_length, embedding_vector_size) dtype float32>, "length": <tensor size (self.batch_size) type int>}
            labels:
                {"targets": <tensor size (self.batch_size, sentence_length,) dtype int32>}

        Args:
            corpus (tf.data.Dataset): dataset in which examples should be 1D string tensor representing sentence (each element of such tensor is one word/token). 
        """
        corpus = self.transform_dataset(corpus)
        if self.batch_size is not None:
            corpus = self.padded_batch(corpus, self.batch_size)
        return corpus

    def padded_batch(self, corpus, batch_size):
        corpus = self._padded_batch(corpus, batch_size)
        corpus = corpus.map(lambda features, labels: self._fix_dimensions(features, labels, batch_size, self._vocab_generalized.vector_size()))
        return corpus

    def make_input_target_example(self, sequence):
        """Transforms sequence into pair of input_sequnce and targets that can be used to learn language model.
        Precisely, output cosist in tuple of features and labels. Features is a dict containing "inputs"
        and "length". Lables is a dict containing "targets". inputs and targets are copies of `sequence` 
        but with special <start> element id prepended in `inputs` and <end> element id appended to `targets`.

        Args:
            sequence: 1-D vector of word ids
        """
        features, labels = {}, {}
        start = self._vocab_generalized.get_special_unit_id(SpecialUnit.START_OF_SEQUENCE)
        stop = self._vocab_generalized.get_special_unit_id(SpecialUnit.END_OF_SEQUENCE)
        sequence = tf.concat(([start], sequence, [stop]), axis=0)
        inputs = sequence[:-1]
        length = tf.shape(inputs)[0]
        features["length"] = length
        features["inputs"] = inputs
        labels["targets"] = sequence[1:]
        return features, labels

    def _padded_batch(self, dataset, batch_size):
        def expand_length(features, labels):
            features["length"] = tf.expand_dims(features["length"], 0)
            return features, labels

        def flatten_length(features, labels):
            features["length"] = tf.squeeze(features["length"], axis=[1])
            return features, labels

        length_expanded_data = dataset.map(expand_length)
        length_expanded_data = length_expanded_data. \
            padded_batch(
            batch_size,
            padded_shapes=(
                {"inputs": tf.TensorShape((tf.Dimension(None), tf.Dimension(None))),
                 "length": tf.TensorShape((tf.Dimension(1),))},
                {"targets": tf.TensorShape((tf.Dimension(None),))}
            )
        )
        return length_expanded_data.map(flatten_length)

    def _fix_dimensions(self, features, labels, batch_size, vector_size):
        """After _padded_batch is performed, some dimesions of tensors are unknown so they have to be fixed "manually" to prevent som errors when using these tensors."""
        features["inputs"].set_shape((batch_size, None, vector_size))
        features["length"].set_shape((batch_size,))
        labels["targets"].set_shape((batch_size, None))
        return features, labels
    
    def get_id_to_embedding_mapping(self):
        return self._vocab_generalized.generalized_id_to_extended_vector()
