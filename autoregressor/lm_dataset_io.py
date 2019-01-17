from contextlib import contextmanager
from functools import partial
from itertools import count
from pathlib import Path
import logging
import re

import tensorflow as tf
from lm_input_data_pipeline import LmInputDataPipeline


class CachedLmInputDataPipeline(LmInputDataPipeline):
    DEFAULT_EXAMPLES_PER_FILE = 2000
    def __init__(self, vocab, data_dir, batch_size=20, examples_per_file=None, hparams=None):
        super(CachedLmInputDataPipeline, self).__init__(vocab, batch_size)
        self.data_dir = data_dir
        # TODO: vvv refactor vvv
        if examples_per_file:
            self.examples_per_file = examples_per_file
        elif hparams is not None:
            try:
                self.examples_per_file = hparams.number_of_cached_examples_in_one_file
            except AttributeError:
                self.examples_per_file = self.DEFAULT_EXAMPLES_PER_FILE
        else:
            self.examples_per_file
        

    def save_data(self, corpus, subset):
        corpus = self.transform_dataset(corpus)
        get_output_file_name_fn = partial(get_output_file_name, self.data_dir, subset)
        writer_manager = EqualSizeRecordFilesWriter(self.examples_per_file, get_output_file_name_fn)
        save_dataset_to_files(
            corpus, 
            make_tf_record_example, 
            writer_manager, 
            self._vocab.after_session_created_hook_fn)

    def load_cached_data(self, subset):
        return read_dataset_from_dir(self.data_dir, subset, self._vocab_generalized.vector_size())



def read_dataset_from_dir(data_dir, only_named_subset, embedding_size):
    """
    Note: side-effect - this function creates tensors in default Graph
    """
    files_paths = list_dataset_files_in_directory(data_dir, only_named_subset)
    files_paths = tf.convert_to_tensor(files_paths, dtype=tf.string)
    dataset = read_dataset_from_files(files_paths, embedding_size)
    return dataset


def list_dataset_files_in_directory(data_dir, only_named_subset=None):
    data_dir = Path(data_dir)
    data_files = [file_path for file_path in data_dir.iterdir() if re.match(r"\w+\.\d+\.tfrecords$",file_path.name)]
    data_files = sorted(data_files, key=lambda p: p.name)
    if only_named_subset is not None:
        data_files = [file_path for file_path in data_files if file_path.name.split(".")[0] == only_named_subset]
    data_files = [str(file_path) for file_path in data_files]
    return data_files


def read_dataset_from_files(input_paths, embedding_size):
    """
    Note: side-effect - may create tensors in default Graph
    """
    input_paths = tf.convert_to_tensor(input_paths)

    feature_description = {
        'inputs': tf.FixedLenSequenceFeature([embedding_size], tf.float32, allow_missing=True, default_value=0.0),
        'length': tf.FixedLenFeature([], tf.int64, default_value=0),
        'targets': tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True, default_value=0),
    }

    def parse_tf_record(example_proto):
        return tf.parse_single_example(example_proto, feature_description)

    def restore_structure(example):
        features = {"inputs": example["inputs"], "length": example["length"]}
        labels = {"targets": example["targets"]}
        return features, labels

    dataset = tf.data.TFRecordDataset(input_paths)
    dataset = dataset.map(parse_tf_record)
    dataset = dataset.map(restore_structure)
    return dataset

def make_tf_record_example(features, labels) -> tf.train.SequenceExample:
    feature_inputs = tf.train.Feature(float_list=tf.train.FloatList(value=features["inputs"].reshape(-1)))
    feature_length = tf.train.Feature(int64_list=tf.train.Int64List(value=[features["length"]]))
    feature_targets = tf.train.Feature(int64_list=tf.train.Int64List(value=labels["targets"]))
    feature_dict = {"inputs": feature_inputs, "length": feature_length, "targets": feature_targets}
    features = tf.train.Features(feature=feature_dict)
    example = tf.train.Example(features=features)
    return example
    

def get_output_file_name(output_path, prefix, example_index):
    output_path = Path(output_path)
    return str(output_path/"{}.{:0=10}.tfrecords".format(prefix, example_index+1))

@contextmanager
def EqualSizeRecordFilesWriter(examples_per_file, get_output_file_name_fn):
    open_writer = None
    filename = None
    def writer_generator():
        nonlocal open_writer
        nonlocal filename
        for file_index in count():
            filename = get_output_file_name_fn(file_index)
            logging.info("Opening file for write {}".format(filename))
            open_writer = tf.python_io.TFRecordWriter(filename)
            for _ in range(examples_per_file):
                yield open_writer
            open_writer.close()
            open_writer = None
    try:
        yield writer_generator()
    finally:
        if open_writer is not None:
            logging.info("Closing leftover tfrecords file {}".format(filename))
            open_writer.close()


def save_dataset_to_files(dataset, make_tf_record_example, file_writer_manager, custom_initialization_fn):
    """
    Args:
        file_writer_manager: context manager that returns generator yielding tf.python_io.TFRecordWriter for every example that is to be wtitten
        custom_initialization_fn: function that is run after sesion is opened and after default initializations are performed, should have following signature 
            Args:
                session: session in which dataset iteration and saving will be performed
        make_tf_record_example: function that converts features and labels from dataset into tf.train.Example that will be serialized and saved
    """
    it = dataset.make_initializable_iterator()
    next_example = it.get_next()

    with tf.Session() as sess:
        if custom_initialization_fn is not None:
            custom_initialization_fn(sess)
        sess.run(tf.tables_initializer())
        sess.run(it.initializer)

        with file_writer_manager as writers_manager:
            for i in count():
                if i % 1000:
                    logging.info("Writting {}-th example".format(i+1))
                try:
                    features, labels = sess.run(next_example)
                    example = make_tf_record_example(features, labels)

                    writer = next(writers_manager)
                    writer.write(example.SerializeToString())
                except tf.errors.OutOfRangeError:
                    logging.info("Finished writting database. Written {} examples".format(i))
                    break
