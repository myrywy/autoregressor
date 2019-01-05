from functools import partial
import shutil

import pytest
from pytest import approx

import tensorflow as tf
import numpy as np

from lm_dataset_io import EqualSizeRecordFilesWriter, read_dataset_from_dir, read_dataset_from_files, make_tf_record_example, get_output_file_name, save_dataset_to_files
from lm_input_data_pipeline import LmInputDataPipeline
from config import TEST_TMP_DIR

def mock_TFRecordWriter_classes_factory(init_callback=None, write_callback=None, close_callback=None):
    class MockTFRecordWriter:
        def __init__(self, filename):
            self.filename = filename
            self.isclosed = False
            init_callback(self, filename)

        def write(self, data):
            if write_callback is not None:
                write_callback(self, data)
        
        def __enter__(self, *a, **k):
            return self

        def __exit__(self, *a):
            self.close()

        def close(self):
            self.isclosed = True
            if close_callback is not None:
                close_callback(self)
    return MockTFRecordWriter


def test_save_and_restore_dataset_one_file():
    if TEST_TMP_DIR.is_dir():
        shutil.rmtree(TEST_TMP_DIR)
    TEST_TMP_DIR.mkdir()
    data_file_path = str(TEST_TMP_DIR/"data.tfrecords")
    examples_per_file = 100000
    examples = [
        ({"inputs": np.array([[1.5,1.5],[2.5,2.5],[3.5,3.5],[4.5,4.5]]), "length": 4}, {"targets": np.array([1,2,3,4], dtype=np.int32)}),
        ({"inputs": np.array([[1.5,1.5],[2.5,2.5],[3.5,3.5]]), "length": 3}, {"targets": np.array([1,2,3], dtype=np.int32)}),
        ({"inputs": np.array([[1.5,1.5],[2.5,2.5]]), "length": 2}, {"targets": np.array([1,2], dtype=np.int32)}),
        ({"inputs": np.array([[1.5,1.5]]), "length": 1},  {"targets": np.array([1], dtype=np.int32)}),
        ({"inputs": np.array([[1.5,1.5],[2.5,2.5]]), "length": 2}, {"targets": np.array([1,2], dtype=np.int32)}),
        ({"inputs": np.array([[1.5,1.5],[2.5,2.5],[3.5,3.5]]), "length": 3}, {"targets": np.array([1,2,3], dtype=np.int32)}),
        ({"inputs": np.array([[1.5,1.5],[2.5,2.5],[3.5,3.5],[4.5,4.5]]), "length": 4}, {"targets": np.array([1,2,3,4], dtype=np.int32)}),
    ]
    def test_data():
        return iter(examples)
    get_output_file_name_fn = lambda i: data_file_path
    file_writer_manager = EqualSizeRecordFilesWriter(examples_per_file, get_output_file_name_fn)
    dataset = tf.data.Dataset.from_generator(
        test_data, 
        output_types=({"inputs": tf.float32, "length": tf.int32}, {"targets": tf.int32}),
        output_shapes=({"inputs": [None, 2], "length": ()}, {"targets": [None]})
        )
    
    sanity_check_examples = []
    it = dataset.make_initializable_iterator()
    next_record = it.get_next()
    with tf.Session() as sess:
        sess.run(it.initializer)
        while True:
            try:
                sanity_check_examples.append(sess.run(next_record))
            except tf.errors.OutOfRangeError:
                break
    assert len(examples) == len(sanity_check_examples)
    for (expected_features, expected_labels), (sanity_features, sanity_labels) in zip(examples, sanity_check_examples):
        assert sanity_features["inputs"] == approx(expected_features["inputs"])
        assert (sanity_features["length"] == expected_features["length"]).all()
        assert (sanity_labels["targets"] == expected_labels["targets"]).all()   
            
    save_dataset_to_files(dataset, make_tf_record_example, file_writer_manager, None)

    
    records_dataset = read_dataset_from_files([data_file_path], embedding_size=2)
    it = records_dataset.make_initializable_iterator()
    next_record = it.get_next()
    with tf.Session() as sess:
        sess.run(it.initializer)
        for expected_features, expected_labels in examples:
            actual_features, actual_labels = sess.run(next_record)
            assert actual_features["inputs"] == approx(expected_features["inputs"])
            assert (actual_features["length"] == expected_features["length"]).all()
            assert (actual_labels["targets"] == expected_labels["targets"]).all()
    

def test_save_and_restore_dataset_multiple_files():
    if TEST_TMP_DIR.is_dir():
        shutil.rmtree(TEST_TMP_DIR)
    TEST_TMP_DIR.mkdir()
    data_dir = str(TEST_TMP_DIR)
    examples_per_file = 10
    examples = [
        ({"inputs": np.array([[1.5,1.5],[2.5,2.5],[3.5,3.5],[4.5,4.5]]), "length": 4}, {"targets": np.array([1,2,3,4], dtype=np.int32)}),
        ({"inputs": np.array([[1.5,1.5],[2.5,2.5],[3.5,3.5]]), "length": 3}, {"targets": np.array([1,2,3], dtype=np.int32)}),
        ({"inputs": np.array([[1.5,1.5],[2.5,2.5]]), "length": 2}, {"targets": np.array([1,2], dtype=np.int32)}),
        ({"inputs": np.array([[1.5,1.5]]), "length": 1},  {"targets": np.array([1], dtype=np.int32)}),
        ({"inputs": np.array([[1.5,1.5],[2.5,2.5]]), "length": 2}, {"targets": np.array([1,2], dtype=np.int32)}),
        ({"inputs": np.array([[1.5,1.5],[2.5,2.5],[3.5,3.5]]), "length": 3}, {"targets": np.array([1,2,3], dtype=np.int32)}),
        ({"inputs": np.array([[1.5,1.5],[2.5,2.5],[3.5,3.5],[4.5,4.5]]), "length": 4}, {"targets": np.array([1,2,3,4], dtype=np.int32)}),
    ]
    def test_data():
        return iter(examples)
    get_output_file_name_fn = partial(get_output_file_name, data_dir, "data")
    file_writer_manager = EqualSizeRecordFilesWriter(examples_per_file, get_output_file_name_fn)
    dataset = tf.data.Dataset.from_generator(
        test_data, 
        output_types=({"inputs": tf.float32, "length": tf.int32}, {"targets": tf.int32}),
        output_shapes=({"inputs": [None, 2], "length": ()}, {"targets": [None]})
        )
            
    save_dataset_to_files(dataset, make_tf_record_example, file_writer_manager, None)

    records_dataset = read_dataset_from_dir(data_dir, "data", embedding_size=2)
    it = records_dataset.make_initializable_iterator()
    next_record = it.get_next()
    with tf.Session() as sess:
        sess.run(it.initializer)
        for expected_features, expected_labels in examples:
            actual_features, actual_labels = sess.run(next_record)
            assert actual_features["inputs"] == approx(expected_features["inputs"])
            assert (actual_features["length"] == expected_features["length"]).all()
            assert (actual_labels["targets"] == expected_labels["targets"]).all()

    
def test_doesnt_raise_make_tf_record_example():
    make_tf_record_example({"inputs": np.array([[1.5,1.5],[2.5,2.5],[3.5,3.5],[4.5,4.5]], dtype=np.float32), "length": np.array(4, dtype=np.int32)}, {"targets": np.array([1,2,3,4], dtype=np.int32)})
    

@pytest.mark.parametrize("examples_per_file, n_writes, exception", 
    [
        (2, 2, False), 
        (1, 2, False), 
        (2, 1, False), 
        (1000, 12, False), 
        (12, 1000, False), 
        (11, 1000, False),
        (2, 1, True),  
        (12, 1000, True), 
        (12, 1001, True)
    ]
)
def test_EqualSizeRecordFilesWriter(monkeypatch, examples_per_file, n_writes, exception):
    def get_output_file_name_fn(number):
        return "data_{}.tfrecords".format(number)
    
    init = False
    write = False
    close = False
    used_writer = None
    current_writer = None
    last_filename = None
    last_written_data = None

    def init_callback(self, filename):
        nonlocal init
        nonlocal used_writer
        nonlocal last_filename
        init = True
        used_writer = self
        last_filename = filename
    def write_callback(self, data):
        nonlocal write
        nonlocal used_writer
        nonlocal last_written_data
        write = True
        used_writer = self
        last_written_data = data
    def close_callback(self):
        nonlocal close
        nonlocal used_writer
        nonlocal current_writer
        close = True
        used_writer = self
        assert current_writer == self


    MockTFRecordWriter = mock_TFRecordWriter_classes_factory(init_callback=init_callback, write_callback=write_callback, close_callback=close_callback)
    monkeypatch.setattr(tf.python_io, "TFRecordWriter", MockTFRecordWriter)

    try:
        with EqualSizeRecordFilesWriter(examples_per_file, get_output_file_name_fn) as writer_generator:
            for i in range(n_writes):
                writer = next(writer_generator)
                if i % examples_per_file == 0:
                    assert init == True
                    init = False
                    assert used_writer != current_writer
                    current_writer = used_writer
                else:
                    assert current_writer == used_writer
                if i % examples_per_file == 0 and i != 0:
                    assert close == True
                    close = False
                
                writer.write("data_{}".format(i))
                assert last_filename == "data_{}.tfrecords".format(i//examples_per_file)
                assert used_writer == current_writer
                assert last_written_data == "data_{}".format(i)
                if exception and i == n_writes - 1:
                    raise RuntimeError()
    except RuntimeError:
        assert close == True
    finally:
        assert close == True
        assert used_writer == current_writer