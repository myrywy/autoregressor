from data_pipeline import DataPipeline

import tensorflow as tf
import numpy as np

import pytest
from pytest import approx


def test_add_structural_transformation():
    def input_data_generator():
        yield np.array([1,2,3,4])
        yield np.array([1,2,3])
        yield np.array([1,2])
        yield np.array([1])
        yield np.array([1,2])
        yield np.array([1,2,3])
        yield np.array([1,2,3,4])

    def expected_output_data_generator():
        yield {"input_sequnce": np.array([1,2,3,4]), "length": 4}
        yield {"input_sequnce": np.array([1,2,3]), "length": 3}
        yield {"input_sequnce": np.array([1,2]), "length": 2}
        yield {"input_sequnce": np.array([1]), "length": 1}
        yield {"input_sequnce": np.array([1,2]), "length": 2}
        yield {"input_sequnce": np.array([1,2,3]), "length": 3}
        yield {"input_sequnce": np.array([1,2,3,4]), "length": 4}

    input_dataset = tf.data.Dataset.from_generator(input_data_generator, output_types=tf.int32)
    expected_output_dataset = tf.data.Dataset.from_generator(expected_output_data_generator, output_types={"input_sequnce": tf.int32, "length": tf.int32})

    def add_length(input_sequnce):
        return {
            "input_sequnce": input_sequnce,
            "length": tf.shape(input_sequnce)[0]
            }
    pipeline = DataPipeline()
    pipeline.add_structural_transformation(add_length)
    output_dataset = pipeline.transform_dataset(input_dataset)

    output_next = output_dataset.make_one_shot_iterator().get_next()
    expected_next = expected_output_dataset.make_one_shot_iterator().get_next()

    with tf.Session() as sess:
        for _ in range(7):
            r_output, r_expected = sess.run((output_next, expected_next))
            assert r_output["input_sequnce"] == approx(r_expected["input_sequnce"])
            assert r_output["length"] == approx(r_expected["length"])


def test_add_unit_transformation():
    def input_data_generator():
        yield {"input_sequence": np.array([1,2,3,4]), "length": 4}
        yield {"input_sequence": np.array([1,2,3]), "length": 3}
        yield {"input_sequence": np.array([1,2]), "length": 2}
        yield {"input_sequence": np.array([1]), "length": 1}
        yield {"input_sequence": np.array([1,2]), "length": 2}
        yield {"input_sequence": np.array([1,2,3]), "length": 3}
        yield {"input_sequence": np.array([1,2,3,4]), "length": 4}

    def expected_output_data_generator():
        yield {"input_sequence": np.array([4,5,6,7]), "length": 4}
        yield {"input_sequence": np.array([4,5,6]), "length": 3}
        yield {"input_sequence": np.array([4,5]), "length": 2}
        yield {"input_sequence": np.array([4]), "length": 1}
        yield {"input_sequence": np.array([4,5]), "length": 2}
        yield {"input_sequence": np.array([4,5,6]), "length": 3}
        yield {"input_sequence": np.array([4,5,6,7]), "length": 4}

    input_dataset = tf.data.Dataset.from_generator(input_data_generator, output_types={"input_sequence": tf.int32, "length": tf.int32})
    expected_output_dataset = tf.data.Dataset.from_generator(expected_output_data_generator, output_types={"input_sequence": tf.int32, "length": tf.int32})

    def add3(x):
        return x+3
    pipeline = DataPipeline()
    pipeline.add_unit_transformation(add3, "input_sequence")
    output_dataset = pipeline.transform_dataset(input_dataset)

    output_next = output_dataset.make_one_shot_iterator().get_next()
    expected_next = expected_output_dataset.make_one_shot_iterator().get_next()

    with tf.Session() as sess:
        for _ in range(7):
            r_output, r_expected = sess.run((output_next, expected_next))
            assert r_output["input_sequence"] == approx(r_expected["input_sequence"])
            assert r_output["length"] == approx(r_expected["length"])


def test_add_unit_transformation_nested():
    def input_data_generator():
        yield {"input_sequence": np.array([1,2,3,4]), "length": 4}, 9
        yield {"input_sequence": np.array([1,2,3]), "length": 3}, 9
        yield {"input_sequence": np.array([1,2]), "length": 2}, 9
        yield {"input_sequence": np.array([1]), "length": 1}, 9
        yield {"input_sequence": np.array([1,2]), "length": 2}, 9
        yield {"input_sequence": np.array([1,2,3]), "length": 3}, 9
        yield {"input_sequence": np.array([1,2,3,4]), "length": 4}, 9

    def expected_output_data_generator():
        yield {"input_sequence": np.array([4,5,6,7]), "length": 4}, 9
        yield {"input_sequence": np.array([4,5,6]), "length": 3}, 9
        yield {"input_sequence": np.array([4,5]), "length": 2}, 9
        yield {"input_sequence": np.array([4]), "length": 1}, 9
        yield {"input_sequence": np.array([4,5]), "length": 2}, 9
        yield {"input_sequence": np.array([4,5,6]), "length": 3}, 9
        yield {"input_sequence": np.array([4,5,6,7]), "length": 4}, 9

    input_dataset = tf.data.Dataset.from_generator(input_data_generator, output_types=({"input_sequence": tf.int32, "length": tf.int32}, tf.int32))
    expected_output_dataset = tf.data.Dataset.from_generator(expected_output_data_generator, output_types=({"input_sequence": tf.int32, "length": tf.int32},tf.int32))

    def add3(x):
        return x+3
    pipeline = DataPipeline()
    pipeline.add_unit_transformation(add3, 0, "input_sequence")
    output_dataset = pipeline.transform_dataset(input_dataset)

    output_next = output_dataset.make_one_shot_iterator().get_next()
    expected_next = expected_output_dataset.make_one_shot_iterator().get_next()

    with tf.Session() as sess:
        for _ in range(7):
            r_output, r_expected = sess.run((output_next, expected_next))
            assert r_output[0]["input_sequence"] == approx(r_expected[0]["input_sequence"])
            assert r_output[0]["length"] == approx(r_expected[0]["length"])
            assert r_output[1] == approx(r_expected[1])


def test_add_unit_transformation_simple():
    def input_data_generator():
        yield np.array([1,2,3,4])
        yield np.array([1,2,3])
        yield np.array([1,2])
        yield np.array([1])
        yield np.array([1,2])
        yield np.array([1,2,3])
        yield np.array([1,2,3,4])

    def expected_output_data_generator():
        yield np.array([4,5,6,7])
        yield np.array([4,5,6])
        yield np.array([4,5])
        yield np.array([4])
        yield np.array([4,5])
        yield np.array([4,5,6])
        yield np.array([4,5,6,7])

    input_dataset = tf.data.Dataset.from_generator(input_data_generator, output_types=tf.int32)
    expected_output_dataset = tf.data.Dataset.from_generator(expected_output_data_generator, output_types=tf.int32)

    def add3(x):
        return x+3
    pipeline = DataPipeline()
    pipeline.add_unit_transformation(add3)
    output_dataset = pipeline.transform_dataset(input_dataset)

    output_next = output_dataset.make_one_shot_iterator().get_next()
    expected_next = expected_output_dataset.make_one_shot_iterator().get_next()

    with tf.Session() as sess:
        for _ in range(7):
            r_output, r_expected = sess.run((output_next, expected_next))
            assert r_output == approx(r_expected)


def test_add_unit_transformation_simple_tensor_slices():
    input_data = np.array(
        [
            [1,2,3,4],
            [1,2,3,0],
            [1,2,0,0],
            [1,0,0,0],
        ]
    )
    expected_output_data = np.array(
        [
            [4, 5, 6, 7],
            [4, 5, 6, 3],
            [4, 5, 3, 3],
            [4, 3, 3, 3],
        ]
    )

    input_dataset = tf.data.Dataset.from_tensor_slices(input_data)
    expected_output_dataset = tf.data.Dataset.from_tensor_slices(expected_output_data)

    def add3(x):
        return x+3
    pipeline = DataPipeline()
    pipeline.add_unit_transformation(add3)
    output_dataset = pipeline.transform_dataset(input_dataset)

    output_next = output_dataset.make_one_shot_iterator().get_next()
    expected_next = expected_output_dataset.make_one_shot_iterator().get_next()

    with tf.Session() as sess:
        for _ in range(4):
            r_output, r_expected = sess.run((output_next, expected_next))
            assert r_output == approx(r_expected)


def test_add_unit_transformation_tuple_tensor_slices():
    input_data = np.array(
        [
            [1,2,3,4],
            [1,2,3,0],
            [1,2,0,0],
            [1,0,0,0],
        ]
    )
    expected_output_data = np.array(
        [
            [4, 5, 6, 7],
            [4, 5, 6, 3],
            [4, 5, 3, 3],
            [4, 3, 3, 3],
        ]
    )

    input_dataset = tf.data.Dataset.from_tensor_slices((input_data, input_data[:,0]))
    expected_output_dataset = tf.data.Dataset.from_tensor_slices((expected_output_data, input_data[:,0]))

    def add3(x):
        return x+3
    pipeline = DataPipeline()
    pipeline.add_unit_transformation(add3, 0)
    output_dataset = pipeline.transform_dataset(input_dataset)

    output_next = output_dataset.make_one_shot_iterator().get_next()
    expected_next = expected_output_dataset.make_one_shot_iterator().get_next()

    with tf.Session() as sess:
        for _ in range(4):
            r_output, r_expected = sess.run((output_next, expected_next))
            assert r_output[0] == approx(r_expected[0])
            assert r_output[1] == approx(r_expected[1])


def test_add_unit_transformation_one_element_tuple_tensor_slices():
    input_data = np.array(
        [
            [1,2,3,4],
            [1,2,3,0],
            [1,2,0,0],
            [1,0,0,0],
        ]
    )
    expected_output_data = np.array(
        [
            [4, 5, 6, 7],
            [4, 5, 6, 3],
            [4, 5, 3, 3],
            [4, 3, 3, 3],
        ]
    )

    input_dataset = tf.data.Dataset.from_tensor_slices((input_data,))
    expected_output_dataset = tf.data.Dataset.from_tensor_slices(expected_output_data)

    def add3(x):
        return x+3
    pipeline = DataPipeline()
    pipeline.add_unit_transformation(add3)
    output_dataset = pipeline.transform_dataset(input_dataset)

    output_next = output_dataset.make_one_shot_iterator().get_next()
    expected_next = expected_output_dataset.make_one_shot_iterator().get_next()

    with tf.Session() as sess:
        for _ in range(4):
            r_output, r_expected = sess.run((output_next, expected_next))
            assert r_output == approx(r_expected)


def test_add_unit_transformation_one_element_tuple():
    def input_data_generator():
        yield np.array([1,2,3,4]),
        yield np.array([1,2,3]),
        yield np.array([1,2]),
        yield np.array([1]),
        yield np.array([1,2]),
        yield np.array([1,2,3]),
        yield np.array([1,2,3,4]),

    def expected_output_data_generator():
        yield np.array([4,5,6,7])
        yield np.array([4,5,6])
        yield np.array([4,5])
        yield np.array([4])
        yield np.array([4,5])
        yield np.array([4,5,6])
        yield np.array([4,5,6,7])

    input_dataset = tf.data.Dataset.from_generator(input_data_generator, output_types=(tf.int32,))
    expected_output_dataset = tf.data.Dataset.from_generator(expected_output_data_generator, output_types=tf.int32)

    def add3(x):
        return x+3
    pipeline = DataPipeline()
    pipeline.add_unit_transformation(add3)
    output_dataset = pipeline.transform_dataset(input_dataset)

    output_next = output_dataset.make_one_shot_iterator().get_next()
    expected_next = expected_output_dataset.make_one_shot_iterator().get_next()

    with tf.Session() as sess:
        for _ in range(7):
            r_output, r_expected = sess.run((output_next, expected_next))
            assert r_output == approx(r_expected)