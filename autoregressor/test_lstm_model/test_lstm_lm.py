"""
VOCABULARY:
id  |   vector
1   |   [1,0,0] (start mark)       
2   |   [0,0,1] (stop mark)
5   |   [1,2,3]
6   |   [1,-2,3]
7   |   [-1,2,-3]
"""
import shutil
import tensorflow as tf
import numpy as np
import itertools
import pytest
from pytest import approx

from lstm_lm import get_language_model_fn, language_model_input_dataset

SEED = 0 # this is used with numpy random seed in each test
tf.set_random_seed(1) # this must be set before graph is made, otherwise it random sequence has already some random values and fixing seed dosn't ensure reproducability

@pytest.fixture
def embedding_lookup_fn():
    return lambda t: tf.nn.embedding_lookup(
        [
            [[0,0,0]], # .
            [[1,0,0]], # 1 (start mark)       
            [[0,0,1]], # 2 (stop mark)
            [[0,0,0]], # .
            [[0,0,0]], # .
            [[1,2,3]], # 5 
            [[1,-2,3]],    # 6
            [[-1,2,-3]],   # 7
        ],
        t
    )


def input_data_minimal():
    examples_raw = [
        [5,6,7],
        [7,6,5],
        [6, 6, 7, 5]
    ]
    examples = [np.array([1]+ex+[2]) for ex in examples_raw]

    def examples_generator():
        for example in examples:
            yield example

    input_data = tf.data.Dataset.from_generator(examples_generator, tf.int32)
    return input_data


def input_data_raw():
    return input_data_minimal().repeat(100000000000000).shuffle(60, seed=SEED)


@pytest.fixture
def input_data(embedding_lookup_fn):
    return lambda: input_data_fn(input_data_raw(), embedding_lookup_fn)

def input_data_fn(dataset, embedding_lookup_fn, batch_size=20):
    def expand_length(features, labels):
        #features, labels = example
        features["length"] = tf.expand_dims(features["length"], 0)
        return (features, labels)
    def flatten_length(features, labels):
        features["length"] = tf.squeeze(features["length"], axis=[1])
        return (features, labels)
    prepared_lm_data = language_model_input_dataset(dataset, embedding_lookup_fn)
    length_expanded_data = prepared_lm_data.map(expand_length)
    length_expanded_data = length_expanded_data.\
        padded_batch(
            batch_size, 
            padded_shapes= (
                    {"inputs": tf.TensorShape((tf.Dimension(None), tf.Dimension(None))), "length": tf.TensorShape((tf.Dimension(1),))},
                    {"targets": tf.TensorShape((tf.Dimension(None),))}
                )
            )
    return length_expanded_data.map(flatten_length)

def test_language_model_input_dataset(embedding_lookup_fn):
    dataset = language_model_input_dataset(input_data_minimal(), embedding_lookup_fn)
    iterator = dataset.make_one_shot_iterator()
    next_example = iterator.get_next()
    expected_features = [
        {"inputs": [[1,0,0], [1,2,3], [1,-2,3],[-1,2,-3]], "length": 4},
        {"inputs": [[1,0,0], [-1,2,-3], [1,-2,3],[1,2,3]], "length": 4},
        {"inputs": [[1,0,0], [1,-2,3], [1,-2,3], [-1,2,-3], [1,2,3]], "length": 5},
    ]
    expected_labels = [
        {"targets": [5,6,7,2], },
        {"targets": [7,6,5,2], },
        {"targets": [6,6,7,5,2], },
    ]
    expected = zip(expected_features, expected_labels)
    
    with tf.Session() as sess:
        for expected_feature, expected_label in expected:
            r_features, r_labels = sess.run(next_example)
            assert r_features["inputs"] == approx(expected_feature["inputs"])
            assert r_features["length"] == approx(expected_feature["length"])
            assert r_labels["targets"] == approx(expected_label["targets"])

def test_get_language_model_fn(input_data, embedding_lookup_fn):
    try:
        shutil.rmtree("./lstm_test_models")
    except FileNotFoundError:
        pass
    def fix_dimensions(features, labels, batch_size):
        features["inputs"] = tf.to_float(features["inputs"])
        features["inputs"].set_shape((batch_size, None, 3))
        features["length"].set_shape((batch_size,))
        labels["targets"].set_shape((batch_size, None))
        return features, labels
    def get_input():
        dataset = input_data()
        return dataset.map(lambda f, l: fix_dimensions(f, l, 20))

    def remove_labels(features, labels):
        return features
    def predict_input():
        #dataset = language_model_input_dataset(input_data_minimal(), embedding_lookup_fn).\
        #    padded_batch(3, 
        #    padded_shapes= (
        #            {"inputs": tf.TensorShape((tf.Dimension(None), tf.Dimension(None))), "length": tf.TensorShape((tf.Dimension(1),))},
        #            {"targets": tf.TensorShape((tf.Dimension(None),))}
        #        )
        #    )
        dataset = input_data_fn(input_data_minimal(), embedding_lookup_fn, 3)
        dataset = dataset.map(lambda f, l: fix_dimensions(f, l, 3))
        return dataset.map(remove_labels)

    def softmax_to_winner(sequences):
        outputs = []
        for sequence in sequences:
            sequence_output = np.argmax(sequence, axis=1)
            outputs.append(sequence_output) 
        return outputs

    #estimator = tf.estimator.Estimator(get_language_model_fn(8), params={"learning_rate": 0.04}, model_dir="./lstm_test_models")
    #estimator.train(lambda: get_input(), steps=200)
    estimator = tf.estimator.Estimator(get_language_model_fn(8), params={"learning_rate": 0.0005}, model_dir="./lstm_test_models")
    for _ in range(10):
        estimator.train(lambda: get_input(), steps=100)
        eval_result = estimator.evaluate(get_input, steps=3)

    '''with tf.Session() as sess:
        d = predict_input()
        it = d.make_one_shot_iterator()
        n = it.get_next()
        r_n = sess.run(n)'''

    predictions = estimator.predict(predict_input)
    predictions = [*itertools.islice(predictions, 3)]
    winners = softmax_to_winner(predictions)
    expected = [
        [6,7,2,0],
        [6,5,2,0],
        [6,7,5,2]
    ]
    result = np.stack(winners)[:, 1:]
    result[0:2, -1] = 0
    assert expected == approx(result)

def test_input_data(embedding_lookup_fn):
    """This test is checking if fixture function produces what is expected. Kind of learning/sanity check."""
    dataset = input_data_minimal().repeat(10)
    dataset = input_data_fn(dataset, embedding_lookup_fn)
    iterator = dataset.make_one_shot_iterator()
    next = iterator.get_next()
    with tf.Session() as sess:
        batch_1 = sess.run(next)
        batch_2 = sess.run(next)

    def py_list_embeddings_mapping(inputs):
        embeddings = {
            0: [0,0,0], # .
            1: [1,0,0], # 1 (start mark)       
            2: [0,0,1], # 2 (stop mark)
            5: [1,2,3], # 5 
            6: [1,-2,3],    # 6
            7: [-1,2,-3],   # 7
        }
        return [[embeddings[id] for id in sequence]for sequence in inputs]

    expected_1_inputs = np.array(py_list_embeddings_mapping([
        [1,5,6,7,0],
        [1,7,6,5,0],
        [1,6, 6, 7, 5],
        [1,5,6,7,0],
        [1,7,6,5,0],
        [1,6, 6, 7, 5],
        [1,5,6,7,0],
        [1,7,6,5,0],
        [1,6, 6, 7, 5],
        [1,5,6,7,0],
        [1,7,6,5,0],
        [1,6, 6, 7, 5],
        [1,5,6,7,0],
        [1,7,6,5,0],
        [1,6, 6, 7, 5],
        [1,5,6,7,0],
        [1,7,6,5,0],
        [1,6, 6, 7, 5],
        [1,5,6,7,0],
        [1,7,6,5,0],
    ]))
    expected_2_inputs = np.array(py_list_embeddings_mapping([
        [1,6, 6, 7, 5],
        [1,5,6,7,0],
        [1,7,6,5,0],
        [1,6, 6, 7, 5],
        [1,5,6,7,0],
        [1,7,6,5,0],
        [1,6, 6, 7, 5],
        [1,5,6,7,0],
        [1,7,6,5,0],
        [1,6, 6, 7, 5],
    ]))
    expected_1_targets = np.array([
        [5,6,7,2,0],
        [7,6,5,2,0],
        [6, 6, 7, 5,2],
        [5,6,7,2,0],
        [7,6,5,2,0],
        [6, 6, 7, 5,2],
        [5,6,7,2,0],
        [7,6,5,2,0],
        [6, 6, 7, 5,2],
        [5,6,7,2,0],
        [7,6,5,2,0],
        [6, 6, 7, 5,2],
        [5,6,7,2,0],
        [7,6,5,2,0],
        [6, 6, 7, 5,2],
        [5,6,7,2,0],
        [7,6,5,2,0],
        [6, 6, 7, 5,2],
        [5,6,7,2,0],
        [7,6,5,2,0],
    ])
    expected_2_targets = np.array([
        [6, 6, 7, 5,2],
        [5,6,7,2,0],
        [7,6,5,2,0],
        [6, 6, 7, 5,2],
        [5,6,7,2,0],
        [7,6,5,2,0],
        [6, 6, 7, 5,2],
        [5,6,7,2,0],
        [7,6,5,2,0],
        [6, 6, 7, 5,2],
    ])
    expected_1_length = [4,4,5,4,4,5,4,4,5,4,4,5,4,4,5,4,4,5,4,4,]
    expected_2_length = [5,4,4,5,4,4,5,4,4,5,]
    # dataset consist of tuple of dicts of tensors interpreted as features (first dict, used as network input) and labels (second dict used as network's gold standard)

    assert batch_1[1]["targets"] == approx(expected_1_targets)
    assert batch_2[1]["targets"] == approx(expected_2_targets)

    assert batch_1[0]["inputs"] == approx(expected_1_inputs)
    assert batch_2[0]["inputs"] == approx(expected_2_inputs)
    assert batch_1[0]["length"] == approx(expected_1_length)
    assert batch_2[0]["length"] == approx(expected_2_length)