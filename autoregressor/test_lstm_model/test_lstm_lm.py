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

from lstm_lm import get_language_model_fn, get_autoregressor_model_fn, language_model_input_dataset
from config import TEST_TMP_DIR

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
            assert r_features["inputs"] == approx(np.array(expected_feature["inputs"]))
            assert r_features["length"] == approx(np.array(expected_feature["length"]))
            assert r_labels["targets"] == approx(np.array(expected_label["targets"]))

# Auxiliary functions TODO: maybe move some of them to  utils, they can be useful outsite test cases
def fix_dimensions(features, labels, batch_size):
    features["inputs"] = tf.to_float(features["inputs"])
    features["inputs"].set_shape((batch_size, None, 3))
    features["length"].set_shape((batch_size,))
    labels["targets"].set_shape((batch_size, None))
    return features, labels
    
def force_float_embeddings(features, labels):
    features["inputs"] = tf.to_float(features["inputs"])
    return features, labels

def remove_labels(features, labels):
    return features

def predict_input(embedding_lookup_fn):
    dataset = input_data_fn(input_data_minimal(), embedding_lookup_fn, 3)
    dataset = dataset.map(lambda f, l: fix_dimensions(f, l, 3))
    return dataset.map(remove_labels)

def softmax_to_winner(sequences):
    outputs = []
    for sequence in sequences:
        sequence_output = np.argmax(sequence, axis=1)
        outputs.append(sequence_output) 
    return outputs

def test_get_language_model_fn(input_data, embedding_lookup_fn):
    try:
        shutil.rmtree(TEST_TMP_DIR)
    except FileNotFoundError:
        pass

    def get_input():
        dataset = input_data()
        return dataset.map(lambda f, l: fix_dimensions(f, l, 20))


    estimator = tf.estimator.Estimator(get_language_model_fn(8), params={"learning_rate": 0.0005}, model_dir=TEST_TMP_DIR)
    for _ in range(10):
        estimator.train(lambda: get_input(), steps=100)
        eval_result = estimator.evaluate(get_input, steps=3)

    predictions = estimator.predict(lambda: predict_input(embedding_lookup_fn))
    predictions = [*itertools.islice(predictions, 3)]
    winners = softmax_to_winner(predictions)
    expected = [
        [6,7,2,0],
        [6,5,2,0],
        [6,7,5,2]
    ]
    result = np.stack(winners)[:, 1:]
    result[0:2, -1] = 0
    assert np.array(expected) == approx(result)

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

    assert batch_1[1]["targets"] == approx(np.array(expected_1_targets))
    assert batch_2[1]["targets"] == approx(np.array(expected_2_targets))

    assert batch_1[0]["inputs"] == approx(np.array(expected_1_inputs))
    assert batch_2[0]["inputs"] == approx(np.array(expected_2_inputs))
    assert batch_1[0]["length"] == approx(np.array(expected_1_length))
    assert batch_2[0]["length"] == approx(np.array(expected_2_length))


def get_test_input_fn(make_example_fn, batch_size, inputs_length):
    def test_input():
        def fix_dimensions(features):
            features["inputs"].set_shape((batch_size,inputs_length))
            features["length"].set_shape((batch_size,))
            return features
        dataset = tf.data.Dataset.from_generator(make_example_fn, {"inputs": tf.int32, "length": tf.int32})
        dataset = dataset.batch(batch_size)
        return dataset.map(fix_dimensions)
    return test_input

def make_test_example_2_first_elements_known():
    yield {"inputs": np.array([1,5]), "length": np.array(4)}
    yield {"inputs": np.array([1,7]), "length": np.array(4)}
    yield {"inputs": np.array([1,6]), "length": np.array(4)}

def make_test_example_first_element_known_trimmed():
    yield {"inputs": np.array([1]), "length": np.array(4)}
def make_test_example_first_element_known_4():
    yield {"inputs": np.array([1]), "length": np.array(4)}
def make_test_example_first_element_known_5():
    yield {"inputs": np.array([1]), "length": np.array(5)}

@pytest.mark.parametrize(
    "make_example_fn, "
    "batch_size, "
    "inputs_length, "
    "prediction_steps, "
    "alternative_paths, "
    "expected_paths, "
    "predictions_mask",
    [
        (make_test_example_2_first_elements_known, 1, 2, 3, 1,
            [
                [
                    [1,5,6,7,2,0],
                ], [
                    [1,7,6,5,2,0],
                ], [
                    [1,6,6,7,5,2]
                ]
            ],
            [
                [
                    [1,1,1,1,1,0],
                ], [
                    [1,1,1,1,1,0],
                ], [
                    [1,1,1,1,1,1]
                ]
            ]
        ), 
        (make_test_example_2_first_elements_known, 3, 2, 3, 1,
            [
                [
                    [1,5,6,7,2,0],
                ], [
                    [1,7,6,5,2,0],
                ], [
                    [1,6,6,7,5,2]
                ]
            ],
            [
                [
                    [1,1,1,1,1,0],
                ], [
                    [1,1,1,1,1,0],
                ], [
                    [1,1,1,1,1,1]
                ]
            ]
        ), 
    ]
)
def test_get_autoregressor_model_fn(
    input_data, 
    embedding_lookup_fn,
    make_example_fn, 
    batch_size, 
    inputs_length,
    prediction_steps,
    alternative_paths,
    expected_paths,
    predictions_mask):
    try:
        shutil.rmtree(TEST_TMP_DIR)
    except FileNotFoundError:
        pass
    def get_input():
        dataset = input_data()
        dataset = dataset.map(force_float_embeddings)
        return dataset.map(lambda f, l: fix_dimensions(f, l, 20))

    params = {"learning_rate": 0.0005, "number_of_alternatives": alternative_paths}

    model_fn = get_autoregressor_model_fn(8, id_to_embedding_mapping=lambda x: tf.to_float(embedding_lookup_fn(x)))

    estimator = tf.estimator.Estimator(model_fn, params=params, model_dir=TEST_TMP_DIR)
    for _ in range(3):
        estimator.train(lambda: get_input(), steps=100)
        eval_result = estimator.evaluate(get_input, steps=3)
        print("Eval results")
        print(eval_result)


    predictions = estimator.predict(get_test_input_fn(make_example_fn, batch_size, inputs_length))
    predictions = [*itertools.islice(predictions, prediction_steps)]
    predictions_stacked = np.stack([o["paths"] for o in predictions])
    
    predictions_stacked *= predictions_mask # because this elements are after the real last element so we don't care (there was no such cases in examples in dataset )
    
    assert predictions_stacked == approx(np.array(expected_paths))



@pytest.mark.parametrize(
    "make_example_fn, "
    "batch_size, "
    "inputs_length, "
    "prediction_steps, "
    "alternative_paths, "
    "expected_paths, "
    "predictions_mask, "
    "allowables",
    [
        (make_test_example_first_element_known_4, 1, 1, 1, 1,
            [
                [
                    [1,5,6,7,2],
                ]
            ],
            [
                [
                    [1,1,1,1,1],
                ]
            ],
            [[5],[],[],[]]
        ), 
        (make_test_example_first_element_known_4, 1, 1, 1, 1,
            [
                [
                    [1,7,6,5,2],
                ]
            ],
            [
                [
                    [1,1,1,1,1],
                ]
            ],
            [[7],[],[],[]]
        ), 
        (make_test_example_first_element_known_5, 1, 1, 1, 1,
            [
                [
                    [1,6,6,7,5,2]
                ]
            ],
            [
                [
                    [1,1,1,1,1,1]
                ]
            ],
            [[6],[],[],[],[]]
        ), 
        (make_test_example_first_element_known_5, 1, 1, 1, 1,
            [
                [
                    [1,6,6,5,0,0]
                ]
            ],
            [
                [
                    [1,1,1,1,0,0]
                ]
            ],
            [[6],[],[5],[],[]]
        ), 
        (make_test_example_first_element_known_5, 1, 1, 1, 1,
            [
                [
                    [1,6,6,6,0,0]
                ]
            ],
            [
                [
                    [1,1,1,1,0,0]
                ]
            ],
            [[6],[],[6],[],[]]
        ), 
        (make_test_example_first_element_known_5, 1, 1, 1, 1,
            [
                [
                    [1,6,6,7,0,0]
                ]
            ],
            [
                [
                    [1,1,1,1,0,0]
                ]
            ],
            [[6],[],[7],[],[]]
        ), 
        (make_test_example_first_element_known_5, 1, 1, 1, 1,
            [
                [
                    [1,6,5,0,0,0]
                ]
            ],
            [
                [
                    [1,1,1,0,0,0]
                ]
            ],
            [[6],[5],[],[],[]]
        ), 
        (make_test_example_first_element_known_5, 1, 1, 1, 1,
            [
                [
                    [1,6,6,0,0,0]
                ]
            ],
            [
                [
                    [1,1,1,0,0,0]
                ]
            ],
            [[6],[6],[],[],[]]
        ), 
        (make_test_example_first_element_known_5, 1, 1, 1, 1,
            [
                [
                    [1,6,7,0,0,0]
                ]
            ],
            [
                [
                    [1,1,1,0,0,0]
                ]
            ],
            [[6],[7],[],[],[]]
        ), 
    ]
)
def test_with_mask_get_autoregressor_model_fn(
    input_data, 
    embedding_lookup_fn,
    make_example_fn, 
    batch_size, 
    inputs_length,
    prediction_steps,
    alternative_paths,
    expected_paths,
    predictions_mask,
    allowables):
    try:
        shutil.rmtree(TEST_TMP_DIR)
    except FileNotFoundError:
        pass
    def get_input():
        dataset = input_data()
        dataset = dataset.map(force_float_embeddings)
        return dataset.map(lambda f, l: fix_dimensions(f, l, 20))

    params = {"learning_rate": 0.0005, "number_of_alternatives": alternative_paths}

    DISTRIBUTION_SIZE = 8
    model_fn = get_autoregressor_model_fn(
        DISTRIBUTION_SIZE, 
        id_to_embedding_mapping=lambda x: tf.to_float(embedding_lookup_fn(x)), 
        mask_allowables=allowables)

    estimator = tf.estimator.Estimator(model_fn, params=params, model_dir=TEST_TMP_DIR)
    for _ in range(3):
        estimator.train(lambda: get_input(), steps=100)
        eval_result = estimator.evaluate(get_input, steps=3)
        print("Eval results")
        print(eval_result)


    predictions = estimator.predict(get_test_input_fn(make_example_fn, batch_size, inputs_length))
    predictions = [*itertools.islice(predictions, prediction_steps)]
    predictions_stacked = np.stack([o["paths"] for o in predictions])
    
    predictions_stacked *= predictions_mask # because this elements are after the real last element so we don't care (there was no such cases in examples in dataset )
    
    assert predictions_stacked == approx(np.array(expected_paths))


@pytest.mark.parametrize(
    "make_example_fn, "
    "batch_size, "
    "inputs_length, "
    "prediction_steps, "
    "alternative_paths, "
    "expected_paths, ",
    [
        (make_test_example_first_element_known_trimmed, 1, 1, 3, 3,
            [
                    np.array([1,5,6,7,2,]),
                    np.array([1,7,6,5,2,]),
                    np.array([1,6,6,7,5,])
            ],
        ), 
    ]
)
def test_generating_only_possible_paths_autoregressor_model_fn(
    input_data, 
    embedding_lookup_fn,
    make_example_fn, 
    batch_size, 
    inputs_length,
    prediction_steps,
    alternative_paths,
    expected_paths):
    """In this setup there are only 3 sentences in training set.
    LSTM language model is trained on the set.
    This test checks if autoregressor with this LM produces (given only first <start> element,
    allowed to generate 3 most probable paths) exactly the same sentences as in training data.
    We generate such a number of elements not to exceed length of shortest sentence."""
    try:
        shutil.rmtree(TEST_TMP_DIR)
    except FileNotFoundError:
        pass
    def get_input():
        dataset = input_data()
        dataset = dataset.map(force_float_embeddings)
        return dataset.map(lambda f, l: fix_dimensions(f, l, 20))

    params = {"learning_rate": 0.0005, "number_of_alternatives": alternative_paths}

    model_fn = get_autoregressor_model_fn(8, id_to_embedding_mapping=lambda x: tf.to_float(embedding_lookup_fn(x)))

    estimator = tf.estimator.Estimator(model_fn, params=params, model_dir=TEST_TMP_DIR)
    for _ in range(10):
        estimator.train(lambda: get_input(), steps=100)
        eval_result = estimator.evaluate(get_input, steps=3)
        print("Eval results")
        print(eval_result)


    predictions = estimator.predict(get_test_input_fn(make_example_fn, batch_size, inputs_length))
    predictions = [*itertools.islice(predictions, prediction_steps)]
    predictions_stacked = np.stack([o["paths"] for o in predictions])
    
    
    for predicted_path in predictions_stacked[0]:
        assert any(all(expected_path == predicted_path) for expected_path in expected_paths)

