import itertools
import tensorflow as tf
import numpy as np
import pytest

TEST_CHECKPOINTS_DIR = "./test_checkpoints"

def constant_model_fn(features, labels, mode, params):
    #a = tf.get_variable("a", shape=(3,), dtype=tf.float32)
    a_layer = MockLayer()
    batched_a_plus = a_layer(tf.ones((1,3), dtype=tf.float32))
    batched_a = a_layer(tf.zeros((1,3), dtype=tf.float32))
    #a.initializer = tf.constant_initializer(np.array([0.0, 0.5, 1.0]))
    if mode == tf.estimator.ModeKeys.PREDICT:
        #output = (batched_a_plus, features)
        spec = tf.estimator.EstimatorSpec(mode=mode, predictions={"batched_a_plus":batched_a_plus, "batched_a": batched_a})
    elif mode == tf.estimator.ModeKeys.TRAIN:
        loss = tf.reduce_sum(tf.reduce_sum(tf.abs(batched_a - tf.constant([[1.0, 1.5, 0.0]]))))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        train_op = optimizer.minimize(
            loss=loss, global_step=tf.train.get_global_step())
        spec = tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,)
    return spec

class MockLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(MockLayer, self).__init__()

    def build(self, batch_size):
        self.a = self.add_variable("a", (1,3,), tf.float32, trainable=True, initializer=tf.constant_initializer([-1,-1,-1]))
    
    def call(self, input):
        return self.a + input

    def output_shape(self):
        return (1,)


def test_constant_model_fn():
    OUTPUT_EXAMPLES = 5
    def input_fn():
        dataset = tf.data.Dataset.range(5)
        dataset = dataset.repeat(1)
        dataset.take(5)
        return dataset
    def input_generator():
        yield input_fn()
    model = tf.estimator.Estimator(
        model_fn=constant_model_fn, 
        model_dir=TEST_CHECKPOINTS_DIR)
    model.train(input_fn, steps=5000)
    output = list(itertools.islice(model.predict(input_generator), OUTPUT_EXAMPLES))
    a_plus = [o["batched_a_plus"] for o in output]
    a = [o["batched_a"] for o in output]
    a_plus = np.stack(a_plus)
    a = np.stack(a)
    assert a == pytest.approx(np.array([[1.0, 1.5, 0.0]]*OUTPUT_EXAMPLES), abs=0.01)
    assert a_plus == pytest.approx(np.array([[2.0, 2.5, 1.0]]*OUTPUT_EXAMPLES), abs=0.01)
