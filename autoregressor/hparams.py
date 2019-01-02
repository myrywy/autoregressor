import tensorflow as tf 
from tensorflow.contrib.training import HParams

hparams = HParams(
    learning_rate=0.1,
    rnn_num_units=100,
    rnn_num_layers=6,
    rnn_last_layer_num_units=50, # None to set the same as in previous layers
    max_training_steps=25,
    rnn_layer="lstm_block_cell"
)