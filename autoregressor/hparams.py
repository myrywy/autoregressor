import tensorflow as tf 
from tensorflow.contrib.training import HParams

hparams = HParams(
    learning_rate=0.1,
    rnn_num_units=100,
    rnn_num_layers=6,
    rnn_last_layer_num_units=50, # None to set the same as in previous layers
    max_training_steps=25,
    rnn_layer="lstm_block_cell",
    profiler=False,
    size_based_device_assignment=False,
    device="CPU",   # "CPU", "GPU" or ""
    batch_size=5,
    shuffle_examples_buffer_size=1000,
    shuffle_examples_seed=0,
    cached_data_dir=None,
    write_target_text_to_summary=False,
    mask_padding_cost=True,
    dynamic_rnn_swap_memory=True,
    predict_top_k=2,
    words_as_text_preview=True,
    time_major_optimization=True,

    # these are only effective when training via lm_training_process
    vocabulary_name="glove300",
    corpus_name="simple_examples",
)