from functools import partial

import tensorflow as tf 
from tensorflow.contrib.training import HParams
import numpy as np

from unittest import mock
import pytest
from pytest import approx

from lm_training_process import LanguageModel
from generalized_vocabulary import GeneralizedVocabulary, SpecialUnit
from vocabularies_preprocessing.mock_vocabulary import MockVocab


@pytest.fixture
def hparams():
    return HParams(
        learning_rate=0.1,
        rnn_num_units=100,
        rnn_num_layers=6,
        rnn_last_layer_num_units=50, # None to set the same as in previous layers
        max_training_steps=25,
        rnn_layer="lstm_block_cell",
        profiler=False,
        size_based_device_assignment=False,
        batch_size=5,
        shuffle_examples_buffer_size=1000,
        shuffle_examples_seed=0,
        cached_data_dir=None,
        write_target_text_to_summary=False,
        mask_padding_cost=True,
        dynamic_rnn_swap_memory=True,
        predict_top_k=5,
        words_as_text_preview=True,
        time_major_optimization=True,
    )


@pytest.mark.parametrize("time_major_optimization, inputs, targets, length, expected_inputs, expected_targets",
    [
        (
            True,
            # input
            np.array([
                [
                    [0.1, 0.2, 0.3],
                    [1.1, 1.2, 1.3],
                    [2.1, 2.2, 2.3],
                ],
                [
                    [3.1, 3.2, 3.3],
                    [4.1, 4.2, 4.3],
                    [5.1, 5.2, 5.3],
                ],
            ]),
            np.array([
                [0,1,2],
                [3,4,5],
            ]),
            np.array([3, 3]),
            # expected output
            np.array([
                [
                    [0.1, 0.2, 0.3],
                    [3.1, 3.2, 3.3],
                ],
                [
                    [1.1, 1.2, 1.3],
                    [4.1, 4.2, 4.3],
                ],
                [
                    [2.1, 2.2, 2.3],
                    [5.1, 5.2, 5.3],
                ],
            ]),
            np.array([
                [0,3,],
                [1,4,],
                [2,5,],
            ]),
        ),
        (
            False,
            # input
            np.array([
                [
                    [0.1, 0.2, 0.3],
                    [1.1, 1.2, 1.3],
                    [2.1, 2.2, 2.3],
                ],
                [
                    [3.1, 3.2, 3.3],
                    [4.1, 4.2, 4.3],
                    [5.1, 5.2, 5.3],
                ],
            ]),
            np.array([
                [0,1,2],
                [3,4,5],
            ]),
            np.array([3, 3]),
            # expected output
            np.array([
                [
                    [0.1, 0.2, 0.3],
                    [1.1, 1.2, 1.3],
                    [2.1, 2.2, 2.3],
                ],
                [
                    [3.1, 3.2, 3.3],
                    [4.1, 4.2, 4.3],
                    [5.1, 5.2, 5.3],
                ],
            ]),
            np.array([
                [0,1,2],
                [3,4,5],
            ]),
        ),
    ]
)
def test_maybe_transpose_batch_time(hparams, time_major_optimization, inputs, targets, length, expected_inputs, expected_targets):
    hparams.set_hparam("time_major_optimization", time_major_optimization)
    lm = LanguageModel({"inputs": inputs, "length": length}, {"targets": targets}, tf.estimator.ModeKeys.TRAIN, None, hparams)
    inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
    targets = tf.convert_to_tensor(targets, dtype=tf.int32)
    t_inputs, t_targets = lm.maybe_transpose_batch_time(inputs, targets)
    with tf.Session() as sess:
        r_inputs, r_targets = sess.run((t_inputs, t_targets))
    
    assert r_inputs == approx(expected_inputs)
    assert (r_targets == expected_targets).all()


@pytest.mark.parametrize("targets, logits, length, expected_cross_entropy",
    [
        (
            np.array([
                [0,1],
                [2,3],
            ]),
            
            np.array([
                [
                    #[0,1,2,3]
                    [ 1, 0, 0, 0],
                    [ 0, 1, 0, 0],
                ],
                [
                    [ 0, 0, 1, 0],
                    [ 0, 0, 0, 1],
                ],
            ]),
            np.array([2,2]),
            np.array([
                [1,1],
                [1,1],
            ]) * (-1) * np.log(np.exp(1)/(np.exp(1)+np.exp(0)+np.exp(0)+np.exp(0)))
        ),
        (
            np.array([
                [0,1],
                [2,3],
            ]),
            
            np.array([
                [
                    #[0,1,2,3]
                    [ 0.5, 0.5, 0, 0],
                    [ 0, 0.6, 0.4, 0],
                ],
                [
                    [ 0, 0, 1, 0],
                    [ 0.7, 0, 0, 0.3],
                ],
            ]),
            np.array([2,2]),
            np.array([
                [
                    (-1) * np.log(np.exp(0.5)/(np.exp(0.5)+np.exp(0.5)+np.exp(0)+np.exp(0))),
                    (-1) * np.log(np.exp(0.6)/(np.exp(0)+np.exp(0.6)+np.exp(0.4)+np.exp(0)))
                ],
                [
                    (-1) * np.log(np.exp(1)/(np.exp(1)+np.exp(0)+np.exp(0)+np.exp(0))),
                    (-1) * np.log(np.exp(0.3)/(np.exp(0.7)+np.exp(0)+np.exp(0)+np.exp(0.3)))
                ],
            ])
        ),
    ]
)
def test_cross_entropy_fn__without_mask(hparams, targets, logits, length, expected_cross_entropy):
    inputs = tf.constant([
        [
            [0.5,0.6],
            [0.5,0.6],
        ],
        [
            [0.5,0.6],
            [0.5,0.6],
        ]
    ])
    targets, logits, length = tf.convert_to_tensor(targets, dtype=tf.int32), tf.convert_to_tensor(logits, dtype=tf.float32), tf.convert_to_tensor(length)
    hparams.set_hparam("mask_padding_cost", False)
    lm = LanguageModel({"inputs": inputs, "length": length}, {"targets": targets}, tf.estimator.ModeKeys.TRAIN, None, hparams)
    t_cross_entropy = lm.cross_entropy_fn(targets, logits, length)
    with tf.Session() as sess:
        r_cross_entropy = sess.run(t_cross_entropy)
    
    assert r_cross_entropy == approx(expected_cross_entropy, abs=0.001)


@pytest.mark.parametrize("targets, logits, length, expected_cross_entropy",
    [
        (
            np.array([
                [0,1],
                [2,3],
            ]),
            
            np.array([
                [
                    #[0,1,2,3]
                    [ 1, 0, 0, 0],
                    [ 0, 1, 0, 0],
                ],
                [
                    [ 0, 0, 1, 0],
                    [ 0, 0, 0, 1],
                ],
            ]),
            np.array([2,1]),
            np.array([
                [1,1],
                [1,0],
            ]) * (-1) * np.log(np.exp(1)/(np.exp(1)+np.exp(0)+np.exp(0)+np.exp(0)))
        ),
        (
            np.array([
                [0,1],
                [2,3],
            ]),
            
            np.array([
                [
                    #[0,1,2,3]
                    [ 1, 0, 0, 0],
                    [ 0, 1, 0, 0],
                ],
                [
                    [ 0, 0, 1, 0],
                    [ 0, 0, 0, 1],
                ],
            ]),
            np.array([1,2]),
            np.array([
                [1,0],
                [1,1],
            ]) * (-1) * np.log(np.exp(1)/(np.exp(1)+np.exp(0)+np.exp(0)+np.exp(0)))
        ),
        (
            np.array([
                [0,1],
                [2,3],
            ]),
            
            np.array([
                [
                    #[0,1,2,3]
                    [ 0.5, 0.5, 0, 0],
                    [ 0, 0.6, 0.4, 0],
                ],
                [
                    [ 0, 0, 1, 0],
                    [ 0.7, 0, 0, 0.3],
                ],
            ]),
            np.array([2,1]),
            np.array([
                [
                    (-1) * np.log(np.exp(0.5)/(np.exp(0.5)+np.exp(0.5)+np.exp(0)+np.exp(0))),
                    (-1) * np.log(np.exp(0.6)/(np.exp(0)+np.exp(0.6)+np.exp(0.4)+np.exp(0)))
                ],
                [
                    (-1) * np.log(np.exp(1)/(np.exp(1)+np.exp(0)+np.exp(0)+np.exp(0))),
                    0
                ],
            ])
        ),
    ]
)
def test_cross_entropy_fn__with_mask_and_batch_major(hparams, targets, logits, length, expected_cross_entropy):
    inputs = tf.constant([
        [
            [0.5,0.6],
            [0.5,0.6],
        ],
        [
            [0.5,0.6],
            [0.5,0.6],
        ]
    ])
    targets, logits, length = tf.convert_to_tensor(targets, dtype=tf.int32), tf.convert_to_tensor(logits, dtype=tf.float32), tf.convert_to_tensor(length)
    hparams.set_hparam("mask_padding_cost", True)
    hparams.set_hparam("time_major_optimization", False)
    lm = LanguageModel({"inputs": inputs, "length": length}, {"targets": targets}, tf.estimator.ModeKeys.TRAIN, None, hparams)
    t_cross_entropy = lm.cross_entropy_fn(targets, logits, length)
    with tf.Session() as sess:
        r_cross_entropy = sess.run(t_cross_entropy)
    
    assert r_cross_entropy == approx(expected_cross_entropy, abs=0.001)



@pytest.mark.parametrize("targets, logits, length, expected_cross_entropy, expected_loss",
    [
        (
            np.array([
                [0,1],
                [2,3],
            ]),
            
            np.array([
                [
                    #[0,1,2,3]
                    [ 1, 0, 0, 0],
                    [ 0, 1, 0, 0],
                ],
                [
                    [ 0, 0, 1, 0],
                    [ 0, 0, 0, 1],
                ],
            ]),
            np.array([2,1]),
            np.array([
                [1,1],
                [1,0],
            ]) * (-1) * np.log(np.exp(1)/(np.exp(1)+np.exp(0)+np.exp(0)+np.exp(0))),
            3/3 * (-1) * np.log(np.exp(1)/(np.exp(1)+np.exp(0)+np.exp(0)+np.exp(0)))
        ),
        (
            np.array([
                [0,1],
                [2,3],
            ]),
            
            np.array([
                [
                    #[0,1,2,3]
                    [ 1, 0, 0, 0],
                    [ 0, 1, 0, 0],
                ],
                [
                    [ 0, 0, 1, 0],
                    [ 0, 0, 0, 1],
                ],
            ]),
            np.array([1,2]),
            np.array([
                [1,1],
                [0,1],
            ]) * (-1) * np.log(np.exp(1)/(np.exp(1)+np.exp(0)+np.exp(0)+np.exp(0))),
            3/3 * (-1) * np.log(np.exp(1)/(np.exp(1)+np.exp(0)+np.exp(0)+np.exp(0)))
        ),
        (
            np.array([
                [0,1],
                [2,3],
            ]),
            
            np.array([
                [
                    #[0,1,2,3]
                    [ 0.5, 0.5, 0, 0],
                    [ 0, 0.6, 0.4, 0],
                ],
                [
                    [ 0, 0, 1, 0],
                    [ 0.7, 0, 0, 0.3],
                ],
            ]),
            np.array([2,1]),
            np.array([
                [
                    (-1) * np.log(np.exp(0.5)/(np.exp(0.5)+np.exp(0.5)+np.exp(0)+np.exp(0))),
                    (-1) * np.log(np.exp(0.6)/(np.exp(0)+np.exp(0.6)+np.exp(0.4)+np.exp(0)))
                ],
                [
                    (-1) * np.log(np.exp(1)/(np.exp(1)+np.exp(0)+np.exp(0)+np.exp(0))),
                    0
                ],
            ]),
            1/3 * ( 
                (-1) * np.log(np.exp(0.5)/(np.exp(0.5)+np.exp(0.5)+np.exp(0)+np.exp(0))) + \
                (-1) * np.log(np.exp(0.6)/(np.exp(0)+np.exp(0.6)+np.exp(0.4)+np.exp(0))) + \
                (-1) * np.log(np.exp(1)/(np.exp(1)+np.exp(0)+np.exp(0)+np.exp(0)))
            )
        ),
        (
            np.array([
                [0,1],
                [2,3],
            ]),
            
            np.array([
                [
                    #[0,1,2,3]
                    [ 0.5, 0.5, 0, 0],
                    [ 0, 0.6, 0.4, 0],
                ],
                [
                    [ 0, 0, 1, 0],
                    [ 0.7, 0, 0, 0.3],
                ],
            ]),
            np.array([1,2]),
            np.array([
                [
                    (-1) * np.log(np.exp(0.5)/(np.exp(0.5)+np.exp(0.5)+np.exp(0)+np.exp(0))),
                    (-1) * np.log(np.exp(0.6)/(np.exp(0)+np.exp(0.6)+np.exp(0.4)+np.exp(0)))
                ],
                [
                    0,
                    (-1) * np.log(np.exp(0.3)/(np.exp(0.7)+np.exp(0)+np.exp(0)+np.exp(0.3))),
                ],
            ]),
            1/3 * ( 
                    (-1) * np.log(np.exp(0.5)/(np.exp(0.5)+np.exp(0.5)+np.exp(0)+np.exp(0))) + \
                    (-1) * np.log(np.exp(0.6)/(np.exp(0)+np.exp(0.6)+np.exp(0.4)+np.exp(0))) + \
                    (-1) * np.log(np.exp(0.3)/(np.exp(0.7)+np.exp(0)+np.exp(0)+np.exp(0.3)))
                )
        ),
    ]
)
def test_cross_entropy_fn__with_mask_and_time_major(hparams, targets, logits, length, expected_cross_entropy, expected_loss):
    inputs = tf.constant([
        [
            [0.5,0.6],
            [0.5,0.6],
        ],
        [
            [0.5,0.6],
            [0.5,0.6],
        ]
    ])
    targets, logits, length = tf.convert_to_tensor(targets, dtype=tf.int32), tf.convert_to_tensor(logits, dtype=tf.float32), tf.convert_to_tensor(length)
    hparams.set_hparam("mask_padding_cost", True)
    hparams.set_hparam("time_major_optimization", True)
    lm = LanguageModel({"inputs": inputs, "length": length}, {"targets": targets}, tf.estimator.ModeKeys.TRAIN, None, hparams)
    t_cross_entropy = lm.cross_entropy_fn(targets, logits, length)
    t_loss = lm.loss_fn(targets, logits, length)
    with tf.Session() as sess:
        r_cross_entropy, r_loss = sess.run([t_cross_entropy, t_loss])
    
    assert r_cross_entropy == approx(expected_cross_entropy, abs=0.001)
    assert r_loss == approx(expected_loss, abs=0.001)



@pytest.mark.parametrize("lengths, max_length, time_major, expected_mask",
    [
        (
            np.array([1,2,3]), 
            3, 
            False, 
            np.array(
                [
                    [1,0,0],
                    [1,1,0],
                    [1,1,1],
                ]
            )
        ),
        (
            np.array([1,2,3]), 
            3, 
            True, 
            np.array(
                [
                    [1,1,1],
                    [0,1,1],
                    [0,0,1],
                ]
            )
        ),
    ]
)
def test_cost_mask(lengths, max_length, time_major, expected_mask):
    lengths = tf.convert_to_tensor(lengths)
    cost_mask_fn = partial(LanguageModel.cost_mask, LanguageModel)
    t_mask = cost_mask_fn(lengths, max_length, time_major)

    with tf.Session() as sess:
        r_mask = sess.run(t_mask)

    assert (r_mask == expected_mask).all()

@pytest.mark.parametrize("device", ["", "GPU", "CPU"])
@pytest.mark.parametrize("logits, targets, expected_score",
    [
        (
            np.array(
                    [
                        [
                            [0.2,0.3,0.4]
                        ]
                    ]
                ),
            np.array(
                    [
                        [0]
                    ]
                ),
            np.array(
                    [
                        [2]
                    ]
                )
        ),
        (
            np.array(
                    [   
                        [
                            [0.2,0.3,0.4]
                        ]
                    ]
                ),
            np.array(
                    [
                        [1]
                    ]
                ),
            np.array(
                    [
                        [1]
                    ]
                )
        ),
        (
            np.array(
                    [   
                        [
                            [0.2,0.3,0.4]
                        ]
                    ]
                ),
            np.array(
                    [
                        [2]
                    ]
                ),
            np.array(
                    [
                        [0]
                    ]
                )
        ),
        (
            np.array(
                    [
                        [
                            [0.2,0.3,0.4],
                            [0.2,0.5,0.4],
                            [0.5,0.3,0.4],
                            [0.5,0.6,0.4],
                        ]
                    ]
                ),
            np.array(
                    [
                        [2,
                        2,
                        2,
                        2]
                    ]
                ),
            np.array(
                    [
                        [0,
                        1,
                        1,
                        2]
                    ]
                )
        ),
        (
            np.array(
                    [
                        [
                            [0.2,0.3,0.4],
                            [0.2,0.5,0.4],
                            [0.5,0.3,0.4],
                            [0.5,0.6,0.4],
                        ],
                        [
                            [0.2,0.5,0.4],
                            [0.5,0.5,0.4],
                            [0.5,0.3,0.4],
                            [0.0,0.0,0.0],
                        ],
                    ]
                ),
            np.array(
                    [
                        [0, 1, 2, 2],
                        [2, 1, 0, 2],
                    ]
                ),
            np.array(
                    [
                        [2, 0, 1, 2],
                        [1, 1, 0, 2],
                    ]
                )
        ),
    ]
)
def test_score_of_true_word_fn(device, logits, targets, expected_score):
    t_logits = tf.convert_to_tensor(logits)
    t_targets = tf.convert_to_tensor(targets)
    
    mockLM = mock.MagicMock()
    mockLM.device = device
    mockLM.score_of_true_word_fn = partial(LanguageModel.score_of_true_word_fn, mockLM)

    t_score = mockLM.score_of_true_word_fn(t_logits, t_targets)

    with tf.Session() as sess:
        r_score = sess.run(t_score)
    
    assert (expected_score == r_score).all()


@pytest.mark.parametrize("k, logits, expected_predictions",
    [
        (
            1,
            np.array(
                    [
                        [
                            [1.0, 2.0, 3.0, 0.0],
                            [5.0, 6.0, 7.0, 8.0],
                            [9.0, 0.0, 1.0, 2.0],
                        ],
                        [
                            [1.0, 3.0, 2.0, 2.0],
                            [5.0, 6.0, 8.0, 8.0],
                            [1.0, 2.0, 1.0, 2.0],
                        ],
                    ]
                ),
            np.array(
                [
                    [
                        [2],[3],[0]
                    ],
                    [
                        [1],[2],[1]
                    ],
                ]
            ), 
        ),
        (
            2,
            np.array(
                    [
                        [
                            [1.0, 2.0, 3.0, 0.0],
                            [5.0, 6.0, 7.0, 8.0],
                            [9.0, 0.0, 1.0, 2.0],
                        ],
                        [
                            [1.0, 3.0, 2.0, 2.0],
                            [5.0, 6.0, 8.0, 8.0],
                            [1.0, 2.0, 1.0, 2.0],
                        ],
                    ]
                ),
            np.array(
                [
                    [
                        [2,1],[3,2],[0,3]
                    ],
                    [
                        [1,2],[2,3],[1,3]
                    ],
                ]
            ), 
        ),
    ]
)
def test_make_predictions__without_converting_to_words(k, logits, expected_predictions):
    mockLanguageModel = mock.Mock()
    mockLanguageModel.make_predictions = LanguageModel.make_predictions
    mockLanguageModel.predict_top_k = k
    mockLanguageModel.words_as_text_preview = False

    t_predictions = mockLanguageModel.make_predictions(mockLanguageModel, logits)

    with tf.Session() as sess:
        r_predictions = sess.run(t_predictions)

    assert (r_predictions == expected_predictions).all()


@pytest.mark.parametrize("top_predicted_words_ids, expected_tokens",
    [
        (
            np.array(
                [
                    [
                        [2,1],[4,2],[0,5]
                    ],
                    [
                        [1,2],[2,4],[1,6]
                    ],
                ]
            ), 
            np.array(
                [
                    [
                        [b"<<END_OF_SEQUENCE>>",b"<<START_OF_SEQUENCE>>"],[b"a",b"<<END_OF_SEQUENCE>>"],[b"<<ZERO>>",b"b"]
                    ],
                    [
                        [b"<<START_OF_SEQUENCE>>",b"<<END_OF_SEQUENCE>>"],[b"<<END_OF_SEQUENCE>>",b"a"],[b"<<START_OF_SEQUENCE>>",b"c"]
                    ],
                ]
            ), 
        )
    ]
)
def test_predictions_ids_to_tokens(top_predicted_words_ids, expected_tokens):
    # b"<<ZERO>>", b"<<START_OF_SEQUENCE>>", b"<<END_OF_SEQUENCE>>", ???, b"a", b"b", b"c"
    mockLanguageModel = mock.Mock()
    mockLanguageModel.predictions_ids_to_tokens = LanguageModel.predictions_ids_to_tokens
    mockLanguageModel.words_as_text_preview = True

    specials = [SpecialUnit.OUT_OF_VOCABULARY, SpecialUnit.START_OF_SEQUENCE, SpecialUnit.END_OF_SEQUENCE]
    vocab = MockVocab()
    generalized = GeneralizedVocabulary(vocab, specials)

    mockLanguageModel.vocabulary_generalized = generalized

    t_top_predicted_words_ids = tf.convert_to_tensor(top_predicted_words_ids)
    t_top_predicted_words_tokens = mockLanguageModel.predictions_ids_to_tokens(mockLanguageModel, t_top_predicted_words_ids)

    with tf.Session() as sess:
        sess.run(tf.tables_initializer())
        r_top_predicted_words_tokens = sess.run(t_top_predicted_words_tokens)

    assert (r_top_predicted_words_tokens == expected_tokens).all()


@pytest.mark.parametrize("top_predicted_words_ids, expected_tokens",
    [
        (
            np.array(
                [
                    [
                        [2,3]
                    ],
                ]
            ), 
            np.array(
                [
                    [
                        [b"<<END_OF_SEQUENCE>>",b""]
                    ],
                ]
            ), 
        )
    ]
)
def test_predictions_ids_to_tokens__undefined_id_proof(top_predicted_words_ids, expected_tokens):
    mockLanguageModel = mock.Mock()
    mockLanguageModel.predictions_ids_to_tokens = LanguageModel.predictions_ids_to_tokens
    mockLanguageModel.words_as_text_preview = True

    specials = [SpecialUnit.OUT_OF_VOCABULARY, SpecialUnit.START_OF_SEQUENCE, SpecialUnit.END_OF_SEQUENCE]
    vocab = MockVocab()
    generalized = GeneralizedVocabulary(vocab, specials)

    mockLanguageModel.vocabulary_generalized = generalized

    t_top_predicted_words_ids = tf.convert_to_tensor(top_predicted_words_ids)
    t_top_predicted_words_tokens = mockLanguageModel.predictions_ids_to_tokens(mockLanguageModel, t_top_predicted_words_ids)

    with tf.Session() as sess:
        sess.run(tf.tables_initializer())
        r_top_predicted_words_tokens = sess.run(t_top_predicted_words_tokens)

    assert (r_top_predicted_words_tokens[:,:,0] == expected_tokens[:,:,0]).all() # second one is immaterial as it is not ID at all, only matters that it doesn't cause error










@pytest.mark.parametrize("targets, logits, length, expected_cross_entropy, expected_log_perplexity",
    [
        (
            np.array([
                [0,1],
                [2,3],
            ]),
            
            np.array([
                [
                    #[0,1,2,3]
                    [ 1, 0, 0, 0],
                    [ 0, 1, 0, 0],
                ],
                [
                    [ 0, 0, 1, 0],
                    [ 0, 0, 0, 1], # time major => masked 
                ],
            ]),
            np.array([2,1]),
            np.array([
                [1,1],
                [1,0],
            ]) * (-1) * np.log(np.exp(1)/(np.exp(1)+np.exp(0)+np.exp(0)+np.exp(0))),
            3/3 * (-1) * np.log(np.exp(1)/(np.exp(1)+np.exp(0)+np.exp(0)+np.exp(0)))
        ),
        (
            np.array([
                [0,1],
                [2,3],
            ]),
            
            np.array([
                [
                    #[0,1,2,3]
                    [ 1, 0, 0, 0],
                    [ 0, 1, 0, 0],
                ],
                [
                    [ 0, 0, 1, 0], # time major => masked 
                    [ 0, 0, 0, 1],
                ],
            ]),
            np.array([1,2]),
            np.array([
                [1,1],
                [0,1],
            ]) * (-1) * np.log(np.exp(1)/(np.exp(1)+np.exp(0)+np.exp(0)+np.exp(0))),
            3/3 * (-1) * np.log(np.exp(1)/(np.exp(1)+np.exp(0)+np.exp(0)+np.exp(0)))
        ),
        (
            np.array([
                [0,1],
                [2,3],
            ]),
            
            np.array([
                [
                    #[0,1,2,3]
                    [ 0.5, 0.5, 0, 0],
                    [ 0, 0.6, 0.4, 0],
                ],
                [
                    [ 0, 0, 1, 0],
                    [ 0.7, 0, 0, 0.3], # time major => masked 
                ],
            ]),
            np.array([2,1]),
            np.array([
                [
                    (-1) * np.log(np.exp(0.5)/(np.exp(0.5)+np.exp(0.5)+np.exp(0)+np.exp(0))),
                    (-1) * np.log(np.exp(0.6)/(np.exp(0)+np.exp(0.6)+np.exp(0.4)+np.exp(0)))
                ],
                [
                    (-1) * np.log(np.exp(1)/(np.exp(1)+np.exp(0)+np.exp(0)+np.exp(0))),
                    0
                ],
            ]),
            1/3 * ( 
                (-1) * np.log(np.exp(0.5)/(np.exp(0.5)+np.exp(0.5)+np.exp(0)+np.exp(0))) + \
                (-1) * np.log(np.exp(0.6)/(np.exp(0)+np.exp(0.6)+np.exp(0.4)+np.exp(0))) + \
                (-1) * np.log(np.exp(1)/(np.exp(1)+np.exp(0)+np.exp(0)+np.exp(0)))
            )
        ),
        (
            np.array([
                [0,1],
                [2,3],
            ]),
            
            np.array([
                [
                    #[0,1,2,3]
                    [ 0.5, 0.5, 0, 0],
                    [ 0, 0.6, 0.4, 0],
                ],
                [
                    [ 0, 0, 1, 0],  # time major => masked 
                    [ 0.7, 0, 0, 0.3],
                ],
            ]),
            np.array([1,2]),
            np.array([
                [
                    (-1) * np.log(np.exp(0.5)/(np.exp(0.5)+np.exp(0.5)+np.exp(0)+np.exp(0))),
                    (-1) * np.log(np.exp(0.6)/(np.exp(0)+np.exp(0.6)+np.exp(0.4)+np.exp(0)))
                ],
                [
                    0,
                    (-1) * np.log(np.exp(0.3)/(np.exp(0.7)+np.exp(0)+np.exp(0)+np.exp(0.3))),
                ],
            ]),
            1/3 * ( 
                    (-1) * np.log(np.exp(0.5)/(np.exp(0.5)+np.exp(0.5)+np.exp(0)+np.exp(0))) + \
                    (-1) * np.log(np.exp(0.6)/(np.exp(0)+np.exp(0.6)+np.exp(0.4)+np.exp(0))) + \
                    (-1) * np.log(np.exp(0.3)/(np.exp(0.7)+np.exp(0)+np.exp(0)+np.exp(0.3)))
                )
        ),
    ]
)
def test_metrics(hparams, targets, logits, length, expected_cross_entropy, expected_log_perplexity):
    inputs = tf.constant([
        [
            [0.5,0.6],
            [0.5,0.6],
        ],
        [
            [0.5,0.6],
            [0.5,0.6],
        ]
    ])
    targets_batch_major = targets.transpose()

    # Now targets are batch-major, logits are time-major because targets are properly passed to constructor and logits are injected manually
    t_targets, t_logits, t_length = tf.convert_to_tensor(targets_batch_major, dtype=tf.int32), tf.convert_to_tensor(logits, dtype=tf.float32), tf.convert_to_tensor(length)
    
    hparams.set_hparam("mask_padding_cost", True)
    hparams.set_hparam("time_major_optimization", True)
    lm = LanguageModel({"inputs": inputs, "length": t_length}, {"targets": t_targets}, tf.estimator.ModeKeys.TRAIN, None, hparams)
    lm.logits = t_logits
    lm.position_of_true_word = tf.constant(
            np.array(
                    [
                        [2, 0, 1, 2],
                        [1, 1, 0, 2],
                    ]
                )
            )
    lm.set_metrics()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES)
        running_vars_initializer = tf.variables_initializer(var_list=running_vars)
        sess.run(running_vars_initializer)
        sess.run([lm.log_perplexity_metric_update_op, lm.position_of_true_word_metric[1]])
        r_log_perplexity_metric, r_perplexity_metric, r_position_of_true_word_metric = \
            sess.run([lm.log_perplexity_metric, lm.perplexity_metric, lm.position_of_true_word_metric[0]])
    
    assert r_position_of_true_word_metric == approx(9/8, abs=0.001)
    assert r_log_perplexity_metric == approx(expected_log_perplexity, abs=0.001)
    assert r_perplexity_metric == approx(np.exp(expected_log_perplexity), abs=0.001)
