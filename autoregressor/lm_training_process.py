from lstm_lm import PredictNext


import tensorflow as tf

class LanguageModel:
    FLOAT_TYPE = tf.float32
    def __init__(self, features, labels, mode, params):
        self.time_major_optimization = NotImplemented
        self.mask_padding_cost = NotImplemented
        self.dynamic_rnn_swap_memory = NotImplemented
        self.num_units = NotImplemented
        self.num_layers = NotImplemented
        self.vocab_size = NotImplemented
        self.last_layer_num_units = NotImplemented
        self.learning_rate = NotImplemented
        self.predict_top_k = NotImplemented
        self.words_as_text_preview = NotImplemented

        self.vocabulary_generalized = None

        self.mode = mode

        self.inputs, self.targets, self.lengths = self.unpack_nested_example(features, labels)
        self.inputs, self.targets = self.maybe_transpose_batch_time(self.inputs, self.targets)

    
    def unpack_nested_example(self, features, labels):
        inputs = features["inputs"]
        lengths = features["lengths"]
        targets = labels["targets"] if labels is not None else None
        return inputs, targets, lengths

    def ____(self):
        self.logits, _ = self.unrolled_rnn(self.inputs, self.lengths)
        self.loss = self.loss_fn(self.logits, self.targets, self.lengths)
        self.predictions_ids, self.predictions = self.make_predictions(self.logits, self.targets)
        if self.mode != tf.estimator.ModeKeys.PREDICT:
            self.train_op = self.optimize(self.loss)
            tf.summary.text("top_k_predictions", self.predictions)
            self.position_of_true_word = self.position_of_true_word_fn(self.logits, self.targets)
            tf.summary.tensor_summary("position_of_true_word", self.position_of_true_word)
            self.mean_position_of_true_word = tf.reduce_mean(self.position_of_true_word)
            tf.summary.scalar("mean_position_of_true_word", self.position_of_true_word)
            tf.summary.scalar("batch_perplexity", self.perplexity_from_loss(self.loss))
            self.set_metrics()

    def perplexity_from_loss(self, loss):
        return tf.exp(loss)

    def set_metrics(self):
        self.position_of_true_word_metric = tf.metrics.mean(self.position_of_true_word)
        self.log_perplexity_metric, self.log_perplexity_metric_update_op = tf.metrics.mean(self.cross_entropy_fn())
        self.perplexity_metric = tf.exp(self.log_perplexity_metric)
        self.metrics = {
            "position_of_true_word": self.position_of_true_word_metric,
            "log_perplexity": (self.log_perplexity_metric, self.log_perplexity_metric_update_op),
            "perplexity": (self.perplexity_metric, None),
        }        

    def score_of_true_word_fn(self, logits, targets):
        flatten_logits = tf.reshape(logits, (-1,tf.shape(logits)[-1]))
        flatten_targets = tf.reshape(targets, (-1,))
        flatten_logits_of_true_words = tf.map_fn((lambda x: tf.gather(x[0], x[1])), [flatten_logits, flatten_targets], dtype=tf.float32)
        logits_of_true_words = tf.reshape(flatten_logits_of_true_words, tf.shape(targets))

        is_score_higher_eq_than_true_one = tf.greater_equal(logits, tf.expand_dims(logits_of_true_words,1))
        n_score_higher_eq_than_true_one = tf.reduce_sum(tf.cast(is_score_higher_eq_than_true_one, tf.int8), axis=1)

        return n_score_higher_eq_than_true_one - 1 # -1 comes from the fact that true index has score equal to true index so it counts to higher-equal

    def probabilities_fn(self, logits):
        return tf.nn.softmax(logits=logits)

    def make_predictions(self, logits):
        top_predicted_words_ids = tf.nn.top_k(logits, self.predict_top_k)
        if self.words_as_text_preview:
            original_shape = tf.shape(top_predicted_words_ids)
            flatten_top_predicted_words_ids = tf.reshape(top_predicted_words_ids, (-1,))
            flatten_top_predicted_words = self.vocabulary_generalized.generalized_id_to_token()(flatten_top_predicted_words_ids)
            top_predicted_words = tf.reshape(flatten_top_predicted_words, shape=original_shape)
        else:
            top_predicted_words = None
        return top_predicted_words_ids, top_predicted_words

    def loss_fn(self, logits, targets, lengths):
        ce = self.cross_entropy_fn(targets, logits, lengths)
        return tf.reduce_mean(self.cross_entropy_based_loss(ce))

    def cross_entropy_based_loss(self, cross_entropy):
        return tf.reduce_mean(cross_entropy)
    
    def cross_entropy_fn(self, targets, logits, lengths):
        try:
            cross_entropy = self.cross_entropy
        except AttributeError:
            cross_entropy = None
        if cross_entropy is None:
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets,
                                                                            logits=logits)
            if self.mask_padding_cost:
                mask = self.cost_mask(targets, tf.shape(targets[1]), self.time_major_optimization)
                cross_entropy = cross_entropy * mask
            self.cross_entropy = cross_entropy
        return cross_entropy

    def cost_mask(self, lengths, max_length, time_major):
        mask = tf.sequence_mask(lengths, max_length, self.FLOAT_TYPE)
        if time_major:
            mask = tf.transpose(mask, (1, 0))
        return mask

    def maybe_transpose_batch_time(self, inputs, targets):
        if self.time_major_optimization:
            inputs = tf.transpose(inputs, (1, 0, 2))
            targets = tf.transpose(targets, (1, 0))

    def unrolled_rnn(self, inputs, lengths):
        logits, state = tf.nn.dynamic_rnn(self.cell(), inputs,
                                            sequence_length=lengths,
                                            dtype=self.FLOAT_TYPE,
                                            time_major=self.time_major_optimization,
                                            swap_memory=self.dynamic_rnn_swap_memory)
        return logits, state

    def cell(self):
        cell = PredictNext(
            self.num_units, self.num_layers, self.vocab_size, last_layer_num_units=self.last_layer_num_units)
        return cell

    def optimize(self, loss):
        optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate)

        train_op = optimizer.minimize(
            loss=loss, global_step=tf.train.get_global_step())
        
        return train_op

    def estimator_spec(self):
        return NotImplemented


class LanguageModelCallable:


    def model_function(features, labels, mode, params):

        if hparams is not None:
            num_units = hparams.rnn_num_units
            num_layers = hparams.rnn_num_layers
            last_layer_num_units = hparams.rnn_last_layer_num_units
            predictor = PredictNext(num_units, num_layers, vocab_size,
                                    last_layer_num_units=last_layer_num_units, rnn_cell_type=hparams.rnn_layer)
        else:
            num_units = NUM_UNITS
            num_layers = NUM_LAYERS
            last_layer_num_units = None
        if mode == tf.estimator.ModeKeys.PREDICT:
            sentence, length = features["inputs"], features["length"]
            if time_major_optimization:
                sentence = tf.transpose(sentence, (1, 0, 2))
            logits, state = tf.nn.dynamic_rnn(predictor, sentence,
                                              sequence_length=length,
                                              dtype=tf.float32,
                                              time_major=time_major_optimization,
                                              swap_memory=True)
            probabilities = tf.nn.softmax(logits=logits)
            if time_major_optimization:
                probabilities = tf.transpose(probabilities, (1, 0, 2))
            predicted_word_id = tf.argmax(probabilities, 2)
            spec = tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={
                    "predicted_word_id": predicted_word_id,
                    "probabilities": probabilities
                })
        else:

            sentence, length = features["inputs"], features["length"]
            targets = labels["targets"]

            masked_cross_entropy = self.masked_cross_entropy(targets, labels, length)

            if hparams.mask_padding_cost == True:
                cross_entropy = cross_entropy * cost_mask

            loss = tf.reduce_mean(cross_entropy)

            probabilities = tf.nn.softmax(logits=logits)
            predicted_word = tf.argmax(probabilities, 2)

            metrics = {
                # "cross_entropy": cross_entropy,
                "accuracy": tf.metrics.accuracy(targets, predicted_word)
            }

            # Wrap all of this in an EstimatorSpec.
            spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op,
                eval_metric_ops=metrics
            )