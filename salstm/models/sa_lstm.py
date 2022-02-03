import time
from random import randint

import numpy as np

import models.initializers
import tensorflow as tf
from salstm.models.base import BaseModel
from salstm.models.meteor import Meteor
from salstm.models.tokens import EOS_TOKEN, PAD_TOKEN, SOS_TOKEN
from salstm.utils import file_utils, graph_utils, misc
from salstm.utils.dbpedia_utils import (pos_process_tokens,
                                        reverse_word_dictionary)
from utils.tensor_utils import pad_sequence, transpose_batch_time


class SALSTM_TextEncoder(BaseModel):
    """
    Implementation of the Sequence Auto-encoder (SA-LSTM) of the paper Dai and Le,
    "Semi-supervised Sequence Learning"
    in https://arxiv.org/pdf/1511.01432.pdf
    The SA-LSTM is mostly based on the auto-encoder of Sutskever et al.,
    Sequence to Sequence Learning with Neural Networks in
    https://arxiv.org/pdf/1409.3215.pdf

    Notes:
        1. Original paper of Dai and Le, use a classic LSTM.
        This code implements a bidirectional LSTM.
        2. Encoder deals with SOS token and EOS token. Inputs must only contain the text.
        3. Different from the original paper we report the METEOR metric too for the inference phase

    References:
        - Tensorflow implementation (first author of the paper is a contributor)
         of https://github.com/tensorflow/models/tree/master/research/adversarial_text
        - Tensorflow implementation of https://github.com/dongjun-Lee/transfer-learning-text-tf
        - Tensorflow implementation of https://github.com/isohrab/semi-supervised-text-classification
        - Tensorflow implementation of https://github.com/JayParks/tf-seq2seq/blob/master/seq2seq_model.py

    TODO:
        - eval graph together with the training graphs
        - it seems that some sequences are generating tokens after the eos token in
        the inference phase. This is not supposed to happen

    """

    def __init__(self, word_dict, initialize, cfg_model, is_training, build_only_encoder=False):
        super(SALSTM_TextEncoder, self).__init__()

        self.word_embedding_size = cfg_model.word_embedding_size
        self.num_hidden_units = cfg_model.num_hidden_units
        self.dropout_rate_embeddings = cfg_model.dropout_rate_embeddings
        self.dropout_rate_output = cfg_model.dropout_rate_output

        self.cfg_model = cfg_model
        # We can try to swap memory between CPU and GPU to allow the computation of long sequences
        self.swap_memory = self.cfg_model.swap_memory

        self.vocabulary_size = len(word_dict)
        self.word_dict = word_dict
        self.reversed_word_dict = reverse_word_dictionary(word_dict)

        self.scope_model = cfg_model.type
        self.is_training = is_training

        # [batch_size, max_text_length]
        self.encoder_inputs = tf.placeholder(
            dtype=tf.int32, shape=(None, None), name="encoder_inputs")
        # Tensor with the batch_size and another for the max_sequence_length
        self._batch_size = tf.shape(self.encoder_inputs)[0]

        # We add a EOS token at the end of the encoder input as in the original paper
        # But since the sequences are padded already, we need to do a trick to insert the
        # EOS token at the right place. Adding one more position to include a EOS to the longest
        # sequence (without this the longest sequence will not have a EOS token at the end)
        encoder_inputs_padded = tf.concat([self.encoder_inputs, tf.ones(
            [self._batch_size, 1], tf.int32) * word_dict[PAD_TOKEN]], axis=1)

        # Original sequence lengths
        self.encoder_inputs_lengths = tf.reduce_sum(tf.sign(encoder_inputs_padded), 1)

        # Creating a matrix where after the last word of each sequence we add a EOS token
        t_indices = tf.stack(
            [tf.range(0, self._batch_size), self.encoder_inputs_lengths], axis=1)
        t_indices = tf.cast(t_indices, tf.int64)

        eos_values = tf.fill([self._batch_size], word_dict[EOS_TOKEN])
        # eos_values = tf.cast(eos_values, tf.int64)
        # print(eos_values)
        dense_shape = tf.cast(tf.shape(encoder_inputs_padded), tf.int64)
        sparse_eos_values = tf.SparseTensor(
            indices=t_indices, values=eos_values, dense_shape=dense_shape)
        dense_eos_values = tf.sparse_tensor_to_dense(sparse_eos_values)

        # self.encoder_inputs_with_eos = tf.concat([self.encoder_inputs, tf.ones([self._batch_size, 1], tf.int32) * word_dict[EOS_TOKEN]], axis=1)
        self.encoder_inputs_with_eos = tf.add(encoder_inputs_padded, dense_eos_values)

        self.decoder_inputs_wo_sos = self.encoder_inputs
        # decoder_inputs = tf.concat([tf.ones([self._batch_size, 1], tf.int32) * word_dict[SOS_TOKEN], self.encoder_inputs], axis=1)
        self.decoder_expected_outputs = tf.add(encoder_inputs_padded, dense_eos_values)
        # tf.concat([self.encoder_inputs, tf.ones([self._batch_size, 1], tf.int32) * word_dict[EOS_TOKEN]], axis=1)

        self._max_sequence_length_decoder = tf.shape(self.decoder_expected_outputs)[1]

        self.sos_token = tf.ones([self._batch_size], tf.int32) * word_dict[SOS_TOKEN]
        self.pad_token = tf.ones([self._batch_size], tf.int32) * word_dict[PAD_TOKEN]

        # [batch_size]
        self.encoder_inputs_lengths = tf.reduce_sum(tf.sign(self.encoder_inputs_with_eos), 1)
        # self.decoder_inputs_lengths = tf.reduce_sum(tf.sign(decoder_inputs), 1)

        # If is not training, we consider the sequence length or maximum number of iterations as
        # 2 times the actual sequence length of the inputs
        if self.is_training:
            self.decoder_inputs_lengths_wo_sos = tf.reduce_sum(
                tf.sign(self.decoder_inputs_wo_sos), 1)
        else:
            self.decoder_inputs_lengths_wo_sos = tf.reduce_max(
                tf.reduce_sum(tf.sign(self.decoder_inputs_wo_sos), 1)) * 2

        # Create variables
        self.embeddings, self.scope_embeddings, self.w_logits, self.b_logits, self.scope_decoder = \
            self.build_variables(initialize, build_only_encoder)

    def build_variables(self, initialize, build_only_encoder):
        """Create the variables, except the ones of the LSTM model."""
        print("Building variables...")
        with tf.variable_scope("embeddings") as scope_embeddings:
            embeddings = models.initializers.create_variable('input_embeddings', [self.vocabulary_size, self.word_embedding_size],
                                                             initialize=initialize, init_type='uniform', min_val=-1, max_val=1)

        w_logits, b_logits, scope_decoder = None, None, None
        if not build_only_encoder:
            with tf.variable_scope("decoder_rnn") as scope_decoder:
                if not self.cfg_model.bidirectional_encoder:
                    num_hidden_units = self.num_hidden_units
                else:
                    num_hidden_units = self.num_hidden_units * 2
                w_logits = models.initializers.create_variable('w_logits', [num_hidden_units, self.vocabulary_size],
                                                               initialize=initialize, init_type='xavier')
                b_logits = models.initializers.create_variable('b_logits', [self.vocabulary_size],
                                                               initialize=initialize, init_type='xavier')

        return embeddings, scope_embeddings, w_logits, b_logits, scope_decoder

    def build_embeddings(self, embeddings_matrix, inputs):
        # Embeddings
        with tf.variable_scope(self.scope_embeddings, reuse=tf.AUTO_REUSE):
            inputs_embedded = tf.nn.embedding_lookup(embeddings_matrix, inputs)

            # Apply dropout as in the paper, and using the following as reference
            # https://github.com/tensorflow/models/blob/56cbd1f2770f1e7386db43af37f6f11b4d85e3da/research/adversarial_text/layers.py#L77
            inputs_embedded = tf.nn.dropout(
                inputs_embedded, rate=self.dropout_rate_embeddings)

            return inputs_embedded

    def _encoder_variables(self):
        encoder_vars_embeddings = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope="embeddings")
        encoder_vars_encoder = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope="encoder_rnn")

        return encoder_vars_embeddings + encoder_vars_encoder

    def build_encoder(self):

        inputs_embedded = self.build_embeddings(
            embeddings_matrix=self.embeddings, inputs=self.encoder_inputs_with_eos)

        with tf.variable_scope("encoder_rnn") as self.scope_encoder:
            # Forward cell (default)
            encoder_fw_cell = tf.nn.rnn_cell.LSTMCell(self.num_hidden_units, name="forward_cell")

            # Encoder
            if self.cfg_model.bidirectional_encoder:
                encoder_bw_cell = tf.nn.rnn_cell.LSTMCell(self.num_hidden_units,
                                                          name="backward_cell")
                encoder_outputs, encoder_states = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=encoder_fw_cell,
                    cell_bw=encoder_bw_cell,
                    inputs=inputs_embedded,
                    sequence_length=self.encoder_inputs_lengths,
                    dtype=tf.float32,
                    swap_memory=self.swap_memory
                )
                # We concatenate the inputs of the forward and backward pass
                outputs_concat = tf.concat(encoder_outputs, -1, name="concat_outputs_fw_and_bw")

                # Similarly we concatenate the variables c and h of the forward and backward pass
                # c is the hidden state, and h is the output
                output_state_fw, output_state_bw = encoder_states

                states_concat_c = tf.concat(
                    [output_state_fw.c, output_state_bw.c], -1, name="state_concat_c")
                states_concat_h = tf.concat(
                    [output_state_fw.h, output_state_bw.h], -1, name="state_concat_h")

                new_state = tf.nn.rnn_cell.LSTMStateTuple(c=states_concat_c, h=states_concat_h)

                return outputs_concat, new_state
            else:
                encoder_outputs, encoder_states = tf.nn.dynamic_rnn(
                    cell=encoder_fw_cell,
                    inputs=inputs_embedded,
                    sequence_length=self.encoder_inputs_lengths,
                    dtype=tf.float32,
                    swap_memory=self.swap_memory
                )
                return encoder_outputs, encoder_states

    def build_loss(self, decoder_outputs, decoder_targets, is_training):

        if not is_training:
            # Inference could have generated sequences with sizes different from what we expected
            # batch size, max steps out, output dimension
            _, decoder_max_steps_out, _ = tf.unstack(tf.shape(decoder_outputs))
            # batch size target, max steps target
            _, decoder_max_steps_target = tf.unstack(tf.shape(decoder_targets))

            # Pad target sequences if decoder_target.shape[1] < decoder_outputs.shape[1]
            # Targets don't need casting to float
            decoder_targets = tf.cond(
                tf.less(decoder_max_steps_target, decoder_max_steps_out),
                lambda: pad_sequence(decoder_targets, desired_size=decoder_max_steps_out,
                                     constant_value=self.word_dict[PAD_TOKEN]),
                lambda: decoder_targets
            )

            # Pad output sequences if decoder_target.shape[1] > decoder_outputs.shape[1]
            decoder_outputs = tf.cond(
                tf.less(decoder_max_steps_out, decoder_max_steps_target),
                lambda: pad_sequence(decoder_outputs, desired_size=decoder_max_steps_target,
                                     constant_value=tf.nn.embedding_lookup(
                                         self.embeddings, self.word_dict[PAD_TOKEN]),
                                     dtype=tf.float32),
                lambda: decoder_outputs
            )

        # Reshaping decoder outputs
        print(f"decoder_outputs.shape={decoder_outputs.shape}")
        decoder_batch_size, decoder_max_steps, decoder_dim = tf.unstack(
            tf.shape(decoder_outputs))
        decoder_outputs_flat = tf.reshape(decoder_outputs, (-1, decoder_dim))
        print(f"decoder_outputs_flat.shape={decoder_outputs_flat.shape}")
        print(f"self.w_logits.shape={self.w_logits.shape}")

        logits_flat = tf.matmul(decoder_outputs_flat, self.w_logits) + self.b_logits
        print(f"logits_flat.shape={logits_flat.shape}")
        targets_flat = tf.reshape(decoder_targets, [-1])
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=targets_flat, logits=logits_flat, name='sparse_cross_entropy')

        # Mask to compute the loss only over the expected outputs, not over the padded outputs
        # Mask the losses
        sequence_mask = tf.sign(tf.cast(targets_flat, tf.float32))
        # tf.reshape(self.sequence_mask, [-1])
        masked_losses = loss * sequence_mask

        # Bring back to shape [batch_size, time_steps]
        masked_losses = tf.reshape(masked_losses, tf.shape(decoder_targets))

        # Average across time steps
        # https://github.com/tensorflow/tensorflow/blob/a6d8ffae097d0132989ae4688d224121ec6d8f35/tensorflow/contrib/seq2seq/python/ops/loss.py#L105
        masked_losses = tf.reduce_sum(masked_losses, axis=[-1])
        total_size = tf.reduce_sum(sequence_mask, axis=[-1])
        total_size += 1e-12  # to avoid division by 0 for all-0 weights/padded positions
        masked_losses /= total_size

        # Average across batch size
        masked_losses = tf.reduce_mean(masked_losses)
        print(f"masked_losses.shape={masked_losses.shape}")

        # Ids of the predicted tokens
        logits = tf.reshape(logits_flat,
                            (decoder_batch_size, decoder_max_steps, self.vocabulary_size))
        predicted_tokens = tf.argmax(logits, axis=-1)
        sequence_mask_int = tf.reshape(tf.dtypes.cast(
            sequence_mask, tf.int64), (decoder_batch_size, decoder_max_steps))
        predicted_tokens = predicted_tokens * sequence_mask_int

        return masked_losses, predicted_tokens

    def build_optimization(self, loss, learning_rate, max_gradient_norm):

        # Get variables of the SA LSTM scopes only
        # Embeddings
        embeddings_trainable_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope="embeddings")
        encoder_trainable_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope="encoder_rnn")
        decoder_trainable_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope="decoder_rnn")

        print(*embeddings_trainable_vars, sep='\n')
        print("\n")
        print(*encoder_trainable_vars, sep='\n')
        print("\n")
        print(*decoder_trainable_vars, sep='\n')
        print("\n")

        trainable_vars = embeddings_trainable_vars + \
            encoder_trainable_vars + decoder_trainable_vars

        opt = tf.train.AdamOptimizer(learning_rate)
        grads, _ = zip(*opt.compute_gradients(loss, trainable_vars))

        # Clip non-embedding grads
        # https://github.com/tensorflow/models/blob/fe1a9089ac65956aea1f92253c3263eca9b19c3a/research/adversarial_text/layers.py#L310
        non_embedding_grads_and_vars = [(g, v) for (g, v) in zip(
            grads, trainable_vars) if 'embeddings' not in v.op.name]
        embedding_grads_and_vars = [(g, v) for (g, v) in zip(
            grads, trainable_vars) if 'embeddings' in v.op.name]

        print(f"Non embedding variables: {non_embedding_grads_and_vars}")
        print(f"Embedding variables: {embedding_grads_and_vars}")

        non_embedding_grads, non_embedding_vars = zip(*non_embedding_grads_and_vars)

        # Calculate global norm to debug on tensorboard
        global_norm_non_emb = tf.global_norm(non_embedding_grads)
        tf.summary.scalar("global_norm_non_emb", global_norm_non_emb)
        non_embedding_grads, _ = tf.clip_by_global_norm(
            non_embedding_grads, max_gradient_norm, use_norm=global_norm_non_emb)

        # non_embedding_grads_and_vars = zip(non_embedding_grads, non_embedding_vars)
        non_embedding_grads_and_vars = [(g, v) for (g, v) in zip(
            non_embedding_grads, non_embedding_vars)]
        grads_and_vars = embedding_grads_and_vars + non_embedding_grads_and_vars

        train_op = opt.apply_gradients(grads_and_vars)

        return train_op

    def build_model(self, merge_summaries=True):
        """Build the Sequence Auto-Encoder model"""
        print("Building model...")

        # Encoder
        _, self.encoder_states = self.build_encoder()

        # Decoder
        decoder_inputs_embedded_wo_sos = self.build_embeddings(
            embeddings_matrix=self.embeddings, inputs=self.decoder_inputs_wo_sos)

        # with tf.variable_scope(self.scope_decoder, reuse=tf.AUTO_REUSE):
        #     decoder_cell = tf.nn.rnn_cell.LSTMCell(self.num_hidden_units)
        #     self.decoder_outputs, _ = tf.nn.dynamic_rnn(cell=decoder_cell, inputs=decoder_inputs_embedded,
        #                                                 sequence_length=self.decoder_inputs_lengths,
        #                                                 initial_state=encoder_states, dtype=tf.float32)

        self.decoder_outputs = self._decoder_raw_rnn(
            decoder_inputs_embedded_wo_sos,
            batch_size=self._batch_size,
            max_sequence_length=self._max_sequence_length_decoder,
            scope=self.scope_decoder,
            num_hidden_units=self.num_hidden_units if not self.cfg_model.bidirectional_encoder else self.num_hidden_units * 2,
            is_training=self.is_training,
            embeddings=self.embeddings,
            sos_token=self.sos_token,
            pad_token=self.pad_token,
            sequence_length=self.decoder_inputs_lengths_wo_sos,
            initial_state=self.encoder_states
        )

        # Original paper use dropout at the word level too, beyond the input embeddings.
        # Following implementation (first author of the paper is a contributor) of
        # https://github.com/tensorflow/models/blob/fe1a9089ac65956aea1f92253c3263eca9b19c3a/research/adversarial_text/layers.py#L129
        self.decoder_outputs = tf.nn.dropout(self.decoder_outputs, rate=self.dropout_rate_output)

        # Define loss
        self.masked_loss, self.predicted_tokens = self.build_loss(
            decoder_outputs=self.decoder_outputs,
            decoder_targets=self.decoder_expected_outputs,
            is_training=self.is_training
        )
        tf.summary.scalar("masked_losses", self.masked_loss)
        self.metrics_to_summarize = {"masked_losses": self.masked_loss}

        # Optimization
        self.train_op = self.build_optimization(
            loss=self.masked_loss,
            learning_rate=self.cfg_model.optimizer.learning_rate,
            max_gradient_norm=self.cfg_model.optimizer.max_gradient_norm
        )

        if merge_summaries:
            # Merge all summaries into a single op
            self.merged_summary_op = tf.summary.merge_all()
            # Set metrics per epoch
            self.metrics_to_summarize = {"loss": self.masked_loss}
            self.metrics_per_epoch()

        print("Build of {} model done.\n".format(self.scope_model.upper()))

    def _decoder_raw_rnn(self, sequence_input, batch_size, max_sequence_length, scope, num_hidden_units,
                         is_training, embeddings, sos_token, sequence_length, pad_token=None, initial_state=None):
        """Implementation of a decoder RNN using Tensorflow function raw_rnn.
           raw_rnn is an experimental function. The code is based on the function from Tensorflow 1.11.

           References:
               - https://hanxiao.github.io/2017/08/16/Why-I-use-raw-rnn-Instead-of-dynamic-rnn-in-Tensorflow-So-Should-You-0/
               - https://github.com/ematvey/tensorflow-seq2seq-tutorials/blob/master/2-seq2seq-advanced.ipynb
        """
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            inputs_ta = None
            if is_training:
                # Only creates the inputs_ta when training, if we create a TensorArray in
                # inference we will see a warning that inputs_ta is not being used
                inputs_ta = tf.TensorArray(size=max_sequence_length,
                                           dtype=tf.float32, name='inputs_ta')
                # We transpose since here we are going to use the shape
                # as [max_time, batch_size, word_embedding_size]
                inputs_ta = inputs_ta.unstack(transpose_batch_time(sequence_input))

            # Sequence_length must not count for sos token, because we do that here
            sequence_length = sequence_length + tf.ones([batch_size], tf.int32)

            # Start of Sequence (SOS) token
            sos_token_embedded = tf.nn.embedding_lookup(embeddings, sos_token)
            pad_token_embedded = tf.nn.embedding_lookup(embeddings, pad_token)

            # Define the LSTM cell
            decoder_cell = tf.nn.rnn_cell.LSTMCell(num_hidden_units, name="raw_lstm_cell")
            output_ta = tf.TensorArray(size=max_sequence_length, dtype=tf.int32)

            def loop_fn(time, cell_output, cell_state, loop_state):
                emit_output = cell_output  # == None for time == 0

                # Determines which sequences are finished and which are not. elements_finished [batch_size]
                elements_finished = (time >= sequence_length)
                # Check if all sequences are finished
                finished = tf.reduce_all(elements_finished)

                if cell_output is None:
                    next_cell_state = initial_state
                    next_input = sos_token_embedded
                    next_loop_state = output_ta
                else:
                    # Pass the last state to the next
                    next_cell_state = cell_state

                    if not is_training:
                        # Inference phase
                        cell_output_drop = tf.nn.dropout(cell_output,
                                                         rate=self.dropout_rate_output)
                        current_logits = tf.matmul(
                            cell_output_drop, self.w_logits) + self.b_logits

                        # Id of the predicted token
                        prediction = tf.argmax(current_logits, axis=-1)

                        # Which elements have the EOS token, indicating the end of the sequence
                        # Note: prediction has tf.int64 data type
                        elements_ended_by_eos_token = tf.equal(prediction, tf.ones(
                            [batch_size], tf.int64) * self.word_dict[EOS_TOKEN])

                        # Check whether the maximum iterations (in inference this is the sequence_length) is reached or if the EOS_TOKEN was generated
                        elements_finished = tf.logical_or(
                            elements_finished, elements_ended_by_eos_token)
                        finished_for_inf = tf.reduce_all(elements_finished)

                        # Prepare next input
                        prediction_embedded = tf.nn.embedding_lookup(
                            embeddings, prediction)

                        next_input = tf.cond(finished_for_inf,
                                             # pad_token_embedded or tf.zeros([batch_size, self.word_embedding_size], dtype=tf.float32))
                                             lambda: pad_token_embedded,
                                             lambda: prediction_embedded)
                    else:
                        # Note: If I use this part, which tries to build both inference and training graphs with the same code
                        # it throws an error when running, saying that there is no memory to allocate the gradients for the false case.
                        # next_input_tmp = tf.cond(is_training,
                        #                     lambda: inputs_ta.read(time - 1),
                        #                     lambda: tf.nn.embedding_lookup(embeddings, tf.argmax(tf.matmul(cell_output, self.w_logits) + self.b_logits, axis=-1))
                        #                     # get_next_input_inference()
                        #                     )

                        # Training phase
                        next_input = tf.cond(
                            finished,
                            # tf.zeros([batch_size, self.word_embedding_size], dtype=tf.float32), #pad_token_embedded,
                            lambda: pad_token_embedded,
                            lambda: inputs_ta.read(time - 1)  # next_input_tmp
                        )

                    next_loop_state = loop_state.write(time - 1, next_input)

                return (elements_finished, next_input, next_cell_state, emit_output, next_loop_state)

            decoder_outputs_ta, final_state, final_loop_state = tf.nn.raw_rnn(
                decoder_cell, loop_fn)

            # Converting back to [batch_size, max_time, cell_size]
            outputs = transpose_batch_time(decoder_outputs_ta.stack())

        return outputs

    def feed_dict_model(self, batch_tokens):
        # Dictionaries for feed_dict
        return {self.encoder_inputs: batch_tokens}

    def train_on_batch(self, session, batch_tokens):

        dict_train_encoder = self.feed_dict_model(batch_tokens)

        # Run optimizer
        _, loss, predicted_tokens, target_tokens, seq_length = \
            session.run([self.train_op, self.masked_loss,
                         self.predicted_tokens, self.decoder_expected_outputs,
                         self.decoder_inputs_lengths_wo_sos], feed_dict=dict_train_encoder)

        # Update metrics that are calculate for each epoch. Get values per video to generate final statistics per epoch
        session.run(self.update_metrics_op, feed_dict=dict_train_encoder)

        predicted_words = list(map(lambda words: list(map(
            lambda word_id: self.reversed_word_dict.get(word_id), words)), predicted_tokens))
        target_words = list(map(lambda words: list(
            map(lambda word_id: self.reversed_word_dict.get(word_id), words)), target_tokens))

        return dict_train_encoder, loss, seq_length, predicted_words, target_words

    def eval_on_batch(self, session, batch_tokens):

        dict_eval_encoder = self.feed_dict_model(batch_tokens)

        # Run optimizer
        loss, predicted_tokens, target_tokens, seq_length = \
            session.run([self.masked_loss, self.predicted_tokens, self.decoder_expected_outputs,
                         self.decoder_inputs_lengths_wo_sos], feed_dict=dict_eval_encoder)

        # Update metrics that are calculate for each epoch. Get values per video to generate final statistics per epoch
        session.run(self.update_metrics_op, feed_dict=dict_eval_encoder)

        predicted_words = list(map(lambda words: list(map(
            lambda word_id: self.reversed_word_dict.get(word_id), words)), predicted_tokens))
        target_words = list(map(lambda words: list(
            map(lambda word_id: self.reversed_word_dict.get(word_id), words)), target_tokens))

        return dict_eval_encoder, loss, seq_length, predicted_words, target_words

    def summary_on_losses(self, losses, predicted_words, target_words, seq_length):
        print(f"Text: {np.mean(losses):.5f}.")
        # Lets see which words were predicted
        id_sample = randint(0, len(predicted_words) - 1)
        str_output = f"Predicted words ({len(predicted_words[id_sample])}): {predicted_words[id_sample]}\n" \
            f"Target words ({len(target_words[id_sample])}, " \
            f"Seq. length: {seq_length[id_sample]}): {target_words[id_sample]}"

        return str_output

    def train(self, train_cfg, training_dataset, nb_documents_train, epochs, batch_size,
              output_checkpoint, output_checkpoint_encoder, restore, output_dir, meteor_dir):

        training_iterator = training_dataset.make_one_shot_iterator()
        # validation_iterator = validation_dataset.make_initializable_iterator()

        next_element_train = training_iterator.get_next()
        # next_element_val = validation_iterator.get_next()

        nb_batch_per_epoch = (nb_documents_train // batch_size) + \
            (1 if nb_documents_train % batch_size > 0 else 0)
        # nb_batch_per_epoch_val = (nb_documents_val // self.batch_size) + (1 if nb_documents_val % self.batch_size > 0 else 0)
        print("batch_size={}, nb_batch_per_epoch={}, epochs={}".format(
            batch_size, nb_batch_per_epoch, epochs))
        # print("batch_size={}, nb_batch_per_epoch_val={}, epochs=1".format(self.batch_size, nb_batch_per_epoch_val, epochs))

        # Create global step (for epochs), to be used when we restore the model to continue training
        global_epoch = tf.Variable(1, trainable=False, name='global_epoch')
        increment_global_epoch = tf.assign_add(global_epoch, 1, name='increment_global_epoch')

        with tf.device('/cpu:0'):
            self.meteor_per_epoch = tf.Variable(0.0)
        merged_summary_per_epoch_op = self.add_common_vars(additional_variables={
            "meteor": self.meteor_per_epoch
        })

        # Add ops to save and restore all the variables, and a saver to only restore the variables of the encoder
        saver = tf.train.Saver(
            max_to_keep=2, keep_checkpoint_every_n_hours=12, name='saver_model')
        saver_encoder = tf.train.Saver(var_list=self._encoder_variables(), max_to_keep=2,
                                       keep_checkpoint_every_n_hours=12, name='saver_encoder')

        # List of the metrics. Note that will be a list of dictionaries, each dictionary corresponding to one epoch.
        training_metrics_per_epoch = []
        # val_metrics_per_epoch = []

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # I will test without allow_growth config. I think for this model memory fragmentation is messing
        # things up when we add memory for new variables giving memory allocation errors
        with tf.Session(config=config) as session:
            if not restore:
                print("Start training")
                # The op for initializing the variables.
                session.run(tf.group(tf.global_variables_initializer(),
                                     tf.local_variables_initializer()))
            else:
                if not graph_utils.restore_model(saver, session, output_checkpoint, output_dir,
                                                 latest_filename='checkpoint'):
                    return

                # Statistics of the metrics
                training_metrics_per_epoch = file_utils.load(f"{output_dir}training_metrics_per_epoch.pkl")
                # val_metrics_per_epoch = myutils.load("{}val_metrics_per_epoch.pkl".format(self.options['dir/models']))

            # Operation to write logs to Tensorboard
            summary_writer = tf.summary.FileWriter(output_dir, graph=tf.get_default_graph())
            # writer_val = tf.summary.FileWriter(model_dir + 'plot_val')
            writer_train = tf.summary.FileWriter(output_dir + 'plot_train')

            try:
                time_per_epoch, max_sequence = 0, 0

                for _ in range(epochs):
                    predicted_sequences, target_sequences = [], []
                    losses = []

                    # train_results_metrics_per_video = {}
                    epoch_start_time = time.time()

                    session.run(self.metrics_init_op)
                    current_epoch = session.run(global_epoch)

                    # Training
                    for current_batch in range(1, nb_batch_per_epoch + 1):
                        # Generates a dictionary {"sequence_length: x, "tokens": [1, 45, etc]"}
                        batch_text = session.run(next_element_train)

                        dict_training_encoder, loss, seq_length, predicted_words, target_words = \
                            self.train_on_batch(session, batch_text["tokens"])

                        predicted_sequences += pos_process_tokens(predicted_words)
                        target_sequences += pos_process_tokens(target_words)
                        losses.append(loss)

                        max_seq_batch = np.amax(batch_text["sequence_length"])
                        if max_seq_batch > max_sequence:
                            max_sequence = max_seq_batch

                        if current_batch % train_cfg.display_loss_every_step == 0:
                            print(f"Epoch {current_epoch}/{epochs}, Batch {current_batch}/{nb_batch_per_epoch} "
                                  f"Time (last epoch): {time_per_epoch:.2f}")
                            print(self.summary_on_losses(
                                losses, predicted_words, target_words, seq_length))

                        if current_batch % train_cfg.save_summary_every_step == 0:
                            summary = session.run(self.merged_summary_op,
                                                  feed_dict=dict_training_encoder)
                            summary_writer.add_summary(
                                summary, (current_epoch - 1) * nb_batch_per_epoch + current_batch)

                    # End of epoch
                    # Saving predicted and target sequences
                    file_utils.save_list(output_dir + "predicted_sequences.txt",
                                         some_list=predicted_sequences)
                    file_utils.save_list(output_dir + "target_sequences.txt",
                                         some_list=target_sequences)

                    # Calculate the METEOR metric for the predictions
                    score_meteor = Meteor(meteor_path=meteor_dir,
                                          file_hypothesis="train_predicted_sequences.txt",
                                          file_reference="train_target_sequences.txt").score

                    training_metrics = self.write_metrics_per_epoch(
                        session, writer_train, current_epoch)
                    training_metrics = misc.merge_dicts(
                        training_metrics, {'meteor': score_meteor})
                    print("training_metrics_statistics={}".format(training_metrics))
                    training_metrics_per_epoch.append(training_metrics.copy())

                    time_per_epoch = time.time() - epoch_start_time
                    summary = session.run(
                        merged_summary_per_epoch_op,
                        feed_dict=misc.merge_dicts(dict_training_encoder,
                                                   {self.time_per_epoch: time_per_epoch,
                                                    self.meteor_per_epoch: score_meteor}))
                    writer_train.add_summary(summary, current_epoch)
                    writer_train.flush()

                    # Saving statistics for all epochs
                    file_utils.save("{}training_metrics_per_epoch.pkl".format(
                        output_dir), training_metrics_per_epoch)
                    print("Metrics statistics per epoch saved at experiment folder!")

                    # Saving checkpoint of model
                    if current_epoch % train_cfg.save_model_every_epoch == 0:
                        saver.save(session, output_checkpoint, global_step=current_epoch,
                                   latest_filename='checkpoint')
                        saver_encoder.save(session, output_checkpoint_encoder,
                                           global_step=current_epoch, latest_filename='checkpoint_encoder')
                        print("Model (epoch={}) saved in file: {}".format(
                            current_epoch, output_checkpoint))

                    session.run([increment_global_epoch])

            except tf.errors.OutOfRangeError:
                print('Training stopped, input queue is empty.')

            # If need to be restored, will restore from this point
            current_epoch = session.run(global_epoch)

            # Save the variables to disk.
            save_path_1 = saver.save(session, output_checkpoint, global_step=current_epoch)
            save_path_2 = saver_encoder.save(
                session, output_checkpoint_encoder, global_step=current_epoch)
            print("Model saved in files: {} and {}".format(save_path_1, save_path_2))

            # Saving statistics for all epochs
            file_utils.save("{}training_metrics_per_epoch.pkl".format(
                output_dir), training_metrics_per_epoch)
            # myutils.save("{}val_metrics_per_epoch.pkl".format(self.options['dir/models']), val_metrics_per_epoch)
            print("Metrics statistics per epoch saved at experiment folder!")

    def val(self, val_cfg, validation_dataset, nb_documents, batch_size, output_checkpoint, output_dir, meteor_dir, tag="val"):
        """
        Performs inference of the model which can include both validation and test phases.
        :param validation_dataset:
        :param nb_documents:
        :param batch_size:
        :param output_checkpoint:
        :return:
        """
        validation_iterator = validation_dataset.make_one_shot_iterator()
        next_element_val = validation_iterator.get_next()

        nb_batch_per_epoch = (nb_documents // batch_size) + \
            (1 if nb_documents % batch_size > 0 else 0)
        print("batch_size={}, nb_batch_per_epoch={}, epochs=1".format(
            batch_size, nb_batch_per_epoch))

        # Create global step (for epochs), to be used when we restore the model to continue training
        # global_epoch = tf.Variable(1, trainable=False, name='global_epoch')
        # increment_global_epoch = tf.assign_add(global_epoch, 1, name='increment_global_epoch')

        # Add ops to save and restore all the variables, and a saver to only restore the variables of the encoder
        saver = tf.train.Saver(max_to_keep=2, keep_checkpoint_every_n_hours=12)
        # List of the metrics. Note that will be a list of dictionaries, each dictionary corresponding to one epoch.
        val_metrics_per_epoch = []
        predicted_sequences, target_sequences = [], []

        config = tf.ConfigProto()
        # config.log_device_placement = True
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as session:
            if not graph_utils.restore_model(saver, session, output_checkpoint, output_dir, latest_filename="checkpoint"):
                return

            # Statistics of the metrics
            val_metrics_per_epoch = file_utils.load(
                "{}{}_metrics_per_epoch.pkl".format(output_dir, tag))

            # Ops to write logs to Tensorboard
            summary_writer = tf.summary.FileWriter(output_dir, graph=tf.get_default_graph())
            writer_val = tf.summary.FileWriter(output_dir + 'plot_{}'.format(tag))

            try:
                max_sequence = 0
                losses = []
                session.run(self.metrics_init_op)

                print(f"Start {tag.upper()} phase...")

                for current_batch in range(1, nb_batch_per_epoch + 1):
                    # Generates a dictionary {"sequence_length: x, "tokens": [1, 45, etc]"}
                    batch_text = session.run(next_element_val)
                    max_seq_batch = np.amax(batch_text["sequence_length"])
                    # Dictionaries for feed_dict
                    dict_val_encoder = {self.encoder_inputs: batch_text["tokens"]}

                    # Run operation to get loss
                    summary, loss, predicted_tokens, target_tokens = \
                        session.run([self.merged_summary_op, self.masked_loss,
                                     self.predicted_tokens, self.decoder_expected_outputs],
                                    feed_dict=dict_val_encoder)
                    losses.append(loss)

                    # Update metrics that are calculate for each epoch. Get values per video to generate final statistics per epoch
                    session.run(self.update_metrics_op, feed_dict=dict_val_encoder)

                    if max_seq_batch > max_sequence:
                        max_sequence = max_seq_batch

                    # predicted_words = list(map(lambda id: self.reversed_word_dict.get(id), predicted_tokens[id_sample]))
                    # target_words = list(map(lambda id: self.reversed_word_dict.get(id), target_tokens[id_sample]))
                    predicted_words = list(map(lambda words: list(map(
                        lambda word_id: self.reversed_word_dict.get(word_id), words)), predicted_tokens))
                    target_words = list(map(lambda words: list(
                        map(lambda word_id: self.reversed_word_dict.get(word_id), words)), target_tokens))

                    predicted_sequences += pos_process_tokens(predicted_words)
                    target_sequences += pos_process_tokens(target_words)

                    if current_batch % val_cfg.display_loss_every_step == 0:
                        print(f"Batch={current_batch}/{nb_batch_per_epoch}, loss: {np.mean(losses):.2f}, max_seq={max_sequence}")

                        # Lets see which words were predicted
                        id_sample = randint(0, batch_text["tokens"].shape[0] - 1)
                        print(f"Predicted words ({len(predicted_words[id_sample])}): {predicted_words[id_sample]}\n"
                              f"Target words ({len(target_words[id_sample])}): {target_words[id_sample]}")

                    if current_batch % val_cfg.save_summary_every_step == 0:
                        summary = session.run(self.merged_summary_op, feed_dict=dict_val_encoder)
                        summary_writer.add_summary_for_block(summary, 1)

                # Saving predicted and target sequences
                file_utils.save_list(f"{output_dir}{tag}_predicted_sequences.txt", some_list=predicted_sequences)
                file_utils.save_list(f"{output_dir}{tag}_target_sequences.txt", some_list=target_sequences)

                # Calculate the METEOR metric for the predictions
                score_meteor = Meteor(meteor_path=meteor_dir,
                                      file_hypothesis=f"{output_dir}{tag}_predicted_sequences.txt",
                                      file_reference=f"{output_dir}{tag}_target_sequences.txt").score

                validation_metrics = self.write_metrics_per_epoch(session, writer_val, 1)
                validation_metrics = misc.merge_dicts(
                    validation_metrics, {'meteor': score_meteor})

                print("validation_metrics={}".format(validation_metrics))
                val_metrics_per_epoch.append(validation_metrics.copy())

                # Saving statistics for all epochs
                file_utils.save(f"{output_dir}{tag}_metrics_per_epoch.pkl", val_metrics_per_epoch)
                print("Metrics statistics per epoch saved at experiment folder!")

            except tf.errors.OutOfRangeError as out_error:
                print(out_error)
                print('Validation stopped, input queue is empty.')
