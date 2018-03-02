import tensorflow as tf
import numpy as np

import utils
import time


class RNNLanguageModel(object):

    def __init__(self, config):

        self.midi_dir = config['midi_dir']
        self.generated_dir = config['generated_dir']
        self.model_checkpoint_dir = config['model_checkpoint_dir']
        self.logs_dir = config['logs_dir']

        self.prev_n_tokens = config['prev_n_tokens']

        self.n_epochs = config['n_epochs']
        self.learning_rate = config['learning_rate']
        self.batch_size = config['batch_size']
        self.rnn_size = config['rnn_size']
        self.num_layers = config['num_layers']
        self.dropout_keep = config['dropout_keep']
        self.vocab_size = config['vocab_size']
        self.n_steps = config['n_steps']  # Number of time steps of future prediction

        self.build_model()

    def build_model(self):
        print("[INFO] Building Model ...")
        self.init_placeholders()
        self.build_encoder()
        self.output_layer()
        self.calculate_loss()
        self.optimization()
        self.summary()

    def init_placeholders(self):
        with tf.name_scope('model_inputs'):
            self.inputs_data = tf.placeholder(dtype=tf.int32,
                                         shape=[self.batch_size, self.prev_n_tokens],
                                         name='input_sequence'
                                         )

            self.output_value = tf.placeholder(dtype=tf.int32,
                                          shape=[self.batch_size, ],
                                          name='output_value'
                                          )

            self.lr = tf.placeholder(dtype=tf.float32,
                                shape=(),
                                name='learning_rate'
                                )

            self.source_sequence_length = tf.placeholder(dtype=tf.int32,
                                                    shape=[self.batch_size, ],
                                                    name='source_sequence_length'
                                                    )

    def build_encoder(self):
        with tf.name_scope('one_hot_layer'):
            self.one_hot_input = tf.one_hot(indices=self.inputs_data,
                                       depth=self.vocab_size,
                                       name='one_hot_input'
                                       )

        with tf.name_scope('encoder_rnn'):
            for layer in range(self.num_layers):
                with tf.variable_scope('encoder_{}'.format(layer + 1)):
                    cell = tf.contrib.rnn.LayerNormBasicLSTMCell(num_units=self.rnn_size)
                    cell = tf.contrib.rnn.DropoutWrapper(cell=cell,
                                                         input_keep_prob=self.dropout_keep)

                    enc_outputs, enc_states = tf.nn.dynamic_rnn(cell=cell,
                                                                inputs=self.one_hot_input,
                                                                sequence_length=self.source_sequence_length,
                                                                dtype=tf.float32)

                    self.h_N = tf.identity(input=enc_states[1],
                                      name='final_output')

    def output_layer(self):
        with tf.name_scope('output_layer'):
            transformed_output = tf.layers.dense(inputs=self.h_N,
                                                 units=self.rnn_size,  # Keep the same dimension as the LSTM dimension
                                                 name='transformation_layer')
            self.output_logits = tf.layers.dense(inputs=transformed_output,
                                            units=self.vocab_size,
                                            name='final_layer')
            self.output_pred = tf.argmax(input=self.output_logits,
                                    axis=-1,
                                    name='output_pred')

    def calculate_loss(self):
        xent_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.output_value,
                                                                   logits=self.output_logits
                                                                   )
        self.cost = tf.reduce_mean(xent_loss)

    def optimization(self):
        optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr)
        var_list = tf.trainable_variables()
        gradients = optimizer.compute_gradients(loss=self.cost,
                                                var_list=var_list)
        capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
        self.train_op = optimizer.apply_gradients(capped_gradients)

    def summary(self):
        with tf.name_scope('summaries'):
            tf.summary.scalar(name='training_loss', tensor=self.cost)
            self.summary_op = tf.summary.merge_all()

    def train(self, X, y):
        print("[INFO] Begin training process ...")

        checkpoint = self.model_checkpoint_dir

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            writer = tf.summary.FileWriter(logdir=self.logs_dir,
                                           graph=sess.graph)

            iter_i = 0

            for epoch_i in range(1, self.n_epochs + 1):

                start_time = time.time()

                for batch_i, (batch_input, batch_output, source_seq_len) in enumerate(utils.get_batches(X, y, self.batch_size, self.prev_n_tokens)):
                    loss_, op_, summary_ = sess.run(fetches=[self.cost, self.train_op, self.summary_op],
                                                    feed_dict={
                                                        self.inputs_data: batch_input,
                                                        self.output_value: batch_output,
                                                        self.source_sequence_length: source_seq_len,
                                                        self.lr: self.learning_rate
                                                    }
                                                    )
                    iter_i += 1

                    writer.add_summary(summary=summary_,
                                       global_step=iter_i)

                saver = tf.train.Saver()
                saver.save(sess=sess,
                           save_path=checkpoint + str(epoch_i) + ".ckpt")

                end_time = time.time()

                print('Epoch: {} \t Loss: {} \t Time: {}'.format(epoch_i, round(loss_, 2),
                                                                 round(end_time - start_time, 2)))

    def generate_midi_array(self, X, checkpoint, sample=True):

        test_indices = np.random.choice(a=range(len(X)), size=self.batch_size, replace=False)
        test_input = X[test_indices, :]

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(sess, checkpoint)

            test_out = np.ones(shape=(self.batch_size, 1))

            for step in range(self.n_steps):
                y_pred = sess.run(fetches=self.output_logits,
                                  feed_dict={self.inputs_data: test_input,
                                             self.source_sequence_length: [self.prev_n_tokens] * self.batch_size})

                if sample:
                    y_pred = utils.sample(y_pred)
                else:
                    y_pred = np.argmax(y_pred, axis = 1)

                # Re-determine the test_input:  test_input[1 to 50] + [y_pred]
                y_pred = y_pred.reshape((self.batch_size, 1))
                test_input = np.concatenate([test_input[:, 1:], y_pred], axis=-1)
                test_out = np.hstack((test_out, y_pred))

            test_out = test_out[:, 1:].astype('int32')

        return test_out


