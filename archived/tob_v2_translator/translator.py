# import future
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# import module
import time
import os
import argparse
import sys

# import file from the directory
import config
import model

# renaming module
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from data import DataHandler

# Enable easier debugging and cleaner terminal
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.logging.set_verbosity(tf.logging.ERROR)


class Translator:

    class Mode:
        '''
        Simple structure representing the different testing modes
        '''
        TRANSLATE = 'translate'
        TRAIN = 'train'

    def __init__(self):
        '''
        '''
        # Parameters
        self.args = None

        # Task specific object
        self.data = None   # DataHandler

        # encoder, decoder, optimizer object from model
        self.encoder = None
        self.decoder = None
        self.optimizer = None

        # Params for training
        self.BUFFER_SIZE = 0
        self.steps_per_epoch = 0
        self.dataset = None
        self.Batch = None  # Batch

        # Tensorflow utilities for convenience saving/logging
        # self.writer = None
        self.checkpoint = None
        self.model_dir = ''  # Where the model is saved
        self.glob_step = 0  # Represent the number of iteration

        # TensorFlow main session (we keep track for the daemon)
        # self.sess = None

        # File name and constant
        self.model_save_dir = ''
        self.MODEL_NAME = ''
        self.CKPT_EXT = ''
        self.ckpt_prefix = ''

    @staticmethod
    def parseArgs(args):
        '''
        Parse the arguments from the given command line
        Args:
            args (list<str>): List of arguments to parse. If None, the default sys.argv will be parsed
        '''
        parser = argparse.ArgumentParser()

        # Global options
        globalArgs = parser.add_argument_group('Global options')
        globalArgs.add_argument('--mode',
                                nargs='?',
                                choices=[Translator.Mode.TRAIN,
                                         Translator.Mode.TRANSLATE],
                                default=Translator.Mode.TRAIN,
                                help='mode: Train/Translate. default: Train'
                                )
        globalArgs.add_argument('--model_tag',
                                type=str,
                                default=None,
                                help='tag to differentiate model to store/load'
                                )
        globalArgs.add_argument('--reset',
                                action='store_true',
                                help='remove everything in model directory'
                                )

        return parser.parse_args(args)

    def main(self, args=None):
        '''
        Launch either train or the translate mode
        '''
        # Tensorflow version check
        print('Tensorflow detected: v{}'.format(tf.__version__))

        # General initialization
        self.args = self.parseArgs(args)

        # callback to DataHandler
        self.data = DataHandler()

        # set checkpoint prefix
        self._get_ckpt_prefix()

        # create BUFFER_SIZE and init steps_per_epoch
        self.BUFFER_SIZE = len(self.data.input_tensor_train)
        self.steps_per_epoch = self.BUFFER_SIZE // config.BATCH_SIZE

        # create Dataset
        self.dataset = tf.data.Dataset.from_tensor_slices(
            (self.data.input_tensor_train, self.data.target_tensor_train)).shuffle(self.BUFFER_SIZE)
        self.Batch = self.dataset.batch(config.BATCH_SIZE, drop_remainder=True)

        # Prepare the model
        with tf.device('/device:GPU:0'):
            self.encoder = model.Encoder(self.data.vocab_in_size,
                                         config.EMBEDDING_DIMENSION,
                                         config.UNITS,
                                         config.BATCH_SIZE)
            self.decoder = model.Decoder(self.data.vocab_out_size,
                                         config.EMBEDDING_DIMENSION,
                                         config.UNITS,
                                         config.BATCH_SIZE)
            self.optimizer = model.optimizer

        # Saver/summaries
        # self.writer = tf.summary.FileWriter(self._get_summary_name())
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer,
                                              encoder=self.encoder,
                                              decoder=self.decoder)

        print('Initialize variables...')

        # Modes: Training / Testing
        if self.args.mode:
            if self.args.mode == Translator.Mode.TRANSLATE:
                # Restore models from the closest checkpoint
                # self.manage_previous_model(self.sess)
                while True:
                    line = self._get_user_input()
                    if len(line) > 0 and line[-1] == '\n':
                        line = line[:-1]
                    if line == '':
                        break
                    self.translate(line)
        else:
            self.train()

    def translate(self, sentence):
        result, sentence, attention_plot = self.evaluate(sentence)
        print('Input: %s' % (sentence))
        print('Predicted translation: {}'.format(result))
        # Uses to draw attention diagram
        # attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]
        # data.plot_attention(attention_plot, sentence.split(' '), result.split(' '))

        def evaluate(self, sentence):
            '''
            Evaluate and format the output for response
            args:
                sentence: input
            returns:
                str(result): output predicted string
                str(sentence): correct output
                attention_plot: attention weight for graphing
            '''
            attention_plot = np.zeros((self.data.max_length_target,
                                       self.data.max_length_input))

            sentence = self.data.preprocess_sentence(sentence)
            inputs = [self.data.input_language.word2idx[i]
                      for i in sentence.split(' ')]
            inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                                   maxlen=self.data.max_length_input,
                                                                   padding='post')
            inputs = tf.convert_to_tensor(inputs)

            result = ''

            hidden = [tf.zeros((1, config.UNITS))]
            enc_out, enc_hidden = self.encoder(inputs, hidden)

            dec_hidden = enc_hidden
            dec_input = tf.expand_dims(
                [self.data.target_language.word2idx['<start>']], 0)

            for t in range(self.data.max_length_target):
                predictions, dec_hidden, attention_weights = self.decoder(dec_input,
                                                                          dec_hidden,
                                                                          enc_out)

                # storing the attention weights to plot later on
                attention_weights = tf.reshape(attention_weights, (-1, ))
                attention_plot[t] = attention_weights.numpy()

                predicted_id = tf.argmax(predictions[0]).numpy()

                result += self.data.target_language.word2idx[predicted_id] + ' '

                if self.data.target_language.word2idx[predicted_id] == '<end>':
                    return result, sentence, attention_plot

                # the predicted ID is fed back into the model
                dec_input = tf.expand_dims([predicted_id], 0)

            return result, sentence, attention_plot

    def train(self):
        '''
        Training loop
        Args:
            sess: The current running session
        '''
        # Define the summary operator (Won't appear on the tensorboard graph)
        # mergedSummaries = tf.summary.merge_all()
        # if self.globStep == 0:  # Not restoring from previous run
        #    self.writer.add_graph(sess.graph)  # First time only
        print('Start training (press Ctrl+C to save and exit)...')
        try:  # If the user exit while training, try to save the model
            for epoch in range(config.EPOCHS):
                print()
                print('----- Epoch {}/{} ; (lr={}) -----'.format(epoch + 1,
                                                                 config.EPOCHS,
                                                                 config.LEARNING_RATE))
                start = time.time()

                enc_hidden = self.encoder.initialize_hidden_state()
                total_loss = 0

                batches = self.Batch.take(self.steps_per_epoch)
                for (batch, (inp, targ)) in tqdm(enumerate(batches), desc='Training'):
                    batch_loss = self.train_step(inp, targ, enc_hidden)
                    # _, loss, summary = sess.run(self.optimizer + (mergedSummaries))
                    # self.writer.add_summary(summary, self.glob_step)
                    self.glob_step += 1
                    total_loss += batch_loss

                    # Output training status
                    if self.glob_step % 100 == 0:
                        tqdm.write('----- Step %d -- Batch %.2f -- Loss %.2f' %
                                   (self.glob_step, batch, batch_loss.numpy()))
                    # Checkpoint
                    if self.glob_step % self.save_every == 0:
                        self.checkpoint.save(file_prefix=self.ckpt_prefix)
                        # self._save_session(sess)

                toc = time.time()
                tqdm.write('----- Epoch {} -- Loss {:.4f}'.format(epoch + 1,
                                                                  total_loss / self.steps_per_epoch))
                print('Epoch finished in {}'.format(toc - start))
        except (KeyboardInterrupt, SystemExit):
            print('Interruption detected, exiting the program...')
        # self._saveSession(sess)  # Ultimate saving before complete exit

        def train_step(self, input, target, enc_hidden):
            loss = 0

            with tf.GradientTape() as tape:
                enc_output, enc_hidden = self.encoder(input, enc_hidden)

                dec_hidden = enc_hidden

                dec_input = tf.expand_dims(
                    [self.data.target_language.word2idx['<start>']] * config.BATCH_SIZE, 1)

                # Teacher forcing - feeding the target as the next input
                for t in range(1, target.shape[1]):
                    # passing enc_output to the decoder
                    predictions, dec_hidden, _ = self.decoder(
                        dec_input, dec_hidden, enc_output)

                    loss += model.loss_function(target[:, t], predictions)

                    # using teacher forcing
                    dec_input = tf.expand_dims(target[:, t], 1)

            batch_loss = (loss / int(target.shape[1]))

            variables = self.encoder.trainable_variables + self.decoder.trainable_variables

            gradients = tape.gradient(loss, variables)

            self.optimizer.apply_gradients(zip(gradients, variables))

            return batch_loss

    def _get_ckpt_prefix(self):
        if self.args.model_tag:
            self.model_save_dir = config.SAVE_PATH + os.sep + 'model'
            self.MODEL_NAME = self.args.model_tag
            self.CKPT_EXT = '.ckpt'
            self.ckpt_prefix = os.path.join(self.model_save_dir,
                                            self.MODEL_NAME,
                                            self.CKPT_EXT)
        return self.ckpt_prefix

    def _get_user_input():
        '''
        Get user's input, which will be transformed into encoder input later
        '''
        print('> ', end='')
        sys.stdout.flush()
        return sys.stdin.readline()
