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
import data

# renaming module
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from data import DataHandler
from model import NMT

# Enable easier debugging and cleaner terminal
tf.enable_eager_execution()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.logging.set_verbosity(tf.logging.ERROR)


class Translator:

    def __init__(self):
        # Model Params
        self.args = None
        # DataHandler
        self.dataHandler = None
        # encoder, decoder, optimizer
        self.encoder = None
        self.decoder = None
        self.optimizer = None

        # Input/Output tensors
        self.input_tensor_train = []
        self.input_tensor_validation = []
        self.target_tensor_train = []
        self.target_tensor_validation = []

        # Params for training
        self.BUFFER_SIZE = 0
        self.steps_per_epoch = 0
        self.dataset = None
        # create batch
        self.Batch = None
        # Tensorflow utilities for convenience saving/logging
        self.writer = None
        self.saver = None
        # Where the model is saved
        self.model_dir = ''
        # Represent the number of iteration for the current model
        self.glob_step = 0
        self.save_every = 0
        # TensorFlow main session (we keep track for the daemon)
        self.sess = None

        # Directory for checkpoint
        self.model_dir_base = config.SAVE_PATH + os.sep + 'model'
        self.MODEL_NAME_BASE = 'model'
        self.MODEL_EXT = '.ckpt'
        self.keep_all = config.KEEP

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
        globalArgs.add_argument('--model_tag', type=str, default=None,
                                help='tag to differentiate which model to store/load')
        globalArgs.add_argument('--reset', action='store_true',
                                help='use this if you want to ignore the previous model present on the model directory (Warning: the model will be destroyed with all the folder content)')

        return parser.parse_args(args)

    def main(self, args=None):
        # Tensorflow version check
        print('Tensorflow detected: v{}'.format(tf.__version__))

        # General init
        self.args = self.parseArgs(args)

        # callback to DataHandler
        self.dataHandler = DataHandler()
        self.dataHandler.load_data(data.data_file)
        # Creating training and validation sets using an 80-20 split
        self.input_tensor_train, self.input_tensor_validation, self.target_tensor_train, self.target_tensor_validation = self.dataHandler.split_set()

        # create BUFFER_SIZE and init steps_per_epoch
        self.BUFFER_SIZE = len(self.input_tensor_train)
        self.steps_per_epoch = self.BUFFER_SIZE // config.BATCH_SIZE

        # create Dataset
        self.dataset = tf.data.Dataset.from_tensor_slices(
            (self.input_tensor_train, self.target_tensor_train)).shuffle(self.BUFFER_SIZE)
        self.Batch = self.dataset.batch(config.BATCH_SIZE, drop_remainder=True)

        # Prepare the model
        with tf.device('/device:GPU:0'):
            self.encoder, self.decoder, self.optimizer = NMT(
                self.dataHandler).buildNet()

        # Saver/summaries
        self.save_every = config.SAVE_EVERY
        self.writer = tf.summary.FileWriter(self._getSummaryName())
        self.saver = tf.train.Checkpoint(optimizer=self.optimizer,
                                         encoder=self.encoder,
                                         decoder=self.decoder)

        # Running session
        self.sess = tf.Session()
        print('Initialize variables...')
        self.sess.run(tf.global_variables_initializer())

        # Modes: Training / Testing
        inp = input('To train the model, type train. To test the model, type translate. Ctrl+C to exit: ')
        try:
            if inp == 'train':
                self.train(self.sess)
            elif inp == 'translate':
                # Restore models from the closest checkpoint
                self.manage_previous_model(self.sess)
                while True:
                    line = self._get_user_input()
                    if len(line) > 0 and line[-1] == '\n':
                        line = line[:-1]
                    if line == '':
                        break
                    self.translate(line)
        except KeyboardInterrupt:
            pass

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
        attention_plot = np.zeros(
            (data.max_length_target, data.max_length_input))

        sentence = self.dataHandler.preprocess_sentence(sentence)
        inputs = [data.input_language.word_index[i]
                  for i in sentence.split(' ')]
        inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                               maxlen=data.max_length_input,
                                                               padding='post')
        inputs = tf.convert_to_tensor(inputs)

        result = ''

        hidden = [tf.zeros((1, config.UNITS))]
        enc_out, enc_hidden = self.encoder(inputs, hidden)

        dec_hidden = enc_hidden
        dec_input = tf.expand_dims(
            [data.target_language.word_index['<start>']], 0)

        for t in range(data.max_length_target):
            predictions, dec_hidden, attention_weights = self.decoder(dec_input,
                                                                      dec_hidden,
                                                                      enc_out)

            # storing the attention weights to plot later on
            attention_weights = tf.reshape(attention_weights, (-1, ))
            attention_plot[t] = attention_weights.numpy()

            predicted_id = tf.argmax(predictions[0]).numpy()

            result += data.target_language.index_word[predicted_id] + ' '

            if data.target_language.index_word[predicted_id] == '<end>':
                return result, sentence, attention_plot

            # the predicted ID is fed back into the model
            dec_input = tf.expand_dims([predicted_id], 0)

        return result, sentence, attention_plot

    def translate(self, sentence):
        result, sentence, attention_plot = self.evaluate(sentence)
        print('Input: %s' % (sentence))
        print('Predicted translation: {}'.format(result))
        # Uses to draw attention diagram
        # attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]
        # data.plot_attention(attention_plot, sentence.split(' '), result.split(' '))

    def train(self, sess):
        '''
        Training loop
        Args:
            sess: The current running session
        '''
        # Define the summary operator (Won't appear on the tensorboard graph)
        mergedSummaries = tf.summary.merge_all()
        if self.globStep == 0:  # Not restoring from previous run
            self.writer.add_graph(sess.graph)  # First time only
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
                    _, loss, summary = sess.run(
                        self.optimizer + (mergedSummaries))
                    self.writer.add_summary(summary, self.glob_step)
                    self.glob_step += 1
                    total_loss += batch_loss

                    # Output training status
                    if self.glob_step % 100 == 0:
                        tqdm.write('----- Step %d -- Batch %.2f -- Loss %.2f' %
                                   (self.glob_step, batch, batch_loss.numpy()))
                    # Checkpoint
                    if self.glob_step % self.save_every == 0:
                        self._save_session(sess)

                toc = time.time()
                tqdm.write('----- Epoch {} -- Loss {:.4f}'.format(epoch + 1,
                                                                  total_loss / self.steps_per_epoch))
                print('Epoch finished in {}'.format(toc - start))
        except (KeyboardInterrupt, SystemExit):
            print('Interruption detected, exiting the program...')
        self._saveSession(sess)  # Ultimate saving before complete exit

    def train_step(self, input, target, enc_hidden):
        loss = 0

        with tf.GradientTape() as tape:
            enc_output, enc_hidden = self.encoder(input, enc_hidden)

            dec_hidden = enc_hidden

            dec_input = tf.expand_dims(
                [data.target_language.word_index['<start>']] * config.BATCH_SIZE, 1)

            # Teacher forcing - feeding the target as the next input
            for t in range(1, target.shape[1]):
                # passing enc_output to the decoder
                predictions, dec_hidden, _ = self.decoder(
                    dec_input, dec_hidden, enc_output)

                loss += NMT.loss_function(target[:, t], predictions)

                # using teacher forcing
                dec_input = tf.expand_dims(target[:, t], 1)

        batch_loss = (loss / int(target.shape[1]))

        variables = self.encoder.trainable_variables + self.decoder.trainable_variables

        gradients = tape.gradient(loss, variables)

        self.optimizer.apply_gradients(zip(gradients, variables))

        return batch_loss

    def manage_previous_model(self, sess):
        '''
        Restore or reset the model, depending of the parameters
        If the destination directory already contains some file, it will handle the conflict as following:
         * If --reset is set, all present files will be removed (warning: no confirmation is asked) and the training
         restart from scratch (globStep & cie reinitialized)
         * Otherwise, it will depend of the directory content. If the directory contains:
           * No model files (only summary logs): works as a reset (restart from scratch)
           * Other model files, but modelName not found (surely keepAll option changed): raise error, the user should
           decide by himself what to do
           * The right model file (eventually some other): no problem, simply resume the training
        In any case, the directory will exist as it has been created by the summary writer
        Args:
            sess: The current running session
        '''

        print('WARNING: ', end='')

        modelName = self._get_model_name()

        if os.listdir(self.model_dir):
            if self.args.reset:
                print('Reset: Destroying previous model at {}'.format(self.model_dir))
            # Analysing directory content
            elif os.path.exists(modelName):  # Restore the model
                print('Restoring previous model from {}'.format(modelName))
                # Will crash when --reset is not activated and the model has not been saved yet
                self.saver.restore(modelName)
            elif self._get_model_list():
                print('Conflict with previous models.')
                raise RuntimeError(
                    'Some models are already present in \'{}\'. You should delete all checkpoints manually'.format(self.modelDir))
            else:  # No other model to conflict with (probably summary files)
                print('No previous model found, but some files found at {}. Cleaning...'.format(
                    self.model_dir))  # Warning: No confirmation asked
                self.args.reset = True

            if self.args.reset:
                fileList = [os.path.join(self.model_dir, f)
                            for f in os.listdir(self.model_dir)]
                for f in fileList:
                    print('Removing {}'.format(f))
                    os.remove(f)

        else:
            print('No previous model found, starting from clean directory: {}'.format(
                self.model_dir))

    def _get_user_input():
        '''
        Get user's input, which will be transformed into encoder input later
        '''
        print('> ', end='')
        sys.stdout.flush()
        return sys.stdin.readline()

    def _save_session(self, sess):
        ''' Save the model parameters and the variables
        Args:
            sess: the current session
        '''
        tqdm.write('Checkpoint reached: saving model (don\'t stop the run)...')
        model_name = self._get_model_name()
        # NOTE: Simulate the old model existance to avoid rewriting the file parser
        with open(model_name, 'w') as f:
            f.write(
                'This file is used internally by the translator to check the model existance. Please do not remove.\n')
        self.saver.save(file_prefix=os.path.join(
            self._set_model_name(), self.MODEL_EXT), session=sess)
        tqdm.write('Model saved.')

    def _set_model_name(self):
        self.model_dir = os.path.join(os.getcwd(), self.model_dir_base)
        if self.args.model_tag:
            self.model_dir += '-' + self.args.model_tag
        return self.model_dir

    def _get_model_name(self):
        '''
        Parse the argument to decide were to save/load the model
        This function is called at each checkpoint and the first
        time the model is load.
        Return:
            str: The path and name were the model need to be saved
        '''
        model_name = os.path.join(self._set_model_name(), self.MODEL_NAME_BASE)
        if self.keep_all:
            # We don't erase previously saved model by including the current step on the name
            model_name += '-' + str(self.glob_step)
        return model_name + self.MODEL_EXT

    def _get_model_list(self):
        '''
        Return the list of the model files inside the model directory
        '''
        return [os.path.join(self.model_dir, f) for f in os.listdir(self.model_dir) if f.endswith(self.MODEL_EXT)]

    def _get_summary_name(self):
        '''
        Parse the argument to decide were to save the summary, at the same
        place that the model. The folder could already contain logs if we
        restore the training, those will be merged.
        Return:
            str: The path and name of the summary
        '''
        return self.model_dir
