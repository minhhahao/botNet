'''
    Descriptions:
        Main script to run the bot. Refers to README.md
'''
# import future
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# import module
import os
import logging  # Removing annoying tf warning
import argparse  # Parsing arguments to commandline
import configparser  # Writing config files
import tensorflow as tf
from packaging import version  # Testing versions
import datetime  # Logging purposes

# import modules from files
from chatbot.data import dataHandler
from chatbot.model import transformer
from chatbot.utils import _get_user_input, preprocess_sentence

# clean terminal view
logging.getLogger('tensorflow').disabled = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    '''
        Custom learning rate schedules
        Subclass of tf.keras.optimizers.schedules.LearningRateSchedule
        Learning rate can be calculated with tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
        with:
            args1 = tf.math.rsqrt(step)
            args2 = step * (self.warmup_steps**-1.5)
        Refers to [this paper](https://arxiv.org/pdf/1706.03762.pdf)
        args:
            d_model <int>: model dimensions
        returns:
            learning_rate <int>: Dynamic learning rate.
    '''
    def __init__(self, d_model, warmup_steps=4000):
        '''
        args:
            d_model <int>: model dimensions
        '''
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        '''
        args:
            step <int>: The amount of steps taken for training
        returns:
            learning_rate <int>; Dunamic learning rate
        '''
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

class botNet:
    '''
    Driver code. Users can choose between testing and interacting with trained model
    '''
    class Mode:
        '''
        Simple structure representing the different modes
        '''
        TRAIN = 'train'
        INTERACTIVE = 'interactive'
        DAEMON = 'daemon'

    def __init__(self):
        '''pretty self-explanatory
        '''
        # Model/dataset parameters
        self.args = None
        # Task-specific object
        self.process = None

        # Tensorflow utilities for convenience saving/logging
        self.tensorboard = None
        self.checkpoint = None

        # Constant dir, filename
        self.model_dir = ''
        self.log_dir = ''
        self.root_dir = os.path.dirname(os.path.abspath(__file__))
        self.LOG_DIRNAME = 'logs'
        self.MODEL_DIR_BASE = 'save' + os.sep + 'model'
        self.MODEL_NAME_BASE = 'model'
        self.MODEL_EXT = '.ckpt'
        self.CONFIG_FILENAME = 'params.ini'
        self.CONFIG_VERSION = '0.5'
        self.OUTPUT_FILE = 'output.txt'

    @staticmethod
    def parse_args(args):
        '''
        Parse the arguments from the given command line
        args:
            args (list<str>): List of arguments to parse. If None, the default sys.argv will be parsed
        '''
        parser = argparse.ArgumentParser()

        # Global options
        global_args = parser.add_argument_group('Global options')
        global_args.add_argument('--mode',
                                 nargs='?',
                                 choices=[botNet.Mode.TRAIN, botNet.Mode.INTERACTIVE, botNet.Mode.DAEMON],
                                 const=botNet.Mode.TRAIN, default=None,
                                 help='Train mode [Default]. Compile model following given parameters. \
                                       Interative mode to talk to the bot. \
                                       Daemon mode for running in background (Web app with Django)(Future development)')
        global_args.add_argument('--verbose', action='store_true', help='When training, print out all outputs')
        global_args.add_argument('--model_tag', type=str, default=None, help='tag to differentiate which model to store/load')
        global_args.add_argument('--continue_training', action='store_true', help='Continue training from saved weight')

        # Dataset options
        dataset_args = parser.add_argument_group('Dataset options')
        dataset_args.add_argument('--corpus', type=str, default='cornell', help='Corpus for trainning data. Adding more dataset (Future development)')
        dataset_args.add_argument('--max_samples', type=int, default=30000, help='Max samples for a dataset')
        dataset_args.add_argument('--vocab_size', type=int, default=2**13, help='Max size for vocab file')
        dataset_args.add_argument('--max_length', type=int, default=40, help='Max length for a sentence')
        dataset_args.add_argument('--buffer_size', type=int, default=20000, help='Amount of data for one buffer')
        dataset_args.add_argument('--batch_size', type=int, default=64, help='Size of a mini-batch')
        dataset_args.add_argument('--epochs', type=int, default=20, help='# of epochs for training')

        # Model parameters
        model_args = parser.add_argument_group('Model options')
        model_args.add_argument('--num_layers', type=int, default=2, help='Number of layers for the architecture')
        model_args.add_argument('--d_model', type=int, default=512, help='Model dimension')
        model_args.add_argument('--heads', type=int, default=8, help='# of parallel layers')
        model_args.add_argument('--units', type=int, default=512, help='The dimensions of output space for Dense layers')
        model_args.add_argument('--dropout', type=float, default=0.1, help='Dropout rate for normalization.')

        return parser.parse_args(args)

    def main(self, args=None):
        '''
        Launch train or interactive mode. (currently working with daemon)
        '''

        print('\nTensorflow detected: {}'.format(tf.__version__))
        assert version.parse(tf.__version__).release[0] >= 2, "Requirements: TensorFlow 2.0 or above."
        # Parsing args
        self.args = self.parse_args(args)
        # Data objects
        self.process = dataHandler(self.args)
        # Misc dir object
        self.model_dir, self.log_dir = self._get_model_dir()
        self.checkpoint = self._get_checkpoint()
        self.checkpoint_dir = os.path.dirname(self.checkpoint)

        # TODO: Fixes tensorboard
        # Tensorflow objects for logging purposes
        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.log_dir)
        self.checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(self.checkpoint,
                                                                      verbose=1,
                                                                      save_weights_only=True,
                                                                      period=5)
        # Either INTERACTIVE or TRAIN (default)
        if self.args.mode:
            if self.args.mode == botNet.Mode.INTERACTIVE:
                self.main_interactive()
            elif self.args.mode == botNet.Mode.DAEMON:
                self.main_daemon()
            else:
                raise RuntimeError('Unknown mode: {}'.format(self.args.mode))  # never going to happen unless some typo in the code segment
        else:
            self.main_train()

    def main_train(self):
        '''Training loop'''
        if self.args.continue_training:
            self.load_model_params()
        else:
            self.save_model_params()
        print('Creating models...')
        model_train = self.model()
        print('\nModel summary: ')
        model_train.summary()
        # Train the model and save checkpoint
        try:
            if self.args.continue_training:
                print('\nStart from saved weights (press Ctrl+C to save and exit)...\n')
                model_train.load_weights(tf.train.latest_checkpoint(self.checkpoint_dir))
                model_train.fit(self.process.dataset,
                                epochs=self.args.epochs,
                                callbacks=[self.checkpoint_callback, self.tensorboard_callback])
                model_train.save_weights(self.checkpoint)
                print('\nFinished training.')
            else:
                print('\nStart training (press Ctrl+C to save and exit)...\n')
                model_train.fit(self.process.dataset,
                                epochs=self.args.epochs,
                                callbacks=[self.checkpoint_callback, self.tensorboard_callback])
                model_train.save_weights(self.checkpoint)
                print('\nFinished training.')
        except (KeyboardInterrupt, SystemExit):
            print('Interruption detected, exiting the program...')
        del model_train

    def main_interactive(self):
        '''Interacting loop'''
        def predict(model_t, sentence):
            def evaluate(model_t, sentence):
                sentence = preprocess_sentence(sentence)
                sentence = tf.expand_dims(self.process.START_TOKEN + self.process.tokenizer.encode(sentence) + self.process.END_TOKEN, axis=0)

                output = tf.expand_dims(self.process.START_TOKEN, 0)

                for i in range(self.args.max_length):
                    predictions = model_t(inputs=[sentence, output], training=False)
                    # select the last word from the seq_len dimension
                    predictions = predictions[:, -1:, :]
                    predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
                    # return the result if the predicted_id is equal to the end token
                    if tf.equal(predicted_id, self.process.END_TOKEN[0]):
                        break
                    # concatenated the predicted_id to the output which is given
                    # to the decoder as its input.
                    output = tf.concat([output, predicted_id], axis=-1)
                return tf.squeeze(output, axis=0)

            prediction = evaluate(model_t, sentence)
            predicted_sentence = self.process.tokenizer.decode(
                [i for i in prediction if i < self.process.tokenizer.vocab_size])
            print('Output: {}'.format(predicted_sentence))
            return predicted_sentence

        model_test = self.model()
        model_test.load_weights(tf.train.latest_checkpoint(self.checkpoint_dir))
        with open(os.path.join(self.process.DATA_PATH, self.process.VOCAB_PATH, self.OUTPUT_FILE), 'a+') as output_file:
            try:
                output_file.write('\n=============================================\n')
                output_file.write(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
                while True:
                    line = _get_user_input()
                    if len(line) > 0 and line[-1] == '\n':
                        line = line[:-1]
                    if line == '':
                        break
                    output_file.write('\nInput: {}\n'.format(line))
                    output_file.write('Output: {}\n'.format(predict(model_test, line)))
                output_file.write('\n=============================================\n')
                output_file.close()
            except KeyboardInterrupt:
                print('\nTerminated')
        del model_test

    def main_daemon(self):
        pass

    def model(self):
        '''Generates models'''
        # Custom params following the paper
        learning_rate = CustomSchedule(self.args.d_model)
        optimizer = tf.keras.optimizers.Adam(learning_rate,
                                             beta_1=0.9,
                                             beta_2=0.98,
                                             epsilon=1e-9)

        def loss_function(y_true, y_pred):
            y_true = tf.reshape(y_true, shape=(-1, self.args.max_length - 1))
            loss = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True, reduction='none')(y_true, y_pred)
            mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
            loss = tf.multiply(loss, mask)
            return tf.reduce_mean(loss)

        def accuracy(y_true, y_pred):
            # ensure labels have shape (batch_size, MAX_LENGTH - 1)
            y_true = tf.reshape(y_true, shape=(-1, self.args.max_length - 1))
            accuracy = tf.metrics.SparseCategoricalAccuracy()(y_true, y_pred)
            return accuracy

        # Create model
        created_model = transformer(vocab_size=self.args.vocab_size,
                                    num_layers=self.args.num_layers,
                                    units=self.args.units,
                                    d_model=self.args.d_model,
                                    num_heads=self.args.heads,
                                    dropout=self.args.dropout)
        created_model.compile(optimizer=optimizer,
                              loss=loss_function,
                              metrics=[accuracy])
        return created_model

    def _get_model_dir(self):
        '''Get model and log directory'''
        self.model_dir = os.path.join(self.root_dir, self.MODEL_DIR_BASE)
        if self.args.model_tag:
            self.model_dir += '-' + self.args.model_tag
            self.log_dir = os.path.join(self.model_dir, self.LOG_DIRNAME)
            self.log_dir += os.sep + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + os.sep
        return self.model_dir, self.log_dir

    def _get_checkpoint(self):
        '''Get model checkpoint (save weights only)'''
        model_ckpt = os.path.join(self.model_dir, self.MODEL_NAME_BASE)
        return model_ckpt + self.MODEL_EXT

    def save_model_params(self):
        '''
        Save the params of the model, like the current globStep value
        Warning: if you modify this function, make sure the changes mirror load_model_params
        '''
        config = configparser.ConfigParser()
        config['Global'] = {}
        config['Global']['version'] = self.CONFIG_VERSION
        config['Global']['corpus'] = str(self.args.corpus)

        config['Dataset'] = {}
        config['Dataset']['max_samples'] = str(self.args.max_samples)
        config['Dataset']['vocab_size'] = str(self.args.vocab_size)
        config['Dataset']['max_length'] = str(self.args.max_length)
        config['Dataset']['buffer_size'] = str(self.args.buffer_size)
        config['Dataset']['epochs'] = str(self.args.epochs)

        config['Model'] = {}
        config['Model']['num_layers'] = str(self.args.num_layers)
        config['Model']['d_model'] = str(self.args.d_model)
        config['Model']['heads'] = str(self.args.heads)
        config['Model']['units'] = str(self.args.units)

        # Keep track of the learning params (but without restoring them)
        config['Training (won\'t be restored)'] = {}
        config['Training (won\'t be restored)']['batch_size'] = str(self.args.batch_size)
        config['Training (won\'t be restored)']['dropout'] = str(self.args.dropout)

        with open(os.path.join(self.model_dir, self.CONFIG_FILENAME), 'w') as config_file:
            config.write(config_file)

    def load_model_params(self):
        '''Load previous model params from save config file'''
        config_name = os.path.join(self.model_dir, self.CONFIG_FILENAME)
        if os.path.exists(config_name):
            # Loading
            config = configparser.ConfigParser()
            config.read(config_name)
            # Check the version
            current_version = config['Global'].get('version')
            if current_version != self.CONFIG_VERSION:
                raise UserWarning('Present configuration version {0} does not match {1}. You can try manual changes on \'{2}\''.format(current_version, self.CONFIG_VERSION, config_name))

            # Restoring the the parameters
            self.args.corpus = config['Global'].get('corpus')

            self.args.max_samples = config['Dataset'].getint('max_samples')
            self.args.vocab_size = config['Dataset'].getint('vocab_size')
            self.args.max_length = config['Dataset'].getint('max_length')
            self.args.buffer_size = config['Dataset'].getint('buffer_size')
            self.args.epochs = config['Dataset'].getint('epochs')

            self.args.num_layers = config['Model'].getint('num_layers')
            self.args.d_model = config['Model'].getint('d_model')
            self.args.heads = config['Model'].getint('heads')
            self.args.units = config['Model'].getint('units')

            # No restoring for training params, batch size or other non model dependent parameters

            # Show the restored params
            print('Warning: Restoring parameters:...')
            print('\nGlobal options:')
            print('Corpus: {}'.format(self.args.corpus))
            print('\nDataset options:')
            print('Sample Size: {}'.format(self.args.max_samples))
            print('Vocab Size: {}'.format(self.args.vocab_size))
            print('Max Sentence Length: {}'.format(self.args.max_length))
            print('Buffer Size: {}'.format(self.args.buffer_size))
            print('# of Epochs: {}'.format(self.args.epochs))
            print('\nModel options:')
            print('Encoder-Decoder layers: {}'.format(self.args.num_layers))
            print('Model Dimension: {}'.format(self.args.d_model))
            print('Parallel layers: {}'.format(self.args.heads))
            print('Dense layers dimension: {}'.format(self.args.units))
        else:
            print('\nNo config file found. Passing...')
