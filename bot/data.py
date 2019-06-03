# Copyright 2019 Aaron Pham. All right reserved

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#   ==========================================================================

'''
    Descriptions: Data Preprocessing with tfds
        Corpus: Cornell Movie Dialogs (default)
        TODO: increase database size
'''
# import future
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# import module
import os
import random  # Debugging purposes
import tensorflow_datasets as tfds  # Dataset processing because I'm kinda lazy
import tensorflow as tf

# import module from file
from .utils import preprocess_sentence
from misc import ROOT_DIR

class dataHandler:
    '''
    Processing datafile
    Returns:
        dataset <obj>: tokenized mini-batch
        questions, answers list(<str>): correspondingly list of inputs and outputs for encoder-decoder
        tokenizer <obj>: Tensorflow tokenizer
        START_TOKEN, END_TOKEN <int>: dynamic START_TOKEN, END_TOKEN for padding (TODO: try normal fixed token)
        t_questions, t_answers list(<str>): tokenized questions, answers
    '''

    def __init__(self, args):
        '''
        Args:
            args (list<str>): arguments for processing data
        '''
        self.args = args
        # Init path
        self.DATA_PATH = 'data'
        self.SAMPLES_PATH = 'samples'

        # Filename
        self.TOKEN_NO_EXT = self._construct_token_path()
        self.TOKEN_FILE = '{}.subwords'.format(self.TOKEN_NO_EXT)
        # self.DATASET_FILE = 'dataset-{}.tfrecord'.format(self.args.corpus)
        # random int for debugging
        self.int = random.randint(0, 10000)
        self.process_data()

    def process_data(self):
        '''driver code'''
        # Preprocessing data from Cornell corpus
        self.questions, self.answers = self.process_cornell()
        # Processing dataset from given data
        self.tokenizer, self.START_TOKEN, self.END_TOKEN = self.load_tokenizer()
        # Padding and tokenizer to the tensor size
        self.t_questions, self.t_answers = self.tokenize_and_filter(self.questions, self.answers)
        # Create Dataset
        self.dataset = self.create_dataset()
        if self.args.verbose:
            print('\nSample question: {}'.format(self.questions[self.int]))
            print('Sample answer: {}'.format(self.answers[self.int]))
            print('\nTokenized sample question: {}'.format(self.tokenizer.encode(self.questions[self.int])))
            print('\nNumber of samples: {}'.format(len(self.t_questions)))
            print('\nCreated dataset.\n')

    def _construct_token_path(self):
        '''Construct path for token file without extension (tfds compatibility)'''
        path = os.path.join(ROOT_DIR, self.DATA_PATH, self.SAMPLES_PATH) + os.sep + 'tokenizer-{}-size{}-samples{}'.format(self.args.corpus, self.args.vocab_size, self.args.max_samples)
        return path

    def process_cornell(self):
        '''Process data from Cornell corpus'''
        # dictionary of line id to text
        movie_lines = os.path.join(ROOT_DIR, self.DATA_PATH, 'cornell', 'movie_lines.txt')
        movie_conversations = os.path.join(ROOT_DIR, self.DATA_PATH, 'cornell', 'movie_conversations.txt')

        # Put lines into dict with {key:line, value:lineID}
        id2line = {}
        with open(movie_lines, 'r', errors='ignore', encoding='utf-8') as file:
            for line in file.readlines():
                parts = line.replace('\n', '').split(' +++$+++ ')
                id2line[parts[0]] = parts[4]

        inputs, outputs = [], []
        with open(movie_conversations, 'r', errors='ignore', encoding='utf-8') as file:
            for line in file.readlines():
                parts = line.replace('\n', '').split(' +++$+++ ')
                # get conversation in a list of line ID
                conversation = [line[1:-1] for line in parts[3][1:-1].split(', ')]
                for i in range(len(conversation) - 1):
                    inputs.append(preprocess_sentence(id2line[conversation[i]]))
                    outputs.append(preprocess_sentence(id2line[conversation[i + 1]]))
                    if len(inputs) >= self.args.max_samples:
                        return inputs, outputs

        return inputs, outputs

    def load_tokenizer(self):
        '''Build tokenizer'''
        if not os.path.exists(self.TOKEN_FILE):
            print('\nCreating new tokenizer...')
            tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(self.questions + self.answers,
                                                                                target_vocab_size=self.args.vocab_size)
            tokenizer.save_to_file(self.TOKEN_NO_EXT)
        else:
            print('\nTokenizer already initialized. Loading from file...')
            tokenizer = tfds.features.text.SubwordTextEncoder.load_from_file(self.TOKEN_NO_EXT)
        # Define start and end token to indicate the start and end of a sentence
        START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]
        return tokenizer, START_TOKEN, END_TOKEN

    # Tokenize, filter and pad sentences
    def tokenize_and_filter(self, inputs, outputs):
        '''Tokenizer and filter inputs-outputs to fit the model hyperparams'''
        tokenized_inputs, tokenized_outputs = [], []
        for (sentence1, sentence2) in zip(inputs, outputs):
            # tokenize sentence
            sentence1 = self.START_TOKEN + self.tokenizer.encode(sentence1) + self.END_TOKEN
            sentence2 = self.START_TOKEN + self.tokenizer.encode(sentence2) + self.END_TOKEN
            # check tokenized sentence max length
            if len(sentence1) <= self.args.max_length and len(sentence2) <= self.args.max_length:
                tokenized_inputs.append(sentence1)
                tokenized_outputs.append(sentence2)
        # pad tokenized sentences
        tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(tokenized_inputs, maxlen=self.args.max_length, padding='post')
        tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(tokenized_outputs, maxlen=self.args.max_length, padding='post')
        return tokenized_inputs, tokenized_outputs

    def create_dataset(self):
        '''Building Dataset using tf.data.Dataset'''
        # decoder inputs use the previous target as input
        # remove START_TOKEN from targets
        dataset = tf.data.Dataset.from_tensor_slices((
            {
                'inputs': self.t_questions,
                'dec_inputs': self.t_answers[:, :-1]
            },
            {
                'outputs': self.t_answers[:, 1:]
            },
        ))
        dataset = dataset.cache()
        dataset = dataset.shuffle(self.args.buffer_size)
        dataset = dataset.batch(self.args.batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset
        # TODO: save dataset to file, probably load dataset if continue training to reduce time?
