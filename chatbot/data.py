# import future
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
# import module
import os
import random
import tensorflow_datasets as tfds
import tensorflow as tf
# import file
from . import config
from .utils import preprocess_sentence


class dataHandler:
    def __init__(self):
        int = random.randint(0, 10000)
        # Preprocessing data from Cornell corpus
        self.movie_lines = os.path.join(config.CHATDATA_PATH, config.LINES_FILE)
        self.movie_conversations = os.path.join(config.CHATDATA_PATH, config.CONVERSATIONS_FILE)
        self.questions, self.answers = self.load_conversations()
        print('\nSample question: {}'.format(self.questions[int]))
        print('Sample answer: {}'.format(self.answers[int]))
        # Processing dataset from given data
        self.tokenizer = self.load_tokenizer()
        self.START_TOKEN, self.END_TOKEN, self.VOCAB_SIZE = self.misc_token(self.tokenizer)
        print('\nTokenized sample question: {}'.format(self.tokenizer.encode(self.questions[int])))
        print('Vocab size: {}'.format(self.VOCAB_SIZE))
        # Padding and tokenizer to the tensor size
        self.t_questions, self.t_answers = self.tokenize_and_filter(self.questions, self.answers)
        print('Number of samples: {}'.format(len(self.t_questions)))
        self.dataset = self.create_dataset()
        print('\nCreated dataset.\n')

    def load_conversations(self):
        # dictionary of line id to text
        id2line = {}
        with open(self.movie_lines, errors='ignore') as file:
            for line in file.readlines():
                parts = line.replace('\n', '').split(' +++$+++ ')
                id2line[parts[0]] = parts[4]

        inputs, outputs = [], []
        with open(self.movie_conversations, 'r') as file:
            for line in file.readlines():
                parts = line.replace('\n', '').split(' +++$+++ ')
                # get conversation in a list of line ID
                conversation = [line[1:-1]
                                for line in parts[3][1:-1].split(', ')]
                for i in range(len(conversation) - 1):
                    inputs.append(preprocess_sentence(
                        id2line[conversation[i]]))
                    outputs.append(preprocess_sentence(
                        id2line[conversation[i + 1]]))
                    if len(inputs) >= config.MAX_SAMPLES:
                        return inputs, outputs
        return inputs, outputs

    def misc_token(self, tokenizer):
        # Return start token, end token, and vocab size
        # Define start and end token to indicate the start and end of a sentence
        START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [
            tokenizer.vocab_size + 1]
        # Vocabulary size plus start and end token
        VOCAB_SIZE = tokenizer.vocab_size + 2
        return START_TOKEN, END_TOKEN, VOCAB_SIZE

    def load_tokenizer(self):
        #  tfds is kinda dank
        name_without_ext = config.VOCAB_FILENAME.format(config.MAX_SAMPLES, config.MAX_VOCAB_SIZE)
        name_with_ext = '{}.subwords'.format(name_without_ext)
        vocab_filename = os.path.join(config.VOCAB_PATH, name_without_ext)
        vocab_filepath = os.path.join(config.VOCAB_PATH, name_with_ext)

        def create_tokenizer():
            # Build tokenizer using tfds for both questions and answers
            tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(self.questions + self.answers,
                                                                                target_vocab_size=config.MAX_VOCAB_SIZE)
            tokenizer.save_to_file(vocab_filename)
            return tokenizer
        if not os.path.exists(vocab_filepath):
            print('\nCreating new tokenizer...')
            tokenizer = create_tokenizer()
        else:
            print('\nTokenizer already initialized. Loading from file...')
            tokenizer = tfds.features.text.SubwordTextEncoder.load_from_file(vocab_filename)
        return tokenizer

    # Tokenize, filter and pad sentences
    def tokenize_and_filter(self, inputs, outputs):
        tokenized_inputs, tokenized_outputs = [], []
        for (sentence1, sentence2) in zip(inputs, outputs):
            # tokenize sentence
            sentence1 = self.START_TOKEN + \
                self.tokenizer.encode(sentence1) + self.END_TOKEN
            sentence2 = self.START_TOKEN + \
                self.tokenizer.encode(sentence2) + self.END_TOKEN
            # check tokenized sentence max length
            if len(sentence1) <= config.MAX_LENGTH and len(sentence2) <= config.MAX_LENGTH:
                tokenized_inputs.append(sentence1)
                tokenized_outputs.append(sentence2)

        # pad tokenized sentences
        tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(
            tokenized_inputs, maxlen=config.MAX_LENGTH, padding='post')
        tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(
            tokenized_outputs, maxlen=config.MAX_LENGTH, padding='post')

        return tokenized_inputs, tokenized_outputs

    def create_dataset(self):
        # Building Dataset using tf.data.Dataset
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
        )).cache().shuffle(config.BUFFER_SIZE).batch(config.BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
        return dataset
