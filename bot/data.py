# import future
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
# import module
import os
import re
import random
import tensorflow_datasets as tfds
import tensorflow as tf
# import file
from . import config


class dataHandler:
    def __init__(self):
        self.movie_lines = os.path.join(config.DATA_PATH, config.LINES_FILE)
        self.movie_conversations = os.path.join(config.DATA_PATH, config.CONVERSATIONS_FILE)
        self.vocab_file = os.path.join('data','samples', config.VOCAB_FILE)
        self.questions, self.answers = self.load_conversations()
        int = random.randint(0, 10000)
        print('\nSample question: {}'.format(self.questions[int]))
        print('\nSample answer: {}'.format(self.answers[int]))
        self.tokenizer, self.START_TOKEN, self.END_TOKEN, self.VOCAB_SIZE = self.tokenizer()
        print('\nTokenized sample question: {}'.format(self.tokenizer.encode(self.questions[int])))
        print('\nVocab size: {}'.format(self.VOCAB_SIZE))
        self.t_questions, self.t_answers = self.tokenize_and_filter(self.questions, self.answers)
        print('\nNumber of samples: {}\n'.format(len(self.t_questions)))
        self.dataset = self.create_dataset()
        print('Created dataset.\n')

    def preprocess_sentence(self, sentence):
        sentence = sentence.lower().strip()
        # creating a space between a word and the punctuation following it
        # eg: "he is a boy." => "he is a boy ."
        sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
        sentence = re.sub(r'[" "]+', " ", sentence)
        # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
        sentence = re.sub(r"[^a-zA-Z?.!,]+", " ", sentence)
        sentence = sentence.strip()
        # adding a start and an end token to the sentence
        return sentence

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
                    inputs.append(self.preprocess_sentence(
                        id2line[conversation[i]]))
                    outputs.append(self.preprocess_sentence(
                        id2line[conversation[i + 1]]))
                    if len(inputs) >= config.MAX_SAMPLES:
                        return inputs, outputs
        return inputs, outputs

    def tokenizer(self):
        # Build tokenizer using tfds for both questions and answers
        tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            self.questions + self.answers, target_vocab_size=2**13)
        if os.path.isfile(self.vocab_file):
            print('Vocab file exists.')
        else:
            tokenizer.save_to_file(self.vocab_file)
        # Define start and end token to indicate the start and end of a sentence
        START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [
            tokenizer.vocab_size + 1]
        # Vocabulary size plus start and end token
        VOCAB_SIZE = tokenizer.vocab_size + 2
        return tokenizer, START_TOKEN, END_TOKEN, VOCAB_SIZE

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
