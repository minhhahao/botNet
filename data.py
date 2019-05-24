# import module
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import io
import re
import unicodedata
import time

# import file from the directory
import config

# renaming module
import numpy as np
import tensorflow as tf
# Draw output attention graph
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class DataHandler():
    def __init__(self):
        '''
        Load all conversations
        Args:
            args: parameters of the model
        '''
        # Path
        self.data_file = self._filePath(config.DATA_PATH, target='spa')
        # Parameters
        self.input_language = []
        self.target_language = []
        self.input_tensor = []
        self.target_tensor = []
        self.input_lang_tokenizer = []  # Index input language
        self.target_lang_tokenizer = []  # Index output language
        self.max_length_input = 0
        self. max_length_target = 0

    # NOTE: Misc
    def _filePath(self, dir, target=''):
        '''
        args:
            dir: directory of dataset
            target: target translate language
        returns:
            data directory
        '''
        data = [f for f in os.listdir(dir)]
        for f in data:
            if f[:-4] == target:
                fileName = f
                file = os.getcwd() + os.sep + config.DATA_PATH + os.sep + fileName
            return file

    def convert(self, lang, tensor):
        '''
        Help visualise index to word mapping
        args:
            lang: input/output language
            tensor: given tensor
        returns:
            pretty print for lit terminal
        '''
        for t in tensor:
            if t != 0:
                print("%d ----> %s" % (t, lang.index_word[t]))

    def len_target(self):
        self.max_length_input = self.max_length(self.input_tensor)
        self.max_length_target = self.max_length(self.target_tensor)

        def max_length(self, tensor):
            '''
            Calculate max_length of input and output tensor
            args:
                tensor: input/output tensor
            return:
                int(): length of the tensor
            '''
            return max(len(t) for t in tensor)
        return self.max_length_input, self.max_length_target

    def plot_attention(attention, sentence, predicted_sentence):
        '''
        plotting the attention weights
        args:
            attention: weight
            sentence: input
            predicted_sentence: output
        returns:
            a nice attention weight plot
        '''
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(1, 1, 1)
        ax.matshow(attention, cmap='viridis')
        fontdict = {'fontsize': 12}
        ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
        ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)
        plt.show()

    def split_set(self):
        '''
        split training, validation set of data
        args:
        returns:
            input_tensor_train
            input_tensor_validation
            target_tensor_train
            target_tensor_validation
        '''
        input_tensor_train, input_tensor_validation, target_tensor_train, target_tensor_validation = train_test_split(
            self.input_tensor, self.target_tensor, test_size=0.2)
        return input_tensor_train, input_tensor_validation, target_tensor_train, target_tensor_validation

    # NOTE: Dataset creation
    def create_data(self, path, num_examples):
        '''
        args:
            path: data_file
            num_examples: # of examples defined in config.NUM_EXAMPLES
        returns:
            word pairs in the format: [in_lang, targ_lang]
        '''
        # TODO: creates more diverse set of input and output languages
        start = time.time()
        lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
        words_pairs = [[self.preprocess_sentence(w) for w in l.split('\t')]  for l in lines[:num_examples]]
        print('Dataset created after {} sec'.format(time.time() - start))
        return zip(*words_pairs)

    def unicode_to_ascii(self, s):
        '''
        Converts the unicode file to ascii for better space allocation
        args:
            s : string
        returns:
            str(): ascii formatted file
        '''
        return ''.join(c for c in unicodedata.normalize('NFD', s)
                       if unicodedata.category(c) != 'Mn')

    def preprocess_sentence(self, w):
        '''
        Processing given sentence
        args:
            w: string
        returns:
            str(): sentence with <start> <end> marks
        '''
        w = self.unicode_to_ascii(w.lower().strip())
        # creating a space between a word and the punctuation following it
        w = re.sub(r"([?.!,¿])", r" \1 ", w)
        w = re.sub(r'[" "]+', " ", w)
        # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
        w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
        w = w.rstrip().strip()
        w = '<start> ' + w + ' <end>'
        return w

    # NOTE: Load dataset and create variables for the model
    def load_data(self, path, num_examples=None):
        '''
        args:
            str(path): data_file
            int(num_examples): # of examples defined in config.NUM_EXAMPLES
        returns:
            list(input_tensor): input tensor with pad_sequences
            list(target_tensor): output tensor with pad_sequences
            str(input_language): index for input language using LanguageIndex
            str(target_language): index for output language using LanguageIndex
        '''
        # creating cleaned input, output pairs
        self.target_language, self.input_language = self.create_data(path, num_examples)

        self.input_tensor, self.input_lang_tokenizer = self.tokenize(self.input_language)
        self.target_tensor, self.target_lang_tokenizer = self.tokenize(
            self.target_language)

        return self.input_tensor, self.target_tensor, self.input_lang_tokenizer, self.target_lang_tokenizer

    def tokenize(self, lang):
        lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
        lang_tokenizer.fit_on_texts(lang)

        tensor = lang_tokenizer.texts_to_sequences(lang)

        tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                               padding='post')

        return tensor, lang_tokenizer
