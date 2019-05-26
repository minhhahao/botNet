# import future
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# import module
import os
import io
import re
import unicodedata
import time

# import file from the directory
import config

# renaming module
import tensorflow as tf
# Draw output attention graph
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class LanguageIndex():
    def __init__(self, lang):
        self.lang = lang
        self.word2idx = {}
        self.idx2word = {}
        self.vocab = set()

        self.create_index()

    def create_index(self):
        for phrase in self.lang:
            self.vocab.update(phrase.split(' '))

        self.vocab = sorted(self.vocab)

        self.word2idx['<pad>'] = 0
        for index, word in enumerate(self.vocab):
            self.word2idx[word] = index + 1

        for word, index in self.word2idx.items():
            self.idx2word[index] = word


class DataHandler:
    def __init__(self):
        '''
        Load all conversations
        '''
        # Path
        self.data_file = self._file_path(config.DATA_PATH, target='spa')

        # Parameters
        self.target_language_list, self.input_language_list = self.create_data(
            self.data_file, config.NUM_EXAMPLES)
        self.input_language = LanguageIndex(sp for sp in self.input_language_list)
        self.target_language = LanguageIndex(en for en in self.target_language_list)

        self.input_tensor, self.target_tensor, \
            self.input_lang_tokenizer, self.target_lang_tokenizer = \
            self.load_data(self.data_file, config.NUM_EXAMPLES)

        # return max(len) for sentence in each tensor
        self.max_length_input = self.max_length(self.input_tensor)
        self.max_length_target = self.max_length(self.target_tensor)

        # Spliting tensors into test set - validation set
        self.input_tensor_train, self.input_tensor_validation, \
            self.target_tensor_train, self.target_tensor_validation = \
            train_test_split(self.input_tensor,
                             self.target_tensor,
                             test_size=0.2)

        print("Input Language; index to word mapping")
        self.convert(self.input_language, self.input_tensor_train[0])
        print()
        print("Target Language; index to word mapping")
        self.convert(self.target_language, self.target_tensor_train[0])
        # return vocab size
        vocab_in_size = len(self.input_language.word2idx) + 1

        vocab_out_size = len(self.target_language.word2idx) + 1

    # NOTE: Misc
    def _file_path(self, dir, target=''):
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
                fileName = os.getcwd() + os.sep + config.DATA_PATH + os.sep + f
            return fileName

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
                print("%d ----> %s" % (t, lang.word2idx[t]))

    def max_length(self, tensor):
        '''
        Calculate max_length of input and output tensor
        args:
            tensor: input/output tensor
        return:
            int(): length of the tensor
        '''
        return max(len(t) for t in tensor)

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

        input_tensor, input_lang_tokenizer = self.tokenize(
            self.input_language)
        target_tensor, target_lang_tokenizer = self.tokenize(
            self.target_language)

        return input_tensor, target_tensor, input_lang_tokenizer, target_lang_tokenizer

    def tokenize(self, lang):
        lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
        lang_tokenizer.fit_on_texts(lang)

        tensor = lang_tokenizer.texts_to_sequences(lang)

        tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                               padding='post')

        return tensor, lang_tokenizer

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
        words_pairs = [[self.preprocess_sentence(
            w) for w in l.split('\t')] for l in lines[:num_examples]]
        print('Dataset created after {} sec'.format(time.time() - start))
        return words_pairs

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
