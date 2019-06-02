# import future
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# Import modules
import os
import re
import sys

# Fixes for drawing images
# https://github.com/AppliedDataSciencePartners/DeepReinforcementLearning/issues/3#issuecomment-420989055
os.environ['PATH'] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

# python main.py --verbose --model_tag drunkboiv2 --epochs 20 --num_layers 3 --max_samples 65000 --buffer_size 25000 --batch_size 128 --max_length 45 --vocab_size 16384

def preprocess_sentence(sentence):
    '''
    Remove any special characters, lower all words in a given sentence for data processing
    Args:
        sentence <str>: input sentence
    Returns:
        sentence <str>: purified sentence uwu
    '''
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


def _get_user_input():
    '''Get user's input, which will be transformed into encoder input later'''
    print("> ", end="")
    sys.stdout.flush()
    return sys.stdin.readline()

# Drawing model structures


class Draw:
    '''
    I'm kinda lazy rn so this is just a subclass to draw the model or
    learning rate for better visualisation
    '''
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from .model import PositionalEncoding, encoder, encoder_layer, decoder, decoder_layer, transformer

    def __init__(self):
        self.units = 512
        self.d_model = 128
        self.num_heads = 4
        self.dropout = 0.3
        self.vocab_size = 8192
        self.num_layers = 2

    def draw_pos_encoding(self):
        sample_pos_encoding = self.PositionalEncoding(50, 512)
        self.plt.pcolormesh(sample_pos_encoding.pos_encoding.numpy()[0], cmap='RdBu')
        self.plt.xlabel('Depth')
        self.plt.xlim((0, 512))
        self.plt.ylabel('Position')
        self.plt.colorbar()
        self.plt.show()

    def draw_encoder_layer(self):
        # Sample encoder layer
        sample_encoder_layer = self.encoder_layer(self.units, self.d_model, self.num_heads, self.dropout, name="sample_encoder_layer")
        self.tf.keras.utils.plot_model(sample_encoder_layer, to_file='data' + os.sep + 'images' + os.sep + 'encoder_layer.png', show_shapes=True)

    def draw_encoder(self):
        # Sample encoder model using n-encoder layer
        sample_encoder = self.encoder(self.vocab_size, self.num_layers, self.units, self.d_model, self.num_heads, self.dropout, name="sample_encoder")
        self.tf.keras.utils.plot_model(sample_encoder, to_file='data' + os.sep + 'images' + os.sep + 'encoder.png', show_shapes=True)

    def draw_decoder_layer(self):
        # Sample decoder layer
        sample_decoder_layer = self.decoder_layer(self.units, self.d_model, self.num_heads, self.dropout, name="sample_decoder_layer")
        self.tf.keras.utils.plot_model(sample_decoder_layer, to_file='data' + os.sep + 'images' + os.sep + 'decoder_layer.png', show_shapes=True)

    def draw_decoder(self):
        # Sample decoder model using n-decoder layer
        sample_decoder = self.decoder(self.vocab_size, self.num_layers, self.units, self.d_model, self.num_heads, self.dropout, name="sample_decoder")
        self.tf.keras.utils.plot_model(sample_decoder, to_file='data' + os.sep + 'images' + os.sep + 'decoder.png', show_shapes=True)

    def draw_transformer(self):
        # Sample seq2seq model with transformer
        # https://arxiv.org/pdf/1706.03762.pdf
        sample_transformer = self.transformer(vocab_size=self.vocab_size, num_layers=4, units=self.units, d_model=self.d_model, num_heads=self.num_heads, drouput=self.dropout, name="sample_transformer")
        self.tf.keras.utils.plot_model(sample_transformer, to_file='data' + os.sep + 'images' + os.sep + 'transformer.png', show_shapes=True)
