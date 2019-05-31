# import future
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
# Import modules
import matplotlib.pyplot as plt
import tensorflow as tf
import os
# Import files
from . import train
from . import model

# https://github.com/AppliedDataSciencePartners/DeepReinforcementLearning/issues/3#issuecomment-420989055
os.environ['PATH'] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'


def draw_learning_rate():
    # Visualise sample learning curve
    sample_learning_rate = train.CustomSchedule(d_model=128)
    plt.ylabel("Learning Rate")
    plt.xlabel("Train Step")
    plt.plot(sample_learning_rate(tf.range(200000, dtype=tf.float32)))


def draw_pos_encoding():
    sample_pos_encoding = model.PositionalEncoding(50, 512)
    plt.pcolormesh(sample_pos_encoding.pos_encoding.numpy()[0], cmap='RdBu')
    plt.xlabel('Depth')
    plt.xlim((0, 512))
    plt.ylabel('Position')
    plt.colorbar()
    plt.show()


def draw_encoder_layer():
    # Sample encoder layer
    sample_encoder_layer = model.encoder_layer(
        units=512,
        d_model=128,
        num_heads=4,
        dropout=0.3,
        name="sample_encoder_layer")
    tf.keras.utils.plot_model(
        sample_encoder_layer, to_file='data' + os.sep + 'images' + os.sep + 'encoder_layer.png', show_shapes=True)


def draw_encoder():
    # Sample encoder model using n-encoder layer
    sample_encoder = model.encoder(
        vocab_size=8192,
        num_layers=2,
        units=512,
        d_model=128,
        num_heads=4,
        dropout=0.3,
        name="sample_encoder")
    tf.keras.utils.plot_model(
        sample_encoder, to_file='data' + os.sep + 'images' + os.sep + 'encoder.png', show_shapes=True)


def draw_decoder_layer():
    # Sample decoder layer
    sample_decoder_layer = model.decoder_layer(
        units=512,
        d_model=128,
        num_heads=4,
        dropout=0.3,
        name="sample_decoder_layer")
    tf.keras.utils.plot_model(
        sample_decoder_layer, to_file='data' + os.sep + 'images' + os.sep + 'decoder_layer.png', show_shapes=True)


def draw_decoder():
    # Sample decoder model using n-decoder layer
    sample_decoder = model.decoder(
        vocab_size=8192,
        num_layers=2,
        units=512,
        d_model=128,
        num_heads=4,
        dropout=0.3,
        name="sample_decoder")
    tf.keras.utils.plot_model(
        sample_decoder, to_file='data' + os.sep + 'images' + os.sep + 'decoder.png', show_shapes=True)


def draw_transformer():
    # Sample seq2seq model with transformer
    # https://arxiv.org/pdf/1706.03762.pdf
    sample_transformer = model.transformer(
        vocab_size=8192,
        num_layers=4,
        units=512,
        d_model=128,
        num_heads=4,
        dropout=0.3,
        name="sample_transformer")
    tf.keras.utils.plot_model(
        sample_transformer, to_file='data' + os.sep + 'images' + os.sep + 'transformer.png', show_shapes=True)
