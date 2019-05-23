from __future__ import absolute_import, division, print_function
from sklearn.model_selection import train_test_split

import time
import os

import config
import data
import model
from model import loss_function
from data import load_dataset, evaluate

import tensorflow as tf
import matplotlib.pyplot as plt

tf.enable_eager_execution()

# Params for training
path_to_file = data._filePath(config.DATA_DIR, target='spa')
input_tensor, target_tensor, inp_lang, targ_lang, max_length_inp, max_length_targ = load_dataset(
    path_to_file, config.NUM_EXAMPLES)

# Creating training and validation sets using an 80-20 split
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(
    input_tensor, target_tensor, test_size=0.2)

BUFFER_SIZE = len(input_tensor_train)
N_BATCH = BUFFER_SIZE // config.BATCH_SIZE
vocab_inp_size = len(inp_lang.word2idx)
vocab_tar_size = len(targ_lang.word2idx)

# Create dataset
dataset = tf.data.Dataset.from_tensor_slices(
    (input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(config.BATCH_SIZE, drop_remainder=True)

# create model and Checkpoint
encoder, decoder, optimizer = model.create_model(
    vocab_inp_size, vocab_tar_size, config.EMBEDDING_DIMENSION, config.UNITS, config.BATCH_SIZE)
ckpt_prefix = os.path.join(config.CKPT_PATH, 'ckpt')
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)


# function for plotting the attention weights
def plot_attention(attention, sentence, predicted_sentence):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')

    fontdict = {'fontsize': 12}

    ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

    plt.show()


def translate(sentence, encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ):
    result, sentence, attention_plot = evaluate(
        sentence, encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ)

    print('Input: {}'.format(sentence))
    print('Predicted translation: {}'.format(result))

    attention_plot = attention_plot[:len(
        result.split(' ')), :len(sentence.split(' '))]
    plot_attention(attention_plot, sentence.split(' '), result.split(' '))


# Training
def train():
    for epoch in range(config.EPOCHS):
        start = time.time()

        hidden = encoder.initialize_hidden_state()
        total_loss = 0

        for (batch, (inp, targ)) in enumerate(dataset):
            loss = 0

            with tf.GradientTape() as tape:
                enc_output, enc_hidden = encoder(inp, hidden)

                dec_hidden = enc_hidden

                dec_input = tf.expand_dims(
                    [targ_lang.word2idx['<start>']] * config.BATCH_SIZE, 1)

                # Teacher forcing - feeding the target as the next input
                for t in range(1, targ.shape[1]):
                    # passing enc_output to the decoder
                    predictions, dec_hidden, _ = decoder(
                        dec_input, dec_hidden, enc_output)

                    loss += loss_function(targ[:, t], predictions)

                    # using teacher forcing
                    dec_input = tf.expand_dims(targ[:, t], 1)

            batch_loss = (loss / int(targ.shape[1]))

            total_loss += batch_loss

            variables = encoder.variables + decoder.variables

            gradients = tape.gradient(loss, variables)

            optimizer.apply_gradients(zip(gradients, variables))

            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f} Time {:.4f}'.format(epoch + 1,
                                                                         batch,
                                                                         batch_loss.numpy(), time.time() - start))
        # saving (checkpoint) the model every 2 epochs
        if (epoch + 1) % 2 == 0:
            checkpoint.save(file_prefix=ckpt_prefix)

        print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                            total_loss / N_BATCH))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))


if __name__ == '__main__':
    inp = input('To train the model, type train. To test the model, type translate. Ctrl+C to exit: ')
    try:
        if inp == 'train':
            train()
        elif inp == 'translate':
            checkpoint.restore(tf.train.latest_checkpoint(config.CKPT_PATH))
            sen = input('Sentence want to translate: ')
            translate(sen, encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ)
    except KeyboardInterrupt:
        pass
