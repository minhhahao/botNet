from __future__ import absolute_import, division, print_function
from sklearn.model_selection import train_test_split

import time
import os

import config
import data
import model

import tensorflow as tf
import matplotlib.pyplot as plt

# Enable easier debugging and cleaner terminal
tf.enable_eager_execution()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.logging.set_verbosity(tf.logging.ERROR)

# Params for training
path_to_file = data._filePath(config.DATA_DIR, target='spa')
in_tensor, targ_tensor, in_lang, targ_lang, max_len_in, max_len_targ = data.load_dataset(
    path_to_file, config.NUM_EXAMPLES)

# Creating training and validation sets using an 80-20 split
in_tensor_train, in_tensor_val, targ_tensor_train, targ_tensor_val = train_test_split(
    in_tensor, targ_tensor, test_size=0.2)

BUFFER_SIZE = len(in_tensor_train)
N_BATCH = BUFFER_SIZE // config.BATCH_SIZE
vocab_inp_size = len(in_lang.word2idx)
vocab_tar_size = len(targ_lang.word2idx)

# Create dataset
dataset = tf.data.Dataset.from_tensor_slices(
    (in_tensor_train, targ_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(config.BATCH_SIZE, drop_remainder=True)

# create model and Checkpoint
encoder, decoder, optimizer = model.create_model(
    vocab_inp_size, vocab_tar_size, config.EMBEDDING_DIMENSION, config.UNITS, config.BATCH_SIZE)
ckpt_prefix = os.path.join(config.CKPT_PATH, 'ckpt')
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)
ckpt_config = tf.estimator.RunConfig(keep_checkpoint_max = 5,)

# function for plotting the attention weights
def plot_attention(attention, sentence, predicted_sentence):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')

    fontdict = {'fontsize': 12}

    ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

    plt.show()


def translate(sentence, encoder, decoder, in_lang, targ_lang, max_len_in, max_len_targ):
    result, sentence, attention_plot = data.evaluate(
        sentence, encoder, decoder, in_lang, targ_lang, max_len_in, max_len_targ)

    print('Input: {}'.format(sentence))
    print('Predicted translation: {}'.format(result))

    attention_plot = attention_plot[:len(
        result.split(' ')), :len(sentence.split(' '))]
    # return a plot for attention
    # plot_attention(attention_plot, sentence.split(' '), result.split(' '))


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

                    loss += model.loss_function(targ[:, t], predictions)

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
        print('Time taken for 1 epoch is {} sec\n'.format(time.time() - start))


if __name__ == '__main__':
    inp = input('To train the model, type train. To test the model, type translate. Ctrl+C to exit: ')
    try:
        if inp == 'train':
            train()
        elif inp == 'translate':
            checkpoint.restore(tf.train.latest_checkpoint(config.CKPT_PATH))
            sen = input('Type stop to exit translation mode. Type continue to input sentence: ')
            while sen != 'stop':
                sens = input('> ')
                if sens == 'exit':
                    break
                translate(sens, encoder, decoder, in_lang, targ_lang, max_len_in, max_len_targ)
    except KeyboardInterrupt:
        pass
