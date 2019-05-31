# import future
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
# import module
import os
import sys
import datetime
import logging
import tensorflow as tf
# import file
from . import config
from . import data
from . import model

# clean terminal view
logging.getLogger('tensorflow').disabled = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# data object
process = data.dataHandler()
# Log directory
# TODO: Fixing tensorboard
chatbotlog = 'save' + os.sep + 'chatbot_logs' + os.sep + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
file_writer = tf.summary.create_file_writer(
    chatbotlog + os.sep + 'scalar' + os.sep + 'metrics')
file_writer.set_as_default()
# Checkpoint for weight
checkpoint_path = os.path.join('save', 'chatbot', 'cp-{epoch:04d}.ckpt')
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 verbose=1,
                                                 save_weights_only=True,
                                                 period=5)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=chatbotlog,
                                                      write_graph=True,
                                                      write_images=True)


# Custom learning rate schedules
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def predict(model_t, sentence):

    def evaluate(model_t, sentence):
        sentence = process.preprocess_sentence(sentence)
        sentence = tf.expand_dims(
            process.START_TOKEN + process.tokenizer.encode(sentence) + process.END_TOKEN, axis=0)

        output = tf.expand_dims(process.START_TOKEN, 0)

        for i in range(config.MAX_LENGTH):
            predictions = model_t(
                inputs=[sentence, output], training=False)
            # select the last word from the seq_len dimension
            predictions = predictions[:, -1:, :]
            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
            # return the result if the predicted_id is equal to the end token
            if tf.equal(predicted_id, process.END_TOKEN[0]):
                break
            # concatenated the predicted_id to the output which is given
            # to the decoder as its input.
            output = tf.concat([output, predicted_id], axis=-1)
        return tf.squeeze(output, axis=0)

    prediction = evaluate(model_t, sentence)
    predicted_sentence = process.tokenizer.decode(
        [i for i in prediction if i < process.tokenizer.vocab_size])
    # print('Input: {}'.format(sente))
    print('Output: {}'.format(predicted_sentence))
    return predicted_sentence


def create_model():
    # Custom params following the paper
    learning_rate = CustomSchedule(config.D_MODEL)
    optimizer = tf.keras.optimizers.Adam(learning_rate,
                                         beta_1=0.9,
                                         beta_2=0.98,
                                         epsilon=1e-9)

    def loss_function(y_true, y_pred):
        y_true = tf.reshape(y_true, shape=(-1, config.MAX_LENGTH - 1))
        loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')(y_true, y_pred)
        mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
        loss = tf.multiply(loss, mask)
        return tf.reduce_mean(loss)

    def accuracy(y_true, y_pred):
        # ensure labels have shape (batch_size, MAX_LENGTH - 1)
        y_true = tf.reshape(y_true, shape=(-1, config.MAX_LENGTH - 1))
        accuracy = tf.metrics.SparseCategoricalAccuracy()(y_true, y_pred)
        return accuracy

    # Create model
    created_model = model.transformer(
        vocab_size=process.VOCAB_SIZE,
        num_layers=config.NUM_LAYERS,
        units=config.UNITS,
        d_model=config.D_MODEL,
        num_heads=config.NUM_HEADS,
        dropout=config.DROPOUT)
    created_model.compile(optimizer=optimizer,
                          loss=loss_function,
                          metrics=[accuracy])
    return created_model


def _get_user_input():
    '''
    Get user's input, which will be transformed into encoder input later
    '''
    print("> ", end="")
    sys.stdout.flush()
    return sys.stdin.readline()


def train():
    print('\n Creating models...')
    model_train = create_model()
    print('\nModel summary: ')
    model_train.summary()
    # Train the model and save checkpoint
    model_train.save_weights(checkpoint_path.format(epoch=0))
    print('\nStart training...\n')
    model_train.fit(process.dataset,
                    epochs=config.EPOCHS,
                    callbacks=[cp_callback, tensorboard_callback])
    print('\nFinished')
    del model_train


def continue_train():
    model_cont = create_model()
    model_cont.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    print('\n Start retraining...\n')
    model_cont.fit(process.dataset,
                   epochs=config.EPOCHS,
                   callbacks=[cp_callback, tensorboard_callback])
    model_cont.save_weights(checkpoint_path.format(epoch=0))
    del model_cont


def test():
    model_test = create_model()
    model_test.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    try:
        while True:
            line = _get_user_input()
            if len(line) > 0 and line[-1] == '\n':
                line = line[:-1]
            if line == '':
                break
            predict(model_test, line)
    except KeyboardInterrupt:
        print('\nTerminated')
    del model_test


def run():
    inp = input(
        'Type "train" to start trainning new weight, "continue" to continue training, "test" to test the model: ')
    if inp == 'train':
        train()
    elif inp == 'continue':
        continue_train()
    elif inp == 'test':
        test()
