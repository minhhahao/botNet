# import future
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
# import module
import os
import datetime
import matplotlib.pyplot as plt
import tensorflow as tf
# import file
import config
import data
import model

tf.random.set_seed(1234)
# clean terminal view
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Clear previous session
# tf.keras.backend.clear_session()


def loss_function(y_true, y_pred):
    y_true = tf.reshape(y_true, shape=(-1, config.MAX_LENGTH - 1))

    loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')(y_true, y_pred)

    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    loss = tf.multiply(loss, mask)

    return tf.reduce_mean(loss)


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

'''
# Visualise sample learning curve
sample_learning_rate = CustomSchedule(d_model=128)

plt.plot(sample_learning_rate(tf.range(200000, dtype=tf.float32)))
plt.ylabel("Learning Rate")
plt.xlabel("Train Step")
'''

learning_rate = CustomSchedule(config.D_MODEL)

optimizer = tf.keras.optimizers.Adam(
    learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)


def accuracy(y_true, y_pred):
    # ensure labels have shape (batch_size, MAX_LENGTH - 1)
    y_true = tf.reshape(y_true, shape=(-1, config.MAX_LENGTH - 1))
    accuracy = tf.metrics.SparseCategoricalAccuracy()(y_true, y_pred)
    return accuracy


model_trans = model.transformer(
    vocab_size=data.VOCAB_SIZE,
    num_layers=config.NUM_LAYERS,
    units=config.UNITS,
    d_model=config.D_MODEL,
    num_heads=config.NUM_HEADS,
    dropout=config.DROPOUT)
model_trans.summary()
# Create checkpoint
ckpt_callback = tf.keras.callbacks.ModelCheckpoint(config.CKPT_PATH,
                                                   save_weights_only=True,
                                                   verbose=1)
log_dir='logs' + os.sep + 'fit'+ os.sep + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                             histogram_freq=1)
model_trans.compile(optimizer=optimizer, loss=loss_function, metrics=[accuracy])
model_trans.fit(data.dataset, epochs=config.EPOCHS, callbacks=[tensorboard, ckpt_callback])
if os.listdir(config.CKPT_PATH):
    # Restore from checkpoint
    model_trans.load_weights(config.CKPT_PATH)
    model_trans.fit(data.dataset, epochs=config.EPOCHS, callbacks=[tensorboard, ckpt_callback])
else:
    pass


def evaluate(sentence):
    sentence = data.preprocess_sentence(sentence)

    sentence = tf.expand_dims(
        data.START_TOKEN + data.tokenizer.encode(sentence) + data.END_TOKEN, axis=0)

    output = tf.expand_dims(data.START_TOKEN, 0)

    for i in range(config.MAX_LENGTH):
        predictions = model(inputs=[sentence, output], training=False)

        # select the last word from the seq_len dimension
        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # return the result if the predicted_id is equal to the end token
        if tf.equal(predicted_id, data.END_TOKEN[0]):
            break

        # concatenated the predicted_id to the output which is given to the decoder
        # as its input.
        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0)


def predict(sentence):
    prediction = evaluate(sentence)

    predicted_sentence = data.tokenizer.decode(
        [i for i in prediction if i < data.tokenizer.vocab_size])

    print('Input: {}'.format(sentence))
    print('Output: {}'.format(predicted_sentence))

    return predicted_sentence


