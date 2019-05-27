# import future
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
# import module
import os
import sys
# import datetime
# import matplotlib.pyplot as plt
import tensorflow as tf
# import file
import config
import data
import model
# Clear previous session
# tf.keras.backend.clear_session()


# Custom learning rate following the paper
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
plt.ylabel("Learning Rate")
plt.xlabel("Train Step")
plt.plot(sample_learning_rate(tf.range(200000, dtype=tf.float32)))
'''

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

# Custom params following the paper
learning_rate = CustomSchedule(config.D_MODEL)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
# log_dir = 'logs' + os.sep + 'fit' + os.sep + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


class Mode:
    def __init__(self):
        # Init data object
        self.process = data.dataHandler()
        self.model_name = 'model-numLayers{}-Dmodel{}-units{}-dropout{}-version{}.h5'.format(
            config.NUM_LAYERS, config.D_MODEL, config.UNITS, config.DROPOUT, config.VERSION)
        self.main()

    def main(self):
        inp = input('\nType "train" to train, "test" to evaluate: ')
        try:
            if inp == 'train':
                self.train()
            elif inp == 'test':
                model = tf.keras.models.load_model(self.model_name)
                while True:
                    line = self._get_user_input()
                    if len(line) > 0 and line[-1] == '\n':
                        line = line[:-1]
                    if line == '':
                        break
                    self.predict(model, line)
        except KeyboardInterrupt:
            print('Terminated')

    def predict(self, model_t, sentence):
        def evaluate(self, model_t, sentence):
            sentence = self.process.preprocess_sentence(sentence)
            sentence = tf.expand_dims(self.process.START_TOKEN + self.process.tokenizer.encode(sentence) + self.process.END_TOKEN, axis=0)

            output = tf.expand_dims(self.process.START_TOKEN, 0)

            for i in range(config.MAX_LENGTH):
                predictions = model_t(inputs=[sentence, output], training=False)
                # select the last word from the seq_len dimension
                predictions = predictions[:, -1:, :]
                predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
                # return the result if the predicted_id is equal to the end token
                if tf.equal(predicted_id, self.process.END_TOKEN[0]):
                    break
                # concatenated the predicted_id to the output which is given
                # to the decoder as its input.
                output = tf.concat([output, predicted_id], axis=-1)
            return tf.squeeze(output, axis=0)

        prediction = evaluate(model_t, sentence)
        predicted_sentence = self.process.tokenizer.decode([i for i in prediction if i < self.process.tokenizer.vocab_size])
        print('Input: {}'.format(sentence))
        print('Output: {}'.format(predicted_sentence))
        return predicted_sentence

    def train(self):
        # Create model
        print('\nCreating model...')
        model_trans = model.transformer(
            vocab_size=self.process.VOCAB_SIZE,
            num_layers=config.NUM_LAYERS,
            units=config.UNITS,
            d_model=config.D_MODEL,
            num_heads=config.NUM_HEADS,
            dropout=config.DROPOUT)
        print('\n Model summary')
        model_trans.summary()
        model_trans.compile(optimizer=optimizer,
                            loss=loss_function, metrics=[accuracy])
        # Train the model and save checkpoint
        print('\nStart training...\n')
        model_trans.fit(self.process.dataset, epochs=config.EPOCHS)
        print('\nSaved models to file')
        tf.keras.experimental.export_saved_model(
            model_trans, os.path.join('data', 'save', self.model_name))
        del model_trans
        print('Finished')

    def _get_user_input(self):
        '''Get user's input, which will be transformed into encoder input later'''
        print("> ", end="")
        sys.stdout.flush()
        return sys.stdin.readline()
