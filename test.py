import data
import config
import model
import train

model_test = train.create_model()
model_test.load_weights(tf.train.latest_checkpoint(config.CKPT_DIR))
sentence = 'I am not crazy, my mother had me tested.'
for _ in range(5):
  sentence = train.predict(sentence)
  print('')