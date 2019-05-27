import config
from train import *
import tensorflow as tf

model_test = create_model()
model_test.load_weights(tf.train.latest_checkpoint(config.CKPT_DIR))
sentence = 'Hello, are you retarded?'
sentence = predict(model_test, sentence)
