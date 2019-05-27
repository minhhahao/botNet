import config
from train import predict, create_model
import tensorflow as tf

model_test = create_model()
model_test.load_weights(tf.train.latest_checkpoint(config.CKPT_DIR))
sentence = 'I am not crazy, my mother had me tested.'
sentence = predict(model_test, sentence)
