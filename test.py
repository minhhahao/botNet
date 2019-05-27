from train import predict, model_name
import tensorflow as tf
import os

model_path = os.path.join(os.getcwd(), 'data', 'save', model_name)
model = tf.keras.models.load_model(model_name)
sentence = 'I am not crazy, my mother had me tested.'
sentence = predict(model, sentence)
