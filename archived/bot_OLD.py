'''
    Running code for tob
'''
import time

import tensorflow as tf
import numpy as np

import config
from data import formatted_vocab, prepare_data, clean_text, sorted_qa, insert_token
from model import model_inputs, seq2seq_model


def question_to_seq(question, vocab_to_int):
    '''Prepare the question for the model'''

    question = clean_text(question)
    return [vocab_to_int.get(word, vocab_to_int['<UNK>']) for word in question.split()]


def pad_sentence_batch(sentence_batch, vocab_to_int):
    ''''
        Pad sentences with <PAD> so that each sentence
        of a batch has the same length
    '''
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [vocab_to_int['<PAD>']] * (max_sentence - len(sentence)) for sentence in sentence_batch]


def batch_data(questions, answers, batch_size):
    '''
    Batch questions and answers together
    '''
    for batch_i in range(0, len(questions) // batch_size):
        start_i = batch_i * batch_size
        questions_batch = questions[start_i:start_i + batch_size]
        answers_batch = answers[start_i:start_i + batch_size]
        pad_questions_batch = np.array(pad_sentence_batch(
            questions_batch, questions_vocab_to_int))
        pad_answers_batch = np.array(pad_sentence_batch(
            answers_batch, answers_vocab_to_int))
        yield pad_questions_batch, pad_answers_batch


def split_data():
    '''
        split data into training and validation set
    '''
    sorted_questions, sorted_answers = prepare_data()
    # Validate the training with 10% of the data
    train_valid_split = int(len(sorted_questions) * 0.15)

    # Split the questions and answers into training and validating data
    train_questions = sorted_questions[train_valid_split:]
    train_answers = sorted_answers[train_valid_split:]

    valid_questions = sorted_questions[:train_valid_split]
    valid_answers = sorted_answers[:train_valid_split]

    return (train_questions, train_answers), (valid_questions, valid_answers)


# Reset the graph to ensure that it is ready for training
tf.reset_default_graph()
# Start the session
sess = tf.InteractiveSession()

# Load the model inputs
input_data, targets, lr, keep_prob = model_inputs()
# Sequence length will be the max line length for each batch
sequence_length = tf.placeholder_with_default(
    config.MAX_LEN, None, name='sequence_length')
# Find the shape of the input data for sequence_loss
input_shape = tf.shape(input_data)

questions_vocab_to_int, answers_vocab_to_int = formatted_vocab()

# Create the training and inference logits
train_logits, inference_logits = seq2seq_model(tf.reverse(input_data, [-1]),
                                               targets,
                                               keep_prob,
                                               config.BATCH_SIZE,
                                               sequence_length,
                                               len(answers_vocab_to_int),
                                               len(questions_vocab_to_int),
                                               config.ENCODING_EMBEDDING_SIZE,
                                               config.DECODING_EMBEDDING_SIZE,
                                               config.RNN_SIZE,
                                               config.NUM_LAYERS,
                                               questions_vocab_to_int)

# Create a tensor for the inference logits, needed if loading a checkpoint version of the model
tf.identity(inference_logits, 'logits')

with tf.name_scope("optimization"):
    # Loss function
    cost = tf.contrib.seq2seq.sequence_loss(
        train_logits,
        targets,
        tf.ones([input_shape[0], sequence_length]))

    # Optimizer
    optimizer = tf.train.AdamOptimizer(config.LEARNING_RATE)

    # Gradient Clipping
    gradients = optimizer.compute_gradients(cost)
    capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var)
                        for grad, var in gradients if grad is not None]
    train_op = optimizer.apply_gradients(capped_gradients)

# Training, Validation starts
(train_questions, train_answers), (valid_questions, valid_answers) = split_data()
# Modulus for checking validation loss
validation_check = ((len(train_questions)) // config.BATCH_SIZE // 2) - 1
summary_valid_loss = []  # Record the validation loss for saving improvements in the mode

checkpoint = "best_model.ckpt"

sess.run(tf.global_variables_initializer())
for epoch_i in range(1, config.EPOCHS + 1):
    for batch_i, (questions_batch, answers_batch) in enumerate(
            batch_data(train_questions, train_answers, config.BATCH_SIZE)):
        start_time = time.time()
        _, loss = sess.run(
            [train_op, cost],
            {input_data: questions_batch,
             targets: answers_batch,
             lr: config.LEARNING_RATE,
             sequence_length: answers_batch.shape[1],
             keep_prob: config.KEEP_PROBABILITY})

        config.TOTAL_TRAIN_LOSS += loss
        end_time = time.time()
        batch_time = end_time - start_time

        if batch_i % config.DISPLAY_STEP == 0:
            print('Epoch {:>3}/{} Batch {:>4}/{} - Loss: {:>6.3f}, Seconds: {:>4.2f}'
                  .format(epoch_i,
                          config.EPOCHS,
                          batch_i,
                          len(train_questions) // config.BATCH_SIZE,
                          config.TOTAL_TRAIN_LOSS / config.DISPLAY_STEP,
                          batch_time * config.DISPLAY_STEP))
            total_train_loss = 0

        if batch_i % validation_check == 0 and batch_i > 0:
            total_valid_loss = 0
            start_time = time.time()
            for batch_ii, (questions_batch, answers_batch) in \
                    enumerate(batch_data(valid_questions, valid_answers, config.BATCH_SIZE)):
                valid_loss = sess.run(
                    cost, {input_data: questions_batch,
                           targets: answers_batch,
                           lr: config.LEARNING_RATE,
                           sequence_length: answers_batch.shape[1],
                           keep_prob: 1})
                total_valid_loss += valid_loss
            end_time = time.time()
            batch_time = end_time - start_time
            avg_valid_loss = total_valid_loss / \
                (len(valid_questions) / config.BATCH_SIZE)
            print('Valid Loss: {:>6.3f}, Seconds: {:>5.2f}'.format(
                avg_valid_loss, batch_time))

            # Reduce learning rate, but not below its minimum value
            config.LEARNING_RATE *= config.LEARNING_RATE_DECAY
            if config.LEARNING_RATE < config.MIN_LEARNING_RATE:
                config.LEARNING_RATE = config.MIN_LEARNING_RATE

            summary_valid_loss.append(avg_valid_loss)
            if avg_valid_loss <= min(summary_valid_loss):
                print('New Record!')
                stop_early = 0
                saver = tf.train.Saver()
                saver.save(sess, checkpoint)

            else:
                print("No Improvement.")
                stop_early += 1
                if stop_early == config.STOP:
                    break

    if stop_early == config.STOP:
        print("Stopping Training.")
        break


# Create your own input question
# input_question = 'How are you?'

# Use a question from the data as your input
short_questions, short_answers = sorted_qa()
_, _, questions_int_to_vocab, answers_int_to_vocab = insert_token()
random = np.random.choice(len(short_questions))
input_question = short_questions[random]

# Prepare the question
input_question = question_to_seq(input_question, questions_vocab_to_int)

# Pad the questions until it equals the max_line_length
input_question = input_question + \
    [questions_vocab_to_int["<PAD>"]] * (config.MAX_LEN - len(input_question))
# Add empty questions so the the input_data is the correct shape
batch_shell = np.zeros((config.BATCH_SIZE, config.MAX_LEN))
# Set the first question to be out input question
batch_shell[0] = input_question

# Run the model with the input question
answer_logits = sess.run(inference_logits, {input_data: batch_shell,
                                            keep_prob: 1.0})[0]

# Remove the padding from the Question and Answer
pad_q = questions_vocab_to_int["<PAD>"]
pad_a = answers_vocab_to_int["<PAD>"]

print('Question')
print('  Word Ids:      {}'.format([i for i in input_question if i != pad_q]))
print('  Input Words: {}'.format(
    [questions_int_to_vocab[i] for i in input_question if i != pad_q]))

print('\nAnswer')
print('  Word Ids:      {}'.format(
    [i for i in np.argmax(answer_logits, 1) if i != pad_a]))
print('  Response Words: {}'.format(
    [answers_int_to_vocab[i] for i in np.argmax(answer_logits, 1) if i != pad_a]))
