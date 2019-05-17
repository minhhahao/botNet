'''
    tob (him/her)self
'''
import dataset
import config
from botNet import tob

import tensorflow as tf
import numpy as np

import time
import sys
import random
import argparse
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def _get_random_bucket(train_buckets_scale):
    '''
    Get a random bucket from which to choose a training sample
    '''
    rand = random.random()
    return min([i for i in range(len(train_buckets_scale))
                if train_buckets_scale[i] > rand])


def _assert_length(encoder_size, decoder_size, encoder_inputs, decoder_inputs, decoder_masks):
    '''
    Assert that the encoder inputs, decoder inputs, and decoder masks are
    of the expected lengths
    '''
    if len(encoder_inputs) != encoder_size:
        raise ValueError("Encoder length must be equal to the one in bucket,"
                         " %d != %d." % (len(encoder_inputs), encoder_size))
    if len(decoder_inputs) != decoder_size:
        raise ValueError("Decoder length must be equal to the one in bucket,"
                         " %d != %d." % (len(decoder_inputs), decoder_size))
    if len(decoder_masks) != decoder_size:
        raise ValueError("Weights length must be equal to the one in bucket,"
                         " %d != %d." % (len(decoder_masks), decoder_size))


def run_step(sess, model, encoder_inputs, decoder_inputs, decoder_masks, bucket_id, forward_only):
    '''
    Run one step in training.
    @forward_only: boolean value to decide whether a backward path should be
    created forward_only is set to True when evaluate on the test set, or when
    the bot is in chat mode.
    '''
    encoder_size, decoder_size = config.BUCKETS[bucket_id]
    _assert_length(encoder_size, decoder_size, encoder_inputs,
                   decoder_inputs, decoder_masks)

    # input feed: encoder_input, decoder_input, target_weights, as provided
    input_feed = {}
    for step in range(encoder_size):
        input_feed[model.encoder_inputs[step].name] = encoder_inputs[step]
    for step in range(decoder_size):
        input_feed[model.decoder_inputs[step].name] = decoder_inputs[step]
        input_feed[model.decoder_masks[step].name] = decoder_masks[step]

    last_target = model.decoder_inputs[decoder_size].name
    input_feed[last_target] = np.zeros([model.batch_size], dtype=np.int32)

    # output feed: depends on whether backward step is done or not
    if not forward_only:
        output_feed = [model.train_ops[bucket_id],  # update op that does SGD.
                       model.gradient_norms[bucket_id],  # gradient norm.
                       model.losses[bucket_id]]  # loss for this batch.
    else:
        output_feed = [model.losses[bucket_id]]  # loss for this batch.
        for step in range(decoder_size):  # output logits.
            output_feed.append(model.outputs[bucket_id][step])

    outputs = sess.run(output_feed, input_feed)
    if not forward_only:
        return outputs[1], outputs[2], None  # Gradient norms, loss, no output
    else:
        # No gradient norms, loss, outputs
        return None, outputs[0], outputs[1:]


def _get_buckets():
    '''
    Load the dataset into buckets based on their lengths.
    Interval: train_buckets_scale to randomise bucket later on
    '''
    test_buckets = dataset.load_data('test_ids.enc', 'test_ids.dec')
    data_buckets = dataset.load_data('train_ids.enc', 'train_ids.dec')
    train_buckets_size = [len(data_buckets[b])
                          for b in range(len(config.BUCKETS))]
    print('Number of sample in each buckets:\n', train_buckets_size)
    train_total_size = sum(train_buckets_size)
    # list of increasing # from 0 to 1 that will used to choose buckets
    train_buckets_scale = [sum(train_buckets_size[:i + 1]) / train_total_size
                           for i in range(len(train_buckets_size))]
    print('Bucket scale:\n', train_buckets_scale)
    return test_buckets, data_buckets, train_buckets_scale


def _get_skip_step(iteration):
    '''
    # of steps should the model train before saving all the weights
    '''
    if iteration < 100:
        return 30
    return 100


def _check_restore_parameters(sess, saver):
    '''
    Restore previous trained parameters if there is any
    '''
    ckpt = tf.train.get_checkpoint_state(
        os.path.dirname(config.CPT_PATH + '/checkpoint'))
    if ckpt and ckpt.model_checkpoint_path:
        print('Loading parameters for tob')
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print('Initialising fresh parameters for tob')


def _eval_test_set(sess, model, test_buckets):
    '''
    Evaluate on test set
    '''
    for bucket_id in range(len(config.BUCKETS)):
        if len(test_buckets[bucket_id]) == 0:
            print(' Test: empty bucket %d' % (bucket_id))
            continue
        start = time.time()
        encoder_inputs, decoder_inputs, decoder_masks = dataset.get_batch(
            test_buckets[bucket_id], bucket_id, batch_size=config.BATCH_SIZE)
        _, step_loss, _ = run_step(
            sess, model, encoder_inputs, decoder_inputs, decoder_masks, bucket_id, True)
        print('Test bucket {}: loss {} time {}'.format(
            bucket_id, step_loss, time.time() - start))


def train():
    '''
    Train tob
    '''
    test_buckets, data_buckets, train_buckets_scale = _get_buckets()
    # need backward pass, therefore, forward_only=False
    model = tob(False, config.BATCH_SIZE)
    model.build_graph()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        print('Running session')
        sess.run(tf.global_variables_initializer())
        _check_restore_parameters(sess, saver)

        iteration = model.global_step.eval()
        total_loss = 0
        while True:
            skip_step = _get_skip_step(iteration)
            bucket_id = _get_random_bucket(train_buckets_scale)
            encoder_inputs, decoder_inputs, decoder_masks = dataset.get_batch(data_buckets[bucket_id],
                                                                              bucket_id,
                                                                              batch_size=config.BATCH_SIZE)
            start = time.time()
            _, step_loss, _ = run_step(
                sess, model, encoder_inputs, decoder_inputs, decoder_masks, bucket_id, False)
            total_loss += step_loss
            iteration += 1

            if iteration % skip_step == 0:
                print('Iter {}: loss {}, time {}'.format(
                    iteration, total_loss / skip_step, time.time() - start))
                start = time.time()
                total_loss = 0
                saver.save(sess, os.path.join(config.CPT_PATH, 'tob'),
                           global_step=model.global_step)
                if iteration % (10 * skip_step) == 0:
                    # Run evals on development set and print their loss
                    _eval_test_set(sess, model, test_buckets)
                    start = time.time()
                sys.stdout.flush()


def _get_user_input():
    '''
    User input, which will transformed into enconder later on
    '''
    print('> ', end='')
    sys.stdout.flush()
    return sys.stdin.readline()


def _find_right_bucket(length):
    '''
    Find proper bucket for encoder input based on the length
    '''
    return min([b for b in range(len(config.BUCKETS))
                if config.BUCKETS[b][0] >= length])


def _construct_response(output_logits, inv_dec_vocab):
    '''
    Construct a response to users' encoder input
    @ output_logits: the outputs from seq2seq wrapper
    output_logits is decoder_size np array, each of dimension 1 x DEC_VOCAB
    Greedy Decoder - outputs are just argmaxes of output_logits, following
    the model implemented from tensorflow/nmt
    '''
    print(output_logits[0])
    outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
    # if EOS exists, cut it off
    if config.EOS_ID in outputs:
        outputs = outputs[:outputs.index(config.EOS_ID)]
    # Print sentence corresponding to the output
    return ' '.join([tf.compat.as_str(inv_dec_vocab[output]) for output in outputs])


def chat():
    '''
    Test mode, forward_only=True
    '''
    _, enc_vocab = dataset.load_vocab(
        os.path.join(config.PROCESSED_PATH, 'vocab.enc'))
    inv_dec_vocab, _ = dataset.load_vocab(
        os.path.join(config.PROCESSED_PATH, 'vocab.dec'))

    model = tob(True, batch_size=1)
    model.build_graph()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        _check_restore_parameters(sess, saver)
        output_file = open(os.path.join(
            config.PROCESSED_PATH, config.OUTPUT_FILE), 'a+')
        # Decode from standard output
        max_length = config.BUCKETS[-1][0]
        print('Hi, my name jeff. exiting @ max length: ', max_length)
        while True:
            line = _get_user_input()
            if len(line) > 0 and line[-1] == '\n':
                line = line[:-1]
            if line == '':
                break
            output_file.write('Human: ' + line + '\n')
            # return token_ids for the input sentence
            token_ids = dataset.sentence2id(enc_vocab, str(line))
            if (len(token_ids) > max_length):
                print("I'm kinda tard, can only handle max length @ ", max_length)
                line = _get_user_input()
                continue
            # return corresponding bucket
            bucket_id = _find_right_bucket(len(token_ids))
            # return a 1-element batch to feed into the model
            encoder_inputs, decoder_inputs, decoder_masks = dataset.get_batch([(token_ids, [])],
                                                                              bucket_id,
                                                                              batch_size=1)
            # return output logits for the sentence
            _, _, output_logits = run_step(sess, model, encoder_inputs, decoder_inputs,
                                           decoder_masks, bucket_id, True)
            response = _construct_response(output_logits, inv_dec_vocab)
            print(response)
            output_file.write('tob: ' + response + '\n')
        output_file.write('================\n')
        output_file.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices={'train', 'chat'}, default='train',
                        help='mode. default: train mode')
    args = parser.parse_args()

    if not os.path.isdir(config.PROCESSED_PATH):
        dataset.prepare_raw_data()
        dataset.process_data()
    print('Finished processing data!')
    # create checkpoint folder if there isn't one
    dataset.make_dir(config.CPT_PATH)

    if args.mode == 'train':
        train()
    elif args.mode == 'chat':
        chat()


if __name__ == '__main__':
    main()
