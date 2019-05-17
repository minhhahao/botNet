
'''
    Processing data from Cornell Corpus
'''
import os
import random
import re

import numpy as np
import config


def getLines():
    '''Attach lineID with the corresponding text'''
    id2line = {}
    file_path = os.path.join(config.DB_PATH, config.LINE_FILE)
    print(config.LINE_FILE)
    with open(file_path, 'r', errors='ignore') as f:
        counter = 0
        try:
            for line in f:
                parts = line.split(' +++$+++ ')
                if len(parts) == 5:
                    if parts[4][-1] == '\n':
                        parts[4] = parts[4][:-1]
                    id2line[parts[0]] = parts[4]
                counter += 1
        except UnicodeDecodeError:
            print(counter, line)
    return id2line


def getConvos():
    '''Get conversation from the file'''
    '''attention to the output of convo : ["L194'"", "L195'", "L196'", 'L197']'''
    file_path = os.path.join(config.DB_PATH, config.CONVO_FILE)
    convos = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            parts = line.split(' +++$+++ ')
            if len(parts) == 4:
                convo = []
                for l in parts[3][1:-2].split(', '):
                    convo.append(l[1:-1])
                convos.append(convo)
    return convos


def QA(id2line, convos):
    '''Create question-answer pair'''
    question, answer = [], []
    for c in convos:
        for index, line in enumerate(c[:-1]):
            question.append(id2line[c[index]])
            answer.append(id2line[c[index + 1]])
    assert len(question) == len(answer)
    return question, answer


def make_dir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
        pass


def prepare_data(question, answer):
    # Create path to store train & test encoder-UnicodeDecodeError
    make_dir(config.PROCESSED)
    # Random convos to create test set
    test_id = random.sample(
        [i for i in range(len(question))], config.TESTSET_SIZE)

    filenames = ['train.enc', 'train.dec', 'test.enc', 'test.dec']
    files = []

    for filename in filenames:
        files.append(open(os.path.join(config.PROCESSED, filename), 'w'))

    for i in range(len(question)):
        if i in test_id:
            files[2].write(question[i] + '\n')
            files[3].write(answer[i] + '\n')
        else:
            files[0].write(question[i] + '\n')
            files[1].write(answer[i] + '\n')

    for file in files:
        file.close()


def tokenizer(line, normalize_digits=True):
    '''
    [Usage]: tokenise word into token for model.py
    '''
    line = re.sub('<u>', '', line)
    line = re.sub('</u>', '', line)
    line = re.sub('\[', '', line)
    line = re.sub('\]', '', line)
    words = []
    _WORD_SPLIT = re.compile("([.,!?\"'-<>:;)(])")
    _DIGIT_RE = re.compile(r'\d')
    for frag in line.strip().lower().split():
        for token in re.split(_WORD_SPLIT, frag):
            if not token:
                continue
            if normalize_digits:
                token = re.sub(_DIGIT_RE, '#', token)
            words.append(token)
    return words


def vocab_builder(filename, normalize_digits=True):
    in_path = os.path.join(config.PROCESSED, filename)
    out_path = os.path.join(config.PROCESSED, 'vocab.{}'.format(filename[-3:]))

    vocab = {}
    with open(in_path, 'r') as f:
        for line in f.readlines():
            for token in tokenizer(line):
                if token not in vocab:
                    vocab[token] = 0
                vocab[token] += 1

    sorted_vocab = sorted(vocab, key=vocab.get, reverse=True)
    with open(out_path, 'w') as f:
        f.write('<pad>' + '\n')
        f.write('<unk>' + '\n')
        f.write('<s>' + '\n')
        f.write('</s>' + '\n')
        index = 4
        for word in sorted_vocab:
            if vocab[word] < config.THRESHOLD:
                break
            f.write(word + '\n')
            index += 1
        with open('config.py', 'a') as cf:
            if filename[-3:] == 'enc':
                cf.write('ENC_VOCAB = ' + str(index) + '\n')
            else:
                cf.write('DEC_VOCAB = ' + str(index) + '\n')


def vocab_load(vocab_path):
    with open(vocab_path, 'r') as f:
        words = f.read().splitlines()
    return words, {words[i]: i for i in range(len(words))}


def sentence2id(vocab, line):
    return [vocab.get(token, vocab['<unk>']) for token in tokenizer(line)]


def token2id(data, mode):
    '''Convert all token in data into corresponding index in vocabulary'''
    vocab_path = 'vocab.' + mode
    in_path = data + '.' + mode
    out_path = data + '.' + mode

    _, vocab = vocab_load(os.path.join(config.PROCESSED, vocab_path))
    in_file = open(os.path.join(config.PROCESSED, in_path), 'r')
    out_file = open(os.path.join(config.PROCESSED, out_path), 'w')

    lines = in_file.read().splitlines()
    for line in lines:
        if mode == 'dec':  # <s> </s> for encoder only
            ids = [vocab['<s>']]
        else:
            ids = []
        ids.extend(sentence2id(vocab, line))
        if mode == 'dec':
            ids.append(vocab['<\s>'])
        out_file.write(' '.join(str(id_) for id_ in ids) + '\n')


def prepare_raw_data():
    print('Preparing RAW data into train set and test set ...')
    id2line = getLines()
    convos = getConvos()
    question, answer = QA(id2line, convos)
    prepare_data(question, answer)


def process_data():
    print('Preparing data to be model-ready ...')
    vocab_builder('train.enc')
    vocab_builder('train.dec')
    token2id('train', 'enc')
    token2id('train', 'dec')
    token2id('test', 'enc')
    token2id('test', 'dec')


def load_data(enc_filename, dec_filename, max_training_size=None):
    encode_file = open(os.path.join(config.PROCESSED, enc_filename), 'r')
    decode_file = open(os.path.join(config.PROCESSED, dec_filename), 'r')

    encode, decode = encode_file.readline(), decode_file.readline()

    data_bucket = [[] for _ in config.BUCKETS]
    i = 0
    while encode and decode:
        if (i + 1) % 10000 == 0:
            print('Bucketing data number', i)
        encode_ids = [int(id_) for id_ in encode.split()]
        decode_ids = [int(id_) for id_ in decode.split()]
        for bucket_id, (encode_max_size, decode_max_size) in enumerate(config.BUCKETS):
            if len(encode_ids) <= encode_max_size and len(decode_ids) <= decode_max_size:
                data_bucket[bucket_id].append([encode_ids, decode_ids])
                break
        # encode, decode = encode_file.readline(), decode_file.readline()
        i += 1
    return data_bucket


def _pad_input(input_, size):
    return input_ + [config.PAD] * (size - len(input_))


def _reshape_batch(inputs, size, batch_size):
    """ Create batch-major inputs (reindexed inputs)"""
    batch_inputs = []
    for length_id in range(size):
        batch_inputs.append(np.array([inputs[batch_id][length_id]
                                      for batch_id in range(batch_size)], dtype=np.int32))
        return batch_inputs


def get_batch(data_bucket, bucket_id, batch_size=1):
    '''Return one batch to fetch into the model'''
    encoder_size, decoder_size = config.BUCKETS[bucket_id]
    encoder_inputs, decoder_inputs = [], []

    for _ in range(batch_size):
        encoder_inputs, decoder_inputs = random.choice(data_bucket)
        # including some path for both encoder and decoder, reverse encoder
        encoder_inputs.append(list(reversed(_pad_input(encoder_inputs, encoder_size))))
        decoder_inputs.append(_pad_input(decoder_inputs, decoder_size))
        # creates batch-major vectors from the selected data
        batch_encoder_input = _reshape_batch(encoder_inputs, encoder_size, batch_size)
        batch_decoder_input = _reshape_batch(decoder_inputs, decoder_size, batch_size)
        # create a decoder_masks = 0 for the decoders that are padding
        batch_masks = []
        for length_id in range(decoder_size):
            batch_mask = np.ones(batch_size, dtypes=np.float32)
            for batch_id in range(batch_size):
                # if target is the PAD symbol set mask = 0
                # the corresponding decoder is decoder_input shifted 1 foward
                if length_id < decoder_size -1:
                    target = decoder_inputs[batch_id][length_id+1]
                if length_id == decoder_size -1 or target == config.PAD:
                    batch_mask[batch_id] = 0.0
            batch_masks.append(batch_mask)
        return batch_encoder_input, batch_decoder_input, batch_masks


if __name__ == '__main__':
    prepare_raw_data()
    process_data()
