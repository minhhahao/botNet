import pandas as pd
import numpy as np
import tensorflow as tf

import re
import time
import os

import config


def get_files():
    '''
        Usage: get inputs from source file
    '''
    line_path = os.path.join(config.DATA_PATH, config.LINE_FILE)
    conv_path = os.path.join(config.DATA_PATH, config.CONV_FILE)
    lines = open(line_path, encoding='utf-8',
                 errors='ignore').read().split('\n')
    conv_lines = open(conv_path, encoding='utf-8',
                      errors='ignore').read().split('\n')
    return lines, conv_lines


def id2line():
    '''
        Create a dictionary to map each line's id with its text
    '''
    id2line = {}
    lines, _ = get_files()
    i = 0
    try:
        for line in lines:
            parts = line.split(' +++$+++ ')
            if len(parts) == 5:
                id2line[parts[0]] = parts[4]
            i += 1
    except UnicodeDecodeError:
        print(i, line)
    return id2line


def convs():
    '''
        Create a list of all of the conversations' lines' ids.
    '''
    convs = []
    _, convs_lines = get_files()
    for line in convs_lines[:-1]:
        _line = line.split(
            ' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ", "")
        convs.append(_line.split(','))
    return convs


def clean_text(text):
    '''
        Formating text
    '''
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "that is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)

    return text


def questions_answers(id2line, convs):
    '''
        Sort the sentences into questions (inputs) and answers (targets)
        Clean the text using clean_text(text)
    '''
    questions, answers = [], []
    formatted_questions, formatted_answers = [], []
    for conv in convs:
        for index, line in enumerate(conv[:-1]):
            questions.append(id2line[conv[index]])
            answers.append(id2line[conv[index + 1]])
    assert len(questions) == len(answers)
    for question in questions:
        formatted_questions.append(clean_text(question))
    for answer in answers:
        formatted_answers.append(clean_text(answer))
    return formatted_questions, formatted_answers


def sorted_qa():
    '''
        Remove too long or too short q-a
    '''
    clean_q, clean_a = questions_answers(id2line(), convs())
    sorted_q_t, sorted_a_t = [], []
    i = 0
    for question in clean_q:
        if len(question.split()) >= config.MIN_LEN and len(question.split()) <= config.MAX_LEN:
            sorted_q_t.append(question)
            sorted_a_t.append(clean_a[i])
            i += 1
    i = 0
    sorted_q, sorted_a = [], []
    for answer in sorted_a_t:
        if len(answer.split()) >= config.MIN_LEN and len(answer.split()) <= config.MAX_LEN:
            sorted_a.append(answer)
            sorted_q.append(sorted_q_t[i])
            i += 1
    print("# of questions:", len(sorted_q))
    print("# of answers:", len(sorted_a))
    print("% of data used: {}%".format(
        round(len(sorted_q) / len(clean_q), 4) * 100))
    return sorted_q, sorted_a


def create_vocab():
    '''
        Create vocab
    '''
    vocab = {}
    questions, answers = sorted_qa()
    for question in questions:
        for word in question.split():
            if word not in vocab:
                vocab[word] = 1
            else:
                vocab[word] += 1
    for answer in answers:
        for word in answer.split():
            if word not in vocab:
                vocab[word] = 1
            else:
                vocab[word] += 1

    # remove rare words from vocab to reduce training time
    count = 0
    for k, v in vocab.items():
        if v >= config.THRESHOLD:
            count += 1
    print("Size of total vocab:", len(vocab))
    print("Size of vocab we will use:", count)
    return vocab


def formatted_vocab():
    '''
    In case we want to use a different vocabulary sizes for the source and
    target text, we can set different threshold values. Nonetheless,
    we will create dictionaries to provide a unique integer for each word.
    '''
    vocab = create_vocab()
    questions_vocab_2_int, answers_vocab_2_int = {}, {}

    wordlen = 0
    for word, count in vocab.items():
        if count >= config.THRESHOLD:
            questions_vocab_2_int[word] = wordlen
            wordlen += 1
    word_len = 0
    for word, count in vocab.items():
        if count >= config.THRESHOLD:
            answers_vocab_2_int[word] = wordlen
            word_len += 1
    return questions_vocab_2_int, answers_vocab_2_int


def insert_token():
    '''
        Add the unique tokens to the vocabulary dictionaries.
    '''
    short_q, short_a = sorted_qa()
    q2int, a2int = formatted_vocab()
    for c in config.UNIQUE:
        q2int[c] = len(q2int) + 1
        a2int[c] = len(a2int) + 1
    qint2vocab = {v_i: v for v, v_i in q2int.items()}
    aint2vocab = {v_i: v for v, v_i in a2int.items()}

    for i in range(len(short_a)):
        short_a[i] += ' <EOS>'

    # convert str to int
    questions_int = []
    for question in short_q:
        ints = []
        for word in question.split():
            if word not in q2int:
                ints.append(q2int['<UNK>'])
            else:
                ints.append(q2int[word])
        questions_int.append(ints)

    answers_int = []
    for answer in short_a:
        ints = []
        for word in answer.split():
            if word not in a2int:
                ints.append(a2int['<UNK>'])
            else:
                ints.append(a2int[word])
        answers_int.append(ints)
    return questions_int, answers_int, qint2vocab, aint2vocab


def prepare_data():
    '''
    sorted according to length to reduce training time
    '''
    sorted_questions, sorted_answers = [], []
    qint, aint, _, _ = insert_token()
    for length in range(1, config.MAX_LEN + 1):
        for i in enumerate(qint):
            if len(i[1]) == length:
                sorted_questions.append(qint[i[0]])
                sorted_answers.append(aint[i[0]])
    return sorted_questions, sorted_answers


if __name__ == '__main__':
    prepare_data()
