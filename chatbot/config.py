import os

VERSION = 1
# Cornell Data Corpus
CHATDATA_PATH = os.path.join('data', 'cornell')
LINES_FILE = 'movie_lines.txt'
CONVERSATIONS_FILE = 'movie_conversations.txt'
# Building Dataset and Vocab
MAX_VOCAB_SIZE = 2**16
VOCAB_PATH = os.path.join('data', 'samples')
VOCAB_FILENAME = 'maxSize{:04d}-maxvocabSize{:04d}'
OUTPUT_FILE = 'output.txt'
# Log directory for Tensorboard
LOG_DIR = os.path.join('save', 'logs', str({}))

# Data processing parameters
MAX_SAMPLES = 100000
MAX_LENGTH = 50
BATCH_SIZE = 64
BUFFER_SIZE = 30000

# Hyper-parameters
NUM_LAYERS = 2
D_MODEL = 256
NUM_HEADS = 8
UNITS = 512
DROPOUT = 0.1
EPOCHS = 100
# 20 is optimum considering learning rate decay
