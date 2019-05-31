import os

VERSION = 1
# Init path
CHATDATA_PATH = os.path.join('data', 'cornell')
LINES_FILE = 'movie_lines.txt'
CONVERSATIONS_FILE = 'movie_conversations.txt'
VOCAB_FILENAME = 'vocab'
OUTPUT_FILE = 'output.txt'

# Data processing parameters
MAX_SAMPLES = 30000
MAX_LENGTH = 40
BATCH_SIZE = 64
BUFFER_SIZE = 20000

# Hyper-parameters
NUM_LAYERS = 2
D_MODEL = 256
NUM_HEADS = 8
UNITS = 512
DROPOUT = 0.1
EPOCHS = 20
# 20 is optimum considering learning rate decay
