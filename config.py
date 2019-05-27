import os

# Init path
DATA_PATH = os.path.join('data', 'cornell')
CKPT_PATH = os.path.join('data', 'save', 'model-Epochs{epoch:04d}-maxlen{maxlen:}-layers{layers:}-dropout{dropout:}.ckpt')
CKPT_DIR = os.path.dirname(CKPT_PATH)
LINES_FILE = 'movie_lines.txt'
CONVERSATIONS_FILE = 'movie_conversations.txt'

# Data processing parameters
MAX_SAMPLES = 50000
MAX_LENGTH = 40
BATCH_SIZE = 64
BUFFER_SIZE = 30000

# Hyper-parameters
NUM_LAYERS = 3
D_MODEL = 512
NUM_HEADS = 8
UNITS = 1024
DROPOUT = 0.1
EPOCHS = 200  # set 20 for first time training
