'''
    [Usage]: hyperparameters for the model
'''
import os

# PATH
DATA_PATH = str(os.path.join(os.getcwd(), 'database',
                             'cornell_movie_dialogs_corpus'))
CONV_FILE = 'movie_conversations.txt'
LINE_FILE = 'movie_lines.txt'
#OUTPUT_FILE = 'output_convo.txt'
# PROCESSED_PATH = 'processed'
# CKPT_PATH = 'checkpoints'

# parameters for processing the dataset
MAX_LEN = 25
MIN_LEN = 2
THRESHOLD = 10
UNIQUE = ['<PAD>', '<EOS>', '<UNK>', '<GO>']

# hyperparameters
EPOCHS = 100
BATCH_SIZE = 128
RNN_SIZE = 512
NUM_LAYERS = 3
ENCODING_EMBEDDING_SIZE = 512
DECODING_EMBEDDING_SIZE = 512
LEARNING_RATE = 0.005
LEARNING_RATE_DECAY = 0.9
MIN_LEARNING_RATE = 0.0001
KEEP_PROBABILITY = 0.75
DISPLAY_STEP = 100  # Check training loss after every 100 batches
STOP_EARLY = 0
STOP = 5  # If the validation loss does decrease in 5 consecutive checks, stop training
TOTAL_TRAIN_LOSS = 0 # Record the training loss for each display step
