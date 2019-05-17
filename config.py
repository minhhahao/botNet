'''
    [Usage]: hyperparameters for the model
'''
import os

# parameters for processing the dataset
DB_PATH = os.path.join(os.getcwd(), 'database', 'cornell_movie_dialogs_corpus')
CONVO_FILE = 'movie_conversations.txt'
LINE_FILE = 'movie_lines.txt'
OUTPUT_FILE = os.path.join(os.getcwd(), 'database', 'output.txt')
PROCESSED = os.path.join('database', 'processed')
CPT_PATH = 'checkpoints'

THRESHOLD = 2

PAD = 0
UNK = 1
START = 2
EOS = 3

TESTSET_SIZE = 25000

BUCKETS = [(19, 19), (28, 28), (33, 33), (40, 43), (50, 53), (60, 63)]


CONTRACTIONS = [("i ' m ", "i 'm "), ("' d ", "'d "), ("' s ", "'s "),
                ("don ' t ", "do n't "), ("didn ' t ",
                                          "did n't "), ("doesn ' t ", "does n't "),
                ("can ' t ", "ca n't "), ("shouldn ' t ",
                                          "should n't "), ("wouldn ' t ", "would n't "),
                ("' ve ", "'ve "), ("' re ", "'re "), ("in ' ", "in' ")]

NUM_LAYERS = 3
HIDDEN_SIZE = 256
BATCH_SIZE = 64

LR = 0.5
MAX_GRAD_NORM = 5.0

NUM_SAMPLES = 512
ENC_VOCAB = 24484
DEC_VOCAB = 24577
