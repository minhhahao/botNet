# parameters for processing the dataset
DATA_PATH = 'database\\cornell_movie_dialogs_corpus'
CONV_FILE = 'movie_conversations.txt'
LINE_FILE = 'movie_lines.txt'
OUTPUT_FILE = 'output.txt'
PROCESSED_PATH = 'processed'
CPT_PATH = 'checkpoints'

THRESHOLD = 5

PAD_ID = 0
UNK_ID = 1
START_ID = 2
EOS_ID = 3

TESTSET_SIZE = 25000

BUCKETS = [(19, 19), (25, 25), (30, 33), (40, 43), (50, 53), (60, 63)]


NUM_LAYERS = 3
HIDDEN_SIZE = 256
BATCH_SIZE = 32

LR = 0.005
MAX_GRAD_NORM = 5.0

NUM_SAMPLES = 512
ENC_VOCAB = 13188
DEC_VOCAB = 13261
ENC_VOCAB = 13195
DEC_VOCAB = 13249
ENC_VOCAB = 13180
DEC_VOCAB = 13303
