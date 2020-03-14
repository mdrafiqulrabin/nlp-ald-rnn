
# Path and Files
ROOT_PATH = "/Users/mdrafiqulrabin/Desktop/nlp-ald-rnn/"
DATA_PATH = "temp/data/raw/"

TRAIN_FILE = ROOT_PATH + DATA_PATH + "train.csv"
DEV_FILE   = ROOT_PATH + DATA_PATH + "dev.csv"
TEST_FILE  = ROOT_PATH + DATA_PATH + "test.csv"

# Output/Target
TEST_LABELED = True
TRUE_LABELS  = {"NAG":0, "OAG":1, "CAG":2}
NUM_TARGET   = 3

# Glove File
GLOVE_FILE = ROOT_PATH + "temp/data/glove.6B/glove.6B.50d.txt"
PAD_TOKEN, UNK_TOKEN = '<PAD>', '<UNK>'
PAD_INDEX, UNK_INDEX = 0, 1
MAX_SEQUENCE_LEN = 20

