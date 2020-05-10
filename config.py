
# Path and Files
ROOT_PATH = "/Users/mdrafiqulrabin/Desktop/nlp-ald-rnn/"
DATA_PATH = "temp/data/raw/"

TRAIN_FILE = ROOT_PATH + DATA_PATH + "train.csv"
DEV_FILE   = ROOT_PATH + DATA_PATH + "dev.csv"
TEST_FILE  = ROOT_PATH + DATA_PATH + "test.csv"

# Output/Target
TEST_LABELED = True
TRUE_LABELS  = {"NAG":0, 0:"NAG", "OAG":1, 1:"OAG", "CAG":2, 2:"CAG"}
NUM_TARGET   = 3

# Glove File
GLOVE_FILE = ROOT_PATH + "temp/data/glove.6B/glove.6B.50d.txt"
PAD_TOKEN, UNK_TOKEN = '<PAD>', '<UNK>'
PAD_INDEX, UNK_INDEX = 0, 1
MAX_SEQUENCE_LEN = 30

# Model Params
MANUAL_SEED = 42

BATCH_SIZE = 30
EPOCH = 1000
PATIENCE = 10

OUTPUT_DIM = NUM_TARGET
LEARNING_RATE = 1e-2
MOMENTUM = 0.99
HIDDEN_LAYER = 2
HIDDEN_DIM = 128
DROPOUT_RATIO = 0.3

# Model Logs
MODE = "test" #train/test
TITLE = "mbgru"

RESULT_PATH  = ROOT_PATH   + "temp/result/"
LOG_PATH     = RESULT_PATH + TITLE + ".log"
MODEL_PATH   = RESULT_PATH + TITLE + ".model"
PREDICT_PATH = RESULT_PATH + "predict.csv"
