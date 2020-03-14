import os, pathlib
import config as cf
import helper as hp
import pandas as pd

import numpy as np
np.random.seed(cf.MANUAL_SEED)
import torch
torch.manual_seed(cf.MANUAL_SEED)

from word_embedder import *
from model_handler import *

# Create Log File
pathlib.Path(cf.LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
open(cf.LOG_PATH, 'w').close()

# Set Device (cuda/cpu)
hp.saveLogMsg("\nAttaching device...")
device = None
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
hp.saveLogMsg("device = {}".format(str(device)))

# Loading Data and Vocabulary
hp.saveLogMsg("\nLoading data...")
my_tweet_dataset = TweetDataset()
train_set, dev_set, test_set = my_tweet_dataset.get_dataset()
hp.saveLogMsg("#Train={}, #Dev={}, #Test={}".format(len(train_set), len(dev_set), len(test_set)))
vocab_tokens = my_tweet_dataset.get_vocabulary(train_set, dev_set)
hp.saveLogMsg('Vocabulary size: {}'.format(len(vocab_tokens)))

# Loading GloVe File
hp.saveLogMsg("\nLoading GloVe from {}...".format(cf.GLOVE_FILE))
encoder = GloVeEmbedder(vocab_tokens, glove_file=cf.GLOVE_FILE, fixed_len=True)
hp.saveLogMsg("\nInitialized GloVe Embedder: {}".format(encoder))

# Run Model
handler = MulBiGRUHandler(encoder, device)
model   = handler.get_model()
assert model is not None

if cf.MODE == "test" and os.path.exists(cf.MODEL_PATH):
    state = torch.load(cf.MODEL_PATH)
    model.load_state_dict(state['model'])
    hp.saveLogMsg('\nLoading best model - [DEV] epoch {}, loss {:.4f}, f1-score {:.4f}, accuracy {:.4f}'.
                  format(state['epoch'], state['loss'], state['f1'], state['acc']))
else:
    model = handler.training_loop(model, train_set, dev_set)

if cf.TEST_LABELED:
    test_true, test_pred, test_loss, test_f1, test_acc = handler.evaluate(model, test_set, batch_size=cf.BATCH_SIZE)
    hp.saveLogMsg('\n[Test] Loss: {:.4f}, F1: {:.4f}, Acc: {:.4f}'.format(test_loss, test_f1, test_acc))
    clf_report = classification_report(test_true, test_pred, output_dict=False)
    hp.saveLogMsg('\n[Test] Classification Report: \n{}'.format(clf_report))
else:
    ids, predictions = handler.predict(model, test_set, batch_size=cf.BATCH_SIZE)
    df = pd.DataFrame(list(zip(ids, predictions)), columns=['ID', 'Label'])
    df["Label"] = df["Label"].apply(lambda x: cf.TRUE_LABELS[x])
    df.to_csv(cf.PREDICT_PATH, index=False)
    hp.saveLogMsg('\nSave prediction at {}'.format(cf.PREDICT_PATH))
