import config as cf
import os, json
import pandas as pd
from data_preprocessor import TweetFilter

class TweetDataset(object):

    def __init__(self):
        pass

    def get_dataset(self):
        datasets   = []
        data_files = [cf.TRAIN_FILE, cf.DEV_FILE, cf.TEST_FILE]
        for fname in data_files:
            tdf = pd.read_csv(fname, encoding='latin-1')
            tdf["Tokens"] = tdf["Comment"].apply(TweetFilter().get_tokens)
            if "Label" in tdf.columns:
                tdf["Label"] = tdf["Label"].apply(lambda x: cf.TRUE_LABELS[x])
            tjs = tdf.to_json(orient='records')
            tjs = json.loads(tjs)
            datasets.append(tjs)
        train_set, dev_set, test_set = datasets[0], datasets[1], datasets[2]
        return train_set, dev_set, test_set

    def get_vocabulary(self, train_set, dev_set):
        train_tokens = [sample['Tokens'] for sample in train_set]
        flat_train_tokens = sum(train_tokens, [])
        dev_tokens = [sample['Tokens'] for sample in dev_set]
        flat_dev_tokens = sum(dev_tokens, [])
        vocab_tokens = list(set(flat_train_tokens + flat_dev_tokens))
        return vocab_tokens

if __name__ == '__main__':
    my_obj = TweetDataset()
    train_set, dev_set, test_set = my_obj.get_dataset()
    vocab_tokens = my_obj.get_vocabulary(train_set, dev_set)
    print(len(vocab_tokens))

