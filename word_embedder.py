import os
import torch
import torch.nn as nn
import numpy as np
import config as cf
from data_loader import TweetDataset
from data_preprocessor import TweetFilter

class GloVeEmbedder(nn.Module):
    def __init__(self, vocab_tokens, glove_file, fixed_len):
        super(GloVeEmbedder, self).__init__()
        assert os.path.exists(glove_file) and glove_file.endswith('.txt'), glove_file

        self.emb_dim   = None

        if fixed_len:
            self.sequence_len = cf.MAX_SEQUENCE_LEN
        else:
            self.sequence_len = -1

        self.PAD_TOKEN = cf.PAD_TOKEN
        self.UNK_TOKEN = cf.UNK_TOKEN

        idx2tok = [self.PAD_TOKEN, self.UNK_TOKEN]
        idx2vec = [None, None]

        with open(glove_file, 'r') as fp:
            for line in fp:
                line = line.split()
                if line[0] not in vocab_tokens:
                    continue

                tok, vec = '', []
                try:
                    tok = line[0]
                    vec = np.array([float(value) for value in line[1:]])
                except:
                    continue

                if self.emb_dim is None:
                    self.emb_dim = vec.shape[0]

                idx2tok.append(tok)
                idx2vec.append(vec)

        if len(idx2vec) == 2:
            print("No match with glove !!")
            idx2vec[0] = np.zeros(self.sequence_length)
            idx2vec[1] = np.zeros(self.sequence_length)
            self.emb_dim = self.sequence_length
        else:
            idx2vec[0] = np.zeros(self.emb_dim)
            idx2vec[1] = np.mean(idx2vec[2:], axis=0)

        self.embeddings = torch.from_numpy(np.array(idx2vec)).float()
        self.embeddings = nn.Embedding.from_pretrained(self.embeddings, freeze=False)

        self.idx2word = {i: t for i, t in enumerate(idx2tok)}
        self.word2idx = {t: i for i, t in self.idx2word.items()}

    def forward(self, samples):
        pad_idx = self.word2idx[self.PAD_TOKEN]
        unk_idx = self.word2idx[self.UNK_TOKEN]

        if self.sequence_len > 0:
            maxlen = self.sequence_len
        else:
            maxlen = max([len(s) for s in samples])

        encoded = [[self.word2idx.get(token, unk_idx) for token in tokens] for tokens in samples]

        padded = np.zeros((len(samples), maxlen), dtype=int)
        masks = torch.zeros(len(samples), maxlen).long()

        for i in range(len(encoded)):
            padded[i, :len(encoded[i])] = np.array(encoded[i])[:maxlen]
            masks[i, :len(encoded[i])] = 1

        encoded = torch.tensor(padded).long()

        if torch.cuda.is_available():
            encoded = encoded.cuda()
            masks = masks.cuda()

        result = {
            'encoded': self.embeddings(encoded),
            'mask': masks,
        }

        return result

if __name__ == '__main__':
    sample_tweets = [
        "This is line number-1",
        "oops! my god ..."
    ]
    print(sample_tweets[0])

    tweet_tokens = []
    for t in sample_tweets:
        tokens = TweetFilter().get_tokens(t)
        tweet_tokens.append(tokens)
    print(tweet_tokens[0])

    my_obj = TweetDataset()
    train_set, dev_set, test_set = my_obj.get_dataset()
    train_set, dev_set, test_set = train_set[:10], dev_set[:10], test_set[:10]
    vocab_tokens = my_obj.get_vocabulary(train_set, dev_set)
    glove_embedder = GloVeEmbedder(vocab_tokens, glove_file=cf.GLOVE_FILE, fixed_len=True)
    result = glove_embedder(tweet_tokens)
    print(result['encoded'][0])
