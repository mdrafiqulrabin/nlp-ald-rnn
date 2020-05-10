import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

import config as cf
import helper as hp

# Multi-layer Bidirectional GRU
class MulBiGRULayer(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers, drop_ratio):
        super(MulBiGRULayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_dim, hidden_dim//2, bidirectional=True,
                          num_layers=num_layers, dropout=drop_ratio)

    def forward(self, vectors, mask):
        vectors = vectors.float()
        batch_size = vectors.size(0)
        maxlen = vectors.size(1)
        lengths = mask.sum(-1)

        gru_out, _ = self.gru(vectors)  # (batch, seq_len, hidden_dim)
        assert gru_out.size(0) == batch_size
        assert gru_out.size(1) == maxlen
        assert gru_out.size(2) == self.hidden_dim

        # Separate the directions of the GRU
        gru_out = gru_out.view(batch_size, maxlen, 2, self.hidden_dim//2)

        # Pick up the last hidden state per direction
        fw_last_hn = gru_out[range(batch_size), lengths - 1, 0]  # (batch, hidden//2)
        bw_last_hn = gru_out[range(batch_size), 0, 1]  # (batch, hidden//2)

        last_hn = torch.cat([fw_last_hn, bw_last_hn], dim=1)  # (batch, hidden//2) -> (batch, hidden)

        return {'output': last_hn, 'outputs': gru_out}

# RNNClassifier with Encoder and Extractor
class RNNClassifier(nn.Module):

    def __init__(self, encoder, extractor):
        super(RNNClassifier, self).__init__()
        self.encoder = encoder
        self.extractor = extractor
        self.classifier = nn.Linear(extractor.hidden_dim, cf.OUTPUT_DIM)

    def forward(self, tokens, targets=None):
        encoded = self.encoder(tokens)
        extracted = self.extractor(encoded['encoded'], encoded['mask'])
        f = nn.Softmax(dim=1)
        output = f(self.classifier(extracted['output']))
        return output

# Model Handler for MulBiGRU
class MulBiGRUHandler:

    def __init__(self, encoder, device):
        self.encoder = encoder
        self.device = device

    # Init Model
    def get_model(self):

        # Set the MulBiGRULayer
        model_layer = MulBiGRULayer(self.encoder.emb_dim, hidden_dim=cf.HIDDEN_DIM,
                                    num_layers=cf.HIDDEN_LAYER, drop_ratio=cf.DROPOUT_RATIO)
        hp.saveLogMsg("\nModel Layer = {}".format(model_layer))

        # Set the RNNClassifier
        model = RNNClassifier(self.encoder, model_layer)
        if torch.cuda.is_available():
            model = model.to(self.device)
        hp.saveLogMsg("\nModel = {}".format(model))

        return model

    # Train the Model
    def train(self, model, optimizer, shuffled_train_set, batch_size):

        model.train()

        total_loss = 0
        batch_tokens, batch_target = [], []

        random.Random(cf.MANUAL_SEED).shuffle(shuffled_train_set)

        for i in range(len(shuffled_train_set)):

            batch_tokens.append(shuffled_train_set[i]['Tokens'])
            batch_target.append([shuffled_train_set[i]['Label']])

            if len(batch_tokens) == batch_size or i == len(shuffled_train_set) - 1:

                optimizer.zero_grad()

                out = model(batch_tokens)

                y_pred = None
                y_target = torch.tensor(batch_target).long()

                if torch.cuda.is_available():
                    y_pred = out.cuda()
                    y_target = y_target.squeeze()
                    if len(batch_tokens) == 1:
                        y_target = y_target.unsqueeze(0)
                    loss_function = nn.CrossEntropyLoss()
                    loss_function.to(self.device)
                    loss = loss_function(y_pred, y_target)
                else:
                    y_pred = out.cpu()
                    y_target = y_target.squeeze()
                    if len(batch_tokens) == 1:
                        y_target = y_target.unsqueeze(0)
                    loss = F.cross_entropy(y_pred, y_target)

                total_loss += loss.item()

                loss.backward()
                optimizer.step()

                batch_tokens, batch_target = [], []

        return model, shuffled_train_set

    # Looping for train
    def training_loop(self, model, train_set, dev_set):

        optimizer = optim.SGD(model.parameters(), lr=cf.LEARNING_RATE, momentum=cf.MOMENTUM)

        shuffled_train_set = train_set
        best_f1 = 0
        patience_track = 0

        for epoch in range(cf.EPOCH):

            epoch_msg = '[Epoch {}] / {}'.format(epoch + 1, cf.EPOCH)

            model, shuffled_train_set = self.train(model, optimizer, shuffled_train_set, batch_size=cf.BATCH_SIZE)

            _, _, train_loss, train_f1, train_acc = self.evaluate(model, shuffled_train_set, batch_size=cf.BATCH_SIZE)
            epoch_msg += ' [TRAIN] Loss: {:.4f}, F1: {:.4f}, Acc: {:.4f}'.format(train_loss, train_f1, train_acc)
            _, _, dev_loss, dev_f1, dev_acc = self.evaluate(model, dev_set, batch_size=cf.BATCH_SIZE)
            epoch_msg += ' [DEV] Loss: {:.4f}, F1: {:.4f}, Acc: {:.4f}'.format(dev_loss, dev_f1, dev_acc)

            best_f1, epoch_track, patience_track = hp.track_best_model(cf.MODEL_PATH, model, epoch + 1,
                                                                       best_f1, dev_f1, dev_acc, dev_loss,
                                                                       patience_track)
            hp.saveLogMsg(epoch_msg + epoch_track)
            if patience_track == int(cf.PATIENCE):
                hp.saveLogMsg('\nNo accuracy improvement for {} consecutive epochs, stopping training!'.format(cf.PATIENCE))
                break

        hp.saveLogMsg('Done Training.')

        state = torch.load(cf.MODEL_PATH)
        model.load_state_dict(state['model'])

        hp.saveLogMsg('\nReturning best model - [DEV] epoch {}, loss {:.4f}, f1-score {:.4f}, accuracy {:.4f}'.
                      format(state['epoch'], state['loss'], state['f1'], state['acc']))

        return model

    # Evaluate the Model
    def evaluate(self, model, dev_set, batch_size):

            model.eval()
            total_loss = 0
            batch_tokens, batch_target = [], []
            predictions, actual = [], []

            with torch.no_grad():
                for i in range(len(dev_set)):

                    batch_tokens.append(dev_set[i]['Tokens'])
                    batch_target.append([dev_set[i]['Label']])

                    if len(batch_tokens) == batch_size or i == len(dev_set) - 1:

                        out = model(batch_tokens)

                        y_pred = None
                        y_target = torch.tensor(batch_target).long()

                        actual.extend(batch_target)
                        if torch.cuda.is_available():
                            y_pred = out.cuda()
                            y_target = y_target.squeeze()
                            if len(batch_tokens) == 1:
                                y_target = y_target.unsqueeze(0)
                            predictions.extend(torch.argmax(y_pred, dim=1).tolist())
                            loss_function = nn.CrossEntropyLoss()
                            loss_function.to(self.device)
                            loss = loss_function(y_pred, y_target)
                        else:
                            y_pred = out.cpu()
                            y_target = y_target.squeeze()
                            if len(batch_tokens) == 1:
                                y_target = y_target.unsqueeze(0)
                            predictions.extend(np.argmax(y_pred, axis=1).tolist())
                            loss = F.cross_entropy(y_pred, y_target)

                        total_loss += loss.item()

                        batch_tokens, batch_target = [], []

            dev_f1 = f1_score(actual, predictions, average='weighted')
            dev_acc = accuracy_score(actual, predictions)

            return actual, predictions, total_loss / len(dev_set), dev_f1, dev_acc

    def predict(self, model, test_set, batch_size):

        model.eval()
        batch_ids, batch_tokens = [], []
        ids, predictions = [], []

        with torch.no_grad():
            for i in range(len(test_set)):

                batch_ids.append(test_set[i]['ID'])
                batch_tokens.append(test_set[i]['Tokens'])

                if len(batch_tokens) == batch_size or i == len(test_set) - 1:

                    out = model(batch_tokens)

                    y_pred = None
                    ids.extend(batch_ids)

                    if torch.cuda.is_available():
                        y_pred = out.cuda()
                        predictions.extend(torch.argmax(y_pred, dim=1).tolist())
                    else:
                        y_pred = out.cpu()
                        predictions.extend(np.argmax(y_pred, axis=1).tolist())

                    batch_ids, batch_tokens = [], []

        return ids, predictions

if __name__ == '__main__':
    pass