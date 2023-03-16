import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pprint import pprint
import random
import time
from sklearn.metrics import classification_report as cr
import os
import json

random.seed(time.time())

sentenceLens = {}
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

CONTEXT = 3  # Left and right context hence total context is 2*CONTEXT
sentenceLimit = 10000
BATCH_SIZE = 32
MODEL = "Test"


# CBOW
class Data(torch.utils.data.Dataset):
    def __init__(self, sentences):
        self.device = device
        self.context = CONTEXT
        self.sentences = [sentence for sentence in sentences if len(sentence) > 2 * self.context]
        idxPtr = 0
        self.w2idx = {}
        self.idx2w = {}
        for sentence in self.sentences:
            for word in sentence:
                if word not in self.w2idx:
                    self.w2idx[word] = idxPtr
                    self.idx2w[idxPtr] = word
                    idxPtr += 1
        self.vocab_size = len(self.w2idx)

        self.inputs = []
        self.outputs = []

        for sentence in self.sentences:
            for i in range(self.context, len(sentence) - self.context):
                self.inputs.append(sentence[i - self.context:i] + sentence[i + 1:i + self.context + 1])
                self.outputs.append(sentence[i])
        # conver to idxs
        self.inputs = [[self.w2idx[word] for word in sentence] for sentence in self.inputs]
        self.outputs = [self.w2idx[word] for word in self.outputs]

        # convert to tensor
        self.inputs = torch.tensor(self.inputs).to(self.device)
        self.outputs = torch.tensor(self.outputs).to(self.device)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]


class NN(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super(NN, self).__init__()  # call the init function of the parent class
        self.device = device
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.softmax = nn.Softmax(dim=1)
        self.elayer = nn.Embedding(self.vocab_size, self.hidden_size)
        self.linear1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(self.hidden_size, self.vocab_size)
        self.to(self.device)

    def forward(self, x, state=None):
        # print(x.shape)
        embeddings = self.elayer(x)
        # print(embeddings.shape)
        embeddings = torch.mean(embeddings, dim=1)
        # print(embeddings.shape)
        # exit(22)
        # linear relu linear
        out = self.linear1(embeddings)
        out = self.relu(out)
        out = self.linear2(out)
        # convert it to prob
        out = self.softmax(out)
        return out


def getLossDataset(data: Data, model):
    model.eval()

    dataL = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)

    criterionL = nn.CrossEntropyLoss()  # ignore_index=data.tagPadIdx)
    loss = 0

    for i, (x, y) in enumerate(dataL):
        x = x.to(model.device)
        y = y.to(model.device)

        output = model(x)

        y = y.view(-1)
        output = output.view(-1, output.shape[-1])

        loss += criterionL(output, y)

    return loss / len(dataL)


def train(model, data, optimizer, criterion):
    epoch_loss = 0
    model.train()

    dataL = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)
    lossDec = True
    prevLoss = 10000000

    epoch = 0
    model.train_data = data
    while lossDec:
        torch.save(model.state_dict(), f"model.pt")
        epoch_loss = 0
        for i, (x, y) in enumerate(dataL):
            optimizer.zero_grad()
            x = x.to(model.device)

            y = y.to(model.device)

            output = model(x)

            y = y.view(-1)
            output = output.view(-1, output.shape[-1])

            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            if i % 100 == 0:
                print(f"Epoch {epoch + 1} Batch {i} loss: {loss.item()}")

        if epoch_loss / len(dataL) > prevLoss:
            lossDec = False
        prevLoss = epoch_loss / len(dataL)

        print(f"Epoch {epoch + 1} loss: {epoch_loss / len(dataL)}")
        epoch += 1


def accuracy(model, data):
    model.eval()
    correct = 0
    total = 0
    for i, (x, y) in enumerate(DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)):
        x = x.to(model.device)
        y = y.to(model.device)

        output = model(x)

        y = y.view(-1)
        output = output.view(-1, output.shape[-1])

        _, predicted = torch.max(output, 1)
        # print x as words
        # print(x)
        # print([model.train_data.idx2w[i] for i in x.view(-1).tolist()])
        # print(x.view(-1).size(0))
        # print(y.size(0))
        # exit(22)
        # only those are suitable in which x is not bos eos or pad
        suitableIdx = [i for i in range(x.view(-1).size(0)) if x.view(-1)[i] != model.train_data.padIdx]
        # test = x.view(-1)[suitableIdx].tolist()
        # print([model.train_data.idx2w[i] for i in test])
        # exit(0)
        y_masked = y[suitableIdx]
        total += y_masked.size(0)
        correct += (predicted[suitableIdx] == y_masked).sum().item()
    return correct / total


def runSkMetric(model, data):
    model.eval()
    y_true = []
    y_pred = []

    for i, (x, y) in enumerate(DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)):
        x = x.to(model.device)
        y = y.to(model.device)

        output = model(x)

        y = y.view(-1)
        output = output.view(-1, output.shape[-1])
        suitableIdx = [i for i in range(x.view(-1).size(0)) if x.view(-1)[i] != model.train_data.padIdx]
        _, predicted = torch.max(output, 1)
        y_masked = y[suitableIdx]
        y_pred_masked = predicted[suitableIdx]
        y_true.extend(y_masked.tolist())
        y_pred.extend(y_pred_masked.tolist())

    y_trueTag = [model.train_data.tagIdx2w[i] for i in y_true]
    y_predTag = [model.train_data.tagIdx2w[i] for i in y_pred]
    return cr(y_trueTag, y_predTag)


jsonInput = open('reviews_Movies_and_TV_processed.json', 'r', encoding="utf8")
curCount = 0
totalSentences = []
for line in jsonInput:
    curCount += 1
    if curCount > sentenceLimit:
        break
    totalSentences.extend(json.loads(line)["reviewText"])
jsonInput.close()
trainData = Data(totalSentences)
print("Loaded")

model = NN(300, trainData.vocab_size)
if os.path.exists("model.pt"):
    model.load_state_dict(torch.load("model.pt"))
    print("Loaded model")
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
train(model, trainData, optimizer, criterion)
