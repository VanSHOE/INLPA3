import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pprint import pprint
import random
import time
from sklearn.metrics import classification_report as cr
from torch.autograd import Variable
import os
import json
import pickle

random.seed(time.time())
logFile = open(f"logs/loss.txt", "w", encoding="utf8")
sentenceLens = {}
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
logFile.write(device + "\n")
logFile.flush()
CONTEXT = 3  # Left and right context hence total context is 2*CONTEXT
sentenceLimit = 10000
BATCH_SIZE = 250
MODEL = "Test"
NEGSAMPLES = 1


# CBOW
class Data(torch.utils.data.Dataset):
    def __init__(self, sentences):
        self.device = device
        self.context = CONTEXT
        self.sentences = [sentence for sentence in sentences if len(
            sentence) > 2 * self.context]
        print(f"Sentences: {len(self.sentences)}")
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
                self.inputs.append(
                    sentence[i - self.context:i] + sentence[i + 1:i + self.context + 1])
                self.outputs.append(sentence[i])
        # conver to idxs
        self.inputs = [[self.w2idx[word] for word in sentence]
                       for sentence in self.inputs]
        self.outputs = [self.w2idx[word] for word in self.outputs]

        # neg sample input outputs
        self.negInputs = []
        self.negOutputs = []
        # context, output_word -> 0 or 1 for all input output (one hot)
        for i in range(len(self.inputs)):
            for word in self.inputs[i]:
                self.negInputs.append((word, self.outputs[i]))
                self.negOutputs.append(1)
            for j in range(NEGSAMPLES):
                negWord = random.randint(0, self.vocab_size - 1)
                while negWord in self.inputs[i]:
                    negWord = random.randint(0, self.vocab_size - 1)
                self.negInputs.append((negWord, self.outputs[i]))
                self.negOutputs.append(0)

        # convert to tensor
        self.negInputs = torch.tensor(self.negInputs).to(self.device)
        self.negOutputs = torch.tensor(self.negOutputs).to(self.device)
        print(f"Inputs: {len(self.negInputs)}")

    def __len__(self):
        return len(self.negInputs)

    def __getitem__(self, idx):
        oneHot = torch.zeros(self.vocab_size, device=self.device)
        oneHot[self.negInputs[idx][0]] = 0.5
        oneHot[self.negInputs[idx][1]] = 0.5
        return oneHot, self.negOutputs[idx]


class NN(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        # call the init function of the parent class
        super(NN, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.elayer = nn.Linear(self.vocab_size, self.hidden_size)
        # self.linear1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(self.hidden_size, 2)
        self.to(self.device)

    def forward(self, x, state=None):
        # print(x.shape)

        embeddings = self.elayer(x)
        # print(embeddings.shape)
        # embeddings = torch.mean(embeddings, dim=1)
        # print(embeddings.shape)
        # exit(22)
        # linear relu linear
        # out = self.linear1(embeddings)
        out = self.relu(embeddings)
        out = self.linear2(out)
        return out


def train(model, data, optimizer, criterion):
    epoch_loss = 0
    model.train()

    dataL = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)
    lossDec = True
    prevLoss = 10000000

    epoch = 0
    model.train_data = data
    pat = 5
    while lossDec:
        torch.save(model.state_dict(), f"modelFixed500_4.pt")
        epoch_loss = 0
        for i, (x, y) in enumerate(dataL):
            optimizer.zero_grad()
            x = x.to(model.device)

            y = y.to(model.device)

            output = model(x)

            y = y.view(-1)
            # output=output.view(-1)
            output = output.view(-1, output.shape[-1])
            # print(output)
            # output = [1.] * len(output)
            # output = torch.tensor(output, requires_grad=True).to(model.device)
            loss = criterion(output, y)
            # loss=torch.abs(output - y)
            # loss=Variable(torch.mean(loss, 0), requires_grad=True)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            if i % 100 == 0:
                print(f"Epoch {epoch + 1} Batch {i} loss: {loss.item()}")

        if epoch_loss / len(dataL) > prevLoss:
            if pat == 0:
                lossDec = False
            else:
                pat -= 1
        else:
            pat = 5

        print(
            f"Epoch {epoch + 1} loss: {epoch_loss / len(dataL)} | Change: {(epoch_loss / len(dataL)) - prevLoss}")
        logFile.write(
            f"Epoch {epoch + 1} loss: {epoch_loss / len(dataL)} | Change: {(epoch_loss / len(dataL)) - prevLoss}" + "\n")
        logFile.flush()
        prevLoss = epoch_loss / len(dataL)
        epoch += 1


if not os.path.exists("trainData4.pkl"):
    print("Dataset not found")
    jsonInput = open('reviews_Movies_and_TV_processed.json',
                     'r', encoding="utf8")
    curCount = 0
    totalSentences = []
    for line in jsonInput:
        curCount += 1
        if curCount > sentenceLimit:
            break
        totalSentences.extend(json.loads(line)["reviewText"])
    jsonInput.close()
    trainData = Data(totalSentences)
    pickle.dump(trainData, open("trainData4.pkl", "wb"))
else:
    print("Dataset found")
    trainData = pickle.load(open("trainData4.pkl", "rb"))

print("Loaded")

model = NN(500, trainData.vocab_size)
if os.path.exists("modelFixed500_4.pt"):
    model.load_state_dict(torch.load("modelFixed500_4.pt"))
    print("Loaded model")
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
criterion = nn.CrossEntropyLoss()
train(model, trainData, optimizer, criterion)
