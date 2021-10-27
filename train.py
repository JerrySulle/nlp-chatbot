import json
from nltk_utils import tokenize, stem, bag_of_words
import numpy as np
from model import NeuralNet

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

with open('intents.json','r') as f:
    intents = json.load(f)

#print(intents)

all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        #split sentences into words
        all_words.extend(w)
        #Append into all_words
        #DO NOT APPEND, BECAUSE THIS IS AN ARRAY
        xy.append((w, tag))
        #tuple pattern and tag
        #print(xy)

ignore_words = ['?', '!', ',', '.']
all_words = [stem(w) for w in all_words if w not in ignore_words]
#STEMMING

all_words = sorted(set(all_words))
tags = sorted(set(tags))
#MAKE SURE THERE ARE NO DUPLICATES
#print(all_words)
#print(tags)

X_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    #put results in training data

    label = tags.index(tag)
    #numbers for the labels
    y_train.append(label)

X_train = np.array(X_train)
print(X_train[0, :])
y_train = np.array(y_train)
print(y_train)

#BATCHING
batch_size = 8
hidden_size = 8
output_size = len(tags)
input_size = len(X_train[0])
#print(input_size, len(all_words))
#print(output_size, tags)
learning_rate = 0.0001
num_epochs = 2000

class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
train_loader = DataLoader(dataset = dataset, batch_size=batch_size, shuffle=True, num_workers=0)
#num workers for multi threading

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)

#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        # Forward pass
        outputs = model(words)
        # if y would be one-hot, we must apply
        # labels = torch.max(labels, 1)[1]
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if(epoch +1) %100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

print(f'final loss: {loss.item():.4f}')

data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. File saved to {FILE}')