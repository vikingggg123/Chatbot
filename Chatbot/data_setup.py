import nltk_utils
import json
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from Model import ChatbotV1
# Open json file:

with open('intents.json', "r") as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []
ignore = ['?',',','!','.']

for intent in intents['intents']:
    tag = intent['tags']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = nltk_utils.Tokenizing(pattern)
        all_words.extend(w)
        xy.append((w, tag))


all_words = [nltk_utils.Stemming(w) for w in all_words if w not in ignore]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

# create training data

X_train = []
y_train = []
for (pattern, tag) in xy:
    bag = nltk_utils.BagOfWord(pattern, all_words)
    X_train.append(bag)
    print(bag)


    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train) 
y_train = np.array(y_train)

# create Dataset class

class Chatbot(Dataset):
    def __init__(self, x, y):
        super().__init__()
        
        self.x = np.array(x)
        self.y = np.array(y)
        self.sample = len(self.x)
    
    def __getitem__(self, index):
        return torch.tensor(self.x[index], dtype=torch.float32), torch.tensor(self.y[index], dtype=torch.float32)
    
    def __len__(self):
        return self.sample

batch = 16

dataset = Chatbot(x=X_train, y=y_train)
train_loader = DataLoader(dataset=dataset, batch_size=batch, shuffle=True)
# initate the model
num_class = len(tags)
input = len(X_train[0])
print(input)
hidden = 16
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Chat_model = ChatbotV1(input=input, output=num_class, hidden=hidden).to(device)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=Chat_model.parameters(),
                             lr=0.001)

EPOCHS = 1000
for epoch in range(EPOCHS):
    Chat_model.train()
    for (word, label) in train_loader:  

        word, label = word.to(device), label.type(torch.LongTensor)
        
        y_pred = Chat_model(word)

        loss = loss_fn(y_pred,label)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        if epoch % 100 == 0:
            print(f"epochs: {epoch} | loss: {loss}")

data = {
    "all_word": all_words,
    "tags": tags
}

print("saving model")
torch.save(Chat_model, "chatbotV2.pth")
print("save successfully")
print("saving all_word and tags")
torch.save(data, "data.pth")