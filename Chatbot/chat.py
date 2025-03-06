import random
import torch
import json
from Model import ChatbotV1
from nltk_utils import Tokenizing, BagOfWord

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open("intent.json","r") as f:
    intents = json.load(f)

model = torch.load("chatbotV3.pth")

data = torch.load('dataV2.pth')

all_words = data['all_word']
tags = data["tags"]

model.eval()

botName = "Kien's chatbot"

print("let talk: type 'quit' to exit")

while True:
    sentence = input("you: ")
    if sentence == 'quit':
        break

    sentence = Tokenizing(sentence)

    X = BagOfWord(sentence,all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    output = model(X)
    _, predict = torch.max(output,dim=1)
    tag = tags[predict.item()]

    probs = torch.softmax(output,dim=1)
    prob = probs[0][predict.item()]

    if prob.item() > 0.7:
        for intent in intents["intents"]:
            if tag == intent["tags"]:
                print(f"{botName}: {random.choice(intent['response'])}")
    else:
        print(f"{botName}: I don't understand sorry")
