import torch
from torch import nn

class ChatbotV1(nn.Module):
    def __init__(self, input, output, hidden):
        super().__init__()
        
        self.l1 = nn.Linear(in_features=input, out_features=hidden)
        self.l2 = nn.Linear(in_features=hidden,out_features=hidden)
        self.l3 = nn.Linear(in_features=hidden,out_features=output)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.l3(x)
        return x