# kuzu.py
# COMP9444, CSE, UNSW

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class NetLin(nn.Module):
    # linear function followed by log_softmax
    def __init__(self):
        super(NetLin, self).__init__()
        # INSERT CODE HERE
        self.linear = torch.nn.Linear(28*28, 10) 
    def forward(self, x):
        x = x.view(-1,28*28) # [batch,1,28,28] => [batch, 784]
        x = self.linear(x)
        x = F.log_softmax(x,dim=1)
        return x # CHANGE CODE HERE

class NetFull(nn.Module):
    # two fully connected tanh layers followed by log softmax
    def __init__(self):
        super(NetFull, self).__init__()
        # INSERT CODE HERE
        self.hidden = torch.nn.Linear(28*28, 300)
        self.out = torch.nn.Linear(300,10)

    def forward(self, x):
        x = x.view(-1,28*28)
        x = F.tanh(self.hidden(x))
        x = F.log_softmax(self.out(x),dim=1)
        return x # CHANGE CODE HERE

class NetConv(nn.Module):
    # two convolutional layers and one fully connected layer,
    # all using relu, followed by log_softmax
    def __init__(self):
        super(NetConv, self).__init__()
        # INSERT CODE HERE
        self.conv1 = nn.Sequential( # input from x: [batch, 1, 28, 28]
            nn.Conv2d(1, 32, 3, 1, 1), # [batch, 1, 28, 28] => [batch, 32, 28, 28]
            # nn.Dropout(0.5),
            nn.ReLU(),
            nn.MaxPool2d(2) # [batch, 32, 28, 28] => [batch, 32, 14, 14]
        )
        self.conv2 = nn.Sequential( # input from conv1: [batch, 32, 14, 14]
            nn.Conv2d(32, 64, 3, 1, 1), # [batch, 32, 14, 14] => [batch, 64, 14, 14]
            # nn.Dropout(0.5),
            nn.ReLU(),
            nn.MaxPool2d(2), # [batch, 64, 14, 14] => [batch, 64, 7, 7]
        )
        self.hidden = nn.Linear(64*7*7, 1000)
        self.out = nn.Linear(1000,10)
    def forward(self, x):
        x = self.conv1(x) # [batch, 1, 28, 28] => [batch, 32, 14, 14]
        x = self.conv2(x) # [batch, 32, 14, 14] => [batch, 64, 7, 7]
        x = x.view(x.size(0), -1) # [batch, 64, 7, 7] => [batch_size, 64*7*7]
        x = F.relu(self.hidden(x))
        x = F.log_softmax(self.out(x),dim=1)
        return x # CHANGE CODE HERE