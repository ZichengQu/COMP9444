# spiral.py
# COMP9444, CSE, UNSW

import torch
from torch import typename
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class PolarNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(PolarNet, self).__init__()
        # INSERT CODE HERE
        self.hidden = torch.nn.Linear(2, num_hid)
        self.out = torch.nn.Linear(num_hid, 1)
    def forward(self, input):
        temp = input.clone()
        input[:,0] = (temp[:,0].pow(2) + temp[:,1].pow(2)).sqrt() # r = sqrt(x^2 + y^2)
        input[:,1] = torch.atan2(temp[:,1],temp[:,0]) # a = atan2(y,x)
        self.hid1 = F.tanh(self.hidden(input))
        output = F.sigmoid(self.out(self.hid1))
        # output = 0*input[:,0] # CHANGE CODE HERE
        return output

class RawNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(RawNet, self).__init__()
        # INSERT CODE HERE
        self.hidden1 = torch.nn.Linear(2, num_hid)
        self.hidden2 = torch.nn.Linear(num_hid, num_hid)
        self.out = torch.nn.Linear(num_hid, 1)
    def forward(self, input):
        self.hid1 = F.tanh(self.hidden1(input))
        self.hid2 = F.tanh(self.hidden2(self.hid1))
        output = F.sigmoid(self.out(self.hid2))
        # output = 0*input[:,0] # CHANGE CODE HERE
        return output

def graph_hidden(net, layer, node):
    xrange = torch.arange(start=-7,end=7.1,step=0.01,dtype=torch.float32)
    yrange = torch.arange(start=-6.6,end=6.7,step=0.01,dtype=torch.float32)
    xcoord = xrange.repeat(yrange.size()[0])
    ycoord = torch.repeat_interleave(yrange, xrange.size()[0], dim=0)
    grid = torch.cat((xcoord.unsqueeze(1),ycoord.unsqueeze(1)),1)

    with torch.no_grad(): # suppress updating of gradients
        net.eval()        # toggle batch norm, dropout
        net(grid)
        net.train() # toggle batch norm, dropout back again

        if layer == 1:
            pred = (net.hid1[:,node] >= 0).float() # Column node

        if layer == 2 and typename(net) != 'spiral.PolarNet':
            pred = (net.hid2[:,node] >= 0).float()

        # plot function computed by model
        plt.clf()
        plt.pcolormesh(xrange,yrange,pred.cpu().view(yrange.size()[0],xrange.size()[0]), cmap='Wistia')
    # INSERT CODE HERE
