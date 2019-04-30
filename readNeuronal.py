import torch
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super(Net,self).__init__()
        self.convolucionalOne = nn.Conv2d(3,10,5)
        #956X716
        self.pool = nn.MaxPool2d(2,2)
        #478X358
        self.convolucionalTwo = nn.Conv2d(10,20,5)
        #474X354
        #237X177
        self.funcionLinealOne = nn.Linear(20*177*237,800)
        self.funcionLinealTwo = nn.Linear(800,460)

    def forward(self,x):
        x = self.convolucionalOne(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.convolucionalTwo(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(-1,20*177*237)
        x = self.funcionLinealOne(x)
        x = F.relu(x)
        x = self.funcionLinealTwo(x)

        return F.log_softmax(x,dim=1)
    

        

