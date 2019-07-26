import torch.nn as nn
from quantization import *
import torch
import numpy as np

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc00 = nn.Linear(8,1)
        self.fc01 = nn.Linear(8,1)
        self.fc02 = nn.Linear(8,1)
        self.fc03 = nn.Linear(8,1)
        self.fc04 = nn.Linear(8,1)
        self.fc05 = nn.Linear(8,1)
        self.fc06 = nn.Linear(8,1)
        self.fc07 = nn.Linear(8,1)
        self.fc08 = nn.Linear(8,1)
        self.fc09 = nn.Linear(8,1)
        #self.batch10 = nn.BatchNorm1d(10,eps = 1e-4)


    def forward(self, x):
        y_final = torch.randn(x.size()[0],10).type(torch.cuda.FloatTensor)
        y_final[:,0] = self.fc00(x[:,0:8])[:,0]
        y_final[:,1] = self.fc01(x[:,8:16])[:,0]
        y_final[:,2] = self.fc02(x[:,16:24])[:,0]
        y_final[:,3] = self.fc03(x[:,24:32])[:,0]
        y_final[:,4] = self.fc04(x[:,32:40])[:,0]
        y_final[:,5] = self.fc05(x[:,40:48])[:,0]
        y_final[:,6] = self.fc06(x[:,48:56])[:,0]
        y_final[:,7] = self.fc07(x[:,56:64])[:,0]
        y_final[:,8] = self.fc08(x[:,64:72])[:,0]
        y_final[:,9] = self.fc09(x[:,72:80])[:,0]

        #x6 = self.batch10(y_final)


        return y_final


