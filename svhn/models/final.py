'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn




class VGG(nn.Module):
    def __init__(self, vgg_name,inputs=6):
        super(VGG, self).__init__()
        self.inputs=inputs
        self.fc00 = nn.Linear(inputs,1)
        self.fc01 = nn.Linear(inputs,1)
        self.fc02 = nn.Linear(inputs,1)
        self.fc03 = nn.Linear(inputs,1)
        self.fc04 = nn.Linear(inputs,1)
        self.fc05 = nn.Linear(inputs,1)
        self.fc06 = nn.Linear(inputs,1)
        self.fc07 = nn.Linear(inputs,1)
        self.fc08 = nn.Linear(inputs,1)
        self.fc09 = nn.Linear(inputs,1)

        self.batch3 = nn.BatchNorm1d(10,eps = 1e-4)


    def forward(self, out):

        y_final = torch.randn(out.size()[0],10).type(torch.cuda.FloatTensor)
        y_final[:,0] = self.fc00(out[:,0:(self.inputs*1)])[:,0]
        y_final[:,1] = self.fc01(out[:,(self.inputs*1):(self.inputs*2)])[:,0]
        y_final[:,2] = self.fc02(out[:,(self.inputs*2):(self.inputs*3)])[:,0]
        y_final[:,3] = self.fc03(out[:,(self.inputs*3):(self.inputs*4)])[:,0]
        y_final[:,4] = self.fc04(out[:,(self.inputs*4):(self.inputs*5)])[:,0]
        y_final[:,5] = self.fc05(out[:,(self.inputs*5):(self.inputs*6)])[:,0]
        y_final[:,6] = self.fc06(out[:,(self.inputs*6):(self.inputs*7)])[:,0]
        y_final[:,7] = self.fc07(out[:,(self.inputs*7):(self.inputs*8)])[:,0]
        y_final[:,8] = self.fc08(out[:,(self.inputs*8):(self.inputs*9)])[:,0]
        y_final[:,9] = self.fc09(out[:,(self.inputs*9):(self.inputs*10)])[:,0]
        x6 = self.batch3(y_final)
        return x6



        
def test():
    net = VGG('VGG11')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# test()
