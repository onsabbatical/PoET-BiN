'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
from models.quantization import *

cfg = {
    'VGG11': [16, 'M', 'L', 'M',  'B'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 'L', 'M', 'B'],
}

class bin_act(nn.Module):
    def __init__(self):
        super(bin_act, self).__init__()
        pass

    def forward(self, x):
        return binary_sigmoid_unit(x)


class VGG(nn.Module):
    def __init__(self, vgg_name,inputs=8):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        #self.classifier = nn.Linear(512, 10)
        self.inputs=inputs
        self.layer_int0 = nn.Linear(512,512)
        self.batch0 = nn.BatchNorm1d(512)
        self.layer_int2 = nn.Linear(512,80)
        self.batch2 = nn.BatchNorm1d(80)
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
        self.act2 = bin_act()
        self.act3 = nn.ReLU()
        #self.act4 = nn.ReLU()


    def forward(self, x):
        out = self.features(x)
        bin_out = out.view(out.size(0), -1)
        #print(bin_out)

        out = self.layer_int0(bin_out)
        out = self.batch0(out)
        out = self.act3(out)
        #out = self.layer_int1(out)
        #out = self.batch1(out)
        #out = self.act4(out)
        out = self.layer_int2(out)
        out = self.batch2(out)
        s_out = self.act2(out)
        #print(out)

        y_final = torch.randn(out.size()[0],10).type(torch.cuda.FloatTensor)
        y_final[:,0] = self.fc00(s_out[:,0:(self.inputs*1)])[:,0]
        y_final[:,1] = self.fc01(s_out[:,(self.inputs*1):(self.inputs*2)])[:,0]
        y_final[:,2] = self.fc02(s_out[:,(self.inputs*2):(self.inputs*3)])[:,0]
        y_final[:,3] = self.fc03(s_out[:,(self.inputs*3):(self.inputs*4)])[:,0]
        y_final[:,4] = self.fc04(s_out[:,(self.inputs*4):(self.inputs*5)])[:,0]
        y_final[:,5] = self.fc05(s_out[:,(self.inputs*5):(self.inputs*6)])[:,0]
        y_final[:,6] = self.fc06(s_out[:,(self.inputs*6):(self.inputs*7)])[:,0]
        y_final[:,7] = self.fc07(s_out[:,(self.inputs*7):(self.inputs*8)])[:,0]
        y_final[:,8] = self.fc08(s_out[:,(self.inputs*8):(self.inputs*9)])[:,0]
        y_final[:,9] = self.fc09(s_out[:,(self.inputs*9):(self.inputs*10)])[:,0]
        x6 = self.batch3(y_final)
        return x6,s_out,bin_out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 1
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif x == 'L':
                layers += [nn.Conv2d(in_channels, 32, kernel_size=5),
                           nn.BatchNorm2d(32)]
            elif x == 'B':
                layers += [bin_act()]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=5),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)




        
def test():
    net = VGG('VGG11')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# test()
