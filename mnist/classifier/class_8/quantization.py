import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter
import numpy as np
import math
from numpy import random

class RoundNoGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.round()
    @staticmethod
    def backward(ctx, g):
        return g 

def hard_sigmoid(x):
    return torch.clamp((x+1.)/2.,0,1)

def hard_sigmoid2(x):
	return torch.clamp((x+1.)/2,-1,1)

def binary_tanh_unit(x):
    return 2. * RoundNoGradient.apply(hard_sigmoid(x)) - 1.

def binary_sigmoid_unit(x):
    return RoundNoGradient.apply(hard_sigmoid(x))

def binarization(W):              #NOT COMPLETE

    weight_b = hard_sigmoid(W)
    weight_b = torch.round(weight_b)
    weight_b = 2*weight_b - 1
    #weight_b[weight_b == 0] = - 1

    return weight_b

def ternarization(W):              #NOT COMPLETE

    weight_b = hard_sigmoid2(W)
    weight_b = torch.round(weight_b)
    #weight_b = 2*weight_b - 1
    #weight_b[weight_b == 0] = - 1

    return weight_b

def Quantize(tensor,  numBits=8):
    #tensor = (2*tensor - torch.max(tensor) - torch.min(tensor))/(torch.max(tensor) - torch.min(tensor))
    tensor = (2.0*tensor)/4.0
    #tensor = tensor.div(2**(numBits-1))
    tensor.clamp_(-1,1- (2**(-numBits+1)))
    tensor=tensor.mul(2**(numBits-1)).round()#.div(2**(numBits-1))
    return tensor
    
def init_weights(m):
    torch.cuda.manual_seed(random.randint(1,2147462579))

    if type(m) == (nn.Conv2d):
        nn.init.uniform(m.weight)
        
    if type(m) == (nn.Linear):
        nn.init.uniform(m.weight)

class Linear(nn.modules.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__(in_features,out_features)
        self.in_features = in_features
        self.out_features = out_features
        self.W_LR_scale = np.float32(1./ np.sqrt(1.5 / (in_features + out_features)))
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward_t(self, input):
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(self.in_features, self.out_features, self.bias is not None)

    def forward(self,input):
        Wr = self.weight.data
        self.Wb = binarization(self.weight.data)
        #self.Wb = ternarization(self.weight.data)
        self.weight.data = self.Wb
        rvalue = self.forward_t(input)
        self.weight.data = Wr
        return rvalue

    def return_W_scale(self):
        return self.W_LR_scale


class Conv2d(nn.modules.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        num_inputs = int(np.prod(kernel_size)*in_channels)
        num_units = int(np.prod(kernel_size)*out_channels)
        self.W_LR_scale = np.float32(1./np.sqrt(1.5 / (num_inputs + num_units)))
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward_t(self, input):
        return F.conv2d(input, self.weight, self.bias, self.stride,self.padding, self.dilation, self.groups)

    def forward(self,input):
        Wr = self.weight.data
        self.Wb = binarization(self.weight.data)
        #self.Wb = ternarization(self.weight.data)
        self.weight.data = self.Wb
        #print(self.weight.data)
        rvalue = self.forward_t(input)
        #print(rvalue.data[0,0,0,0] - self.bias.data[0])
        self.weight.data = Wr
        return rvalue

    def return_W_scale(self):
        return self.W_LR_scale

