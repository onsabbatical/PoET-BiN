import torchvision
import torchvision.transforms as transforms
import torch
from train_test import *
from model import *
import os
my_batch = 100
transform = transforms.Compose([transforms.ToTensor()])
testset = torchvision.datasets.MNIST(root='./data', train=False,download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=my_batch,shuffle=False, num_workers=2)

PATH = os.path.dirname(os.path.abspath(__file__))+'/state1.pt'
net = Net()
net.cuda()

checkpoint = torch.load(PATH)
net.load_state_dict(checkpoint)
test_acc(net,1,testloader)