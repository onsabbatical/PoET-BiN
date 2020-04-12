'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
os.environ['CUDA_VISIBLE_DEVICES']='0'
from models import *
from utils import progress_bar
from itertools import chain

from storage import *

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.0005, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    #transforms.RandomCrop(32, padding=4),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.SVHN(root='./data', split = 'train', download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,shuffle=True, num_workers=2)

extraset = torchvision.datasets.SVHN(root='./data', split = 'extra', download=True, transform=transform_train)
extraloader = torch.utils.data.DataLoader(extraset, batch_size=100,shuffle=True, num_workers=2)

testset = torchvision.datasets.SVHN(root='./data', split = 'test', download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100,shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
net = VGG('VGG11')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
#net = ShuffleNetV2(1)
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()

print(len(extraloader))

# Training
def train(epoch,cur_lr):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    optimizer = optim.SGD(net.parameters(), lr=cur_lr, momentum=0.9, weight_decay=5e-4)

    all_train_loader = chain(trainloader,extraloader)

    for batch_idx, (inputs, targets) in enumerate(all_train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs,sec_out,sec_in = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, (len(trainloader) + len(extraloader)), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs,sec_out,sec_in = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc
        if acc > 50:
            all_train_loader = chain(trainloader,extraloader)
            storage(all_train_loader,'train')
            storage(testloader,'test')


def storage(loader_v,nme):
    net.eval()
    main_array1 = torch.ByteTensor(1,512)
    main_array2 = torch.ByteTensor(1,60)
    out_label = torch.ByteTensor(1)
    with torch.no_grad():
        for ct,(inputs,targets) in enumerate(loader_v):
            fl_st = 0
            if targets.size()[0] != 100:
                fl_st = 1
            elif (ct+1) % 100 == 0:
                fl_st = 1
            inputs, targets = inputs.to(device), targets.to(device)
            outputs,sec_out,sec_in = net(inputs)
            main_array1 = store_value_2d(main_array1,sec_in.data,fl_st,nme + '_input_',ct)
            main_array2 = store_value_2d(main_array2,sec_out.data,fl_st,nme + '_output_',ct)
            out_label = store_value2(out_label,targets,fl_st, nme + '_labels_',ct)

LR_start = args.lr
LR_fin = 0.0000003
LR_decay = (LR_fin/LR_start)**(1./(200)) 

for epoch in range(start_epoch, start_epoch+200):
    if epoch == 0:
        cur_lr = args.lr
    else:
        cur_lr = cur_lr * LR_decay
    print('epoch : ', epoch, 'cur_lr : ', cur_lr)
    train(epoch,cur_lr)
    test(epoch)
