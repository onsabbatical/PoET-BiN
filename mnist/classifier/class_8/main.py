import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch
from train_test import *

my_batch = 100

num_epochs = 600
LR_start = 0.0001
LR_fin = 0.0000003
LR_decay = (LR_fin/LR_start)**(1./(num_epochs)) 

H = 1.
N = 1.
th = 3.

np_train = np.load('predicted_train_outputs.npy')
np_test = np.load('predicted_test_outputs.npy')

print(np.shape(np_train))
print(np.shape(np_test))

torch_train = torch.from_numpy(np_train.astype(float))
torch_test  = torch.from_numpy(np_test.astype(float))
train_labels = np.load('train_labels.npy')
test_labels = np.load('test_labels.npy')


torch_tr_l_temp = torch.from_numpy(train_labels.astype(float))
torch_te_l_temp = torch.from_numpy(test_labels.astype(float))


torch_tr_l = torch_tr_l_temp.reshape(500,100).type(torch.LongTensor)
torch_te_l = torch_te_l_temp[:10000].reshape(100,100).type(torch.LongTensor)

trainloader = (torch_train.reshape(500,100,80)).type(torch.FloatTensor)
testloader  = torch_test[:10000,:].reshape(100,100,80).type(torch.FloatTensor)


train(num_epochs,LR_start,LR_fin,LR_decay,H,N,th,trainloader,testloader,my_batch,torch_tr_l,torch_te_l)
test_acc()		
