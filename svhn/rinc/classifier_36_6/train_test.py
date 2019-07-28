import torch
import time
import os
from torch.autograd import Variable
from model import *
from storage import *
from opt_loss import *
from quantization import *

import torch.optim as optim


def train(num_epochs,LR_start,LR_fin,LR_decay,H,N,th,trainloader,testloader,my_batch,torch_tr_l,torch_te_l):
	net = Net()
	net.cuda()
	net.apply(init_weights)
	#model_dict = net.state_dict()
	#checkpoint = torch.load('ckpt.t7')
	#pretrained_dict = checkpoint['net']
	#pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
	#model_dict.update(pretrained_dict) 
	#net.load_state_dict(pretrained_dict)

	optimizer = Adam(net.parameters(),lr = LR_start)
	best_epoch = 1
	best_acc = 0.0


	t1 = time.clock()
	for epoch in range(num_epochs):
		running_loss = 0.0
		if epoch == 0:
			current_lr = LR_start;
		else:
			current_lr = current_lr * LR_decay
		print(current_lr)
		net.train()

		for i in range(trainloader.size()[0]):
			inputs,labels = trainloader[i,:,:].reshape(100,60), torch_tr_l[i,:].reshape(100)
			inputs, labels = Variable(inputs.cuda()), labels.cuda()
			net.train()
			net.zero_grad()
			outputs = net(inputs.cuda())
			loss= hinge_loss(outputs,labels,my_batch)
			loss.backward()
			delta = optimizer.false_step(current_lr)
			clipping_scaling(delta,net)
			running_loss += loss.data
		t2 = time.clock()
		print('Epoch : %d Time : %.3f sec and loss: %.3f' % (epoch + 1, (t2 - t1), running_loss / i))
		train_acc(net,epoch,trainloader,torch_tr_l,0)

		accuracy_load = test_acc(net,epoch,testloader,torch_te_l,0)
		if accuracy_load > best_acc:
			best_acc = accuracy_load
			best_epoch = epoch + 1
			torch.save(net.state_dict(),os.path.dirname(os.path.abspath(__file__))+'/state_n.pt')
			store_all_weights(net.state_dict())
			print('saving : ', accuracy_load)
			train_acc(net,epoch,trainloader,torch_tr_l,1)
			test_acc(net,epoch,testloader,torch_te_l,1)

	print('Finished_Training')

def train_acc(net,epoch,trainloader,torch_tr_l,save_flag):
	correct = 0.0
	total = 0.0
	main_array1 = torch.IntTensor(1,10)
	inp_array = torch.ByteTensor(1,60)
	net.eval()
	#main_array1 = torch.ByteTensor(1,960)
	#main_array2 = torch.ByteTensor(1,60)
	out_label = torch.ByteTensor(1)
	for ct in range(trainloader.size()[0]):
		images, labels = trainloader[ct,:,:].reshape(100,60), torch_tr_l[ct,:].reshape(100)
		#image = 2. * images - 1
		fl_st = 0
		if labels.size()[0] != 100:
			fl_st = 1
		elif (ct+1) % 100 == 0:
			fl_st = 1
		outputs = net(Variable(images.cuda()))
		#print(torch.max(Quantize(outputs.data)),'max')
		#print(torch.min(Quantize(outputs.data)),'min')
		_, predicted = torch.max(Quantize(outputs.data), 1)
		total += labels.size(0)
		correct += (predicted == labels.cuda()).sum()
		if save_flag == 1:
			main_array1 = store_value_2d_char(main_array1,Quantize(outputs.data),fl_st,'train_output_',ct)
			inp_array = store_value_2d(inp_array,images,fl_st,'train_input_',ct)



	print('Accuracy on 60000 train images: %.3f correct: %d total examples: %d' % (100.0 * correct / total ,correct, total))



def test_acc(net,epoch,testloader,torch_te_l,save_flag):
	net.eval()
	correct = 0.0
	total = 0.0
	main_array1 = torch.IntTensor(1,10)
	inp_array = torch.ByteTensor(1,60)
	net.eval()
	#main_array1 = torch.ByteTensor(1,960)
	#main_array2 = torch.ByteTensor(1,60)
	#out_label = torch.ByteTensor(1)	
	for ct in range(testloader.size()[0]):
		images, labels = testloader[ct,:,:].reshape(100,60), torch_te_l[ct,:].reshape(100)
		#image = 2* images - 1
		fl_st = 0
		if labels.size()[0] != 100:
			fl_st = 1
		elif (ct+1) % 100 == 0:
			fl_st = 1
		outputs = net(Variable(images.cuda()))
		_, predicted = torch.max(Quantize(outputs.data),1)
		total += labels.size(0)
		correct += (predicted == labels.cuda()).sum()
		if save_flag == 1:
			main_array1 = store_value_2d_char(main_array1,Quantize(outputs.data),fl_st,'test_output_',ct)
			inp_array = store_value_2d(inp_array,images,fl_st,'test_input_',ct)

	print('Accuracy on 10000 test images: %.3f correct: %d total examples: %d' % (100.0 * correct / total ,correct, total))
	return (100.0 * float(correct)/float(total))
