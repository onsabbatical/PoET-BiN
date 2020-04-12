import numpy as np
import torch


train_output_store = np.uint8(np.zeros((1,60)))
train_output_t_store = torch.zeros([1,60],dtype=torch.uint8)

for i in range(10):
	train_output = np.uint8(np.load('train_output_' + str(i) + '.npy'))
	train_output_t = torch.from_numpy(train_output).type(torch.ByteTensor)
	train_output_t_store = torch.cat((train_output_t_store,train_output_t),0)

train_output_store = train_output_t_store.numpy()
np.save('dt_train_out.npy',train_output_store[1:,:])

train_input_store = np.uint8(np.zeros((1,512)))
train_input_t_store = torch.zeros([1,512],dtype=torch.uint8)

for i in range(10):
	train_input = np.uint8(np.load('train_input_' + str(i) + '.npy'))
	train_input_t = torch.from_numpy(train_input).type(torch.ByteTensor)
	train_input_t_store = torch.cat((train_input_t_store,train_input_t),0)

train_input_store = train_input_t_store.numpy()
np.save('dt_train_inp.npy',train_input_store[1:,:])


train_labels_store = np.uint8(np.zeros(1))
train_labels_t_store = torch.zeros([1],dtype=torch.uint8)

for i in range(10):
	train_labels = np.uint8(np.load('train_labels_' + str(i) + '.npy'))
	train_labels_t = torch.from_numpy(train_labels).type(torch.ByteTensor)
	train_labels_t_store = torch.cat((train_labels_t_store,train_labels_t),0)

train_labels_store = train_labels_t_store.numpy()
np.save('train_labels.npy',train_labels_store[1:])




test_output_store = np.uint8(np.zeros((1,60)))
test_output_t_store = torch.zeros([1,60],dtype=torch.uint8)

for i in range(3):
	test_output = np.uint8(np.load('test_output_' + str(i) + '.npy'))
	test_output_t = torch.from_numpy(test_output).type(torch.ByteTensor)
	test_output_t_store = torch.cat((test_output_t_store,test_output_t),0)

test_output_store = test_output_t_store.numpy()
np.save('dt_test_out.npy',test_output_store[1:,:])

test_input_store = np.uint8(np.zeros((1,512)))
test_input_t_store = torch.zeros([1,512],dtype=torch.uint8)

for i in range(3):
	test_input = np.uint8(np.load('test_input_' + str(i) + '.npy'))
	test_input_t = torch.from_numpy(test_input).type(torch.ByteTensor)
	test_input_t_store = torch.cat((test_input_t_store,test_input_t),0)

test_input_store = test_input_t_store.numpy()
np.save('dt_test_inp.npy',test_input_store[1:,:])


test_labels_store = np.uint8(np.zeros(1))
test_labels_t_store = torch.zeros([1],dtype=torch.uint8)

for i in range(10):
	test_labels = np.uint8(np.load('test_labels_' + str(i) + '.npy'))
	test_labels_t = torch.from_numpy(test_labels).type(torch.ByteTensor)
	test_labels_t_store = torch.cat((test_labels_t_store,test_labels_t),0)

test_labels_store = test_labels_t_store.numpy()
np.save('test_labels.npy',test_labels_store[1:])


test_input = np.load('test_input_0.npy')
test_output = np.load('test_output_0.npy')
test_label = np.load('test_labels_0.npy')
np.save('dt_test_inp.npy',test_input)
np.save('dt_test_out.npy',test_output)
np.save('test_labels.npy',test_label)

'''
train_input_0 = np.load('train_input_0.npy')
train_input_1 = np.load('train_input_1.npy')
train_input_2 = np.load('train_input_2.npy')
train_input_3 = np.load('train_input_3.npy')
train_input_4 = np.load('train_input_4.npy')

train_input_i1 = np.append(train_input_0,train_input_1,axis=0)
train_input_i2 = np.append(train_input_2,train_input_3,axis=0)
train_input_f1 = np.append(train_input_i1,train_input_i2,axis=0)
train_input = np.append(train_input_f1,train_input_4,axis=0)

train_output_0 = np.load('train_output_0.npy')
train_output_1 = np.load('train_output_1.npy')
train_output_2 = np.load('train_output_2.npy')
train_output_3 = np.load('train_output_3.npy')
train_output_4 = np.load('train_output_4.npy')

train_output_i1 = np.append(train_output_0,train_output_1,axis=0)
train_output_i2 = np.append(train_output_2,train_output_3,axis=0)
train_output_f1 = np.append(train_output_i1,train_output_i2,axis=0)
train_output = np.append(train_output_f1,train_output_4,axis=0)

test_input = np.load('test_input_0.npy')
test_output = np.load('test_output_0.npy')

np.save('dt_train_inp.npy',train_input)
np.save('dt_train_out.npy',train_output)
np.save('dt_test_inp.npy',test_input)
np.save('dt_test_out.npy',test_output)


train_label_0 = np.load('train_labels_0.npy')
train_label_1 = np.load('train_labels_1.npy')
train_label_2 = np.load('train_labels_2.npy')
train_label_3 = np.load('train_labels_3.npy')
train_label_4 = np.load('train_labels_4.npy')


train_label_i1 = np.append(train_label_0,train_label_1,axis=0)
train_label_i2 = np.append(train_label_2,train_label_3,axis=0)
train_label_f1 = np.append(train_label_i1,train_label_i2,axis=0)
train_label = np.append(train_label_f1,train_label_4,axis=0)

test_label = np.load('test_labels_0.npy')

np.save('train_labels.npy',train_label)
np.save('test_labels.npy',test_label)
'''
