import numpy as np

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
