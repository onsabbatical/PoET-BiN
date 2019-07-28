import torch 
import numpy as np

def store_value(main_array,cu_fl,i,name,ct):

	cu_uint8 = cu_fl.type(torch.ByteTensor)
	main_array = torch.cat((main_array,cu_uint8),0)
	#print(i)

	if i == 1:
		main_array_np = main_array.cpu().numpy()
		np.save(name + str(int(ct/100)) + '.npy',main_array[1:,:,:,:])
		main_array = torch.ByteTensor(1,np.shape(main_array)[1],np.shape(main_array)[2],np.shape(main_array)[3])
	return main_array


def store_value_3d(main_array,cu_fl,i,name,ct):

	cu_uint8 = cu_fl.type(torch.ByteTensor)
	cu_uint8 = torch.reshape(cu_uint8,(cu_fl.size()[0],cu_fl.size()[2],cu_fl.size()[3]))
	main_array = torch.cat((main_array,cu_uint8),0)
	#print(i)

	if i == 1:
		main_array_np = main_array.cpu().numpy()
		np.save(name + str(int(ct/100)) + '.npy',main_array[1:,:,:])
		main_array = torch.ByteTensor(1,np.shape(main_array)[1],np.shape(main_array)[2])
	return main_array

def store_value_2d(main_array,cu_fl,i,name,ct):

	cu_uint8 = cu_fl.type(torch.ByteTensor)
	main_array = torch.cat((main_array,cu_uint8),0)
	#print(i)

	if i == 1:
		main_array_np = main_array.cpu().numpy()
		np.save(name + str(int(ct/100)) + '.npy',main_array[1:,:])
		main_array = torch.ByteTensor(1,np.shape(main_array)[1])
		print(ct)
	return main_array

def store_value_2d_char(main_array,cu_fl,i,name,ct):

	cu_uint8 = cu_fl.type(torch.IntTensor)
	main_array = torch.cat((main_array,cu_uint8),0)
	#print(i)

	if i == 1:
		main_array_np = main_array.cpu().numpy()
		np.save(name + str(int(ct/100)) + '.npy',main_array[1:,:])
		main_array = torch.IntTensor(1,np.shape(main_array)[1])
		print(ct)
	return main_array


def store_value2(main_array,cu_fl,i,name,ct):

	cu_uint8 = cu_fl.type(torch.ByteTensor)
	main_array = torch.cat((main_array,cu_uint8),0)
	#print(i)

	if i == 1:
		main_array_np = main_array.cpu().numpy()
		np.save(name + str(int(ct/100)) + '.npy',main_array[1:])
		main_array = torch.ByteTensor(1)
	return main_array

def store_all_weights(dict_wb):
	weight_matrix = torch.Tensor(1,6).type(torch.cuda.FloatTensor)
	bias_matrix = torch.Tensor(1).type(torch.cuda.FloatTensor)

	for items in dict_wb:
		print(weight_matrix.size())
		if 'fc' in items and 'weight' in items:
			#print(dict_wb[items].size())
			weight_matrix = torch.cat((weight_matrix,dict_wb[items]),0)

		if 'fc' in items and  'bias' in items:
			bias_matrix = torch.cat((bias_matrix,dict_wb[items]),0)
	np.save('weight_matrix.npy',weight_matrix[1:,:].cpu().numpy())
	np.save('bias_matrix.npy',bias_matrix[1:].cpu().numpy())