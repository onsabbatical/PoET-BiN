import numpy as np
import os

cur_path = os.path.abspath(__file__)
file_name = 'comp_gen.txt'
f = open(cur_path+file_name,'w')

#First input and last output###
inp_data = np.load('dt_test_inp.npy').astype(int)
out_matrix = np.load('classifier_36_6/test_output_0.npy')

#Two's complement#
def twos_comp(inp):
	temp_out = 'b'
	for i in range(8):
		if inp[-1-i] == '0':
			temp_out += '1'
		else :
			temp_out += '0'
	cin = 1
	out = 'b'
	inp = temp_out[1:]
	for i in range(8):
		temp = int(inp[i]) + cin
		if temp == 0:
			out += '0'
			cin = 0
		elif temp == 1:
			out += '1'
			cin = 0
		elif temp == 2:
			out += '0'
			cin = 1
	fin_out = out[::-1]
	return fin_out[:-1]
#Out matrix in python is in integer, needs to be converted to binary#
out_matrix_bin = np.zeros((np.shape(out_matrix)[0],10,8))
for i in range(np.shape(out_matrix)[0]):
	for j in range(10):
		temp = format(int(out_matrix[i,j]),'#010b')
		if temp[0] == '-':
			temp = format(int(out_matrix[i,j]),'#011b')
			temp = twos_comp(temp)
		else:
			temp = temp[2:]

		for k in range(8):
			out_matrix_bin[i,j,k] = int(temp[k])


out_matrix_bin = out_matrix_bin.astype(int)

#Testbench#
for i in range(100):
	f.write('inp_feat <= \"' );
	for j in range(np.shape(inp_data)[1]):
		f.write(str(inp_data[i,np.shape(inp_data)[1] - j - 1]))
	f.write('\";')
	f.write('cor_in <= \"' )
	for k in range(10):
		for j in range(8):
			f.write(str(out_matrix_bin[i,k,j]))
	f.write('\" ; wait for 10 ns; \n')
