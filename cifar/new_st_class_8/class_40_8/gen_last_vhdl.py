import numpy as np
import os


####Add paths to save vhdl file####
cur_path = os.path.abspath(__file__)
file_name = 'comp_gen.txt'
f = open(cur_path+file_name,'w')

####Read Weights and bias Matrices of final layer#####
MAT_weights = np.load('weight_matrix.npy')  #### no_of_class * LUT inputs, (10 * 8)#####
MAT_bias = np.load('bias_matrix.npy')      #### no_of_class * 1  (10 * 1) ####

inp_matrix = np.load('test_input_0.npy').astype(int)   ##### no_of_class * LUT inputs ####
out_matrix = np.load('test_output_0.npy')  #### no of class * num_bits ####

def Quantize(tensor,  numBits=8):
    tensor = (2.0*tensor)/4.0   ###Scaling to get it all the ranges closer, prevents many values in the saturation region####
    tensor = np.clip(tensor,-1,1 - (2**(-numBits+1)))   ####[-1,1)###
    tensor= np.round(tensor * (2**(numBits-1)))    ####[-128,127) integers###
    return tensor

#### Generating 0 to 2**8, all possible inputs ####
cur_inputs = np.zeros((2**np.shape(MAT_weights)[1],np.shape(MAT_weights)[1]))
for i in range(2**np.shape(MAT_weights)[1]):
	seq_string = format(i,'#010b') #0bXXXX XXXX, MSB to LSB#
	for j in range(np.shape(MAT_weights)[1]):
		cur_inputs[i,j] = int(seq_string[(-1)-j]) #Reversing the LSB is in position 0#

### Multiplying 256 * 8 with 8 * 1 + 1*1 generating a vector of 256 ###
MAT_labels = np.zeros((10,2**np.shape(MAT_weights)[1]))
for i in range(10):
	t_b_q = MAT_bias[i] + np.matmul(cur_inputs,MAT_weights[i,:])  
	MAT_labels[i,:] = Quantize(t_b_q)    ### Quantize ### 



print(MAT_labels[1,4])
print(MAT_labels[1,59])
print(MAT_labels[1,63])

print(np.min(MAT_labels))


####Random Prints###
print(format(int(MAT_labels[1,4]),'#012b'))
print(format(int(MAT_labels[1,59]),'#012b'))
print(format(int(MAT_labels[1,63]),'#012b'))

###Two's complement of a number###
def twos_comp(inp):
	temp_out = 'b'    #Kick starter#
	for i in range(8):  ##Invert the bits, here indiates the numbits###
		if inp[-1-i] == '0':   ###Processing from LSB to MSB###
			temp_out += '1'
		else :
			temp_out += '0'  
	cin = 1
	out = 'b'
	inp = temp_out[1:] ###Removed kick starter###
	for i in range(8): ###Adding and carry forward###
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
	fin_out = out[::-1]  #### MSB -> LSB ####
	return fin_out[:-1]

### Printing Intermediated stuff for debugging ###
def twos_complement_debug(inp):
	temp_out = 'b' 
	print('inp : ',inp)
	for i in range(8):  
		if inp[-1-i] == '0':
			temp_out += '1'
		else :
			temp_out += '0'
	cin = 1
	out = 'b'
	inp = temp_out[1:]
	print('switched : ',inp)
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
		print('out : ', out)
	fin_out = out[::-1]
	return fin_out[:-1]

### The integers are converted to their binary form 10,256,8###
MAT_labels_bin = np.zeros((10,2**np.shape(MAT_weights)[1],8))
for i in range(10):
	for j in range(2**np.shape(MAT_weights)[1]):
		temp = format(int(MAT_labels[i,j]),'#010b') #normal covert#
		if temp[0] == '-': #negative cases#
			temp = format(int(MAT_labels[i,j]),'#011b')
			temp = twos_comp(temp)
		else:
			temp = temp[2:]

		for k in range(8):
			MAT_labels_bin[i,j,7 - k] = int(temp[k])

#contains zeros and ones#
MAT_labels_bin = MAT_labels_bin.astype(int)

print(MAT_labels_bin[1,4,:])
result = twos_complement_debug(format(int(MAT_labels[1,59]),'#011b'))

print(MAT_labels_bin[1,59,:])
print(MAT_labels_bin[1,63,:])

###Printing the signals###
for i in range(10):
	for j in range(8):
		f.write('signal C_' + str(i) + '_B_' + str(7-j) + '_out : std_logic := \'0\'; \n ')

f.write('\n');

###Just before merging###
for i in range(80):
	f.write('C_' + str(i) + '_out <= inp_feat(' + str(i) + '); \n')


f.write('\n');
###Signal required VHDL###
for i in range(80):
	f.write('signal C_' + str(i) +'_out : std_logic := \'0\'; \n ')

f.write('\n');

###Write labels and inputs###
for i in range(10):
	for j in range(8):
		f.write('C_'+str(i)+'_B_'+str(j)+'_inst : LUT8 generic map(INIT => "')  ###LUT 8###
		for k in range(2**np.shape(MAT_weights)[1]):
			f.write(str(MAT_labels_bin[i,np.shape(MAT_labels_bin)[1] - k - 1,j]))###In reverse order###
		f.write('") port map( O =>' + 'C_'+str(i)+'_B_'+str(j)+'_out')
		for k in range(np.shape(MAT_weights)[1]):
			f.write(', I' + str(k) + ' =>  C_'+str(i*8 + k)+'_out') ###Inputs Linearly###
		f.write('); \n')
	f.write('\n');

f.write('\n');

###At end to append one after another to collate all LUT output###
f.write('out_fin <= ')
for i in range(10):
	for j in range(8):
		f.write('C_' + str(i) + '_B_' + str(7-j) + '_out ')
		if i == 9 and j == 7:
			f.write('; \n')
		else:
			f.write (' & ')

f.write('\n')

###The out matrix has integers from python -> needs to be converted###
print(np.min(out_matrix))
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

#### Test bench###
for k in range(100):
	f.write('inp_feat <= \"' );
	for i in range(80):
		f.write(str(inp_matrix[k,79-i]))
	f.write('\";')
	f.write('cor_in <= \"' )
	for i in range(10):
		for j in range(8):
			f.write(str(out_matrix_bin[k,i,j]))
	f.write('\" ; wait for 10 ns; \n')





f.close();

