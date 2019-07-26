import numpy as np
import os

###Paths to save file###
cur_path = os.path.abspath(__file__)
file_name = 'comp_gen.txt'
f = open(cur_path+file_name,'w')

### Read files generated from my_multi###
RINC_0_labels = np.load('label_store.npy').astype(int)  ###the class one or zero###
RINC_0_feats = np.load('feats_store.npy').astype(int) ###the features that are selected###
MAT_weights = np.load('weights_store.npy') ###Adaboost weights###
inp_data = np.load('dt_train_inp.npy').astype(int) ###inp data###
out_data = np.load('predicted_train_outputs.npy').astype(int) ###out data from python###

### Signal RINC L0###
for i in range(np.shape(RINC_0_feats)[0]):
	f.write('signal C_'+str(i)+'_out : std_logic := \'0\'; \n')

f.write('\n')

### Signal RINC L1###
for i in range(np.shape(RINC_0_feats)[0]):
	for j in range(np.shape(RINC_0_feats)[1]):
		for k in range(np.shape(RINC_0_feats)[2]):
			f.write('signal C_'+str(i)+'_S_'+str(j)+'_L_'+str(k)+'_out : std_logic := \'0\'; \n')
f.write('\n')

### Signal RINC L2###
for i in range(np.shape(RINC_0_feats)[0]):
	for j in range(np.shape(RINC_0_feats)[1]):
		f.write('signal C_'+str(i)+'_S_'+str(j)+'_out : std_logic := \'0\'; \n')

f.write('\n')

### First level of RINC ###
for i in range(np.shape(RINC_0_feats)[0]):
	for j in range(np.shape(RINC_0_feats)[1]):
		for k in range(np.shape(RINC_0_feats)[2]):
			f.write('C_'+str(i)+'_S_'+str(j)+'_L_'+str(k)+'_inst : LUT8 generic map(INIT => "')
			for l in range(np.shape(RINC_0_labels)[3]):
				f.write(str(RINC_0_labels[i,j,k,np.shape(RINC_0_labels)[3] - l - 1]))
			f.write('") port map( O =>' + 'C_'+str(i)+'_S_'+str(j)+'_L_'+str(k)+'_out')
			for l in range(np.shape(RINC_0_feats)[3]):
				f.write(', I' + str(l) + ' =>  inp_feat(' + str(RINC_0_feats[i,j,k,np.shape(RINC_0_feats)[3]-l-1]) + ')')
			f.write('); \n')

f.write('\n');
### For first level adaboost, all possible inputs###
cur_inputs = np.zeros((2**np.shape(RINC_0_feats)[3],np.shape(RINC_0_feats)[3]))
for i in range(2**np.shape(RINC_0_feats)[3]):
	seq_string = format(i,'#010b')
	for j in range(np.shape(RINC_0_feats)[3]):
		cur_inputs[i,j] = int(seq_string[(-1)-j])
### For multiply and threshold###
MAT_labels = np.zeros((np.shape(MAT_weights)[0],np.shape(MAT_weights)[1],np.shape(RINC_0_labels)[3]))
for i in range(np.shape(MAT_weights)[0]):
	for j in range(np.shape(MAT_weights)[1]):
		sum_cur = np.sum(MAT_weights[i,j,:])
		th = sum_cur * 0.5
		MAT_labels[i,j,:] = np.matmul(cur_inputs,MAT_weights[i,j,:].reshape(np.shape(MAT_weights)[2],1))[:,0] > th
		
MAT_labels = MAT_labels.astype(int)
###The above generates the labels for all possible input combinations###
for i in range(np.shape(MAT_labels)[0]):
	for j in range(np.shape(MAT_labels)[1] - 1): ###Not the last level###
		f.write('C_'+str(i)+'_S_'+str(j)+'_inst : LUT8 generic map(INIT => "')
		for l in range(np.shape(MAT_labels)[2]):
			f.write(str(MAT_labels[i,j,np.shape(MAT_labels)[2] - l - 1]))
		f.write('") port map( O =>' + 'C_'+str(i)+'_S_'+str(j)+'_out')
		for l in range(np.shape(RINC_0_feats)[3]):
			f.write(', I' + str(l) + ' =>  C_'+str(i)+'_S_'+str(j)+'_L_'+str(l)+'_out')
		f.write('); \n')

	f.write('\n');
	#### Last level, considering the possibly it might not be full###

	f.write('C_'+str(i)+'_inst : LUT8 generic map(INIT => "')
	for l in range(np.shape(MAT_labels)[2]):
		f.write(str(MAT_labels[i,-1,np.shape(MAT_labels)[2] - l - 1]))
	f.write('") port map( O =>' + 'C_'+str(i)+'_out')
	for l in range(np.shape(RINC_0_feats)[3]):
		if l < (np.shape(MAT_labels)[1] - 1):
			f.write(', I' + str(l) + ' =>  C_'+str(i)+'_S_'+str(l)+'_out')
		else:
			f.write(', I' + str(l) + ' => \' 0 \' ')  ### Need this when we don not use all###

	f.write('); \n')
	f.write('\n \n');
### Append all LUT outputs at the last level###
f.write('pred_out <= ')
for i in range(np.shape(RINC_0_feats)[0] - 1):
	f.write('C_'+str(i)+'_out &')
f.write('C_' + str(np.shape(RINC_0_feats)[0] - 1) + '_out ; \n')
#### Test bench using the input and output###
for i in range(100):
	f.write('inp_feat <= \"' );
	for j in range(np.shape(inp_data)[1]):
		f.write(str(inp_data[i,np.shape(inp_data)[1] - j - 1]))
	f.write('\";')
	f.write('cor_in <= \"' )
	for j in range(np.shape(out_data)[1]):
		f.write(str(out_data[i,j]))
	f.write('\" ; wait for 10 ns; \n');








#print(MAT_labels[0,:,:])
f.close()
