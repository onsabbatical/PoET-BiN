import numpy as np

'''
import gzip,pickle
f=gzip.open(r'D:\siva\Masters\Thesis\04_ETE_18\W1\mnist.pkl.gz')
MNIST=pickle.load(f)

logFile=r'D:\siva\Masters\Thesis\06_AUT_18\W3\Mec\logFile.log'
logFile2=r'D:\siva\Masters\Thesis\06_AUT_18\W3\Mec\logFile2.log'
log=open(logFile,'w')
log2=open(logFile2,'w')
'''
#%%

def best_feature(X,y,feature_list,lists,lvl,weights):

	if lvl == 0:
		min_entropy = 1

	else:
		min_entropy = 2 ** (lvl)
	best_feat = feature_list[0]

	for j,cur_feature in enumerate(feature_list):
		entropy_lvl = 0
		#print('feat' + str(cur_feature))

		for cur_node in range(np.shape(lists)[0]):
			ent1 = 0
			ent2 = 0

			#node_list = np.array(np.sum(cur_node != -1))
			node_list = lists[cur_node, lists[cur_node] != -1 ]
			#print('node_list' + str(node_list))
			#y_int = y_train[node_list]
			#ln = np.sum(y[X[node_list,cur_feature] == 0] == 0) 
			#a = node_list[X[node_list,cur_feature] == 0]
			#print('a ' + str(a))
			subl_nodelist = node_list[X[node_list,cur_feature] == 0]
			ln = np.sum(weights[subl_nodelist[y[subl_nodelist] == 0]])
			#print('ln : ' + str(ln))
			#lp = np.sum(y[X[node_list,cur_feature] == 0] == 1)
			lp = np.sum(weights[subl_nodelist[y[subl_nodelist] == 1]])
			#print('lp : ' + str(lp))

			total_l = (ln + lp).astype(float)

			#rn = np.sum(y[X[node_list,cur_feature] == 1] == 0)
			subr_nodelist = node_list[X[node_list,cur_feature] == 1]
			rn = np.sum(weights[subr_nodelist[y[subr_nodelist] == 0]])
			#rp = np.sum(y[X[node_list,cur_feature] == 1] == 1)
			rp = np.sum(weights[subr_nodelist[y[subr_nodelist] == 1]])

			total_r = (rn + rp).astype(float)

			if total_l == 0:
				ent1 = 0
			elif ln == 0:
				ent1 = 0
			elif lp ==0:
				ent1 = 0
			else:
				ent1 = -1 / (total_l + total_r ) *(ln*np.log2(ln/total_l)  + lp * np.log2(lp/total_l))
				#print('hi_l' + str(ent1))

			if total_r == 0:
				ent2 = 0
			elif rn == 0:
				ent2 = 0
			elif rp == 0:
				ent2 = 0
			else:	
				ent2 = -1 / ( total_l + total_r )*(rn*np.log2(rn/total_r)  + rp * np.log2(rp/total_r))
				#print('hi_r' + str(ent2))

			entropy_node = ent1 + ent2

			entropy_lvl += entropy_node


		if entropy_lvl < min_entropy:

			min_entropy = entropy_lvl
			best_feat = cur_feature
			#print(best_feat)



	list_new = np.ones((2*np.shape(lists)[0],np.shape(X)[0])).astype(int)
	list_new = -1 * list_new

	for cur_node in range(np.shape(lists)[0]):
		node_list = lists[ cur_node, lists[cur_node] != -1 ]
		list_new[2*cur_node,:np.sum(X[node_list,best_feat] == 0) ] = node_list[X[node_list,best_feat] == 0]
		list_new[2*cur_node + 1,:np.sum(X[node_list,best_feat] == 1) ] =  node_list[X[node_list,best_feat] == 1]

	#print('lvl = ' + str(lvl)  + ', best_feat = ' + str(best_feat))
		

	return (best_feat,list_new)

def assign_class(X,y,lists,weights):

	label_class = np.zeros(np.shape(lists)[0])

	for cur_node in range(np.shape(lists)[0]):
		node_list = lists[cur_node,  lists[cur_node] != -1 ]
		if np.sum(weights[node_list[y[node_list] == 1]]) > np.sum(weights[node_list[y[node_list] == 0]]):
			label_class[cur_node] = 1

	
	return label_class		

#%%


def construct_lut(X,y,max_features,weights):

	feature_list = np.arange(np.size(X,1))
	lists = np.arange(np.size(X,0))
	lists_new = np.zeros((1,len(lists))).astype(int)
	lists_new[0] = lists

	feature_array = np.zeros(max_features)

	for i in range(max_features):
		feature_list_new = feature_list[feature_list >=0]
		(best_feat2,lists_new) = best_feature(X,y,feature_list_new,lists_new,i,weights)
		feature_array[i] = best_feat2.astype(int)
		feature_list[best_feat2] = -1 


	label_class = assign_class(X,y,lists_new,weights)
	#print(label_class)
	return (feature_array, lists_new, label_class)


#%%


def predict(X_train,y_train,feature_array,label_class,lvl):
	m=lvl
	indices_my = np.arange(2**lvl)
	sel_ind = indices_my[label_class == 1]
	sparse_mat = np.flip((((sel_ind[:,None] & (1 << np.arange(m)))) > 0).astype(int),1)
	feature_array = feature_array.astype(int)
	y_predicted = np.zeros(np.shape(X_train)[0])
	for i in range(np.shape(X_train)[0]):
		X_t = (X_train[i,feature_array] == sparse_mat)
		y_predicted[i] = np.any(np.sum(X_t,axis = 1) == lvl).astype(int)

	accuracy = np.sum(y_predicted == y_train).astype(float) / np.shape(X_train)[0]

	return accuracy

def predict_out_lut(X_train,y_train,feature_array,label_class,lvl):
	m=lvl
	indices_my = np.arange(2**lvl)
	sel_ind = indices_my[label_class == 1]
	sparse_mat = np.flip((((sel_ind[:,None] & (1 << np.arange(m)))) > 0).astype(int),1)
	feature_array = feature_array.astype(int)
	y_predicted = np.zeros(np.shape(X_train)[0])
	for i in range(np.shape(X_train)[0]):
		X_t = (X_train[i,feature_array] == sparse_mat)
		y_predicted[i] = np.any(np.sum(X_t,axis = 1) == lvl).astype(int)

	return y_predicted
'''
def vhdl_gen(feature_array,label_class):
	dict1 = {'1' : ' ', '0' : ' not('}
	dict2 = {'1' : ' ', '0' : ' ) '}
	pos = np.where(label_class == 1)
	for i in range(len(pos)):

		not_pos = '{0:06b}'.format(pos[i])
		log.write('tree_0_' + 'path_' + str(i) + ' = ')
		for j in range(5):
			log.write(dict1[not_pos[j]] 'feat_' + str(j) + dict2[not_pos[j]] + ' and ')
		log.write(dict1[not_pos[5]] 'feat_' + str(5) + dict2[not_pos[5]] + '; \n')


	log2.write('lut_0 = ')
	for i in range(len(pos) - 1):
		log2.write('tree_0_' + 'path_' + str(i) + ' or ')

	log2.write('tree_0_' + 'path_' + str(len(pos)) + ' ; ')

'''

'''
train_set = np.append(MNIST[0][0],MNIST[0][1].reshape(-1,1),axis=1)
valid_set = np.append(MNIST[1][0],MNIST[1][1].reshape(-1,1),axis=1)
test_set = np.append(MNIST[2][0],MNIST[2][1].reshape(-1,1),axis=1)

train_set = np.append (train_set,valid_set,axis = 0)

X_train, y_train  = train_set[:,:-1], train_set[:,-1]
X_test, y_test = test_set[:,:-1], test_set[:,-1]

X_train[X_train >= 0.4] = 1 
X_train[X_train < 0.4] = 0

X_test[X_test >= 0.4] = 1 
X_test[X_test < 0.4] = 0

y_train[y_train != 1 ] = 0
y_train[ y_train == 1] = 1


y_test[ y_test != 1 ] = 0
y_test[ y_test == 1] = 1

weights = np.ones(np.shape(X_train)[0])/np.shape(X_train)[0]
 
lvl = 6
(feature_array, lists_new, label_class) = construct_lut(X_train,y_train,lvl,weights)

accuracy = predict(X_train,y_train,feature_array,label_class,lvl)

vhdl_gen(feature_array,label_class)

print('VHDL Generated')

print( 'Training Accuracy is  : ' + str(accuracy))

accuracy = predict(X_test,y_test,feature_array,label_class,lvl)
print( 'Test Accuracy is  : ' + str(accuracy))
'''

'''

X_train = np.zeros((7,3))
y_train = np.zeros(7)

X_train[1,2] = X_train[2,1] =  X_train[3,1] = X_train[3,2] = X_train[4,0] = X_train[5,0] = X_train[5,2] = X_train[6,0] = X_train[6,1] = y_train[1] = y_train[3] = y_train[6] = 1
''' 