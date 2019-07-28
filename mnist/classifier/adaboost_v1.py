# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 16:25:18 2018

@author: siva

Adaboost Implementation

"""

import numpy as np
import lvl_wise_copy2
'''
import gzip,pickle
f=gzip.open(r'/home/siva/Thesis/Ete18/W1/mnist.pkl.gz')
MNIST=pickle.load(f)
'''

#from sklearn.tree import DecisionTreeClassifier
#from sklearn.externals.six.moves import zip

def adaboost_train(X_train,y_train,X_test,y_test,n_trees,lvl,init_weights):

	#weights = np.ones(np.shape(X_train)[0])/np.shape(X_train)[0]
	weights = np.copy(init_weights)
	#new_weights = np.ones(np.shape(X_train)[0])/np.shape(X_train)[0]
	label_array = np.zeros((n_trees,2**lvl))
	feats_array = np.zeros((n_trees,lvl))

	new_weights = np.copy(init_weights)
	alpha = np.zeros(n_trees)
	predicted_outputs_store = np.zeros((n_trees,np.shape(X_train)[0]))
	predicted_test_out_store = np.zeros((n_trees,np.shape(X_test)[0]))
	#clf = DecisionTreeClassifier(max_depth=3,criterion='entropy')
	final_decision = np.zeros(np.shape(X_train)[0])
	final_test_decision = np.zeros(np.shape(X_test)[0])

	for i in range(n_trees):
		(feature_array, lists_new, label_class) = lvl_wise_copy2.construct_lut(X_train,y_train,lvl,weights)
		#print(feature_array)
		label_array[i,:] = label_class
		feats_array[i,:] = feature_array
		predicted_outputs = lvl_wise_copy2.predict_out_lut(X_train,y_train,feature_array,label_class,lvl)
		
		#clf.fit(X_train,y_train,sample_weight = weights)
		#predicted_outputs = clf.predict(X_train)
		predicted_outputs_store[i] = predicted_outputs
		#predicted_test_out = clf.predict(X_test)
		predicted_test_out = lvl_wise_copy2.predict_out_lut(X_test,y_test,feature_array,label_class,lvl)
		predicted_test_out_store[i] = predicted_test_out

		
		cur_error = np.sum(weights[np.where(predicted_outputs != y_train)])
		#print('tree : ' + str(i) + '  cur_error : ' + str(cur_error))



		#test_curr_accuracy = lvl_wise_copy2.predict(X_test,y_test,feature_array,label_class,lvl)

		#print( 'Test Accuracy is  : ' + str(test_curr_accuracy))

		if cur_error <= 0:
			alpha[i] = 10.36
		else:
			alpha[i] = 0.5 * np.log2((1-cur_error)/cur_error)

		#print(alpha[i])

		#final_decision += alpha[i] * (2 * predicted_outputs - 1) 
		#print (np.sum(final_decision>0))

		new_weights[np.where(predicted_outputs != y_train)] = 0.5  * weights[np.where(predicted_outputs != y_train)] / cur_error
		new_weights[np.where(predicted_outputs == y_train)] = 0.5  * weights[np.where(predicted_outputs == y_train)] / (1 - cur_error)

		weights = new_weights

		#fd_sign = (np.sign(final_decision) + 1)/2
		#accuracy = np.sum(fd_sign == y_train).astype(float) / np.shape(X_train)[0]

		#print('tree : ' + str(i) + '  Training Accuracy : ' + str(accuracy))


		#final_test_decision += alpha[i] * (2 * predicted_test_out - 1) 
		#fd_test_sign = (np.sign(final_test_decision) + 1)/2
		#accuracy_test = np.sum(fd_test_sign == y_test).astype(float) / np.shape(X_test)[0]

		#print('tree : ' + str(i) + '  Test Accuracy : ' + str(accuracy_test))
		#print(init_weights[0])


	return predicted_test_out_store,predicted_outputs_store,alpha,label_array,feats_array


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

y_train[y_train == 1 ] = 10
y_train[y_train != 10] =  0
y_train[y_train == 10] =  1

y_test[y_test == 1] =  10
y_test[y_test != 10 ] = 0
y_test[y_test == 10] =  1
predicted_outputs_store = adaboost_train(X_train,y_train,X_test,y_test,20,7)

'''


'''
y_train[y_train != 8 ] = 0
y_train[ y_train == 8] = 1


y_test[ y_test != 8 ] = 0
y_test[ y_test == 8] = 1
'''