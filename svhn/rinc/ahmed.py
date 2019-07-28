# Author: Noel Dawe <noel.dawe@gmail.com>
#
# License: BSD 3 clause

import gzip,pickle
import numpy as np
#f=gzip.open(r'D:\siva\Masters\Thesis\ETE_18\W1\mnist.pkl.gz')
#MNIST=pickle.load(f)

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
from sklearn.externals.six.moves import zip
#%%

X_train = np.load('dt_train_inp.npy')
y_train = np.load('train_labels.npy')

X_test = np.load('dt_test_inp.npy')
y_test = np.load('test_labels.npy')

MulticlassBNNout = np.zeros((26032,10))
MuliclassConfidence = np.zeros((26032,10))
MuliclassConfidence2 = np.zeros((26032,10))
MulticlassBNNout = np.zeros((26032,10))
for cur_class in range(10):

    cur_y_train = np.zeros(np.shape(y_train))
    cur_y_train[y_train == cur_class] = 1

    cur_y_test = np.zeros(np.shape(y_test))
    cur_y_test[y_test == cur_class] = 1


    cur_X_train = X_train
    cur_X_test = X_test

#%%

    total_estimators = 15
    bdt_real = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),n_estimators=total_estimators,learning_rate=1,algorithm  = 'SAMME')

    bdt_real.fit(cur_X_train,cur_y_train)
#%%
    all_estimators = bdt_real.estimators_

    BNNout = np.zeros((np.shape(cur_X_test)[0],total_estimators))
    for col,esti in zip(range(total_estimators),all_estimators):
        BNNout[:,col] = esti.predict(cur_X_test)

    all_weights = np.zeros((total_estimators,1))
    all_weights[:,0] = bdt_real.estimator_weights_
    sum_aw = np.sum(all_weights)

    BNNout_f = np.matmul(BNNout,all_weights)

    Confid = BNNout_f /  (np.sum(all_weights[:,0])/2)   
    BNNout_f = BNNout_f >  (np.sum(all_weights[:,0])/2)   
    
    MuliclassConfidence[:,cur_class] = Confid[:,0] 
    MulticlassBNNout[:,cur_class] = BNNout_f[:,0]

    BNNacc = np.sum((cur_y_test == BNNout_f[:,0].astype(int)).astype(int)).astype(float) / np.shape(BNNout_f)[0] *100.0;
    print "Accuracy of class: ",cur_class," is: ", BNNacc," Sum is : " , sum_aw
#%%
MuliclassConfidence2[(MuliclassConfidence[:,:]>=0) & (MuliclassConfidence[:,:] <0.7)] = 0
MuliclassConfidence2[(MuliclassConfidence[:,:]>=0.7) & (MuliclassConfidence[:,:] <1)] = 1
MuliclassConfidence2[(MuliclassConfidence[:,:]>=1) & (MuliclassConfidence[:,:] <1.3)] = 2
MuliclassConfidence2[(MuliclassConfidence[:,:]>=1.3)] = 3

f_o = np.zeros(26032)
    
for i in range(26032):
    ind = MulticlassBNNout[i,:]!=0
    if(ind.any() == False):
        f_o[i] = np.argmax(MuliclassConfidence2[i,:])
    else:
        temp = MuliclassConfidence2[i,:]
        temp[MulticlassBNNout[i,:]==0] = 0
        f_o[i] = np.argmax(temp)
    
accuracy = np.sum(f_o == y_test).astype(float)/26032.0

print "Overall Adaboost  Accuracy: " , accuracy


#%%
# bdt_discrete = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),n_estimators=600,learning_rate=1.5,algorithm="SAMME")

count_wrong =0
count_missclassified = 0

for i in range(26032):

    if(np.all(MulticlassBNNout[i,:] == 0) | np.sum(MulticlassBNNout[i,:]!=0) > 1):
        count_wrong = count_wrong + 1;
        ind = MulticlassBNNout[i,:]!=0
        if(ind.any() == False):
            f_o[i] = np.argmax(MuliclassConfidence2[i,:])
        else:
            temp = MuliclassConfidence2[i,:]
            temp[MulticlassBNNout[i,:]==0] = 0
            f_o[i] = np.argmax(temp)
        if(f_o[i] != y_test[i]):
            count_missclassified = count_missclassified + 1
        
        
        
