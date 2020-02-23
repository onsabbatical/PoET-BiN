import numpy as np
from multiprocessing import Pool
import adaboost_v1

X_train = np.load('dt_train_inp.npy')
y_train = np.load('dt_train_out.npy')

X_test = np.load('dt_test_inp.npy')
y_test = np.load('dt_test_out.npy')

#X_train = np.append(X_train,X_test,axis=0)
#y_train = np.append(y_train,y_test,axis=0)

#X_test = np.append(X_test,X_train[320000:,:],axis=0)
#y_test = np.append(y_test,y_train[320000:,:],axis=0)

#X_train = X_train[:32000,:]
#y_train = y_train[:32000,:]

print(np.shape(X_train))
print(np.shape(X_test))
print(np.shape(y_train))
print(np.shape(y_test))

cur_pred_y_train_store = np.zeros((np.shape(y_train)[0],np.shape(y_train)[1])).astype(bool)
cur_pred_y_test_store = np.zeros((np.shape(y_test)[0],np.shape(y_test)[1])).astype(bool)


tot_est = 40  
hyp_inp = 8
total_estimators = tot_est
c_feats_store = np.zeros((10*hyp_inp,int(total_estimators/hyp_inp),hyp_inp,hyp_inp))
c_weights_store = np.zeros((10*hyp_inp,int(total_estimators/hyp_inp) + 1,hyp_inp))
def adaboost_train_req(cur_y_train,cur_y_test):
    Features_next_level_test = np.zeros((np.shape(cur_y_test)[0],int(total_estimators/hyp_inp)))
    Features_next_level_train = np.zeros((np.shape(cur_y_train)[0],int(total_estimators/hyp_inp)))
    alpha_next_level = np.zeros(int(total_estimators/hyp_inp))
    new_init_weights = np.ones(np.shape(X_train)[0])/np.shape(X_train)[0]
    init_weights = np.ones(np.shape(X_train)[0])/np.shape(X_train)[0]
    c_label_array = np.zeros((int(total_estimators/hyp_inp),hyp_inp,2**hyp_inp))
    c_feats_array = np.zeros((int(total_estimators/hyp_inp),hyp_inp,hyp_inp))
    c_weights_array = np.zeros((int(total_estimators/hyp_inp) + 1,hyp_inp))

    for i in range(int(total_estimators/hyp_inp)):
        BNNout_temp,BNNout_train_temp,alpha,label_array,feats_array = adaboost_v1.adaboost_train(X_train,cur_y_train,X_test,cur_y_test,hyp_inp,hyp_inp,init_weights)
        c_label_array[i,:,:] = label_array
        c_feats_array[i,:,:] = feats_array
        BNNout = np.transpose(BNNout_temp)
        c_weights_array[i,:] = alpha 
        BNNout_train = np.transpose(BNNout_train_temp)
        cur_weights = np.zeros((hyp_inp,1))
        cur_weights[:,0] = alpha
        sum_cur = np.sum(cur_weights)
        th = sum_cur * 0.5
        Features_next_level_test[:,i] = np.matmul(BNNout,cur_weights)[:,0] > th
        Features_next_level_train[:,i] = np.matmul(BNNout_train,cur_weights)[:,0] > th
        cur_error = np.sum(init_weights[cur_y_train != Features_next_level_train[:,i]])
        new_init_weights[np.where(Features_next_level_train[:,i] != cur_y_train)] = 0.5  * init_weights[np.where(Features_next_level_train[:,i] != cur_y_train)] / cur_error
        new_init_weights[np.where(Features_next_level_train[:,i] == cur_y_train)] = 0.5  * init_weights[np.where(Features_next_level_train[:,i] == cur_y_train)] / (1 - cur_error)
        if cur_error <= 0:
            alpha_next_level[i] = 10.36
        else:
            alpha_next_level[i] = 0.5 * np.log2((1-cur_error)/cur_error)
        init_weights = new_init_weights

    all_weights = np.zeros((int(total_estimators/hyp_inp),1))
    all_weights[:,0] = alpha_next_level
    c_weights_array[-1,:int(total_estimators/hyp_inp)] = alpha_next_level
    sum_aw = np.sum(all_weights)
    th = sum_aw / 2
    #all_weights[:,0] = all_weights[:,0] / sum_aw
    BNNout_f = np.matmul(Features_next_level_test,all_weights)
    BNNout_f_train = np.matmul(Features_next_level_train,all_weights)
    BNNout_f_bin = BNNout_f >  th 
    BNNout_f_train_bin = BNNout_f_train > th
    BNNacc = np.sum((cur_y_test == BNNout_f_bin[:,0].astype(int)).astype(int)).astype(float) / np.shape(BNNout_f_bin)[0] *100.0;
    print ("Accuracy of pixel is ", BNNacc," Sum is : " , sum_aw)

    return BNNout_f_train_bin[:,0],BNNout_f_bin[:,0],c_label_array,c_feats_array,c_weights_array
'''
def adaboost_train_req(cur_y_train,cur_y_test):

	for i in range(int(total_estimators/6)):
		cur_pred_y_train1,cur_pred_y_test1 = adaboost_v1.adaboost_train(X_train,cur_y_train,X_test,cur_y_test,6,6)
	return cur_pred_y_train1,cur_pred_y_test1 
'''
if __name__ == '__main__':
    p = Pool(30)
    c_label_store = np.zeros((10*hyp_inp,int(total_estimators/hyp_inp),hyp_inp,2**hyp_inp))

    cur_y_train_list = [y_train[:,0]]
    cur_y_test_list = [y_test[:,0]]

    for i in range(79):
        cur_y_train_list.append(y_train[:,i+1])
        cur_y_test_list.append(y_test[:,i+1])

    cur_pred_y_train_list ,cur_pred_y_test_list,c_label_array_list,c_feats_array_list,c_weights_array_list  = zip(*p.starmap(adaboost_train_req,zip(cur_y_train_list, cur_y_test_list)))

    for i in range(80):
        cur_pred_y_train_store[:,i] = cur_pred_y_train_list[i]
        cur_pred_y_test_store[:,i] = cur_pred_y_test_list[i]
        c_label_store[i,:,:,:] = c_label_array_list[i]
        c_feats_store[i,:,:,:] = c_feats_array_list[i]
        c_weights_store[i,:,:] = c_weights_array_list[i]

    #print('current_output : ' , i)
    np.save("predicted_train_outputs.npy",cur_pred_y_train_store)
    np.save("predicted_test_outputs.npy",cur_pred_y_test_store)
    np.save("label_store.npy",c_label_store)
    np.save("feats_store.npy",c_feats_store)
    np.save("weights_store.npy",c_weights_store)
