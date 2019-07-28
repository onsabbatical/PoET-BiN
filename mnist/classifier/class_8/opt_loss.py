import torch.optim as optim
import numpy as np
from numpy import random
import math
import torch
from torch.autograd import Variable

def clipping_scaling(delta,net):
	#new_params = []
	for (name,i),grads_i in zip(net.named_parameters(),delta):
		#print(name)
		if (('conv' in name) or ('fc' in name)) and ('weight' in name):
			#print(name)
			i.data = torch.clamp(i.data - 1*(grads_i),-1,1)
			#i.data = i.data - grads_i
			#if 'fc00' in name:
				#print(i)

		else:
			i.data = i.data - grads_i

def update(delta,net):
	for i,new_params_i in zip(net.parameters(),new_params_t):
		i.data = new_params_i



class Adam(optim.Adam):
    def __init__(self,params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,weight_decay=0):
        super(Adam,self).__init__(params,lr=lr, betas=betas, eps=eps,weight_decay=weight_decay)

    def false_step(self,cur_lr):

        W_delta = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = cur_lr * math.sqrt(bias_correction2) / bias_correction1
                
                W_delta.append(step_size * torch.div(exp_avg, denom))
        return W_delta    

def hinge_loss(outputs,labels,my_batch):
        lb2 = torch.LongTensor(my_batch,1).cuda()
        lb2[:,0] = labels
        y_onehot = (torch.FloatTensor(my_batch, 10)).cuda()
        y_onehot.zero_()
        y_onehot.scatter_(1, lb2, 1)

        y_final = Variable(y_onehot.cuda())
        y_f = 2*y_final - 1

        l1 = outputs * y_f
        l2 = 1. - l1
        l_zero = Variable(torch.zeros(my_batch,10).cuda())
        l3 = torch.max(l_zero,l2)
        l4 = torch.mul(l3,l3)
        loss = l4.mean()
        return loss	