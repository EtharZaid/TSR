#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 08:01:52 2022
Transductive Survival Ranking
@author: u1876024
"""

import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from lifelines.utils import concordance_index

USE_CUDA = torch.cuda.is_available() 
from torch.autograd import Variable
def cuda(v):
    if USE_CUDA:
        return v.cuda()
    return v
def toTensor(v,dtype = torch.float,requires_grad = False):       
    return cuda(Variable(torch.tensor(v)).type(dtype).requires_grad_(requires_grad))
def toNumpy(v):
    if USE_CUDA:
        return v.detach().cpu().numpy()
    return v.detach().numpy()

#print('Using CUDA:',USE_CUDA)

def L2(z):
    """
    g = 5
    zz = torch.zeros((2,len(z)))
    zz[1] = z
    az = (torch.logsumexp(-g*zz,0)+torch.logsumexp(g*zz,0))/g #smooth approximation of abs https://math.stackexchange.com/questions/728094/approximate-x-with-a-smooth-function
    #az = torch.logsumexp(g*zz,0)*2/g-z-(2/g)*torch.log(torch.tensor(2)) #smooth approximation of abs 
    zz[1] = 1-az
    
    closs= torch.logsumexp(g*zz,0)/g #approx of max(0,1-abs(z))
    """
    closs = torch.exp(-3*(z**2)) #original approximation used in the paper 
    #closs = torch.exp(-0.5*(z**2)) #original approximation used in the paper "LARGE SCALE TRANSDUCTIVE SVMS"  by Collobert 2006    
    #closs= torch.max(toTensor([0],requires_grad=False), 1 - torch.abs(z)) #max(0,1-abs(z))
    return closs
# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
class TransductiveSurvivalRanker:
    def __init__(self,model=None,lambda_w=0.01,lambda_u = 0.0,p=1,lr=1e-2,Tmax = 200):#,lambdaw=0.010,p=1,Tmax = 100,lr=1e-1
        self.lambda_w = lambda_w
        self.lambda_u = lambda_u
        self.p = p
        self.Tmax = Tmax
        self.lr = lr
        self.model = model
        
    def fit(self,X_train,T_train,E_train,X_test = None):        
        #from sklearn.preprocessing import MinMaxScaler
        #self.MMS = MinMaxScaler().fit(T_train.reshape(-1, 1))#rescale y-values
        #T_train = 1e-3+self.MMS.transform(T_train.reshape(-1, 1)).flatten()        
        x = toTensor(X_train)
        if X_test is not None:
            X_test = toTensor(X_test)
        y = toTensor(T_train)
        e = toTensor(E_train)
        N,D_in = x.shape        
        H, D_out = D_in, 1
        if self.model is None:                    
            self.model = torch.nn.Sequential( 
                torch.nn.Linear(H, D_out,bias=True),
                #torch.nn.ReLU(),
                #torch.nn.Linear(H, D_out,bias=True),
                #torch.nn.Linear(int(H/2), D_out,bias=True),
                torch.nn.Tanh()
            )
        model = self.model
        model=cuda(model)
        learning_rate = self.lr
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay = 0.0)
        TT = self.Tmax  
        lambda_w = self.lambda_w 
        p = self.p
        L = []
        dT = T_train[:, None] - T_train[None, :] #dT_ij = T_i-T_j
        dP = (dT>0)*E_train
        dP = toTensor(dP,requires_grad=False)>0 # P ={(i,j)|T_i>T_j ^ E_j=1}
        dY = (y.unsqueeze(1) - y)[dP] #y.unsqueeze(1) - y is the same as dT?
        #import pdb;pdb.set_trace()
        self.bias = 0.0
        loss_uv = 0.0
        for t in (range(TT)):
            y_pred = model(x).flatten()
            #self.bias = torch.mean(y_pred)
            dZ = (y_pred.unsqueeze(1) - y_pred)[dP]  
            loss = torch.mean(torch.max(toTensor([0],requires_grad=False),1.0-dZ)) #hinge loss
            if X_test is not None and self.lambda_u > 0:
                y_tt = model(X_test).flatten()
                #self.bias = (torch.sum(y_pred)+torch.sum(y_tt))/(len(y_pred)+len(y_tt))#torch.mean(y_tt)#
                #y_tt = y_tt-self.bias
                #y_pred = y_pred - self.bias
                loss_u = torch.mean(L2(y_tt))#(torch.sum(L2(y_tt))+torch.sum(L2(y_pred)))/(len(y_pred)+len(y_tt))#
                loss+=self.lambda_u*loss_u
                loss_uv = loss_u.item()
            #import pdb;pdb.set_trace();
            #ww = torch.cat([w_.view(-1) for w_ in model[0].parameters()]) #weights of input
            ww = model[0].weight.view(-1) #only input layer weighths (excl bias from regularization)
            loss+=lambda_w*torch.norm(ww, p)**p #regularize
            L.append([loss.item(),loss_uv])
            model.zero_grad()
            loss.backward() # Calculate the gradient during the backward pass
            optimizer.step() #Performs a single optimization step (parameter update)
        ww = model[0].weight.view(-1)#torch.cat([w_.view(-1) for w_ in model[0].parameters()])
        self.ww = ww
        self.L = L
        self.model = model
        return self
    def decision_function(self,x):
        x = toTensor(x)
        return toNumpy(self.model(x)-self.bias).flatten()
    def getW(self):
        return toNumpy(self.ww/torch.linalg.norm(self.ww,ord=1))
    #By Ethar
    def score(self,X,Time,Event):
        return concordance_index(Time, X, Event)