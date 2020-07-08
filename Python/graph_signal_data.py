# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 12:58:55 2020

@author: vascodebruijn
"""
import sys
import os
import torch
cluster = True
module_path = os.path.abspath(os.path.join('..'))
if cluster:
    sys.path.insert(1, '\\home\\nfs\\vascodebruijn\\graph-neural-networks-networks')
else:    
    sys.path.insert(1, 'C:\\Users\\vascodebruijn\\Documents\\GitHub\\graph-neural-networks')
import Utils.dataTools
import Modules.evaluation as ev
from sklearn.model_selection import train_test_split

class signal_data(Utils.dataTools._dataForClassification):
    
    def __init__(self,x,y,G,nTrain,nValid,nTest):
        super().__init__()
        (A,V) = G
        (N,_) = x.shape
        self.adjacencyMatrix = A
        self.nNodes = V
        self.nTotal = N
        self.nTrain = nTrain
        self.nValid = nValid
        self.nTest = nTest
        assert nTrain+nValid+nTest <= self.nTotal, "Issue with splitting the dataset in test and train"
        
        X_train, X_tmp, y_train,y_tmp = train_test_split(x,y,train_size=nTrain)
        X_test,X_valid,y_test,y_valid = train_test_split(X_tmp,y_tmp,train_size=nValid, test_size=nTest)
        
        self.samples['train']['signals'] = x[0:nTrain, :]
        self.samples['train']['targets'] = y[0:nTrain]
        self.samples['valid']['signals'] = x[nTrain:nTrain+nValid, :]
        self.samples['valid']['targets']=y[nTrain:nTrain+nValid]
        self.samples['test']['signals'] = x[nTrain+nValid:nTrain+nValid+nTest, :]
        self.samples['test']['targets'] =y[nTrain+nValid:nTrain+nValid+nTest]
        self.astype(torch.float64)
        
        self.samples['train']['targets'] = self.samples['train']['targets'].long()
        self.samples['valid']['targets'] = self.samples['valid']['targets'].long()
        self.samples['test']['targets'] = self.samples['test']['targets'].long()
        
        
        self.expandDims()

    def evaluate_GE(self, yHat, y, tol = 1e-9):
        guessing_vector = torch.argsort(yHat, descending=True)
        y_t = torch.reshape(y, (-1,1))
        (_,pos) = torch.where(guessing_vector == y_t)
        GE = torch.mean(pos.float())
        return GE