# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 12:58:55 2020

@author: vascodebruijn
"""
import sys
import os
import torch
cluster = os.getcwd() != 'C:\\Users\\vascodebruijn\\Documents\\GitHub\\Thesis\\Python'
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
        X_valid,X_test,y_valid,y_test = train_test_split(X_tmp,y_tmp,train_size=nValid, test_size=nTest)
        
        self.samples['train']['signals'] = X_train
        self.samples['train']['targets'] = y_train
        self.samples['valid']['signals'] = X_valid
        self.samples['valid']['targets']= y_valid
        self.samples['test']['signals'] = X_test
        self.samples['test']['targets'] =y_test
        self.astype(torch.float64)
        
        self.samples['train']['targets'] = self.samples['train']['targets'].long()
        self.samples['valid']['targets'] = self.samples['valid']['targets'].long()
        self.samples['test']['targets'] = self.samples['test']['targets'].long()
        
        
        self.expandDims()

    def evaluate_GE(self, yHat, y, tol = 1e-9):
        """
        evaluates the guessing entropy 
        yHat: key guessing vector 
        y: (Ground Truth) Classification vector
        """
        guessing_vector = torch.argsort(yHat, descending=True)
        y_t = torch.reshape(y, (-1,1))
        (_,pos) = torch.where(guessing_vector == y_t)
        GE = torch.mean(pos.float())
        return float(GE)