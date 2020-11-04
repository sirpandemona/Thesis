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
    
    def __init__(self,x,y,G,nTrain,nValid,nTest,cross_eval=False, X_train=None,X_test=None,Y_train=None,Y_test=None):
        super().__init__()
        (A,V) = G
        (N,_) = x.shape
        self.adjacencyMatrix = A
        self.nNodes = V
        self.nTotal = N
        self.nTrain = nTrain
        self.nValid = nValid
        self.nTest = nTest
        #assert nTrain+nValid+nTest <= self.nTotal, "Issue with splitting the dataset in test and train"
        
        X_valid,Y_valid = X_test,Y_test

        
        self.samples['train']['signals'] = X_train
        self.samples['train']['targets'] = Y_train
        self.samples['valid']['signals'] = X_valid
        self.samples['valid']['targets']= Y_valid
        self.samples['test']['signals'] = X_test
        self.samples['test']['targets'] =Y_test
        self.astype(torch.float64)
        
        self.samples['train']['targets'] = self.samples['train']['targets'].long()
        self.samples['valid']['targets'] = self.samples['valid']['targets'].long()
        self.samples['test']['targets'] = self.samples['test']['targets'].long()
        
        
        self.expandDims()