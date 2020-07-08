# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 13:54:28 2020

@author: vascodebruijn
"""

import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True # Comment this line if no LaTeX installation is available
matplotlib.rcParams['font.family'] = 'serif' # Comment this line if no LaTeX installation is available
matplotlib.rcParams['text.latex.preamble']=[r'\usepackage{amsmath}']
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats
from pyvis.network import Network
import plotly.graph_objects as go
import networkx as nx
import math
import dgl
import sys
import os
import pickle
import datetime
from copy import deepcopy

import torch; torch.set_default_dtype(torch.float64)
import torch.nn as nn
import torch.optim as optim

#import Albertos lib
module_path = os.path.abspath(os.path.join('..'))
sys.path.insert(1, 'C:\\Users\\vascodebruijn\\Documents\\GitHub\\graph-neural-networks')

import Utils.graphTools as graphTools
import Utils.dataTools
import Utils.graphML as gml

import Modules.architectures as archit
import Modules.model as model
import Modules.training as training
import Modules.evaluation as evaluation
import Modules.loss as loss


from Utils.miscTools import writeVarValues
from Utils.miscTools import saveSeed


#import own stuff
if module_path not in sys.path:
    sys.path.append(module_path+"\\Python")
import import_traces

sbox=(
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16) 


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
        
def generate_graph(x):
    """
    Generates graph used for graph signal classification
    Params:
        x: Traces in np-format (num_traces*num_features)
    Returns:
        A: adjacency matrix representing the edges
        V: number of vertixes (A=V*V)  
    """
    
    (_,V) = x.shape
    A = seq_connection(x)
    
    return (A,V)
    
def seq_connection(x):
    """
    Connects all nodes corresponding to temporal successive nodes
    Params:
        x: Traces in np-format (num_traces*num_features)
    Out:
        A: Adjacency matrix
    """
    (_,N) = x.shape
    A = np.zeros((N,N))
    for i in range(N-1):
        j= i+1
        A[i][j] = 1
        A[j][i] = 1
    return A

def corr_tresh_connection(x,c):
    """
    Connects nodes using Pearson Correlation as treshold value
    Params:
        x: Traces in np-format (num_traces*num_features)
        c: treshold value
    Out:
        A: Adjacency matrix
    """
    (_,N) = x.shape
    A = np.zeros((N,N))
    for i in range (N):
        for j in range(i,N):
            v_i= x[:,i]
            v_j= x[:,j]
            (corr,_) = scipy.stats.pearsonr(v_i, v_j)
            if(corr >= c):
                A[i,j] = 1
                A[j,i] = 1
    return A

def corr_knn_connection(x,k):
    """
    Connects nodes using Pearson Correlation with their K closest neighbours
    Params:
        x: Traces in np-format (num_traces*num_features)
        k: amount of neighbours
    Out:
        A: Adjacency matrix
    """
    (N,f) = x.shape
    A = np.zeros((N,N))
    C = np.zeros((N,N))
    #put all correlations row-wise in a matrix
    for i in range (N):
        v_i= x[:,i]
        for j in range(i,N):
            v_j= x[:,j]
            (corr,_) = scipy.stats.pearsonr(v_i, v_j)
            C[i,j] = corr
            C[j,i] = corr
        #when a row is full, select the k highest values         
        c_i = C[i,:]
        sort_idx = np.argsort(c_i)
        top = sort_idx[0:k]
        A[i, top] =1
    return A    

(traces, keys) = import_traces.get_DPA_traces()
keys=keys.flatten()
G = generate_graph(traces)
(A,V) = G
data = signal_data(traces.copy(),keys.copy(),G,8000,1000,1000 )
n = len(keys)

nFeatureBank = 1
nClasses = 9
k = 3

dimNodeSignals =[1, nFeatureBank, nFeatureBank]
dimLayersMLP= [nClasses]
nShiftTaps =[k, k]
nFilterNodes = [50, 50]
bias = False 
nonlinearity = nn.ReLU
nSelectedNodes= [50, 50] 
poolingSize=[50, 50]
GSO = A

EdgeNet = archit.EdgeVariantGNN(dimNodeSignals, nShiftTaps,nFilterNodes,bias,nonlinearity,nSelectedNodes,Utils.graphML.NoPool,poolingSize,dimLayersMLP, GSO)

lossFunction = nn.CrossEntropyLoss
optimAlg = 'ADAM'
learningRate = 0.001
beta1 = 0.9
beta2 = 0.999
trainer = training.Trainer
evaluator = evaluation.evaluate
thisOptim = optim.Adam(EdgeNet.parameters(), lr = learningRate, betas = (beta1,beta2))
useGPU = False

if useGPU and torch.cuda.is_available():
    device = 'cuda:0'
    torch.cuda.empty_cache()
else:
    device = 'cpu'
    
EdgeNetGNN = model.Model(EdgeNet,
                     lossFunction(),
                     thisOptim,
                     trainer,
                     evaluator,
                     device,
                     "EdgeNet",
                     'test') 

nEpochs = 100 # Number of epochs
batchSize = 5 # Batch size
validationInterval = 50 # How many training steps to do the validation
trainingOptions = {}

thisTrainVars = EdgeNetGNN.train(data, nEpochs, batchSize, **trainingOptions)

lossTrain = thisTrainVars['lossTrain']
costTrain = thisTrainVars['costTrain']
lossValid = thisTrainVars['lossValid']
costValid = thisTrainVars['costValid']

plt.plot(costValid)