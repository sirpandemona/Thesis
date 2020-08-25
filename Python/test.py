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
import sys
import os
import datetime
import time
import torch; torch.set_default_dtype(torch.float64)
import torch.nn as nn
import torch.optim as optim
import json
#shortcut to check which path should be used
cluster = os.getcwd() != 'C:\\Users\\vascodebruijn\\Documents\\GitHub\\Thesis\\Python'
#import Albertos lib
module_path = os.path.abspath(os.path.join('..'))
if cluster:
    sys.path.insert(1, '\\home\\nfs\\vascodebruijn\\graph-neural-networks-networks')
else:    
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
import graph_generation as gg
import graph_signal_data as gsd
import utils
#File handling 
graphType = 'Signal Graph' # Type of graph
thisFilename = 'SigGNN' # This is the general name of all related files
saveDirRoot = 'experiments' # Relative location where to save the file
saveDir = os.path.join(saveDirRoot, thisFilename) # Dir where to save all the results from each run
today = datetime.datetime.now().strftime("%Y%m%d%H%M") #Hour should be large enought discriminator
saveDir = saveDir + '-' + graphType + '-' + today
#writer = SummaryWriter()
# Create directory
if not os.path.exists(saveDir):
    os.makedirs(saveDir)
    
#save params    
varsFile = os.path.join(saveDir,'hyperparameters.txt')
with open(varsFile, 'w+') as file:
    file.write('%s\n\n' % datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"))

resultsFile =  os.path.join(saveDir,'results.npy')  
hyperParam_list = os.path.join(saveDir,'hyper_params.list')  


#training params
nTrain = 8000
nValid = 1000
nTest = 1000
nEpochs = 10 # Number of epochs
batchSize = 100 # Batch size
validationInterval = 1000 # How many training steps to do the validation
trainingOptions = {} 
printInterval = 500
   
trainingOptions['saveDir'] = saveDir
trainingOptions['printInterval'] = printInterval
trainingOptions['validationInterval'] = validationInterval

writeVarValues(varsFile, {'nTrain': nTrain,
                          'nValid': nValid,
                          'nTest': nTest,
                          'nEpochs': nEpochs,
                          'batchSize': batchSize,
                          'validationInterval': validationInterval})
   

#Moar training params
lossFunction = nn.CrossEntropyLoss
optimAlg = 'ADAM'
learningRate = 0.001
beta1 = 0.9
beta2 = 0.999
trainer = training.Trainer
evaluator = gg.evaluateGE
useGPU = False
if useGPU and torch.cuda.is_available():
    device = 'cuda:0'
    torch.cuda.empty_cache()
else:
    device = 'cpu'
    
writeVarValues(varsFile, {'lossFunction': lossFunction,
                          'OptimAlg' : optimAlg,
                          'learningRate': learningRate,
                          'beta1': beta1,
                          'beta2': beta2,
                          'device': device,
                          })
  
#Hyperparams
hyperparam_settings =[]
hyperparams_base =  {'F': 4,
                          'nClasses' : 9,
                          'k' : 2,
                          'nLayers': 3,
                          'Feature Reduction': True,
                          'Class Reduction' : True,
                          'Used Architecture': 'ConvNet',
                          'Edge Function': "Successive",
                          'EdgeFn Threshold': 0.5,
                          'dataset': 'dpa4'
                          }
hyperparam_settings.append(hyperparams_base)

candidateF = [8,12,16,20,24]
candidateL = [2,3,4]
candidateK = [2,3,4,5]
candidateFn= [("Threshold Correlation",0.8),("Threshold Correlation",0.5),("Threshold Correlation",0.2),("Successive",0.5),("KNN Correlation",1),("KNN Correlation",2),("KNN Correlation",4),("KNN Correlation",8)]

candidateFn= [("Successive",0.5)]
candidateF = [16]
candidateL = [3]


results = np.empty((0,100))

for F  in candidateF:
    for L in candidateL:
        for K in candidateK:
            for efn in candidateFn:
                (fn,c) = efn
                hyperparam_settings.append({'F': F,
                          'nClasses' : 9,
                          'k' : K,
                          'nLayers': L,
                          'Feature Reduction': True,
                          'Class Reduction' : True,
                          'Used Architecture': 'ConvNet',
                          'Edge Function': fn,
                          'EdgeFn Threshold': c,
                          'dataset': 'dpa4'
                          })
#diff k
with open (hyperParam_list, 'w') as f:
    f.write(json.dumps(hyperparam_settings))
    
    
for hyperparams in hyperparam_settings:
    nFeatureBank = hyperparams['F']
    k = hyperparams['k']
    nLayers =  hyperparams['nLayers']
    fr = hyperparams['Feature Reduction']
    cr = hyperparams['Class Reduction']
    dataset = hyperparams['dataset']
    chosen_arch = hyperparams['Used Architecture']
    edge_function = hyperparams['Edge Function']
    threshold = hyperparams['EdgeFn Threshold']
    edge_fn = gg.get_edge_fn(edge_function,threshold)
    if (cr):
        nClasses = 9
    else:
        nclasses = 256
      
    #get the data and transform it into a graph
    (traces, keys) = import_traces.import_traces(cluster,cr,fr, dataset)
    G = gg.generate_graph(traces,edge_fn)
    
    (A,V) = G
    
    #create datamodel to use in GNN framework
    data = gsd.signal_data(traces.copy(),keys.copy(),G,nTrain,nValid,nTest )
    (nTraces,nFeatures) = traces.shape
    
    #Moar architecture params
    bias = False 
    nonlinearity = nn.ReLU
    dimNodeSignals =[1] + [nFeatureBank]*nLayers
    dimLayersMLP= [nClasses]
    nShiftTaps =[k] * nLayers
    nFilterTaps = [k] * nLayers
    nFilterNodes = [5] * nLayers
    GSO = A
    
    writeVarValues(varsFile, {'bias': bias,
                              'nonlinearity' : nonlinearity,
                              'dimNodeSignals': dimNodeSignals,
                              'nShiftTaps': nShiftTaps,
                              'nFilterNodes': nFilterNodes,
                              'dimLayersMLP': dimLayersMLP,
                              })
    #pooling stuff
    poolingFn = Utils.graphML.NoPool
    nSelectedNodes= [nFeatures] *nLayers
    poolingSize=[nFeatures]*nLayers
    
    #Put all the vars in the architecture
    EdgeNet = archit.EdgeVariantGNN(dimNodeSignals, nShiftTaps,nFilterNodes,bias,nonlinearity,nSelectedNodes,poolingFn,poolingSize,dimLayersMLP, GSO)
    ConvNet = archit.SelectionGNN(dimNodeSignals, nFilterTaps, bias, nonlinearity,nSelectedNodes,poolingFn, poolingSize,dimLayersMLP,GSO)
    
    architectures = {}
    architectures['EdgeNet'] = EdgeNet
    architectures['ConvNet'] = ConvNet
    netArch = architectures[chosen_arch]
    
    
    netArch.to(device)
    thisOptim = optim.Adam(EdgeNet.parameters(), lr = learningRate, betas = (beta1,beta2))
    
    GNNModel = model.Model(netArch,
                         lossFunction(),
                         thisOptim,
                         trainer,
                         evaluator,
                         device,
                         chosen_arch,
                         'test') 
    
    print("Start Training")
    
    
    writeVarValues(varsFile, {'nFeatureBank': nFeatureBank,
                              'nClasses' : nClasses,
                              'k' : k,
                              'nLayers': nLayers,
                              'Feature Reduction': fr,
                              'Class Reduction' : cr,
                              'Used Architecture': chosen_arch,
                              'Edge Function': edge_function,
                              'EdgeFn Threshold': threshold
                              })
    
    start = time.perf_counter()
    
    thisTrainVars = GNNModel.train(data, nEpochs, batchSize, **trainingOptions)
    evalVars = GNNModel.evaluate(data)
    
    finish = time.perf_counter()
    runtime = finish-start
        
    writeVarValues(varsFile, {'Runtime':runtime})  
    
    writeVarValues(varsFile, evalVars)    
    results = np.append(results,[evalVars['GE_best']],axis=0)
    np.save(resultsFile,results)
#utils.make_fig(thisTrainVars, saveDir,nEpochs, validationInterval)

