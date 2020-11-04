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
from sklearn.model_selection import KFold
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
import random

from Utils.miscTools import writeVarValues
from Utils.miscTools import saveSeed

#import own stuff
if module_path not in sys.path:
    sys.path.append(module_path+"\\Python")
    
import import_traces
import graph_generation as gg
import graph_signal_data as gsd
import utils
import pdb
import mod_architectures as mod_archit

def nonlin(x):
    return x

#File handling 
thisFilename = 'GenericRun' # This is the general name of all related files
saveDirRoot = 'experiments' # Relative location where to save the file
hw = [bin(x).count("1") for x in range(256)]
#Hyperparams
hyperparam_settings =[]

input_args = sys.argv
if len(input_args) > 1:
    hyperparam_doc = input_args[1] 
    with open(hyperparam_doc,'r') as f:
        hyperparam_settings = json.loads(f.read())
    thisFilename = hyperparam_doc
    print(thisFilename)
else:
   hyperparam_settings=utils.generate_hyperparamlist([8,10,12,16,20],[3,4,5],[2,3,4],[("Threshold Correlation",0.2)]) 

fullCV = True
randomCV = False
attack_size = 0

if len( input_args) > 2:
    fullCV = input_args[2] == 'True' or input_args[2] == True
    
if len(input_args)>3:
    randomCV = input_args[3] == 'True ' or input_args[3] == True
    
if len(input_args)>4:
    attack_size = int(input_args[4])
#save params

 
saveDir = os.path.join(saveDirRoot, thisFilename) # Dir where to save all the results from each 

today = datetime.datetime.now().strftime("%Y%m%d")
saveDir=os.path.join(saveDir,today)
date = datetime.datetime.now().strftime("%Y%m%d%H%M")
 #Hour should be large enought discriminator
saveDir = saveDir + '-' + date
#writer = SummaryWriter()
# Create directory
if not os.path.exists(saveDir):
    os.makedirs(saveDir)
 
varsFile = os.path.join(saveDir,'hyperparameters.txt')
with open(varsFile, 'w+') as file:
    file.write('%s\n\n' % datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"))

resultsFile =  os.path.join(saveDir,'results.npy')  
results_best_File =  os.path.join(saveDir,'results_best.npy')  

predictionsFile = os.path.join(saveDir,'predictions.npy')
hyperParam_list = os.path.join(saveDir,'hyper_params.list')  

print (input_args)

if attack_size>0:
    results = np.empty((0,attack_size))
    results_best= np.empty((0,attack_size))
else:
    results = np.empty((0,1000))
    results_best= np.empty((0,1000))

#training params
nTrain = 0
nValid = 0
nTest = 1000

nEpochs = 100 # Number of epochs
batchSize = 200 # Batch size
validationInterval = 100 # How many training steps to do the validation
trainingOptions = {} 
printInterval = 5000
   
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
  #diff k
#hyperparam_settings.reverse()  
with open (hyperParam_list, 'w') as f:
    f.write(json.dumps(hyperparam_settings))
    
iteration = 0    
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
    size_dataset = hyperparams['Size Dataset']
    
    if cr:
        nClasses = 9
    else:
        nClasses = 256
        
    mask = None
    if dataset == 'ascad':
        mask = hyperparams['mask']
        nFeatures=700
    if dataset == 'dpa4':
        nFeatures=50
    #get the data and transform it into a graph
    (traces,keys,ptxts,masks)= import_traces.import_traces(cluster, dataset,mask, size_dataset) 
    
    (iv, hw) = import_traces.leakage_model(keys, ptxts, masks,dataset = dataset)
    leakage_model = ''
    if(cr):
        lm = np.asarray(hw)
        leakage_model = "HW"
    else:
        lm = np.asarray(iv)
        leakage_model="IV"
    
    if 'Feature Reduction Method' in hyperparams.keys():
        FR_method = hyperparams['Feature Reduction Method']
        traces = utils.feature_reduction(FR_method, 50, traces,lm, dataset,leakage_model,size_dataset)
        nFeatures = 50
        
    G = gg.get_graph(dataset,nFeatures,edge_function,threshold,traces)    
    (A,_) = G
    
    #create datamodel to use in GNN framework
    splits = 10
    if  randomCV:
        rnd_st = None
    else:
        rnd_st = 42
    #if len(hyperparam_settings) > 20:
    #    splits = 3
    cros_valid = KFold(n_splits = splits, shuffle=True,random_state=rnd_st)
    
    for train, test in cros_valid.split(traces):
        X_train, X_test, y_train, y_test = traces[train], traces[test], lm[train], lm[test]
        cv_ptxts, cv_masks,cv_keys = ptxts[test],masks[test],keys[test]
        data = gsd.signal_data(traces.copy(),keys.copy(),G,len(X_train),len(X_test),len(X_test),cross_eval=True, X_train=X_train,X_test=X_test,Y_train=y_train,Y_test=y_test )
    
    #data = gsd.signal_data(traces.copy(),keys.copy(),G,nTrain,nValid,nTest )
        (nTraces,nFeatures) = traces.shape
        
        #Moar architecture params
        bias = False 
        nonlinearity = nn.ReLU
        dimNodeSignals =[1] + [nFeatureBank]*nLayers
        dimLayersMLP= [nClasses]
        ML_dimLayersMLP = [nFeatures,nClasses]
        nShiftTaps =[k] * nLayers
        nFilterTaps = [k] * nLayers
        nFilterNodes = [nFeatures] * nLayers
        nAttentionHeads = [k] * nLayers
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
        GCATNet = archit.GraphConvolutionAttentionNetwork(dimNodeSignals, nFilterTaps, nAttentionHeads, bias, nonlinearity(), nSelectedNodes, poolingFn, poolingSize, dimLayersMLP, GSO)
        EdgeNet = archit.EdgeVariantGNN(dimNodeSignals, nShiftTaps,nFilterNodes,bias,nonlinearity,nSelectedNodes,poolingFn,poolingSize,dimLayersMLP, GSO)
        ConvNet = archit.SelectionGNN(dimNodeSignals, nFilterTaps, bias, nonlinearity,nSelectedNodes,poolingFn, poolingSize,dimLayersMLP,GSO)
        
        LinConvNet = mod_archit.SelectionGNN_Lin(dimNodeSignals, nFilterTaps, bias, nonlinearity,nSelectedNodes,poolingFn, poolingSize,dimLayersMLP,GSO)
        MLPConvNet = archit.SelectionGNN(dimNodeSignals, nFilterTaps, bias, nonlinearity,nSelectedNodes,poolingFn, poolingSize,ML_dimLayersMLP,GSO)
        LinMLPConvNet = mod_archit.SelectionGNN_Lin(dimNodeSignals, nFilterTaps, bias, nonlinearity,nSelectedNodes,poolingFn, poolingSize,ML_dimLayersMLP,GSO)

        LinGCATNet = mod_archit.GraphConvolutionAttentionNetwork_Lin(dimNodeSignals, nFilterTaps, nAttentionHeads, bias, nonlinearity(), nSelectedNodes, poolingFn, poolingSize, dimLayersMLP, GSO)
        LinEdgeNet = mod_archit.EdgeVariantGNN_Lin(dimNodeSignals, nShiftTaps,nFilterNodes,bias,nonlinearity,nSelectedNodes,poolingFn,poolingSize,dimLayersMLP, GSO)

        
        architectures = {}
        architectures['EdgeNet'] = EdgeNet
        architectures['ConvNet'] = ConvNet
        architectures['GCATNet'] = GCATNet
        architectures['LinConvNet'] = LinConvNet
        architectures['MLPConvNet'] = MLPConvNet
        architectures['LinMLPConvNet'] = LinMLPConvNet
        architectures['LinGCATNet'] =LinGCATNet
        architectures['LinEdgeNet'] = LinEdgeNet

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
                             saveDir) 
        
        print("Start Training")
        print (iteration)
        iteration = iteration +1
        
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
        print (attack_size)
        thisTrainVars = GNNModel.train(data, nEpochs, batchSize, **trainingOptions)
        if attack_size > 0:
            evalVars = GNNModel.evaluate(data, ptx = cv_ptxts, offset = cv_masks, keys =cv_keys,lm=leakage_model,dataset=dataset ,nRuns=100,attack_size=attack_size)
        else:
            evalVars = GNNModel.evaluate(data, ptx = cv_ptxts, offset = cv_masks, keys =cv_keys,lm=leakage_model,dataset=dataset ,nRuns=100)
        
        finish = time.perf_counter()
        runtime = finish-start
        print(runtime)    
        writeVarValues(varsFile, {'Runtime':runtime})  
        
        writeVarValues(varsFile, evalVars)    
        results = np.append(results,[evalVars['GE_last']],axis=0)
        results_best = np.append(results_best,[evalVars['GE_best']],axis=0)

        np.save(resultsFile,results)
        np.save(results_best_File,results_best)
        np.save (predictionsFile+str(iteration),evalVars['Prediction'])
        #if were hyperparameter tuning, only run 1 iteration
        if not fullCV:
            break
#utils.make_fig(thisTrainVars, saveDir,nEpochs, validationInterval)
if not cluster:
    plt.plot(np.transpose(results))
    plt.plot(np.mean(results,axis=0))
