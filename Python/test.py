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
from torch.utils.tensorboard import SummaryWriter


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
today = datetime.datetime.now().strftime("%Y%m%d%H") #Hour should be large enought discriminator
saveDir = saveDir + '-' + graphType + '-' + today
#writer = SummaryWriter()
# Create directory
if not os.path.exists(saveDir):
    os.makedirs(saveDir)
    
#save hyperparams    
varsFile = os.path.join(saveDir,'hyperparameters.txt')
with open(varsFile, 'w+') as file:
    file.write('%s\n\n' % datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"))

#Hyperparams
nFeatureBank = 2
k = 3
nLayers = 2 
fr = True
cr = True
dataset = "dpa4"
chosen_arch = 'ConvNet'
edge_function = "Threshold Correlation"
threshold = 0.5
edge_fn = gg.get_edge_fn(edge_function,threshold)
if (cr):
    nClasses = 9
else:
    nclasses = 256
    

#training params
nTrain = 8000
nValid = 1000
nTest = 1000
nEpochs = 100 # Number of epochs
batchSize = 5 # Batch size
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
dimNodeSignals =[1, nFeatureBank, nFeatureBank]
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
nSelectedNodes= [nFeatures, nFeatures] 
poolingSize=[nFeatures, nFeatures]

#Put all the vars in the codestuff
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

utils.make_fig(thisTrainVars, saveDir,nEpochs, validationInterval)

