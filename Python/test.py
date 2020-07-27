# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 13:54:28 2020

@author: vascodebruijn
"""
cluster = False

import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True # Comment this line if no LaTeX installation is available
matplotlib.rcParams['font.family'] = 'serif' # Comment this line if no LaTeX installation is available
matplotlib.rcParams['text.latex.preamble']=[r'\usepackage{amsmath}']
import matplotlib.pyplot as plt
import sys
import os
import datetime

import torch; torch.set_default_dtype(torch.float64)
import torch.nn as nn
import torch.optim as optim

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


#init the training params
nTrain = 8000
nValid = 1000
nTest = 1000
nEpochs = 20 # Number of epochs
batchSize = 20 # Batch size
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
   
#Architecture params
nFeatureBank = 2
nClasses = 256
k = 3
writeVarValues(varsFile, {'nFeatureBank': nFeatureBank,
                          'nClasses' : nClasses,
                          'k' : k
                          })
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
(traces, keys) = import_traces.get_DPA_traces(cluster,feat_red=True, hw=False)
keys=keys.flatten()
G = gg.generate_graph(traces)
(A,V) = G

#create datamodel to use in GNN framework
data = gsd.signal_data(traces.copy(),keys.copy(),G,nTrain,nValid,nTest )
(nTraces,nFeatures) = traces.shape

#Moar architecture params
bias = False 
nonlinearity = nn.ReLU
dimNodeSignals =[1, nFeatureBank, nFeatureBank]
dimLayersMLP= [nClasses]
nShiftTaps =[k, k]
nFilterNodes = [40, 40]
GSO = A

writeVarValues(varsFile, {'bias': bias,
                          'nonlinearity' : nonlinearity,
                          'dimNodeSignals': dimNodeSignals,
                          'nShiftTaps': nShiftTaps,
                          'nFilterNodes': nFilterNodes,
                          'dimLayersMLP': dimLayersMLP,
                          })
#pooling stuff
nSelectedNodes= [nFeatures, nFeatures] 
poolingSize=[nFeatures, nFeatures]

#Put all the vars in the codestuff
EdgeNet = archit.EdgeVariantGNN(dimNodeSignals, nShiftTaps,nFilterNodes,bias,nonlinearity,nSelectedNodes,Utils.graphML.NoPool,poolingSize,dimLayersMLP, GSO)

EdgeNet.to(device)
thisOptim = optim.Adam(EdgeNet.parameters(), lr = learningRate, betas = (beta1,beta2))

EdgeNetGNN = model.Model(EdgeNet,
                     lossFunction(),
                     thisOptim,
                     trainer,
                     evaluator,
                     device,
                     "EdgeNet",
                     'test') 

#writer.add_graph(EdgeNet,data.samples['train']['signals'])
#writer.close()

print("Start Training")
thisTrainVars = EdgeNetGNN.train(data, nEpochs, batchSize, **trainingOptions)

lossTrain = thisTrainVars['lossTrain']
costTrain = thisTrainVars['costTrain']
lossValid = thisTrainVars['lossValid']
costValid = thisTrainVars['costValid']
evalVars = EdgeNetGNN.evaluate(data)

#\\\ FIGURES DIRECTORY:
saveDirFigs = os.path.join(saveDir,'figs')
# If it doesn't exist, create it.
if not os.path.exists(saveDirFigs):
    os.makedirs(saveDirFigs)
    
xAxisMultiplierTrain = 100 # How many training steps in between those shown in
    # the plot, i.e., one training step every xAxisMultiplierTrain is shown.
xAxisMultiplierValid = 1 # How many validation steps in between those shown,
    # same as above.
figSize = 5 # Overall size of the figure that contains the plot
lineWidth = 2 # Width of the plot lines
markerShape = 'o' # Shape of the markers
markerSize = 3 # Size of the markers    

nBatches = thisTrainVars['nBatches']

# Compute the x-axis
xTrain = np.arange(0, nEpochs * nBatches, xAxisMultiplierTrain)
xValid = np.arange(0, nEpochs * nBatches, \
                      validationInterval*xAxisMultiplierValid)

if xAxisMultiplierTrain > 1:
    # Actual selected samples
    selectSamplesTrain = xTrain
    # Go and fetch tem
    lossTrain = lossTrain[selectSamplesTrain]
    costTrain = costTrain[selectSamplesTrain]
# And same for the validation, if necessary.
if xAxisMultiplierValid > 1:
    selectSamplesValid = np.arange(0, len(lossValid), \
                                   xAxisMultiplierValid)
    lossValid = lossValid[selectSamplesValid]
    costValid = costValid[selectSamplesValid]

costFig = plt.figure(figsize=(1.61*figSize, 1*figSize))
plt.plot(xTrain, costTrain,color='#01256E',linewidth = lineWidth,
             marker = markerShape, markersize = markerSize )
plt.plot(xValid,costValid,color = '#95001A',linewidth = lineWidth,
             marker = markerShape, markersize = markerSize)

plt.ylabel(r'Error rate')
plt.xlabel(r'Training steps')
plt.legend([r'Training', r'Validation'])
plt.title(r'Results')
costFig.savefig(os.path.join(saveDirFigs,'eval.pdf'),
                    bbox_inches = 'tight')