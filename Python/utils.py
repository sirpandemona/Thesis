# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 15:23:25 2020

@author: vascodebruijn
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import json
def make_fig(thisTrainVars, saveDir,nEpochs,  validationInterval):
    #\\\ FIGURES DIRECTORY:
    lossTrain = thisTrainVars['lossTrain']
    costTrain = thisTrainVars['costTrain']
    lossValid = thisTrainVars['lossValid']
    costValid = thisTrainVars['costValid']
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
    
def get_all_results():
    results = {}
    for folder in [x[0] for x in os.walk('experiments')]:
        results[folder]={}
        res_path = os.path.join(folder,'results.npy')
        hyper_param_path = os.path.join(folder,'hyper_params.list')
        if os.path.isfile(res_path):
            results[folder]['res'] = np.load(res_path)
            
        if os.path.isfile(hyper_param_path):
            with open(hyper_param_path,'r') as f:
                results[folder]['hyperparam'] = json.loads(f.read())
    return results

def generate_hyperparamlist(candidateF,candidateL,candidateK,candidateFn):
    hyperparam_settings = []
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
                              'dataset': 'dpa4',
                              "Size Dataset":10000
                              })
    return hyperparam_settings

def gen_hyperparam_scatterplot(res):
    params = res['hyperparam']
    data =res['res'][:,0]
    vals = []
    K = []
    F = []
    L = []
    for i in range(0,len(data)):
        hp=params[i]
        vals.append(data[i])
        K.append(hp['k'])
        L.append(hp['nLayers'])
        F.append(hp['F'])
    return (K,F,L,vals)

def map_hws(y):
    hw = [bin(x).count("1") for x in range(256)]
    yhw = []
    for k in y:
        yhw.append(hw[int(k)])
    return np.asarray(yhw)