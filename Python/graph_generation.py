# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 12:18:43 2020

@author: vascodebruijn
"""

import numpy as np
import scipy
import sys
import os
import torch
import pickle
import random
from itertools import zip_longest

cluster = os.getcwd() != 'C:\\Users\\vascodebruijn\\Documents\\GitHub\\Thesis\\Python'
module_path = os.path.abspath(os.path.join('..'))
if cluster:
    sys.path.insert(1, '\\home\\nfs\\vascodebruijn\\graph-neural-networks-networks')
else:    
    sys.path.insert(1, 'C:\\Users\\vascodebruijn\\Documents\\GitHub\\graph-neural-networks')

import import_traces

def generate_graph(x,edge_fn):
    """
    Generates graph used for graph signal classification
    Params:
        x: Traces in np-format (num_traces*num_features)
        edge_fn: method which is used to generate the edges\adj. matrix
    Returns:
        A: adjacency matrix representing the edges
        V: number of vertixes (A=V*V)  
    """
    
    (_,V) = x.shape
    A = edge_fn(x)
    
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

def get_edge_fn(fn_name,threshold):
    """
    gets the actual python function for the edge generation method
    name: name of the edge generation method
    """
    seq_fn = seq_connection
    corr_thresh_fn = lambda x: corr_tresh_connection(x,threshold)
    corr_knn_fn = lambda x: corr_tresh_connection(x,threshold)
    
    edge_dic = {
        "Successive": seq_fn,
        "Threshold Correlation" : corr_thresh_fn,
        "KNN Correlation": corr_knn_fn
        }
    return edge_dic[fn_name]
    
def evaluateGE(model, data, **kwargs):
    """
    evaluate: evaluate a model using classification error and guessing entropy
    
    Input:
        model (model class): class from Modules.model
        data (data class): a data class from the Utils.dataTools; it needs to
            have a getSamples method and an evaluate method.
        doPrint (optional, bool): if True prints results
    
    Output:
        evalVars (dict): 'errorBest' contains the error rate for the best
            model, 'errorLast' contains the error rate for the last model, 'GE' guessing entropy
    """

    # Get the device we're working on
    device = model.device
    
    if 'doSaveVars' in kwargs.keys():
        doSaveVars = kwargs['doSaveVars']
    else:
        doSaveVars = True

    plaintxts = kwargs['ptx']
    offsets = kwargs['offset']
    keys = kwargs['keys']
    ########
    # DATA #
    ########

    xTest, yTest = data.getSamples('test')
    xTest = xTest.to(device)
    yTest = yTest.to(device)
    
    ##############
    # BEST MODEL #
    ##############

    model.load(label = 'Best')

    with torch.no_grad():
        # Process the samples
        yHatTest = model.archit(xTest)
        key_probs = import_traces.process_y(yHatTest,plaintxts,offsets,'HW')
        # yHatTest is of shape
        #   testSize x numberOfClasses
        # We compute the error
        costBest = data.evaluate(yHatTest, yTest)
        #GE_best = data.evaluate_GE(yHatTest,yTest)
        GE_best = evaluate_traces(key_probs,keys,100,data)
    ##############
    # LAST MODEL #
    ##############

    model.load(label = 'Last')

    with torch.no_grad():
        # Process the samples
        yHatTest = model.archit(xTest)
        key_probs = import_traces.process_y(yHatTest,plaintxts,offsets,'HW')

        # yHatTest is of shape
        #   testSize x numberOfClasses
        # We compute the error
        costLast = data.evaluate(yHatTest, yTest)
        GE_last = evaluate_traces(key_probs,keys,100,data)

    evalVars = {}
    evalVars['costBest'] = costBest.item()
    evalVars['costLast'] = costLast.item()
    evalVars['GE_best'] = GE_best
    evalVars['GE_last'] = GE_last
    
    if doSaveVars:
        saveDirVars = os.path.join(model.saveDir, 'evalVars')
        if not os.path.exists(saveDirVars):
            os.makedirs(saveDirVars)
        pathToFile = os.path.join(saveDirVars, model.name + 'evalVars.pkl')
        with open(pathToFile, 'wb') as evalVarsFile:
            pickle.dump(evalVars, evalVarsFile)

    return evalVars    
         
def evaluate_traces(probs,y,n,data):
    """
    Calculates the guessing entropy for different amount of traces
    probs: probabilities of each key for each input
    y: array of correct keys
    n: max amount of traces to evaluate
    """
    results = []
    for i in range(1,n+1):
        (comb_probs, keys) = combine_traces(probs,y,i)
        GE = data.evaluate_GE(comb_probs,keys)
        results.append(GE)
    return results
       
def combine_traces(guesses, y, n):            
    """
    Calculate the key guessing vector over multimple samples
    probs: probabilities of each key for each input
    y: array of correct keys
    n: amount of traces
    """  
    
    grouped_props = {}
    guesses_t = torch.from_numpy(guesses).double()
    for i in range(len(y)):
        label = int(y[i])
        if not label in grouped_props:
            grouped_props[label] = []
        grouped_props[label].append(guesses_t[i])
    
    combined_probs=[]
    combined_y = []
    
    for label in grouped_props:
        random.shuffle(grouped_props[label])
        for prob_slice in grouper(n,grouped_props[label]):
            ps = list(prob_slice)
            ll = torch.sum(torch.log_softmax(torch.stack(ps),1),axis=0)
            combined_probs.append(ll)
            combined_y.append(label)
            
    return (torch.stack(combined_probs), torch.tensor(combined_y))    

def grouper(n, iterable, fillvalue=None):
    #helper function to group a set in subsets
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    tail = len(iterable) % n
    if tail > 0:
        iterable = iterable[:len(iterable)-tail]
     
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)