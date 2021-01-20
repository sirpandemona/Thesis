# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 12:18:43 2020

@author: vascodebruijn
"""

import numpy as np
import scipy
import scipy.special
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
import utils


def get_graph(dataset,nFeatures,edge_function,threshold,traces):
    """
    General function to retrieve graph, 
    or to generate graph is no existing graph is found
    """
    
    
    graph_name = 'A_' + dataset + '_'+str(nFeatures) +'_'+edge_function+'_'+str(threshold)+'.npy'
    edge_fn = get_edge_fn(edge_function,threshold)

    print(graph_name)
    if os.path.isfile(graph_name):
        A = np.load(graph_name,'r')
        G = (A,len(A))
    else:
        G = generate_graph(traces,edge_fn)    
        (A,_) = G
        np.save(graph_name,A)
    return G
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
    Connects all nodes corresponding to the cyclic graph
    Params:
        x: Traces in np-format (num_traces*num_features)
    Out:
        A: Adjacency matrix
    """
    (_,N) = x.shape
    A = np.zeros((N,N))
    A[N-1][0] = 1
    A[0][N-1] = 1
    for i in range(N-1):
        j= i+1
        A[i][j] = 1
        A[j][i] = 1
        
    return A

def random_graph(x,k):
    """
    Connects nodes by randomlygenerating k*n edges
    Params:
        x: Traces in np-format (num_traces*num_features)
        k: amount of neighbours
    Out:
        A: Adjacency matrix
    """
    (_,N) = x.shape
    A = np.zeros((N,N))
    
    M= N*k
    
    idx = np.random.randint(low=0, high=N, size=(M,2))
    
    for (i,j) in idx:
        A[i][j] = 1
        
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
    A = np.zeros((f,f))
    C = np.zeros((f,f))
    #put all correlations row-wise in a matrix
    for i in range (f):
        v_i= x[:,i]
        for j in range(i,f):
            v_j= x[:,j]
            (corr,_) = scipy.stats.pearsonr(v_i, v_j)
            C[i,j] = corr
            C[j,i] = corr
        #when a row is full, select the k highest values         
        c_i = C[i,:]
        sort_idx = np.argsort(c_i)
        top = sort_idx[-k:len(sort_idx)]
        A[i, top] =1
    return A  

def full_connected(x):
    """Generates a fully connected graph based on X"""
    (N,f) = x.shape
    return np.ones((f,f))     
    
def identity_graph(x):
    (N,f) = x.shape
    return np.identity(f)

def get_edge_fn(fn_name,threshold):
    """
    gets the actual python function for the edge generation method
    name: name of the edge generation method
    """
    seq_fn = seq_connection
    corr_thresh_fn = lambda x: corr_tresh_connection(x,threshold)
    corr_knn_fn = lambda x: corr_knn_connection(x,threshold)
    full_con_fn = full_connected
    rnd_fn = lambda x: random_graph(x,threshold)
    id_fn = identity_graph
    edge_dic = {
        "Successive": seq_fn,
        "Threshold Correlation" : corr_thresh_fn,
        "KNN Correlation": corr_knn_fn,
        "Fully Connected" : full_con_fn,
        "Random": rnd_fn,
        "Identity":  id_fn
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
    lm = kwargs['lm']
    dataset = kwargs['dataset']
    nRuns = kwargs['nRuns']
    
    attack_size = len(keys)
    GE_method = 'Simple'
    if 'attack_size' in kwargs.keys():
        attack_size = kwargs['attack_size']
        GE_method = 'General'
        
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
        #yHatTest_norm =scipy.special.softmax(yHatTest.numpy())
        # yHatTest is of shape
        #   testSize x numberOfClasses
        # We compute the error
        costBest = data.evaluate(yHatTest, yTest)
        #GE_best = data.evaluate_GE(yHatTest,yTest)
        #GE_best = evaluate_traces(key_probs,keys,500,data)
        #GE_best = utils.evaluate_GE(yHatTest_norm,plaintxts,offsets,keys,attack_size, lm,dataset,nRuns,GE_method)
    ##############
    # LAST MODEL #
    ##############
    evalVars = {}

    model.load(label = 'Last')

    with torch.no_grad():
        # Process the samples
        yHatTest = model.archit(xTest)
        yHatTest_norm =scipy.special.softmax(yHatTest.numpy())

        # yHatTest is of shape
        #   testSize x numberOfClasses
        # We compute the error
        costLast = data.evaluate(yHatTest, yTest)
        #GE_last = evaluate_traces(key_probs,keys,500,data)
        GE_last = utils.evaluate_GE(yHatTest_norm,plaintxts,offsets,keys,attack_size, lm,dataset,nRuns,GE_method)
    
    evalVars['Prediction'] = yHatTest_norm
    evalVars['costBest'] = costBest.item()
    evalVars['costLast'] = costLast.item()
    #evalVars['GE_best'] = GE_best
    evalVars['GE_last'] = GE_last
    
    if doSaveVars:
        saveDirVars = os.path.join(model.saveDir, 'evalVars')
        if not os.path.exists(saveDirVars):
            os.makedirs(saveDirVars)
        pathToFile = os.path.join(saveDirVars, model.name + 'evalVars.pkl')
        with open(pathToFile, 'wb') as evalVarsFile:
            pickle.dump(evalVars, evalVarsFile)

    return evalVars    
   

