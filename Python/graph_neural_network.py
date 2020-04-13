# -*- coding: utf-8 -*-


import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import numpy as np
import random
from itertools import zip_longest

def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)
# Sends a message of node feature h.
msg = fn.copy_src(src='h', out='m')

def reduce(nodes):
    """Take an average over all neighbor node features hu and use it to
    overwrite the original node feature."""
    accum = torch.mean(nodes.mailbox['m'], 1)
    return {'h': accum}

class NodeApplyModule(nn.Module):
    """Update the node feature hv with ReLU(Whv+b)."""
    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, node):
        h = self.linear(node.data['h'])
        h = self.activation(h)
        return {'h' : h}

class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(GCN, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)

    def forward(self, g, feature):
        # Initialize the node features with h.
        g.ndata['h'] = feature
        g.update_all(msg, reduce)
        g.apply_nodes(func=self.apply_mod)
        return g.ndata.pop('h')

class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(Classifier, self).__init__()

        self.layers = nn.ModuleList([
            GCN(in_dim, hidden_dim, F.relu),
            GCN(hidden_dim, hidden_dim, F.relu)])
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g):
        # For undirected graphs, in_degree is the same as
        # out_degree.
        h = g.in_degrees().view(-1, 1).float()
        for conv in self.layers:
            h = conv(g, h)
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')
        return self.classify(hg)    



def calc_results(probs,y):
    """
    Calculate the Guessing Entropy and Success Rate for output of a given NN
    probs: probabilities of each key for each input
    y: array of correct keys
    """
    guess_v = np.argsort(probs)
    sr = 0
    ge = 0
    n = len(y)
    for i in range(n):
        v = guess_v[i]
        key = y[i]
        ge += np.where(v == key)[0]
        if v[0] == key:
            sr+=1
    return (ge/n, sr/n)
 
def grouper(n, iterable, fillvalue=None):
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    tail = len(iterable) % n
    if tail > 0:
        iterable = iterable[:len(iterable)-tail]
     
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)
       
def combine_probs(probs, y, n):  
          
    """
    Calculate the key guessing vector over multimple samples
    probs: probabilities of each key for each input
    y: array of correct keys
    n: amount of samples
    """  
    
    grouped_props = {}
    
    for i in range(len(y)):
        label = y[i][0]
        if not label in grouped_props:
            grouped_props[label] = []
        grouped_props[label].append(probs[i])
    
    combined_probs=[]
    combined_y = []
    
    for label in grouped_props:
        random.shuffle(grouped_props[label])
        for prob_slice in grouper(n,grouped_props[label]):
            ps = list(prob_slice)
            ll = np.sum(np.log(ps),axis=0)
            combined_probs.append(ll)
            combined_y.append(label)
            
    return (combined_probs, combined_y)
        
    
    
    
    