# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 14:01:52 2020

@author: vascodebruijn
"""

import numpy as np
import networkx as nx
import math
from pyvis.network import Network

#Markov Based Methods

#Root function
def simple_markov(x):
    
    """
    Creates a Markov Chain Graph from a single trace
    Returns a tuple with the list of nodes and list of edges with weights corresponding to the frequency 
    x: Trace in np-format
    """
    
    top = np.max(x)
    bottom = np.min(x)
    vals = range(bottom,top+1)
    size = len(vals)+1
    markov = np.zeros((size, size))
    trans = {}
    g=nx.Graph()
    
    #Get all the transitions
    for i in range(0,len(x)-1):
        src = int(x[i])
        dst = int(x[i+1])
        markov[src-bottom][dst-bottom] += 1

        if (src,dst) in trans:
            trans[src,dst] += 1
        else:
            trans[src,dst] = 1
            
    #Convert the transition dictionary to a list of edges
    edges = trans.keys()
    wedges = []
    for src, dst in edges:
        w = trans[src,dst]
        #Note that w is the amount of occurrences, and you want the frequency
        wdict = {'weight': 1/w}
        wedges.append((src,dst,wdict))
        
    nodes = list(vals)
    nodes = [(i,{'val':i}) for i in nodes]
    g.add_nodes_from(nodes)        
    g.add_edges_from(wedges)  
    return g
    #return (nodes, wedges)


def ngram_markov(x,n):
    
    """
    Creates a Markov Chain Graph based on ngrams from a single trace
    x: Trace in np-format
    n: symbol length of ngram
    """
    top = np.max(x)
    bottom = np.min(x)
    vals = range(bottom,top+1)
    size = len(vals)+1
    
    #make ngram traces
    ngram_trace = []
    for i in range(len(x)-(n-1)):
        j = i+ (n)
        ngram = x[i:j]
        ngram_trace.append(ngram)
        
    #Get the transitions
    trans = {}    
    nodes = [(str(ngram_trace[0]),{'ngram':ngram_trace[0]})]
    for i in range(len(ngram_trace)-1):
        src = str(ngram_trace[i])
        dst = str(ngram_trace[i+1])
        if not dst in nodes:
            nodes.append((dst,{'ngram':ngram_trace[i+1]}))
            nodes.add(dst)
            
        if (src,dst) in trans:
            trans[src,dst] += 1
        else:
            trans[src,dst] = 1
            
    #Concert transitions to edges        
    edges = trans.keys()
    wedges = []
    for src, dst in edges:
        w = trans[src,dst]
        #Note that w is the amount of occurrences, and you want the frequency
        wdict = {'weight': 1/w}
        wedges.append((src,dst,wdict))
        
    
    g =nx.Graph()
    g.add_nodes_from(nodes)        
    g.add_edges_from(wedges)  
        
    return g

#Derived functions
def bin_markov(x, bin_num=16, bin_range=range(256)):
    
    """
    Creates a Markov Chain from a single trace
    Slots the values in the given amount of bins
    x: Trace in np-format
    bin_num: Amount of bins
    bin_range: The range of values which the traces may contain
    """
    
    x_bin = bin_trace(x, bin_num,bin_range)
    g = simple_markov(x_bin)
    return g
    
def dst_markov(x):
    
    """    
    Creates a Markov Chain from a single trace
    Uses the distances between points as values
    x: Trace in np-format
    """
    
    x_dst = dst_trace(x)
    g = simple_markov(x_dst)
    return g
    
    
def dst_trace(x):
    
    """
    Gets the distances between each pair of subsequent points in the trace
    x: Trace in np-format
    """
    
    dst = []
    for i in range(0,len(x)-1):
        val_i = x[i]
        val_j = x[i+1]
        dst.append(val_i -val_j)
    return np.asarray(dst)


#Help functinons
def bin_trace(x, bin_num=16, bin_range=range(256)):  
    
    """
    Slots the values in trace to a given amount of bins
    x: Trace in np-format
    bin_num: Amount of bins
    bin_range: The range of values which the traces may contain
    """   
    
    #If no range is given, take the range from the tracevalues
    if bin_range is None:
        bin_range = range(min(x),max(x))
    
    top = max(bin_range)
    bottom = min(bin_range)
    bin_width = (top-bottom)/bin_num
    x_bin = np.zeros_like(x)
    for i in range(len(x)):
        val = x[i]
        x_bin[i] = math.floor((val-bottom)/bin_width)
    return x_bin       
    
def make_similarity_matrix(graphs):
    
    """
    
    Creates a similarity matrix for all graphs in given list
    graphs: list of nx-graphs
    
    """
    
    n_graphs = len(graphs)
    sim_mat = np.zeros((n_graphs, n_graphs))
    for i in range(0,n_graphs):
        for j in range(0, n_graphs):
            g_1 = graphs[i]
            g_2 = graphs[j]
            dst = nx.graph_edit_distance(g_1, g_2)
            sim_mat[i][j] = dst
    return sim_mat

def gen_pyvis_graph(g):
    
    """
    Generates pyvis_graph
    g:NetX graph to visualize
    """
    nodes = g.nodes
    edges = g.edges
    net = Network(directed=True, notebook=True)
    
    for n in nodes:
        net.add_node(n, label = n)
        
    for edge in edges:
        w= g.edges[edge]['weight']
        net.add_edge(*edge,width=min(1/w, 10))
    net.prep_notebook()
    return net    