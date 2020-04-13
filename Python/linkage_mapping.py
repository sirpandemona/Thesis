# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 14:18:38 2020

@author: vascodebruijn
"""
import numpy as np
import networkx as nx

#Linkage based Methods

def simple_linkage(x):
    """
    Creates a graph by connecting all subsequent measures in the trace
    x: Trace in np-format
    """
    
    nodes = []
    edges = []
    for i in range(len(x)):
        node_attr ={"lvl":x[i]}
        nodes.append((i, node_attr))
        edges.append((i,i+1,{'weight':1}))
    edges.pop()
    
    g =nx.Graph()
    g.add_nodes_from(nodes)        
    g.add_edges_from(edges)  
    return g
    