# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 14:26:50 2020

@author: vascodebruijn
"""
import numpy as np
import scipy
import torch
import torch.nn as nn
import os
import sys

cluster = os.getcwd() != 'C:\\Users\\vascodebruijn\\Documents\\GitHub\\Thesis\\Python'
module_path = os.path.abspath(os.path.join('..'))
if cluster:
    sys.path.insert(1, '\\home\\nfs\\vascodebruijn\\graph-neural-networks-networks')
else:    
    sys.path.insert(1, 'C:\\Users\\vascodebruijn\\Documents\\GitHub\\graph-neural-networks')


import Utils.graphML as gml
import Utils.graphTools

from Utils.dataTools import changeDataType

zeroTolerance = 1e-9 # Absolute values below this number are considered zero.

class SelectionGNN_Lin(nn.Module):
    
    """
    SelectionGNN: implement the selection GNN architecture
    Initialization:
        SelectionGNN(dimNodeSignals, nFilterTaps, bias, # Graph Filtering
                     nonlinearity, # Nonlinearity
                     nSelectedNodes, poolingFunction, poolingSize, # Pooling
                     dimLayersMLP, # MLP in the end
                     GSO, order = None, # Structure
                     coarsening = False)
        Input:
            /** Graph convolutional layers **/
            dimNodeSignals (list of int): dimension of the signals at each layer
                (i.e. number of features at each node, or size of the vector
                 supported at each node)
            nFilterTaps (list of int): number of filter taps on each layer
                (i.e. nFilterTaps-1 is the extent of neighborhoods that are
                 reached, for example K=2 is info from the 1-hop neighbors)
            bias (bool): include bias after graph filter on every layer
            >> Obs.: dimNodeSignals[0] is the number of features (the dimension
                of the node signals) of the data, where dimNodeSignals[l] is the
                dimension obtained at the output of layer l, l=1,...,L.
                Therefore, for L layers, len(dimNodeSignals) = L+1. Slightly
                different, nFilterTaps[l] is the number of filter taps for the
                filters implemented at layer l+1, thus len(nFilterTaps) = L.
                
            /** Activation function **/
            nonlinearity (torch.nn): module from torch.nn non-linear activations
            
            /** Pooling **/
            nSelectedNodes (list of int): number of nodes to keep after pooling
                on each layer
            >> Obs.: The selected nodes are the first nSelectedNodes[l] starting
                from the first element in the order specified by the given GSO
            >> Obs.: If coarsening = True, this variable is ignored since the
                number of nodes in each layer is given by the graph coarsening
                algorithm.
            poolingFunction (nn.Module in Utils.graphML or in torch.nn): 
                summarizing function
            >> Obs.: If coarsening = True, then the pooling function is one of
                the regular 1-d pooling functions available in torch.nn (instead
                of one of the summarizing functions in Utils.graphML).
            poolingSize (list of int): size of the neighborhood to compute the
                summary from at each layer
            >> Obs.: If coarsening = True, then the pooling size is ignored 
                since, due to the binary tree nature of the graph coarsening
                algorithm, it always has to be 2.
                
            /** Readout layers **/
            dimLayersMLP (list of int): number of output hidden units of a
                sequence of fully connected layers after the graph filters have
                been applied
                
            /** Graph structure **/
            GSO (np.array): graph shift operator of choice.
            order (string or None, default = None): determine the criteria to
                use when reordering the nodes (i.e. for pooling reasons); the
                string has to be such that there is a function named 
                'perm' + order in Utils.graphTools that takes as input the GSO
                and returns a new GSO ordered by the specified criteria and
                an order array
            coarsening (bool, default = False): if True uses graph coarsening
                instead of zero-padding to reduce the number of nodes.
            >> Obs.: (i) Graph coarsening only works when the number
                 of edge features is 1 -scalar weights-. (ii) The graph
                 coarsening forces a given order of the nodes, and this order
                 has to be used to reordering the GSO as well as the samples
                 during training; as such, this order is internally saved and
                 applied to the incoming samples in the forward call -it is
                 thus advised to use the identity ordering in the model class
                 when using the coarsening method-.
        Output:
            nn.Module with a Selection GNN architecture with the above specified
            characteristics.
    Forward call:
        SelectionGNN(x)
        Input:
            x (torch.tensor): input data of shape
                batchSize x dimFeatures x numberNodes
        Output:
            y (torch.tensor): output data after being processed by the selection
                GNN; shape: batchSize x dimLayersMLP[-1]
                
    Other methods:
        
        .changeGSO(S, nSelectedNodes = [], poolingSize = []): takes as input a
        new graph shift operator S as a tensor of shape 
            (dimEdgeFeatures x) numberNodes x numberNodes
        Then, next time the SelectionGNN is run, it will run over the graph 
        with GSO S, instead of running over the original GSO S. This is 
        particularly useful when training on one graph, and testing on another
        one. The number of selected nodes and the pooling size will not change
        unless specifically consider those as input. Those lists need to have
        the same length as the number of layers. There is no need to define
        both, unless they change.
        >> Obs.: The number of nodes in the GSOs need not be the same, but
            unless we want to risk zero-padding beyond the original number
            of nodes (which just results in disconnected nodes), then we might
            want to update the nSelectedNodes and poolingSize accordingly, if
            the size of the new GSO is different.
            
        y, yGNN = .splitForward(x): gives the output of the entire GNN y,
        which is of shape batchSize x dimLayersMLP[-1], as well as the output
        of all the GNN layers (i.e. before the MLP layers), yGNN of shape
        batchSize x nSelectedNodes[-1] x dimFeatures[-1]. This can be used to
        isolate the effect of the graph convolutions from the effect of the
        readout layer.
    """

    def __init__(self,
                 # Graph filtering
                 dimNodeSignals, nFilterTaps, bias,
                 # Nonlinearity
                 nonlinearity,
                 # Pooling
                 nSelectedNodes, poolingFunction, poolingSize,
                 # MLP in the end
                 dimLayersMLP,
                 # Structure
                 GSO,
                 # Ordering
                 order = None,
                 # Coarsening
                 coarsening = False):
        # Initialize parent:
        super().__init__()
        # dimNodeSignals should be a list and of size 1 more than nFilter taps.
        assert len(dimNodeSignals) == len(nFilterTaps) + 1
        # nSelectedNodes should be a list of size nFilterTaps, since the number
        # of nodes in the first layer is always the size of the graph
        assert len(nSelectedNodes) == len(nFilterTaps)
        # poolingSize also has to be a list of the same size
        assert len(poolingSize) == len(nFilterTaps)
        # Check whether the GSO has features or not. After that, always handle
        # it as a matrix of dimension E x N x N.
        assert len(GSO.shape) == 2 or len(GSO.shape) == 3
        if len(GSO.shape) == 2:
            assert GSO.shape[0] == GSO.shape[1]
            GSO = GSO.reshape([1, GSO.shape[0], GSO.shape[1]]) # 1 x N x N
        else:
            assert GSO.shape[1] == GSO.shape[2] # E x N x N
        # Store the values (using the notation in the paper):
        self.L = len(nFilterTaps) # Number of graph filtering layers
        self.F = dimNodeSignals # Features
        self.K = nFilterTaps # Filter taps
        self.E = GSO.shape[0] # Number of edge features
        if order is not None:
            # If there's going to be reordering, then the value of the
            # permutation function will be given by the criteria in 
            # self.reorder. For instance, if self.reorder = 'Degree', then
            # we end up calling the function Utils.graphTools.permDegree.
            # We need to be sure that the function 'perm' + self.reorder
            # is available in the Utils.graphTools module.
            self.permFunction = eval('Utils.graphTools.perm' + order)
        else:
            self.permFunction = Utils.graphTools.permIdentity
            # This is overriden if coarsening is selected, since the ordering
            # function is native to that pooling method.
     
        self.coarsening = coarsening # Whether to do coarsening or not
        # If we have to do coarsening, then note that it can only be done if
        # we have a single edge feature, otherwise, each edge feature could be
        # coarsed (and thus, ordered) in a different way, and there is no
        # sensible way of merging back this different orderings. So, we will
        # only do coarsening if we have a single edge feature; otherwise, we
        # will default to selection sampling (therefore, always specify
        # nSelectedNodes)
        if self.coarsening and self.E == 1:
            self.permFunction = Utils.graphTools.permCoarsening # Override
                # permutation function for the one corresponding to coarsening
            GSO = scipy.sparse.csr_matrix(GSO[0])
            GSO, self.order = Utils.graphTools.coarsen(GSO, levels=self.L,
                                                       self_connections=False)
            # Now, GSO is a list of csr_matrix with self.L+1 coarsened GSOs,
            # we need to torch.tensor them and put them in a list.
            # order is just a list of indices to reorder the nodes.
            self.S = []
            self.N = [] # It has to be reset, because now the number of
                # nodes is determined by the coarsening scheme
            for S in GSO:
                S = S.todense().A.reshape([self.E, S.shape[0], S.shape[1]])
                    # So, S.todense() returns a numpy.matrix object; a numpy
                    # matrix cannot be converted into a tensor (i.e., added
                    # the third dimension), therefore we need to convert it to
                    # a numpy.array. According to the documentation, the 
                    # attribute .A in a numpy.matrix returns self as an ndarray
                    # object. So that's why the .A is there.
                self.S.append(torch.tensor(S))
                self.N.append(S.shape[1])
            # Finally, because the graph coarsening algorithm is a binary tree
            # pooling, we always need to force a pooling size of 2
            self.alpha = [2] * self.L
        else:
            # Call the corresponding ordering function. Recall that if no
            # order was selected, then this is permIdentity, so that nothing
            # changes.
            self.S, self.order = self.permFunction(GSO)
            if 'torch' not in repr(self.S.dtype):
                self.S = torch.tensor(self.S)
            self.N = [GSO.shape[1]] + nSelectedNodes # Number of nodes
            self.alpha = poolingSize
            self.coarsening = False # If it failed because there are more than
                # one edge feature, then just set this to false, so we do not
                # need to keep checking whether self.E == 1 or not, just this
                # one
        # See that we adding N_{0} = N as the number of nodes input the first
        # layer: this above is the list containing how many nodes are between
        # each layer.
        self.bias = bias # Boolean
        # Store the rest of the variables
        self.sigma = nonlinearity
        self.rho = poolingFunction
        self.dimLayersMLP = dimLayersMLP
        # And now, we're finally ready to create the architecture:
        #\\\ Graph filtering layers \\\
        # OBS.: We could join this for with the one before, but we keep separate
        # for clarity of code.
        gfl = [] # Graph Filtering Layers
        for l in range(self.L):
            #\\ Graph filtering stage:
            gfl.append(gml.GraphFilter(self.F[l], self.F[l+1], self.K[l],
                                              self.E, self.bias))
            # There is a 3*l below here, because we have three elements per
            # layer: graph filter, nonlinearity and pooling, so after each layer
            # we're actually adding elements to the (sequential) list.
            if self.coarsening:
                gfl[2*l].addGSO(self.S[l])
            else:
                gfl[2*l].addGSO(self.S)
            #\\ Nonlinearity
            #lol, no nonlinearity
            #gfl.append(self.sigma())
            #\\ Pooling
            if self.coarsening:
                gfl.append(self.rho(self.alpha[l]))
            else:
                gfl.append(self.rho(self.N[l], self.N[l+1], self.alpha[l]))
                # Same as before, this is 3*l+2
                gfl[2*l+1].addGSO(self.S)
        # And now feed them into the sequential
        self.GFL = nn.Sequential(*gfl) # Graph Filtering Layers
        #\\\ MLP (Fully Connected Layers) \\\
        fc = []
        if len(self.dimLayersMLP) > 0: # Maybe we don't want to MLP anything
            # The first layer has to connect whatever was left of the graph
            # signal, flattened.
            dimInputMLP = self.N[-1] * self.F[-1]
            # (i.e., we have N[-1] nodes left, each one described by F[-1]
            # features which means this will be flattened into a vector of size
            # N[-1]*F[-1])
            fc.append(nn.Linear(dimInputMLP, dimLayersMLP[0], bias = self.bias))
            # The last linear layer cannot be followed by nonlinearity, because
            # usually, this nonlinearity depends on the loss function (for
            # instance, if we have a classification problem, this nonlinearity
            # is already handled by the cross entropy loss or we add a softmax.)
            for l in range(len(dimLayersMLP)-1):
                # Add the nonlinearity because there's another linear layer
                # coming
                #fc.append(self.sigma())
                # And add the linear layer
                fc.append(nn.Linear(dimLayersMLP[l], dimLayersMLP[l+1],
                                    bias = self.bias))
        # And we're done
        self.MLP = nn.Sequential(*fc)
        # so we finally have the architecture.
        
    def changeGSO(self, GSO, nSelectedNodes = [], poolingSize = []):
        
        # We use this to change the GSO, using the same graph filters.
        
        # Check that the new GSO has the correct
        assert len(GSO.shape) == 2 or len(GSO.shape) == 3
        if len(GSO.shape) == 2:
            assert GSO.shape[0] == GSO.shape[1]
            GSO = GSO.reshape([1, GSO.shape[0], GSO.shape[1]]) # 1 x N x N
        else:
            assert GSO.shape[1] == GSO.shape[2] # E x N x N
        
        # Get dataType and device of the current GSO, so when we replace it, it
        # is still located in the same type and the same device.
        dataType = self.S.dtype
        if 'device' in dir(self.S):
            device = self.S.device
        else:
            device = None
            
        # Now, if we don't have coarsening, then we need to reorder the GSO,
        # and since this GSO reordering will affect several parts of the non
        # coarsening algorithm, then we will do it now
        # Reorder the GSO
        if not self.coarsening:
            self.S, self.order = self.permFunction(GSO)
            # Change data type and device as required
            self.S = changeDataType(self.S, dataType)
            if device is not None:
                self.S = self.S.to(device)
            
        # Before making decisions, check if there is a new poolingSize list
        if len(poolingSize) > 0 and not self.coarsening:
            # (If it's coarsening, then the pooling size cannot change)
            # Check it has the right length
            assert len(poolingSize) == self.L
            # And update it
            self.alpha = poolingSize
        
        # Now, check if we have a new list of nodes (this only makes sense
        # if there is no coarsening, because if it is coarsening, the list with
        # the number of nodes to be considered is ignored.)
        if len(nSelectedNodes) > 0 and not self.coarsening:
            # If we do, then we need to change the pooling functions to select
            # less nodes. This would allow to use graphs of different size.
            # Note that the pooling function, there is nothing learnable, so
            # they can easily be re-made, re-initialized.
            # The first thing we need to check, is that the length of the
            # number of nodes is equal to the number of layers (this list 
            # indicates the number of nodes selected at the output of each
            # layer)
            assert len(nSelectedNodes) == self.L
            # Then, update the N that we have stored
            self.N = [GSO.shape[1]] + nSelectedNodes
            # And get the new pooling functions
            for l in range(self.L):
                # For each layer, add the pooling function
                self.GFL[2*l+2] = self.rho(self.N[l], self.N[l+1],
                                           self.alpha[l])
                self.GFL[2*l+2].addGSO(self.S)
        elif len(nSelectedNodes) == 0 and not self.coarsening:
            # Just update the GSO
            for l in range(self.L):
                self.GFL[2*l+2].addGSO(self.S)
        
        # If it's coarsening, then we need to compute the new coarsening
        # scheme
        if self.coarsening and self.E == 1:
            device = self.S[0].device
            GSO = scipy.sparse.csr_matrix(GSO[0])
            GSO, self.order = Utils.graphTools.coarsen(GSO, levels=self.L,
                                                       self_connections=False)
            # Now, GSO is a list of csr_matrix with self.L+1 coarsened GSOs,
            # we need to torch.tensor them and put them in a list.
            # order is just a list of indices to reorder the nodes.
            self.S = []
            self.N = [] # It has to be reset, because now the number of
                # nodes is determined by the coarsening scheme
            for S in GSO:
                S = S.todense().A.reshape([self.E, S.shape[0], S.shape[1]])
                    # So, S.todense() returns a numpy.matrix object; a numpy
                    # matrix cannot be converted into a tensor (i.e., added
                    # the third dimension), therefore we need to convert it to
                    # a numpy.array. According to the documentation, the 
                    # attribute .A in a numpy.matrix returns self as an ndarray
                    # object. So that's why the .A is there.
                self.S.append(torch.tensor(S).to(device))
                self.N.append(S.shape[1])
            # And we need to update the GSO in all the places.
            #   Note that we do not need to change the pooling function, because
            #   it is the standard pooling function that doesn't care about the
            #   number of nodes: it still takes one every two of them.
            for l in range(self.L):
                self.GFL[2*l].addGSO(self.S[l]) # Graph convolutional layer
        else:
            # And update in the LSIGF that is still missing (recall that the
            # ordering for the non-coarsening case has already been done)
            for l in range(self.L):
                self.GFL[2*l].addGSO(self.S) # Graph convolutional layer

    def splitForward(self, x):
        
        # Reorder the nodes from the data
        # If we have added dummy nodes (which, has to happen when the size
        # is different and we chose coarsening), then we need to use the
        # provided permCoarsening function (which acts on data to add dummy
        # variables)
        if x.shape[2] != self.N[0] and self.coarsening:
            thisDevice = x.device # Save the device we where operating on
            x = x.cpu().numpy() # Convert to numpy
            x = Utils.graphTools.permCoarsening(x, self.order) 
                # Re order and add dummy values
            x = torch.tensor(x).to(thisDevice)
        else:
        # If not, simply reorder the nodes
            x = x[:, :, self.order]

        # Now we compute the forward call
        assert len(x.shape) == 3
        batchSize = x.shape[0]
        assert x.shape[1] == self.F[0]
        assert x.shape[2] == self.N[0]
        # Let's call the graph filtering layer
        y = self.GFL(x)
        # Flatten the output
        yFlat = y.reshape(batchSize, self.F[-1] * self.N[-1])
        # And, feed it into the MLP
        return self.MLP(yFlat), y
        # If self.MLP is a sequential on an empty list it just does nothing.
    
    def forward(self, x):
        
        # Most of the times, we just need the actual, last output. But, since in
        # this case, we also want to compare with the output of the GNN itself,
        # we need to create this other forward funciton that takes both outputs
        # (the GNN and the MLP) and returns only the MLP output in the proper
        # forward function.
        output, _ = self.splitForward(x)
        
        return output

    def to(self, device):
        # Because only the filter taps and the weights are registered as
        # parameters, when we do a .to(device) operation it does not move the
        # GSOs. So we need to move them ourselves.
        # Call the parent .to() method (to move the registered parameters)
        super().to(device)
        # Move the GSO
        if self.coarsening:
            for l in range(self.L):
                self.S[l] = self.S[l].to(device)
                self.GFL[2*l].addGSO(self.S[l])
        else:
            self.S = self.S.to(device)
            # And all the other variables derived from it.
            for l in range(self.L):
                self.GFL[2*l].addGSO(self.S)
                self.GFL[2*l+1].addGSO(self.S)

class GraphConvolutionAttentionNetwork_Lin(nn.Module):
    """
    GraphConvolutionAttentionNetwork: implement the graph convolution attention
        network (GCAT) architecture

    Initialization:

        GraphConvolutionAttentionNetwork(dimNodeSignals, 
                                         nFilterTaps,
                                         nAttentionHeads,
                                         bias, # Graph Filtering
                                         nonlinearity, # Nonlinearity
                                         nSelectedNodes,
                                         poolingFunction, 
                                         poolingSize,
                                         dimLayersMLP, # MLP in the end
                                         GSO, order = None) # Structure

        Input:
            /** Graph attention convolutional layers **/
            dimNodeSignals (list of int): dimension of the signals at each layer
            nFilterTaps (list of int): number of filter taps on each layer
            nAttentionHeads (list of int): number of attention heads on each
                layer
            bias (bool): include bias after the graph filter on each layer
            >> Obs.: dimNodeSignals[0] is the number of features (the dimension
                of the node signals) of the data, where dimNodeSignals[l] is the
                dimension obtained at the output of layer l, l=1,...,L.
                Therefore, for L layers, len(dimNodeSignals) = L+1. Slightly
                different, nAttentionHeads[l] is the number of filter taps for
                the filters implemented at layer l+1, thus
                len(nAttentionHeads) = L. Same for len(nFilterTaps) = L.
                
            /** Activation function **/
            nonlinearity (torch.nn.functional): function from module
                torch.nn.functional for non-linear activations
                
            /** Pooling **/
            nSelectedNodes (list of int): number of nodes to keep after pooling
                on each layer
            >> Obs.: The selected nodes are the first nSelectedNodes[l] starting
                from the first element in the order specified by the given GSO
            poolingFunction (nn.Module in Utils.graphML): summarizing function
            poolingSize (list of int): size of the neighborhood to compute the
                summary from at each layer
                
            /** Readout layer **/
            dimLayersMLP (list of int): number of output hidden units of a
                sequence of fully connected layers after the graph filters have
                been applied
                
            /** Graph structure **/
            GSO (np.array): graph shift operator of choice
            order (string or None, default = None): determine the criteria to
                use when reordering the nodes (i.e. for pooling reasons); the
                string has to be such that there is a function named 
                'perm' + order in Utils.graphTools that takes as input the GSO
                and returns a new GSO ordered by the specified criteria and
                an order array

        Output:
            nn.Module with a Graph Convolutional Attention Network architecture
            with the above specified characteristics.

    Forward call:

        GraphConvolutionAttentionNetwork(x)

        Input:
            x (torch.tensor): input data of shape
                batchSize x dimFeatures x numberNodes

        Output:
            y (torch.tensor): output data after being processed by the selection
                GNN; shape: batchSize x dimLayersMLP[-1]
    """

    def __init__(self,
                 # Graph attentional layer
                 dimNodeSignals, nFilterTaps, nAttentionHeads, bias,
                 # Nonlinearity (nn.functional)
                 nonlinearity,
                 # Pooling
                 nSelectedNodes, poolingFunction, poolingSize,
                 # MLP in the end
                 dimLayersMLP,
                 # Structure
                 GSO, order = None):
        # Initialize parent:
        super().__init__()
        # dimNodeSignals should be a list and of size 1 more than nFilterTaps
        # and nAttentionHeads
        assert len(dimNodeSignals) == len(nFilterTaps) + 1
        assert len(dimNodeSignals) == len(nAttentionHeads) + 1
        # nSelectedNodes should be a list of size nFilterTaps, since the number
        # of nodes in the first layer is always the size of the graph
        assert len(nSelectedNodes) == len(nAttentionHeads)
        # poolingSize also has to be a list of the same size
        assert len(poolingSize) == len(nAttentionHeads)
        # Check whether the GSO has features or not. After that, always handle
        # it as a matrix of dimension E x N x N.
        assert len(GSO.shape) == 2 or len(GSO.shape) == 3
        if len(GSO.shape) == 2:
            assert GSO.shape[0] == GSO.shape[1]
            GSO = GSO.reshape([1, GSO.shape[0], GSO.shape[1]]) # 1 x N x N
        else:
            assert GSO.shape[1] == GSO.shape[2] # E x N x N
        # Store the values (using the notation in the paper):
        self.L = len(nAttentionHeads) # Number of graph filtering layers
        self.F = dimNodeSignals # Features
        self.K = nFilterTaps # Number of filter taps
        self.P = nAttentionHeads # Attention Heads
        self.E = GSO.shape[0] # Number of edge features
        self.N = [GSO.shape[1]] + nSelectedNodes # Number of nodes
        # See that we adding N_{0} = N as the number of nodes input the first
        # layer: this above is the list containing how many nodes are between
        # each layer.
        if order is not None:
            # If there's going to be reordering, then the value of the
            # permutation function will be given by the criteria in 
            # self.reorder. For instance, if self.reorder = 'Degree', then
            # we end up calling the function Utils.graphTools.permDegree.
            # We need to be sure that the function 'perm' + self.reorder
            # is available in the Utils.graphTools module.
            self.permFunction = eval('Utils.graphTools.perm' + order)
        else:
            self.permFunction = Utils.graphTools.permIdentity
            # This is overriden if coarsening is selected, since the ordering
            # function is native to that pooling method.
        self.S, self.order = self.permFunction(GSO)
        if 'torch' not in repr(self.S.dtype):
            self.S = torch.tensor(self.S)
        self.sigma = nonlin # This has to be a nn.functional instead of
            # just a nn
        self.rho = poolingFunction
        self.alpha = poolingSize
        self.dimLayersMLP = dimLayersMLP
        self.bias = bias
        # And now, we're finally ready to create the architecture:
        #\\\ Graph Attentional Layers \\\
        # OBS.: The last layer has to have concatenate False, whereas the rest
        # have concatenate True. So we go all the way except for the last layer
        gat = [] # Graph Attentional Layers
        if self.L > 1:
            # First layer (this goes separate because there are not attention
            # heads increasing the number of features)
            #\\ Graph attention stage:
            gat.append(gml.GraphFilterAttentional(self.F[0],
                                                  self.F[1],
                                                  self.K[0],
                                                  self.P[0],
                                                  self.E,
                                                  self.bias,
                                                  self.sigma,
                                                  True))
            gat[0].addGSO(self.S)
            #\\ Pooling
            gat.append(self.rho(self.N[0], self.N[1], self.alpha[0]))
            gat[1].addGSO(self.S)
            # All the next layers (attention heads appear):
            for l in range(1, self.L-1):
                #\\ Graph attention stage:
                gat.append(gml.GraphFilterAttentional(self.F[l] * self.P[l-1],
                                                      self.F[l+1],
                                                      self.K[l],
                                                      self.P[l],
                                                      self.E,
                                                      self.bias,
                                                      self.sigma,
                                                      True))
                # There is a 2*l below here, because we have two elements per
                # layer: graph filter and pooling, so after each layer
                # we're actually adding elements to the (sequential) list.
                gat[2*l].addGSO(self.S)
                #\\ Pooling
                gat.append(self.rho(self.N[l], self.N[l+1], self.alpha[l]))
                # Same as before, this is 2*l+1
                gat[2*l+1].addGSO(self.S)
            # And the last layer (set concatenate to False):
            #\\ Graph attention stage:
            gat.append(gml.GraphFilterAttentional(self.F[self.L-1] \
                                                             * self.P[self.L-2],
                                                  self.F[self.L],
                                                  self.K[self.L-1],
                                                  self.P[self.L-1],
                                                  self.E,
                                                  self.bias,
                                                  self.sigma,
                                                  False))
            gat[2* (self.L - 1)].addGSO(self.S)
            #\\ Pooling
            gat.append(self.rho(self.N[self.L-1], self.N[self.L],
                                self.alpha[self.L-1]))
            gat[2* (self.L - 1) +1].addGSO(self.S)
        else:
            # If there's only one layer, it just go straightforward, adding a
            # False to the concatenation and no increase in the input features
            # due to attention heads
            gat.append(gml.GraphFilterAttentional(self.F[0],
                                                  self.F[1],
                                                  self.K[0],
                                                  self.P[0],
                                                  self.E,
                                                  self.bias,
                                                  self.sigma,
                                                  False))
            gat[0].addGSO(self.S)
            #\\ Pooling
            gat.append(self.rho(self.N[0], self.N[1], self.alpha[0]))
            gat[1].addGSO(self.S)
        # And now feed them into the sequential
        self.GCAT = nn.Sequential(*gat) # Graph Attentional Layers
        #\\\ MLP (Fully Connected Layers) \\\
        fc = []
        if len(self.dimLayersMLP) > 0: # Maybe we don't want to MLP anything
            # The first layer has to connect whatever was left of the graph
            # signal, flattened.
            # NOTE: Because sigma is a functional, instead of the layer, then
            # we need to pick up the layer for the MLP part.
            if str(self.sigma).find('relu') >= 0:
                self.sigmaMLP = nn.ReLU()
            elif str(self.sigma).find('tanh') >= 0:
                self.sigmaMLP = nn.Tanh()
                
            dimInputMLP = self.N[-1] * self.F[-1]
            # (i.e., we have N[-1] nodes left, each one described by F[-1]
            # features which means this will be flattened into a vector of size
            # N[-1]*F[-1])
            fc.append(nn.Linear(dimInputMLP, dimLayersMLP[0], bias = self.bias))
            # The last linear layer cannot be followed by nonlinearity, because
            # usually, this nonlinearity depends on the loss function (for
            # instance, if we have a classification problem, this nonlinearity
            # is already handled by the cross entropy loss or we add a softmax.)
            for l in range(len(dimLayersMLP)-1):
                # Add the nonlinearity because there's another linear layer
                # coming
                fc.append(self.sigmaMLP())
                # And add the linear layer
                fc.append(nn.Linear(dimLayersMLP[l], dimLayersMLP[l+1],
                                    bias = self.bias))
        # And we're done
        self.MLP = nn.Sequential(*fc)
        # so we finally have the architecture.

    def forward(self, x):
        # Now we compute the forward call
        assert len(x.shape) == 3
        batchSize = x.shape[0]
        assert x.shape[1] == self.F[0]
        assert x.shape[2] == self.N[0]
        # Reorder
        x = x[:, :, self.order] # B x F x N
        # Let's call the graph attentional layers
        y = self.GCAT(x)
        # Flatten the output
        y = y.reshape(batchSize, self.F[-1] * self.N[-1])
        # And, feed it into the MLP
        return self.MLP(y)
        # If self.MLP is a sequential on an empty list it just does nothing.

    def to(self, device):
        # Because only the filter taps and the weights are registered as
        # parameters, when we do a .to(device) operation it does not move the
        # GSOs. So we need to move them ourselves.
        # Call the parent .to() method (to move the registered parameters)
        super().to(device)
        # Move the GSO
        self.S = self.S.to(device)
        # And all the other variables derived from it.
        for l in range(self.L):
            self.GCAT[2*l].addGSO(self.S)
            self.GCAT[2*l+1].addGSO(self.S)

class EdgeVariantGNN_Lin(nn.Module):
    """
    EdgeVariantGNN: implement the selection GNN architecture using edge variant
        graph filters (through masking, not placement)

    Initialization:

        EdgeVariantGNN(dimNodeSignals, nShiftTaps, nFilterNodes, bias,
                       nonlinearity, # Nonlinearity
                       nSelectedNodes, poolingFunction, poolingSize,
                       dimLayersMLP, # MLP in the end
                       GSO, order = None) # Structure

        Input:
            /** Graph filtering layers **/
            dimNodeSignals (list of int): dimension of the signals at each layer
                (i.e. number of features at each node, or size of the vector
                 supported at each node)
            nShiftTaps (list of int): number of shift taps on each layer
                (i.e. information is gathered from up to the (nShiftTaps-1)-hop
                 neighborhood)
            nFilterNodes (list of int): number of nodes selected for the EV part
                of the hybrid EV filtering (recall that the first ones in the
                given permutation of S are the nodes selected; if any element in
                nFilterNodes is equal to the number of nodes, then we have a
                full edge-variant filter, not an hybrid one)
            bias (bool): include bias after graph filter on every layer
            >> Obs.: dimNodeSignals[0] is the number of features (the dimension
                of the node signals) of the data, where dimNodeSignals[l] is the
                dimension obtained at the output of layer l, l=1,...,L.
                Therefore, for L layers, len(dimNodeSignals) = L+1. Slightly
                different, nShiftTaps[l] is the number of filter taps for the
                filters implemented at layer l+1, thus len(nShiftTaps) = L.
                
            /** Activation function **/
            nonlinearity (torch.nn): module from torch.nn non-linear activations
            
            /** Pooling **/
            nSelectedNodes (list of int): number of nodes to keep after pooling
                on each layer
            >> Obs.: The selected nodes are the first nSelectedNodes[l] starting
                from the first element in the order specified by the given GSO
            poolingFunction (nn.Module in Utils.graphML): summarizing function
            poolingSize (list of int): size of the neighborhood to compute the
                summary from at each layer
                
            /** Readout layer **/
            dimLayersMLP (list of int): number of output hidden units of a
                sequence of fully connected layers after the graph filters have
                been applied
                
            /** Graph structure **/
            GSO (np.array): graph shift operator of choice.
            order (string or None, default = None): determine the criteria to
                use when reordering the nodes (i.e. for pooling reasons); the
                string has to be such that there is a function named 
                'perm' + order in Utils.graphTools that takes as input the GSO
                and returns a new GSO ordered by the specified criteria and
                an order array

        Output:
            nn.Module with a Selection GNN architecture with the above specified
            characteristics, implementing edge-variant graph filters.

    Forward call:

        EdgeVariantGNN(x)

        Input:
            x (torch.tensor): input data of shape
                batchSize x dimFeatures x numberNodes

        Output:
            y (torch.tensor): output data after being processed by the selection
                GNN; shape: batchSize x dimLayersMLP[-1]
                
    Other methods:
            
        y, yGNN = .splitForward(x): gives the output of the entire GNN y,
        which is of shape batchSize x dimLayersMLP[-1], as well as the output
        of all the GNN layers (i.e. before the MLP layers), yGNN of shape
        batchSize x nSelectedNodes[-1] x dimFeatures[-1]. This can be used to
        isolate the effect of graph filtering from the effect of the readout
        layer.
    """

    def __init__(self,
                 # Graph filtering
                 dimNodeSignals, nShiftTaps, nFilterNodes, bias,
                 # Nonlinearity
                 nonlinearity,
                 # Pooling
                 nSelectedNodes, poolingFunction, poolingSize,
                 # MLP in the end
                 dimLayersMLP,
                 # Structure
                 GSO, order = None):
        # Initialize parent:
        super().__init__()
        # dimNodeSignals should be a list and of size 1 more than the number of
        # filter taps (because of the input number of features)
        assert len(dimNodeSignals) == len(nShiftTaps) + 1
        # Filter nodes is a list of int with the number of nodes to select for
        # the EV part at each layer; it should have the same length as the
        # number of filter taps
        assert len(nFilterNodes) == len(nShiftTaps)
        # nSelectedNodes should be a list of size nShiftTaps, since the number
        # of nodes in the first layer is always the size of the graph
        assert len(nSelectedNodes) == len(nShiftTaps)
        # poolingSize also has to be a list of the same size
        assert len(poolingSize) == len(nShiftTaps)
        # Check whether the GSO has features or not. After that, always handle
        # it as a matrix of dimension E x N x N.
        assert len(GSO.shape) == 2 or len(GSO.shape) == 3
        if len(GSO.shape) == 2:
            assert GSO.shape[0] == GSO.shape[1]
            GSO = GSO.reshape([1, GSO.shape[0], GSO.shape[1]]) # 1 x N x N
        else:
            assert GSO.shape[1] == GSO.shape[2] # E x N x N
        # Store the values (using the notation in the paper):
        self.L = len(nShiftTaps) # Number of graph filtering layers
        self.F = dimNodeSignals # Features
        self.K = nShiftTaps # Filter Shift taps
        self.M = nFilterNodes
        self.E = GSO.shape[0] # Number of edge features
        self.N = [GSO.shape[1]] + nSelectedNodes # Number of nodes
        # See that we adding N_{0} = N as the number of nodes input the first
        # layer: this above is the list containing how many nodes are between
        # each layer.
        self.bias = bias # Boolean
        if order is not None:
            # If there's going to be reordering, then the value of the
            # permutation function will be given by the criteria in 
            # self.reorder. For instance, if self.reorder = 'Degree', then
            # we end up calling the function Utils.graphTools.permDegree.
            # We need to be sure that the function 'perm' + self.reorder
            # is available in the Utils.graphTools module.
            self.permFunction = eval('Utils.graphTools.perm' + order)
        else:
            self.permFunction = Utils.graphTools.permIdentity
            # This is overriden if coarsening is selected, since the ordering
            # function is native to that pooling method.
        self.S, self.order = self.permFunction(GSO)
        if 'torch' not in repr(self.S.dtype):
            self.S = torch.tensor(self.S)
        self.rho = poolingFunction
        self.alpha = poolingSize
        self.dimLayersMLP = dimLayersMLP
        # And now, we're finally ready to create the architecture:
        #\\\ Graph filtering layers \\\
        # OBS.: We could join this for with the one before, but we keep separate
        # for clarity of code.
        evgfl = [] # Node Variant GF Layers
        for l in range(self.L):
            #\\ Graph filtering stage:
            evgfl.append(gml.EdgeVariantGF(self.F[l], self.F[l+1],
                                            self.K[l], self.M[l], self.N[0],
                                            self.E, self.bias))
            # There is a 3*l below here, because we have three elements per
            # layer: graph filter, nonlinearity and pooling, so after each layer
            # we're actually adding elements to the (sequential) list.
            evgfl[2*l].addGSO(self.S)
            #\\ Nonlinearity
            #\\ Pooling
            evgfl.append(self.rho(self.N[l], self.N[l+1], self.alpha[l]))
            # Same as before, this is 3*l+2
            evgfl[2*l+1].addGSO(self.S)
        # And now feed them into the sequential
        self.EVGFL = nn.Sequential(*evgfl) # Graph Filtering Layers
        #\\\ MLP (Fully Connected Layers) \\\
        fc = []
        if len(self.dimLayersMLP) > 0: # Maybe we don't want to MLP anything
            # The first layer has to connect whatever was left of the graph
            # signal, flattened.
            dimInputMLP = self.N[-1] * self.F[-1]
            # (i.e., we have N[-1] nodes left, each one described by F[-1]
            # features which means this will be flattened into a vector of size
            # N[-1]*F[-1])
            fc.append(nn.Linear(dimInputMLP, dimLayersMLP[0], bias = self.bias))
            # The last linear layer cannot be followed by nonlinearity, because
            # usually, this nonlinearity depends on the loss function (for
            # instance, if we have a classification problem, this nonlinearity
            # is already handled by the cross entropy loss or we add a softmax.)
            for l in range(len(dimLayersMLP)-1):
                # Add the nonlinearity because there's another linear layer
                # coming
                # And add the linear layer
                fc.append(nn.Linear(dimLayersMLP[l], dimLayersMLP[l+1],
                                    bias = self.bias))
        # And we're done
        self.MLP = nn.Sequential(*fc)
        # so we finally have the architecture.
        
    def splitForward(self, x):
        # Now we compute the forward call
        assert len(x.shape) == 3
        batchSize = x.shape[0]
        assert x.shape[1] == self.F[0]
        assert x.shape[2] == self.N[0]
        # Reorder
        x = x[:, :, self.order] # B x F x N
        # Let's call the graph filtering layer
        y= self.EVGFL(x)
        # Flatten the output
        yFlat = y.reshape(batchSize, self.F[-1] * self.N[-1])
        # And, feed it into the MLP
        return self.MLP(yFlat), y
        # If self.MLP is a sequential on an empty list it just does nothing.
        
    def forward(self, x):
        
        # Most of the times, we just need the actual, last output. But, since in
        # this case, we also want to compare with the output of the GNN itself,
        # we need to create this other forward funciton that takes both outputs
        # (the GNN and the MLP) and returns only the MLP output in the proper
        # forward function.
        output, _ = self.splitForward(x)
        
        return output

    def to(self, device):
        # Because only the filter taps and the weights are registered as
        # parameters, when we do a .to(device) operation it does not move the
        # GSOs. So we need to move them ourselves.
        # Call the parent .to() method (to move the registered parameters)
        super().to(device)
        # Move the GSO
        self.S = self.S.to(device)
        # And all the other variables derived from it.
        for l in range(self.L):
            self.EVGFL[2*l].addGSO(self.S)
            self.EVGFL[2*l+1].addGSO(self.S)

def nonlin(x):
    return x