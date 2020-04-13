# -*- coding: utf-8 -*-
#!/bin/env python


import sys
import os
sys.path.append(os.getcwd()) 

import import_traces
import markov_mapping
import linkage_mapping
import graph_neural_network as gnn
import dgl
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch

#test pipeline
print("Start!")


(traces_aes, keys_aes) = import_traces.get_aes_hd()

#Begin with 2-classes, so only get items with 8 keys
#labels = ['100','110','111', '112', '113','114','115','116']
labels = ['100','110']
idxs = import_traces.get_mul_indexes(labels,keys_aes)
dataset = []

for j in range(len(labels)):
    ids = idxs[j]
   #y = labels[j]
    y = j
    for i in range(len(ids)):
        x = traces_aes.values[i]
        net = markov_mapping.bin_markov(x)
        g = dgl.DGLGraph()
        g.from_networkx(net)
        dataset.append((g,y)) 
        
#To start with, let the train and testset be identical
#We get to actual testing later
trainset = dataset
testset = dataset

#trainset = MiniGCDataset(320, 10, 20)
#testset = MiniGCDataset(80, 10, 20)
data_loader = DataLoader(trainset, batch_size=32, shuffle=True,
                         collate_fn=gnn.collate)

# Create model
model = gnn.Classifier(1, 256, len(labels))
#model = Classifier(1, 256, trainset.num_classes)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
model.train()

epoch_losses = []
for epoch in range(50):
    epoch_loss = 0
    for iter, (bg, label) in enumerate(data_loader):
        prediction = model(bg)
        loss = loss_func(prediction, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
    epoch_loss /= (iter + 1)
    print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
    epoch_losses.append(epoch_loss)
    
test_X, test_Y = map(list, zip(*testset))
test_bg = dgl.batch(test_X)
test_Y = torch.tensor(test_Y).float().view(-1, 1)
probs_Y = torch.softmax(model(test_bg), 1)
sampled_Y = torch.multinomial(probs_Y, 1)
argmax_Y = torch.max(probs_Y, 1)[1].view(-1, 1)

(p,y) = gnn.combine_probs(probs_Y.detach().numpy(),test_Y.numpy(),2)
(GE, SR) = gnn.calc_results(p,y)

print('Accuracy of sampled predictions on the test set: {:.4f}%'.format(
    (test_Y == sampled_Y.float()).sum().item() / len(test_Y) * 100))
print('Accuracy of argmax predictions on the test set: {:4f}%'.format(
    (test_Y == argmax_Y.float()).sum().item() / len(test_Y) * 100))

#f= open("/home/nfs/vascodebruijn/thesis/results.txt","w+")
#f.write("Guessing Entropy: %f\r\n"%GE)
#f.write("Success Rate %f\r\n"%SR)
#f.close()
