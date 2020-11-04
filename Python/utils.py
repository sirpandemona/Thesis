# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 15:23:25 2020

@author: vascodebruijn
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import json
import random
import pickle
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
from  scipy.stats import pearsonr
from sklearn.decomposition import PCA
mask = np.array([0x03, 0x0c, 0x35, 0x3a, 0x50, 0x5f, 0x66, 0x69, 0x96, 0x99, 0xa0, 0xaf, 0xc5, 0xca, 0xf3, 0xfc])
hw = [bin(x).count("1") for x in range(256)] 

sbox = np.array([
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
])

def feature_reduction(method, k, X,Y, dataset,LM,N):
    """
    Applies feature reduction to given dataset, but first tries to see if a
    reduced dataset with the correct parameters does already exist
    method: method used for feature selection
    k: number of features after selection
    X: actual data
    Y: labels of the data
    LM: used linkage model
    N: number of datasamples
    dataset: name of the dataset
    """
    filename = str(k)+'-'+method+'-'+dataset+str(N)+'-'+LM+'.npy'
    
    scaler=MinMaxScaler()
    X = scaler.fit_transform(X)
    reduced_data = []
    if os.path.isfile(filename):
        reduced_data = np.load(filename,'r')
    else:
        if method == 'ChiSquare':
            data_model =  SelectKBest(chi2,k=k).fit(X, Y)
            indices=data_model.get_support(indices = True)
            reduced_data = X[:,indices]
        if method == 'Pearson':
            Y=np.reshape(Y,(N,1))
            app = np.append(X,Y,axis=1)
            corr_matrix= np.corrcoef(np.transpose(app))
            corrs = corr_matrix[len(corr_matrix)-1]
            sort_idx = np.argsort(corrs)
            top = sort_idx[-k-1:len(sort_idx)-1]
            reduced_data = X[:,top]
        if method == 'PCA':
            pca = PCA(n_components=k)
            pca.fit(X)
            reduced_data = pca.transform(X)
            
        np.save(filename,reduced_data)
    return reduced_data



def intermediate_value_dpa4(pt, keyguess, offset):
    return sbox[pt ^ keyguess] ^ mask[(offset+1)%16]

def intermediate_value(pt, keyguess, mask):
    return sbox[pt ^ keyguess] ^ mask


def evaluate_GE(predictions,plaintxts,offsets,keys,attack_size, model,dataset,n_shuffles,method):
    if method == 'Simple':
        return evaluate_GE_SP(predictions,plaintxts,offsets,keys,attack_size, model,dataset,n_shuffles)
    if method == 'General':
        return evaluate_GE_Gen(predictions,plaintxts,offsets,keys,attack_size, model,dataset,n_shuffles)

    
def evaluate_GE_Gen(predictions,plaintxts,offsets,keys,attack_size, model,dataset,n_shuffles):
    """
    Better GE calculation method
    predictions: C*N tensor of class predictions 
    plaintexts: N plaintexts corresponding to the predictions
    offsets: N masks/offsets corresponding to the predictions
    keys: N keys corresponding to the the predictions
    attack_size: number of traces used for attack
    model: leakage model, either IV(immediate value) or HW (Hamming Weight)
    """
    
    (test_size,nClasses) = predictions.shape
    ge_m = []
    for j in range(n_shuffles):
        ge_x = []
        pred = np.zeros(256)
        cand_ids = list(range(test_size))
        ids = random.choices(cand_ids, k=attack_size)
        random.shuffle(ids)    
        for idx in ids:
            for keyGuess in range(256):
                sbox_out = intermediate_value(plaintxts[idx], keyGuess,offsets[idx])
                if dataset == 'dpa4':
                    sbox_out = intermediate_value_dpa4(plaintxts[idx], keyGuess,offsets[idx])
                lv = sbox_out
                if model == "HW":
                    lv = hw[sbox_out]
                pred[keyGuess] += np.log(predictions[idx][lv]+ 1e-36)
        
            # Calculate key rank
            res = np.argmax(np.argsort(pred)[::-1] == keys[0]) #argsort sortira argumente od najmanjeg do najveceg, [::-1} okrece to, argmax vraca redni broj gdje se to desilo
            ge_x.append(res)
        ge_m.append(ge_x)    
        
    ge_m =np.array(ge_m)
    
    return np.mean(ge_m,axis=0)

def evaluate_GE_SP(predictions,plaintxts,offsets,keys,attack_size, model,dataset,n_shuffles):
    """
    Better GE calculation method
    predictions: C*N tensor of class predictions 
    plaintexts: N plaintexts corresponding to the predictions
    offsets: N masks/offsets corresponding to the predictions
    keys: N keys corresponding to the the predictions
    attack_size: number of traces used for attack
    model: leakage model, either IV(immediate value) or HW (Hamming Weight)
    """
    
    (test_size,nClasses) = predictions.shape
    ge_m = []
    for j in range(n_shuffles):
        ge_x = []
        pred = np.zeros(256)
        ids = list(range(attack_size))
        random.shuffle(ids)    
        for idx in ids:
            for keyGuess in range(256):
                sbox_out = intermediate_value(plaintxts[idx], keyGuess,offsets[idx])
                if dataset == 'dpa4':
                    sbox_out = intermediate_value_dpa4(plaintxts[idx], keyGuess,offsets[idx])
                lv = sbox_out
                if model == "HW":
                    lv = hw[sbox_out]
                pred[keyGuess] += np.log(predictions[idx][lv]+ 1e-36)
        
            # Calculate key rank
            res = np.argmax(np.argsort(pred)[::-1] == keys[0]) #argsort sortira argumente od najmanjeg do najveceg, [::-1} okrece to, argmax vraca redni broj gdje se to desilo
            ge_x.append(res)
        ge_m.append(ge_x)    
        
    ge_m =np.array(ge_m)
    
    return np.mean(ge_m,axis=0)

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
        
        for item in os.listdir(folder):
            if '.pkl' in item:
                path = os.path.join(folder,item)
                with (open(path,'rb')) as picklefile:
                    res = pickle.load(picklefile)
                    results[folder][item.replace('.pkl','')] = res
    return results

def reshape_results(data,F,L,K,efn=1):
    (N,ntraces) = data.shape
    total = F*L*K*efn
    if N<total:
        data = np.pad(data,[(0,total-N),(0,0)],mode='constant',constant_values=np.nan)
    res = np.reshape(data, (F,L,K,ntraces))
    return res

def generate_hyperparamlist(candidateF,candidateL,candidateK,candidateFn,dataset='dpa4',arch='ConvNet'):
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
                              'Used Architecture': arch,
                              'Edge Function': fn,
                              'EdgeFn Threshold': c,
                              'dataset': dataset,
                              "Size Dataset":10000
                              })
    return hyperparam_settings

def save_hyperparamlist(hplist, name):
    with open (name+'.json', 'w') as f:
        f.write(json.dumps(hplist))

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