# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 13:25:13 2020

@author: vascodebruijn
"""

import pandas as pd
import scipy.io
import numpy as np

sbox=(
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16) 
hw = [bin(x).count("1") for x in range(256)] 
 
def import_traces(cluster, hw, feat_red,name,mask):
    """Overarching import function"""
    traces = []
    keys = []
    if name == "dpa4":
        (traces,keys) =  get_DPA_traces(cluster, hw,feat_red)
    if name == "aes_hd":
        (traces, keys) =  get_aes_hd()
    if name == 'delay':
        (traces,keys) =  get_delayed_traces()
    if name == 'ascad':
        (traces,keys) = get_ascad_traces(cluster,hw,feat_red)
    keys=keys.flatten()
    return (traces,keys)

def get_ascad_traces(cluster,hw,mask=False,data='train'):
    """
    Gets traces from the ASCAD dataset
    path: default folder
    hw: Whether the full key, or the HW model of the key should be loaded
    feat_red: Whether the full dataset should be used, or the one with feature reduction
    mask: whether the masked or unmasked dataset should be used
    """
    
    #use fixed key
    key = 34

    if cluster:
        path=r'/home/nfs/vascodebruijn/thesis/ASCAD_Keys'
    else:
        path=r'C:/Users/vascodebruijn/Thesis_Data/student-datasets/ASCAD_Keys'
        
    train_x, test_x = r'%s/traces/train_traces.npy'%path, r'%s/traces/test_traces.npy'%path
    traces_train = np.load(train_x)
    traces_test = np.load(test_x)
    keys_train=  np.full(len(traces_train),key)
    keys_test = np.full(len(traces_test),key)

    if data == 'train':
        return (traces_train, keys_train)
    else:
        return (traces_train, keys_train,traces_test,keys_test)

def get_ascad_info(cluster):
    if cluster:
        path = '/home/nfs/vascodebruijn/thesis/ASCAD_Keys/Value/test_plaintexts.npy'
    else:
        path=r'C:/Users/vascodebruijn/Thesis_Data/student-datasets/ASCAD_Keys/Value/test_plaintexts.npy'
        
    ptxts = np.load(path)
    offset = np.zeros(ptxts.shape)
    return (ptxts, offset)
    
def get_DPA_traces(cluster,hw,feat_red ):    
    """
    Gets traces from the DPA dataset
    path: default folder
    hw: Whether the full key, or the HW model of the key should be loaded
    feat_red: Whether the full dataset should be used, or the one with feature reduction
    """
    if cluster:
        path=r'/home/nfs/vascodebruijn/thesis/DPAv4'
    else:
        path=r'C:/Users/vascodebruijn/Thesis_Data/student-datasets\DPAv4'
        
    traces_path = r'%s/traces/traces_complete_10000.npy' % path
    fr_traces_path = r'%s/traces/traces_50_Value.npy'% path
    fr_hw_traces_path = r'%s/traces/traces_50_HW.npy'% path
    
    
    key = 108
    
    tp = traces_path
        
    if(feat_red):
        tp = fr_traces_path
        
    if(feat_red and hw):
        tp = fr_hw_traces_path
    
    traces = np.load(tp, mmap_mode='r')
    keys = np.full(len(traces),key)
        
    return (traces,keys)

def get_DPA_info(cluster):
    if cluster:
        path = '/home/nfs/vascodebruijn/thesis/dpav4_rsm_index.txt'
    else:
        path = 'C:\\Users\\vascodebruijn\\Documents\\GitHub\\Thesis\\Python\\dpav4_rsm_index.txt'
    file = open(path,'r')
    keys = []
    ptxts = []
    ctxts = []
    offsets = []
    for line in file.readlines():
        data = line.split()
        key = int(data[0][0:2],16)
        ptxt = int(data[1][0:2],16)
        ctxt = int(data[2][0:2],16)
        offset = int(data[3],16)
        
        keys.append(key)
        ptxts.append(ptxt)
        ctxts.append(ctxt)
        offsets.append(offset)
    
    return (np.asarray(keys),np.asarray(ptxts),np.asarray(ctxts),np.asarray(offsets))

def leakage_model(keys, ptxts, offsets, model):
    IVs =[]
    HWs = []
    for i in range(len(keys)):
        key = keys[i]
        plaintxt = ptxts[i]
        offset = offsets[i]
        
        IV = sbox[plaintxt ^ key] ^ offset
        HW = hw[IV]
        IVs.append(IV)
        HWs.append(HW)
    return (IVs, HWs)

def process_y(y,ptxts,offsets,model):
    res = []
    for i in range(len(y)):
        plaintxt = ptxts[i]
        offset = offsets[i]
        y_i = y[i]
        ykeys = np.zeros((256))
        for key in range(0,256):
            IV = sbox[plaintxt ^ key] ^ offset        
            HW = hw[IV]
            if model == 'IV':
                ykeys[key] = y_i[IV]
            if model == 'HW':
                ykeys[key] = y_i[HW]
        res.append(ykeys)
    return np.asarray(res)
      
def get_aes_hd(path = r'C:\Users\vascodebruijn\Documents\GitHub\AES_HD_Dataset\\'):
    
    """
    Gets traces from the AES dataset
    path: default folder
    """
    
    
    label_path = path+r'labels.csv'
    trace_path= path+r'traces_1.csv'
    labels = open(label_path, 'r')
    keys = labels.read().splitlines()
    traces_df = pd.read_csv(trace_path, delimiter=' ',header=None)
    return (traces_df, keys)

def get_delayed_traces(path = r'C:\Users\vascodebruijn\Documents\GitHub\randomdelays-traces\ctraces_fm16x4_2.mat'):
    
    """
    Gets traces from the random delays dataset
    path: default file
    """
    
    
    data = scipy.io.loadmat(path)
    plaintxt = data['plaintext']
    traces =data['CompressedTraces']
    return (traces, plaintxt)
    

def get_balancedTraces(info_lines, count):
    
    """
    Get a balanced set of traces from the DPA dataset without having to load all of them
    info_lines: Metadata file from the DPA dataset
    count: amount of traces of each class
    """

    sorted_hws = get_trace_hws(info_lines)
    ids = []   
    for i in range(9):
        for j in range(count):
            ids.append(sorted_hws[i][j])
    return ids

def get_trace_hws(info_lines) : 
    
    """
    Get all Hamming  Weights for the info file
    info_lines:  info file lines from the DPA dataset

    """
    
   
    sorted_hw = [[] for _ in range(9)]
    
    for i in range(len(info_lines)):
        data = info_lines[i].split()
        key = bytes.fromhex(data[0])
        plain = bytes.fromhex(data[1])
        cipher = bytes.fromhex(data[2])
        
        sbox_line = sbox[plain[0] ^ key[0]]
        hw_line = hw[sbox_line]
        sorted_hw[hw_line].append(i)                    
    return sorted_hw

def get_mul_indexes(vals, arr):
    
    """
    Searches for the indices of values in an array
    vals: array of values which to look for
    arr: array for which the indices has to be seached
    """
    idx_arr = []
    get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]
    for val in vals:
        idxs = get_indexes(val, arr)
        idx_arr.append(idxs)
    return idx_arr