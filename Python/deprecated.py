# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 16:30:47 2020

@author: vascodebruijn
"""

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

def evaluate_GE(self, yHat, y, tol = 1e-9):
    """
    evaluates the guessing entropy 
    yHat: key guessing vector 
    y: (Ground Truth) Classification vector
    """
    guessing_vector = torch.argsort(yHat, descending=True)
    y_t = torch.reshape(y, (-1,1))
    (_,pos) = torch.where(guessing_vector == y_t)
    GE = torch.mean(pos.float())
    return float(GE)


def process_y(y,ptxts,offsets,model, dataset = 'ascad'):
    res = []
    for i in range(len(y)):
        plaintxt = ptxts[i]
        offset = offsets[i]
        y_i = y[i]
        ykeys = np.zeros((256))
        for key in range(0,256):
            if dataset == 'dpa4':
                IV = sbox[plaintxt ^ key] ^ mask[(offset+1)%16]
            else:
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