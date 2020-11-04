import sys, os
from readTrc import Trc
import bz2
import matplotlib.pyplot as plt
import numpy as np
import zipfile
import glob
from tqdm import tqdm

central_pos = 272000 #center value of the cutted traces.
left = central_pos-2000
right = central_pos+2000
#The "methodology paper" traces are windowed from sample 270000 to 274000.
#The "make some noise paper" traces are probably more around samples 268000 to 272000 (in the paper says they use 4k samples but snr trace only have 3k samples..)

root_dir = os.getcwd() + '/'
compressed_dir = root_dir + 'compressed/'#contains compressed files from http://www.dpacontest.org/v4/ (DPA Contest v4.2 consists of 16 files).

filepath = root_dir + 'DPA_contestv4_2/'#destination of uncompressed files (default when unzip at root_dir)


l_files = ['DPA_contestv4_2_k00.zip']

##
for item in tqdm(l_files):#os.listdir(filepath)
    zipname = compressed_dir + item
    #try to decompress the zipfile, cancel if already exists
    if not(os.path.isdir(filepath + item[-7:-4])):
        print('Extracting file :'+item)
        bigzip = zipfile.ZipFile(zipname,'r')
        bigzip.extractall(root_dir)
        bigzip.close()

#try to allocate array in RAM
dpav4 = np.empty((5000,4000),dtype=np.float)

ind = 0
for item in tqdm(sorted(os.listdir(filepath))):#gives a list of working directories containing list of traces in .trc.bz2
    print(item)
    for bz2item in tqdm(sorted(glob.glob(filepath + item + '/*.trc.bz2'))):
        myzip = bz2.BZ2File(bz2item)
        trace = myzip.read()
        trcfile = bz2item[:-4]
        open(trcfile, 'wb').write(trace)
        trc=Trc()
        datX, datY, m = trc.open(trcfile)
        dpav4[ind] = datY[central_pos-2000:central_pos+2000]
        ind+=1

np.savez(root_dir + '{}_{}_DPAV4V2.npz'.format(left,right),dpav4)

