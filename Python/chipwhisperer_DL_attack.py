# -*- coding: utf-8 -*-


import numpy as np

from keras.models import Sequential
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path+"\\Python")
import import_traces

'''
Helpful tools:
- Bytearray to hex: import binascii; binascii.hexlify(bytearray(plain[0]))
- Integer to hex: hex(43)
- Hex to integer: 0xDC
'''

# NN architectures
def small_MLP(num_features, num_classes):
    # Small MLP: 2 deep layers of 250 neurons, then output
    model = Sequential()
    
    model.add(Dense(250, activation='relu', input_shape=(num_features, )))
    model.add(Dense(250, activation='relu'))
    model.add(Dense(100, activation='selu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    return model

def small_CNN(num_features, num_classes):
    # Small CNN
    model = Sequential()
    
    # Convolutional layer with 8 filters of size 3x1, step size 1
    model.add(Conv1D(filters=8, kernel_size=3, strides=1, activation='relu', input_shape=(num_features, 1)))
    
    # Maximum pooling (optional) to reduce the number of parameters and accelerate training
    model.add(MaxPooling1D(pool_size=3))
    

    # Flatten the output, so it can be inputted to the final layer
    model.add(Flatten())            
    model.add(Dense(num_classes, activation='softmax'))
    
    return model

# Sbox
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
mask = np.array([0x03, 0x0c, 0x35, 0x3a, 0x50, 0x5f, 0x66, 0x69, 0x96, 0x99, 0xa0, 0xaf, 0xc5, 0xca, 0xf3, 0xfc])

def intermediate_value(pt, keyguess, offset):
    return sbox[pt ^ keyguess] ^ mask[(offset+1)%16]




# Set training, validation and attack size
dataset_size        = 10000
training_size       = 4500
validation_size     = 500
attack_size         = 250

subkey              = 0    # which byte of key to attack (should be in interval [0, 15])
num_classes         = 256  # intermediate value model
num_epochs          = 10   # number of epochs for training the NN
batch_size_local    = 128  # batch size for training the NN



# Load data
#trace_array = np.load(r'D:\chipwhisperer_data\data\traces.npy')[0:dataset_size]
(trace_array,key,plain,masks) = import_traces.import_traces(False, 'dpa4', False, dataset_size)
numtraces = np.shape(trace_array)[1] #number of measurements per encryption

#key = np.load(r'D:\chipwhisperer_data\data\key.npy')[0:dataset_size]
#plain = np.load(r'D:\chipwhisperer_data\data\plain.npy')[0:dataset_size]



# Compute labels. We use the intermediate value model of sbox outputs
labels = np.ndarray(dataset_size)
for i in range(dataset_size):
    labels[i] = intermediate_value(plain[i], key[i],masks[i])
    


# Split up in train, validation and test set
assert training_size + validation_size + attack_size <= dataset_size
# Measurements
x_train         = trace_array[0:training_size]
x_validation    = trace_array[training_size:training_size+validation_size]
x_attack        = trace_array[training_size+validation_size:training_size+validation_size+attack_size]

# Scale measurements, to get mean 0 and variance 1
# To avoid bias for conducting a 'real' attack, fit only on training data
scaler = StandardScaler()
x_train         = scaler.fit_transform(x_train)                # If training_size is small, this will not be a good fit!
x_validation    = scaler.transform(x_validation)
x_attack        = scaler.transform(x_attack)

# Labels, using hot encoding
y = to_categorical(labels, num_classes)
y_train         = y[0:training_size]
y_validation    = y[training_size:training_size+validation_size]
y_attack        = y[training_size+validation_size:training_size+validation_size+attack_size]


# ===============================================================================
# When using a CNN, reshape the traces like this:
#x_train         = np.reshape(x_train, (training_size, numtraces, 1))
#x_validation    = np.reshape(x_validation, (validation_size, numtraces, 1))
#x_attack        = np.reshape(x_attack, (attack_size, numtraces, 1))
# Comment the lines above when using the MLP
# ===============================================================================

# Train model
model = small_MLP(numtraces, num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
print("Model compiled, summary of the layers:")
print(model.summary())
model.fit(x_train, y_train, batch_size=batch_size_local, validation_data=(x_validation, y_validation), epochs=num_epochs, verbose=1)

# Conduct attack
predictions = model.predict(x_attack)

# Compute guessing entropy
ge_x = []
pred = np.zeros(256)
for i in range(attack_size):
    for keyGuess in range(256):
        sbox_out = intermediate_value(plain[training_size + validation_size + i], keyGuess,masks[training_size + validation_size + i])
        pred[keyGuess] += np.log(predictions[i][sbox_out]+ 1e-36)

    # Calculate key rank
    res = np.argmax(np.argsort(pred)[::-1] == key[0]) #argsort sortira argumente od najmanjeg do najveceg, [::-1} okrece to, argmax vraca redni broj gdje se to desilo
    ge_x.append(res)



# Report
print("Attack completed")
#print('Real key, byte {}/16: {}. Entire key: {}'.format(subkey+1, key[0][subkey], key[0]))
key_guess = np.argmax(pred)

wrong_guess_reset = True
consistently_correct_since = np.inf
for i in range(attack_size):
    if wrong_guess_reset and ge_x[i] == 0:
        wrong_guess_reset = False
        consistently_correct_since = i
    elif ge_x[i] != 0:
        wrong_guess_reset = True
    
print('Guess of the key after {} attack traces: {}'.format(attack_size, key_guess))
if ge_x[attack_size-1] == 0:
    print('Key guess was consistently correct since {} attack traces.'.format(consistently_correct_since+1))

plt.grid(True)
plt.title('Guessing entropy')
plt.plot(ge_x)
plt.show() 
