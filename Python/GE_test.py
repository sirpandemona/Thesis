# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 12:07:48 2020

@author: vascodebruijn
"""
import numpy as np
import scipy
import scipy.special
import matplotlib.pyplot as plt
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path+"\\Python")
import import_traces

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

test=[   0,    3,    8,   14,   23,   29,   31,   33,   35,   36,   39,
         47,   50,   76,   80,   88,   92,   96,  101,  103,  106,  107,
        108,  119,  131,  135,  144,  169,  245,  248,  251,  252,  259,
        263,  265,  267,  274,  286,  290,  291,  303,  304,  311,  315,
        316,  318,  321,  330,  333,  335,  346,  349,  360,  367,  376,
        381,  388,  410,  416,  439,  447,  450,  457,  461,  483,  487,
        500,  510,  518,  532,  543,  563,  567,  568,  576,  577,  582,
        586,  590,  592,  621,  623,  624,  637,  647,  655,  668,  673,
        683,  713,  733,  735,  742,  761,  764,  765,  782,  789,  794,
        795,  799,  850,  856,  872,  932,  952,  958,  960,  965,  970,
        971,  993, 1026, 1056, 1068, 1071, 1078, 1084, 1087, 1090, 1107,
       1111, 1112, 1121, 1123, 1145, 1147, 1149, 1155, 1156, 1175, 1180,
       1183, 1188, 1190, 1208, 1217, 1224, 1247, 1253, 1254, 1261, 1315,
       1330, 1339, 1347, 1355, 1360, 1372, 1373, 1383, 1393, 1408, 1413,
       1418, 1453, 1472, 1494, 1498, 1501, 1509, 1512, 1513, 1516, 1558,
       1562, 1568, 1569, 1575, 1578, 1593, 1597, 1599, 1606, 1608, 1617,
       1618, 1623, 1631, 1650, 1655, 1658, 1660, 1672, 1683, 1684, 1691,
       1692, 1702, 1713, 1726, 1730, 1731, 1732, 1758, 1760, 1768, 1775,
       1785, 1794, 1795, 1803, 1833, 1835, 1851, 1862, 1864, 1891, 1941,
       1963, 1965, 2018, 2020, 2025, 2029, 2045, 2067, 2075, 2098, 2107,
       2110, 2124, 2145, 2147, 2149, 2165, 2168, 2184, 2189, 2213, 2215,
       2232, 2233, 2236, 2239, 2249, 2251, 2252, 2254, 2260, 2286, 2287,
       2304, 2310, 2316, 2340, 2344, 2358, 2360, 2392, 2412, 2417, 2420,
       2458, 2473, 2484, 2492, 2498, 2516, 2522, 2523, 2534, 2545, 2586,
       2592, 2602, 2609, 2614, 2615, 2620, 2627, 2664, 2673, 2678, 2688,
       2737, 2750, 2753, 2769, 2771, 2802, 2818, 2820, 2823, 2855, 2868,
       2876, 2884, 2894, 2906, 2908, 2910, 2913, 2922, 2926, 2927, 2973,
       3006, 3007, 3015, 3019, 3023, 3032, 3033, 3039, 3045, 3047, 3061,
       3070, 3079, 3095, 3100, 3101, 3105, 3122, 3123, 3130, 3145, 3151,
       3154, 3160, 3187, 3194, 3204, 3231, 3238, 3251, 3257, 3279, 3305,
       3309, 3312, 3314, 3317, 3334, 3337, 3350, 3355, 3379, 3383, 3387,
       3393, 3396, 3399, 3400, 3404, 3410, 3412, 3434, 3464, 3465, 3473,
       3482, 3501, 3519, 3526, 3529, 3541, 3550, 3570, 3595, 3597, 3646,
       3662, 3686, 3703, 3704, 3706, 3723, 3767, 3771, 3787, 3790, 3817,
       3834, 3837, 3842, 3857, 3869, 3872, 3887, 3898, 3912, 3921, 3937,
       3973, 3992, 3999, 4003, 4038, 4049, 4058, 4067, 4071, 4080, 4081,
       4084, 4098, 4102, 4112, 4113, 4122, 4123, 4130, 4139, 4140, 4151,
       4170, 4185, 4190, 4194, 4202, 4210, 4216, 4217, 4269, 4302, 4313,
       4321, 4337, 4338, 4362, 4367, 4381, 4386, 4395, 4397, 4398, 4414,
       4423, 4438, 4442, 4475, 4477, 4502, 4504, 4511, 4513, 4514, 4516,
       4521, 4540, 4575, 4613, 4625, 4630, 4631, 4638, 4640, 4672, 4684,
       4695, 4704, 4712, 4716, 4721, 4731, 4734, 4742, 4747, 4751, 4768,
       4771, 4773, 4775, 4801, 4812, 4813, 4819, 4822, 4827, 4830, 4851,
       4856, 4872, 4879, 4890, 4891, 4898, 4900, 4903, 4908, 4919, 4929,
       4942, 4947, 4949, 4955, 4957, 4963, 4983, 4993, 5019, 5026, 5049,
       5050, 5053, 5062, 5082, 5090, 5094, 5098, 5100, 5123, 5128, 5144,
       5156, 5170, 5183, 5196, 5198, 5202, 5205, 5206, 5221, 5234, 5250,
       5266, 5269, 5272, 5284, 5306, 5309, 5323, 5331, 5359, 5381, 5403,
       5414, 5421, 5440, 5459, 5467, 5481, 5489, 5503, 5505, 5521, 5527,
       5537, 5544, 5547, 5550, 5559, 5572, 5589, 5605, 5606, 5609, 5627,
       5635, 5648, 5653, 5662, 5665, 5674, 5702, 5703, 5718, 5735, 5741,
       5748, 5764, 5766, 5786, 5794, 5795, 5798, 5802, 5805, 5814, 5822,
       5824, 5826, 5833, 5851, 5853, 5861, 5872, 5885, 5903, 5908, 5910,
       5928, 5938, 5940, 5968, 5970, 5983, 5998, 6001, 6004, 6005, 6021,
       6027, 6033, 6039, 6045, 6060, 6062, 6070, 6077, 6080, 6084, 6085,
       6100, 6120, 6149, 6161, 6163, 6176, 6200, 6202, 6213, 6216, 6231,
       6252, 6261, 6266, 6289, 6292, 6300, 6328, 6329, 6340, 6349, 6363,
       6408, 6409, 6410, 6428, 6442, 6472, 6482, 6496, 6501, 6516, 6517,
       6547, 6550, 6552, 6561, 6590, 6595, 6599, 6603, 6608, 6621, 6623,
       6625, 6630, 6631, 6634, 6646, 6650, 6681, 6685, 6687, 6690, 6727,
       6728, 6743, 6753, 6754, 6765, 6789, 6790, 6802, 6808, 6826, 6830,
       6834, 6843, 6850, 6870, 6897, 6906, 6918, 6939, 6940, 6951, 6967,
       6968, 6977, 6997, 7016, 7017, 7032, 7047, 7051, 7058, 7072, 7077,
       7086, 7093, 7110, 7144, 7146, 7150, 7155, 7169, 7201, 7203, 7204,
       7205, 7216, 7220, 7228, 7231, 7235, 7261, 7262, 7282, 7301, 7315,
       7323, 7333, 7376, 7377, 7383, 7388, 7396, 7399, 7414, 7415, 7417,
       7420, 7436, 7438, 7444, 7454, 7457, 7464, 7471, 7485, 7487, 7488,
       7492, 7493, 7500, 7507, 7514, 7516, 7522, 7529, 7535, 7539, 7548,
       7578, 7590, 7601, 7618, 7620, 7636, 7640, 7642, 7644, 7646, 7653,
       7684, 7695, 7708, 7713, 7727, 7728, 7742, 7744, 7749, 7753, 7770,
       7772, 7791, 7801, 7807, 7811, 7821, 7825, 7828, 7841, 7850, 7872,
       7882, 7887, 7894, 7896, 7907, 7911, 7930, 7931, 7932, 7938, 7942,
       7952, 7966, 7968, 8000, 8001, 8015, 8020, 8023, 8069, 8074, 8076,
       8080, 8093, 8103, 8107, 8109, 8119, 8127, 8134, 8137, 8138, 8142,
       8146, 8158, 8159, 8174, 8178, 8181, 8187, 8191, 8205, 8214, 8235,
       8239, 8243, 8260, 8264, 8267, 8276, 8278, 8279, 8283, 8284, 8296,
       8304, 8310, 8313, 8328, 8336, 8341, 8342, 8355, 8358, 8360, 8361,
       8362, 8365, 8382, 8384, 8403, 8413, 8439, 8446, 8447, 8467, 8481,
       8500, 8501, 8517, 8518, 8520, 8532, 8537, 8574, 8575, 8612, 8618,
       8629, 8660, 8663, 8670, 8672, 8674, 8683, 8694, 8699, 8703, 8706,
       8708, 8709, 8714, 8720, 8728, 8729, 8731, 8743, 8757, 8762, 8764,
       8771, 8773, 8791, 8794, 8799, 8819, 8827, 8834, 8846, 8847, 8849,
       8864, 8868, 8881, 8894, 8895, 8897, 8921, 8933, 8944, 8946, 8961,
       8965, 8972, 9001, 9030, 9038, 9039, 9049, 9054, 9063, 9085, 9088,
       9096, 9100, 9102, 9115, 9123, 9134, 9148, 9149, 9155, 9161, 9176,
       9187, 9189, 9190, 9202, 9214, 9217, 9233, 9238, 9275, 9308, 9312,
       9317, 9335, 9340, 9362, 9393, 9400, 9406, 9411, 9412, 9414, 9417,
       9420, 9445, 9475, 9485, 9495, 9515, 9547, 9574, 9587, 9615, 9643,
       9644, 9655, 9696, 9701, 9709, 9731, 9733, 9753, 9765, 9783, 9791,
       9808, 9813, 9828, 9833, 9857, 9860, 9871, 9873, 9893, 9896, 9909,
       9919, 9920, 9930, 9964, 9965, 9966, 9972, 9975, 9989, 9993]
def intermediate_value(pt, keyguess, offset):
    return sbox[pt ^ keyguess] ^ mask[(offset+1)%16]

def ge_and_sr(runs, output_probabilities, param, leakage_model, test_trace_data, step, fraction, byte=0):
    nt = len(output_probabilities)
    nt_kr = int(nt / fraction)
    nt_interval = int(nt / (step * fraction))
    key_ranking_sum = np.zeros(nt_interval)
    success_rate_sum = np.zeros(nt_interval)
    key_probabilities_key_ranks = np.zeros((runs, nt, 256))

    # ---------------------------------------------------------------------------------------------------------#
    # compute labels for all key hypothesis
    # ---------------------------------------------------------------------------------------------------------#
    labels_key_hypothesis = np.zeros((256, nt))
    for key_byte_hypothesis in range(0, 256):
        key_h = bytearray.fromhex(param["key"])
        key_h[byte] = key_byte_hypothesis
        labels_key_hypothesis[key_byte_hypothesis][:] = aes_labelize(test_trace_data, leakage_model, byte, param, key=key_h)

    # ---------------------------------------------------------------------------------------------------------#
    # predict output probabilities for shuffled test or validation set
    # ---------------------------------------------------------------------------------------------------------#

    probabilities_kg_all_traces = np.zeros((nt, 256))
    for index in range(nt):
        probabilities_kg_all_traces[index] = output_probabilities[index][
            np.asarray([int(leakage[index]) for leakage in labels_key_hypothesis[:]])
        ]

    for run in range(runs):

        probabilities_kg_all_traces_shuffled = shuffle(probabilities_kg_all_traces, random_state=random.randint(0, 100000))
        key_probabilities = np.zeros(256)
        kr_count = 0
        for index in range(nt_kr):
            key_probabilities += np.log(probabilities_kg_all_traces_shuffled[index] + 1e-36)
            key_probabilities_key_ranks[run][index] = probabilities_kg_all_traces_shuffled[index]
            key_probabilities_sorted = np.argsort(key_probabilities)[::-1]
            if (index + 1) % step == 0:
                key_ranking_good_key = list(key_probabilities_sorted).index(param["good_key"]) + 1
                key_ranking_sum[kr_count] += key_ranking_good_key
                if key_ranking_good_key == 1:
                    success_rate_sum[kr_count] += 1
                kr_count += 1
        print(
            "KR: {} | GE for correct key ({}): {})".format(run, param["good_key"], key_ranking_sum[nt_interval - 1] / (run + 1)))

    guessing_entropy = key_ranking_sum / runs
    success_rate = success_rate_sum / runs

    hf = h5py.File('output_prob.h5', 'w')
    hf.create_dataset('output_probabilities', data=output_probabilities)
    hf.close()

    return guessing_entropy, success_rate, key_probabilities_key_ranks

attack_size = 250
training_size = 9000
test_size = 1000
#predictions = scipy.special.softmax((np.load('predictions.npy')),axis=1)

predictions = scipy.special.softmax((np.load('predictions_3.npy')),axis=1)
(_,keys,ptxts,masks)= import_traces.import_traces(False, 'dpa4','', 10000) 
#plain, offset,key = ptxts[test],masks[test],keys[test]
plain, offset,key = ptxts,masks,keys

# Compute guessing entropy
ge_m = []
for j in range(int(test_size/attack_size)):
    ge_x = []
    pred = np.zeros(256)
    for i in range(attack_size):
        idx = i+training_size+(j*attack_size)
        for keyGuess in range(256):
            sbox_out = intermediate_value(plain[idx], keyGuess,offset[idx])
            lv = hw[sbox_out]
            pred[keyGuess] += np.log(predictions[i][lv]+ 1e-36)
    
        # Calculate key rank
        res = np.argmax(np.argsort(pred)[::-1] == key[0]) #argsort sortira argumente od najmanjeg do najveceg, [::-1} okrece to, argmax vraca redni broj gdje se to desilo
        ge_x.append(res)
    ge_m.append(ge_x)
    
ge_m =np.array(ge_m)

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
