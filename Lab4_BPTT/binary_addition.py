import random
import numpy as np

def string2array(string):
    a = np.zeros((len(string),1))
    for i in range(len(string)):
        a[i,0] = float(string[i])
    return np.flip(a)

def ba():
    a = random.randint(-128, 127)
    if a > 0:
        b = random.randint(-128, 127-a)
    else:
        b = random.randint(-128-a, 127)

    a_s = np.binary_repr(a, width=8)
    a_array = string2array(a_s)
    b_s = np.binary_repr(b, width=8)
    b_array = string2array(b_s)
    c_s = np.binary_repr(a+b, width=8)
    c_array = string2array(c_s)

    # print("a", a, a_array)
    # print("b", b, b_array)
    # print("c", a+b, c_array)s
    return np.hstack((a_array,b_array)) , c_array

# def ba_batch_generator(batch_size):
#     ba()