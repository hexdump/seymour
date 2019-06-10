import math
import numpy as np
# class Network(object):

#     def __init__(self, ni, no, nl):
#         self.ni = ni
#         self.no = no
#         self.nl = nl
#         self.genome = np.random.randn(n)

#     def evaluate()

def sig(x):
    return 1 / (1 + math.exp(-x))

def sig_vec(v):
    out = np.zeros((len(v), 1))
    for i, val in enumerate(v):
        out[i, 0] = sig(val[0])
    return np.asarray(out)
        
def evaluate(genome, inp, ni, nl, no):

    d = ni * ni
    matrices = []

    for i in range(0, d * nl, d):
        matrices.append(np.reshape(genome[i : i + d], (ni, ni)))

    matrices.append(np.reshape(genome[d * nl : d * nl + ni * no], (no, ni)))

    for matrix in matrices[:-1]:

        inp = np.matmul(matrix, inp)

        # pr
        # print(np)
        
        inp = np.reshape(sig_vec(inp), (ni, 1))

    inp = np.matmul(matrices[-1], inp)
    inp = np.reshape(sig_vec(inp), (no, 1))

    return inp


ni = 2
no = 1
nl = 3

genome = np.random.randn(ni * ni * nl + ni * no)

i = np.array([[1], [3]])

# print(evaluate(genome, i, ni, nl, no))
