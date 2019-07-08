import math
import numpy as np
import seymour.ga as ga

from libc.math cimport exp as cexp
from libc.math cimport abs as abs
#cython: boundscheck=False, wraparound=False, nonecheck=False

def sig(x):
    cdef double e
    try:
        e = cexp(-x)
    except OverflowError:
        return 0
    return 1 / (1 + e)

def sig_vec(v):
    out = np.zeros((len(v), 1))
    for i, val in enumerate(v):
        out[i, 0] = sig(val[0])
    return np.asarray(out)

def rpd(double est, double act):
    # relative percent difference
    if est == act:
        return 0
    else:
        return 2 * (est - act) / (abs(est) + abs(act))

def se(double est, double act):
    return (act - est) ** 2

class Network(ga.Individual):

    def __init__(self, inputs, outputs, nl=5, genome=None):
        # self.inputs = np.asmatrix(inputs)
        # self.outputs = np.asmatrix(outputs)

        self.inputs = inputs
        self.outputs = outputs
        
        self.ni = len(inputs[0])
        self.no = len(outputs[0])
        self.nl = nl
        self.genome = genome
        self.genome_size = (self.ni * self.ni + self.ni) * self.nl + self.ni * self.no

        super().__init__()

    def reproduce(self, genome):
        return Network(self.inputs, self.outputs, self.nl, genome)

    def evaluate(self, inp):

        idx = 0
        layers = []
        
        for i in range(0, self.nl):
            coef = self.genome[idx: idx + self.ni * self.ni]
            coef = np.reshape(coef, (self.ni, self.ni))
            idx += self.ni * self.ni
            bias = self.genome[idx: idx + self.ni]
            bias = np.reshape(bias, (self.ni, 1))
            idx += self.ni
            layers.append((coef, bias))
        
        trans = self.genome[idx: idx + self.ni * self.no]
        idx += self.ni * self.no
        trans = np.reshape(trans, (self.no, self.ni))

        for (coef, bias) in layers:
            inp = sig_vec(np.matmul(coef, inp) + bias)
            
        return sig_vec(np.matmul(trans, inp))

    def fitness_function(self):
        score = 0
        for (input, output) in zip(self.inputs, self.outputs):
            exp = self.evaluate(input)
            act = output

            score += np.sum(np.array(list(map(se, exp, act))))
        return score

def columnize(item):
    return np.asmatrix(item).reshape(len(item), 1)

def columnize_list(list):
    return [columnize(x) for x in list]
