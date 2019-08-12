import math
import numpy as np
import seymour.ga as ga
import numpy as np

from libc.math cimport exp as cexp
from libc.math cimport abs as abs
#cython: boundscheck=False, wraparound=False, nonecheck=False
from libc.math cimport round

import seymour.common as common

def columnize(item):
    return np.asmatrix(item).reshape(len(item), 1)

def columnize_list(list):
    return [columnize(x) for x in list]

def sig(double x):
    cdef double e
    #x = round(x, 3)
    try:
        e = cexp(-x)
    except OverflowError:
        return 0
    return (1 / (1 + e)) - 0.5

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

#def build_model(genome):
#def evaluate_function(model):

def make_evaluate_function(genome,
                           ni, no, nl=5):
    
        idx = 0
        layers = []

        for i in range(0, nl):
            coef = genome[idx: idx + ni * ni]
            coef = np.reshape(coef, (ni, ni))
            idx += ni * ni
            bias = genome[idx: idx + ni]
            bias = np.reshape(bias, (ni, 1))
            idx += ni
            layers.append((coef, bias))
        
        trans = genome[idx: idx + ni * no]
        idx += ni * no
        trans = np.reshape(trans, (no, ni))

        def evaluate(inp):
            for (coef, bias) in layers:
                inp = sig_vec(np.matmul(coef, inp) + bias * 0.01)

            return sig_vec(np.matmul(trans, inp))

        return evaluate

def make_fitness_function(inputs, outputs,
                          ni, no, nl=5):
    
    def fitness(genome):
        f = make_evaluate_function(genome,
                                   ni, no, nl)
        
        exp = np.asarray([f(i) for i in inputs])

#        print(exp.flatten())
#        print(outputs.flatten())
        
        return common.list_rpd(exp.flatten(), outputs.flatten())

    return fitness

def network_genome_size(ni, no, nl):
    return (ni * ni + ni) * nl + ni * no
