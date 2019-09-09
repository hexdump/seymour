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
    return np.asarray([columnize(x) for x in list])

def sig(double x):
    cdef double e
    #x = round(x, 3)
    try:
        e = cexp(-x)
    except OverflowError:
        return 0
    return ((1 / (1 + e)) - 0.5) * 2

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
                           ni,
                           no,
                           nl=5,
                           nw=None):
    if nw is None:
        nw = ni
    
    idx = 0
    layers = []

    in_trans = genome[idx: idx + ni * nw]
    idx += ni * nw
    in_trans = np.reshape(in_trans, (nw, ni))
    
    for i in range(0, nl):
        coef = genome[idx: idx + nw * nw]
        coef = np.reshape(coef, (nw, nw))
        idx += nw * nw
        bias = genome[idx: idx + nw]
        bias = np.reshape(bias, (nw, 1))
        idx += nw
        layers.append((coef, bias))
        
    out_trans = genome[idx: idx + nw * no]
    idx += nw * no
    out_trans = np.reshape(out_trans, (no, nw))

    def evaluate(inp):
        inp = sig_vec(np.matmul(in_trans, inp))
        
        for (coef, bias) in layers:
            inp = sig_vec(np.matmul(coef, inp) + bias * 0.01)

        return sig_vec(np.matmul(out_trans, inp) * (1/nw))

    return evaluate

def make_fitness_function(inputs, outputs,
                          ni, no, nl=5, nw=None):
    
    def fitness(genome):
        f = make_evaluate_function(genome,
                                   ni, no, nl, nw)
        
        exp = np.asarray([f(i) for i in inputs])

#        print(exp)
#        print(outputs)
#        print(np.hstack(exp))
#        print(np.hstack(outputs))
#        print(outputs)
        
        return common.list_rpd(np.hstack(exp), np.hstack(outputs))

    return fitness

def network_genome_size(ni, no, nl=5, nw=None):
    if nw is None:
        nw = ni
    return ni * nw + (nw * nw + nw) * nl + nw * no
