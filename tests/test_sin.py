#!/usr/bin/env python3

#
# [tests/test_xor.py]
##
# A binary addition model trained using a Seymour net.
# Copyright (C) 2019, Liam Schumm
#

import seymour.net as net
import seymour.ga as ga
from seymour.common import list_rpd
from seymour.manifold import cossim, layer, list_rpd

# from progressbar import ProgressBar

# with open("train.csv") as f:
#     lines = f.read().split("\n")[1:-1]
#     print("loading mnist dataset...")
#     pbar  = ProgressBar(maxval=len(lines))
#     pbar.start()
#     labels = []
#     inputs = []
#     for (i, line) in enumerate(lines[:1]):
#         pbar.update(i + 1)
#         data = [int(x) for x in line.split(",")]

#         labels.append(onehot(data[0], 10))
#         inputs.append(data[1:])
#     inputs = net.columnize_list(inputs)
# #    print(inputs)
# #    outputs = net.columnize_list(labels)
#     outputs = net.columnize_list(labels) #numpy.asarray(labels)
# #    print(net.columnize_list(outputs))
# #    print(np.asarray(labels).shape)
# #    print(outputs)

# #    print(inputs.shape)
# #    print(outputs.shape)
#     pbar.finish()

# genome_size = net.network_genome_size(ni=784,
#                                       no=9,
#                                       nl=2,
#                                       nw=300)

# gp = ga.Population(genome_size, 100)
# error = net.make_fitness_function(inputs,
#                                   outputs,
#                                   784, 9, 2, 300)

# while True:
#     gp.optimize(error, 1)

#     g = gp.best_genome()
#     evaluate_function = net.make_evaluate_function(g,
#                                                    784, 9, 2, 300)

#     for input in inputs:
#         print(evaluate_function(input))



inputs = [
    [0,0],
    [0,1],
    [1,0],
    [1,1]
]

outputs = [
    [0],
    [1],
    [1],
    [0]
]

# from math import floor
# def layer(*argv):
#     total = 0
#     for arg in argv:
#         total += sin(arg)#sin(floor(arg * 2)) * sin(arg) 
#     return total

import math
from math import sin

def rpd(est, act):
    return abs(est - act)
    
layers = [3, 3, 1]
def error(genome):
    exp = []
    for (i, o) in zip(inputs, outputs):
        g = 0
        v = i
        for l in layers:
            v0 = v
            v = []
            for j in range(l):
                v.append(layer(genome[g], *v0))
                g += 1
        exp.append(v)
    return sum(list_rpd(exp[i], outputs[i]) for i in range(len(exp)))
#    return exp

def evaluate(genome, i):
    g = 0
    v = i
    for l in layers:
        v0 = v
        v = []
        for j in range(l):
            v.append(layer(genome[g], *v0))
            g += 1
    return v

gp = ga.Population(sum(layers), 1000)
gp.optimize(error, 100)
g = gp.best_genome()
for input in inputs:
    print(evaluate(g, input))
