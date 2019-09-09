#!/usr/bin/env python3

#
# [tests/test_xor.py]
#
# A binary addition model trained using a Seymour net.
# Copyright (C) 2019, Liam Schumm
#

import seymour.net as net
import seymour.ga as ga
from seymour.common import list_rpd


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
    [0, 0],
    [0, 0.5],
    [0.5, 0],
    [0.5, 0.5]
]

outputs = [
    [0],
    [0.5],
    [0.5],
    [1]
]

import math
from math import sin

def rpd(est, act):
    return abs(est - act)

def list_rpd(x1, x2):
    err = 0
    for x, y in zip(x1, x2):
        if isinstance(x, list):
            err += list_rpd(x, y)
        else:
            err += rpd(x, y)
    return err
    

def layer(*argv):
    return sum(sin(arg) for arg in argv) #/ len(argv)

layers = [3, 2, 1]
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
    return list_rpd(exp, outputs)
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
