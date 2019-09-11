#!/usr/bin/env python3

#
# [tests/test_xor.py]
#
# A binary addition model trained using a Seymour net.
# Copyright (C) 2019, Liam Schumm
#

import seymour.net as net
import seymour.ga as ga
from seymour.manifold import layer, list_rpd
import numpy as np

np.set_printoptions(suppress=True)

# inputs = net.columnize_list([
#     [0, 0],
#     [0, 0.5],
#     [0.5, 0],
#     [0.5, 0.5]
# ])

# outputs = np.asarray(net.columnize_list([
#     [0],
#     [0.5],
#     [0.5],
#     [1]
# ]))

import numpy

def onehot(i, n):
    array = [0] * n
    array[i] = 1
    return array

#onehot = lambda i, n: numpy.array([0] * (i - 1) + [1] + [0] * (n - i - 1))

from progressbar import ProgressBar

with open("train.csv") as f:
    lines = f.read().split("\n")[1:-1]
    print("loading mnist dataset...")
    pbar  = ProgressBar(maxval=len(lines))
    pbar.start()
    inputs = []
    outputs = []
    for (i, line) in enumerate(lines):
        pbar.update(i + 1)
        data = [int(x) for x in line.split(",")]

        inputs.append(onehot(data[0], 10))
        outputs.append(data[1:])
#    inputs = net.columnize_list(inputs)
#    print(inputs)
#    outputs = net.columnize_list(labels)
#    outputs = net.columnize_list(labels) #numpy.asarray(labels)
#    print(net.columnize_list(outputs))
#    print(np.asarray(labels).shape)
#    print(outputs)

#    print(inputs.shape)
#    print(outputs.shape)
    pbar.finish()

# inputs = [
#     [0, 0],
#     [0, 1],
#     [1, 0],
#     [1, 1]
# ]

# outputs = [
#     [0],
#     [1],
#     [1],
#     [0]
# ]

import math
from math import sin

test_inputs = inputs
test_outputs = outputs

layers = [300, 100, 10]
def error(genome):
    global test_inputs, test_outputs
    exp = []
    for (i, o) in zip(test_inputs, test_outputs):
        g = 0
        v = i
        for l in layers:
            v0 = v
            v = []
#            c = genome[g]
            x = layer(*v0)
            for j in range(l):
                v.append(layer(genome[g], *v0))
                g += 1
                #v.append(x + sin(genome[g]))
                #g += 1
                
        exp.append([(x + 1)/2 for x in v])
#    exp = [(x + 1) / 2 for x in exp]
    return list_rpd(exp, outputs)
#    return exp

def evaluate(genome, i):
    g = 0
#    exp =
#    i
    v = i
    for l in layers:
        v0 = v
        v = []
        for j in range(l):
            v.append(layer(genome[g], *v0))
            g += 1
    return [(x+1)/2 for x in v]

gp = ga.Population(sum(layers), 10)
import random
for i in range(500):
    random.shuffle(inputs)
    random.shuffle(outputs)
    test_inputs = inputs[:100]
    test_outputs = outputs[:100]
    gp.optimize(error, 1)
    g = gp.best_genome()
    test_inputs = inputs
    test_outputs = outputs
    print(evaluate(g, inputs[0]))
#    test_inputs = inputs[
    
    
g = gp.best_genome()
for input in inputs:
    print(evaluate(g, input))
    
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

