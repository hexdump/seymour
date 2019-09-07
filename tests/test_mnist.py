#!/usr/bin/env python3

#
# [tests/test_xor.py]
#
# A binary addition model trained using a Seymour net.
# Copyright (C) 2019, Liam Schumm
#

import seymour.net as net
import seymour.ga as ga

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

onehot = lambda i, n: [0] * (i - 1) + [1] + [0] * (n - i - 1)

with open("train.csv") as f:
    lines = f.read().split("\n")[1:-1]
    for line in lines:
        for x in line.split(","):
            try:
                int(x)
            except:
                print(line)
                print(x)
    data = [[int(x) for x in line.split(",")] for line in lines]

    labels = [onehot(row[0], 784) for row in data]
    
    inputs = [row[1:] for row in data]
    outputs = net.columnize_list(labels)


genome_size = net.network_genome_size(ni=784,
                                      no=9,
                                      nl=3,
                                      nw=784)

gp = ga.Population(genome_size, 500)
error = net.make_fitness_function(inputs,
                                  outputs,
                                  784, 9, 3, 784)

gp.optimize(error, 1000)

g = gp.best_genome()
evaluate_function = net.make_evaluate_function(g,
                                               784, 9, 3, 784)

for input in inputs:
    print(evaluate_function(input))
