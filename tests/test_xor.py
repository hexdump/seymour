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

inputs = net.columnize_list([
    [0, 0],
    [0, 0.5],
    [0.5, 0],
    [0.5, 0.5]
])

outputs = np.asarray(net.columnize_list([
    [0],
    [0.5],
    [0.5],
    [1]
]))

genome_size = net.network_genome_size(ni=2,
                                      no=1,
                                      nl=2,
                                      nw=3)

gp = ga.Population(genome_size, 500)
error = net.make_fitness_function(inputs,
                                  outputs,
                                  2, 1, 2, 3)

gp.optimize(error, 1000)

g = gp.best_genome()
evaluate_function = net.make_evaluate_function(g,
                                               2, 1, 2, 3)

for input in inputs:
    print(evaluate_function(input))
