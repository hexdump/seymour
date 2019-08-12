#!/usr/bin/env python3

#
# [tests/test_addition.py]
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
    [0, 1],
    [1, 0],
    [1, 1]
])

outputs = np.asarray(net.columnize_list([
    [0],
    [0.58],
    [0],
    [0.23]
]))

genome_size = net.network_genome_size(ni=2,
                                      no=1,
                                nl=5)

gp = ga.Population(genome_size, 500)
#gt = ga.GeneticTrainer(net.Network, (inputs, outputs, 2))
#s = gt.train_until(.01)

#evaluate_function = net.make_evaluate_function(
#    
#    ni=2, no=2, nl=3)

error = net.make_fitness_function(inputs,
                                  outputs,
                                  2, 1, 5)

gp.optimize(error, 1000)

g = gp.best_genome()
evaluate_function = net.make_evaluate_function(g,
                                               2, 1, 5)

f = gp.population[-1].genome
fevaluate_function = net.make_evaluate_function(f,
                                                2, 1, 5)


for input in inputs:
    print(evaluate_function(input))

# print('---')
    
# for input in inputs:
#     print(fevaluate_function(input))

#    print(fevaluate_function(np.asarray([0])))

