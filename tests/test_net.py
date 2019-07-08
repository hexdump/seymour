#!/usr/bin/env python3

#
# [test.py]
#
# Runs some Seymour test examples.
# Copyright (C) 2019, Liam Schumm
#

import seymour.net as net
import seymour.ga as ga

inputs = net.columnize_list([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

outputs = net.columnize_list([
    [0],
    [1],
    [1],
    [0]
])

gt = ga.GeneticTrainer(net.Network, (inputs, outputs, 3))

s = gt.train_until(.1)

for input in inputs:
    print(s.evaluate(input))
