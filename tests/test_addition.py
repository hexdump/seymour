#!/usr/bin/env python3

#
# [tests/test_addition.py]
#
# A binary addition model trained using a Seymour net.
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
    [0, 0],
    [0, 1],
    [0, 1],
    [1, 1]
])

gt = ga.GeneticTrainer(net.Network, (inputs, outputs, 2))

s = gt.train_until(.01)

for input in inputs:
    print(s.evaluate(input))
