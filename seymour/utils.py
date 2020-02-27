#!/usr/bin/env python3

#
# [utils.py]
#
# Utilities for Seymour models.
# Copyright (C) 2019, Liam Schumm
#

import random
import numpy as np

def random_boolean():
    return random.choice([True, False])

def breed_booleans(this, that):
    return random.choice([this, that])

def mutate_boolean(this, prob_flip):
    return not this if random.random() < prob_flip else this

def boolean_to_float(boolean):
    return 1.0 if boolean else 0.0

def probability(p):
    return random.random() < p

array = np.asarray

def tensor_difference(tensor_a, tensor_b):
    assert isinstance(tensor_a, np.ndarray)
    assert isinstance(tensor_b, np.ndarray)

    error = 0
    
    for (a, b) in zip(tensor_a.flatten(), tensor_b.flatten()):
        error += abs(a - b)

    size_difference  = abs(len(tensor_a.flatten()) - len(tensor_b.flatten()))
    error *= 1.5 ** size_difference
    error += size_difference
    
    return error
