#!/usr/bin/env python3

#
# [utils.py]
#
# Utilities for Seymour models.
# Copyright (C) 2019, Liam Schumm
#

import random as rd
import numpy as np

def random_boolean():
    return rd.choice([True, False])

def breed_booleans(this, that):
    return rd.choice([this, that])

def breed_lists(l1, l2):
    assert len(l1) == len(l2)
    return [(l1[i] if i % 2 == 0 else l2[i]) for i in range(len(l1))]

def mutate_boolean(this, prob_flip):
    return not this if rd.random() < prob_flip else this

def boolean_to_float(boolean):
    return 1.0 if boolean else 0.0

def mutate_float(x, sd):
    return np.random.normal(x, sd)

def probability(p):
    return rd.random() < p

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

def random(shape=()):
    return np.random.random(shape) * 2 - 1

