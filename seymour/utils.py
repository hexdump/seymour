#!/usr/bin/env python3

#
# [utils.py]
#
# Utilities for Seymour models.
# Copyright (C) 2019, Liam Schumm
#

import random

def random_boolean():
    return random.choice([True, False])

def breed_booleans(this, that):
    return random.choice([this, that])

def mutate_boolean(this, prob_flip):
    return not this if random.random() < prob_flip else this

def boolean_to_float(boolean):
    return 1.0 if boolean else 0.0
