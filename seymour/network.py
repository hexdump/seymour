#!/usr/bin/env python3

#
# [network.py]
#
# Dynamic neural network model definition.
# Copyright (C) 2019, Liam Schumm
#

import numpy as np
from numpy import dot
from scipy.special import expit
from seymour import Model
import sympy
from seymour.utils import random_boolean, breed_booleans, mutate_boolean, probability

sympy.init_printing()

def random(shape=()):
    return np.random.random(shape) * 2 - 1
    
class FullyConnectedNet(Model):

    weights = []
    biases = []

    input_size  = 1
    output_size = 1

    # hyperparameters
    hp_prob_split = 0.05
    hp_prob_join  = 0.0
    hp_prob_mut = 0.5
    hp_bias_coeff = 0.1
    
    def __init__(self):        
        self.weights = [random((self.input_size, self.input_size)),
                        random((self.input_size, self.input_size)),
                        random((self.input_size, self.output_size))]
        self.biases = [random(self.input_size),
                       random(self.input_size),
                       random(self.output_size)]
        
    def mutate(self, alpha):
        i = 0
        while i < len(self.weights):
            if probability(self.hp_prob_mut):
                weight_delta = random(self.weights[i].shape) * self.error * alpha
                self.weights[i] += weight_delta
                bias_delta = random(self.biases[i].shape) * self.error * alpha
                self.biases[i] += bias_delta
                
            if probability(self.hp_prob_split):
                starting_dim = self.weights[i].shape[0]
                ending_dim = self.weights[i].shape[1]
                inner_dim = int(np.random.random() * (self.input_size + self.output_size)) + 1
                
                self.weights = (self.weights[:i]
                               + [random((starting_dim, inner_dim)),
                                  random((inner_dim, ending_dim))]
                               + self.weights[i + 1:])
                self.biases = (self.biases[:i]
                               + [random(inner_dim),
                                  random(ending_dim)]
                               + self.biases[i + 1:])

            if probability(self.hp_prob_join) and (i < len(self.weights) - 1):
                self.weights = self.weights[:i] + [dot(self.weights[i], self.weights[i + 1])] + self.weights[i + 2:]
                self.biases = self.biases[:i] + [self.biases[i + 1]] + self.biases[i + 2:]
            
            i += 1
            
    def evaluate(self, i):
        """
        takes in a user function as an input that operates on the nicely structured
        graph object.
        """
        
        for weight, bias in zip(self.weights, self.biases):
            i = expit(dot(i, weight) + bias * self.hp_bias_coeff)
        return i

    def display(self):
        for weight in self.weights:
            sympy.pprint(sympy.Matrix(weight))

        for bias in self.biases:
            sympy.pprint(sympy.Matrix(bias))
