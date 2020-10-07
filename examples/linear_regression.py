#!/usr/bin/env python3

#
# [examples/linear_regression.py]
#
# An example of doing a simple linear regression with Seymour.
# Copyright (C) 2020, Leslie Schumm
#

import matplotlib.pyplot as plt

from seymour import Model, Optimizer
from seymour.utils import probability, breed_lists, mutate_float

data_inputs = [
    [1, 2, 1],
    [3, 8, 4],
    [9, 6, 3],
    [4, 9, 4],
]

data_outputs = [
    [1, 0, 4],
    [3, 9, 15],
    [9, 4, 18],
    [4, 20, 17],
]

class LinearRegressionModel(Model):
    def __init__(self, inputs, outputs, inputs_dim, outputs_dim):
        self.inputs = inputs
        self.outputs = outputs
        self.inputs_dim = inputs_dim
        self.outputs_dim = outputs_dim

        # initialize all weights to 0
        self.weights = [[0 for __ in range(inputs_dim)] for _ in range(outputs_dim)]

    def evaluate(self, inputs):
        predictions = []
        for output_idx in range(self.outputs_dim):
            prediction = sum([self.weights[output_idx][i] * inputs[i] for i in range(self.inputs_dim)])
            predictions.append(prediction)
        return predictions
            
    def update_error(self):
        self.error = 0
        
        for (inputs, outputs) in zip(data_inputs, data_outputs):
            predicted_outputs = self.evaluate(inputs)
            for (predicted_output, output) in zip(predicted_outputs, outputs):
                self.error += (predicted_output - output) ** 2

    def mutate(self, alpha):
        for output_set in self.weights:
            for (weight_idx, weight) in enumerate(output_set):
                output_set[weight_idx] = mutate_float(weight, alpha)
                
    def reproduce_sexually(self, other):
        child = self.reproduce_asexually()
        for (output_set_idx, (self_output_set, other_output_set)) in enumerate(zip(self.weights, other.weights)):
            child.weights[output_set_idx] = breed_lists(self_output_set, other_output_set)
        return child

    def display(self):
        print(self.weights)
    
optimizer = Optimizer(model = LinearRegressionModel)
model = optimizer.optimize((data_inputs, data_outputs, 3, 3), 10000, 100, 0.05)
