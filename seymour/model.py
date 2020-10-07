#!/usr/bin/env python3

#
# [seymour/model.py]
#
# Model interface definition.
# Copyright (C) 2019-2020, Leslie Schumm
#

import copy

class Model(object):

    error = None
    
    def __init__(self):
         """
         initialize a model with a random set of instance variables.
         """
         pass
    
    def copy_self(self):
         """
         returns a deep copy of this instance of the Model.
         """
         return copy.deepcopy(self)

    def mutate(self, alpha):
         """
         (should be defined by implementations of Model)
         mutates this instantation of Model.
         """
         raise NotImplementedError('.mutate() not implemented')

    def reproduce_asexually(self):
         """
         returns a new instance of this Model with a mutated instance.
         """

         child = self.copy_self()
         return child

    def reproduce_sexually(self, other):
        """
        returns a new instance of a Model with a genome that is a
        combination between two models.
        """
        raise NotImplementedError(".reproduce_sexually() not implemented.")

    def update_error(self, i):
        """
        (should be defined by implementations of Model)
        updates the .error attribute to the current
        error of the model (inverse performance,
        0 is maximum possible performance, while a high
        number is low performance).
        """
        raise NotImplementedError('.error() not implemented.')
    
    def evaluate(self, i):
        """
        (should be defined by implementations of Model)
        returns the model's output for the given input `i`.
        """
        raise NotImplementedError('.evaluate() not implemented.')
    
    def display(self):
        """
        (should be defined by implementations of Model)
        shows a glimpse of what this agent's parameters are like.
        """
        raise NotImplementedError('.display() not implemented.')
        
