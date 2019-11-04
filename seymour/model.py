#!/usr/bin/env python3

#
# [model.py]
#
# Model interface definition.
# Copyright (C) 2019, Liam Schumm
#

import copy

class Model(object):

    error = 0
    
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

    def mutate(self):
         """
         (should be defined by implementations of Model)
         mutates this instantation of Model.
         """
         raise NotImplementedError('.mutate() not implemented')

    def reproduce(self):
         """
         returns a new instance of this Model with a mutated instance.
         """

         child = self.copy_self()
         child.mutate()
         return child

    def evaluate(self, i):
         """
         (should be defined by implementations of Model)
         returns the Model output for the given input `i`.
         """
         raise NotImplementedError('.evaluate() not implemented')
