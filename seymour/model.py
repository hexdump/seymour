#!/usr/bin/env python3

#
# [model.py]
#
# Model interface definition.
# Copyright (C) 2019, Liam Schumm
#

import copy

class Model(object):
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
         
     
     def reproduce(self):
         """
         (should be defined by implementations of Model)
         returns a new instance of this Model with a mutated instance.
         """
         
         raise NotImplementedError('.reproduce() not implemented')

     def evaluate(self, i):
         """
         (should be defined by implementations of Model)
         returns the Model output for the given input `i`.
         """
         raise NotImplementedError('.evaluate() not implemented')
