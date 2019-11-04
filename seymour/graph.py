#!/usr/bin/env python3

#
# [graph.py]
#
# Graph model definition.
# Copyright (C) 2019, Liam Schumm
#

from model import Model
from utils import random_boolean, breed_booleans, mutate_boolean

class FixedSizeGraph(Model):

    prob_flip = 0.1
    
    num_nodes = None
    node_ids = None
    
    num_relations = None
    relation_ids = None

    # list of matrices, one for each relation. a matrix for
    # relation R is a grid where x * y is the truth value of x R y.
    relations = []
    
    def __init__(self):        
        self.relations = [[
            [random_boolean() for node in range(self.num_nodes)] for node in range(self.num_nodes)]
                      for relation in range(self.num_relations)]

    def as_graph(self):
        nodes = {node_id: {} for node_id in self.node_ids}

        for relation, relation_id in zip(self.relations, self.relation_ids):
            for node in nodes:
                nodes[node][relation_id] = []
        
        for relation, relation_id in zip(self.relations, self.relation_ids):
            for i in range(self.num_nodes):
                for j in range(self.num_nodes):
                    if relation[j][i]:
                        nodes[self.node_ids[j]][relation_id] += [self.node_ids[i]]
        return nodes
            
        
    def mutate(self):
        for relation, relation_id in zip(self.relations, self.relation_ids):
            for i in range(self.num_nodes):
                for j in range(self.num_nodes):
                    relation[i][j] = mutate_boolean(relation[i][j], self.prob_flip)
        
    def evaluate(self, f):
        """
        takes in a user function as an input that operates on the nicely structured
        graph object.
        """
        
        return f(self.as_graph())
