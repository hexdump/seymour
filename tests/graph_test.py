import json
from utils import boolean_to_float
from seymour.graph import FixedSizeGraph
from seymour.optimizer import Optimizer

class SimpleFixedSizeGraphTest(FixedSizeGraph):
    prob_flip = 0.01
    
    num_nodes = 4
    node_ids = ['a', 'b', 'c', 'd']
    
    num_relations = 2
    relation_ids = ['is_parent_of', 'is_child_of']
    
    # list of matrices, one for each relation. a matrix for
    # relation R is a grid where x * y is the truth value of x R y.
    relations = None

def parent_child(parent, child):
    def inner(graph):
        return boolean_to_float((parent in graph[child]['is_child_of']) and (child in graph[parent]['is_parent_of']))
    return inner

dataset = [
    [parent_child('a', 'b'), 1],
    [parent_child('b', 'c'), 1],
    [parent_child('c', 'd'), 1],

    [parent_child('a', 'a'), 0],
    [parent_child('b', 'b'), 0],
    [parent_child('c', 'c'), 0],
    [parent_child('d', 'd'), 0],
    
    [parent_child('b', 'a'), 0],
    [parent_child('c', 'b'), 0],
    [parent_child('d', 'c'), 0],    
]
    
g = SimpleFixedSizeGraphTest()

o = Optimizer(model = SimpleFixedSizeGraphTest,
              dataset = dataset)

o.optimize(10000, 10)
