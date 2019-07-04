import seymour.ga as ga
from random import random

steps = 10

target = (random() * 2 * steps - steps,
          random() * 2 * steps - steps)

def pair_coords(genome, dim):
    return [tuple(genome[i:i+dim]) for i in range(0, len(genome), dim)]

def sum_coords(coords, dim):
    init = [0 for _ in range(dim)]

    for coord in coords:
        for i in range(len(init)):
            init[i] += coord[i]
    return init

def logistic(x):
    return (2 / (1 + 2.718 ** (-x))) - 1

def normalize_coords(coords):
    return [(logistic(x) for x in coord) for coord in coords]

def distance(p1, p2):
    total = 0
    for i in range(len(p1)):
        total += (p1[i] - p2[i]) ** 2
    return total ** 0.5

class Path(ga.Individual):
    def __init__(self, target, path_length, dimension=2, genome=None):
        self.genome_size = path_length * dimension
        self.target = target
        self.path_length = path_length
        self.dimension = dimension

        super().__init__()

    def reproduce(self, genome):
        return Path(self.target, self.path_length, self.dimension, self.genome)

    def evaluate(self):
        pairs = pair_coords(self.genome, self.dimension)
        norm_pairs = normalize_coords(pairs)
        total = sum_coords(pairs, self.dimension)
        return total
        
    def fitness_function(self):
        return distance(self.evaluate(), self.target)

gt = ga.GeneticTrainer(Path, (target, steps))
s = gt.train(600)

print(target)
print(s.evaluate())
    


