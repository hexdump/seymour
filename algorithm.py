#!/usr/bin/env python3

#
# [algorithm.py]
#
# A genetic algorithm trainer.
# Copyright (C) 2019, Liam Schumm
#

from random import random
import numpy as np

SD_ERR_COEFF = 0.05

deviate = np.random.normal

def mutate_genome(genome, err):
    return [deviate(x, SD_ERR_COEFF * abs(err ** 2)) for x in genome]

def mix_genomes(l1, l2):
    assert len(l1) == len(l2)
    return [l1[i] if random() > 0.5 else l2[i]
            for i in range(len(l1))]

class Individual(object):
    genome = None
    def __init__(self):
        if self.genome is None:
            self.genome = [deviate(0, 1) for _ in range(self.genome_size)]
        self.fitness = self.fitness_function()

        
    def sexually_reproduce(self, mate):
        assert len(self.genome) == len(mate.genome)
        [c1, c2, c3, c4] = [self.reproduce(mutate_genome(mix_genomes(
            self.genome,
            mate.genome),
                                           self.fitness/2 + mate.fitness/2))
                            for _ in range(4)]
        return [c1, c2, c3, c4]
        
    def asexually_reproduce(self):
        return [self.reproduce(mutate_genome(self.genome, self.fitness))
                for _ in range(2)]

class Population(object):
    def __init__(self, member_class=Individual, init_args=()):
        self.population_size = 1000
        self.population = [member_class(*init_args)
                           for _ in range(self.population_size)]
        
    def sort(self):
        self.population.sort(key=lambda i: i.fitness)

    def asexually_breed(self):
        children = []
        for i in self.population:
            children += i.asexually_reproduce()
        self.population = children
        self.sort()
        self.population = self.population[:len(self.population) // 2]

    def best_individual(self):
        return self.population[0]

class GeneticTrainer(object):
    def __init__(self, member_class, init_args):
        self.population = Population(member_class, init_args)

    def train(self, iterations):
        for i in range(iterations):
            #if i % 10 == 0:
            #    print(i)
            print(i)
            print(self.population.best_individual().fitness)
            self.population.asexually_breed()

        return self.population.best_individual()
