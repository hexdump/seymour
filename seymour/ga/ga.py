#!/usr/bin/env python3

#
# [algorithm.py]
#
# A genetic algorithm trainer.
# Copyright (C) 2019, Liam Schumm
#

from random import random, shuffle
import numpy as np

SD_ERR_COEFF = 0.1
CONV_POW = 2

def deviate(mean, stddev):
    #if random() > 0.9:
    #    return np.random.normal(mean, 5)
    #else:
    return np.random.normal(mean, stddev)

def mutate_genome(genome, err):
    return [deviate(x, SD_ERR_COEFF * abs(err ** CONV_POW)) for x in genome]

def mix_genomes(l1, l2):
    assert len(l1) == len(l2)
    return [l1[i] if random() > 0.5 else l2[i]
            for i in range(len(l1))]

def rpd(est, act):
    # relative percent difference
    if est == act:
        return 0
    else:
        return 2 * (est - act) / (abs(est) + abs(act))

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

    def genome_similarity(self, other):
        total = 0
        for (s, o) in zip(self.genome, other.genome):
            total += rpd(s, o)
        return total

class Population(object):
    def __init__(self, member_class=Individual, init_args=()):
        self.population_size = 250
        self.population = [member_class(*init_args)
                           for _ in range(self.population_size)]
        
    def sort(self):
        self.population.sort(key=lambda i: i.fitness)

    def select(self):
        self.sort()
        self.population = self.population[:int(len(self.population) * 0.5)]
        
    def asexually_breed(self):
        children = []
        for i in self.population:
            children += i.asexually_reproduce()
        self.population = children

    def sexually_breed(self):
        children = []
        pop = self.population
        while len(pop) > 0:
            p1 = pop.pop()
            pop.sort(key=lambda x: p1.genome_similarity(x))
            p2 = pop[0]
            pop.pop()
            shuffle(pop)
            # sexual reproduction is symmetric, so it doesn't
            # matter that p2 isn't .sexually_reproducing with p1
            children += p1.sexually_reproduce(p2)
            
        self.population = children

    def mixed_breed(self):
        children = []
        unfit = []
        for i in self.population:
            if i.fitness < 0.5:
                children += i.asexually_reproduce()
            else:
                unfit.append(i)

        shuffle(unfit)
        if len(unfit) % 2 != 0:
            unfit.append(unfit[0])
            
        while len(unfit) > 0:
            p1 = unfit.pop()
            p2 = unfit.pop()
            #unfit.sort(key=lambda x: p1.genome_similarity(x))
            #p2 = unfit[0]
            #unfit.pop()
            #shuffle(unfit)
            children += p1.sexually_reproduce(p2)

        self.population = children        

    def best_individual(self):
        return self.population[0]

class GeneticTrainer(object):
    def __init__(self, member_class, init_args):
        self.population = Population(member_class, init_args)

    def train(self, iterations):
        for i in range(iterations):
            #if i % 10 == 0:
            #    print(i)
            print('round=' + str(i) + " fitness=" + str(self.population.best_individual().fitness))
            self.population.mixed_breed()
            self.population.select()

        return self.population.best_individual()

    def train_until(self, fitness, max_rounds=10000):
        for i in range(max_rounds):
            print('round=' + str(i) + " fitness=" + str(self.population.best_individual().fitness))
            if self.population.best_individual().fitness <= fitness:
                break
            self.population.mixed_breed()
            self.population.select()
        return self.population.best_individual()
            
        
