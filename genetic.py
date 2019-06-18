#!/usr/bin/env python3

#
# [genetic.py]
#
# Genetic algorithm optimizer library.
# Copyright (C) 2019, Liam Schumm.
#

import os
import math
import random
import numpy as np

rand = lambda: random.random() * 2 - 1

from utils import *

def pair(population, pair_pct=1):

    population = population[:int(len(population) * pair_pct)]

    if len(population) % 2 != 0:
        population = population[:-1]

    for i in range(0, len(population), 2):
        yield (population[i], population[i + 1])

def mksquare(arr, n):
    mat = []
    for i in range(n):
        mat.append(arr[i * n : i * n + 1])
    return mat

def dot(weights, vals):
    if len(weights) != len(vals):
        raise Exception()

    for i in range(len(weights)):
        weights[i] * vals[i]

def deviate(x, err):
    return numpy.random.normal(x, 0.5 * (err ** 0.1))

def _error(v1, v2):
    if v1 == v2:
        return 0
    else:
        return abs((v1 - v2 / abs(v1) + abs(v2)))

def sigmoid(x, derivative=False):
    sigm = 1. / (1. + np.exp(-x))
    if derivative:
        return sigm * (1. - sigm)
    return sigm

class Individual(object):

    def __init__(self, n_genome=50, genome=[]):
        if not genome:
            self.genome = [rand() for _ in range(n_genome)]
        else:
            self.genome = genome
        self.error = 10000

    def breed(self, p2):
        new_genome = []
        for i in range(len(self.genome)):
            r = rand()
            n = r * self.genome[i] + (1 - r) * p2.genome[i]
            new_genome.append(n)
        return Individual(genome=new_genome)

    def mutate(self):
        for (i, weight) in enumerate(self.genome):
            if chance(0.8):
                self.genome[i] = deviate(weight, self.error)

class GeneticAlgorithm(object):
    def __init__(self, error, n=20):
        self.error = error
        self.n = n
        self.population = []

    def selection(self):        
        for i in self.population:
            i.error = self.error(i)
        self.population.sort(key=lambda m: m.error)

        self.population = self.population[:int(len(self.population) * .5)]

    def breed(self):  
        new_population = []
        for p in pair(self.population):
            p1 = p[0]
            p2 = p[1]

            o1 = p1.breed(p2)
            o2 = p2.breed(p1)
            new_population += [o1, o2, o1, o2]
        self.population = new_population

    def mutate(self):
        for p in self.population:
            if chance(0.1):
                p.mutate()

    def train(self, initial_pop=1000, rounds=2000):
#        print(self.n)
        self.population = [Individual(self.n) for x in range(initial_pop)]
        for _ in range(2000):
            if True:
                os.system("clear")
                print("{} rounds completed.".format(_))
                print('ERR')
                print(self.error(self.best_individual()))
                #print(self.best_individual().genome)
                for k,v in [(np.asarray([[0], [0]]), 0),
                            (np.asarray([[1], [0]]), 1),
                            (np.asarray([[0], [1]]), 1),
                            (np.asarray([[1], [1]]), 0)]:
                    
                    o = evaluate(self.best_individual().genome, k, ni, nl, no)
                    print(o)

                
            self.selection()
            self.breed()
            self.mutate()

    def best_individual(self):
        return self.population[0]

ex = [1, 2, 3]

def _basic_error(l):
    return sum([abs(ex[j] - l[j]) for j in range(3)])

def basic_error(i):
    return _basic_error(i.genome)

#print(_basic_error([1,2,3.2]))
        

