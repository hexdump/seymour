#!/usr/bin/env python3

#
# [algorithm.py]
#
# A genetic algorithm trainer.
# Copyright (C) 2019, Liam Schumm
#

from random import random, shuffle
import numpy as np

import seymour.genome as genome
import seymour.common as common

SD_ERR_COEFF = 0.5
CONV_POW = 2

# TODO: convert to cython struct
class Individual(object):
    genome = None
    error = None

class Population(object):
    def __init__(self, genome_size, population_size=250):
        self.population_size = population_size
        self.population = [Individual()
                           for _ in range(self.population_size)]
        for individual in self.population:
            individual.genome = genome.generate_genome(genome_size,
                                                       mean = 0,
                                                       err  = 1
            )

    def optimize(self, error, rounds=50, sd_err_coeff=SD_ERR_COEFF, conv_pow=CONV_POW):
        for _ in range(rounds):
            for individual in self.population:
                individual.error = error(individual.genome)
            print(str(_) + ' ' + str(self.population[0].error))
            self.asexually_breed(sd_err_coeff, conv_pow)
            self.population.sort(key=lambda i: abs(i.error))
            self.population = self.population[:self.population_size]
                    
    def asexually_breed(self, sd_err_coeff, conv_pow):
        children_genomes = []
        for i in self.population:
            i.genome = genome.asexually_reproduce(i.genome, i.error,
                                                  sd_err_coeff, conv_pow)
            i.fitness = None
            
    def sexually_breed(self, sd_err_coeff, conv_pow):
        children = []
        pop = self.population
        final = []
        while len(pop) > 0:
            p1 = pop.pop()
            pop.sort(key=lambda x: genome.genome_difference(p1.genome,
                                                            x.genome))
            p2 = pop.pop(0)
            shuffle(pop)
            
            # sexual reproduction is symmetric, so it doesn't
            # matter that p2 isn't .sexually_reproducing with p1
            genomes = genome.sexually_reproduce(p1.genome, p1.error,
                                                p2.genome, p2.error,
                                                sd_err_coeff, conv_pow)
            p1.genome = genomes[0]
            p2.genome = genomes[1]
            
            final += [p1, p2]
        
        self.population = final

    # def mixed_breed(self):
    #     children = []
    #     unfit = []
    #     for i in self.population:
    #         if i.error < 0.15:
    #             children += i.asexually_reproduce()
    #         else:
    #             unfit.append(i)

    #     shuffle(unfit)
    #     if len(unfit) % 2 != 0:
    #         unfit.append(unfit[0])
            
    #     while len(unfit) > 0:
    #         p1 = unfit.pop()
    #         p2 = unfit.pop()
    #         children += p1.sexually_reproduce(p2)

    #     self.population = children        

    def best_genome(self):
        return self.population[0].genome
