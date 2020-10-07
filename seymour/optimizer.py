#!/usr/bin/env python3

#
# [seymour/optimizer.py]
#
# Optimizer code for Seymour.
# Copyright (C) 2019-2020, Leslie Schumm
#

import time
from statistics import median

class Optimizer(object):
    def __init__(self, model):
        self.model = model
        self.population = []
        
        self.max_errors = []
        self.min_errors = []
        self.median_errors = []

        self.best_agent = None

    def optimize(self, init_args, population_size, epochs, alpha):
        if self.population == []:
            print('initializing population...')
            self.population = [self.model(*init_args) for _ in range(population_size)]

        try:
            for _ in range(epochs):
                # sort agents by error
                for agent in self.population:
                    agent.update_error()
                self.population.sort(key = lambda agent: agent.error)

                # print and record statistics about the population
                print("epoch: " + str(_))
                min_error = self.population[0].error
                max_error = self.population[-1].error
                median_error = median(agent.error for agent in self.population)
                print(f"min error: {min_error}")
                print(f"max error: {max_error}")
                print(f"median error: {median_error}")
                self.min_errors.append(min_error)
                self.max_errors.append(max_error)
                self.median_errors.append(median_error)
                
                # now, let's reevaluate our current best (if we have one)
                if self.best_agent is not None:
                    self.best_agent.update_error()
                    if self.best_agent.error > self.population[0].error:
                        self.best_agent = self.population[0].reproduce_asexually()
                else:
                    self.best_agent = self.population[0].reproduce_asexually()
                
                # pick best ones as parents
                parents = self.population[:int(population_size / 2)]

                # breed
                children = []
                for i in range(0, len(parents), 2):
                    children.append(parents[i].reproduce_sexually(parents[i + 1]))
                    children.append(parents[i + 1].reproduce_sexually(parents[i]))

                # mutate children
                for child in children:
                    child.mutate(alpha)
    
                self.population = children
                self.population[0].display()
                
        except KeyboardInterrupt:
            pass

        return self.best_agent
                    

            
            
