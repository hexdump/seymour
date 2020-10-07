#!/usr/bin/env python3

#
# [seymour/optimizer.py]
#
# Optimizer code for Seymour.
# Copyright (C) 2019-2020, Leslie Schumm
#

import time

class Optimizer(object):
    def __init__(self, model):
        self.model = model
        self.population = []
        
        self.max_errors = []
        self.min_errors = []
        self.mean_errors = []

        self.best_agent = None

    def optimize(self, init_args, population_size, epochs, alpha):
        if self.population == []:
            print('initializing population...')
            self.population = [self.model(*init_args) for _ in range(population_size)]

        try:
            for _ in range(epochs):
                start_time = time.time()
                
                children_population = [agent.reproduce(alpha) for agent in self.population * 2]
    
                self.population = children_population
    
                for agent in self.population:
                    agent.update_error()
	
                self.population.sort(key=lambda agent: agent.error)
                self.population = self.population[:int(len(self.population) / 2)]
    
                print()
                print("epoch: " + str(_))
                print("min error: " + str(self.population[0].error))

                # now, let's reevaluate our current best
                if self.best_agent:
                    self.best_agent.update_error()
                    if self.best_agent.error > self.population[0].error:
                        self.best_agent = self.population[0].reproduce_asexual()
                
                self.min_errors.append(self.population[0].error)
                self.max_errors.append(self.population[-1].error)
                self.mean_errors.append(sum(agent.error for agent in self.population) / len(self.population))
                
                elapsed_time = time.time() - start_time
                print("elapsed time: " + str(elapsed_time))
    
                self.population[0].display()
        except KeyboardInterrupt:
            pass

        return self.population[0]
                    

            
            
