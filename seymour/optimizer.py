import time
from numba import jit
from seymour.utils import tensor_difference

import dill
from pathos.multiprocessing import Pool

class Optimizer(object):
    def __init__(self, model):
        self.model = model

    def optimize(self, init_args, population_size, epochs, alpha):
        print('initializing population...')
        self.population = [self.model(*init_args) for _ in range(population_size)]

        try:
            for _ in range(epochs):
                start_time = time.time()
                
                children_population = [agent.reproduce(alpha) for agent in self.population * 2]
    
                self.population = children_population
    
                for agent in self.population:
                    agent.update_error()

                pool = Pool(4)
                pool.map(lambda x: x.update_error(), self.population)
                pool.close()
                    
                self.population.sort(key=lambda agent: agent.error)
                self.population = self.population[:int(len(self.population) / 2)]
    
                print()
                print("📆  epoch: " + str(_))
                print("💪  min error: " + str(self.population[0].error))
                elapsed_time = time.time() - start_time
                print("⏰ elapsed time: " + str(elapsed_time))
    
                self.population[0].display()
        except KeyboardInterrupt:
            pass

        return self.population[0]
                    

            
            
