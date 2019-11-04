class Optimizer(object):
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset

    def optimize(self, population_size, epochs):
        self.population = [self.model() for _ in range(population_size)]

        for _ in range(epochs):
            children_population = [agent.reproduce() for agent in self.population]

            for child in children_population:
                error = 0
                for (i, o) in self.dataset:
                    error += abs(child.evaluate(i) - o)
                child.error = error
                print(error)
            self.population = self.population + children_population

            self.population.sort(key=lambda agent: agent.error)
            self.population = self.population[:int(len(self.population) / 2)]
        return self.population[0]
                    

            
            
