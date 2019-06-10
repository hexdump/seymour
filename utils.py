import math
import random

def pair(population, pair_pct=1):

    population = population[:int(len(population) * pair_pct)]

    if len(population) % 2 != 0:
        population = population[:-1]

    for i in range(0, len(population), 2):
        yield (population[i], population[i + 1])

def sig(x):
    return 1 / (1 + math.exp(-x))

def chance(p):
    return random.random() < p
