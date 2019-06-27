# seymour

A genetic algorithm solver library, primarily used for the solving of fixed-size deep neural nets.

## Genetic Algorithm How-To

In order to create a genetic optimizer, you'll first need a fitness function. Each individual in a Seymour instance will have an arbitrarily long list of floating-point numbers to represent its "genome"; thus the fitness function will take in a list of floating point numbers and return a fitness value (with `0` being most fit and large negative and positive numbers being least).

For this example, we'll be optimizing a `n=4` genome so the individual components will sum to `4`. Here's the fitness function for this problem:

```
def fitness(genome):
    return 4 - sum(genome)
```

After we've defined our fitness function, we'll initialize a `GeneticTrainer` object with our fitness function and our genome size (`4`):

```
gt = GeneticTrainer(fitness, 4)
```

Then, let's train it and get the most fit individual after `200` rounds:

```
best_individual = gt.train(200)
```

And let's take a look at its genome and the sum:

```
print(best_individual.genome)
print(sum(best_individual.genome))
```
