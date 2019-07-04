# seymour

*Feed me training data, Seymour!*

A genetic algorithm solver library, primarily used for the solving of fixed-size deep neural nets.

## Genetic Algorithm How-To

Genetic algorithms are handled as large sets of objects. These objects are defined by the user, so all of the objects simulated can also be evaluated in whatever application is being optimized.

In this example, we'll be definining a neural network object that can be optimized over a population. Key things to note about this example are:

- Definition of a `reproduce` function
- Definition of `fitness_function`

Without these components, an `Individual` object will not function.

```
import seymour as sy

class Network(sy.ga.Individual):

    def __init__(self, inputs, outputs, nl=5, genome=None):
        # self.inputs = np.asmatrix(inputs)
        # self.outputs = np.asmatrix(outputs)

        self.inputs = inputs
        self.outputs = outputs
        
        self.ni = len(inputs[0])
        self.no = len(outputs[0])
        self.nl = nl
        self.genome = genome
        self.genome_size = (self.ni * self.ni + self.ni) * self.nl + self.ni * self.no

        super().__init__()

    def reproduce(self, genome):
        return Network(self.inputs, self.outputs, self.nl, genome)

    def evaluate(self, inp):

        idx = 0
        layers = []
        
        for i in range(0, self.nl):
            coef = self.genome[idx: idx + self.ni * self.ni]
            coef = np.reshape(coef, (self.ni, self.ni))
            idx += self.ni * self.ni
            bias = self.genome[idx: idx + self.ni]
            bias = np.reshape(bias, (self.ni, 1))
            idx += self.ni
            layers.append((coef, bias))
        
        trans = self.genome[idx: idx + self.ni * self.no]
        idx += self.ni * self.no
        trans = np.reshape(trans, (self.no, self.ni))

        for (coef, bias) in layers:
            inp = sig_vec(np.matmul(coef, inp) + bias)
            
        return sig_vec(np.matmul(trans, inp))

    def fitness_function(self):
        score = 0
        for (input, output) in zip(self.inputs, self.outputs):
            exp = self.evaluate(input)
            act = output

            score += abs(sum(map(se, exp, act)))
        return score
```

## Neural Network How-To

``
import seymour.net as net
import seymour.ga as ga

inputs = net.columnize_list([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

outputs = net.columnize_list([
    [0],
    [1],
    [1],
    [0]
])

gt = ga.GeneticTrainer(net.Network, (inputs, outputs, 3))

s = gt.train(100)

for input in inputs:
    print(s.evaluate(input))
```