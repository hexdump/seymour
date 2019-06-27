import math
import numpy as np
import algorithm as alg

def sig(x):
    return 1 / (1 + math.exp(-x))

def sig_vec(v):
    out = np.zeros((len(v), 1))
    for i, val in enumerate(v):
        out[i, 0] = sig(val[0])
    return np.asarray(out)

def rpd(est, act):
    # relative percent difference
    if est == act:
        return 0
    else:
        return 2 * (est - act) / (abs(est) + abs(act))

class Network(alg.Individual):

    def __init__(self, inputs, outputs, nl=5, genome=None):
        # self.inputs = np.asmatrix(inputs)
        # self.outputs = np.asmatrix(outputs)

        self.inputs = inputs
        self.outputs = outputs
        
        self.ni = len(inputs[0])
        self.no = len(outputs[0])
        self.nl = nl
        self.genome = genome
        self.genome_size = self.ni * self.ni * self.nl + self.ni * self.no

        super().__init__()

    def reproduce(self, genome):
        return Network(self.inputs, self.outputs, self.nl, self.genome)

    def evaluate(self, inp):

        d = self.ni * self.ni
        matrices = []

        for i in range(0, d * self.nl, d):
            matrices.append(np.reshape(self.genome[i : i + d], (self.ni, self.ni)))

        matrices.append(np.reshape(self.genome[d * self.nl : d * self.nl + self.ni * self.no], (self.no, self.ni)))


        for matrix in matrices[:-1]:
            
            inp = np.matmul(matrix, inp)

            inp = np.reshape(sig_vec(inp), (self.ni, 1))

        inp = np.matmul(matrices[-1], inp)
        inp = np.reshape(sig_vec(inp), (self.no, 1))

        return inp

    def fitness_function(self):
        score = 0
        for (input, output) in zip(self.inputs, self.outputs):
            exp = self.evaluate(np.asmatrix(input).reshape(len(input), 1))
            act = output

            score += sum(map(rpd, exp, act))
        return score


inputs = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]

outputs = [
    [0],
    [1],
    [1],
    [0]
]

gt = alg.GeneticTrainer(Network, (inputs, outputs))

print(gt.train(200).genome)
