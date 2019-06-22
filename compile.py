import numpy as np
from sympy import Symbol, Matrix
from sympy.matrices import Transpose

# %{NUM_ROUNDS}
# %{NUM_INDIVIDUALS}
# %{EVALUATION}

def gen_matrix(y, x, i0=0):

    num_symbols = x * y
    symbols = []

    for i in range(num_symbols):
        symbols.append(Symbol("this.genome[" + str(i + i0) + "]"))

    matrix = []

    index = 0
    for row in range(y):
        matrix.append([])
        for column in range(x):
            matrix[-1].append(symbols[index])
            index += 1

    return Matrix(matrix)

def rpd(est, act):
    # relative percent difference
    if est == act:
        return 0
    else:
        return 2 * (est - act) / (abs(est) + abs(act))

class Model(object):

    built = False
    
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    def build(self):

        final = 0
        
        matrices = []
        t1, t2, t3 = None, None, None

        # TODO: assert that inputs and outputs are homogeneous length
        ni = len(self.inputs[0])
        no = len(self.outputs[0])

        offset = 0

        t1 = gen_matrix(1, ni, offset)
        offset += 1 * ni
        
        for i in range(5):
            matrices.append(gen_matrix(ni, ni, offset))
            offset += ni * ni

        t2 = gen_matrix(ni, no, offset)
        offset += ni * no

        t3 = gen_matrix(ni, 1, offset)
        offset += ni * no

        for (input, output) in zip(self.inputs, self.outputs):

            est = Matrix(input) * t1
            
            for matrix in matrices:
                est = est * matrix

            est = Transpose(est * t2)
            est = est * t3

            for i in range(no):
                final += rpd(est[i], output[i])

        print(final)
            
        built = True

    def evaluate(self, input):
        if not built:
            raise Exception("CNS model not built.")
        

inputs = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
]

outputs = [
    [0],
    [1],
    [1],
    [0],
]

m = Model(inputs, outputs)
m.build()
