import json
from seymour.network import FullyConnectedNet
from seymour.optimizer import Optimizer
from seymour.utils import array

class SimpleNetworkTest(FullyConnectedNet):

    input_size = 2
    output_size = 1

dataset = [
    [array([0, 0]), array(0)],
    [array([1, 0]), array(1)],
    [array([0, 1]), array(1)],
    [array([1, 1]), array(0)],
]

o = Optimizer(model = SimpleNetworkTest,
              dataset = dataset)

m = o.optimize(1000, 1000, 1)

m.display()

print(m.evaluate(array([0, 0])))
print(m.evaluate(array([1, 0])))
print(m.evaluate(array([0, 1])))
print(m.evaluate(array([1, 1])))
