from seymour.utils import tensor_difference
from seymour.model import Model

class SupervisedModel(Model):

    dataset = None
    
    def update_error(self):
        error = 0
        for (i, o) in self.dataset:
            error += tensor_difference(self.evaluate(i), o)
        self.error = error
