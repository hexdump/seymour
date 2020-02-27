import json
from seymour.network import FullyConnectedNet
from seymour.optimizer import Optimizer
from seymour.utils import array
import numpy as np

import gym

class MountainCarTest(FullyConnectedNet):
    
    input_size = 2 + 1 + 1
    output_size = 3 + 1

    def update_error(self, display=False):
        fitness = 0

        env = gym.make('MountainCar-v0')
        
        observation = env.reset()
        state = 0
        reward = 0
        done = False

        max_position = 0
        
        for i in range(500):
            (position, _) = observation
            max_position = max(max_position, position)
            
            if position >= 0.5:
                break
            
            # we've survived for one round, so increment fitness
            fitness += 1

            # render the environment on the screen
            if display:
                env.render()

            # determine action
            i = np.append(observation, [reward, state])
            r = self.evaluate(i)
            action_onehot = r[:3]
            action = list(action_onehot).index(max(action_onehot))
            state = r[3]

            # iterate
            observation, reward, done, info = env.step(action)

        self.error = fitness * (1/(max_position + 0.1))
        
o = Optimizer(model = MountainCarTest)

m = o.optimize(10, 100, 1)

m.update_error(True)
