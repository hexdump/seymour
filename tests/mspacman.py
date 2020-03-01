import json
from seymour import Model
from seymour.network import FullyConnectedNet
from seymour.optimizer import Optimizer
from seymour.utils import array, random
import numpy as np
from scipy.signal import convolve2d
from skimage.color import rgb2gray

import gym
import time

class MsPacman(Model):

    def __init__(self):
        self.net = FullyConnectedNet(228, 8 + 10)
#        self.kernel1 = random((50, 50))
#        self.kernel2 = random((50, 50))

    def mutate(self, alpha):
        self.net.mutate(alpha)
#        self.kernel1 = self.kernel1 + random(self.kernel1.shape) * alpha

    def update_error(self):
        return self.display(False)
        
    def display(self, display=True):
        fitness = 0

        env = gym.make('MsPacman-ram-v0')
        
        observation = env.reset()
        state = np.zeros(10)
        reward = 0
        done = False

        max_position = 0
        
        for i in range(500):
            fitness += reward

            # render the environment on the screen
            if display:
                env.render()
                time.sleep(0.1)

            #
            # determine action
            #

            # greyscalify and convolve observation, and flatten
#            observation = rgb2gray(observation)
            # (210, 160) -> (161, 111)
#            observation = convolve2d(observation, self.kernel1, mode='valid')
            # (161, 111) -> (112, 62)
#            observation = convolve2d(observation, self.kernel2, mode='valid')
            # (112, 62) -> (6944,)
            
            observation = observation.flatten()
            
            # put together observation with reward and state. end with an array
            # of shape (7044,)
            reward_and_state = array([reward]) + state
            for i in range(10):
                observation = np.concatenate([observation, reward_and_state])
                
            # evaluate through network
            result = self.net.evaluate(observation)

            # separate action and state update
            action_onehot = result[:8]
            state = result[-10:]
            
            # calculate onehot
            action = list(action_onehot).index(max(action_onehot))

            # iterate
            observation, reward, done, info = env.step(action)

        self.error = 1 / fitness

o = Optimizer(model = MsPacman)
m = o.optimize(1, 1, 1)

import pickle
with open('model', 'wb') as f:
    pickle.dump(m, f)
