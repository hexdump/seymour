import json
from seymour import Model
from seymour.network import FullyConnectedNet
from seymour.optimizer import Optimizer
from seymour.utils import array, random
import numpy as np
from scipy.signal import convolve2d
from skimage.color import rgb2gray
from skimage.transform import resize

import scipy.misc

import gym
import time

class MsPacman(Model):

    def __init__(self):
        self.net = FullyConnectedNet(3944, 8 + 10)
        self.kernel1 = random((20, 20))
        self.kernel2 = random((20, 20))

    def mutate(self, alpha):
        self.net.mutate(alpha)
#        self.kernel1 = self.kernel1 + random(self.kernel1.shape) * alpha

    def update_error(self):
        return self.display(False)
        
    def display(self, display=True):
        fitness = 0

        env = gym.make('MsPacman-v0')

        env.env.ale.saveScreenPNG(b'test_image.png')

        
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

            # greyscalify
            observation = rgb2gray(observation)
            observation = observation[1:168,]
            observation = resize(observation, (100, 100))
            observation = convolve2d(observation, self.kernel1, mode='valid')
            observation = convolve2d(observation, self.kernel2, mode='valid')
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
m = o.optimize(100, 100, 1)

import pickle
with open('model', 'wb') as f:
    pickle.dump(m, f)
