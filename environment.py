
from gym.core import Env
import numpy as np
from gym import spaces
class Environment(Env):
    metadata = {'render.modes': ['human'] }
    def __init__(self, data, task=None):
        self.initial_request = 0.01
        self.request = 0.01
        self.min_action = -1.0
        self.max_action = 1.5
        self.min_position = 0.0
        self.max_position = 1.5
        self.max_ratio = 1.0
        self.uppser_bound = 0.8
        self.lower_bound = 0.2
        self.data = data
        self.low_state = np.array(
            [self.min_position, self.min_position, self.min_position, self.min_position], dtype=np.float32
        )
        self.high_state = np.array(
            [self.max_position, self.max_position, self.max_position, self.max_position], dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=self.min_action, high=self.max_action, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=self.low_state, high=self.high_state, dtype=np.float32
        )


    def reset(self):
        resource = self.getResource(0)
        request = self.initial_request
        state = np.array([request, resource/request, resource, 0], dtype=np.float32).reshape(4)
        return state

    def seed(self, seed=None):
        pass

    def getResource(self, step, n=2):
        step = step % len(self.data)
        block = self.data[step:step + n] if step < len(self.data) - n - 1 else self.data[len(self.data) - n - 1:len(self.data)]

        # state = []
        resource = []
        for i in range(n - 1):
            resource.append(block[i])
        return np.array(resource)


    def step(self, step, mode, action):

        done = False
        info = {}
        resource = self.getResource(step)
        scaled = 0
        if (self.request + action) <= self.min_position :
            reward = 0
            done = True
        else :
            if ((resource / (self.request + action)) < self.uppser_bound) & ((resource / (self.request + action)) > self.lower_bound):
                self.request = self.request + action
                reward = 1
                if action > 0:
                    scaled = 1
                elif action < 0:
                    scaled = -1
            elif resource / (self.request + action) <= self.lower_bound:
                reward = -0.5
            elif (resource / (self.request + action) >= self.uppser_bound) & (resource / (self.request + action) < 1.0):
                reward = -0.5
            else:
                reward = -1

        usage = resource / self.request


        state = np.array([self.request, usage, resource, scaled], dtype=np.float32).reshape(4)
        if mode == 'test':
            print('action:{} | state :{} |  reward :{} | request :{} | ratio :{}'
                  .format(action, resource, reward, self.request, usage))
        return state, reward, done, info