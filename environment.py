import gym
import numpy as np
import util
import math
from gym import spaces
class Environment(gym.Env):
    metadata = {'render.modes': ['human'] }
    def __init__(self, data):
        self.initial_request = 0.1
        self.request = 0.1
        self.min_action = -1.0
        self.max_action = 1.0
        self.min_position = 0.0
        self.max_position = 1.0
        self.data = data
        self.low_state = np.array(
            [self.min_position, self.min_position], dtype=np.float32
        )
        self.high_state = np.array(
            [self.max_position, self.max_position], dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=self.min_action, high=self.max_action, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=self.low_state, high=self.high_state, dtype=np.float32
        )

    def reset(self):
        resource = self.getState(0)
        request = self.initial_request
        state = np.array([request, resource], dtype=np.float32).reshape(2)
        return state

    def seed(self, seed=None):
        pass

    def getState(self, step, n=2):
        step = step % len(self.data)
        block = self.data[step:step + n] if step < len(self.data) - n - 1 else self.data[len(self.data) - n - 1:len(self.data)]

        # state = []
        resource = []
        for i in range(n - 1):
            resource.append(block[i])
        return np.array(resource)

    def decide_scaling_unit(self, state, action):
        request = self.request
        if action > 0:
            scaling_unit = abs(request - (100 / 75 * state))
        else:
            scaling_unit = abs((100 / 75 * state) - request)
        return round(float(scaling_unit),3)

    def step(self, step, action):

        done = False
        info = {}
        resource = self.getState(step)

        if (self.request + action) < self.min_position:
            action = self.max_position

        if ((resource / (self.request + action)) < 0.75) & ((resource / (self.request + action)) > 0.10):
            if (self.request + action) <= self.min_position:
                reward = 1
            else:
                reward = 1
                self.request = self.request + action
        elif ((resource / (self.request + action)) <= 0.10) | (((resource / (self.request + action)) >= 0.75) & (resource / (self.request + action) < 1.0)):
            reward = 0
        else:
            reward = -1
        state = np.array([self.request, resource], dtype=np.float32).reshape(2)
        print('action:{} | state :{} |  reward :{} | request :{} | ratio :{}'.format(action, resource, reward, self.request, (resource / (self.request + action))))
        return state, reward, done, info