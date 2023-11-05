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
        self.max_position = 4.0
        self.max_ration = 1.0
        self.data = data
        self.low_state = np.array(
            [self.min_position, self.min_position, self.min_position], dtype=np.float32
        )
        self.high_state = np.array(
            [self.max_position, self.max_position, self.max_ration], dtype=np.float32
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
        state = np.array([request, 0.5, False], dtype=np.float32).reshape(3)
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

        # if ((resource/self.request) < 0.75) & ((resource/self.request) > 0.10):
        #     action = 0.0

        scaled = False
        if ((resource / (self.request + action)) < 0.75) & ((resource / (self.request + action)) > 0.10):
            if (self.request + action) <= self.min_position:
                reward = 0
            else:
                if ((resource/self.request) < 0.75) & ((resource/self.request) > 0.10):
                    reward = 1
                    action = 0.0
                else:
                    reward = 1
                    self.request = self.request + action
                    scaled = True
        elif (((resource / (self.request + action)) <= 0.10)
              | (((resource / (self.request + action)) >= 0.75) & (resource / (self.request + action) < 1.0))):
            reward = -0.1
        else:
            reward = -1

        usage = resource / self.request
        if scaled == False:
            usage = 0.5
        state = np.array([self.request, usage, scaled], dtype=np.float32).reshape(3)
        if mode == 'test':
            print('scaled: {} | action:{} | state :{} |  reward :{} | request :{} | ratio :{}'.format(scaled, action, resource, reward, self.request, (resource / self.request)))
        return state, reward, done, info