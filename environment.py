import numpy as np
import util
import math
from gym import spaces
class Environment(object):
    def __init__(self):
        self.request = 2
        self.num_up = 0  # upscaling 횟수
        self.num_down = 0  # downscaling 횟수
        self.num_stay = 0  # stay 횟수
        self.num_scaling = 0
        self.min_action = -1.0
        self.max_action = 1.0
        self.min_position = 0.0
        self.max_position = 15.0
        self.max_scale = 0.07
        self.low_state = np.array(
            [self.min_position, -self.max_scale], dtype=np.float32
        )
        self.high_state = np.array(
            [self.max_position, self.max_scale],  dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=self.min_action, high=self.max_action, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=self.min_position, high=self.max_position, shape=(1,), dtype=np.float32
        )
    # def _action(self, action):
    #     act_k = (self.action_space.high - self.action_space.low)/ 2.
    #     act_b = (self.action_space.high + self.action_space.low)/ 2.
    #     return act_k * action + act_b
    #
    # def _reverse_action(self, action):
    #     act_k_inv = 2./(self.action_space.high - self.action_space.low)
    #     act_b = (self.action_space.high + self.action_space.low)/ 2.
    #     return act_k_inv * (action - act_b)

    def reset(self, data, nb_states, request):
        state = self.getState(data, 0, nb_states + 1)
        return state

    def getState(self, data, step, n):
        step = step % len(data)
        block = data[step:step + n] if step < len(data) - n - 1 else data[len(data) - n - 1:len(data)]
        if step == 0:
            self.request = 1
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

    def step(self, action, state):
        scaling_unit = self.decide_scaling_unit(state[0], action)
        if (action * scaling_unit) + self.request > self.max_position:
            scaling_unit = 1.0
        if (action * scaling_unit) + self.request < self.min_position:
            scaling_unit = 1.0

        if ((action * scaling_unit) + self.request < state) & ((action * scaling_unit) + self.request > self.min_position):
            reward = 1
            self.request = (action * scaling_unit) + self.request
        else :
            reward = 0
        # print('action:{} | state :{} | request :{} | scaling :{}'.format(action, state, self.request, action * scaling_unit))
        return reward, self.request