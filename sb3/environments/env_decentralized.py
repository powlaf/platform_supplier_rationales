import glob
import os
import random
import pickle
import numpy as np

import gymnasium as gym
from gymnasium import spaces


class DecentralizedEnvTraining(gym.Env):

    metadata = {"render_modes": ["console"]}

    def __init__(self, mode, name):
        super(DecentralizedEnvTraining, self).__init__()

        self.file_names = glob.glob(f'drl_data/{mode}_data/{name}_*.pkl')
        self.current_file_names = self.file_names.copy()

        self.env_t, self.env_g = {}, {}
        self.env_supplier, self.state = [], []
        self.t, self.max_t = 0, 0

        # Define action and observation space
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=100.0, shape=(4,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        # get new training file
        if len(self.env_supplier) == 0:
            random.shuffle(self.current_file_names)
            if len(self.current_file_names) == 0:
                self.current_file_names = self.file_names.copy()

            file_name = self.current_file_names.pop()

            with open(file_name, 'rb') as f:
                tracking = pickle.load(f)

            self.env_t = tracking['env_t']  # supplier_cost, customer_valuations, # suppliers, # customers, price
            for t in range(len(self.env_t)):  # append prob of being selected
                self.env_t[t].append(min(1, self.env_t[t][3] / self.env_t[t][2]))
            self.env_g = tracking['env_g']  # patience, arrival_c/s, commission
            self.env_supplier = tracking['supplier']  # [t_arriving, own_costs]

            self.max_t = len(self.env_t) - 1

        # get new supplier (= episode)
        random.shuffle(self.env_supplier)
        self.data_supplier = self.env_supplier.pop()

        # get state from data
        self.t, cost = self.data_supplier
        patience, commission = self.env_g[0], self.env_g[3]
        min_price = cost / (1 - commission)

        self.state = [cost, min_price, patience, self.t]

        return np.array(self.state).astype(np.float32), {}

    def step(self, action):

        reward = 0
        terminated = False
        prob = self.env_t[self.t][4]

        if random.random() < prob:
            c = random.choice(self.env_t[self.t][1])

            if c > action:
                reward = (action - self.state[1]) * 10
                terminated = True

        # update state
        self.t += 1
        self.state[3] += 1
        self.state[2] -= 1

        # termination for impatient suppliers
        if self.state[2] <= 0:
            terminated = True

        truncated = True if self.t >= self.max_t else False

        info = {}

        return (
            np.array(self.state).astype(np.float32),
            reward,
            terminated,
            truncated,
            info,
        )


class DecentralizedEnvEvaluation(gym.Env):

    metadata = {"render_modes": ["console"]}

    def __init__(self, mode, name):
        super(DecentralizedEnvEvaluation, self).__init__()

        self.file_names = glob.glob(f'drl_data/{mode}_data/{name}_*.pkl')
        self.current_file_names = self.file_names.copy()

        self.env_t, self.env_g = {}, {}
        self.env_supplier, self.state = [], []
        self.t, self.max_t = 0, 0

        # Define action and observation space
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=100.0, shape=(4,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        # get new training file
        if len(self.env_supplier) == 0:
            if len(self.current_file_names) == 0:
                self.current_file_names = self.file_names.copy()

            file_name = self.current_file_names.pop()

            with open(file_name, 'rb') as f:
                tracking = pickle.load(f)

            self.env_t = tracking['env_t']  # supplier_cost, customer_valuations, # suppliers, # customers
            for t in range(len(self.env_t)):  # append prob of being selected
                self.env_t[t].append(min(1, self.env_t[t][3] / self.env_t[t][2]))
            self.env_g = tracking['env_g']  # patience, arrival_c/s, commission
            self.env_supplier = tracking['supplier']  # [t_arriving, own_costs]

            self.max_t = len(self.env_t) - 1

        # get new supplier (= episode)
        random.shuffle(self.env_supplier)
        self.data_supplier = self.env_supplier.pop()

        # get state from data
        self.t, cost = self.data_supplier
        patience, commission = self.env_g[0], self.env_g[3]
        min_price = cost / (1 - commission)

        self.state = [cost, min_price, patience, self.t]

        return np.array(self.state).astype(np.float32), {}

    def step(self, action):
        # termination = max_timesteps or patience
        # trunctuation = successfull match
        reward = 0
        terminated = False
        prob = self.env_t[self.t][4]

        if random.random() < prob:
            c = random.choice(self.env_t[self.t][1])

            if c > action:
                reward = (action - self.state[1]) * 10
                terminated = True

        # update state
        self.t += 1
        self.state[3] += 1
        self.state[2] -= 1

        # termination and truncation
        if self.state[2] <= 0:
            terminated = True

        truncated = self.t >= self.max_t

        # Optionally we can pass additional info, we are not using that for now
        info = {}

        return (
            np.array(self.state).astype(np.float32),
            reward,
            terminated,
            truncated,
            info
        )


class DecentralizedEnvTesting(gym.Env):

    metadata = {"render_modes": ["console"]}

    def __init__(self, mode, name):
        super(DecentralizedEnvTesting, self).__init__()

        self.file_names = glob.glob(f'drl_data/{mode}_data_stable/{name}_*.pkl')
        self.current_file_names = self.file_names.copy()

        self.env_t, self.env_g = {}, {}
        self.env_supplier, self.state = [], []
        self.t, self.max_t = 0, 0

        # Define action and observation space
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=100.0, shape=(4,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        # get new training file
        if len(self.env_supplier) == 0:

            if len(self.current_file_names) == 0:
                self.current_file_names = self.file_names.copy()

            file_name = self.current_file_names.pop()

            with open(file_name, 'rb') as f:
                tracking = pickle.load(f)

            self.env_t = tracking['env_t']  # { t: supplier_cost, customer_valuations, # suppliers, # customers, prob of being selected }
            self.env_g = tracking['env_g']  # patience, arrival_c/s, commission
            self.env_supplier = tracking['supplier']  # [t_arriving, own_costs, [random prob, random customer], [], []]

            self.max_t = len(self.env_t) - 1

        # get new supplier (= episode)
        self.data_supplier = self.env_supplier.pop()

        # get state from data
        self.t, cost = self.data_supplier[0], self.data_supplier[1]
        patience, commission = self.env_g[0], self.env_g[3]
        min_price = cost / (1 - commission)

        self.state = [cost, min_price, patience, self.t]

        return np.array(self.state).astype(np.float32), {}

    def step(self, action):
        # trunctuation = max_timesteps or patience
        # termination = successfull match

        reward = 0
        terminated = False
        truncated = False

        prob = self.env_t[self.t][4]
        p = self.env_g[0] - self.state[2] + 2

        if self.data_supplier[p][0] < prob:
            c = self.data_supplier[p][1]

            if c > action:
                reward = (action - self.state[1]) * 10
                terminated = True

        # update state
        self.t += 1
        self.state[3] += 1
        self.state[2] -= 1

        # termination and truncation
        if self.state[2] <= 0:
            truncated = True

        if self.t >= self.max_t:
            truncated = True

        info = {}

        return (
            np.array(self.state).astype(np.float32),
            reward,
            terminated,
            truncated,
            info
        )

