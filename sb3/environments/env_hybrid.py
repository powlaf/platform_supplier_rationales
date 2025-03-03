import glob
import random
import pickle
import warnings
import numpy as np

import gymnasium as gym
from gymnasium import spaces

warnings.filterwarnings("ignore")


class HybridEnvTraining(gym.Env):

    metadata = {"render_modes": ["console"]}

    def __init__(self, mode, name):
        super(HybridEnvTraining, self).__init__()

        self.file_names = glob.glob(f'drl_data/{mode}_data/{name}_*.pkl')
        self.current_file_names = self.file_names.copy()

        self.env_t, self.env_g = {}, {}
        self.env_supplier, self.state = [], []
        self.t, self.max_t = 0, 0

        # Define action and observation space
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=100.0, shape=(6,), dtype=np.float32)

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

            self.env_t = tracking['env_t']  # supplier_cost, customer_valuations, # suppliers, # customers, pmin, pmax
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
        pmin, pmax = self.env_t[self.t][4], self.env_t[self.t][5]

        self.state = [cost, min_price, pmin, pmax, patience, self.t]
        # cost, min_price, pmin, pmax, patience, t

        return np.array(self.state).astype(np.float32), {}

    def step(self, action):

        reward = 0
        terminated = False
        if self.state[2] <= action <= self.state[3]:

            prob = min(1, self.env_t[self.t][3] / self.env_t[self.t][2])

            if random.random() < prob:
                c = random.choice(self.env_t[self.t][1])

                if c > action:
                    reward = (action - self.state[1]) * 10
                    terminated = True

        # update state
        self.t += 1
        self.state[5] += 1
        self.state[4] -= 1
        if self.t < self.max_t:
            self.state[2], self.state[3] = self.env_t[self.t][4], self.env_t[self.t][5]

        # termination and truncation
        if self.state[4] <= 0:
            terminated = True

        truncated = True if self.t >= self.max_t else False

        # Optionally we can pass additional info, we are not using that for now
        info = {}

        return (
            np.array(self.state).astype(np.float32),
            reward,
            terminated,
            truncated,
            info,
        )


class HybridEnvEvaluation(gym.Env):

    metadata = {"render_modes": ["console"]}

    def __init__(self, mode, name):
        super(HybridEnvEvaluation, self).__init__()

        self.file_names = glob.glob(f'drl_data/{mode}_data/{name}_*.pkl')
        self.current_file_names = self.file_names.copy()

        self.env_t, self.env_g = {}, {}
        self.env_supplier, self.state = [], []
        self.t, self.max_t = 0, 0

        # Define action and observation space
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=100.0, shape=(6,), dtype=np.float32)

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

            self.env_t = tracking['env_t']  # supplier_cost, customer_valuations, # suppliers, # customers, pmin, pmax
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
        pmin, pmax = self.env_t[self.t][4], self.env_t[self.t][5]

        self.state = [cost, min_price, pmin, pmax, patience, self.t]
        # cost, min_price, pmin, pmax, patience, t

        return np.array(self.state).astype(np.float32), {}

    def step(self, action):

        reward = 0
        terminated = False

        if self.state[2] <= action <= self.state[3]:
            prob = min(1, self.env_t[self.t][3] / self.env_t[self.t][2])
            if random.random() < prob:
                c = random.choice(self.env_t[self.t][1])

                if c > action:
                    reward = (action - self.state[1]) * 10
                    terminated = True

        # update state
        self.t += 1
        self.state[5] += 1
        self.state[4] -= 1
        if self.t < self.max_t:
            self.state[2], self.state[3] = self.env_t[self.t][4], self.env_t[self.t][5]

        # termination and truncation
        if self.state[4] <= 0:
            terminated = True

        truncated = True if self.t >= self.max_t else False

        # Optionally we can pass additional info, we are not using that for now
        info = {}

        return (
            np.array(self.state).astype(np.float32),
            reward,
            terminated,
            truncated,
            info,
        )


class HybridEnvTesting(gym.Env):

    metadata = {"render_modes": ["console"]}

    def __init__(self, mode, name):
        super(HybridEnvTesting, self).__init__()

        self.file_names = glob.glob(f'drl_data/{mode}_data_stable/{name}_*.pkl')
        self.current_file_names = self.file_names.copy()

        self.env_t, self.env_g = {}, {}
        self.env_supplier, self.state = [], []
        self.t, self.max_t = 0, 0

        # Define action and observation space
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=100.0, shape=(6,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        # get new training file
        if len(self.env_supplier) == 0:

            if len(self.current_file_names) == 0:
                self.current_file_names = self.file_names.copy()

            file_name = self.current_file_names.pop()

            with open(file_name, 'rb') as f:
                tracking = pickle.load(f)

            self.env_t = tracking['env_t']  # supplier_cost, customer_valuations, # suppliers, # customers, pmin, pmax
            self.env_g = tracking['env_g']  # patience, arrival_c/s, commission
            self.env_supplier = tracking['supplier']  # [t_arriving, own_costs]

            self.max_t = len(self.env_t) - 1

        # get new supplier (= episode)
        random.shuffle(self.env_supplier)
        self.data_supplier = self.env_supplier.pop()

        # get state from data
        self.t, cost = self.data_supplier[0], self.data_supplier[1]
        patience, commission = self.env_g[0], self.env_g[3]
        min_price = cost / (1 - commission)
        pmin, pmax = self.env_t[self.t][4], self.env_t[self.t][5]

        self.state = [cost, min_price, pmin, pmax, patience, self.t]
        # cost, min_price, pmin, pmax, patience, t

        return np.array(self.state).astype(np.float32), {}

    def step(self, action):

        reward = 0
        terminated = False
        truncated = False

        if self.state[2] <= action <= self.state[3]:

            prob = self.env_t[self.t][6]
            p = self.env_g[0] - self.state[4] + 2

            if self.data_supplier[p][0]  < prob:
                c = self.data_supplier[p][1]

                if c > action:
                    reward = (action - self.state[1]) * 10
                    terminated = True

        # update state
        self.t += 1
        self.state[5] += 1
        self.state[4] -= 1
        if self.t < self.max_t:
            self.state[2], self.state[3] = self.env_t[self.t][4], self.env_t[self.t][5]

        # termination and truncation
        if self.state[4] <= 0:
            truncated = True

        if self.t >= self.max_t:
            truncated = True

        # Optionally we can pass additional info, we are not using that for now
        info = {}

        return (
            np.array(self.state).astype(np.float32),
            reward,
            terminated,
            truncated,
            info,
        )

