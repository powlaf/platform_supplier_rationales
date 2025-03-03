import glob
import pickle
import random
import warnings
import numpy as np

from gymnasium import spaces
import gymnasium as gym

warnings.filterwarnings("ignore")


class CentralizedEnvTraining(gym.Env):

    metadata = {"render_modes": ["console"]}

    # Define action (0 = do not participate in market; 1 = participate in market)
    NOT = 0
    PARTICIPATE = 1

    def __init__(self, mode, name):
        super(CentralizedEnvTraining, self).__init__()

        self.file_names = glob.glob(f'drl_data/{mode}_data/{name}_*.pkl')
        self.current_file_names = self.file_names.copy()

        self.env_t, self.env_g = {}, {}
        self.env_supplier, self.state = [], []
        self.t, self.max_t = 0, 0

        # Define action and observation space
        n_actions = 2
        self.action_space = spaces.Discrete(n_actions)
        self.observation_space = spaces.Box(low=-1.0, high=100.0, shape=(6,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        # get new training file
        if len(self.env_supplier) == 0:

            if len(self.current_file_names) == 0:
                self.current_file_names = self.file_names.copy()

            random.shuffle(self.current_file_names)
            file_name = self.current_file_names.pop()

            with open(file_name, 'rb') as f:
                tracking = pickle.load(f)

            self.env_t = tracking['env_t']  # {t: [supplier_cost, customer_valuations, # active_suppliers, # customers, price, prob]}
            for t in range(len((self.env_t))):  # append prob of being selected
                self.env_t[t].append(min(1, self.env_t[t][3] / self.env_t[t][2]))
            self.env_g = tracking['env_g']  # patience, arrival_c/s, commission
            self.env_supplier = tracking['supplier']  # [[t_arriving, own_costs],...]

            self.max_t = len(self.env_t) - 1

        # get new supplier (= episode)
        random.shuffle(self.env_supplier)
        self.data_supplier = self.env_supplier.pop()

        # get state from data
        self.t, cost = self.data_supplier
        commission, patience = self.env_g[3], self.env_g[0]
        price = self.env_t[self.t][4]

        profit = price * (1 - commission) - cost
        potential = 1 / (patience + 1) if profit > 0 else 0

        self.state = [price, cost, profit, patience, potential, self.t]

        return np.array(self.state).astype(np.float32), {}

    def step(self, action):
        reward = 0
        terminated = False
        truncated = False

        if action == self.PARTICIPATE:
            prob = self.env_t[self.t][5]

            if random.random() < prob:
                c = random.choice(self.env_t[self.t][1])

                if c > self.state[0]:
                    terminated = True
                    reward = self.state[2] * 10

            self.state[3] -= 1
            self.state[5] += 1
            self.t += 1

        elif action == self.NOT:
            self.state[3] -= 1
            self.state[5] += 1
            self.t += 1

            commission = self.env_g[3]
            if self.t < self.max_t:
                price = self.env_t[self.t][4]
            else:
                price = self.state[0]

            cost = self.data_supplier[1]
            profit = price * (1 - commission) - cost
            potential = 1 / (self.state[3] + 1) if profit > 0 else 0

            self.state[0] = price
            self.state[2] = profit
            self.state[4] = potential

        else:
            raise ValueError('`action` should be 0 or 1.')

        # Termination and truncation
        if self.state[3] <= 0:
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

    def close(self):
        pass


class CentralizedEnvEvaluation(gym.Env):

    metadata = {"render_modes": ["console"]}

    # Define action (0 = do not participate in market; 1 = participate in market)
    NOT = 0
    PARTICIPATE = 1

    def __init__(self, mode, name):
        super(CentralizedEnvEvaluation, self).__init__()

        self.file_names = glob.glob(f'drl_data/{mode}_data/{name}_*.pkl')
        self.current_file_names = self.file_names.copy()

        self.env_t, self.env_g = {}, {}
        self.env_supplier, self.state = [], []
        self.t, self.max_t = 0, 0

        # Define action and observation space
        n_actions = 2
        self.action_space = spaces.Discrete(n_actions)
        self.observation_space = spaces.Box(low=-1.0, high=100.0, shape=(6,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        # get new training file
        if len(self.env_supplier) == 0:

            if len(self.current_file_names) == 0:
                self.current_file_names = self.file_names.copy()

            #random.shuffle(self.current_file_names)
            file_name = self.current_file_names.pop()

            with open(file_name, 'rb') as f:
                tracking = pickle.load(f)

            self.env_t = tracking['env_t']  # {t: [supplier_cost, customer_valuations, # active_suppliers, # customers, price, prob]}
            for t in range(len((self.env_t))):  # append prob of being selected
                self.env_t[t].append(min(1, self.env_t[t][3] / self.env_t[t][2]))
            self.env_g = tracking['env_g']  # patience, arrival_c/s, commission
            self.env_supplier = tracking['supplier']  # [[t_arriving, own_costs],...]

            self.max_t = len(self.env_t) - 1

        # get new supplier (= episode)
        #random.shuffle(self.env_supplier)
        self.data_supplier = self.env_supplier.pop()

        # get state from data
        self.t, cost = self.data_supplier
        commission, patience = self.env_g[3], self.env_g[0]
        price = self.env_t[self.t][4]

        profit = price * (1 - commission) - cost
        potential = 1 / (patience + 1) if profit > 0 else 0

        self.state = [price, cost, profit, patience, potential, self.t]

        return np.array(self.state).astype(np.float32), {}

    def step(self, action):

        reward = 0
        terminated = False
        truncated = False

        if action == self.PARTICIPATE:
            prob = self.env_t[self.t][5]

            if random.random() < prob:
                c = random.choice(self.env_t[self.t][1])

                if c > self.state[0]:
                    terminated = True
                    reward = self.state[2] * 10

            self.state[3] -= 1
            self.state[5] += 1
            self.t += 1

        elif action == self.NOT:
            self.state[3] -= 1
            self.state[5] += 1
            self.t += 1

            commission = self.env_g[3]
            if self.t < self.max_t:
                price = self.env_t[self.t][4]
            else:
                price = self.state[0]

            cost = self.data_supplier[1]
            profit = price * (1 - commission) - cost
            potential = 1 / (self.state[3] + 1) if profit > 0 else 0

            self.state[0] = price
            self.state[2] = profit
            self.state[4] = potential

        else:
            raise ValueError('`action` should be 0 or 1.')

        # Termination and truncation
        if self.state[3] <= 0:
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

    def close(self):
        pass


class CentralizedEnvTesting(gym.Env):

    metadata = {"render_modes": ["console"]}

    # Define action (0 = do not participate in market; 1 = participate in market)
    NOT = 0
    PARTICIPATE = 1

    def __init__(self, mode, name):
        super(CentralizedEnvTesting, self).__init__()

        self.file_names = glob.glob(f'drl_data/{mode}_data_stable/{name}_*.pkl')
        self.current_file_names = self.file_names.copy()

        self.env_t, self.env_g = {}, {}
        self.env_supplier, self.state = [], []
        self.t, self.max_t = 0, 0

        # Define action and observation space
        n_actions = 2
        self.action_space = spaces.Discrete(n_actions)
        self.observation_space = spaces.Box(low=-1.0, high=100.0, shape=(6,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        # get new training file
        if len(self.env_supplier) == 0:

            if len(self.current_file_names) == 0:
                self.current_file_names = self.file_names.copy()

            #random.shuffle(self.current_file_names)
            file_name = self.current_file_names.pop()

            with open(file_name, 'rb') as f:
                tracking = pickle.load(f)

            self.env_t = tracking['env_t']  # {t: [supplier_cost, customer_valuations, # active_suppliers, # customers, price]}
            self.env_g = tracking['env_g']  # patience, arrival_c/s, commission
            self.env_supplier = tracking['supplier']  # [[t_arriving, own_costs],...]

            self.max_t = len(self.env_t) - 1

        # get new supplier (= episode)
        #random.shuffle(self.env_supplier)
        self.data_supplier = self.env_supplier.pop()

        # get state from data
        self.t, cost = self.data_supplier[0], self.data_supplier[1]
        commission, patience = self.env_g[3], self.env_g[0]
        price = self.env_t[self.t][4]

        profit = price * (1 - commission) - cost
        potential = 1 / (patience + 1) if profit > 0 else 0

        self.state = [price, cost, profit, patience, potential, self.t]

        return np.array(self.state).astype(np.float32), {}

    def step(self, action):

        reward = 0
        terminated = False
        truncated = False

        if action == self.PARTICIPATE:
            prob = min(1, self.env_t[self.t][3] / self.env_t[self.t][2])
            p = self.env_g[0] - self.state[3] + 2

            if self.data_supplier[p][0] < prob:  # TODO
                c = self.data_supplier[p][1]

                if c > self.state[0]:
                    terminated = True
                    reward = self.state[2] * 10

            self.state[3] -= 1
            self.state[5] += 1
            self.t += 1

        elif action == self.NOT:
            self.state[3] -= 1
            self.state[5] += 1
            self.t += 1

            commission = self.env_g[3]
            if self.t < self.max_t:
                price = self.env_t[self.t][4]
            else:
                price = self.state[0]

            cost = self.data_supplier[1]
            profit = price * (1 - commission) - cost
            potential = 1 / (self.state[3] + 1) if profit > 0 else 0

            self.state[0] = price
            self.state[2] = profit
            self.state[4] = potential

        else:
            raise ValueError('`action` should be 0 or 1.')

        # Termination and truncation
        if self.state[3] <= 0:
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

    def close(self):
        pass

