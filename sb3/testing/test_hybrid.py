import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy

from sb3.environments.env_hybrid import HybridEnvTesting
from pipeline_helper import save_testing_results


def test_hybrid(name):

    # 1. Create environment --------------------------------------------------------------------------------------------
    mode = 'testing'

    test_env_drl = HybridEnvTesting(mode, name)
    test_env_heuristic = HybridEnvTesting(mode, name)
    test_env_optimal = HybridEnvTesting(mode, name)

    # 2. Run drl env ---------------------------------------------------------------------------------------------------

    model = SAC.load(f"./sb3/logs/{name}/best_model")

    returns_drl = []
    terminations_drl = []
    for _ in range(5):
        for e in range(9999):

            obs, info = test_env_drl.reset()
            done = False
            # obs = [cost, min_price, patience, self.t]

            while not done:
                action = model.predict(obs)[0]
                obs, reward, terminated, truncated, info = test_env_drl.step(action)

                # Check if the episode has ended
                done = terminated or truncated

                if done:
                    returns_drl.append(reward)
                    terminations_drl.append(terminated)
                    break

    print(f'DRL results      : {round(sum(returns_drl), 2)} ({round(np.mean(returns_drl), 2)} +- {round(np.std(returns_drl), 2)})')
    save_testing_results(name, 'drl', returns_drl, terminations_drl)

    #mean_reward, std_reward = evaluate_policy(model, test_env_drl, n_eval_episodes=9999*5, deterministic=True)
    #print(f'DRL results      : {round(mean_reward*9999*5,2)} ({round(mean_reward,2)} +- {round(std_reward,2)})')

    # 3. Run heuristic env ---------------------------------------------------------------------------------------------
    ## Price = min_cost

    returns_heuristic = []
    terminations_heuristic = []
    for _ in range(5):
        for e in range(9999):

            obs, info = test_env_heuristic.reset()
            done = False
            # obs = [cost, min_price, pmin, pmax, patience, self.t]

            while not done:
                if obs[2] <= obs[1] <= obs[3]:
                    action = obs[1]
                elif obs[1] < obs[2]:
                    action = obs[2]
                else:
                    action = 1
                obs, reward, terminated, truncated, info = test_env_heuristic.step(action)

                # Check if the episode has ended
                done = terminated or truncated

                if done:
                    returns_heuristic.append(reward)
                    terminations_heuristic.append(terminated)
                    break

    print(f'Heuristic results: {round(sum(returns_heuristic),2)} ({round(np.mean(returns_heuristic),2)} +- {round(np.std(returns_heuristic),2)})')
    save_testing_results(name, 'heuristic', returns_heuristic, terminations_heuristic)

    # 4. Run semi-optimal env ------------------------------------------------------------------------------------------
    ## Price = min_cost + 10%

    returns_optimal = []
    terminations_optimal = []
    for _ in range(5):
        for e in range(9999):

            obs, info = test_env_optimal.reset()
            done = False
            # obs = [cost, min_price, patience, self.t]

            while not done:
                action = obs[1] * 1.1
                obs, reward, terminated, truncated, info = test_env_optimal.step(action)

                # Check if the episode has ended
                done = terminated or truncated

                if done:
                    returns_optimal.append(reward)
                    terminations_heuristic.append(terminated)
                    break

    print(f'Semi-optimal results: {round(sum(returns_optimal),2)} ({round(np.mean(returns_optimal),2)} +- {round(np.std(returns_optimal),2)})')
    save_testing_results(name, 'semi-optimal', returns_optimal, terminations_optimal)

