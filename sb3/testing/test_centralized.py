import numpy as np

from stable_baselines3 import DQN

from sb3.environments.env_centralized import CentralizedEnvTesting
from pipeline_helper import save_testing_results


def test_centralized(name, printl=False):

    # 1. Create environment --------------------------------------------------------------------------------------------
    mode = 'testing'

    test_env_drl = CentralizedEnvTesting(mode, name)
    test_env_heuristic = CentralizedEnvTesting(mode, name)
    test_env_optimal = CentralizedEnvTesting(mode, name)

    # 2. Run drl env ---------------------------------------------------------------------------------------------------
    model = DQN.load(f"./sb3/logs/{name}/best_model")

    returns_drl = []
    terminations_drl = []
    for _ in range(5):
        for e in range(9999):
            obs, info = test_env_drl.reset()
            done = False

            while not done:
                action = model.predict(obs)[0]
                obs, reward, terminated, truncated, info = test_env_drl.step(action)

                # Check if the episode has ended
                done = terminated or truncated

                if done:
                    returns_drl.append(reward)
                    terminations_drl.append(terminated)

    if printl:
        print(f'DRL results      : {round(sum(returns_drl),2)} ({round(np.mean(returns_drl),2)} +- {round(np.std(returns_drl),2)})')
    save_testing_results(name, 'drl', returns_drl, terminations_drl)
    #mean_reward, std_reward = evaluate_policy(model, test_env_drl, n_eval_episodes=9999*5, deterministic=True)
    #print(f'DRL results      : {round(mean_reward*9999*5,2)} ({round(mean_reward,2)} +- {round(std_reward,2)})')

    # 3. Run heuristic env ---------------------------------------------------------------------------------------------

    returns_heuristic = []
    terminations_heuristic = []
    for i in range(5):
        for e in range(9999):
            obs, info = test_env_heuristic.reset()
            done = False

            while not done:
                if obs[2] > 0:
                    action = 1
                else:
                    action = 0
                obs, reward, terminated, truncated, info = test_env_heuristic.step(action)

                # Check if the episode has ended
                done = terminated or truncated

                if done:
                    returns_heuristic.append(reward)
                    terminations_heuristic.append(terminated)

    if printl:
        print(f'Heuristic results: {round(sum(returns_heuristic),2)} ({round(np.mean(returns_heuristic),2)} +- {round(np.std(returns_heuristic),2)})')
    save_testing_results(name, 'heuristic', returns_heuristic, terminations_heuristic)

    # 4. Run optimal env ---------------------------------------------------------------------------------------------------

    returns_optimal = []
    terminations_optimal = []
    for _ in range(5):
        for e in range(9999):
            #print(e)
            obs, info = test_env_optimal.reset()
            done = False
            max_profit = 0
            while not done:
                max_profit = max(max_profit, obs[2])
                action = 0
                obs, reward, terminated, truncated, info = test_env_optimal.step(action)

                # Check if the episode has ended
                done = terminated or truncated

                if done:
                    returns_optimal.append(max_profit * 10)
                    if max_profit > 0:
                        terminations_optimal.append(True)
                    else:
                        terminations_optimal.append(False)

    if printl:
        print(f'Optimal results  : {round(sum(returns_optimal),2)} ({round(np.mean(returns_optimal),2)} +- {round(np.std(returns_optimal),2)})')
    save_testing_results(name, 'optimal', returns_optimal, terminations_optimal)
