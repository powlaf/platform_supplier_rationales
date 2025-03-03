import os
from tqdm import tqdm

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.env_util import make_vec_env

from sb3.environments.env_decentralized import DecentralizedEnvTraining, DecentralizedEnvEvaluation


def train_decentralized(env_name, total_timesteps, seed):
    # hyperparameters
    n_training_envs = 1
    n_eval_envs = 5

    # check environment
    # env = DecentralizedEnvTraining()
    # check_env(env, warn=True)

    # log dir for evaluation
    log_dir = f"./sb3/logs/{env_name}/"
    model_path = os.path.join(log_dir, "best_model.zip")
    os.makedirs(log_dir, exist_ok=True)

    # train and evaluation environment
    train_env = make_vec_env(DecentralizedEnvTraining, n_envs=n_training_envs, seed=seed, monitor_dir=log_dir,
                             env_kwargs={'mode': 'training', 'name': env_name})
    eval_env = make_vec_env(DecentralizedEnvEvaluation, n_envs=n_eval_envs, seed=seed,
                            env_kwargs={'mode': 'training', 'name': env_name})

    # load or create model
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")

        log_dir = f"./sb3/logs/{env_name}_2/"
        os.makedirs(log_dir, exist_ok=True)

        train_env = make_vec_env(DecentralizedEnvTraining, n_envs=n_training_envs, seed=seed, monitor_dir=log_dir,
                                 env_kwargs={'mode': 'training', 'name': env_name})

        model = SAC.load(model_path, env=train_env)

    else:
        print("No existing model found. Training from scratch.")
        model = SAC('MlpPolicy', train_env)

    # callback for model saving and progress bar
    eval_callback = EvalCallback(eval_env, best_model_save_path=log_dir, log_path=log_dir,
                                 eval_freq=max(5_000 // n_training_envs, 1), n_eval_episodes=9999*5,
                                 deterministic=True, render=False)

    class ProgressBarCallback(BaseCallback):
        def __init__(self, total_timesteps):
            super().__init__()
            self.total_timesteps = total_timesteps
            self.pbar = None

        def _on_training_start(self):
            self.pbar = tqdm(total=self.total_timesteps, desc="Training Progress")

        def _on_step(self):
            self.pbar.update(self.model.n_envs)  # Update progress
            return True  # Continue training

        def _on_training_end(self):
            self.pbar.close()  # Close the progress bar

    # create and train model
    model.learn(total_timesteps, callback=[eval_callback, ProgressBarCallback(total_timesteps)])

