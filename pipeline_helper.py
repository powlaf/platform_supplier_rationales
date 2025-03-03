import os
import json
import warnings
import subprocess
import numpy as np
import pandas as pd
from stable_baselines3 import DQN, SAC
warnings.filterwarnings("ignore")


# see how many timesteps each algo trained and when the last model was saved
def check_training_steps():

    data = []
    for name in [i for i in os.listdir('sb3/logs') if i[0] != '.']:
        if name[0] == 'c':
            model = DQN.load(f"./sb3/logs/{name}/best_model")
        else:
            model = SAC.load(f"./sb3/logs/{name}/best_model")
        print(name, model._total_timesteps, model.num_timesteps)

        data.append({'env_name': name, 'total_training_steps': model._total_timesteps,
                'model_saved_steps': model.num_timesteps})

    df = pd.DataFrame(data, columns=['env_name', 'total_training_steps', 'model_saved_steps'])

    return df


# get number of entries per environment
def get_generation_info():
    data = []
    for env in list(set(['_'.join(i.split('_')[0:3]) for i in os.listdir('./drl_data/training_data') if i[-4:]=='.pkl'])):
        no_training = len([i for i in os.listdir('./drl_data/training_data') if '_'.join(i.split('_')[0:3]) == env])
        no_validation = len([i for i in os.listdir('./drl_data/validation_data') if '_'.join(i.split('_')[0:3]) == env])
        no_testing = len([i for i in os.listdir('./drl_data/testing_data_stable') if '_'.join(i.split('_')[0:3]) == env])
        data.append((env, no_training, no_validation, no_testing))

    data = sorted(data, key=lambda x: x[0])
    max_key_length = max(len(row[0]) for row in data)

    for row in data:
        print(f"{row[0]:<{max_key_length}} {row[1]:>3} {row[2]:>2} {row[3]:>2}")

    df = pd.DataFrame(data, columns=['env_name', 'training', 'validation', 'testing'])
    df['price_setter'] = [x.split('_')[0] for x in df.env_name]
    df['patience'] = [x.split('_')[1][1:] for x in df.env_name]
    df['sbratio'] = [x.split('_')[2][1:] for x in df.env_name]

    return df


# pop up and sound when pipe step is finished
def notify(type, env_name):

    say = {
        'data': 'Data generation finished',
        'train': 'Training finished',
        'test': 'Testing finished',
        'sim': 'Simulation runs finished'
    }

    os.system(f'say "{say[type]}"')

    script = {
        'data': f"""
        display dialog "Data generation finished" ¬
        with title "{env_name}" ¬
        with icon caution ¬
        buttons {{"OK"}}
        """,
        'train': f"""
        display dialog "Algo finished training" ¬   
        with title "{env_name}" ¬
        with icon caution ¬
        buttons {{"OK"}}
        """,
        'test': f"""
        display dialog "Testing finished - copy results to test_results.txt" ¬
        with title "{env_name}" ¬
        with icon caution ¬
        buttons {{"OK"}}
        """,
        'sim': f"""
        display dialog "Simulation finished" ¬
        with title "{env_name}" ¬
        with icon caution ¬
        buttons {{"OK"}}
        """,
    }

    subprocess.call("osascript -e '{}'".format(script[type]), shell=True)

    print(type, env_name)


# save a single test run
def save_testing_results(env_name, decision_rationale, returns, terminations):

    file_path = f'sb3/testing/data/data_testing.json'

    data = [{
        'env_name': env_name,
        'decision_rationale': decision_rationale,
        'returns': returns,
        'terminations': terminations,
        'returns_sum': sum(returns),
        'returns_avg': np.mean(returns),
        'returns_std': np.std(returns),
        'terminations_pos': sum(terminations),
        'terminations_total': len(terminations),
        'terminations_mean': np.mean(terminations)
    }]

    # append to existing file
    if os.path.exists(file_path):
        with open(file_path, "r") as json_file:
            existing_data = json.load(json_file)
        existing_data.extend(data)
        with open(file_path, "w") as json_file:
            json.dump(existing_data, json_file, indent=4)

    # create new file
    else:
        with open(file_path, "w") as json_file:
            json.dump(data, json_file, indent=4)


# return a simulation view for one environment
def view_simulation(env_name):
    # load data
    data_file_names = os.listdir('./simulations/data/')
    data_file_names = [i for i in data_file_names if i.endswith('.json')]

    file_path = f'simulations/data/{env_name}.json'

    with open(file_path, "r") as json_file:
        loaded_data = json.load(json_file)

    # view data as df
    df = pd.DataFrame(loaded_data)
    temp_df = pd.json_normalize(df['platform_params'])
    df = pd.concat([df, temp_df], axis=1).drop(columns=['platform_params'])

    temp_df = pd.json_normalize(df['results'])
    df = pd.concat([df, temp_df], axis=1).drop(columns=['results'])

    # readjust visualization
    normal_cols = ['timestamp', 'platform_type', 'decision_rationale', 'timesteps', 'patience', 'arrival_customers',
                   'arrival_suppliers', 'commission', 'p_range']
    welfare_cols = ['welfare_total', 'welfare_supplier', 'welfare_customer', 'welfare_platform', 'costs', 'valuations',
                    'revenue', 'matches', 'total_customers', 'active_customers', 'total_suppliers', 'active_suppliers']

    df.columns = normal_cols + ['t_' + i for i in welfare_cols]

    for col in welfare_cols:
        df[col] = [sum(i) / len(i) for i in df['t_' + col]]

    df_wot = df[normal_cols + [i for i in welfare_cols]]

    return df_wot


# return and save testing results as csv
def view_testing():
    # load data
    file_path = f'sb3/testing/data/data_testing.json'

    with open(file_path, "r") as json_file:
        loaded_data = json.load(json_file)

    # view data as df
    df = pd.DataFrame(loaded_data)
    df.drop_duplicates(subset=['env_name', 'decision_rationale'], keep='last', inplace=True)

    sorted(list(set(df.env_name)))

    df.to_csv('sb3/testing/data/data_testing.csv')

    return df


# return and save full and aggregated simulation results
def simulation_dataframe():
    # create dataframe with aggregated data (one row per environment per decision_rationale)
    all_cols = ['welfare_total', 'welfare_supplier', 'welfare_customer', 'welfare_platform', 'costs', 'valuations',
                'revenue', 'matches', 'total_customers', 'active_customers', 'total_suppliers', 'active_suppliers']
    df_total_agg = pd.DataFrame(columns=['env_name', 'decision_rationale', 'runs'] + all_cols)

    env_names = [x for x in os.listdir('simulations/data') if x[-5:] == '.json']
    for env_name in env_names:
        env_name = env_name[0:len(env_name) - 5]
        df = view_simulation(env_name)

        # only keep last 10 rows of each combination (to drop earlier runs)
        df = df.groupby(['decision_rationale'], group_keys=False).tail(10)

        over = df.groupby('decision_rationale')[all_cols].mean()
        counts = df.groupby('decision_rationale').size()
        over = over.join(counts.rename('runs'))

        over = over.reset_index()
        over['env_name'] = env_name

        df_total_agg = pd.concat([df_total_agg, over], ignore_index=True)

    # save aggregated simulation df
    df_total_agg.to_csv('simulations/data/simulation_data_agg.csv')

    # create dataframe with full data (10 rows per environment per decision_rationale)
    all_cols = ['welfare_total', 'welfare_supplier', 'welfare_customer', 'welfare_platform', 'costs', 'valuations',
                'revenue', 'matches', 'total_customers', 'active_customers', 'total_suppliers', 'active_suppliers']
    df_total_full = pd.DataFrame(columns=['env_name', 'decision_rationale'] + all_cols)

    env_names = [x for x in os.listdir('simulations/data') if x[-5:] == '.json']
    for env_name in env_names:
        env_name = env_name[0:len(env_name) - 5]
        df = view_simulation(env_name)

        df = df[['decision_rationale'] + all_cols]
        df['env_name'] = env_name
        df_total_full = pd.concat([df_total_full, df], ignore_index=True)

    # save full simulation df
    df_total_full.to_csv('simulations/data/simulation_data_full.csv')

    return df_total_agg, df_total_full

