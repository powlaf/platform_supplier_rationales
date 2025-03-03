from drl_data.data_generation.data_generation import generate_data, create_stable_testing_data

from sb3.algos.sac_decentralized import train_decentralized
from sb3.algos.dqn_centralized import train_centralized
from sb3.algos.sac_hybrid import train_hybrid

from sb3.testing.test_decentralized import test_decentralized
from sb3.testing.test_centralized import test_centralized
from sb3.testing.test_hybrid import test_hybrid

from simulations.simulation import simulate

from results.hypothesis_testing.hypothesis_testing import create_hypothesis_test_testing, \
    create_hypothesis_test_simulation

from pipeline_helper import get_generation_info, notify, check_training_steps, view_simulation, view_testing, \
    simulation_dataframe


# 0. Parameters and Environment Definition -----------------------------------------------------------------------------

for supplier_buyer_ration in [0.1]:
    for patience in [1]:
        for platform_type in ['centralized']: #,  'decentralized', 'hybrid']:
            #patience = 10  # 1, 5, 10, 50, 100
            #supplier_buyer_ration = 1  # 0.1, 0.5, 1.0, 2.0, 10.0
            #platform_type = 'centralized'  # decentralized, hybrid
            timesteps = 100
            arrival_suppliers = 100
            arrival_customers = int(arrival_suppliers / supplier_buyer_ration)
            commission = 0.1
            p_range = 0.1

            env_name = f'{platform_type}_p{patience}_r{supplier_buyer_ration}'
            print('ENV_NAME', env_name)

            runs_heuristic = 10
            simulate(env_name, platform_type, 'heuristic', runs_heuristic, patience, arrival_customers, arrival_suppliers,
                     timesteps, commission, p_range)
            runs_drl = 10
            simulate(env_name, platform_type, 'drl', runs_drl, patience, arrival_customers, arrival_suppliers, timesteps,
                     commission, p_range)


# 1. Data Generation ---------------------------------------------------------------------------------------------------

# training
no_train_episodes = 20
generate_data(env_name, 'training', platform_type, no_train_episodes, patience, arrival_customers, arrival_suppliers,
              timesteps, commission, p_range)
# validation
no_eval_episodes = 5
generate_data(env_name, 'validation', platform_type, no_eval_episodes, patience, arrival_customers, arrival_suppliers,
              timesteps, commission, p_range)
# testing
no_test_episodes = 5
generate_data(env_name, 'testing', platform_type, no_test_episodes, patience, arrival_customers, arrival_suppliers,
              timesteps, commission, p_range)
# stable testing (only run once)
create_stable_testing_data(platform_type)

#notify('data', env_name)
df_gen_info = get_generation_info()


# 2. Algo Training -----------------------------------------------------------------------------------------------------

train_timesteps = 300_000
seed = 0

algo_train = {'centralized': train_centralized,
              'hybrid': train_hybrid,
              'decentralized': train_decentralized
              }
algo_train[platform_type](env_name, train_timesteps, seed)
#notify('train', env_name)

df_train_steps = check_training_steps()


# 3. Algo Testing ------------------------------------------------------------------------------------------------------
## one supplier behaves differently

algo_test = {'centralized': test_centralized,
             'hybrid': test_hybrid,
             'decentralized': test_decentralized
             }
algo_test[platform_type](env_name)
#notify('test', env_name)

# return and save testing data
df_testing = view_testing()


# 4. Simulating --------------------------------------------------------------------------------------------------------
## all suppliers behave differently

runs_heuristic = 10
simulate(env_name, platform_type, 'heuristic', runs_heuristic, patience, arrival_customers, arrival_suppliers,
         timesteps, commission, p_range)
runs_drl = 10
simulate(env_name, platform_type, 'drl', runs_drl, patience, arrival_customers, arrival_suppliers, timesteps,
         commission, p_range)
#notify('sim', env_name)

# return data for one simulation
df_simulation_env = view_simulation(env_name)
# return and save full simulation data
df_simulation_agg, df_simulation_full = simulation_dataframe()


# 5. Hypothesis testing ------------------------------------------------------------------------------------------------

# hypothesis testing
df_hypothesis_testing = create_hypothesis_test_testing()
df_hypothesis_simulation = create_hypothesis_test_simulation()

