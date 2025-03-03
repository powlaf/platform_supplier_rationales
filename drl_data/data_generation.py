import glob
import pickle
import random
from simulations.classes import Customer, Supplier
from drl_data.data_generation.data_generation_hybrid import hybrid_data_generation
from drl_data.data_generation.data_generation_centralized import centralized_data_generation
from drl_data.data_generation.data_generation_decentralized import decentralized_data_generation


def generate_data(env_name, mode, price_setting_entity, no_episodes, patience, arrival_customers, arrival_suppliers,
                  timesteps, commission, p_range):
    printl = False

    for e in range(no_episodes):

        print(f'-- {mode} data generation for {env_name} in episode {e}')

        # run one episode
        if price_setting_entity == 'decentralized':
            tracking = decentralized_data_generation(patience, arrival_customers, arrival_suppliers, timesteps, commission, printl)
        elif price_setting_entity == 'centralized':
            tracking = centralized_data_generation(patience, arrival_customers, arrival_suppliers, timesteps, commission, printl)
        elif price_setting_entity == 'hybrid':
            tracking = hybrid_data_generation(patience, arrival_customers, arrival_suppliers, timesteps, commission, p_range, printl)
        else:
            AttributeError('Please choose one of the following price_setting_entities: centralized, hybrid or decentralized!')

        # reset ids
        Supplier.reset()
        Customer.reset()

        # print
        if printl:
            for key, value in tracking.items():
                print(key, value)
                break

        # write to file
        file_name = int(round(random.random()*10000000000,0))
        print(file_name)

        with open(f'drl_data/{mode}_data/{env_name}_{file_name}.pkl', 'wb') as f:
            pickle.dump(tracking, f)


def create_stable_testing_data_centralized():
    price_setter = 'centralized'
    mode = 'testing'

    file_names = glob.glob(f'drl_data/{mode}_data/{price_setter}_*.pkl')
    for file_name in file_names:
        # file_name = file_names.pop()
        with open(file_name, 'rb') as f:
            tracking = pickle.load(f)

        env_t = tracking['env_t']  # { t: supplier_cost, customer_valuations, # suppliers, # customers }
        for t in range(len(env_t)):  # append prob of being selected
            env_t[t].append(min(1, env_t[t][3] / env_t[t][2]))

        env_g = tracking['env_g']  # patience, arrival_c/s, commission

        env_supplier = tracking['supplier']  # [t_arriving, own_costs]
        for supp in range(len(env_supplier)):
            for p in range(env_g[0]):
                c = random.choice(env_t[env_supplier[supp][0]][1])
                env_supplier[supp].append([random.random(), c])

        new_tracking = {
            'env_t': env_t,
            'env_g': env_g,
            'supplier': env_supplier,
            'totals': tracking['totals']
        }

        with open(f'{file_name.split("/")[0]}/{mode}_data_stable/{file_name.split("/")[2]}', 'wb') as f:
            pickle.dump(new_tracking, f)


def create_stable_testing_data_decentralized():
    price_setter = 'decentralized'
    mode = 'testing'

    file_names = glob.glob(f'drl_data/{mode}_data/{price_setter}_*.pkl')
    for file_name in file_names:
        #file_name = file_names.pop()
        with open(file_name, 'rb') as f:
            tracking = pickle.load(f)

        env_t = tracking['env_t']  # { t: supplier_cost, customer_valuations, # suppliers, # customers }
        for t in range(len(env_t)):  # append prob of being selected
            env_t[t].append(min(1, env_t[t][3] / env_t[t][2]))

        env_g = tracking['env_g']  # patience, arrival_c/s, commission

        env_supplier = tracking['supplier']  # [t_arriving, own_costs]
        for supp in range(len(env_supplier)):
            for p in range(env_g[0]):
                c = random.choice(env_t[env_supplier[supp][0]][1])
                env_supplier[supp].append([random.random(), c])

        new_tracking = {
            'env_t': env_t,
            'env_g': env_g,
            'supplier': env_supplier,
            'totals': tracking['totals']
        }

        with open(f'{file_name.split("/")[0]}/{mode}_data_stable/{file_name.split("/")[2]}', 'wb') as f:
            pickle.dump(new_tracking, f)


def create_stable_testing_data_hybrid():
    price_setter = 'hybrid'
    mode = 'testing'

    file_names = glob.glob(f'drl_data/{mode}_data/{price_setter}_*.pkl')
    for file_name in file_names:
        print(file_name)
        #file_name = file_names.pop()
        with open(file_name, 'rb') as f:
            tracking = pickle.load(f)

        env_t = tracking['env_t']  # { t: supplier_cost, customer_valuations, # suppliers, # customers }
        for t in range(len(env_t)):  # append prob of being selected
            env_t[t].append(min(1, env_t[t][3] / env_t[t][2]))

        env_g = tracking['env_g']  # patience, arrival_c/s, commission

        env_supplier = tracking['supplier']  # [t_arriving, own_costs]
        for supp in range(len(env_supplier)):
            for p in range(env_g[0]):
                t = env_supplier[supp][0]
                if len(env_t[t][1]) > 0:
                    c = random.choice(env_t[t][1])
                else:
                    c = 0
                env_supplier[supp].append([random.random(), c])

        new_tracking = {
            'env_t': env_t,
            'env_g': env_g,
            'supplier': env_supplier,
            'totals': tracking['totals']
        }

        with open(f'{file_name.split("/")[0]}/{mode}_data_stable/{file_name.split("/")[2]}', 'wb') as f:
            pickle.dump(new_tracking, f)


def create_stable_testing_data(price_setter):
    if price_setter == 'centralized':
        create_stable_testing_data_centralized()
    elif price_setter == 'hybrid':
        create_stable_testing_data_hybrid()
    elif price_setter == 'decentralized':
        create_stable_testing_data_decentralized()

