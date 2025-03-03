import os
import json
from datetime import datetime

from simulations.sim_hybrid import hybrid_heuristic, hybrid_drl
from simulations.sim_centralized import centralized_heuristic, centralized_drl
from simulations.sim_decentralized import decentralized_heuristic, decentralized_drl

def simulate(env_name, platform_type, decision_rationale, runs, patience, arrival_customers, arrival_suppliers, timesteps, commission, p_range):

    printl = False
    ## running -------------------------------------------------------------------------------------------------------------
    for run in range(runs):
        print(f'-- Simulation {decision_rationale} for {env_name} in run {run}')

        '''    
        if platform_type == 'centralized':
            if decision_rationale == 'heuristic':
                totals = centralized_heuristic(patience, arrival_customers, arrival_suppliers, timesteps, commission, printl=printl)
            elif decision_rationale == 'drl':
                totals = centralized_drl(patience, arrival_customers, arrival_suppliers, timesteps, commission, env_name, printl=printl)
            else:
                ValueError('Choose one of the following decision rationales: heuristic or drl')
    
        elif platform_type == 'hybrid':
            if decision_rationale == 'heuristic':
                totals = hybrid_heuristic(patience, arrival_customers, arrival_suppliers, timesteps, commission, p_range, printl=printl)
            elif decision_rationale == 'drl':
                totals = hybrid_drl(patience, arrival_customers, arrival_suppliers, timesteps, commission, p_range, printl=printl)
            else:
                ValueError('Choose one of the following decision rationales: heuristic or drl')
    
        elif platform_type == 'decentralized':
            if decision_rationale == 'heuristic':
                totals = hybrid_heuristic(patience, arrival_customers, arrival_suppliers, timesteps, commission, p_range, printl=printl)
            elif decision_rationale == 'drl':
                totals = hybrid_drl(patience, arrival_customers, arrival_suppliers, timesteps, commission, p_range, printl=printl)
            else:
                ValueError('Choose one of the following decision rationales: heuristic or drl')
    
        else:
            ValueError('Choose one of the following modes: centralized, hybrid, or decentralized')
            totals = {}
        '''
        # function mapping for cleaner code
        decision_functions = {
            "centralized": {
                "heuristic": centralized_heuristic,
                "drl": centralized_drl
            },
            "hybrid": {
                "heuristic": hybrid_heuristic,
                "drl": hybrid_drl
            },
            "decentralized": {
                "heuristic": decentralized_heuristic,
                "drl": decentralized_drl
            }
        }

        # validate inputs
        if platform_type not in decision_functions:
            raise ValueError("Choose one of the following modes: centralized, hybrid, or decentralized")
        if decision_rationale not in decision_functions[platform_type]:
            raise ValueError("Choose one of the following decision rationales: heuristic or drl")

        # get simulation function and parameters
        decision_function = decision_functions[platform_type][decision_rationale]
        params = [patience, arrival_customers, arrival_suppliers, timesteps, commission, printl]
        if platform_type in ["hybrid"]:
            params.insert(-1, p_range)
        if decision_rationale in ["drl"]:
            params.append(env_name)

        # run function
        totals = decision_function(*params)

        ## save results ----------------------------------------------------------------------------------------------------
        file_path = f'simulations/data/{env_name}.json'

        data = [{
            'timestamp': datetime.now().strftime("%Y%m%d-%H%M%S"),
            'platform_type': platform_type,
            'decision_rationale': decision_rationale,
            'platform_params': {
                'timesteps': timesteps,
                'patience': patience,
                'arrival_customers': arrival_customers,
                'arrival_suppliers': arrival_suppliers,
                'commission': commission,
                'p_range': p_range,
            },
            'results': totals
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

