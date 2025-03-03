import time
import gurobipy as gp
import pandas as pd
from gurobipy import GRB
import os
import csv


def optimize_centralized(costs, valuations, no_supplier, no_buyer, commission):
    start_time = time.time()

    S = len(no_supplier)
    B = len(no_buyer)

    # model
    m = gp.Model()
    m.setParam('OutputFlag', 0)

    # parameters
    x = m.addVars(S, B, vtype=GRB.CONTINUOUS, lb=0, ub=1)
    p = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1)
    z_s = m.addVars(S, B, vtype=GRB.BINARY)
    z_b = m.addVars(S, B, vtype=GRB.BINARY)

    # objective
    m.setObjective(gp.quicksum(x[s, b] for s in range(S) for b in range(B)) * p * commission, GRB.MAXIMIZE)

    # constraints
    M = 1e6
    for s in range(S):
        for b in range(B):
            m.addConstr(x[s, b] <= (no_buyer[b] * no_supplier[s]) / (sum(no_buyer) * sum(no_supplier)))

            m.addConstr(x[s, b] <= z_s[s, b] * M)
            m.addConstr(x[s, b] <= z_b[s, b] * M)

            m.addConstr(costs[s] - p * (1 - commission) <= M * (1 - z_s[s, b]))
            m.addConstr(p - valuations[b] <= M * (1 - z_b[s, b]))

    # solve
    m.optimize()

    # track time
    elapsed_time = time.time() - start_time
    print(f"Time elapsed for optimization model: {elapsed_time:.6f} seconds")

    return p.x


'''
costs = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # 0.1*i for i in range(0,11)]
valuations = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
no_supplier = [10, 10, 10, 10, 20, 20, 20, 20, 20, 20, 20]  # [1 for _ in range(10)]
no_buyer = [10, 20, 20, 50, 50, 14, 1, 1, 1, 1, 1]  # 10, 11, 12, 13, 14, 14, 17, 18, 19, 10]
commission = 0.1

p = optimize_centralized(costs, valuations, no_supplier, no_buyer, commission)
'''

'''
def printing(p, B, S, x):
    print('Price: ', p.x)
    for b in range(B):
        print(f"Buyer {b} with valuation {valuations[b]}")
        print(f"c: {'  | '.join(['%.1f' % round(costs[s],2) for s in range(S)])}")
        print(f"x: {' | '.join(['%.2f' % round(x[s,b].x,2) for s in range(S)])}")

#printing(p, B, S, x)
'''


def optimize_hybrid(costs, valuations, no_supplier, no_buyer, commission, p_range):
    start_time = time.time()

    S = len(costs)
    B = len(valuations)

    # model
    m = gp.Model()
    m.setParam('OutputFlag', 0)

    # parameters
    x = m.addVars(S, B, vtype=GRB.CONTINUOUS, lb=0, ub=1)
    p = m.addVars(S, B, vtype=GRB.CONTINUOUS, lb=0, ub=1)
    pmin = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1)
    pmax = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1)
    z_s = m.addVars(S, B, vtype=GRB.BINARY)
    z_b = m.addVars(S, B, vtype=GRB.BINARY)
    z_pmin = m.addVars(S, B, vtype=GRB.BINARY)
    z_pmax = m.addVars(S, B, vtype=GRB.BINARY)

    # objective
    m.setObjective(
        gp.quicksum(x[s, b] * no_supplier[s] * no_buyer[b] * p[s, b] for s in range(S) for b in range(B)) * commission,
        GRB.MAXIMIZE)

    # constraints
    M = 1.01
    for s in range(S):
        for b in range(B):
            m.addConstr(x[s, b] <= (no_supplier[s] * no_buyer[b]) / (sum(no_supplier) * sum(no_buyer)))

            m.addConstr(x[s, b] <= z_s[s, b] * M)
            m.addConstr(x[s, b] <= z_b[s, b] * M)

            m.addConstr(costs[s] - p[s, b] * (1 - commission) <= M * (1 - z_s[s, b]))
            m.addConstr(p[s, b] - valuations[b] <= M * (1 - z_b[s, b]))

            m.addConstr(pmax - pmin == p_range)
            # m.addConstr(p[s, b] == costs[s] * (1 + commission))
            m.addConstr(x[s, b] <= z_pmin[s, b] * M)
            m.addConstr(x[s, b] <= z_pmax[s, b] * M)
            m.addConstr(pmin - p[s, b] <= M * (1 - z_pmin[s, b]))
            m.addConstr(p[s, b] - pmax <= M * (1 - z_pmax[s, b]))

    # solve
    m.optimize()

    # print('Price: ', pmin.x, pmax.x)

    # track time
    elapsed_time = time.time() - start_time
    print(f"Time elapsed for optimization model: {elapsed_time:.6f} seconds")

    return pmin.x, pmax.x


def optimize_hybrid_test(costs, valuations, no_supplier, no_buyer, commission, p_range, M):
    start_time = time.time()

    S = len(costs)
    B = len(valuations)

    # model
    m = gp.Model()
    m.setParam('OutputFlag', 0)

    # parameters
    x = m.addVars(S, B, vtype=GRB.CONTINUOUS, lb=0, ub=1)
    p = m.addVars(S, B, vtype=GRB.CONTINUOUS, lb=0, ub=1)
    pmin = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1)
    pmax = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1)
    z_s = m.addVars(S, B, vtype=GRB.BINARY)
    z_b = m.addVars(S, B, vtype=GRB.BINARY)
    z_pmin = m.addVars(S, B, vtype=GRB.BINARY)
    z_pmax = m.addVars(S, B, vtype=GRB.BINARY)

    # objective
    m.setObjective(
        gp.quicksum(x[s, b] * no_supplier[s] * no_buyer[b] * p[s, b] for s in range(S) for b in range(B)) * commission,
        GRB.MAXIMIZE)

    # constraints
    # M = 100
    for s in range(S):
        for b in range(B):
            m.addConstr(x[s, b] <= (no_supplier[s] * no_buyer[b]) / (sum(no_supplier) * sum(no_buyer)))

            m.addConstr(x[s, b] <= z_s[s, b] * M)
            m.addConstr(x[s, b] <= z_b[s, b] * M)

            m.addConstr(costs[s] - p[s, b] * (1 - commission) <= M * (1 - z_s[s, b]))
            m.addConstr(p[s, b] - valuations[b] <= M * (1 - z_b[s, b]))

            m.addConstr(pmax - pmin == p_range)
            # m.addConstr(p[s, b] == costs[s] * (1 + commission))
            m.addConstr(x[s, b] <= z_pmin[s, b] * M)
            m.addConstr(x[s, b] <= z_pmax[s, b] * M)
            m.addConstr(pmin - p[s, b] <= M * (1 - z_pmin[s, b]))
            m.addConstr(p[s, b] - pmax <= M * (1 - z_pmax[s, b]))

    # solve
    m.optimize()

    # print('Price: ', pmin.x, pmax.x)

    # track time
    elapsed_time = time.time() - start_time
    print(f"Time elapsed for optimization model: {elapsed_time:.6f} seconds")

    return elapsed_time, pmin.x, pmax.x


def test_runtime():
    file_path = 'time_trials_unbalanced_real_distrs_multiple_3.csv'

    distrs = [
        [[10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
         [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]],
        [[16.14, 16.14, 16.14, 16.14, 16.14, 16.14, 16.14, 20.0, 20.0, 20.0, 20.0],
         [20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 13.5, 13.17, 13.17, 13.17]],
        [[10.0, 10.0, 10.0, 10.0, 10.0, 25.48, 26.14, 30.0, 30.0, 30.0, 30.0],
         [30.0, 30.0, 30.0, 30.0, 30.0, 17.17, 14.77, 10.0, 10.0, 10.0, 10.0]],
        [[17.0, 17.0, 17.0, 17.0, 17.0, 32.48, 33.14, 37.0, 40.0, 40.0, 40.0],
         [40.0, 40.0, 40.0, 40.0, 40.0, 27.17, 24.77, 20.0, 13.0, 11.5, 11.5]],
        [[14.74, 14.74, 14.74, 14.74, 14.74, 30.21, 42.71, 47.0, 50.0, 50.0, 50.0],
         [50.0, 50.0, 50.0, 50.0, 50.0, 37.17, 21.57, 15.05, 10.0, 10.0, 10.0]],
        [[10.0, 10.0, 10.0, 22.99, 24.74, 40.21, 52.71, 57.0, 60.0, 60.0, 60.0],
         [60.0, 60.0, 60.0, 51.75, 49.75, 36.92, 21.32, 14.8, 10.0, 10.0, 10.0]],
        [[12.87, 12.87, 12.87, 25.86, 27.61, 43.09, 55.59, 66.88, 70.0, 70.0, 70.0],
         [70.0, 70.0, 70.0, 61.75, 59.75, 46.92, 31.32, 15.55, 10.0, 10.0, 10.0]],
        [[10.0, 10.0, 10.0, 16.1, 17.85, 51.92, 65.59, 76.88, 80.0, 80.0, 80.0],
         [80.0, 80.0, 80.0, 71.75, 69.75, 43.42, 24.02, 10.0, 10.0, 10.0, 10.0]],
        [[16.51, 16.51, 16.51, 22.61, 24.36, 58.44, 72.1, 83.39, 89.89, 90.0, 90.0],
         [90.0, 90.0, 90.0, 81.75, 79.75, 53.42, 34.02, 20.0, 11.67, 10.17, 10.17]],
        [[14.61, 14.61, 14.61, 20.71, 22.46, 56.53, 81.53, 93.39, 99.89, 100.0, 100.0],
         [100.0, 100.0, 100.0, 91.75, 89.75, 63.42, 32.82, 14.8, 10.0, 10.0, 10.0]]
    ]

    for M in [1.01, 2, 10, 100, 1000, 10000]:
        print('M: ', M)

        for t in range(len(distrs)):
            print('t: ', t)
            costs = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # 0.1*i for i in range(0,11)]
            valuations = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            no_supplier = distrs[t][0]  # [i*100 for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]  # [1 for _ in range(11)]
            no_buyer = distrs[t][
                1]  # [i*100 for i in [11, 10, 8, 8, 7, 6, 5, 4, 3, 2, 1]]  # 10, 11, 12, 13, 14, 14, 17, 18, 19, 10]
            commission = 0.1
            p_range = 0.1
            elapsingtime = []

            for i in range(10):
                print('i: ', i)
                etime, _, _ = optimize_hybrid_test(costs, valuations, no_supplier, no_buyer, commission, p_range, M)
                elapsingtime.append(etime)

            data = {'M': M,
                    't': t,
                    'time_elapsed': elapsingtime,
                    }

            if os.path.exists(file_path):
                # append
                with open(file_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(data.values())

            else:
                # save
                with open(file_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(data.keys())
                    writer.writerow(data.values())


test_runtime()


def read_test():
    import pandas as pd

    file_path = 'time_trials_unbalanced_real_distrs.csv'
    df = pd.read_csv(file_path)
    df.columns = ['M', 't', 'time_elapsed', 'pmin', 'pmax']

    df_t = df[['M', 't', 'time_elapsed']].set_index(['t', 'M']).unstack(level='M')
    df_t.columns = [second for first, second in df_t.columns]

    df_t.loc['Column_Avg', :] = df_t.mean(numeric_only=True, axis=0)
    df_t.loc[:, 'Row_Avg'] = df_t.mean(numeric_only=True, axis=1)


def read_time_trials_unbalanced_real_distrs_multiple():
    data = {}
    data_std = {}
    import re
    import numpy as np

    file_name = 'time_trials_unbalanced_real_distrs_multiple_2.txt'
    with open(file_name, 'r') as f:
        content = f.read()

    Ms = content.split('M:  ')[1:]

    for x in Ms:
        # x = Ms[0]

        elements = x.split('\nt:  ')

        M = elements[0]
        data[M] = {}
        data_std[M] = {}

        occurences = elements[1:]
        for i, occ in enumerate(occurences):
            # occ = occurences[0]
            x = occ.split('\nTime elapsed for optimization model: ')[1:]
            numbers = [float(re.search(r'\d+\.\d+', item).group()) for item in x]

            data[M][i] = sum(numbers) / len(numbers)
            data_std[M][i] = np.std(numbers)

        df = pd.DataFrame(data)
        df.loc['Column_Avg', :] = df.mean(numeric_only=True, axis=0)
        df.loc[:, 'Row_Avg'] = df.mean(numeric_only=True, axis=1)

        df_std = pd.DataFrame(data_std)
        df_std.loc['Column_Avg', :] = df_std.mean(numeric_only=True, axis=0)
        df_std.loc[:, 'Row_Avg'] = df_std.mean(numeric_only=True, axis=1)

        '''
        import ast
        file_name = 'time_trials_unbalanced_real_distrs_multiple_3.csv'
        df = pd.read_csv(file_name)

        df['time_elapsed'] = [ast.literal_eval(x) for x in df['time_elapsed']]
        df['time_avg'] = [sum(x)/len(x) for x in df['time_elapsed']]
        df['time_std'] = [np.std(x) for x in df['time_elapsed']]
        pivot_df = df.pivot(index='t', columns='M', values='time_avg')
        pivot_df.columns = [f"M={col}" for col in pivot_df.columns]

        '''


'''
def printing():
    print('Price: ', pmin.x, pmax.x)
    for b in range(B):
        print(f"Buyer {b} with valuation {valuations[b]}")
        print(f"c: {'  | '.join(['%.1f' % round(costs[s],2) for s in range(S)])}")
        print(f"x: {' | '.join(['%.2f' % round(x[s,b].x,2) for s in range(S)])}")
        print(f"p: {' | '.join(['%.2f' % round(p[s, b].x, 2) for s in range(S)])}")
'''