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
    # print(f"Time elapsed for optimization model: {elapsed_time:.6f} seconds")

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
    #start_time = time.time()

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
    m.setObjective(gp.quicksum(x[s, b] * no_supplier[s] * no_buyer[b] * p[s, b] for s in range(S) for b in range(B)) * commission, GRB.MAXIMIZE)

    # constraints
    M = 1000
    for s in range(S):
        for b in range(B):
            m.addConstr(x[s, b] <= (no_supplier[s] * no_buyer[b]) / (sum(no_supplier) * sum(no_buyer)))

            m.addConstr(x[s, b] <= z_s[s, b] * M)
            m.addConstr(x[s, b] <= z_b[s, b] * M)

            m.addConstr(costs[s] - p[s, b] * (1 - commission) <= M * (1 - z_s[s, b]))
            m.addConstr(p[s, b] - valuations[b] <= M * (1 - z_b[s, b]))

            m.addConstr(pmax - pmin == p_range)
            #m.addConstr(p[s, b] == costs[s] * (1 + commission))
            m.addConstr(x[s, b] <= z_pmin[s, b] * M)
            m.addConstr(x[s, b] <= z_pmax[s, b] * M)
            m.addConstr(pmin - p[s, b] <= M * (1 - z_pmin[s, b]))
            m.addConstr(p[s, b] - pmax <= M * (1 - z_pmax[s, b]))

    # solve
    m.optimize()

    #print('Price: ', pmin.x, pmax.x)

    # track time
    #elapsed_time = time.time() - start_time
    # print(f"Time elapsed for optimization model: {elapsed_time:.6f} seconds")

    return pmin.x, pmax.x


'''
def printing():
    print('Price: ', pmin.x, pmax.x)
    for b in range(B):
        print(f"Buyer {b} with valuation {valuations[b]}")
        print(f"c: {'  | '.join(['%.1f' % round(costs[s],2) for s in range(S)])}")
        print(f"x: {' | '.join(['%.2f' % round(x[s,b].x,2) for s in range(S)])}")
        print(f"p: {' | '.join(['%.2f' % round(p[s, b].x, 2) for s in range(S)])}")
'''