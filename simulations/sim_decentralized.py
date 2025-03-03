import random
import numpy as np
from stable_baselines3 import SAC
from simulations.classes import Customer, Supplier, Match


def decentralized_heuristic(patience, arrival_customers, arrival_suppliers, timesteps, commission, printl=True):
    # all C and S in the platform
    customers = []
    suppliers = []

    # valuations and cost types
    delta = 10
    valuations = [1/delta*i for i in range(delta+1)]
    costs = [0.1 * i for i in range(delta+1)]

    # distribution of valuations and costs anticipated by the platform
    dist_suppliers = [0 for _ in range(delta+1)]
    dist_customers = [0 for _ in range(delta+1)]

    # track matches and overall statistics
    matches_total = {}
    totals = {
        'welfare_total': [0 for _ in range(timesteps)],
        'welfare_supplier': [0 for _ in range(timesteps)],
        'welfare_customer': [0 for _ in range(timesteps)],
        'welfare_platform': [0 for _ in range(timesteps)],
        'costs': [0 for _ in range(timesteps)],
        'valuations': [0 for _ in range(timesteps)],
        'revenue': [0 for _ in range(timesteps)],
        'matches': [0 for _ in range(timesteps)],
        'total_customers': [0 for _ in range(timesteps)],
        'active_customers': [0 for _ in range(timesteps)],
        'total_suppliers': [0 for _ in range(timesteps)],
        'active_suppliers': [0 for _ in range(timesteps)]
    }

    # run simulation for t timesteps
    for t in range(timesteps):

        if printl:
            print('TIMESTEP ', t)

        # C & S arrival for overall tracking and platform anticipationn
        customers += [Customer(patience) for _ in range(arrival_customers)]
        suppliers += [Supplier(patience) for _ in range(arrival_suppliers)]

        new_customers = [arrival_customers/(delta+1) for _ in range(len(dist_customers))]
        dist_customers = [ni + oi for (ni, oi) in zip(new_customers, dist_customers)]
        new_suppliers = [arrival_suppliers / (delta + 1) for _ in range(len(dist_suppliers))]
        dist_suppliers = [ni + oi for (ni, oi) in zip(new_suppliers, dist_suppliers)]
        
        totals['active_suppliers'][t] = len(suppliers)
        totals['total_suppliers'][t] = len(suppliers)
        totals['active_customers'][t] = len(customers)
        totals['total_customers'][t] = len(customers)

        if printl:
            print(f'Customer: {len(customers)} ({[round(c.valuation,2) for c in customers]})')
            print(f'Supplier: {len(suppliers)} ({[round(s.cost, 2) for s in suppliers]})')

        # uniform random matching
        matches = []
        short_side = 's' if len(customers) > len(suppliers) else 'c'
        primary, secondary = (suppliers, customers) if short_side == 's' else (customers, suppliers)
        remaining = secondary.copy()

        for p in primary:
            sec = random.choice(remaining)
            remaining.remove(sec)

            # supplier sets own costs (plus commission) as price
            price = sec.cost / (1 - commission) + 0.001 if short_side == 'c' else p.cost / (1 - commission) + 0.001

            m = Match(p if short_side == 'c' else sec, sec if short_side == 'c' else p, price, commission)
            if m.status:
                matches.append(m)
                if printl:
                    print(f'Achieved match between supplier (cost: {round(m.supplier.cost, 2)}) and customer (valuation: {round(m.customer.valuation,2)}) '
                          f'  -- Price: {round(price, 2)} -- Welfare: {m.welfare} ({m.welfare_supplier}/{m.welfare_customer}/{m.welfare_platform})'
                          )
            else:
                if printl:
                    print(
                        f'Failed match between supplier (cost: {round(m.supplier.cost, 2)}) and customer (valuation: {round(m.customer.valuation,2)}) '
                        f'  -- Price: {round(price, 2)} -- Welfare: {m.welfare} ({m.welfare_supplier}/{m.welfare_customer}/{m.welfare_platform})'
                    )

        # get matching statistics
        matches_total[t] = matches
        for m in matches:
            totals['welfare_total'][t] += m.welfare
            totals['welfare_supplier'][t] += m.welfare_supplier
            totals['welfare_customer'][t] += m.welfare_customer
            totals['welfare_platform'][t] += m.welfare_platform
            totals['revenue'][t] += m.price
            totals['costs'][t] += m.supplier.cost
            totals['valuations'][t] += m.customer.valuation
            totals['matches'][t] += 1

        if printl:
            print(f'Before: Customers ({len(customers)}) | Suppliers ({len(suppliers)})')

        # matches C & S leave
        for m in matches:
            customers.remove(m.customer)
            suppliers.remove(m.supplier)
        if printl:
            print(f'After matching: Customers ({len(customers)}) | Suppliers ({len(suppliers)})')

        # impatient C & S leave
        for entity in customers + suppliers:
            entity.waiting_time += 1

        customers = [c for c in customers if c.waiting_time < c.patience]
        suppliers = [s for s in suppliers if s.waiting_time < s.patience]
        if printl:
            print(f'After patience: Customers ({len(customers)}) | Suppliers ({len(suppliers)})')

        # platform updates anticipated C & S type distribution
        for m in matches:
            c_inactive = sum(1 for v in valuations if v < m.price)
            c_active = len(valuations) - c_inactive

            c_update = [0. for _ in range(c_inactive)] + [-1/c_active for _ in range(c_active)]
            dist_customers = [max(0, oi + li) for (oi, li) in zip(dist_customers, c_update)]

            s_inactive = sum(1 for c in costs if c > m.price * (1-commission))
            s_active = len(costs) - s_inactive

            s_update = [-1/s_active for _ in range(s_active)] + [0. for _ in range(s_inactive)]
            dist_suppliers = [max(0, oi + li) for (oi, li) in zip(dist_suppliers, s_update)]

        if printl:
            print(f'Updated supplier types: ', [round(c, 2) for c in dist_suppliers])
            print(f'Updated customer types: ', [round(v, 2) for v in dist_customers])

    return totals


def decentralized_drl(patience, arrival_customers, arrival_suppliers, timesteps, commission, printl=True, env_name='xxx'):

    # load agent
    dir = f"./sb3/logs/{env_name}/best_model"
    model = SAC.load(dir)

    # all C and S in the platform
    customers = []
    suppliers = []

    # valuations and cost types
    delta = 10
    valuations = [1 / delta * i for i in range(delta + 1)]
    costs = [0.1 * i for i in range(delta + 1)]

    # distribution of valuations and costs anticipated by the platform
    dist_suppliers = [0 for _ in range(delta + 1)]
    dist_customers = [0 for _ in range(delta + 1)]

    # track matches and overall statistics
    matches_total = {}
    totals = {
        'welfare_total': [0 for _ in range(timesteps)],
        'welfare_supplier': [0 for _ in range(timesteps)],
        'welfare_customer': [0 for _ in range(timesteps)],
        'welfare_platform': [0 for _ in range(timesteps)],
        'costs': [0 for _ in range(timesteps)],
        'valuations': [0 for _ in range(timesteps)],
        'revenue': [0 for _ in range(timesteps)],
        'matches': [0 for _ in range(timesteps)],
        'total_customers': [0 for _ in range(timesteps)],
        'active_customers': [0 for _ in range(timesteps)],
        'total_suppliers': [0 for _ in range(timesteps)],
        'active_suppliers': [0 for _ in range(timesteps)]
    }

    # run simulation for t timesteps
    for t in range(timesteps):

        if printl:
            print('TIMESTEP ', t)

        # C & S arrival for overall tracking and platform anticipationn
        customers += [Customer(patience) for _ in range(arrival_customers)]
        suppliers += [Supplier(patience) for _ in range(arrival_suppliers)]

        new_customers = [arrival_customers / (delta + 1) for _ in range(len(dist_customers))]
        dist_customers = [ni + oi for (ni, oi) in zip(new_customers, dist_customers)]
        new_suppliers = [arrival_suppliers / (delta + 1) for _ in range(len(dist_suppliers))]
        dist_suppliers = [ni + oi for (ni, oi) in zip(new_suppliers, dist_suppliers)]

        totals['active_suppliers'][t] = len(suppliers)
        totals['total_suppliers'][t] = len(suppliers)
        totals['active_customers'][t] = len(customers)
        totals['total_customers'][t] = len(customers)

        if printl:
            print(f'Customer: {len(customers)} ({[round(c.valuation, 2) for c in customers]})')
            print(f'Supplier: {len(suppliers)} ({[round(s.cost, 2) for s in suppliers]})')

        # uniform random matching
        matches = []
        short_side = 's' if len(customers) > len(suppliers) else 'c'
        primary, secondary = (suppliers, customers) if short_side == 's' else (customers, suppliers)
        remaining = secondary.copy()

        for p in primary:
            sec = random.choice(remaining)
            remaining.remove(sec)

            # supplier sets price
            cost = sec.cost if short_side == 'c' else p.cost
            waiting_time = sec.waiting_time if short_side == 'c' else p.waiting_time
            observation = np.array([cost, cost / (1 - commission), patience - waiting_time, t]).astype(np.float32)
            price = model.predict(observation=observation)[0].item()

            m = Match(p if short_side == 'c' else sec, sec if short_side == 'c' else p, price, commission)

            if m.status:
                matches.append(m)
                if printl:
                    print(
                        f'Achieved match between supplier (cost: {round(m.supplier.cost, 2)}) and customer (valuation: {round(m.customer.valuation, 2)}) '
                        f'  -- Price: {round(price, 2)} -- Welfare: {m.welfare} ({m.welfare_supplier}/{m.welfare_customer}/{m.welfare_platform})'
                        )
            else:
                if printl:
                    print(
                        f'Failed match between supplier (cost: {round(m.supplier.cost, 2)}) and customer (valuation: {round(m.customer.valuation, 2)}) '
                        f'  -- Price: {round(price, 2)} -- Welfare: {m.welfare} ({m.welfare_supplier}/{m.welfare_customer}/{m.welfare_platform})'
                    )

        # get matching statistics
        matches_total[t] = matches
        for m in matches:
            totals['welfare_total'][t] += m.welfare
            totals['welfare_supplier'][t] += m.welfare_supplier
            totals['welfare_customer'][t] += m.welfare_customer
            totals['welfare_platform'][t] += m.welfare_platform
            totals['revenue'][t] += m.price
            totals['costs'][t] += m.supplier.cost
            totals['valuations'][t] += m.customer.valuation
            totals['matches'][t] += 1

        if printl:
            print(f'Before: Customers ({len(customers)}) | Suppliers ({len(suppliers)})')

        # matches C & S leave
        for m in matches:
            customers.remove(m.customer)
            suppliers.remove(m.supplier)
        if printl:
            print(f'After matching: Customers ({len(customers)}) | Suppliers ({len(suppliers)})')

        # impatient C & S leave
        for entity in customers + suppliers:
            entity.waiting_time += 1

        customers = [c for c in customers if c.waiting_time < c.patience]
        suppliers = [s for s in suppliers if s.waiting_time < s.patience]
        if printl:
            print(f'After patience: Customers ({len(customers)}) | Suppliers ({len(suppliers)})')

        # platform updates anticipated C & S type distribution
        for m in matches:
            c_inactive = sum(1 for v in valuations if v < m.price)
            c_active = len(valuations) - c_inactive

            c_update = [0. for _ in range(c_inactive)] + [-1 / c_active for _ in range(c_active)]
            dist_customers = [max(0, oi + li) for (oi, li) in zip(dist_customers, c_update)]

            s_inactive = sum(1 for c in costs if c > m.price * (1 - commission))
            s_active = len(costs) - s_inactive

            s_update = [-1 / s_active for _ in range(s_active)] + [0. for _ in range(s_inactive)]
            dist_suppliers = [max(0, oi + li) for (oi, li) in zip(dist_suppliers, s_update)]

        if printl:
            print(f'Updated supplier types: ', [round(c, 2) for c in dist_suppliers])
            print(f'Updated customer types: ', [round(v, 2) for v in dist_customers])

    return totals

