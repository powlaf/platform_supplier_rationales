import ast
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest


def create_hypothesis_test_testing():
    # load testing df
    path_df = 'sb3/testing/data/data_testing.csv'
    df_testing = pd.read_csv(path_df)
    env_names = list(set(df_testing.env_name))

    df_testing['price_setter'] = [x.split('_')[0] for x in df_testing.env_name]
    df_testing['patience'] = [x.split('_')[1][1:] for x in df_testing.env_name]
    df_testing['ratio'] = [x.split('_')[2][1:] for x in df_testing.env_name]

    # create df to save hypothesis results
    df_hypothesis = pd.DataFrame(columns=['env_name',
                                          't-test_returns_statistic', 't-test_returns_p-value',
                                          't-test_terminations_statistic', 't-test_terminations_p-value'
                                          ])

    # hypothesis testing for all environments
    # H0: the mean of env_name for a heuristic compared to a smart agents is the same
    # HA: the mean of env_name for a heuristic compared to a smart agent differs
    for env_name in env_names:
        print(env_name)

        df_testing_env = df_testing.loc[(df_testing['env_name'] == env_name), :]

        # t-test test for hypothesis testing (avg return)
        data_drl_returns = ast.literal_eval(df_testing_env.loc[df_testing_env['decision_rationale'] == 'drl', 'returns'].iloc[0])
        data_heuristic_returns = ast.literal_eval(
            df_testing_env.loc[df_testing_env['decision_rationale'] == 'heuristic', 'returns'].iloc[0])

        returns_statistic, returns_p_value = stats.ttest_ind(data_drl_returns, data_heuristic_returns, equal_var=True)
        print("RETURNS:", "t-statistic:", round(returns_statistic, 3), "p-value:", "{:.3f}".format(returns_p_value))

        # t-test test for hypothesis testing (average positive terminations / successfully matched episodes)
        data_drl_terminations = np.array(ast.literal_eval(
            df_testing_env.loc[df_testing_env['decision_rationale'] == 'drl', 'terminations'].iloc[0]), dtype=int)
        data_heuristic_terminations = np.array(ast.literal_eval(
            df_testing_env.loc[df_testing_env['decision_rationale'] == 'heuristic', 'terminations'].iloc[0]), dtype=int)

        terminations_statistic, terminations_p_value = proportions_ztest(
            np.array([data_drl_terminations.sum(), data_heuristic_terminations.sum()]),
            np.array([len(data_drl_terminations), len(data_heuristic_terminations)]))

        print("TERMINATIONS:", "t-statistic:", round(terminations_statistic, 3), "p-value:",
              "{:.3f}".format(terminations_p_value))

        # append data to df
        new_row = {
            'env_name': env_name,
            't-test_returns_statistic': returns_statistic,
            't-test_returns_p-value': returns_p_value,
            't-test_terminations_statistic': terminations_statistic,
            't-test_terminations_p-value': terminations_p_value,
        }

        new_row_df = pd.DataFrame([new_row], index=['new_row'])
        df_hypothesis = pd.concat([df_hypothesis, new_row_df])

    df_hypothesis = df_hypothesis.reset_index().drop(['index'], axis=1).sort_values(['env_name'], axis=0)
    df_hypothesis.to_excel('results/hypothesis_testing/hypothesis_testing.xlsx', merge_cells=True)

    return df_hypothesis


def create_hypothesis_test_simulation():
    # load testing df
    path_df = 'simulations/data/simulation_data_full.csv'
    df_simulation = pd.read_csv(path_df)
    env_names = list(set(df_simulation.env_name))

    df_simulation['%active_supplier_matched'] = df_simulation['matches'] / df_simulation['active_suppliers']
    df_simulation['price'] = df_simulation['revenue'] / df_simulation['matches']

    # create df to save hypothesis results
    result_indicators = ['welfare_supplier', 'welfare_customer', 'welfare_platform', 'matches',
                         '%active_supplier_matched', 'price']
    df_hypothesis = pd.DataFrame(columns=['env_name']+['statistic-' + x for x in result_indicators]+['p_value-' + x for x in result_indicators])

    # perform hypothesis test for all environments
    for env_name in env_names:

        df_hypothesis = pd.concat([df_hypothesis, pd.DataFrame({'env_name': env_name}, index=[0])], ignore_index=True)

        df_drl = df_simulation.loc[(df_simulation['decision_rationale'] == 'drl') & (df_simulation['env_name'] == env_name),:]
        df_heuristic = df_simulation.loc[(df_simulation['decision_rationale'] == 'heuristic') & (df_simulation['env_name'] == env_name),:]

        # perform hypothesis test for all relevant result indicators
        for col in result_indicators:

            statistic, p_value = stats.ttest_ind(df_drl[col],  df_heuristic[col], equal_var=True)

            df_hypothesis.loc[df_hypothesis['env_name'] == env_name, 'statistic-' + col] = statistic
            df_hypothesis.loc[df_hypothesis['env_name'] == env_name, 'p_value-' + col] = p_value

    df_hypothesis.to_excel('results/hypothesis_testing/hypothesis_simulation.xlsx', merge_cells=True)

    return df_hypothesis

