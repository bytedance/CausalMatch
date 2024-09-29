import pandas as pd
import numpy as np
from scipy.stats import norm
def sensitivity_test(match_obj, gamma, y_i):

    id_list_t = match_obj.df_out_final['user_id_treat'].to_list()
    id_list_c = match_obj.df_out_final['user_id_control'].to_list()

    df_pair = pd.DataFrame()
    df_pair['y_t'], df_pair['y_c'] = match_obj.data.iloc[id_list_t][y_i].values, match_obj.data.iloc[id_list_c][y_i].values
    df_pair['absolute_diff'] = np.abs(df_pair['y_t'] - df_pair['y_c'])
    df_pair['absolute_diff_rank'] = df_pair['absolute_diff'].rank()
    df_pair['cs1'] = (df_pair['y_t'] > df_pair['y_c']) * 1
    df_pair['cs2'] = (df_pair['y_c'] > df_pair['y_t']) * 1
    df_pair['cz'] = df_pair['cs1'] * 1 + df_pair['cs2'] * 0
    df_pair['dcz'] = df_pair['cz'] * df_pair['absolute_diff_rank']
    test_stat = df_pair['dcz'].sum()

    cols = ['Wilcoxon-statistic', 'gamma'
            , 'stat upper bound', 'stat_lower_bound'
            , 'z-score upper bound', 'z-score lower bound'
            , 'p-val upper bound', 'p-val lower bound']
    lst = []
    for gamma_i in gamma:
        df_pair['ps_plus'] = gamma_i / (1 + gamma_i)
        df_pair['ps_plus'][(df_pair['cs1'] == 0) & (df_pair['cs2'] == 0)] = 0
        df_pair['ps_plus'][(df_pair['cs1'] == 1) & (df_pair['cs2'] == 1)] = 1

        df_pair['ps_minus'] = 1 / (1 + gamma_i)
        df_pair['ps_minus'][(df_pair['cs1'] == 0) & (df_pair['cs2'] == 0)] = 0
        df_pair['ps_minus'][(df_pair['cs1'] == 1) & (df_pair['cs2'] == 1)] = 1

        gamma_1_t_plus = np.sum(df_pair['absolute_diff_rank'] * df_pair['ps_plus'])

        gamma_1_t_minus = np.sum(df_pair['absolute_diff_rank'] * df_pair['ps_minus'])

        var_t_plus = np.sum(
            df_pair['absolute_diff_rank'] ** 2 * (df_pair['ps_plus'] * (1 - df_pair['ps_plus'])))

        deviate_ub = (df_pair['dcz'].sum() - gamma_1_t_plus) / np.sqrt(var_t_plus)
        deviate_lb = (df_pair['dcz'].sum() - gamma_1_t_minus) / np.sqrt(var_t_plus)

        if test_stat > round(gamma_1_t_plus, 2):
            lst.append([test_stat
                           , gamma_i
                           , round(gamma_1_t_plus, 2)
                           , round(gamma_1_t_minus, 2)
                           , deviate_ub
                           , deviate_lb
                           , np.round(norm.sf(abs(deviate_ub)), 5)
                           , np.round(norm.sf(abs(deviate_lb)), 5)])
        else:
            lst.append([test_stat
                           , gamma_i
                           , round(gamma_1_t_plus, 2)
                           , round(gamma_1_t_minus, 2)
                           , None
                           , None
                           , None
                           , None])

    df_sensitivity_test = pd.DataFrame(lst, columns=cols)
    return df_sensitivity_test
