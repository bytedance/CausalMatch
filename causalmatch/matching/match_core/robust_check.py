import pandas as pd
import numpy as np
from scipy.stats import norm
from causalmatch.matching.match_core.preprocess import preprocess
import copy


def sensitivity_test(match_obj, gamma, y_i) :
    id_treat = match_obj.id + "_treat"
    id_control = match_obj.id + "control"

    id_list_t = match_obj.df_out_final[id_treat].to_list()
    id_list_c = match_obj.df_out_final[id_control].to_list()

    df_pair = pd.DataFrame()
    # ------------------------------------------------------------------------------
    # BUGFIX: 20241018, Xiaoyu Zhou
    # df_pair['y_t'], df_pair['y_c'] = match_obj.data.iloc[id_list_t][y_i].values, match_obj.data.iloc[id_list_c][y_i].values
    df_pair['y_t'] = match_obj.data[match_obj.data[match_obj.id].isin(id_list_t)][y_i].values
    df_pair[id_treat] = id_list_t
    df_pair[id_control] = id_list_c
    df_pair_ = df_pair.merge(match_obj.data[[match_obj.id, y_i]], how='left', left_on=id_control,
                             right_on=match_obj.id)
    df_pair_.rename(columns={y_i : "y_c"}, inplace=True)
    df_pair_.drop(columns=match_obj.id, inplace=True)
    df_pair = df_pair_.copy()
    # ------------------------------------------------------------------------------

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
    for gamma_i in gamma :
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

        if test_stat > round(gamma_1_t_plus, 2) :
            lst.append([test_stat
                           , gamma_i
                           , round(gamma_1_t_plus, 2)
                           , round(gamma_1_t_minus, 2)
                           , deviate_ub
                           , deviate_lb
                           , np.round(norm.sf(abs(deviate_ub)), 5)
                           , np.round(norm.sf(abs(deviate_lb)), 5)])
        else :
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


def placebo_treatment_estimate(match_obj, n, b):
    np.random.seed(123456)

    # Sample sub data from original dataset
    data_b = match_obj.data.sample(n=n).copy()
    data_b.reset_index(inplace=True, drop=True)

    # Create random treatment variable based on binomial(0.5) distribution
    rand_discrete = np.random.binomial(1, 0.5, [n, b])
    ate_list = []

    pseudo_t_list = []
    for i in range(b) :
        t_i_name = 't_{}'.format(i)
        pseudo_t_list.append(t_i_name)
        data_b[t_i_name] = rand_discrete[:, i]

        match_obj_i = copy.deepcopy(match_obj)

        # specify parameter change with loop
        match_obj_i.data = data_b
        match_obj_i.T = t_i_name
        preprocess(match_obj_i)
        match_obj_i.psm(n_neighbors=1,
                        model=match_obj.model,
                        caliper=match_obj.caliper,
                        trim_percentage=match_obj.trim_percentage,
                        drop_duplicates=match_obj.drop_duplicates,
                        model_list=match_obj.model_list,
                        test_size=match_obj.test_size,
                        verbose=None)

        ate_i = match_obj_i.ate().iloc[0]['ate']
        ate_list.append(ate_i)

    return ate_list
