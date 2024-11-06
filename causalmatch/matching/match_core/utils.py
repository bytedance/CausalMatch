# Copyright 2024 Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy as scipy
import warnings
from distutils.version import LooseVersion, StrictVersion

warnings.simplefilter(action='ignore', category=FutureWarning)
def data_process_bc(match_obj,
                    include_discrete) :
    """
    Data preprocess before balance check.

    Parameters
    ----------
    match_obj: object after matching
    include_discrete: if discrete variables are included. if True, then do one-hot.
    """

    x_numeric = match_obj.X_numeric
    x_discrete = match_obj.X_discrete
    df_out_ = match_obj.df_out_final
    df_cat = match_obj.data_with_categ
    df_psm_post_trim = match_obj.df_out_final_post_trim
    df_raw = match_obj.data

    T = match_obj.T
    id = match_obj.id

    if match_obj.method == 'cem':
        # if process with cem method, previously did not do one-hot,
        # do here for balance check
        if include_discrete and len(x_discrete)>0:
            # post-matching
            df_post_validate_x = pd.concat([
                df_out_[x_numeric],
                pd.get_dummies(df_out_[x_discrete], columns=x_discrete, drop_first=True)
                # categorical features converted to dummies
            ], axis=1)

            # pre-matching
            df_pre_validate_x = pd.concat([
                df_raw[x_numeric],
                pd.get_dummies(df_raw[x_discrete], columns=x_discrete, drop_first=True)
                # categorical features converted to dummies
            ], axis=1)
        else :
            df_post_validate_x = df_out_[x_numeric]
            df_pre_validate_x = df_raw[x_numeric]

        df_post_validate = pd.concat([df_out_[[T, id]], df_post_validate_x], axis=1)
        df_pre_validate = pd.concat([df_raw[[T, id]], df_pre_validate_x], axis=1)
        df_x = df_post_validate_x.columns
    else :
        # data with x post match
        if include_discrete or x_discrete is None:
            df_right = (df_cat) * 1
        else :
            df_right = df_cat[x_numeric]

        df_x = df_right.columns
        df_right[id] = df_raw[id]
        df_post_validate = df_psm_post_trim.merge(df_right, how='left', on=id)

        # data with x pre match
        df_pre_validate = df_right
        df_pre_validate[T] = df_raw[T]

    return df_x, df_post_validate, df_pre_validate

def data_process_ate(match_obj, df_post_validate):
    y = match_obj.y
    id = match_obj.id

    method = match_obj.method
    data = match_obj.data
    df_out_final = match_obj.df_out_final

    if method == 'cem':
        # 1. merge y variables from raw df to matched df
        df_post_validate_y_ = df_post_validate.merge(data[y + [id]], how='left', on=id)

        # 2. for CEM need one more step due to WLS: merge weight to outcome dataframe
        df_post_validate_y = df_post_validate_y_.merge(df_out_final[[id, 'weight_adjust']], how='left', on=id)
        weight = df_post_validate_y['weight_adjust']
    else :
        df_post_validate_y = df_post_validate.merge(data[y + [id]], how='left', on=id)
        weight = None

    return df_post_validate_y, weight
def balance_check_x(match_obj,
                    X,
                    df_post,
                    df_pre):
    treat_var = match_obj.T
    threshold_smd = match_obj.threshold_smd
    threshold_vr = match_obj.threshold_vr

    smd_match_df_post = smd_df(df_post, X, treat_var, threshold_smd, threshold_vr, 'post')
    smd_match_df_pre = smd_df(df_pre, X, treat_var, threshold_smd, threshold_vr, 'pre')

    return smd_match_df_post, smd_match_df_pre


def smd_df(df, X, treat_var, threshold_smd, threshold_vr, type) :
    smd_all = {"Covariates" : [],
               "Mean Treated {}-match".format(type) : [],
               "Mean Control {}-match".format(type) : [],
               "SMD" : [],
               "Var Ratio" : [],
               "ks-p_val" : [],
               "ttest-p_val" : []}

    for col in X :
        col, t_avg, c_avg, t_avg, c_avg, smd, vr, pval, p = smd_x(col, df, treat_var, threshold_smd, threshold_vr)

        smd_all["Covariates"].append(col)
        smd_all["Mean Treated {}-match".format(type)].append(t_avg)
        smd_all["Mean Control {}-match".format(type)].append(c_avg)
        smd_all["SMD"].append(smd)
        smd_all["Var Ratio"].append(vr)
        smd_all["ks-p_val"].append(pval)
        smd_all["ttest-p_val"].append(p)

    df_res = pd.DataFrame(smd_all)
    return df_res


def smd_x(col, df, treat_var, threshold_smd, threshold_vr) :
    # Xiaoyu Zhou 08/11/2024: cast boolean type dataframe to int or stats raise error
    c_array = (df[df[treat_var] == 0][col].values) * 1
    t_array = (df[df[treat_var] == 1][col].values) * 1

    # post-matching balance test
    t_avg, c_avg, smd, pass_smd, vr, pass_vr = calculate_smd(c_array,
                                                             t_array,
                                                             t_array,
                                                             threshold_smd,
                                                             threshold_vr)

    # Performs the two-sample Kolmogorov-Smirnov test for goodness of fit.
    if LooseVersion(scipy.__version__) >= LooseVersion('1.12.0') :
        ks_stats, pvalue = stats.ks_2samp(c_array,
                                          t_array,
                                          method='asymp')
    else:
        ks_stats, pvalue = stats.ks_2samp(c_array,
                                          t_array)

    # Perform Levene test for equal variances.
    _, levene_p = stats.levene(c_array, t_array)

    if levene_p > 0.05 :
        t, p = stats.ttest_ind(c_array, t_array, equal_var=True)
    else :
        t, p = stats.ttest_ind(c_array, t_array, equal_var=False)

    return col, t_avg, c_avg, t_avg, c_avg, smd, vr, np.round(pvalue, 3), np.round(p, 3)


def calculate_smd(c_array,
                  t_array,
                  all_t_array,
                  threshold_smd,
                  threshold_vr) :
    c_avg = np.round(c_array.mean(), 4)
    t_avg = np.round(t_array.mean(), 4)
    sf = all_t_array.std()
    smd = np.round((t_avg - c_avg) / sf, 2)
    if np.abs(smd) > threshold_smd :
        pass_smd = False
    else :
        pass_smd = True
    if np.all((c_array == 0) | (c_array == 1)) and np.all((t_array == 0) | (t_array == 1)) :
        vr = np.nan
        pass_vr = np.nan
    else :
        vr = np.round(np.var(t_array) / np.var(c_array), 2)
        if (vr < threshold_vr) and (vr > 1 / threshold_vr) :
            pass_vr = True
        else :
            pass_vr = False
    return t_avg, c_avg, smd, pass_smd, vr, pass_vr


def gen_test_data(n=5000, c_ratio=0.1) :
    """
    :n: number of observations
    :c_ratio: fraction of control obs
    """
    np.random.seed(123456)

    # generate sudo-data for matching
    ## 1. generate categorical data

    k_continuous = 3
    k_discrete = 3
    n_obs = n

    col_name_x = ['c_1', 'c_2', 'c_3', 'd_1', 'gender', 'd_3']
    col_name_list = ['c_1', 'c_2', 'c_3', 'd_1', 'gender', 'd_3', 'treatment']
    col_name_y = ['y']

    list_choice = list(range(10))
    rand_vec_1 = np.random.rand(10)
    array_prob = rand_vec_1 / np.sum(rand_vec_1)
    list_prob = list(array_prob)

    rand_continuous = np.random.rand(n_obs, k_continuous)
    rand_discrete = np.random.choice(a=list_choice,
                                     size=[n_obs, k_discrete],
                                     p=list_prob)
    rand_treatment = np.random.choice(a=[0, 1], size=[n_obs, 1], p=[c_ratio, 1 - c_ratio])
    rand_error = np.random.normal(loc=0.0, scale=1.0, size=[n_obs, 1])
    rand_true_param = np.random.normal(loc =5.0, scale=1.0, size=[k_continuous, 1])
    rand_full = np.concatenate((rand_continuous, rand_discrete, rand_treatment), axis=1)
    param_te = 0.5

    df = pd.DataFrame(data=rand_full,
                      columns=col_name_list)

    df['gender'] = df['gender'].replace({0 : "male",
                                         1 : "female",
                                         2 : "cat",
                                         3 : "dog",
                                         4 : "pig",
                                         5 : "cat1",
                                         6 : "cat2",
                                         7 : "cat3",
                                         8 : "cat4",
                                         9 : "cat5", })
    df[col_name_y] = rand_continuous @ rand_true_param + param_te * rand_treatment + rand_error

    df['y2'] = 0.1 * rand_treatment + rand_error

    df['user_id'] = df.index

    df['d_1'] = df['d_1'].astype('str')
    df['d_1'].replace(['0.0', '1.0', '2.0'], 'apple', inplace=True)
    df['d_1'].replace(['3.0', '4.0'], 'pear', inplace=True)
    df['d_1'].replace(['5.0', '6.0'], 'cat', inplace=True)
    df['d_1'].replace(['7.0', '8.0'], 'dog', inplace=True)
    df['d_1'].replace(['9.0'], 'bee', inplace=True)

    df['d_1'].value_counts()

    df['d_3'] = df['d_3'].astype('str')

    return df, rand_continuous, rand_true_param, param_te , rand_treatment, rand_error


def gen_test_data_panel(N, T, beta, ate, exp_date, unbalance=False) :
    """
    :N: number of cross sections
    :T: number of time periods
    :beta: true beta
    :ate: true ate
    :exp_date: experiment date
    """
    if exp_date > T :
        raise NameError('exp_date must be smaller than the number of time periods')

    if len(beta) <= 1 :
        raise NameError('length of array beta must be greater than or equal to 2')

    k = len(beta)
    id_list = np.linspace(1, N, num=N)
    time_list = np.linspace(1, T, num=T)

    np.random.seed(0)
    error = np.random.normal(0, 1, N * T)
    t_effect = np.random.normal(ate, 1, N * T)

    c_i = np.random.normal(0, 1, N)
    a_t = np.random.normal(0, 1, T)
    x1 = np.random.normal(0, 1, N * T * (k - 1))
    x1.shape = (N * T, (k - 1))
    const = np.ones(N * T)
    const.shape = (N * T, 1)
    x = np.concatenate((const, x1), axis=1)

    j_N = np.ones(N)
    time_full = np.kron(j_N, time_list)
    a_t_full = np.kron(j_N, a_t)
    id_full = np.repeat(id_list, T, axis=0)
    c_i_full = np.repeat(c_i, T, axis=0)
    treatment_group = np.random.choice(id_list, round(N / 4), replace=False)

    df = pd.DataFrame(data=x)
    for name in df.columns :
        new_name = "x_" + str(name)
        df = df.rename(columns={name : new_name})

    beta = np.reshape(beta, (-1, k))
    df['xb'] = np.dot(x, beta.T)

    df['id'] = id_full
    df['time'] = time_full
    df['c_i'] = c_i_full
    df['a_t'] = a_t_full
    df['error'] = error

    df['post'] = (df['time'] >= exp_date) * 1
    df['treatment'] = df['id'].apply(lambda x : 1 if x in treatment_group else 0)

    df['y'] = df['xb'] + ate * df['treatment'] * df['post'] + df['c_i'] + df['a_t'] + df['error']

    df2 = df.copy()
    for tt in time_list :
        if tt < 10 :
            time_str = "date_0" + str(int(tt))
        else :
            time_str = "date_" + str(int(tt))
        df2 = df2.replace({'time' : tt}, time_str)

    if unbalance :
        np.random.seed(10)
        drop_indices = np.random.choice(df.index, N - 1, replace=False)
        df2 = df2.drop(drop_indices)

    return df2


def psm_trim_caliper(match_obj,
                     df_pre,
                     caliper: float = 0.05) :

    df_post = df_pre.copy()
    if caliper > 0 :
        df_pre['pscore_diff'] = np.abs(df_pre['pscore_treat'] - df_pre['pscore_control'])
        valid_pair_indices = df_pre[df_pre['pscore_diff'] <= caliper].index
        df_post = df_pre.iloc[valid_pair_indices, :].copy()
        df_post.reset_index(inplace=True, drop=True)

    # stack up all observations
    df_post_treat = df_post[[match_obj.id + "_treat", match_obj.T + "_treat", 'pscore_treat']].copy()
    df_post_control = df_post[[match_obj.id + "_control", match_obj.T + "_control", 'pscore_control']].copy()

    df_post_treat_ = df_post_treat.rename(
        columns={match_obj.id + "_treat" : match_obj.id, match_obj.T + "_treat" : match_obj.T, 'pscore_treat' : 'pscore'})
    df_post_control_ = df_post_control.rename(
        columns={match_obj.id + "_control" : match_obj.id, match_obj.T + "_control" : match_obj.T, 'pscore_control' : 'pscore'})

    df_full = pd.concat([df_post_treat_, df_post_control_], axis=0, ignore_index=True)
    df_full.drop_duplicates(subset=[match_obj.id], inplace=True, ignore_index=True)

    return df_post, df_full

def psm_trim_percent(match_obj,
                     df_pre,
                     percentage: float = 0.00) :
    df_post = df_pre.copy()

    # stack up all observations
    df_post_treat = df_post[[match_obj.id + "_treat", match_obj.T + "_treat", 'pscore_treat']]
    df_post_control = df_post[[match_obj.id + "_control", match_obj.T + "_control", 'pscore_control']]

    df_post_treat_ = df_post_treat.rename(
        columns={match_obj.id + "_treat" : match_obj.id, match_obj.T + "_treat" : match_obj.T, 'pscore_treat' : 'pscore'})
    df_post_control_ = df_post_control.rename(
        columns={match_obj.id + "_control" : match_obj.id, match_obj.T + "_control" : match_obj.T, 'pscore_control' : 'pscore'})

    df_full = pd.concat([df_post_treat_, df_post_control_], axis=0, ignore_index=True)

    if (percentage > 0) and (percentage < 1) :
        p_score_ub = df_full['pscore'].quantile(q=1 - percentage / 2)
        p_score_lb = df_full['pscore'].quantile(q=percentage / 2)
        df_post = df_full[(df_full['pscore'] <= p_score_ub) & (df_full['pscore'] >= p_score_lb)]

    elif percentage == 0 :
        df_post = df_full

    else :
        raise TypeError('Trim percentage should a value between 0 and 1.')

    df_post.reset_index(inplace=True, drop=True)
    return df_post


