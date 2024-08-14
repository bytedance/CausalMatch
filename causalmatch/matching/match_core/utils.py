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
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def data_process_bc(match_obj,
                    include_discrete):
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
        if include_discrete:
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
    else:
        # data with x post match
        if include_discrete:
            df_right = (df_cat) * 1
        else:
            df_right = df_cat[x_numeric]

        df_x = df_right.columns
        df_right[id] = df_raw[id]
        df_post_validate = df_psm_post_trim.merge(df_right, how='left', on=id)

        # data with x pre match
        df_pre_validate = df_right
        df_pre_validate[T] = df_raw[T]

    return df_x, df_post_validate, df_pre_validate

def balance_check_x(match_obj,
                    X,
                    df_post,
                    df_pre):

    treat_var = match_obj.T
    threshold_smd = match_obj.threshold_smd
    threshold_vr = match_obj.threshold_vr

    smd_match_df_post = smd_df(df_post, X, treat_var, threshold_smd, threshold_vr, 'post')
    smd_match_df_pre = smd_df(df_pre, X , treat_var, threshold_smd, threshold_vr, 'pre')

    return smd_match_df_post, smd_match_df_pre


def smd_df(df, X, treat_var, threshold_smd, threshold_vr, type):


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



def smd_x(col, df, treat_var, threshold_smd, threshold_vr):

    # Xiaoyu Zhou 08/11/2024: cast boolean type dataframe to int or stats raise error
    c_array = (df[df[treat_var] == 0][col].values)*1
    t_array = (df[df[treat_var] == 1][col].values)*1

    # post-matching balance test
    t_avg, c_avg, smd, pass_smd, vr, pass_vr = calculate_smd(c_array,
                                                             t_array,
                                                             t_array,
                                                             threshold_smd,
                                                             threshold_vr)

    # Performs the two-sample Kolmogorov-Smirnov test for goodness of fit.
    ks_stats, pvalue = stats.ks_2samp(c_array,
                                      t_array,
                                      method='asymp')

    # Perform Levene test for equal variances.
    _, levene_p = stats.levene(c_array, t_array)

    if levene_p > 0.05 :
        t, p = stats.ttest_ind(c_array, t_array, equal_var=True)
    else :
        t, p = stats.ttest_ind(c_array, t_array, equal_var=False)

    return col,t_avg,c_avg,t_avg,c_avg,smd,vr,np.round(pvalue, 3),np.round(p, 3)


def calculate_smd(c_array,
                  t_array,
                  all_t_array,
                  threshold_smd,
                  threshold_vr):
    c_avg = np.round(c_array.mean(), 4)
    t_avg = np.round(t_array.mean(), 4)
    sf = all_t_array.std()
    smd = np.round((t_avg - c_avg) / sf, 2)
    if np.abs(smd) > threshold_smd:
        pass_smd = False
    else:
        pass_smd = True
    if np.all((c_array == 0) | (c_array == 1)) and np.all((t_array == 0) | (t_array == 1)):
        vr = np.nan
        pass_vr = np.nan
    else:
        vr = np.round(np.var(t_array) / np.var(c_array), 2)
        if (vr < threshold_vr) and (vr > 1 / threshold_vr):
            pass_vr = True
        else:
            pass_vr = False
    return t_avg, c_avg, smd, pass_smd, vr, pass_vr


def gen_test_data(n = 5000, c_ratio = 0.1):
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
    rand_treatment = np.random.choice(a=[0, 1], size=[n_obs, 1], p=[c_ratio, 1-c_ratio])
    rand_error = np.random.normal(loc=0.0, scale=1.0, size=[n_obs, 1])
    rand_true_param = np.random.normal(loc=0.0, scale=1.0, size=[k_continuous, 1])
    rand_full = np.concatenate((rand_continuous, rand_discrete, rand_treatment), axis=1)
    param_te = 0.5

    df = pd.DataFrame(data=rand_full,
                      columns=col_name_list)

    df['gender'] = df['gender'].replace({0: "male",
                                         1: "female",
                                         2: "cat",
                                         3: "dog",
                                         4: "pig",
                                         5: "cat1",
                                         6: "cat2",
                                         7: "cat3",
                                         8: "cat4",
                                         9: "cat5", })
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

    return df