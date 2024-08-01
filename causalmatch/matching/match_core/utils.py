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

def data_process_bc(match_obj,
                    include_discrete):
    if match_obj.method == 'cem':
        # if process with cem method, previously did not do one-hot, do here for balance check
        if include_discrete:
            df_post_validate_x = pd.concat([
                match_obj.df_out_final[match_obj.X_numeric],
                pd.get_dummies(match_obj.df_out_final[match_obj.X_discrete],
                               columns = match_obj.X_discrete,
                               drop_first=True)  # categorical features converted to dummies
            ], axis=1)
        else:
            df_post_validate_x = match_obj.df_out_final[match_obj.X_numeric]

        df_post_validate = pd.concat([match_obj.df_out_final[[match_obj.T,match_obj.id]],
                                      df_post_validate_x], axis=1)
        X_balance_check = df_post_validate_x.columns
    else:
        if include_discrete:
            df_right = (match_obj.data_with_categ) * 1
        else:
            df_right = match_obj.data_with_categ[match_obj.X_numeric]

        X_balance_check = df_right.columns
        df_right[match_obj.id] = match_obj.data[match_obj.id]
        df_post_validate = match_obj.df_out_final_post_trim.merge(df_right, how='left', on=match_obj.id)
    return X_balance_check, df_post_validate

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