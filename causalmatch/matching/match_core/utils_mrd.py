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

import pandas as pd
import numpy as np
import scipy.stats as stats
from causalmatch.matching.match_core.utils import demean

def mrd_estimation(mrd_obj) :
    df = mrd_obj.data
    idb = mrd_obj.idb
    ids = mrd_obj.ids
    tb = mrd_obj.tb
    ts = mrd_obj.ts
    y = mrd_obj.y
    user_exp_list = mrd_obj.user_exp_list
    shop_exp_list = mrd_obj.shop_exp_list
    n_users = mrd_obj.n_users
    n_shops = mrd_obj.n_shops
    n_users_t = n_users - len(user_exp_list)
    n_shops_t = n_shops - len(shop_exp_list)
    n_pairs = (n_users - 1) * (n_shops - 1)
    dof = df[((df['treatment_u'] == 1) & (df['treatment_s'] == 1)) | (
                (df['treatment_u'] == 0) & (df['treatment_s'] == 0))].shape[0]

    df['status'] = 'unknown'
    df['status'].loc[((df[idb].isin(user_exp_list)) & df[ids].isin(shop_exp_list))] = 't'
    df['status'].loc[((df[idb].isin(user_exp_list)) & ~df[ids].isin(shop_exp_list))] = 'ib'
    df['status'].loc[(~(df[idb].isin(user_exp_list)) & df[ids].isin(shop_exp_list))] = 'is'
    df['status'].loc[(~(df[idb].isin(user_exp_list)) & ~df[ids].isin(shop_exp_list))] = 'c'

    # group mean
    y_11_mean = df[(df[tb] == 1) & (df[ts] == 1)][y].mean()
    y_00_mean = df[(df[tb] == 0) & (df[ts] == 0)][y].mean()
    y_10_mean = df[(df[tb] == 1) & (df[ts] == 0)][y].mean()
    y_01_mean = df[(df[tb] == 0) & (df[ts] == 1)][y].mean()

    # Bajari P, Burdick B, Imbens G W, et al. Multiple randomization designs[J]. p17, treatment effect:
    t = y_11_mean - y_00_mean  # 1. tau
    tdirect = y_11_mean - y_10_mean - y_01_mean + y_00_mean  # 2. t-direct
    tb = y_10_mean - y_00_mean  # 3. t-spillover b
    ts = y_01_mean - y_00_mean  # 4. t-spillover s

    df_pivot_y = df.pivot(index=idb, columns=ids, values=y)
    df_pivot_status = df.pivot(index=idb, columns=ids, values=['status'])
    y_t = df_pivot_y.values * ((df_pivot_status.values == 't') * 1)
    y_ib = df_pivot_y.values * ((df_pivot_status.values == 'ib') * 1)
    y_is = df_pivot_y.values * ((df_pivot_status.values == 'is') * 1)
    y_c = df_pivot_y.values * ((df_pivot_status.values == 'c') * 1)

    # for missing values, replace nan with 0
    y_t[np.isnan(y_t)]   = 0
    y_ib[np.isnan(y_ib)] = 0
    y_is[np.isnan(y_is)] = 0
    y_c[np.isnan(y_c)]   = 0

    y_t_b_dot, y_t_s_dot, y_t_dot, sigma_t_2b, sigma_t_2s, sigma_t_bs = demean(y_t, n_pairs, n_users, n_shops, n_users_t, n_shops_t, 't')
    y_c_b_dot, y_c_s_dot, y_c_dot, sigma_c_2b, sigma_c_2s, sigma_c_bs = demean(y_c, n_pairs, n_users, n_shops, n_users_t, n_shops_t, 'c')
    y_ib_b_dot, y_ib_s_dot, y_ib_dot, sigma_ib_2b, sigma_ib_2s, sigma_ib_bs = demean(y_ib, n_pairs, n_users, n_shops, n_users_t, n_shops_t, 'ib')
    y_is_b_dot, y_is_s_dot, y_is_dot, sigma_is_2b, sigma_is_2s, sigma_is_bs = demean(y_is, n_pairs, n_users, n_shops, n_users_t, n_shops_t, 'is')

    # Covariance matrix used for Lemma A.5
    sigma_tc_2b = 1 / (n_users - 1) * np.sum((y_t_b_dot - y_c_b_dot) * (y_t_b_dot - y_c_b_dot))
    sigma_tc_2s = 1 / (n_shops - 1) * np.sum((y_t_s_dot - y_c_s_dot) * (y_t_s_dot - y_c_s_dot))
    sigma_tc_2bs = 1 / ((n_users - 1) * (n_shops - 1)) * np.sum((y_t_dot - y_c_dot) * (y_t_dot - y_c_dot))

    sigma_tib_2b = 1 / (n_users - 1) * np.sum((y_t_b_dot - y_ib_b_dot) * (y_t_b_dot - y_ib_b_dot))
    sigma_tib_2s = 1 / (n_shops - 1) * np.sum((y_t_s_dot - y_ib_s_dot) * (y_t_s_dot - y_ib_s_dot))
    sigma_tib_2bs = 1 / ((n_users - 1) * (n_shops - 1)) * np.sum((y_t_dot - y_ib_dot) * (y_t_dot - y_ib_dot))

    sigma_tis_2b = 1 / (n_users - 1) * np.sum((y_t_b_dot - y_is_b_dot) * (y_t_b_dot - y_is_b_dot))
    sigma_tis_2s = 1 / (n_shops - 1) * np.sum((y_t_s_dot - y_is_s_dot) * (y_t_s_dot - y_is_s_dot))
    sigma_tis_2bs = 1 / ((n_users - 1) * (n_shops - 1)) * np.sum((y_t_dot - y_is_dot) * (y_t_dot - y_is_dot))

    sigma_ibis_2b = 1 / (n_users - 1) * np.sum((y_ib_b_dot - y_is_b_dot) * (y_ib_b_dot - y_is_b_dot))
    sigma_ibis_2s = 1 / (n_shops - 1) * np.sum((y_ib_s_dot - y_is_s_dot) * (y_ib_s_dot - y_is_s_dot))
    sigma_ibis_2bs = 1 / ((n_users - 1) * (n_shops - 1)) * np.sum((y_ib_dot - y_is_dot) * (y_ib_dot - y_is_dot))

    sigma_ibc_2b = 1 / (n_users - 1) * np.sum((y_ib_b_dot - y_c_b_dot) * (y_ib_b_dot - y_c_b_dot))
    sigma_ibc_2s = 1 / (n_shops - 1) * np.sum((y_ib_s_dot - y_c_s_dot) * (y_ib_s_dot - y_c_s_dot))
    sigma_ibc_2bs = 1 / ((n_users - 1) * (n_shops - 1)) * np.sum((y_ib_dot - y_c_dot) * (y_ib_dot - y_c_dot))

    sigma_isc_2b = 1 / (n_users - 1) * np.sum((y_is_b_dot - y_c_b_dot) * (y_is_b_dot - y_c_b_dot))
    sigma_isc_2s = 1 / (n_shops - 1) * np.sum((y_is_s_dot - y_c_s_dot) * (y_is_s_dot - y_c_s_dot))
    sigma_isc_2bs = 1 / ((n_users - 1) * (n_shops - 1)) * np.sum((y_is_dot - y_c_dot) * (y_is_dot - y_c_dot))

    # Lemma A.4
    i = n_users
    j = n_shops
    i_c = n_users - len(user_exp_list)
    i_t = len(user_exp_list)
    j_c = n_shops - len(shop_exp_list)
    j_t = len(shop_exp_list)

    v_t = i_c / (i_t * i) * sigma_t_2b + j_c / (j_t * j) * sigma_t_2s + i_c / (i_t * i) * j_c / (j_t * j) * sigma_t_bs
    v_ib = i_c / (i_t * i) * sigma_ib_2b + j_c / (j_t * j) * sigma_ib_2s + i_c / (i_t * i) * j_c / (
                j_t * j) * sigma_ib_bs
    v_is = i_c / (i_t * i) * sigma_is_2b + j_c / (j_t * j) * sigma_is_2s + i_c / (i_t * i) * j_c / (
                j_t * j) * sigma_is_bs
    v_c = i_c / (i_t * i) * sigma_c_2b + j_c / (j_t * j) * sigma_c_2s + i_c / (i_t * i) * j_c / (j_t * j) * sigma_c_bs

    # Lemma A.5
    c_t_ib = i_c / (2 * i_t * i) * (sigma_t_2b + sigma_ib_2b - sigma_tib_2b) \
             - 1 / (2 * j) * (sigma_t_2s + sigma_ib_2s - sigma_tib_2s) \
             - i_c / (2 * i_t * i * j) * (sigma_t_bs + sigma_ib_bs - sigma_tib_2bs)

    c_t_is = -1 / (2 * i) * (sigma_t_2b + sigma_is_2b - sigma_tis_2b) \
             + j_c / (2 * j_t * j) * (sigma_t_2s + sigma_is_2s - sigma_tis_2s) \
             - j_c / (2 * i * j_t * j) * (sigma_t_bs + sigma_is_bs - sigma_tis_2bs)

    c_t_c = -1 / (2 * i) * (sigma_t_2b + sigma_c_2b - sigma_tc_2b) \
            - 1 / (2 * j) * (sigma_t_2s + sigma_c_2s - sigma_tc_2s) \
            + 1 / (2 * i * j) * (sigma_t_bs + sigma_c_bs - sigma_tc_2bs)

    c_ib_is = -1 / (2 * i) * (sigma_ib_2b + sigma_is_2b - sigma_ibis_2b) \
              - 1 / (2 * j) * (sigma_ib_2s + sigma_is_2s - sigma_ibis_2s) \
              + 1 / (2 * i * j) * (sigma_ib_bs + sigma_is_bs - sigma_ibis_2bs)

    c_ib_c = -1 / (2 * i) * (sigma_ib_2b + sigma_c_2b - sigma_ibc_2b) \
             + j_c / (2 * j_t * j) * (sigma_ib_2s + sigma_c_2s - sigma_ibc_2s) \
             - j_c / (2 * i * j_t * j) * (sigma_ib_bs + sigma_c_bs - sigma_ibc_2bs)

    c_is_c = i_c / (2 * i_t * i) * (sigma_is_2b + sigma_c_2b - sigma_isc_2b) \
             - 1 / (2 * j) * (sigma_is_2s + sigma_c_2s - sigma_isc_2s) \
             - i_c / (2 * i_t * i * j) * (sigma_is_bs + sigma_c_bs - sigma_isc_2bs)

    v_tau_direct = v_t + v_ib + v_is + v_c - 2 * c_t_ib - 2 * c_t_is + 2 * c_t_c + 2 * c_ib_is - 2 * c_ib_c - 2 * c_is_c
    v_s_spillover = v_is + v_c - 2 * c_is_c
    v_b_spillover = v_ib + v_c - 2 * c_ib_c
    v_tau = v_t + v_c - 2 * c_t_c

    t_stat_direct = tdirect / np.sqrt(v_tau_direct)
    t_stat_s_spillover = ts / np.sqrt(v_s_spillover)
    t_stat_b_spillover = tb / np.sqrt(v_b_spillover)
    t_stat_tau = t / np.sqrt(v_tau)

    pval_tau_direct = stats.t.sf(np.abs(t_stat_direct), dof - 1) * 2
    pval_s_spillover = stats.t.sf(np.abs(t_stat_s_spillover), df.shape[0] - 1) * 2
    pval_b_spillover = stats.t.sf(np.abs(t_stat_b_spillover), df.shape[0] - 1) * 2
    pval_tau = stats.t.sf(np.abs(t_stat_tau), dof - 1) * 2

    df_res = pd.DataFrame()
    df_res['parameters'] = ['tau', 'tau_tdirect', 'tau_b_spillover', 'tau_s_spillover']
    df_res['mean'] = [t, tdirect, tb, ts]
    df_res['variance'] = [v_tau, v_tau_direct, v_b_spillover, v_s_spillover]
    df_res['t_stat'] = [t_stat_tau, t_stat_direct, t_stat_b_spillover, t_stat_s_spillover]
    df_res['p_values'] = [pval_tau, pval_tau_direct, pval_b_spillover, pval_s_spillover]

    return df_res



def mrd_estimation2(mrd_obj) :
    df = mrd_obj.data
    idb = mrd_obj.idb
    ids = mrd_obj.ids
    tb = mrd_obj.tb
    ts = mrd_obj.ts
    y = mrd_obj.y
    user_exp_list = mrd_obj.user_exp_list
    shop_exp_list = mrd_obj.shop_exp_list
    n_users = mrd_obj.n_users
    n_shops = mrd_obj.n_shops
    n_users_t = n_users - len(user_exp_list)
    n_shops_t = n_shops - len(shop_exp_list)
    n_pairs = (n_users - 1) * (n_shops - 1)

    user_exp_list = df[df[tb] == 1][idb].value_counts().index.to_list()
    shop_exp_list = df[df[ts] == 1][ids].value_counts().index.to_list()

    df['status'] = 'unknown'
    df['status'].loc[((df[idb].isin(user_exp_list)) & df[ids].isin(shop_exp_list))] = 't'
    df['status'].loc[((df[idb].isin(user_exp_list)) & ~df[ids].isin(shop_exp_list))] = 'ib'
    df['status'].loc[(~(df[idb].isin(user_exp_list)) & df[ids].isin(shop_exp_list))] = 'is'
    df['status'].loc[(~(df[idb].isin(user_exp_list)) & ~df[ids].isin(shop_exp_list))] = 'c'

    n_users = df['user_id'].value_counts().shape[0]
    n_users_t = len(user_exp_list)
    n_users_c = n_users - n_users_t

    n_shops = df['shop_id'].value_counts().shape[0]
    n_shops_t = len(shop_exp_list)
    n_shops_c = n_shops - n_shops_t

    n_pairs = (n_users - 1) * (n_shops - 1)
    # n_pairs_ib = df[df['status'] == 'ib'].shape[0]
    # n_pairs_is = df[df['status'] == 'is'].shape[0]
    # n_pairs_t = df[df['status'] == 't'].shape[0]
    # n_pairs_c = df[df['status'] == 'c'].shape[0]

    user_list = df[idb].value_counts().index.to_list()
    shop_list = df[ids].value_counts().index.to_list()

    df_u = pd.DataFrame(data=user_list, columns=['user_id'])
    df_s = pd.DataFrame(data=shop_list, columns=['shop_id'])



    y_11_mean = df[(df[tb] == 1) & (df[ts] == 1)][y].sum()/(n_users_t * n_shops_t)
    y_00_mean = df[(df[tb] == 0) & (df[ts] == 0)][y].sum()/(n_users_c * n_shops_c)
    y_10_mean = df[(df[tb] == 1) & (df[ts] == 0)][y].sum()/(n_users_t * n_shops_c)
    y_01_mean = df[(df[tb] == 0) & (df[ts] == 1)][y].sum()/(n_users_c * n_shops_t)

    # 1. tau
    t = y_11_mean - y_00_mean
    # 2. t-direct
    tdirect = y_11_mean - y_10_mean - y_01_mean + y_00_mean
    # 3. t-spillover b
    tb = y_10_mean - y_00_mean
    # 4. t-spillover s
    ts = y_01_mean - y_00_mean


    df['y_t']  = np.where((df[idb].isin(user_exp_list) & df[ids].isin(shop_exp_list)), df[y],0)
    df['y_ib'] = np.where(((df[idb].isin(user_exp_list)) & ~df[ids].isin(shop_exp_list)), df[y],0)
    df['y_is'] = np.where((~(df[idb].isin(user_exp_list)) & df[ids].isin(shop_exp_list)), df[y],0)
    df['y_c']  = np.where((~(df[idb].isin(user_exp_list)) & ~df[ids].isin(shop_exp_list)), df[y],0)


    ## 非矩阵的形式

    df_y_t_b_bar = pd.DataFrame(df.groupby('user_id')['y_t'].sum()/ n_shops_t)
    df_y_t_s_bar = pd.DataFrame(df.groupby('shop_id')['y_t'].sum()/ n_users_t)

    df_y_t_b_bar.reset_index(inplace=True)
    df_y_t_s_bar.reset_index(inplace=True)

    df_y_t_b_bar.rename(columns={'y_t' : "y_t_b_bar"}, inplace=True)
    df_y_t_s_bar.rename(columns={'y_t' : "y_t_s_bar"}, inplace=True)
    df_y_t_bar_bar = df['y_t'].sum() / ((n_users_t - 1) *(n_shops_t - 1))

    # 这三个需要output
    df_y_t_b_bar['y_t_b_dot'] = df_y_t_b_bar['y_t_b_bar'] - df_y_t_bar_bar
    df_y_t_s_bar['y_t_s_dot'] = df_y_t_s_bar['y_t_s_bar'] - df_y_t_bar_bar
    df = df.merge(df_y_t_b_bar[['user_id'
                                ,'y_t_b_bar'
                                ,'y_t_b_dot']], how='left', on='user_id').merge(df_y_t_s_bar[['shop_id'
                                                                                              ,'y_t_s_bar'
                                                                                              ,'y_t_s_dot']], how='left', on='shop_id')
    df['y_t_dot'] = df['y_t'] - df['y_t_b_bar'] - df['y_t_s_bar'] + df_y_t_bar_bar


    sigma_t_2b_ = np.sum(df_y_t_b_bar['y_t_b_dot'] * df_y_t_b_bar['y_t_b_dot']) / (n_users_t - 1)
    sigma_t_2s_ = np.sum(df_y_t_s_bar['y_t_s_dot'] * df_y_t_s_bar['y_t_s_dot']) / (n_shops_t - 1)
    sigma_t_bs_ = np.sum(df['y_t_dot'] * df['y_t_dot']) / ((n_users_t - 1) *(n_shops_t - 1))

    # 每个用户的店均
    # 每个店铺的人均
    df_y_ib_b_bar = pd.DataFrame(df.groupby('user_id')['y_ib'].sum() / n_shops_c)
    df_y_ib_s_bar = pd.DataFrame(df.groupby('shop_id')['y_ib'].sum() / n_users_t)
    df_y_ib_b_bar.reset_index(inplace=True)
    df_y_ib_s_bar.reset_index(inplace=True)

    df_y_ib_b_bar.rename(columns={'y_ib' : "y_ib_b_bar"}, inplace=True)
    df_y_ib_s_bar.rename(columns={'y_ib' : "y_ib_s_bar"}, inplace=True)
    df_y_ib_bar_bar = df['y_ib'].sum() / ((n_users_t - 1) * (n_shops_c - 1))

    # 这三个需要output
    df_y_ib_b_bar['y_ib_b_dot'] = df_y_ib_b_bar['y_ib_b_bar'] - df_y_ib_bar_bar
    df_y_ib_s_bar['y_ib_s_dot'] = df_y_ib_s_bar['y_ib_s_bar'] - df_y_ib_bar_bar
    df = df.merge(df_y_ib_b_bar[['user_id', 'y_ib_b_bar', 'y_ib_b_dot']], how='left', on='user_id').merge(
        df_y_ib_s_bar[['shop_id', 'y_ib_s_bar', 'y_ib_s_dot']], how='left', on='shop_id')
    df['y_ib_dot'] = df['y_ib'] - df['y_ib_b_bar'] - df['y_ib_s_bar'] + df_y_ib_bar_bar

    sigma_ib_2b_ = 1 / (n_users_t - 1) * np.sum(df_y_ib_b_bar['y_ib_b_dot'] * df_y_ib_b_bar['y_ib_b_dot'])
    sigma_ib_2s_ = 1 / (n_shops_c - 1) * np.sum(df_y_ib_s_bar['y_ib_s_dot'] * df_y_ib_s_bar['y_ib_s_dot'])
    sigma_ib_bs_ = 1 / ((n_users_t - 1) * (n_shops_c - 1)) * np.sum(df['y_ib_dot'] * df['y_ib_dot'])

    # df['y_is']

    # 每个用户的店均
    # 每个店铺的人均
    df_y_is_b_bar = pd.DataFrame(df.groupby('user_id')['y_is'].sum() / n_shops_t)
    df_y_is_s_bar = pd.DataFrame(df.groupby('shop_id')['y_is'].sum() / n_users_c)
    df_y_is_b_bar.reset_index(inplace=True)
    df_y_is_s_bar.reset_index(inplace=True)

    df_y_is_b_bar.rename(columns={'y_is' : "y_is_b_bar"}, inplace=True)
    df_y_is_s_bar.rename(columns={'y_is' : "y_is_s_bar"}, inplace=True)
    df_y_is_bar_bar = df['y_is'].sum() / ((n_users_c - 1) * (n_shops_t - 1))

    # 这三个需要output
    df_y_is_b_bar['y_is_b_dot'] = df_y_is_b_bar['y_is_b_bar'] - df_y_is_bar_bar
    df_y_is_s_bar['y_is_s_dot'] = df_y_is_s_bar['y_is_s_bar'] - df_y_is_bar_bar
    df = df.merge(df_y_is_b_bar[['user_id', 'y_is_b_bar', 'y_is_b_dot']], how='left', on='user_id').merge(
        df_y_is_s_bar[['shop_id', 'y_is_s_bar', 'y_is_s_dot']], how='left', on='shop_id')
    df['y_is_dot'] = df['y_is'] - df['y_is_b_bar'] - df['y_is_s_bar'] + df_y_is_bar_bar

    sigma_is_2b_ = 1 / (n_users_c - 1) * np.sum(df_y_is_b_bar['y_is_b_dot'] * df_y_is_b_bar['y_is_b_dot'])
    sigma_is_2s_ = 1 / (n_shops_t - 1) * np.sum(df_y_is_s_bar['y_is_s_dot'] * df_y_is_s_bar['y_is_s_dot'])
    sigma_is_bs_ = 1 / ((n_users_c - 1) * (n_shops_t - 1)) * np.sum(df['y_is_dot'] * df['y_is_dot'])
    # sigma_is_bs_ = 1 / (n_pairs) * np.sum(df['y_is_dot'] * df['y_is_dot'])

    # df['y_c']

    # 每个用户的店均
    # 每个店铺的人均
    df_y_c_b_bar = pd.DataFrame(df.groupby('user_id')['y_c'].sum() / n_shops_c)
    df_y_c_s_bar = pd.DataFrame(df.groupby('shop_id')['y_c'].sum() / n_users_c)
    df_y_c_b_bar.reset_index(inplace=True)
    df_y_c_s_bar.reset_index(inplace=True)

    df_y_c_b_bar.rename(columns={'y_c' : "y_c_b_bar"}, inplace=True)
    df_y_c_s_bar.rename(columns={'y_c' : "y_c_s_bar"}, inplace=True)
    df_y_c_bar_bar = df['y_c'].sum() / ((n_users_c - 1) * (n_shops_c - 1))

    # 这三个需要output
    df_y_c_b_bar['y_c_b_dot'] = df_y_c_b_bar['y_c_b_bar'] - df_y_c_bar_bar
    df_y_c_s_bar['y_c_s_dot'] = df_y_c_s_bar['y_c_s_bar'] - df_y_c_bar_bar
    df = df.merge(df_y_c_b_bar[['user_id', 'y_c_b_bar', 'y_c_b_dot']], how='left', on='user_id').merge(
        df_y_c_s_bar[['shop_id', 'y_c_s_bar', 'y_c_s_dot']], how='left', on='shop_id')
    df['y_c_dot'] = df['y_c'] - df['y_c_b_bar'] - df['y_c_s_bar'] + df_y_c_bar_bar

    sigma_c_2b_ = 1 / (n_users_c - 1) * np.sum(df_y_c_b_bar['y_c_b_dot'] * df_y_c_b_bar['y_c_b_dot'])
    sigma_c_2s_ = 1 / (n_shops_c - 1) * np.sum(df_y_c_s_bar['y_c_s_dot'] * df_y_c_s_bar['y_c_s_dot'])
    sigma_c_bs_ = 1 / ((n_users_c - 1) * (n_shops_c - 1)) * np.sum(df['y_c_dot'] * df['y_c_dot'])

    sigma_tc_2b_ = 1 / (n_users - 1) * np.sum((df_y_t_b_bar['y_t_b_dot'] - df_y_c_b_bar['y_c_b_dot']) * (
                df_y_t_b_bar['y_t_b_dot'] - df_y_c_b_bar['y_c_b_dot']))
    sigma_tc_2s_ = 1 / (n_shops - 1) * np.sum((df_y_t_s_bar['y_t_s_dot'] - df_y_c_s_bar['y_c_s_dot']) * (
                df_y_t_s_bar['y_t_s_dot'] - df_y_c_s_bar['y_c_s_dot']))
    sigma_tc_2bs_ = 1 / (n_pairs) * np.sum((df['y_t_dot'] - df['y_c_dot']) * (df['y_t_dot'] - df['y_c_dot']))

    sigma_tib_2b_ = 1 / (n_users - 1) * np.sum((df_y_t_b_bar['y_t_b_dot'] - df_y_ib_b_bar['y_ib_b_dot']) * (
                df_y_t_b_bar['y_t_b_dot'] - df_y_ib_b_bar['y_ib_b_dot']))
    sigma_tib_2s_ = 1 / (n_shops - 1) * np.sum((df_y_t_s_bar['y_t_s_dot'] - df_y_ib_s_bar['y_ib_s_dot']) * (
                df_y_t_s_bar['y_t_s_dot'] - df_y_ib_s_bar['y_ib_s_dot']))
    sigma_tib_2bs_ = 1 / (n_pairs) * np.sum((df['y_t_dot'] - df['y_ib_dot']) * (df['y_t_dot'] - df['y_ib_dot']))

    sigma_tis_2b_ = 1 / (n_users - 1) * np.sum((df_y_t_b_bar['y_t_b_dot'] - df_y_is_b_bar['y_is_b_dot']) * (
                df_y_t_b_bar['y_t_b_dot'] - df_y_is_b_bar['y_is_b_dot']))
    sigma_tis_2s_ = 1 / (n_shops - 1) * np.sum((df_y_t_s_bar['y_t_s_dot'] - df_y_is_s_bar['y_is_s_dot']) * (
                df_y_t_s_bar['y_t_s_dot'] - df_y_is_s_bar['y_is_s_dot']))
    sigma_tis_2bs_ = 1 / (n_pairs) * np.sum((df['y_t_dot'] - df['y_is_dot']) * (df['y_t_dot'] - df['y_is_dot']))

    sigma_ibis_2b_ = 1 / (n_users - 1) * np.sum((df_y_ib_b_bar['y_ib_b_dot'] - df_y_is_b_bar['y_is_b_dot']) * (
                df_y_ib_b_bar['y_ib_b_dot'] - df_y_is_b_bar['y_is_b_dot']))
    sigma_ibis_2s_ = 1 / (n_shops - 1) * np.sum((df_y_ib_s_bar['y_ib_s_dot'] - df_y_is_s_bar['y_is_s_dot']) * (
                df_y_ib_s_bar['y_ib_s_dot'] - df_y_is_s_bar['y_is_s_dot']))
    sigma_ibis_2bs_ = 1 / (n_pairs) * np.sum((df['y_ib_dot'] - df['y_is_dot']) * (df['y_ib_dot'] - df['y_is_dot']))

    sigma_ibc_2b_ = 1 / (n_users - 1) * np.sum((df_y_ib_b_bar['y_ib_b_dot'] - df_y_c_b_bar['y_c_b_dot']) * (
                df_y_ib_b_bar['y_ib_b_dot'] - df_y_c_b_bar['y_c_b_dot']))
    sigma_ibc_2s_ = 1 / (n_shops - 1) * np.sum((df_y_ib_s_bar['y_ib_s_dot'] - df_y_c_s_bar['y_c_s_dot']) * (
                df_y_ib_s_bar['y_ib_s_dot'] - df_y_c_s_bar['y_c_s_dot']))
    sigma_ibc_2bs_ = 1 / (n_pairs) * np.sum((df['y_ib_dot'] - df['y_c_dot']) * (df['y_ib_dot'] - df['y_c_dot']))

    sigma_isc_2b_ = 1 / (n_users - 1) * np.sum((df_y_is_b_bar['y_is_b_dot'] - df_y_c_b_bar['y_c_b_dot']) * (
                df_y_is_b_bar['y_is_b_dot'] - df_y_c_b_bar['y_c_b_dot']))
    sigma_isc_2s_ = 1 / (n_shops - 1) * np.sum((df_y_is_s_bar['y_is_s_dot'] - df_y_c_s_bar['y_c_s_dot']) * (
                df_y_is_s_bar['y_is_s_dot'] - df_y_c_s_bar['y_c_s_dot']))
    sigma_isc_2bs_ = 1 / (n_pairs) * np.sum((df['y_is_dot'] - df['y_c_dot']) * (df['y_is_dot'] - df['y_c_dot']))



    # 新的
    # Lemma A.4
    i = n_users
    j = n_shops
    i_c = n_users - len(user_exp_list)
    i_t = len(user_exp_list)
    j_c = n_shops - len(shop_exp_list)
    j_t = len(shop_exp_list)


    v_t  = i_c / (i_t * i) * sigma_t_2b_  + j_c / (j_t * j) * sigma_t_2s_  + i_c / (i_t * i) * j_c / (j_t * j) * sigma_t_bs_
    v_ib = i_c / (i_t * i) * sigma_ib_2b_ + j_c / (j_t * j) * sigma_ib_2s_ + i_c / (i_t * i) * j_c / (j_t * j) * sigma_ib_bs_
    v_is = i_c / (i_t * i) * sigma_is_2b_ + j_c / (j_t * j) * sigma_is_2s_ + i_c / (i_t * i) * j_c / (j_t * j) * sigma_is_bs_
    v_c  = i_c / (i_t * i) * sigma_c_2b_  + j_c / (j_t * j) * sigma_c_2s_  + i_c / (i_t * i) * j_c / (j_t * j) * sigma_c_bs_

    # Lemma A.5
    c_t_ib = i_c / (2 * i_t * i) * (sigma_t_2b_ + sigma_ib_2b_ - sigma_tib_2b_) \
             - 1 / (2 * j) * (sigma_t_2s_ + sigma_ib_2s_ - sigma_tib_2s_) \
             - i_c / (2 * i_t * i * j) * (sigma_t_bs_ + sigma_ib_bs_ - sigma_tib_2bs_)

    c_t_is = -1 / (2 * i) * (sigma_t_2b_ + sigma_is_2b_ - sigma_tis_2b_) \
             + j_c / (2 * j_t * j) * (sigma_t_2s_ + sigma_is_2s_ - sigma_tis_2s_) \
             - j_c / (2 * i * j_t * j) * (sigma_t_bs_ + sigma_is_bs_ - sigma_tis_2bs_)

    c_t_c = -1 / (2 * i) * (sigma_t_2b_ + sigma_c_2b_ - sigma_tc_2b_) \
            - 1 / (2 * j) * (sigma_t_2s_ + sigma_c_2s_ - sigma_tc_2s_) \
            + 1 / (2 * i * j) * (sigma_t_bs_ + sigma_c_bs_ - sigma_tc_2bs_)

    c_ib_is = -1 / (2 * i) * (sigma_ib_2b_ + sigma_is_2b_ - sigma_ibis_2b_) \
              - 1 / (2 * j) * (sigma_ib_2s_ + sigma_is_2s_ - sigma_ibis_2s_) \
              + 1 / (2 * i * j) * (sigma_ib_bs_ + sigma_is_bs_ - sigma_ibis_2bs_)

    c_ib_c = -1 / (2 * i) * (sigma_ib_2b_ + sigma_c_2b_ - sigma_ibc_2b_) \
             + j_c / (2 * j_t * j) * (sigma_ib_2s_ + sigma_c_2s_ - sigma_ibc_2s_) \
             - j_c / (2 * i * j_t * j) * (sigma_ib_bs_ + sigma_c_bs_ - sigma_ibc_2bs_)

    c_is_c = i_c / (2 * i_t * i) * (sigma_is_2b_ + sigma_c_2b_ - sigma_isc_2b_) \
             - 1 / (2 * j) * (sigma_is_2s_ + sigma_c_2s_ - sigma_isc_2s_) \
             - i_c / (2 * i_t * i * j) * (sigma_is_bs_ + sigma_c_bs_ - sigma_isc_2bs_)

    v_tau_direct = v_t + v_ib + v_is + v_c - 2 * c_t_ib - 2 * c_t_is + 2 * c_t_c + 2 * c_ib_is - 2 * c_ib_c - 2 * c_is_c
    v_s_spillover = v_is + v_c - 2 * c_is_c
    v_b_spillover = v_ib + v_c - 2 * c_ib_c
    v_tau = v_t + v_c - 2 * c_t_c

    # dof   = df[((df['treatment_u'] ==1) & (df['treatment_s'] == 1))|((df['treatment_u'] ==0) & (df['treatment_s'] == 0))].shape[0]
    # dof_b = df[((df['treatment_u'] ==1) & (df['treatment_s'] == 0))|((df['treatment_u'] ==0) & (df['treatment_s'] == 0))].shape[0]
    # dof_s = df[((df['treatment_u'] ==0) & (df['treatment_s'] == 1))|((df['treatment_u'] ==0) & (df['treatment_s'] == 0))].shape[0]

    dof = (n_shops_t-1) * (n_users_t-1) + (n_shops_c-1) * (n_users_c-1)
    dof_b = (n_shops_c-1) * (n_users_t-1) + (n_shops_c-1) * (n_users_c-1)
    dof_s = (n_shops_t-1) * (n_users_c-1) + (n_shops_c-1) * (n_users_c-1)

    t_stat_direct = tdirect / np.sqrt(v_tau_direct)
    t_stat_tau = t / np.sqrt(v_tau )
    t_stat_b_spillover = tb / np.sqrt(v_b_spillover)
    t_stat_s_spillover = ts / np.sqrt(v_s_spillover)


    pval_tau_direct  = stats.t.sf(np.abs(t_stat_direct), dof - 1) * 2
    pval_tau = stats.t.sf(np.abs(t_stat_tau), dof - 1) * 2
    pval_b_spillover = stats.t.sf(np.abs(t_stat_b_spillover), dof_b - 1) * 2
    pval_s_spillover = stats.t.sf(np.abs(t_stat_s_spillover), dof_s - 1) * 2



    df_res = pd.DataFrame()
    df_res['parameters'] = ['tau', 'tau_tdirect', 'tau_b_spillover', 'tau_s_spillover']
    df_res['mean'] = [t, tdirect, tb, ts]
    df_res['variance'] = [v_tau, v_tau_direct, v_b_spillover, v_s_spillover]
    df_res['t_stat'] = [t_stat_tau, t_stat_direct, t_stat_b_spillover, t_stat_s_spillover]
    df_res['p_values'] = [pval_tau, pval_tau_direct, pval_b_spillover, pval_s_spillover]

    return df_res
