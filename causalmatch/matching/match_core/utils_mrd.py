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

    y_t_b_dot, y_t_s_dot, y_t_dot, sigma_t_2b, sigma_t_2s, sigma_t_bs = demean(y_t, n_users, n_shops)
    y_c_b_dot, y_c_s_dot, y_c_dot, sigma_c_2b, sigma_c_2s, sigma_c_bs = demean(y_c, n_users, n_shops)
    y_ib_b_dot, y_ib_s_dot, y_ib_dot, sigma_ib_2b, sigma_ib_2s, sigma_ib_bs = demean(y_ib, n_users, n_shops)
    y_is_b_dot, y_is_s_dot, y_is_dot, sigma_is_2b, sigma_is_2s, sigma_is_bs = demean(y_is, n_users, n_shops)

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

    t_stat_direct = tdirect / np.sqrt(v_tau_direct / df.shape[0])
    t_stat_s_spillover = ts / np.sqrt(v_s_spillover / df.shape[0])
    t_stat_b_spillover = tb / np.sqrt(v_b_spillover / df.shape[0])
    t_stat_tau = t / np.sqrt(v_tau / df.shape[0])

    pval_tau_direct = stats.t.sf(np.abs(t_stat_direct), df.shape[0] - 1) * 2
    pval_s_spillover = stats.t.sf(np.abs(t_stat_s_spillover), df.shape[0] - 1) * 2
    pval_b_spillover = stats.t.sf(np.abs(t_stat_b_spillover), df.shape[0] - 1) * 2
    pval_tau = stats.t.sf(np.abs(t_stat_tau), df.shape[0] - 1) * 2

    df_res = pd.DataFrame()
    df_res['parameters'] = ['tau', 'tau_tdirect', 'tau_b_spillover', 'tau_s_spillover']
    df_res['mean'] = [t, tdirect, tb, ts]
    df_res['variance'] = [v_tau, v_tau_direct, v_b_spillover, v_s_spillover]
    df_res['t_stat'] = [t_stat_tau, t_stat_direct, t_stat_b_spillover, t_stat_s_spillover]
    df_res['p_values'] = [pval_tau, pval_tau_direct, pval_b_spillover, pval_s_spillover]

    return df_res