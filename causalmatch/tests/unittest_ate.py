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

import unittest
import pandas as pd
import numpy as np
from causalmatch.matching import matching
from sklearn.linear_model import LogisticRegression
def gen_unittest_data(n=5000, c_ratio=0.1) :
    """
    :n: number of observations
    :c_ratio: fraction of control obs
    """
    np.random.seed(123456)
    k_continuous = 3
    k_discrete = 3
    n_obs = n
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

    return df

class Testing(unittest.TestCase):

    @classmethod
    def test_ate(self):
        df = gen_unittest_data(n=10000, c_ratio=0.5)

        X_1 = ['c_1']
        X_2 = ['d_1']
        X_3 = ['d_1', 'd_3']
        X_4 = ['c_1', 'c_2', 'c_3']
        X_5 = ['c_1', 'd_1', 'd_3']
        y = ['y']
        id = 'user_id'
        T = 'treatment'


        # test when X only include 1 continuous variable
        match_obj = matching(data=df, T=T, X=X_1, id=id, y = y)
        match_obj.psm(n_neighbors=1, model=LogisticRegression(), trim_percentage=0.1, caliper=0.1, verbose=True)
        res_post, res_pre = match_obj.balance_check(include_discrete=True)
        df_res = match_obj.ate()

        test_1 = [res_post['Mean Treated post-match'].sum(), res_post['Mean Control post-match'].sum()]
        test_2 = [res_pre['Mean Treated pre-match'].sum(), res_pre['Mean Control pre-match'].sum()]
        test_3 = np.round(df_res['ate'].values[0], 2)
        np.testing.assert_almost_equal(test_1, [0.5005, 0.5008])
        np.testing.assert_almost_equal(test_2, [0.5004, 0.4984])
        np.testing.assert_almost_equal(test_3, 0.5)

        # test when X only include 1 discrete variable
        match_obj = matching(data=df, T=T, X=X_2, id=id, y = y)
        match_obj.psm(n_neighbors=1, model=LogisticRegression(), trim_percentage=0.1, caliper=0.1)
        res_post, res_pre = match_obj.balance_check(include_discrete=True)
        df_res = match_obj.ate()
        test_1 = [res_post['Mean Treated post-match'].sum(), res_post['Mean Control post-match'].sum()]
        test_2 = [res_pre['Mean Treated pre-match'].sum(), res_pre['Mean Control pre-match'].sum()]
        test_3 = np.round(df_res['ate'].values[0], 2)
        np.testing.assert_almost_equal(test_1, [0.7185, 0.7185])
        np.testing.assert_almost_equal(test_2, [0.7185, 0.7316])
        np.testing.assert_almost_equal(test_3, -0.78)

        # test when X only include multiple discrete variable
        match_obj = matching(data=df, T=T, X=X_3, id=id, y = y)
        match_obj.psm(n_neighbors=1, model=LogisticRegression(), trim_percentage=0.1, caliper=0.1)
        res_post, res_pre = match_obj.balance_check(include_discrete=True)
        df_res = match_obj.ate()
        test_1 = [res_post['Mean Treated post-match'].sum(), res_post['Mean Control post-match'].sum()]
        test_2 = [res_pre['Mean Treated pre-match'].sum(), res_pre['Mean Control pre-match'].sum()]
        test_3 = np.round(df_res['ate'].values[0], 2)
        np.testing.assert_almost_equal(test_1, [1.7193, 1.7193])
        np.testing.assert_almost_equal(test_2, [1.6933, 1.7007])
        np.testing.assert_almost_equal(test_3, 0.45)

        # test when X only include multiple continuous variable
        match_obj = matching(data=df, T=T, X=X_4, id=id, y = y)
        match_obj.psm(n_neighbors=1, model=LogisticRegression(), trim_percentage=0.1, caliper=0.1)
        res_post, res_pre = match_obj.balance_check(include_discrete=True)
        df_res = match_obj.ate()
        test_1 = [res_post['Mean Treated post-match'].sum(), res_post['Mean Control post-match'].sum()]
        test_2 = [res_pre['Mean Treated pre-match'].sum(), res_pre['Mean Control pre-match'].sum()]
        test_3 = np.round(df_res['ate'].values[0], 2)
        np.testing.assert_almost_equal(test_1, [1.5051, 1.5147])
        np.testing.assert_almost_equal(test_2, [1.5061, 1.5099])
        np.testing.assert_almost_equal(test_3, 0.44)

        # test when X only include both continuous and discrete variable
        match_obj = matching(data=df, T=T, X=X_5, id=id, y = y)
        match_obj.psm(n_neighbors=1, model=LogisticRegression(), trim_percentage=0.1, caliper=0.1)
        res_post, res_pre = match_obj.balance_check(include_discrete=True)
        df_res = match_obj.ate()
        test_1 = [res_post['Mean Treated post-match'].sum(), res_post['Mean Control post-match'].sum()]
        test_2 = [res_pre['Mean Treated pre-match'].sum(), res_pre['Mean Control pre-match'].sum()]
        test_3 = np.round(df_res['ate'].values[0], 2)
        np.testing.assert_almost_equal(test_1, [2.2399, 2.2389])
        np.testing.assert_almost_equal(test_2, [2.1937, 2.1991])
        np.testing.assert_almost_equal(test_3, 0.47)

if __name__ == '__main__':
    unittest.main()
