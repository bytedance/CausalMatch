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
from pandas.api.types import is_string_dtype,is_numeric_dtype,is_float_dtype

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from typing import List, Dict
import scipy.stats as stats
from .psm import psm
from .cem import calculate_weight, bin_cut, sample_k2k
from .utils import calculate_smd, data_process_bc
import warnings
import statsmodels.api as sm

class matching:
    def __init__(self,
                 data: pd.DataFrame,
                 T: str,
                 X: List[str],
                 y: List[str]=[],
                 method: str = "psm",
                 id: str = None):
        """
        Initialize matching object, include two methods: psm and cem.

        Parameters
        ----------
        :param data: Pandas dataframe, should include feature columns and treatment column.
        :param T: str, must input.
            Treatment variable name included in dataframe. This version only support
            dummy treatment, i.e 0-1 binary treatment.
        :param X: list[str], must input
            List of feature names you want to include. Can be float, int, or str.
        :param method: str, default psm.
            Whether to use psm method or cem method for matching.
        :param id: str, default none.
            ID column, can be user id or device id or any level id that you want to match.
            ID should be row-unique, ie. one id only appear once in your dataframe input.
        """

        # original input
        self.data = data
        self.T = T
        self.X = X
        self.y = y
        self.method = method
        self.id = id

        # reserve for output
        self.data_with_categ  = None
        self.col_name_x_expand = None

        self.df_out_final     = None
        self.data_out_treat   = None
        self.data_out_control = None
        self.model            = None
        self.df_out_final_trim_caliper    = None
        self.df_out_final_trim_percentage = None
        self.df_out_final_post_trim       = None
        self.n_bins = None
        self.break_points = None
        self.cluster_criteria = None
        self.replace = None

        # data-preprocess
        self.preprocess()


    def preprocess(self):

        # 1. whether T is numeric type
        if is_numeric_dtype(self.data[self.T]) == False:
            raise TypeError('Treatment column in your dataframe is not numeric type, please transfer to numeric type.')


        # 2.x and t are contained in dataframe
        col_list = set(self.data.columns)
        input_vars = [self.T] + self.X
        for i in input_vars:
            if i not in col_list:
                raise TypeError('Variable {} is not in the input dataframe.'.format(i))

        # 3. if ID column is generated
        if self.id is None:
            id = 'id'
            self.data[id] = self.data.index
            self.id = id
        else:
            id = self.id
            if is_float_dtype(self.data[id]) is True:
                warnings.warn("ID columns is float type, cannot support, convert to int type.")
                self.data[id] = self.data[id].astype(int)
            else:
                self.data[id] = self.data[id]


        data = self.data
        # 4. If ID is unique, if is float
        id_check = data.groupby(id)[id].nunique() > 1
        if len(id_check[id_check == True])>0:
            raise TypeError('Your dataframe contains duplicate ID, please make sure ID column is unique for each row.')

        # 5. If treatment or x has no variation
        err_msg_variation = 'The feature column {} has only one value, please exclude this column from dataset.'
        err_msg_missing = 'Column {} has missing value, please fill these missing.'

        df_numeric = self.data[self.X].select_dtypes(include='number')
        df_discrete = self.data[self.X].select_dtypes(exclude='number')
        X_numeric = list(df_numeric.columns)
        X_discrete = list(df_discrete.columns)


        if X_numeric is not None:
            for i in X_numeric:
                if data[i].var() == 0:
                    raise TypeError(err_msg_variation.format(i))

        if X_discrete is not None:
            for i in X_discrete:
                if data[i].value_counts().shape[0] <= 1:
                    raise TypeError(err_msg_variation.format(i))

        # 5. If missing values exists
        for i in self.X:
            if data[i].isna().sum()>0:
                raise TypeError(err_msg_missing.format(i))

        self.df_numeric = df_numeric
        self.df_discrete = df_discrete
        self.X_numeric = X_numeric
        self.X_discrete = X_discrete

        return

    def preprocess_psm(self):

        if (self.caliper <= 0) or (self.caliper > 1):
            raise TypeError('Caliper should be a number between [0,1]. If you want to keep all observation, please set caliper to 1.')

        # for psm, need one-hot encoding
        if len(self.X_discrete) > 0:
            data_with_categ = pd.concat([
                self.df_numeric,  # dataset without the categorical features
                pd.get_dummies(self.df_discrete,
                               columns = self.X_discrete,
                               drop_first = True)  # categorical features converted to dummies
            ], axis=1)
            col_name_x_expand = data_with_categ.columns
        elif len(self.X_discrete) == 0:
            data_with_categ = self.df_numeric
            col_name_x_expand = data_with_categ.columns
        else:
            data_with_categ = None
            col_name_x_expand = []

        self.data_with_categ = data_with_categ
        self.col_name_x_expand = col_name_x_expand

        return

    def psm(self,
            n_neighbors: int = 1,
            model: LogisticRegression = LogisticRegression(random_state=0, C=1e6),
            caliper: float = 0.05,
            trim_percentage: float = 0.00,
            drop_duplicates: bool = False,
            model_list = None,
            test_size = 0) -> None:
        """
        Initialize matching object, include two methods: psm and cem.

        Parameters
        ----------
        :param n_neighbors: int, default is 1
            The number of neighbors you want to match. Current version only support
            NN method, so if you set n_neighbors=2. then 1 treatment obs match 2 control obs.
        :param model: sklearn model, default LogisticRegression(random_state=0, C=1e6).
            We support any sklearn classification model that can use "predict_proba"
            function to calculate p-score. You can also try following models:
                ps_model = LogisticRegression(C=1e6)
                ps_model = SVC(probability=True)
                ps_model = GaussianNB()
                ps_model = KNeighborsClassifier()
                ps_model = DecisionTreeClassifier()
                ps_model = RandomForestClassifier()
                ps_model = GradientBoostingClassifier()
                ps_model = LGBMClassifier()
                ps_model = XGBClassifier()
        :param caliper: float, default is 0.05
            If p-score diff between treat and control is greater than caliper,
            then trim this pair.
        :param trim_percentage: float, default is 0
            Must be smaller than 1. The percentage of obs to be trimmed. If equals 0.02,
            means trim p-score distribution's left 1% and right 1%'s observations.

        Returns
        -------
        df_out_final: pandas dataframe
             Matched Dataframe results.Include columns ['user_id_treat', 'treatment_treat', 'pscore_treat',
                     'user_id_control','treatment_control', 'pscore_control',
                     'pscore_diff']
        df_out_final_post_trim: pandas dataframe
             Matched Dataframe results after trimming. If caliper=0 and trim_percentage=0,
             df_out_final_post_trim is identical to df_out_final.
        """
        # Data preprocessing for psm only
        self.method = "psm"
        self.caliper = caliper
        self.drop_duplicates = drop_duplicates
        self.preprocess_psm()

        id = self.id
        data = self.data
        T = self.T

        # step 1: matching
        df_out_final, data_out, data_out_control, ps_model = psm(model, data, self.data_with_categ, self.col_name_x_expand, T, id, n_neighbors,model_list, test_size)

        # step 2: trim pairs with caliper
        df_out_final_trim_caliper, df_out_final_post_trim = self.psm_trim_caliper(df_out_final, caliper)

        # step 3: trim p-score percentile
        df_out_final_trim_percentage = self.psm_trim_percent(df_out_final_trim_caliper,trim_percentage)

        # step 4: trim
        if drop_duplicates is True:
            df_out_final_trim_percentage.drop_duplicates(subset=[self.id], inplace=True, ignore_index=True)

        df_out_final_post_trim = df_out_final_trim_percentage.copy()

        self.data_out_treat   = data_out
        self.data_out_control = data_out_control
        self.df_out_final     = df_out_final
        self.df_out_final_post_trim = df_out_final_post_trim
        self.model            = ps_model
        self.df_out_final_trim_caliper = df_out_final_trim_caliper
        self.df_out_final_trim_percentage = df_out_final_trim_percentage
        self.col_name_x_expand = self.col_name_x_expand

    def psm_trim_caliper(self,
                         df_pre,
                         caliper: float = 0.05):

        df_post = df_pre.copy()
        if caliper > 0:
            df_pre['pscore_diff'] = np.abs(df_pre['pscore_treat'] - df_pre['pscore_control'])
            valid_pair_indices = df_pre[df_pre['pscore_diff'] <= caliper].index
            df_post = df_pre.iloc[valid_pair_indices, :].copy()
            df_post.reset_index(inplace=True, drop=True)

        # stack up all observations
        df_post_treat = df_post[[self.id + "_treat", self.T + "_treat", 'pscore_treat']]
        df_post_control = df_post[[self.id + "_control", self.T + "_control", 'pscore_control']]

        df_post_treat.rename(columns={self.id + "_treat":self.id, self.T + "_treat":self.T,'pscore_treat':'pscore'},inplace=True)
        df_post_control.rename(columns={self.id + "_control": self.id, self.T + "_control": self.T, 'pscore_control': 'pscore'},inplace=True)

        df_full = pd.concat([df_post_treat, df_post_control], axis=0, ignore_index=True)
        df_full.drop_duplicates(subset=[self.id], inplace=True, ignore_index=True)

        return df_post, df_full

    def psm_trim_percent(self,
                         df_pre,
                         percentage: float = 0.00):
        df_post = df_pre.copy()

        # stack up all observations
        df_post_treat = df_post[[self.id + "_treat", self.T + "_treat", 'pscore_treat']]
        df_post_control = df_post[[self.id + "_control", self.T + "_control", 'pscore_control']]

        df_post_treat.rename(columns={self.id + "_treat":self.id, self.T + "_treat":self.T,'pscore_treat':'pscore'},inplace=True)
        df_post_control.rename(columns={self.id + "_control": self.id, self.T + "_control": self.T, 'pscore_control': 'pscore'},inplace=True)

        df_full = pd.concat([df_post_treat, df_post_control], axis=0, ignore_index=True)


        if (percentage > 0) and (percentage < 1):

            # df_full.drop_duplicates(subset=[self.id], inplace=True, ignore_index=True)

            p_score_ub = df_full['pscore'].quantile(q = 1-percentage/2)
            p_score_lb = df_full['pscore'].quantile(q = percentage/2)
            df_post = df_full[(df_full['pscore'] <= p_score_ub) & (df_full['pscore'] >= p_score_lb)]

        elif percentage == 0:
            df_post = df_full

        else:
            raise TypeError('Trim percentage should a value between 0 and 1.')

        df_post.reset_index(inplace=True, drop=True)
        return df_post

    def cem(self,
            n_bins: int = 5,
            break_points: Dict[str,List[float]] = None,
            cluster_criteria: Dict[str, List] = None,
            k2k = False):

        """
        Coarsened exact matching.

        Parameters
        ----------
        :param n_bins: int, default is 5.
            The number of bins you want to cut your continuous
            feature variables into.
        :param break_points: Dict, default is None.
            Dict type, key is the name of the column you'd like to break down,
            key's value is a list of float you'd like to break into. For example, if a feature is
            percentage type between 0 and 1,
            you can specify "break_points = {'c_1': [-1, 0.3, 0.6, 2]}"
            If break_points is not None, we'd first break values using breakpoints input,
            for the rest continuous columns we'd break them using n_bins.
        :param cluster_criteria: Dict, default is None.
            Dict type, key is the name of the column you'd like to break down, For example,
            if you have two object type columns "d_1" and "d_3", you can specify:
            cluster_criteria = {'d_1': [['apple','pear'],['cat','dog'],['bee']],
                                'd_3': [['0.0','1.0','2.0'],
                                        ['3.0','4.0','5.0'],
                                        ['6.0','7.0','8.0','9.0']]})
            key's value is a list of list string you'd like to cluster.
        :param k2k: Bool, default is False.
            If the number of observations are unequal between treatment group and control group,
            random sample from the group which has more observations with the obs num equals to the smaller one:
            for example, if treatment group has N1 observations, control group has N2, and N1>N2. K2K will sample
            N2 sample from treatment group.


        Returns
        -------
        df_out_final: Pandas dataframe.
            Including 'bin_id', 'bin_cnt_control','bin_cnt_treatment','weight_adjust'
            columns.
            'bin_id': the tag we put for bin.
            'bin_cnt_treatment': in this bin_id, how many treatment obs are included.
            'bin_cnt_control': in this bin_id, how many control obs are included.
            'weight_adjust': The weight used to calculate treatment effect if you use cem.

        Examples
        --------
            from matching import matching
            match_obj_1 = matching(data = df,
                                     T = 'treatment',
                                     X = ['c_1','c_2','d_1', 'gender', 'd_3'],
                                     id = 'user_id')

            match_obj_1.cem(n_bins=5,
                         break_points = {'c_1': [-1, 0.3, 0.6, 2]},
                         cluster_criteria = {'d_1': [['apple','pear'],['cat','dog'],['bee']],
                                            'd_3': [['0.0','1.0','2.0'],
                                                    ['3.0','4.0','5.0'],
                                                    ['6.0','7.0','8.0','9.0']]})
            match_obj_1.df_out_final.shape
        """

        self.method = "cem"


        self.n_bins = n_bins
        self.break_points = break_points
        self.cluster_criteria = cluster_criteria

        data = self.data
        T = self.T
        X = self.X

        df_x_numeric_cut, df_x_string_grouped = bin_cut(self, cluster_criteria, break_points)
        if (df_x_numeric_cut.shape[1]>0) and (df_x_string_grouped.shape[1]>0):
            df_coarsened_full = pd.DataFrame(data=np.concatenate((df_x_numeric_cut, df_x_string_grouped), axis=1),
                                             columns=list(df_x_numeric_cut.columns) + list(df_x_string_grouped.columns))
        elif (df_x_numeric_cut.shape[1]>0) and (df_x_string_grouped.shape[1]==0):
            df_coarsened_full = df_x_numeric_cut
        else:
            df_coarsened_full = df_x_string_grouped

        # generate bin_id
        x_cols_name_full_coarsened = list(df_coarsened_full.columns)
        df_coarsened_full['bin_id'] = ''

        for x in x_cols_name_full_coarsened:
            df_coarsened_full['bin_id'] = df_coarsened_full['bin_id'].astype(str) + df_coarsened_full[x].astype(str)
        df_coarsened_full['treatment'] = data[T]

        # calculate weight
        df_coarsened_full[self.id] = data[self.id]
        df_coarsened_full_out = calculate_weight(self, df_coarsened_full)

        # TODO: 用merge u_id/d_id. 把权重放回原始数据
        data['weight_adjust'] = df_coarsened_full_out['weight_adjust']
        data['weight'] = df_coarsened_full_out['weight']
        data['bin_id'] = df_coarsened_full_out['bin_id']
        data['bin_cnt_control'] = df_coarsened_full_out[0.0]
        data['bin_cnt_treatment'] = df_coarsened_full_out[1.0]

        df_matched = data[data['weight_adjust'] > 0].copy()
        df_matched.reset_index(inplace=True, drop=True)

        if k2k is True:
            df_matched = sample_k2k(self, df_matched)


        print('number of matched obs', df_matched.shape, 'number of total obs ', data.shape)

        self.df_x_numeric_cut = df_x_numeric_cut
        self.df_x_string_grouped = df_x_string_grouped
        self.df_out_final = df_matched
        self.data = data

    def ate(self):
        X_balance_check, df_post_validate = data_process_bc(self, True)
        df_post_validate_y = df_post_validate.merge(self.data[self.y + [self.id]], how='left', on = self.id)


        dict_ate = {"y": [], "ate": [], "p_val": []}

        for y_i in self.y:
            Y = df_post_validate_y[y_i]
            x = df_post_validate_y[self.T]

            x = sm.add_constant(x)

            model = sm.OLS(Y, x)
            results = model.fit()

            dict_ate['y'].append(y_i)
            dict_ate['ate'].append(results.params[self.T])
            dict_ate['p_val'].append(results.pvalues[self.T])

        return pd.DataFrame(dict_ate)



    def balance_check(self,
                      include_discrete = False,
                      threshold_smd = 0.1,
                      threshold_vr = 2):

        treat_var = self.T
        X_balance_check, df_post_validate = data_process_bc(self, include_discrete)

        smd_all = {"Covariates": [],
                   "Mean Treated": [],
                   "Mean Control": [],
                   "SMD": [],
                   "Var Ratio": [],
                   "ks-p_val": [],
                   "ttest-p_val": []}

        for col in X_balance_check:
            control_array = df_post_validate[df_post_validate[treat_var] == 0][col].values
            treatment_array = df_post_validate[df_post_validate[treat_var] == 1][col].values
            t_avg, c_avg, smd, pass_smd, vr, pass_vr = calculate_smd(control_array,
                                                                     treatment_array,
                                                                     treatment_array,
                                                                     threshold_smd,
                                                                     threshold_vr)

            # Performs the two-sample Kolmogorov-Smirnov test for goodness of fit.
            ks_stats, pvalue = stats.ks_2samp(control_array,
                                              treatment_array)

            # Perform Levene test for equal variances.
            _, levene_p = stats.levene(control_array, treatment_array)

            if levene_p > 0.05:
                t, p = stats.ttest_ind(control_array, treatment_array, equal_var=True)
            else:
                t, p = stats.ttest_ind(control_array, treatment_array, equal_var=False)

            smd_all["Covariates"].append(col)
            smd_all["Mean Treated"].append(t_avg)
            smd_all["Mean Control"].append(c_avg)
            smd_all["SMD"].append(smd)
            smd_all["Var Ratio"].append(vr)
            smd_all["ks-p_val"].append(np.round(pvalue, 3))
            smd_all["ttest-p_val"].append(np.round(p, 3))

        smd_match_df = pd.DataFrame(smd_all)
        return smd_match_df


if __name__ == "__main__":
    np.random.seed(123456)

    # generate sudo-data for matching
    ## 1. generate categorical data

    n_obs = 5000
    k_continuous = 3
    k_discrete = 3

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
    rand_treatment = np.random.choice(a=[0, 1], size=[n_obs, 1], p=[0.8, 0.2])
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
    df['user_id'] = df.index

    df['d_1'] = df['d_1'].astype('str')
    df['d_1'].replace(['0.0', '1.0', '2.0'], 'apple', inplace=True)
    df['d_1'].replace(['3.0', '4.0'], 'pear', inplace=True)
    df['d_1'].replace(['5.0', '6.0'], 'cat', inplace=True)
    df['d_1'].replace(['7.0', '8.0'], 'dog', inplace=True)
    df['d_1'].replace(['9.0'], 'bee', inplace=True)

    df['d_1'].value_counts()

    df['d_3'] = df['d_3'].astype('str')

    # --------------------------------------------- #
    match_obj = matching(data=df,
                         T='treatment',
                         X=['c_1', 'c_2', 'c_3', 'd_1', 'gender'],
                         id='user_id')

    match_obj.psm(n_neighbors = 2,
                  model = GradientBoostingClassifier(),
                  trim_percentage=0.1,
                  caliper = 0.000005)


    smd_match_df = match_obj_1.balance_check()

    print('smd_match_df')
