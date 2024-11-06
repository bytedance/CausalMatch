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
from sklearn.linear_model import LogisticRegression
from typing import List, Dict
from causalmatch.matching.match_core.psm import psm
from causalmatch.matching.match_core.cem import calculate_weight, bin_cut, sample_k2k
from causalmatch.matching.match_core.preprocess import preprocess,preprocess_psm
from causalmatch.matching.match_core.utils import data_process_bc, data_process_ate, balance_check_x, gen_test_data, psm_trim_caliper, psm_trim_percent
from causalmatch.matching.match_core.ate import ate
from causalmatch.matching.match_core.meta_learner import s_learner_linear
from causalmatch.matching.match_core.robust_check import sensitivity_test, placebo_treatment_estimate
from causalmatch.matching.match_core.TableFormat import gen_fmt, fmt_2
from statsmodels.iolib.table import SimpleTable

import warnings


warnings.simplefilter(action='ignore', category=FutureWarning)
class matching :
    def __init__(self,
                 data: pd.DataFrame,
                 T: str,
                 X: List[str],
                 y: List[str] = [],
                 method: str = "psm",
                 id: str = None) :
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
        :param y: list[str], optional input
            List of outcome variables (dependent variables) names you want to calculate ATE with.
        :param method: str, default psm, optional input.
            Take two values only, either 'cem' or 'psm'.
        :param id: str, default none.
            ID column, can be user id or device id or any level id that you want to match.
            ID should be row-unique, i.e. one id only appear once in your dataframe input.
        """

        # reserve for input
        self.threshold_smd = None
        self.data = data
        self.T = T
        self.X = X
        self.y = y
        self.method = method
        self.id = id
        self.verbose = None
        self.n_neighbors = None
        self.trim_percentage = None
        self.drop_duplicates = None
        self.model_list = None
        self.test_size = None

        # reserve for output
        self.data_with_categ = None
        self.col_name_x_expand = None

        self.data_ps = None
        self.df_out_final = None
        self.data_out_treat = None
        self.data_out_control = None
        self.model = None
        self.df_out_final_trim_caliper = None
        self.df_out_final_trim_percentage = None
        self.df_out_final_post_trim = None
        self.n_bins = None
        self.break_points = None
        self.cluster_criteria = None
        self.replace = None

        self.df_pre_validate = None
        self.df_post_validate = None
        self.lb_index = None
        self.ub_index = None
        self.critical_value_lb = None
        self.critical_value_ub = None
        self.true_ate = None
        self.ps_model = None


        # data-preprocess
        preprocess(self)

    def psm(self,
            n_neighbors: int = 1,
            model: LogisticRegression = LogisticRegression(random_state=0, C=1e6),
            caliper: float = 0.05,
            trim_percentage: float = 0.00,
            drop_duplicates: bool = False,
            model_list = None,
            test_size = 0,
            verbose = None) -> None :
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
        self.n_neighbors = n_neighbors
        self.trim_percentage = trim_percentage
        self.model_list = model_list
        self.test_size = test_size

        preprocess_psm(self)

        id = self.id
        data = self.data
        T = self.T

        # Step 1: matching
        data_ps, df_out_final, data_out, data_out_control, ps_model = psm(model,
                                                                          data,
                                                                          self.data_with_categ,
                                                                          self.col_name_x_expand,
                                                                          T,
                                                                          id,
                                                                          n_neighbors,
                                                                          model_list,
                                                                          test_size,
                                                                          verbose=verbose)

        # Step 2: trim pairs with caliper
        df_out_final_trim_caliper, df_out_final_post_trim = psm_trim_caliper(self,df_out_final, caliper)

        # Step 3: trim p-score percentile
        df_out_final_trim_percentage = psm_trim_percent(self,df_out_final_trim_caliper, trim_percentage)

        # Step 4: trim
        if drop_duplicates is True :
            df_out_final_trim_percentage.drop_duplicates(subset=[self.id], inplace=True, ignore_index=True)

        df_out_final_post_trim = df_out_final_trim_percentage.copy()

        self.data_ps = data_ps
        self.data_out_treat = data_out
        self.data_out_control = data_out_control
        self.df_out_final = df_out_final
        self.df_out_final_post_trim = df_out_final_post_trim
        self.model = ps_model
        self.df_out_final_trim_caliper = df_out_final_trim_caliper
        self.df_out_final_trim_percentage = df_out_final_trim_percentage
        self.col_name_x_expand = self.col_name_x_expand



    def cem(self,
            n_bins: int = 5,
            break_points: Dict[str, List[float]] = None,
            cluster_criteria: Dict[str, List] = None,
            k2k = False) :

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
        if (df_x_numeric_cut.shape[1] > 0) and (df_x_string_grouped.shape[1] > 0) :
            df_coarsened_full = pd.DataFrame(data=np.concatenate((df_x_numeric_cut, df_x_string_grouped), axis=1),
                                             columns=list(df_x_numeric_cut.columns) + list(df_x_string_grouped.columns))
        elif (df_x_numeric_cut.shape[1] > 0) and (df_x_string_grouped.shape[1] == 0) :
            df_coarsened_full = df_x_numeric_cut
        else :
            df_coarsened_full = df_x_string_grouped

        # generate bin_id
        x_cols_name_full_coarsened = list(df_coarsened_full.columns)
        df_coarsened_full['bin_id'] = ''

        for x in x_cols_name_full_coarsened :
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

    def ate(self, use_weight = False) :
        """
        Average treatment effect calculation function.

        Parameters
        ----------
        :param use_weight: boolean, default is False.
            If use CEM, number of treatment obs can differ from the number of control.
            This option use weighted least square for CEM method to solve this issue.

        Returns
        -------
        df_param: Pandas dataframe.
            Including 'y', 'ate','p-value' columns.
            'y': name of the dependent variable.
            'ate': treatment effect coefficient.
            'p-value': treatment effect p-value.

        """
        if self.y is None or len(self.y)==0:
            raise TypeError('Please specify y variable as an input in the matching function.')

        # 1. post-process data
        X_balance_check, df_post_validate, df_pre_validate = data_process_bc(self, True)
        self.df_post_validate = df_post_validate
        self.df_pre_validate = df_pre_validate

        # 2. OLS and get treatment effect
        df_param = ate(self, use_weight, df_post_validate)
        return df_param

    def hte(self):
        """
        Average treatment effect calculation function.

        `match_obj.hte()` returns the treatment effect from a linear single learner model
        $y=f(X,T)+\varepsilon $, where $f(X,T)$ is a first order polynomial. For example, if $X=[X_1,X_2]$, $y=\alpha_0+\alpha_1 X_1 T+\alpha_2 X_2 T+\alpha_3 X_1 +\alpha_4 X_2+\alpha_5 T  +  \varepsilon$.
        We also provide MAPE score based on linear model.

        Returns
        -------
        hte_linear: 2D Numpy array with shape N*1.
        """

        T = self.T

        if self.y is None or len(self.y)==0:
            raise TypeError('Please specify y variable as an input in the matching function.')
        else:
            y = self.y[0]
            if len(self.y) > 1 :
                warnings.warn("We take the first y variable from your multiple y variables")

        # 1. post-process data
        X_balance_check, df_post_validate, df_pre_validate = data_process_bc(self, True)
        self.df_post_validate = df_post_validate
        self.df_pre_validate = df_pre_validate

        df_post_validate_y, weight = data_process_ate(self, df_post_validate)

        # 2. Single learner
        X_mat = df_post_validate[X_balance_check].values
        T_mat = df_post_validate[[T]].values
        y_mat = df_post_validate_y[[y]].values

        est_linear = s_learner_linear()
        est_linear.fit(y_mat, T_mat, X_mat)
        hte_linear = est_linear.predict(X_mat)

        return hte_linear

    def balance_check(self,
                      include_discrete = False,
                      threshold_smd = 0.1,
                      threshold_vr = 2):

        """
        Balance check function post matching.

        Parameters
        ----------
        :param include_discrete: boolean, default is False.
            Whether generate 2sample t-test for discrete X.
        :param threshold_smd: float, default is 0.1.
            When calculating SMD, threshold used to decide whether pass the test.
        :param threshold_vr: float, default is 2.
            When calculating SMD, threshold used to decide whether pass the test.


        Returns
        -------
        smd_match_df_post: Pandas dataframe.
            Post matching balance check.
            Including 'covariates', 'post treatment mean','post control mean', SMD, two-sample t-test
            result test columns.

        smd_match_df_pre: Pandas dataframe.
            Pre matching balance check.
            Including 'covariates', 'pre treatment mean','pre control mean', SMD, two-sample t-test
            result test columns.
        """
        self.threshold_smd = threshold_smd
        self.threshold_vr = threshold_vr

        X_balance_check, df_post_validate, df_pre_validate = data_process_bc(self, include_discrete)
        self.df_post_validate = df_post_validate
        self.df_pre_validate = df_pre_validate


        smd_match_df_post, smd_match_df_pre = balance_check_x(self, X_balance_check, df_post_validate, df_pre_validate)

        return smd_match_df_post, smd_match_df_pre

    def sensitivity_test(self,
                         gamma: List[int]):
        """
        Sensitivity test, following Rosenbaum (2005) "Sensitivity analysis in observational studies".
        You can start with fitting a list of integers starting from 1, such as [1,2,3,...] and
        see from which gamma, the "p-val upper bound" start to fall below 0.05

        Parameters
        ----------
        :param gamma: List[int], must input.
            Gamma measures the odds of treatment that is up to gamma times greater
            than the odds for another subject. When Gamma equals to 1, everyone has
            the same chance of receiving the treatment.


        Returns
        -------
        df: Pandas dataframe of test results with different gamma and y.
            The 'Wilcoxon-statistic' column is the one-sided Wilcoxon signed-rank test statistics
            computed using the matched pair's y values based on pair info from match_obj.df_out_final.
            The 'gamma' column is the gamma value from your input that is used to compute the upper
            bound and lower bound's test statistics.
            The 'stat upper bound' column is the upper bound value for the inferenced quantities,
            The 'lower upper bound' column is the lower bound value for the inferenced quantities,
            The 'z-score upper bound' column is the upper bound z-score for the inferenced quantities,
            The 'z-score lower bound' column is the lower bound z-score for the inferenced quantities,
            The 'y' column indicates which y is computed,

        """
        if self.method != 'psm':
            raise TypeError('This function only supports psm method.')

        if self.y is None or len(self.y)==0:
            raise TypeError('Please specify y variable in matching function.')

        df_res_full = pd.DataFrame()
        for y_i in self.y:
            df_res = sensitivity_test(self, gamma, y_i)
            df_res['y'] = y_i
            df_res_full = pd.concat([df_res_full, df_res], ignore_index=True)

        return df_res_full


    def placebo_treatment(self,
                          n: int = 1000,
                          b: int = 100):
        """
        Verifies the effect disappears when the Treatment is replaced with a placebo.

        Parameters
        ----------
        :param n: int, must input.
            Number of items from x axis to return, i.e how many rows are sample with replacement
            from original dataframe.
        :param b: int, must input.
            The number of bootstrap sample, default equals 100.


        Returns
        -------
        ate: Numpy array to store ATE estimate for each bootstrap sample with length b X 1.

        """
        if self.method != 'psm':
            raise TypeError('This function only supports psm method.')

        if self.model_list != None:
            raise TypeError('This function only supports model option input, does not support model_list option input.')

        if len(self.y) > 1 :
            raise TypeError(
                'Placebo test only support estimator for one y variable, '
                "please restrict your y input to a list with length of 1, for example, change 'y=[y0,y1]' to 'y=[y0]' .")

        if self.y is None:
            raise TypeError('Please input one y variable.')


        true_ate = self.ate().iloc[0]['ate']
        ate_array = placebo_treatment_estimate(self, n, b)

        # empirical confidence interval
        lb_index = np.round(b * 0.025)
        ub_index = np.round(b * (1 - 0.025))
        critical_value_lb = np.sort(np.array(ate_array))[int(lb_index)]
        critical_value_ub = np.sort(np.array(ate_array))[int(ub_index)]

        self.true_ate = true_ate
        self.lb_index = lb_index
        self.ub_index = ub_index
        self.critical_value_lb = critical_value_lb
        self.critical_value_ub = critical_value_ub

        return ate_array


    def robust_check(self,
                     gamma: List[int],
                     n: int = 1000,
                     b: int = 100):

        if len(self.y) > 1 :
            raise TypeError(
                'Placebo test only support estimator for one y variable, '
                "please restrict your y input to a list with length of 1, for example, change 'y=[y0,y1]' to 'y=[y0]' .")

        if (self.df_post_validate is None) or (self.df_pre_validate is None):
            X_balance_check, df_post_validate, df_pre_validate = data_process_bc(self, True)
        else:
            df_post_validate = self.df_post_validate
            df_pre_validate = self.df_pre_validate

        # 1. The remaining fraction of observations
        remain_sample_fraction = df_post_validate.shape[0] / df_pre_validate.shape[0]
        remain_sample_fraction_t = df_post_validate[df_post_validate[self.T] == 1].shape[0] / \
                                   df_pre_validate[df_pre_validate[self.T] == 1].shape[0]
        remain_sample_fraction_c = df_post_validate[df_post_validate[self.T] != 1].shape[0] / \
                                   df_pre_validate[df_pre_validate[self.T] != 1].shape[0]

        b_pass_fraction_test = 'Pass' if remain_sample_fraction >= 0.9 else 'Fail'
        b_pass_fraction_test_t = 'Pass' if remain_sample_fraction_t >= 0.9 else 'Fail'
        b_pass_fraction_test_c = 'Pass' if remain_sample_fraction_c >= 0.9 else 'Fail'


        # 2. Which observation from control group has been used multiple times
        df_repeat_ = pd.DataFrame(df_post_validate[df_post_validate[self.T]!=1][self.id].value_counts())
        df_repeat_.reset_index(inplace=True)
        max_repeat_time = df_repeat_[self.id].max()
        max_repeat_control_id_list = list(df_repeat_[df_repeat_[self.id] == max_repeat_time]['index'])
        max_repeat_control_id_list_str = ''.join(str(e) + ',' for e in max_repeat_control_id_list)


        # 3. Gamma sensitivity test, only works for PSM
        if self.method == 'psm':
            if gamma is None:
                gamma_ = [1,1.5,2,2.5,3]
            else:
                gamma_ = gamma

            df_gamma = self.sensitivity_test(gamma=gamma_)
            df_gamma_ = df_gamma[~df_gamma['z-score upper bound'].isna()].groupby(['y']).max()['gamma']
            gamma_test_stat = df_gamma_.values[0]
        else:
            gamma_test_stat = np.nan

        b_pass_gamma_test = 'Fail' if (gamma_test_stat <= 1 or gamma_test_stat==np.nan) else 'Pass'

        # TODO: 4. Refutation test, keep adding
        pseudo_ate_list = self.placebo_treatment(n=n,b=b)
        critical_val = ['{:.2f}'.format(self.critical_value_lb),'{:.2f}'.format(self.critical_value_ub)]
        critical_val_str = ''.join(str(e) + ',' for e in critical_val)
        b_pass_placebo_test = 'Fail' if (self.critical_value_lb <= self.ate().iloc[0]['ate'] and self.critical_value_ub >= self.ate().iloc[0]['ate']) else 'Pass'

        summary_stat = {"y" : self.y[0],
                        "ATE": '{:.2f}'.format(self.ate().iloc[0]['ate']),
                        "remain_sample_fraction": '{:.2f}%'.format(remain_sample_fraction*100),
                        "remain_sample_fraction_t": '{:.2f}%'.format(remain_sample_fraction_t*100),
                        "remain_sample_fraction_c": '{:.2f}%'.format(remain_sample_fraction_c*100),
                        "max_repeat_time": max_repeat_time,
                        "max_repeat_control_id_list_str": max_repeat_control_id_list_str,
                        "gamma_test_stat": gamma_test_stat,
                        "conf. interval": critical_val_str}


        # 5. Simpletable should be able to handle the formatting
        params_data_ = (['{:.2f}'.format(self.ate().iloc[0]['ate']), '{:.2f}%'.format(self.ate().p_val[0])],
                        [' ', ' '],
                        ['{:.2f}%'.format(remain_sample_fraction * 100), b_pass_fraction_test],
                        ['{:.2f}%'.format(remain_sample_fraction_t * 100), b_pass_fraction_test_t],
                        ['{:.2f}%'.format(remain_sample_fraction_c * 100), b_pass_fraction_test_c],
                        [max_repeat_time, '-'],
                        [gamma_test_stat, b_pass_gamma_test],
                        [critical_val_str, b_pass_placebo_test])

        params_stubs_ = ['Average Treatment Effect: '
            , ' '
            , '1. Total % of obs remained: '
            , ' -- Treated % obs remained: '
            , ' -- Control % obs remained: '
            , '    -- The most repeated times of a control obs:'
            , '2. Sensitivity test result, Gamma statistics'
            , '3. Placebo Test Result conf. interval']

        gen_title = 'Robustness Check Output Table for Dep. Variable {}'.format(self.y[0])
        gen_header = ['coef', 'P>|t| ']

        gen_table_parm = SimpleTable(params_data_,
                                     gen_header,
                                     params_stubs_,
                                     title=gen_title,
                                     txt_fmt=gen_fmt)

        print(gen_table_parm)

        return


if __name__ == "__main__" :
    df, rand_continuous, rand_true_param, param_te , rand_treatment, rand_error = gen_test_data(n = 10000, c_ratio=0.5)

    df.head()

    # X = ['c_1', 'c_2', 'c_3', 'd_1', 'gender']
    X = ['c_1', 'c_2', 'c_3']
    y = ['y', 'y2']
    id = 'user_id'
    T = 'treatment'

    # STEP 1: initialize matching object
    match_obj_cem = matching(data=df,
                             y=['y'],
                             T='treatment',
                             X=X,
                             id='user_id')

    # STEP 2: coarsened exact matching
    match_obj_cem.cem(n_bins=10, k2k=True)

    # STEP 3: balance check after propensity score matching
    print(match_obj_cem.balance_check(include_discrete=True))

    # STEP 4: obtain average partial effect
    print(match_obj_cem.ate())

    linear_hte = match_obj_cem.hte()

