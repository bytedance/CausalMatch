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
import warnings
from causalmatch.matching.match_core.preprocess import preprocess_mrd
from causalmatch.matching.match_core.utils import gen_test_data_mrd
from causalmatch.matching.match_core.utils_mrd import mrd_estimation, mrd_estimation2
warnings.simplefilter(action='ignore', category=FutureWarning)
class mrd:
    def __init__(self,
                 data: pd.DataFrame,
                 idb: str,
                 ids: str,
                 tb: str,
                 ts: str,
                 y: str):
        """
        Obtain estimators from :
        Bajari, Patrick, et al. "Multiple randomization designs." arXiv preprint arXiv:2112.13495 (2021).

        Parameters
        ----------
        :param data: Pandas dataframe, should include feature columns and treatment column.
        :param idb: str, default none.
            Buyer id column name, here refers to row-id in equation (3.1).
        :param ids: str, default none.
            Seller id column name, here refers to column-id in equation (3.1).
            idb*ids pair should be pairwise unique.
        :param tb: str, must input.
            Treatment status for idb. Only support dummy treatment, i.e 0-1 binary treatment.
        :param ts: str, must input.
            Treatment status for ids. Only support dummy treatment, i.e 0-1 binary treatment.
        :param y: str, must input
            Dependent variable of paired result. For example, if id1 is user id, id2 is shop id, each row of y can be the
            payment amount that one user pay one shop.

        Returns
        -------
        None
            An initialized object.

        Examples
        --------
        >>> df_raw = gen_test_data_mrd(n_shops = 5
        ...                           , n_users = 10
        ...                           , ate = 1.5
        ...                           , uflow = 0.2
        ...                           , sflow = 0.3)
        >>> df_raw
              shop_id	  user_id	treatment	treatment_u	treatment_s	      error	      status	y_clean	    y_overflow
        0	     1	        1	        0	        0	        1	        -0.156717	    is	   -0.156717	 0.143283
        1	     2	        1	        0	        0	        0	         0.056958	    c	    0.056958	 0.056958
        2	     3	        1	        0	        0	        1	        -0.202036	    is	   -0.202036	 0.097964
        3	     4	        1	        0	        0	        0	        -0.144839	    c	   -0.144839	-0.144839
        4	     5	        1	        0	        0	        0	        -0.019864	    c	   -0.019864	-0.019864
        In the test dataset, shop_id mimics seller id in Bajari, Patrick, et al. (2021),
        user_id mimics buyer id, treatment_u is the treatment status of the user_id,
        treatment_s is the treatment status of the shop_id, y_overflow is the dependent variable
        target to be estimated. Then we initialize a matching object:
        >>> mrd_obj = mrd(data = df_raw,
        ...               idb = 'user_id',
        ...               ids = 'shop_id',
        ...               tb  = 'treatment_u',
        ...               ts  = 'treatment_s',
        ...               y   = 'y_overflow')
        """

        # reserve for input
        self.data = data
        self.idb = idb
        self.ids = ids
        self.tb = tb
        self.ts = ts
        self.y = y
        self.user_exp_list = None
        self.shop_exp_list = None
        self.n_users = None
        self.n_shops = None

        # data-preprocess
        preprocess_mrd(self)

    def ate(self):
        """
        Calculate average treatment effect based on Bajari, Patrick, et al. (2021) formulas.

        Parameters
        ----------
        None

        Returns
        -------
        df_res: pandas dataframe
             The resulting dataframe include four columns: parameters, mean, variance, t_stat, p_values.
             "parameters" has four types: tau(pb,ps), tau direct, tau b spillover, tau s spillover.
             "mean" is the mean estimates of the four types of parameter following equations from p17
             "variance" is the variance estimates of the four types of parameter following Lemma A.4 and Lemma A.5.
             "t_stat" is the t-statistic
             "p_values" is the p-value.

        Examples
        --------
        >>> mrd_obj.ate()
                        parameters	    mean	 variance	 t_stat	    p_values
            0	     tau	          2.061481	 0.125891	41.083462	1.25E-39
            1	     tau_tdirect	  1.523736	 0.074238	39.544176	7.68E-39
            2	     tau_b_spillover  0.234169	 0.002311	34.441344	5.31E-36
            3	     tau_s_spillover  0.303575	 0.00317	38.12586	4.35E-38
        """
        df_res = mrd_estimation(self)
        return df_res

    def ate2(self):
        df_res = mrd_estimation2(self)
        return df_res



if __name__ == "__main__" :
    df_raw = gen_test_data_mrd(n_shops=10
                               , n_users=100
                               , ate=1
                               , uflow=0.05
                               , sflow=0.05)
    mrd_obj = mrd(data=df_raw,
                  idb='user_id',
                  ids='shop_id',
                  tb='treatment_u',
                  ts='treatment_s',
                  y='y_overflow')

    mrd_obj.ate()


