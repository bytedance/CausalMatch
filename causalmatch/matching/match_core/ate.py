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
import statsmodels.api as sm
from .utils import data_process_ate
warnings.simplefilter(action='ignore', category=FutureWarning)

def ate(match_obj, use_weight, df_post_validate):
    """

    @type match_obj: object
    """
    T = match_obj.T
    method = match_obj.method
    y = match_obj.y

    df_post_validate_y, weight = data_process_ate(match_obj, df_post_validate)
    dict_ate = {"y": [], "ate": [], "p_val": []}

    x = df_post_validate_y[T]
    x = sm.add_constant(x)

    # Loop all y variables to do OLS
    for y_i in y:
        Y = df_post_validate_y[y_i]

        if method == 'cem' and use_weight is True:
            model = sm.WLS(Y, x, weights=weight)
        else :
            model = sm.OLS(Y, x)

        results = model.fit()

        dict_ate['y'].append(y_i)
        dict_ate['ate'].append(results.params[T])
        dict_ate['p_val'].append(results.pvalues[T])
    df_param = pd.DataFrame(dict_ate)
    return df_param
