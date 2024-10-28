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
from pandas.api.types import is_numeric_dtype, is_float_dtype
import warnings


warnings.simplefilter(action='ignore', category=FutureWarning)
def preprocess(match_obj) :
    # 1. whether T is numeric type
    if not is_numeric_dtype(match_obj.data[match_obj.T]) :
        raise TypeError('Treatment column in your dataframe is not numeric type, please transfer to numeric type.')

    # 2.x and t are contained in dataframe
    col_list = set(match_obj.data.columns)
    input_vars = [match_obj.T] + match_obj.X
    for i in input_vars :
        if i not in col_list :
            raise TypeError('Variable {} is not in the input dataframe.'.format(i))

    # 3. if ID column is generated
    if match_obj.id is None :
        id = 'id'
        match_obj.data[id] = match_obj.data.index
        match_obj.id = id
    else :
        id = match_obj.id
        if is_float_dtype(match_obj.data[id]) is True :
            warnings.warn("ID columns is float type, cannot support, convert to int type.")
            match_obj.data[id] = match_obj.data[id].astype(int)
        else :
            match_obj.data[id] = match_obj.data[id]

    data = match_obj.data
    # 4. If ID is unique, if is float
    id_check = data.groupby(id)[id].nunique() > 1
    if len(id_check[id_check == True]) > 0 :
        raise TypeError('Your dataframe contains duplicate ID, please make sure ID column is unique for each row.')

    # 5. If treatment or x has no variation
    err_msg_variation = 'The feature column {} has only one value, please exclude this column from dataset.'
    err_msg_missing = 'Column {} has missing value, please fill these missing.'

    df_numeric = match_obj.data[match_obj.X].select_dtypes(include='number')
    df_discrete = match_obj.data[match_obj.X].select_dtypes(exclude='number')
    X_numeric = list(df_numeric.columns)
    X_discrete = list(df_discrete.columns)

    if X_numeric is not None :
        for i in X_numeric :
            if data[i].var() == 0 :
                raise TypeError(err_msg_variation.format(i))

    if X_discrete is not None :
        for i in X_discrete :
            if data[i].value_counts().shape[0] <= 1 :
                raise TypeError(err_msg_variation.format(i))

    # 5. If missing values exists
    for i in match_obj.X :
        if data[i].isna().sum() > 0 :
            raise TypeError(err_msg_missing.format(i))

    match_obj.df_numeric = df_numeric
    match_obj.df_discrete = df_discrete
    match_obj.X_numeric = X_numeric
    match_obj.X_discrete = X_discrete

    return

def preprocess_psm(match_obj) :

    if (match_obj.caliper <= 0) or (match_obj.caliper > 1) :
        raise TypeError(
            'Caliper should be a number between [0,1]. If you want to keep all observation, please set caliper to 1.')

    # for psm, need one-hot encoding
    if len(match_obj.X_discrete) > 0 :
        data_with_categ = pd.concat([
            match_obj.df_numeric,  # dataset without the categorical features
            pd.get_dummies(match_obj.df_discrete,
                           columns=match_obj.X_discrete,
                           drop_first=True)  # categorical features converted to dummies
        ], axis=1)
        col_name_x_expand = data_with_categ.columns
    elif len(match_obj.X_discrete) == 0 :
        data_with_categ = match_obj.df_numeric
        col_name_x_expand = data_with_categ.columns
    else :
        data_with_categ = None
        col_name_x_expand = []

    match_obj.data_with_categ = data_with_categ
    match_obj.col_name_x_expand = col_name_x_expand

    return