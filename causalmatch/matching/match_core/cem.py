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
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None


def sample_k2k(match_obj, df_matched) :
    T = match_obj.T

    df_t = df_matched[df_matched[T] == 1].copy()
    df_c = df_matched[df_matched[T] == 0].copy()

    nt = df_t.shape[0]
    nc = df_c.shape[0]

    if nt < nc :
        df_c_ = df_c.sample(n=nt, replace=False, random_state=1).copy()
        df_matched_k2k = pd.concat([df_c_, df_t], axis=0, ignore_index=True)
    elif nt > nc :
        df_t_ = df_t.sample(n=nc, replace=False, random_state=1).copy()
        df_matched_k2k = pd.concat([df_c, df_t_], axis=0, ignore_index=True)
    else :
        df_matched_k2k = df_matched

    return df_matched_k2k


def bin_cut(match_obj,
            cluster_criteria,
            break_points) :
    X = match_obj.X
    df_x = match_obj.data[X].copy()
    df_x_string = match_obj.df_discrete
    df_x_string_grouped = pd.DataFrame()
    df_x_numeric_cut = pd.DataFrame()

    # bin-cut discrete variables
    for x in df_x_string.columns :
        df_x_string_grouped[x] = df_x[x]
        if cluster_criteria is not None :
            if x in list(cluster_criteria.keys()) :
                for x_i in cluster_criteria[x] :
                    group_name_i = str(x_i)
                    mapping = {}
                    for i in x_i :
                        mapping[i] = group_name_i
                    df_x_string_grouped[x] = df_x_string_grouped.apply(
                        lambda row : mapping[row[x]] if row[x] in x_i else row[x], axis=1)

    # bin-cut continuous variables
    for x in match_obj.X_numeric :
        if break_points is not None :
            if x in list(break_points.keys()) :
                x_cut = pd.cut(df_x[x], bins=break_points[x])
                df_x_numeric_cut[x] = x_cut.astype(str)
            else :
                x_cut = pd.cut(df_x[x], bins=match_obj.n_bins)
                df_x_numeric_cut[x] = x_cut.astype(str)
        else :
            x_cut = pd.qcut(df_x[x], q=match_obj.n_bins, duplicates='drop')
            df_x_numeric_cut[x] = x_cut.astype(str)

    return df_x_numeric_cut, df_x_string_grouped


def calculate_weight(match_obj,
                     df_coarsened_full):
    """
    For CEM, if do not use is_k2k option, should user weighted least square when
    calculating average treatment effect. This function generate each obs weight based
    on "[PUT REFERENCED EQUATION WEBPAGE LINK HERE]". Formula follows:

            Weight = (Treatnment_N / Control_N) / ( Total_Control_N / Total_Treatment_N)
    where Treatnment_N is the number of treament obs in the bin, Control_N is the number of control obs
    in the bin, Total_Control_N  is the total number of control obs, Total_Treatment_N is the total number
    of obs in the treatment group.

    Parameters
    ----------
    :param match_obj: object, must input.
    :param df_coarsened_full: Pandas dataframe, must input.
        Treatment variable name included in dataframe. This version only support
        dummy treatment, i.e 0-1 binary treatment.
    """


    id = match_obj.id
    df_coarsened_full_groupby_treatment = df_coarsened_full[['bin_id', 'treatment', id]].groupby(['bin_id', 'treatment']).count()
    df_coarsened_full_groupby_treatment.reset_index(inplace=True)

    df_coarsened_full_groupby_treatment_pivot = df_coarsened_full_groupby_treatment.pivot(index='bin_id',
                                                                                          columns='treatment',
                                                                                          values=match_obj.id)
    df_coarsened_full_groupby_treatment_pivot.reset_index(inplace=True)
    df_coarsened_full_groupby_treatment_pivot.fillna(0, inplace=True)

    denominator = df_coarsened_full_groupby_treatment_pivot[0.0].sum() / df_coarsened_full_groupby_treatment_pivot[
        1.0].sum()
    df_coarsened_full_groupby_treatment_pivot['weight'] = (df_coarsened_full_groupby_treatment_pivot[1.0] /
                                                           df_coarsened_full_groupby_treatment_pivot[0.0]
                                                           ) / denominator
    df_coarsened_full_groupby_treatment_pivot['weight'].replace([np.inf, -np.inf], 0, inplace=True)

    # merge the weight of every bin back to original dataframe
    df_coarsened_full_out = df_coarsened_full.merge(df_coarsened_full_groupby_treatment_pivot,
                                                    how='left',
                                                    on=['bin_id'])

    # recalculate weight, initialize weight to a 99999 number so later can drop those empty
    df_coarsened_full_out['weight_adjust'] = 999999.0

    # 1. treatment obs with no matched control, set weight to 0
    df_coarsened_full_out['weight_adjust'].loc[
        (df_coarsened_full_out['treatment'] == 1.0) & (df_coarsened_full_out['weight'] == 0.0)] = 0.0

    # 2. treatment with control, set weight to 1
    df_coarsened_full_out['weight_adjust'].loc[
        (df_coarsened_full_out['treatment'] == 1.0) & (df_coarsened_full_out['weight'] > 0)] = 1.0

    # 3. control obs with no treatment, set weight to 0
    df_coarsened_full_out['weight_adjust'].loc[
        (df_coarsened_full_out['treatment'] == 0.0) & (df_coarsened_full_out['weight'] == 0.0)] = 0.0

    # 4. control obs with matched treatment, set weight to calculated values
    df_coarsened_full_out['weight_adjust'].loc[
        (df_coarsened_full_out['treatment'] == 0.0) & (df_coarsened_full_out['weight'] > 0.0)] = \
        df_coarsened_full_out['weight'].loc[
            (df_coarsened_full_out['treatment'] == 0.0) & (df_coarsened_full_out['weight'] > 0.0)].values

    return df_coarsened_full_out
