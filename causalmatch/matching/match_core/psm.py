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

from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, roc_auc_score, f1_score
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

def psm(model,
        data,
        data_with_categ,
        col_name_x_expand,
        T, id,
        n_neighbors,
        model_list,
        test_size):
    # initialize the list to store f1 score
    score_list = []

    if test_size == 0:
        X_train, y_train = data_with_categ[col_name_x_expand], data[T]
        ps_model = model.fit(X_train, y_train)

    elif 0 < test_size < 1:
        X_train, X_test, y_train, y_test = train_test_split(data_with_categ[col_name_x_expand],
                                                            data[T],
                                                            test_size=test_size,
                                                            random_state=42)
        if model_list is not None:

            for model in model_list:
                ps_model = model.fit(X_train, y_train)
                y_pred = ps_model.predict(X_test)
                f1score = f1_score(y_test.values, y_pred)
                score_list.append(f1score)

            # choose the best model to review
            best_model_index = np.argmax(score_list, axis=0)
            ps_model = model_list[best_model_index].fit(data_with_categ[col_name_x_expand], data[T])
            print('The f1 score for all models you specify is:', score_list)
            print('The best model is the {} model'.format(best_model_index))

        else:
            ps_model = model.fit(X_train, y_train)
            y_pred = ps_model.predict(X_test.values)
            f1score = f1_score(y_test.values, y_pred)
            score_list.append(f1score)

    else:
        raise TypeError('Test size should be a value between 0 and 1.')

    p_score_val = ps_model.predict_proba(data_with_categ[col_name_x_expand])[:, 1]
    data_ps = data.assign(pscore=p_score_val)
    treated_indices = data_ps[data_ps[T] == 1].index
    control_indices = data_ps[data_ps[T] == 0].index

    # The NearestNeighbors model with n_neighbors=1 and algorithm='ball_tree' will return
    # the indices of the nearest points in the population matrix, rather than the actual values of those points.
    nbrs = NearestNeighbors(n_neighbors=n_neighbors,
                            algorithm='ball_tree').fit(np.reshape(p_score_val[control_indices], (-1, 1)))

    # K-neighbors
    distances, indices = nbrs.kneighbors(np.reshape(p_score_val[treated_indices], (-1, 1)))
    matched_control_indices = control_indices[indices.flatten()]

    data_out = data_ps[[id, T, 'pscore']].iloc[treated_indices, :].copy()
    data_out.reset_index(inplace=True, drop=True)
    data_out_ = data_out.loc[data_out.index.repeat(n_neighbors)]
    data_out_.reset_index(inplace=True, drop=True)
    data_out_.rename(columns={id: id + "_treat",
                              T: T + "_treat",
                              "pscore": "pscore" + "_treat"}, inplace=True)
    data_out_control = data_ps[[id, T, 'pscore']].iloc[matched_control_indices, :].copy()
    data_out_control.reset_index(inplace=True, drop=True)

    frames = [data_out_, data_out_control]
    df_out_final = pd.concat(frames, ignore_index=True, axis=1)
    df_out_final.columns = list(data_out_.columns) + list(data_out_control.columns)

    df_out_final.rename(columns={id: id + "_control",
                                 T: T + "_control",
                                 "pscore": "pscore" + "_control"}, inplace=True)

    return df_out_final, data_out, data_out_control, ps_model
