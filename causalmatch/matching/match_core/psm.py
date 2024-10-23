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

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from math import ceil
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, roc_auc_score, f1_score
import warnings
from tqdm import tqdm

warnings.simplefilter(action='ignore', category=FutureWarning)

def psm(model,
        data,
        data_with_categ,
        col_name_x_expand,
        T, id,
        n_neighbors,
        model_list,
        test_size,
        verbose):
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
            y_pred = ps_model.predict(X_test)
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
    if verbose is not None:
        distances = []
        indices = []
        treated_pscore = np.reshape(p_score_val[treated_indices], (-1, 1))
        total_samples =  treated_pscore.shape[0]
        # Show progress
        with tqdm(total=total_samples, desc="Processed Samples", unit="sample") as pbar:
            step_length = max(1000, ceil(total_samples/20) )
            for i in range(0, total_samples, step_length):  # Assume we process 100 samples at a time
                end_idx = min(i + step_length, total_samples)  # Ensure the last batch does not exceed the total count
                dist, idx = nbrs.kneighbors(treated_pscore[i:end_idx])
                distances.append(dist)
                indices.append(idx)

                # Update the progress bar
                pbar.update(end_idx - i)

        # Combine results from all batches
        distances = np.vstack(distances)
        indices = np.vstack(indices)
        print('*'*30)
    else:
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

    return data_ps, df_out_final, data_out, data_out_control, ps_model


def main():
    # Generate synthetic data
    np.random.seed(42)
    num_samples = 50000
    data = pd.DataFrame({
        'id': np.arange(num_samples),  # Add an id column
        'feature1': np.random.rand(num_samples),
        'feature2': np.random.rand(num_samples),
        'T': np.random.choice([0, 1], size=num_samples)  # Treatment indicator
    })

    # Creating categorical variables
    data_with_categ = pd.get_dummies(data, columns=['feature1'], drop_first=True)

    col_name_x_expand = data_with_categ.columns.difference(['T', 'id'])
    T = 'T'
    id_col = 'id'
    n_neighbors = 1
    model_list = None  # Change to LogisticRegression
    test_size = 0
    verbose = True

    # Call the psm function
    data_ps, df_out_final, data_out, data_out_control, ps_model = psm(
        model = LogisticRegression(random_state=0, C=1e6),
        data = data,
        data_with_categ=data_with_categ,
        col_name_x_expand=col_name_x_expand,
        T=T,
        id=id_col,  # Pass the id column
        n_neighbors=n_neighbors,
        model_list=model_list,
        test_size=test_size,
        verbose=verbose
    )

    #match_obj.psm(n_neighbors=1, model=LogisticRegression(), trim_percentage=0.1, caliper=0.1)

    # Print the output dataframes
    print("Data with propensity scores:")
    print(data_ps.head())
    print("\nFinal output dataframe:")
    print(df_out_final.head())
    print("\nTreated data output:")
    print(data_out.head())
    print("\nMatched control data output:")
    print(data_out_control.head())

if __name__ == "__main__":
    main()
