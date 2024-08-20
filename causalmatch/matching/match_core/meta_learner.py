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
import warnings
from sklearn.linear_model import LinearRegression
from functools import reduce
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold, cross_val_score
warnings.simplefilter(action='ignore', category=FutureWarning)


def NMAPE(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true))

def SMAPE(y_true, y_pred) -> float :
    # Convert actual and predicted to numpy
    # array data type if not already
    if not all([isinstance(y_true, np.ndarray),isinstance(y_pred, np.ndarray)]) :
        y_true, y_pred = np.array(y_true), np.array(y_pred)
    return round(np.mean(np.abs(y_pred - y_true) /((np.abs(y_pred) + np.abs(y_true)) / 2)), 2)

def cross_product(*XS) :
    for X in XS :
        assert 2 >= X.ndim >= 1
    n = XS[0].shape[0]
    for X in XS :
        assert n == X.shape[0]

    def cross(XS) :
        k = len(XS)
        XS = [XS[i].reshape((n,) + (1,) * (k - i - 1) + (-1,) + (1,) * i) for i in range(k)]
        res = reduce(np.multiply, XS).reshape((n, -1))
        return res

    res = cross(XS)
    return res

def _combine(X, T, fitting=True) :
    if X is not None :
        F = X
    return cross_product(F, T)


def prep_x(T, X, if_intercept=1) :

    nobs = T[:, 0].shape[0]
    if if_intercept == 1:
        intercept = np.ones([nobs, 1])
        x_int = np.round(np.append(intercept, X, axis=1), 4)
    else :
        x_int = np.round(X, 4)
    fts = np.round(_combine(x_int, T), 4)

    x_full = np.concatenate([fts, x_int], axis=1)
    return x_full

def cross_validate(X, T, y, n_splits = 5) :
    nmape_scorer = make_scorer(SMAPE)

    # For linear model, interact X and T
    X_mat_full = prep_x(T, X, if_intercept=0)

    k_folds = KFold(n_splits = n_splits, shuffle=True, random_state=111)
    scores = cross_val_score(LinearRegression(), X_mat_full, y, scoring=nmape_scorer, cv=k_folds)

    print("Cross Validation Scores: ", scores)
    print("Average CV Score: ", scores.mean())
    print("Number of CV Scores used in Average: ", len(scores))


class s_learner_linear :
    def __init__(self, model_final = LinearRegression()):
        self.model_final = model_final

    def fit(self, y, T, X):
        model_final = self.model_final
        X = X * 1
        T = T.astype(float)
        X = X.astype(float)

        if model_final.__class__.__name__== 'LinearRegression':
            cross_validate(X, T, y, n_splits = 5)

        x_full = prep_x(T, X, if_intercept = 1)
        model_final = model_final.fit(x_full, y)
        # model_final = model_final.fit(X, y)

        self.model_final = model_final
        self.param = model_final.coef_
        self.x_full = x_full

    def predict(self, X) :
        X = X.astype(float)
        nobs = X.shape[0]
        intercept = np.ones([nobs, 1])


        x_int = np.round(np.append(intercept, X, axis=1), 4)
        fts_1 = np.round(_combine(x_int, np.ones([nobs, ])), 4)
        fts_0 = np.round(_combine(x_int, np.zeros([nobs, ])), 4)
        x_full_1 = np.concatenate([fts_1, x_int], axis=1)
        x_full_0 = np.concatenate([fts_0, x_int], axis=1)

        hte = self.model_final.predict(x_full_1) - self.model_final.predict(x_full_0)

        return hte


