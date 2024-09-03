<h1>
<a href="">
<img src="doc/logo_nobc.png" width="80px" align="left" style="margin-right: 10px;", alt="causamatch-logo"> 
</a> CausalMatch: A Python Package for Propensity Score Matching and Coarsened Exact Matching 
</h1>

[![PyPI version](https://badge.fury.io/py/causalmatch.svg)](https://badge.fury.io/py/causalmatch)
[![Downloads](https://static.pepy.tech/badge/causalmatch)](https://pepy.tech/project/causalmatch)
[![Downloads](https://static.pepy.tech/badge/causalmatch/month)](https://pepy.tech/project/causalmatch)
[![Downloads](https://static.pepy.tech/badge/causalmatch/week)](https://pepy.tech/project/causalmatch)

**CausalMatch** is a Python package that implements two classic matching methods, propensity score matching (PSM) and coarsened exact matching (CEM), to estimate average treatment effects from observational data. 
This package was designed and built as part of the ByteDance data science research program with the goal of combining state-of-the-art machine learning techniques with econometrics to bring automation to complex causal inference problems.
Our toolkit possess the following features:
* Implement classic matching techniques in the literature at the intersection of econometrics and machine learning
* Maintain flexibility in modeling the propensity score model (via various machine learning classification models), while preserving the causal interpretation of the learned model and often offering valid confidence intervals
* Use a unified API
* Build on standard Python packages for Machine Learning and Data Analysis

[//]: # (For information on use cases and background material on causal inference and heterogeneous treatment effects see our webpage at [webpage here])

<details>
<summary><strong><em>Table of Contents</em></strong></summary>

- [News](#news)
- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Usage Examples](#usage-examples)
    - [Estimation Methods](#estimation-methods)
- [References](#references)
</details>

# News

If you'd like to contribute to this project, please contact xiaoyuzhou@bytedance.com. 
If you have any questions, feel free to raise them in the issues section.

**August 20, 2024:** Release v0.0.2, see release notes [here](https://github.com/bytedance/CausalMatch/releases/tag/v0.0.2)

<details><summary>Previous releases</summary>

**August 2, 2024:** Release 0.0.1.

</details>

</details>

# Getting Started

## Installation

Install the latest release from [PyPI]:
```
pip install causalmatch==0.0.2
```


## Usage Examples
### Estimation Methods

<details>
  <summary>Propensity Score Matching (aka PSM) (click to expand)</summary>

  * Simple PSM

  ```Python
from causalmatch import matching,gen_test_data
from sklearn.ensemble import GradientBoostingClassifier
import statsmodels.api as sm

df = gen_test_data(n = 10000, c_ratio=0.5)
df.head()


X = ['c_1', 'c_2', 'c_3', 'd_1', 'gender']
y = ['y', 'y2']
id = 'user_id'
T = 'treatment' # treatment variable must be binary with 0/1 values

# STEP 1: initialize object
match_obj = matching(data = df,     
                     T = T,
                     X = X,
                     id = id)

# STEP 2: propensity score matching

match_obj.psm(n_neighbors = 1,                      # number of neighbors
                model = GradientBoostingClassifier(), # p-score model
                trim_percentage = 0.1,                # trim x percent of data based on propensity score
                caliper = 0.1)                        # caliper for p-score diff

# STEP 3: balance check after propensity score matching
match_obj.balance_check(include_discrete = True)


# STEP 4: obtain output dataframe, and merge X and y
df_out = match_obj.df_out_final_post_trim.merge(df[y + X + [id]], how='left', on = id)

# STEP 5: calculate average treatment effect on treated

X_mat = df_out[T]
y_mat = df_out[y]

X_mat = sm.add_constant(X_mat)
model = sm.OLS(y_mat,X_mat)
results = model.fit()
print(results.params)
  ```

  * PSM with multiple p-score models and select the best one based on f1 score 

  ```Python
# STEP 0: define all classification model you need
from causalmatch import matching
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

ps_model1 = LogisticRegression(C=1e6)
ps_model2 = SVC(probability=True)
ps_model3 = GaussianNB()
ps_model4 = KNeighborsClassifier()
ps_model5 = DecisionTreeClassifier()
ps_model6 = RandomForestClassifier()
ps_model7 = GradientBoostingClassifier()
ps_model8 = LGBMClassifier()
ps_model9 = XGBClassifier()

model_list = [ps_model1, ps_model2, ps_model3,  ps_model4, ps_model5, ps_model6,  ps_model7, ps_model8, ps_model9]
match_obj = matching(data = df, T = T, X = X, id = id)
match_obj.psm(n_neighbors = 1,
              model_list = model_list, # input list of models you want to try
              trim_percentage = 0,
              caliper = 1,              
              test_size = 0.2) # train-test split, what portion does test sample takes
print(match_obj.balance_check(include_discrete = True))
df_out = match_obj.df_out_final_post_trim.merge(df[y + X + [id]], how='left', on = id)

  ```

</details>



<details>
  <summary>Coarsened Exact Matching (click to expand)</summary>

  * Simple CEM


  ```Python

match_obj_cem = matching(data = df,  y = ['y'], T = 'treatment',  X = ['c_1','d_1','d_3'], id = 'user_id')
# coarsened exact matching
match_obj_cem.cem(n_bins = 10, # number of bins for continuous x variables, cut by percentile
                  k2k = True)  # k2k: trim exp/base to have same observation numbers
print(match_obj_cem.balance_check(include_discrete=True))
print(match_obj_cem.ate())
  ```

  * CEM with customized bin cut

  ```Python

match_obj_cem = matching(data = df,  y = ['y'], T = 'treatment',  X = ['c_1','d_1','d_3'], id = 'user_id')
match_obj_cem.cem(n_bins = 10,                                     
                  break_points = {'c_1': [-1, 0.3, 0.6, 2]},  # cut point for continuous variable
                  cluster_criteria = {'d_1': [['apple','pear'],['cat','dog'],['bee']],
                                      'd_3': [['0.0','1.0','2.0'], ['3.0','4.0','5.0'], ['6.0','7.0','8.0','9.0']]}, # group values for discrete variables
                  k2k = True) 
  ```
</details>



See the <a href="#references">References</a> section for more details.

# References

S. Athey, J. Tibshirani, S. Wager.
**Generalized random forests.**
[*Annals of Statistics, 47, no. 2, 1148--1178*](https://projecteuclid.org/euclid.aos/1547197251), 2019.

V. Chernozhukov, D. Nekipelov, V. Semenova, V. Syrgkanis.
**Plug-in Regularized Estimation of High-Dimensional Parameters in Nonlinear Semiparametric Models.**
[*Arxiv preprint arxiv:1806.04823*](https://arxiv.org/abs/1806.04823), 2018.

S. Wager, S. Athey.
**Estimation and Inference of Heterogeneous Treatment Effects using Random Forests.**
[*Journal of the American Statistical Association, 113:523, 1228-1242*](https://www.tandfonline.com/doi/citedby/10.1080/01621459.2017.1319839), 2018.

V. Chernozhukov, D. Chetverikov, M. Demirer, E. Duflo, C. Hansen, and a. W. Newey. **Double Machine Learning for Treatment and Causal Parameters.** [*ArXiv preprint arXiv:1608.00060*](https://arxiv.org/abs/1608.00060), 2016.
