# %% Imports
import re
from pathlib import Path
from pprint import pp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    PredictionErrorDisplay,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion

from P03_run_diff_sigma.T00_lib.classes_ml import DataHandler, MyUtil
from P03_run_diff_sigma.T00_lib.optuna_ml import (
    OptunaUtil,
    optuna_objective_with_data_input,
)
from P03_run_diff_sigma.T00_lib.utils import check_jupyter
from P05_run_center_loc_7.T00_lib.classes import MyEval

# %% Initialize paths and settings
if check_jupyter():
    BASE_DIR = Path.cwd()  # Current directory of the running file
    DATA_DIR = BASE_DIR.parent / "T02_combine_features"
    CURRENT_DIR = BASE_DIR
else:
    BASE_DIR = Path.cwd()  # Base directory of the project
    DATA_DIR = BASE_DIR / "src/P03_run_diff_sigma/T02_combine_features"
    CURRENT_DIR = Path(__file__).resolve().parent

dt = MyUtil.get_dt()
print(f"Current Directory: {CURRENT_DIR}")
print(f"Current Date and Time: {dt}")

# %% Load data
df = pd.read_excel(DATA_DIR / "S02_data_combined_loc.xlsx")
print(f"df.shape: {df.shape}")

# Extract features and targets
_X = df.iloc[:, :-3].values
_Y = df.iloc[:, -3:].values
print(f"_X.shape: {_X.shape}")
print(f"_Y.shape: {_Y.shape}")

# Create DataHandler instance
data_handler = DataHandler(
    _X=_X, _Y=_Y, scalerX=StandardScaler(), scalerY=StandardScaler()
)

# Create parameter grid
param_study_grid = [
    {
        "random_state": [1, 2, 3, 4, 5],
        "test_size": [0.3],
        "model": ["TabPFN"],
        "n_trials": [30],
    },
]
param_study_list = list(ParameterGrid(param_study_grid))
pp(param_study_list)


# Initialize the regressor
reg = MultiOutputRegressor(TabPFNRegressor())


df_arr = []
for idx_study, param_study in enumerate(param_study_list[:]):
    model = param_study["model"]
    random_state = param_study["random_state"]
    test_size = param_study["test_size"]
    n_trials = param_study["n_trials"]

    print(
        f"Processing study {idx_study + 1}/{len(param_study_list)}: model={model}, random_state={random_state}, test_size={test_size}, n_trials={n_trials}"
    )

    # Extract parameters for the study
    model = param_study["model"]
    random_state = param_study["random_state"]
    test_size = param_study["test_size"]
    n_trials = param_study["n_trials"]

    # Prepare data
    data_handler.split_and_scale(random_state=random_state, test_size=test_size)
    X_train, Y_train = data_handler.get_train()
    X_test, Y_test = data_handler.get_test()

    reg.fit(X_train, Y_train)

    Y_test_pred = reg.predict(X_test)
    # TabPFN does not support train prediction separately, just use random values
    Y_train_pred = np.random.random(Y_train.shape)

    _df = MyEval.eval(
        Y_train=Y_train,
        Y_train_pred=Y_train_pred,
        Y_test=Y_test,
        Y_test_pred=Y_test_pred,
        random_state=random_state,
        test_size=test_size,
        model=model,
    )
    df_arr.append(_df)

df_performances = pd.concat(df_arr, ignore_index=True)
output_filename = "S03_calculate_performance.xlsx"
df_performances.to_excel(CURRENT_DIR / output_filename, index=False)
print(f"Saved performances to {output_filename}")
