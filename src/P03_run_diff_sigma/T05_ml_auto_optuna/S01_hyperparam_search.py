# cd .\src\P03_run_diff_sigma\T05_ml_auto_optuna\
# optuna-dashboard sqlite:///storage.db
# %%
import logging
import os
import pickle
import sys
from functools import partial
from pathlib import Path
from pprint import pp

import optuna
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, ParameterGrid, cross_val_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from P03_run_diff_sigma.T00_lib.classes import DataHandler, MyUtil
from P03_run_diff_sigma.T00_lib.utils import check_jupyter

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
param_grid_split = [{"random_state": [1, 2, 3, 4, 5], "test_size": [0.3]}]
param_list_split = list(ParameterGrid(param_grid_split))
pp(param_list_split)


# %% Define Optuna objective function
def _objective(trial, X_train, Y_train, model="RandomForest"):
    if model == "RandomForest":
        n_estimators = trial.suggest_int("n_estimators", 1, 1000, log=True)
        max_depth = trial.suggest_int("max_depth", 1, 32, log=True)
        reg = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42
        )
    scores = cross_val_score(
        reg, X_train, Y_train, cv=3, scoring="neg_mean_squared_error"
    )
    mse = -scores.mean()  # We want to minimize mean of MSE
    return mse


# %% Prepare optuna study
# Add stream handler of stdout to show the messages
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

# %% Run optimization
model = "RandomForest"
n_trials = 1
study_info_arr = []

for idx_split, param_split in enumerate(param_list_split[:]):
    random_state = param_split["random_state"]
    test_size = param_split["test_size"]
    print(
        f"Running split {idx_split} with random_state={random_state}, test_size={test_size}"
    )

    data_handler.split_and_scale(**param_split)
    X_train, Y_train = data_handler.get_train()

    objective = partial(_objective, X_train=X_train, Y_train=Y_train, model=model)

    base_name = f"{model}_RS-{random_state}_TS-{test_size}".replace(".", "_")
    study_name = f"study_{base_name}"
    sampler_name = f"sampler_{base_name}"

    # Load or create the sampler
    if not os.path.exists(f"{CURRENT_DIR}/{sampler_name}.pickle"):
        sampler = optuna.samplers.CmaEsSampler(seed=42)
    else:
        with open(f"{CURRENT_DIR}/{sampler_name}.pickle", "rb") as fin:
            sampler = pickle.load(fin)

    # Load or create the study
    storage_name = f"sqlite:///{CURRENT_DIR}/storage.db"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        sampler=sampler,
    )

    # Optimize
    study.optimize(objective, n_trials=n_trials)

    # Save sampler state for reproducibility
    with open(f"{CURRENT_DIR}/{sampler_name}.pickle", "wb") as fout:
        pickle.dump(sampler, fout)

    _study_info = dict(
        model=model,
        **param_split,
        study_name=study_name,
        best_param=study.best_params,
        best_value=study.best_value,
    )
    study_info_arr.append(_study_info)

study_info = pd.DataFrame.from_dict(study_info_arr)
study_info.to_excel(CURRENT_DIR / "S01_hyperparam_search.xlsx", index=False)
