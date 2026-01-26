# %%
# import logging
# import sys
import os
from functools import partial
from pathlib import Path
from pprint import pp

import optuna
import pandas as pd
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler

from P07_af_all.T00_lib.classes_ml import DataHandler, MyUtil
from P07_af_all.T00_lib.utils import check_jupyter
from P07_af_all.T00_lib.optuna_ml import (
    OptunaUtil,
    optuna_objective_with_data_input,
)

# %% Initialize paths and settings
if check_jupyter():
    BASE_DIR = Path.cwd()  # Current directory of the running file
    DATA_DIR = BASE_DIR.parent / "T02_combine_features"
    CURRENT_DIR = BASE_DIR
else:
    BASE_DIR = Path.cwd()  # Base directory of the project
    DATA_DIR = BASE_DIR / "src/P07_af_all/T02_combine_features"
    CURRENT_DIR = Path(__file__).resolve().parent

dt = MyUtil.get_dt()
print(f"Current Directory: {CURRENT_DIR}")
print(f"Current Date and Time: {dt}")

# %% Load data
df = pd.read_excel(DATA_DIR / "S02_data_combined_loc.xlsx")
print(f"df.shape: {df.shape}")

# Extract features and targets
_X = df.iloc[:, :-3].values

# Extract target variables (last three columns)
_dfY = df.iloc[:, -3:]

# Need only 5052 prediction
_Y = _dfY.iloc[:, [0]].values

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
        "model": [
            "RandomForest",
            "KNR",
            "GradientBoosting",
            "SVR",
            "ElasticNet",
            "XGBR",
        ],
        "n_trials": [200],
    },
]
param_study_list = list(ParameterGrid(param_study_grid))
pp(param_study_list)


# %% Prepare optuna study
# Add stream handler of stdout to show the messages
# optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

# %% Run optimization
study_info_arr = []

for idx_study, param_study in enumerate(param_study_list[:]):
    # Extract parameters for the study
    model = param_study["model"]
    random_state = param_study["random_state"]
    test_size = param_study["test_size"]
    n_trials = param_study["n_trials"]

    # Create OptunaUtil instance
    optuna_util = OptunaUtil(
        model=model,
        random_state=random_state,
        test_size=test_size,
        current_dir=CURRENT_DIR,
    )

    # Print study information
    pp(
        f"Running study {idx_study}: model={model}, random_state={random_state}, test_size={test_size}, n_trials={n_trials}"
    )

    # Prepare data
    data_handler.split_and_scale(random_state=random_state, test_size=test_size)
    X_train, Y_train = data_handler.get_train()

    # Define objective function with fixed data
    optuna_objective_include_input = partial(
        optuna_objective_with_data_input,
        X_train=X_train,
        Y_train=Y_train,
        model=model,
        objective_score="r2",
    )

    # Load or create the sampler
    if not os.path.exists(optuna_util.sampler_filename):
        sampler = optuna.samplers.TPESampler(seed=42)
    else:
        sampler = optuna_util.load_sampler()

    # Load or create the study
    study = optuna.create_study(
        study_name=optuna_util.study_name,
        storage=optuna_util.storage_path,
        load_if_exists=True,
        sampler=sampler,
    )

    # Optimize
    study.optimize(optuna_objective_include_input, n_trials=n_trials)

    # Save sampler state for reproducibility
    optuna_util.save_sampler(sampler)

    # Collect study results
    best_trial = study.best_trial
    best_trial_scores = best_trial.user_attrs["scores"]
    best_trial_model_params = best_trial.user_attrs["model_params"]
    _study_info = dict(
        dt=dt,
        **param_study,
        study_name=optuna_util.study_name,
        best_param=study.best_params,
        best_value=study.best_value,
        total_trial=study.trials_dataframe().shape[0],
        **best_trial_scores,
        model_params=best_trial_model_params,
    )
    study_info_arr.append(_study_info)

study_info = pd.DataFrame.from_dict(study_info_arr)
study_info.to_excel(CURRENT_DIR / f"S01_search_{dt}.xlsx", index=False)
