# cd .\src\P03_run_diff_sigma\T05_ml_auto_optuna\
# optuna-dashboard sqlite:///storage.db
# %%
import logging
import os
import sys
from functools import partial
from pathlib import Path
from pprint import pp

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import ParameterGrid, cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler

from P03_run_diff_sigma.T00_lib.classes import DataHandler, MyUtil, OptunaUtil, MyEval
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
param_study_grid = [
    {
        "random_state": [1],
        "test_size": [0.3],
        "model": ["RandomForest"],
        "n_trials": [50],
    },
]
param_study_list = list(ParameterGrid(param_study_grid))
pp(param_study_list)


# %% Define Optuna objective function
def _objective(
    trial: optuna.trial.Trial,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    model: str,
    cv=3,
    objective_score="mse_mean",
):
    if model == "RandomForest":
        n_estimators = trial.suggest_int("n_estimators", 50, 500, log=True)
        max_depth = trial.suggest_int("max_depth", 3, 128, log=True)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 50)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 50)
        max_features = trial.suggest_categorical(
            "max_features", ["sqrt", "log2", 0.2, 0.5, 0.8, 1.0]
        )
        bootstrap = trial.suggest_categorical("bootstrap", [True, False])
        criterion = trial.suggest_categorical(
            "criterion", ["squared_error", "absolute_error"]
        )
        reg = OptunaUtil.get_model(
            model_name=model,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap,
            criterion=criterion,
        )
    elif model == "SVR":
        kernel = trial.suggest_categorical(
            "kernel", ["rbf", "linear", "poly", "sigmoid"]
        )
        C = trial.suggest_float("C", 1e-6, 1e6, log=True)
        epsilon = trial.suggest_float("epsilon", 1e-6, 1.0, log=True)
        # gamma: allow 'scale'/'auto' or numeric
        gamma_choice = trial.suggest_categorical(
            "gamma_choice", ["scale", "auto", "float"]
        )
        if gamma_choice == "float":
            gamma = trial.suggest_float("gamma", 1e-6, 1e1, log=True)
        else:
            gamma = gamma_choice
        coef0 = trial.suggest_float("coef0", -1.0, 1.0)
        degree = trial.suggest_int("degree", 2, 5) if kernel == "poly" else 3
        shrinking = trial.suggest_categorical("shrinking", [True, False])
        reg = OptunaUtil.get_model(
            model_name=model,
            kernel=kernel,
            C=C,
            gamma=gamma,
            epsilon=epsilon,
            coef0=coef0,
            degree=degree,
            shrinking=shrinking,
        )
    else:
        raise ValueError(f"Model {model} not recognized in objective function")

    # Perform cross-validation

    scoring = ["neg_mean_squared_error", "neg_mean_absolute_percentage_error", "r2"]

    cv_results = cross_validate(
        reg,
        X_train,
        Y_train,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
    )

    scores = dict(
        mse_mean=-cv_results["test_neg_mean_squared_error"].mean(),
        mse_std=cv_results["test_neg_mean_squared_error"].std(),
        mape_mean=-cv_results["test_neg_mean_absolute_percentage_error"].mean(),
        mape_std=cv_results["test_neg_mean_absolute_percentage_error"].std(),
        r2_mean=cv_results["test_r2"].mean(),
        r2_std=cv_results["test_r2"].std(),
    )

    # Store all scores as user attributes
    trial.set_user_attr(key="scores", value=scores)

    # Return the objective score
    if objective_score == "mse_mean":
        return scores["mse_mean"]
    else:
        raise ValueError(f"Unsupported objective_score: {objective_score}")


# %% Prepare optuna study
# Add stream handler of stdout to show the messages
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

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
    objective = partial(_objective, X_train=X_train, Y_train=Y_train, model=model)

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
    study.optimize(objective, n_trials=n_trials)

    # Save sampler state for reproducibility
    optuna_util.save_sampler(sampler)

    # Collect study results
    best_trial = study.best_trial
    best_trial_scores = best_trial.user_attrs["scores"]
    _study_info = dict(
        **param_study,
        study_name=optuna_util.study_name,
        best_param=study.best_params,
        best_value=study.best_value,
        total_trial=study.trials_dataframe().shape[0],
        **best_trial_scores,
    )
    study_info_arr.append(_study_info)

study_info = pd.DataFrame.from_dict(study_info_arr)
study_info.to_excel(CURRENT_DIR / "S01_hyperparam_search.xlsx", index=False)
