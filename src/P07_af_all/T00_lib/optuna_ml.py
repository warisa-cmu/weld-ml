import os
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import optuna
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet
from xgboost import XGBRegressor
from sklearn.model_selection import cross_validate
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR


def optuna_objective_with_data_input(
    trial: optuna.trial.Trial,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    model: str,
    cv=3,
    objective_score="mse_mean",
):
    if model == "RandomForest":
        # ! For faster tuning
        n_estimators = trial.suggest_int("n_estimators", 50, 100, log=True)
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

        model_params = dict(
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

        # Prepare model parameters
        model_params = dict(
            kernel=kernel,
            C=C,
            gamma=gamma,
            epsilon=epsilon,
            coef0=coef0,
            degree=degree,
            shrinking=shrinking,
            max_iter=int(1e6),
        )

    elif model == "GradientBoosting":
        n_estimators = trial.suggest_int("n_estimators", 50, 500, log=True)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1.0, log=True)
        max_depth = trial.suggest_int("max_depth", 1, 10)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 100)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 50)
        subsample = trial.suggest_float("subsample", 0.5, 1.0)
        max_features = trial.suggest_categorical(
            "max_features", ["sqrt", "log2", 0.2, 0.5, 0.8, 1.0]
        )
        loss = trial.suggest_categorical(
            "loss", ["squared_error", "absolute_error", "huber"]
        )

        model_params = dict(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            subsample=subsample,
            max_features=max_features,
            loss=loss,
        )

    elif model == "ElasticNet":
        # ElasticNet: tune regularization and mixing between L1/L2
        # alpha: overall regularization strength
        alpha = trial.suggest_float("alpha", 1e-8, 1e2, log=True)
        # l1_ratio: mix between L1 (1.0) and L2 (0.0)
        l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)
        fit_intercept = trial.suggest_categorical("fit_intercept", [True, False])
        positive = trial.suggest_categorical("positive", [True, False])
        # max iterations and tolerance
        max_iter = trial.suggest_int("max_iter", 100, 10000, log=True)
        tol = trial.suggest_float("tol", 1e-6, 1e-1, log=True)
        # coordinate descent selection strategy
        selection = trial.suggest_categorical("selection", ["cyclic", "random"])
        warm_start = trial.suggest_categorical("warm_start", [True, False])

        model_params = dict(
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=fit_intercept,
            positive=positive,
            max_iter=max_iter,
            tol=tol,
            selection=selection,
            warm_start=warm_start,
        )

    elif model == "KNR":
        # KNeighborsRegressor hyperparameters
        n_neighbors = trial.suggest_int("n_neighbors", 1, 50)
        weights = trial.suggest_categorical("weights", ["uniform", "distance"])
        algorithm = trial.suggest_categorical(
            "algorithm", ["auto", "ball_tree", "kd_tree", "brute"]
        )
        leaf_size = trial.suggest_int("leaf_size", 10, 60)
        p = trial.suggest_int("p", 1, 2)
        # metric: allow common metrics; if non-minkowski, `p` is ignored by sklearn
        metric = trial.suggest_categorical(
            "metric", ["minkowski", "euclidean", "manhattan", "chebyshev"]
        )

        model_params = dict(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p,
            metric=metric,
        )

    elif model == "XGBR":
        # XGBRegressor
        n_estimators = trial.suggest_int("n_estimators", 50, 500, log=True)
        max_depth = trial.suggest_int("max_depth", 1, 12)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1.0, log=True)
        subsample = trial.suggest_float("subsample", 0.3, 1.0)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.3, 1.0)
        colsample_bylevel = trial.suggest_float("colsample_bylevel", 0.3, 1.0)
        gamma = trial.suggest_float("gamma", 0.0, 10.0)
        min_child_weight = trial.suggest_float("min_child_weight", 1e-3, 10.0, log=True)
        reg_alpha = trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True)
        reg_lambda = trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True)
        booster = trial.suggest_categorical("booster", ["gbtree", "dart"])
        tree_method = trial.suggest_categorical(
            "tree_method", ["auto", "hist", "approx"]
        )

        model_params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            colsample_bylevel=colsample_bylevel,
            gamma=gamma,
            min_child_weight=min_child_weight,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            booster=booster,
            tree_method=tree_method,
            random_state=42,
            verbosity=0,
        )

    else:
        raise ValueError(f"Model {model} not recognized in objective function")

    reg = OptunaUtil.get_model(model_name=model, **model_params)
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

    # Store scores and model parameters in trial user attributes
    trial.set_user_attr(key="scores", value=scores)
    trial.set_user_attr(key="model_params", value=model_params)

    # Return the objective score
    if objective_score == "mse_mean":
        return scores["mse_mean"]
    elif objective_score == "center":
        if len(cv_results["test_r2"]) != 3:
            raise Exception("This option needs 3 Ys")
        return -cv_results["test_r2"][2]
    else:
        raise ValueError(f"Unsupported objective_score: {objective_score}")


@dataclass
class OptunaUtil:
    model: str
    random_state: int
    test_size: float
    current_dir: Path
    base_name = ""
    study_name = ""
    sampler_name = ""
    sampler_filename = ""
    storage_path = ""
    db_name = "storage"
    storage_dirname = "S01"
    storage_dirpath = ""

    def __post_init__(self):
        names = self.get_names(
            model=self.model,
            random_state=self.random_state,
            test_size=self.test_size,
        )
        self.base_name = names["base_name"]
        self.sampler_name = names["sampler_name"]
        self.study_name = names["study_name"]
        self.storage_dirpath = f"{self.current_dir}/{self.storage_dirname}"

        if not os.path.exists(self.storage_dirpath):
            os.makedirs(self.storage_dirpath)

        self.storage_path = (
            f"sqlite:///{self.current_dir}/{self.storage_dirname}/{self.db_name}.db"
        )
        self.sampler_filename = (
            f"{self.current_dir}/{self.storage_dirname}/{self.sampler_name}.pickle"
        )

        print(f"Database path: {self.storage_path}")

    @staticmethod
    def get_names(model: str, random_state: int, test_size: float) -> str:
        base_name = f"{model}_RS-{random_state}_TS-{test_size}".replace(".", "_")
        study_name = f"study_{base_name}"
        sampler_name = f"sampler_{base_name}"
        return dict(
            base_name=base_name, sampler_name=sampler_name, study_name=study_name
        )

    def load_sampler(self):
        if not os.path.exists(self.sampler_filename):
            raise FileNotFoundError(f"Sampler file {self.sampler_filename} not found")
        with open(self.sampler_filename, "rb") as file:
            sampler = pickle.load(file)
        return sampler

    def save_sampler(self, sampler):
        with open(self.sampler_filename, "wb") as file:
            pickle.dump(sampler, file)

    @staticmethod
    def get_model(model_name: str, **params) -> MultiOutputRegressor:
        if model_name == "RandomForest":
            base_model = RandomForestRegressor(**params)
        elif model_name == "KNR":
            base_model = KNeighborsRegressor(**params)
        elif model_name == "GradientBoosting":
            base_model = GradientBoostingRegressor(**params)
        elif model_name == "SVR":
            base_model = SVR(**params)
        elif model_name == "ElasticNet":
            base_model = ElasticNet(**params)
        elif model_name == "XGBR":
            base_model = XGBRegressor(**params)
        else:
            raise ValueError(f"Model {model_name} not recognized")
        reg = MultiOutputRegressor(base_model)
        return reg
