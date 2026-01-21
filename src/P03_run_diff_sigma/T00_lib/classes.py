import os
from pathlib import Path
import pickle
from dataclasses import dataclass
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    PredictionErrorDisplay,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR


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

    def __post_init__(self):
        names = self.get_names(
            model=self.model,
            random_state=self.random_state,
            test_size=self.test_size,
        )
        self.base_name = names["base_name"]
        self.sampler_name = names["sampler_name"]
        self.sampler_filename = f"{self.current_dir}/{self.sampler_name}.pickle"
        self.study_name = names["study_name"]
        self.storage_path = f"sqlite:///{self.current_dir}/{self.db_name}.db"

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
        elif model_name == "GradientBoosting":
            base_model = GradientBoostingRegressor(**params)
        elif model_name == "SVR":
            base_model = SVR()
        elif model_name == "LinearRegression":
            base_model = LinearRegression(**params)
        else:
            raise ValueError(f"Model {model_name} not recognized")
        reg = MultiOutputRegressor(base_model)
        return reg


class MyUtil:
    @staticmethod
    def save_data(filename, data):
        with open(filename, "wb") as file:
            pickle.dump(data, file)

    @staticmethod
    def load_data(filename):
        with open(filename, "rb") as file:
            data = pickle.load(file)
        return data

    @staticmethod
    def get_dt():
        return datetime.now().strftime("%Y-%m-%d_%H-%M")


class DataHandler:
    def __init__(self, _X, _Y, scalerX, scalerY):
        self._X = _X
        self._Y = _Y
        self.scalerX = scalerX
        self.scalerY = scalerY
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None

    def split_and_scale(self, test_size, random_state):
        _X_train, _X_test, _Y_train, _Y_test = train_test_split(
            self._X, self._Y, test_size=test_size, random_state=random_state
        )
        self.X_train = self.scalerX.fit_transform(_X_train)
        self.X_test = self.scalerX.transform(_X_test)

        self.Y_train = self.scalerY.fit_transform(_Y_train)
        self.Y_test = self.scalerY.transform(_Y_test)

    def get_train(self):
        return self.X_train, self.Y_train

    def get_test(self):
        return self.X_test, self.Y_test


class RegSwitcher(BaseEstimator):
    def __init__(self, base=None):
        self.base = base

    def fit(self, X, Y):
        self.base.fit(X, Y)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        return self.base.predict(X)


class MyEval:
    @staticmethod
    def eval_perf(y_true, y_pred):
        mse = mean_squared_error(y_true=y_true, y_pred=y_pred)
        mape = mean_absolute_percentage_error(y_true=y_true, y_pred=y_pred)
        r2 = r2_score(y_true=y_true, y_pred=y_pred)
        return mse, mape, r2

    @classmethod
    def eval(cls, Y_train, Y_test, Y_train_pred, Y_test_pred, **kwargs):
        data_arr = []
        for i in range(0, Y_train.shape[1]):
            mse_train, mape_train, r2_train = cls.eval_perf(
                y_true=Y_train[:, i], y_pred=Y_train_pred[:, i]
            )
            mse_test, mape_test, r2_test = cls.eval_perf(
                y_true=Y_test[:, i], y_pred=Y_test_pred[:, i]
            )

            data = {
                **kwargs,
                "Y": f"Y-{i + 1}",
                "MSE Train (No Val)": mse_train,
                "MSE Test": mse_test,
                "MAPE Train (No Val)": mape_train,
                "MAPE Test": mape_test,
                "R2 Train (No Val)": r2_train,
                "R2 Test": r2_test,
            }
            data_arr.append(data)

        mse_train, mape_train, r2_train = cls.eval_perf(
            y_true=Y_train, y_pred=Y_train_pred
        )
        mse_test, mape_test, r2_test = cls.eval_perf(y_true=Y_test, y_pred=Y_test_pred)
        data = {
            **kwargs,
            "Y": "Y-All",
            "MSE Train (No Val)": mse_train,
            "MSE Test": mse_test,
            "MAPE Train (No Val)": mape_train,
            "MAPE Test": mape_test,
            "R2 Train (No Val)": r2_train,
            "R2 Test": r2_test,
        }
        data_arr.append(data)
        df_eval = pd.DataFrame.from_dict(data_arr)
        return df_eval

    @classmethod
    def plot_res(
        cls,
        Y_train,
        Y_test,
        Y_train_pred,
        Y_test_pred,
        current_dir=".",
        dt="",
        save=False,
        show=False,
        file_prefix="",
    ):
        for i in range(0, Y_train.shape[1]):
            fig, axes = plt.subplots(
                nrows=1,
                ncols=2,
                figsize=(10, 5),
                constrained_layout=True,
                sharex=True,
                sharey=True,
            )

            display_train = PredictionErrorDisplay(
                y_true=Y_train[:, i], y_pred=Y_train_pred[:, i]
            )
            display_train.plot(ax=axes[0])
            axes[0].set_title("Train")

            display_train = PredictionErrorDisplay(
                y_true=Y_test[:, i], y_pred=Y_test_pred[:, i]
            )
            display_train.plot(ax=axes[1])
            axes[1].set_title("Test")

            if file_prefix != "":
                fig.suptitle(file_prefix)

            if save:
                if file_prefix == "":
                    raise Exception("Please specify file prefix")

                filename = f"{current_dir}/{file_prefix}_{dt}_{i}.png"
                fig.savefig(filename, dpi=300)

            if show:
                plt.show()
            else:
                plt.close(fig)
