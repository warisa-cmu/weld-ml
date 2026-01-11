# %%
import re
from pathlib import Path
from pprint import pp

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from P03_run_diff_sigma.T00_lib.classes import DataHandler, MyUtil, RegSwitcher
from P03_run_diff_sigma.T00_lib.utils import check_jupyter

if check_jupyter():
    BASE_DIR = Path.cwd()  # Current directory of the running file
    DATA_DIR = BASE_DIR.parent / "T02_combine_features"
    CURRENT_DIR = BASE_DIR
else:
    BASE_DIR = Path.cwd()  # Base directory of the project
    DATA_DIR = BASE_DIR / "src/P3_run_diff_sigma/T02_combine_features"
    CURRENT_DIR = Path(__file__).resolve().parent

dt = MyUtil.get_dt()
print(f"Current Directory: {CURRENT_DIR}")
print(f"Current Date and Time: {dt}")

SAVE_DATA = True

# Load data
df = pd.read_excel(DATA_DIR / "S02_data_combined_loc.xlsx")
print(df.shape)

# %%
# Extract features and targets
_X = df.iloc[:, :-3].values
_Y = df.iloc[:, -3:].values
print(_X.shape)
print(_Y.shape)

# Create DataHandler instance
data_handler = DataHandler(
    _X=_X, _Y=_Y, scalerX=StandardScaler(), scalerY=StandardScaler()
)

# Create parameter grid
param_grid_split = [{"random_state": [1, 2, 3, 4, 5], "test_size": [0.3]}]
param_list_split = list(ParameterGrid(param_grid_split))
pp(param_list_split)
base_lr = MultiOutputRegressor(estimator=LinearRegression())
base_svr = MultiOutputRegressor(estimator=SVR())
base_rf = MultiOutputRegressor(estimator=RandomForestRegressor())
base_gbr = MultiOutputRegressor(estimator=GradientBoostingRegressor())

param_grid_hyper = [
    {"base": [base_lr]},
    {"base": [base_svr], "base__estimator__C": [1, 100]},
    {"base": [base_rf], "base__estimator__n_estimators": [100]},
    {"base": [base_gbr], "base__estimator__n_estimators": [100]},
]

# Initialize blank model (optional)
reg = RegSwitcher(base=None)

df_arr = []
for idx_split, param_split in enumerate(param_list_split):
    data_handler.split_and_scale(**param_split)
    X_train, Y_train = data_handler.get_train()

    gs = GridSearchCV(
        estimator=reg,
        param_grid=param_grid_hyper,
        cv=3,
        # scoring="neg_mean_squared_error",
        scoring="r2",
        n_jobs=-1,
    )
    gs.fit(X_train, Y_train)
    _df = pd.DataFrame(gs.cv_results_)
    _df["id_split"] = idx_split
    _df["param_split"] = [param_split for _ in range(_df.shape[0])]
    df_arr.append(_df)

df_cv = pd.concat(df_arr)
df_cv = df_cv.reset_index().rename(columns={"index": "id_gs"})

# Process results
df_cv["estimator"] = df_cv["param_base"].apply(lambda x: x.estimator.__class__.__name__)
pattern = r"split\d+_test_score"
colsSplitTestScore = [col for col in df_cv.columns if re.fullmatch(pattern, col)]
df_cv["validation_scores"] = df_cv[colsSplitTestScore].apply(
    lambda row: row.values, axis=1
)
cols = [
    "id_split",
    "param_split",
    "id_gs",
    "params",
    "estimator",
    "mean_test_score",
    "std_test_score",
    "rank_test_score",
    "validation_scores",
]
df_cv = df_cv[cols]
df_cv

if SAVE_DATA:
    filename = CURRENT_DIR / f"S01_{dt}.pkl"

    data_save = {
        "desc": "Results of ML model selection using GridSearchCV",
        "data_handler": data_handler,
        "param_split": param_split,
        "param_grid_hyper": param_grid_hyper,
        "df_cv": df_cv,
    }

    # Save the model
    MyUtil.save_data(filename=filename, data=data_save)
