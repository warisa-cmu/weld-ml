# %% Imports
import re
from pathlib import Path
from pprint import pp

import pandas as pd
from sklearn.model_selection import ParameterGrid
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from tabpfn import TabPFNRegressor

from run1.lib.classes_ml import DataHandler, MyEval
from run1.lib.utils import check_jupyter, MyUtil

# %% Initialize paths and settings
if check_jupyter():
    BASE_DIR = Path.cwd()  # Current directory of the running file
    ROOT_DIR = BASE_DIR.parent.parent.parent
    DATA_DIR = ROOT_DIR / "run1" / "data"
    CURRENT_DIR = BASE_DIR
else:
    BASE_DIR = Path.cwd()  # Base directory of the project
    ROOT_DIR = BASE_DIR
    DATA_DIR = ROOT_DIR / "run1" / "data"
    CURRENT_DIR = Path(__file__).resolve().parent

dt = MyUtil.get_dt()
print(f"Current Directory: {CURRENT_DIR}")
print(f"Data Directory: {DATA_DIR}")
print(f"Current Date and Time: {dt}")

# %% Load data
_df = pd.read_excel(DATA_DIR / "S02_data_exp.xlsx")
print(f"df.shape: {_df.shape}")

# Select columns for features and targets
colsY = [c for c in _df.columns if re.search(r"stress_value", c)]
colsX = [c for c in _df.columns if c in ["R", "W", "D", "position"]]
_dfY = _df[colsY]
_dfX = _df[colsX]
print("Selected feature columns:", colsX)
print("Selected target columns:", colsY)
print(f"dfX.shape: {_dfX.shape}")
print(f"dfY.shape: {_dfY.shape}")

# %% Extract features and targets
_X = _dfX.values
_Y = _dfY.values
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
    },
]
param_study_list = list(ParameterGrid(param_study_grid))
pp(param_study_list)

# %% Run studies

# Initialize the regressor
reg = MultiOutputRegressor(TabPFNRegressor())

df_arr = []
for idx_study, param_study in enumerate(param_study_list[:]):
    # Extract parameters for the study
    model = param_study["model"]
    random_state = param_study["random_state"]
    test_size = param_study["test_size"]
    print(
        f"Processing study {idx_study + 1}/{len(param_study_list)}: model={model}, random_state={random_state}, test_size={test_size}"
    )

    # Prepare data
    data_handler.split_and_scale(random_state=random_state, test_size=test_size)
    X_train, Y_train = data_handler.get_train()
    X_test, Y_test = data_handler.get_test()

    # Fit the model
    reg.fit(X_train, Y_train)

    # Make predictions
    # Training prediction is computationly expensive for TabPFN, so we skip it here
    Y_test_pred = reg.predict(X_test)

    _df = MyEval.eval_single(
        Y_true=Y_test,
        Y_pred=Y_test_pred,
        random_state=random_state,
        test_size=test_size,
        model=model,
    )
    df_arr.append(_df)

df_performances = pd.concat(df_arr, ignore_index=True)
output_filename = "S01_calculate_performance.xlsx"
df_performances.to_excel(CURRENT_DIR / output_filename, index=False)
print(f"Saved performances to {output_filename}")
