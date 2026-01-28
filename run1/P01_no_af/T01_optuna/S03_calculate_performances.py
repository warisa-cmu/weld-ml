# %% Imports
from pathlib import Path
import ast
import re
import pandas as pd
from sklearn.preprocessing import StandardScaler

from run1.lib.classes_ml import (
    DataHandler,
    MyUtil,
    MyEval,
)
from run1.lib.utils import check_jupyter
from run1.lib.optuna_ml import (
    OptunaUtil,
)


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
study_info_filename = "S02_combine_study.xlsx"
study_info = pd.read_excel(CURRENT_DIR / study_info_filename)
study_info["model_params"] = study_info["model_params"].apply(ast.literal_eval)
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

# %% Load paramlist
df_arr = []
for idx, study in study_info.iterrows():
    random_state = study["random_state"]
    test_size = study["test_size"]
    model = study["model"]
    model_params = study["model_params"]

    print(
        f"Processing study {idx + 1}/{len(study_info)}: model={model}, random_state={random_state}, test_size={test_size}"
    )
    data_handler.split_and_scale(random_state=random_state, test_size=test_size)

    X_train, Y_train = data_handler.get_train()
    X_test, Y_test = data_handler.get_test()

    reg = OptunaUtil.get_model(model_name=model, **model_params)
    reg.fit(X_train, Y_train)
    Y_train_pred = reg.predict(X_train)
    Y_test_pred = reg.predict(X_test)

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
output_filename = "S03_calculate_performances.xlsx"
df_performances.to_excel(CURRENT_DIR / output_filename, index=False)
print(f"Saved performances to {output_filename}")
