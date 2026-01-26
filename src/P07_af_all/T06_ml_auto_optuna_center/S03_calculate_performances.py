# %% Imports
from pathlib import Path
import ast
import pandas as pd
from sklearn.preprocessing import StandardScaler

from P07_af_all.T00_lib.classes_ml import (
    DataHandler,
    MyUtil,
    MyEval,
)
from P07_af_all.T00_lib.utils import check_jupyter
from P07_af_all.T00_lib.optuna_ml import OptunaUtil

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
study_info_filename = "S02_combine_study.xlsx"
study_info = pd.read_excel(CURRENT_DIR / study_info_filename)
study_info["model_params"] = study_info["model_params"].apply(ast.literal_eval)
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
output_filename = "S03_calculate_performance.xlsx"
df_performances.to_excel(CURRENT_DIR / output_filename, index=False)
print(f"Saved performances to {output_filename}")
