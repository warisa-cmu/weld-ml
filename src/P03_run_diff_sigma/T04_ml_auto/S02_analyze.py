from pathlib import Path
from pprint import pp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from P03_run_diff_sigma.T00_lib.classes import MyEval, MyUtil, RegSwitcher
from P03_run_diff_sigma.T00_lib.utils import check_jupyter


if check_jupyter():
    BASE_DIR = Path.cwd()  # Current directory of the running file
    CURRENT_DIR = BASE_DIR
else:
    BASE_DIR = Path.cwd()  # Base directory of the project
    CURRENT_DIR = Path(__file__).resolve().parent

filenameInput = CURRENT_DIR / "S01_2026-01-03_10-42.pkl"

# Residual plots
IS_PLOT_RES = False
SAVE_PLOT_RES = False

SAVE_PLOT = True
SAVE_DATA = False

dt = MyUtil.get_dt()
data_load = MyUtil.load_data(filename=filenameInput)
data_handler = data_load["data_handler"]
df_cv = data_load["df_cv"]

# Sort the DataFrame by "rank_test_score"
df_cv = df_cv.sort_values(by="rank_test_score")
df_fit_select = df_cv.groupby(["id_split", "estimator"]).first().reset_index()

# Initialize blank model (optional)
reg = RegSwitcher(base=None)
df_arr = []
for idx, fit in df_fit_select.iterrows():
    param_split = fit["param_split"]
    data_handler.split_and_scale(**param_split)

    X_train, Y_train = data_handler.get_train()
    X_test, Y_test = data_handler.get_test()

    params = fit["params"]
    reg.set_params(**params)

    reg.fit(X_train, Y_train)

    Y_train_pred = reg.predict(X_train)
    Y_test_pred = reg.predict(X_test)

    _df = MyEval.eval(
        Y_train=Y_train,
        Y_train_pred=Y_train_pred,
        Y_test=Y_test,
        Y_test_pred=Y_test_pred,
        id_split=fit["id_split"],
        id_gs=fit["id_gs"],
        estimator=fit["estimator"],
    )
    df_arr.append(_df)

    if IS_PLOT_RES:
        id_split = fit["id_split"]
        estimator = fit["estimator"]
        MyEval.plot_res(
            Y_train=Y_train,
            Y_train_pred=Y_train_pred,
            Y_test=Y_test,
            Y_test_pred=Y_test_pred,
            current_dir=CURRENT_DIR,
            dt=dt,
            save=SAVE_PLOT_RES,
            file_prefix=f"S02-{estimator}-{id_split}",
        )

df_eval = pd.concat(df_arr).reset_index(drop=True)

df_cv["Y"] = "Y-All"
colsToMerge = ["id_split", "id_gs", "Y", "validation_scores"]
df_eval = df_eval.merge(df_cv[colsToMerge], on=["id_split", "id_gs", "Y"], how="left")


def expandCV(_df):
    val_scores = np.concatenate(_df["validation_scores"].values)
    return pd.DataFrame(data={"cv_results": val_scores})


filt = df_eval["Y"] == "Y-All"
cv_data = (
    df_eval[filt]
    .groupby(by=["estimator"])
    .apply(expandCV, include_groups=False)
    .reset_index(drop=False)
    .drop(columns=["level_1"])
)

fig, axes = plt.subplots(1, 3, figsize=(20, 4))

# Plot CV results
sns.boxplot(cv_data, x="estimator", y="cv_results", ax=axes[0])
axes[0].set_ylim([0, 1])
axes[0].set_title("CV Results")

# Plot test results
ax = sns.boxplot(data=df_eval, x="estimator", y="R2 Test", hue="Y", ax=axes[1])
axes[1].set_title("Test Result")

# Plot train (no cv) results
sns.boxplot(data=df_eval, x="estimator", y="R2 Train (No Val)", hue="Y", ax=axes[2])
axes[2].set_title("Train Result (No Validation)")

if SAVE_PLOT:
    filename = CURRENT_DIR / f"S02_{dt}.png"
    fig.savefig(filename, dpi=300, bbox_inches="tight")

if SAVE_DATA:
    filename = CURRENT_DIR / f"S02_{dt}.pkl"

    data_save = {
        "desc": "Results of ML model evaluation after selection",
        "filenameInput": filenameInput,
        "df_eval": df_eval,
        "cv_data": cv_data,
    }

    # Save the model
    MyUtil.save_data(filename=filename, data=data_save)
