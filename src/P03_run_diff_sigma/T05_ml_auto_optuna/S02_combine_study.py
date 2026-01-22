# %% Imports
import os
from pathlib import Path
import pandas as pd

from P03_run_diff_sigma.T00_lib.classes import MyUtil
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
os.chdir(CURRENT_DIR)
df_array = []
for file in os.listdir(CURRENT_DIR):
    if "S01" in file and file.endswith(".xlsx"):
        # Load each Excel file and append to list
        print(f"Loading file: {file}")
        df = pd.read_excel(file)
        df_array.append(df)

df = pd.concat(df_array, ignore_index=True)
# %% Process data


def choose_latest_study(df: pd.DataFrame) -> pd.Series:
    dft = df.copy()
    dft = dft.sort_values(by="dt", ascending=False)
    return dft.iloc[0]


dfG = df.groupby(by="study_name").apply(choose_latest_study, include_groups=False)
dfG = dfG.reset_index(drop=False)

dfG.to_excel(f"S02_comb_{dt}.xlsx", index=False)
