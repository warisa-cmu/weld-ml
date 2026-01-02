# %%
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path
from pprint import pp

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split

from P04_ml.T00_lib.classes import BaseEstimator, DataHandler, MyUtil


def is_running_in_jupyter_sys():
    """Checks for the presence of the ipykernel module in sys.modules."""
    return "ipykernel" in sys.modules


if is_running_in_jupyter_sys():
    print("Code is running in a Jupyter environment.")
else:
    print("Code is running in a standard Python environment.")

is_jupyter = is_running_in_jupyter_sys()

if is_jupyter:
    CURRENT_DIR = Path.cwd()
    PARENT_DIR = CURRENT_DIR.parent.parent
    DATA_DIR = PARENT_DIR / "P03_data_preprocess"
else:
    CURRENT_DIR = Path.cwd()
    DATA_DIR = CURRENT_DIR / "src/P03_data_preprocess"
print(DATA_DIR)
df = pd.read_excel(DATA_DIR / "S07_data_combined_loc.xlsx")
print(df.head())

SAVE_DATA = True
