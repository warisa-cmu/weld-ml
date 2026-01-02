# %%
import os
import pickle
from datetime import datetime
from pathlib import Path
from pprint import pp

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split

from P04_ml.T00_lib.classes import BaseEstimator, DataHandler, MyUtil

is_jupyter = True

if is_jupyter:
    CURRENT_DIR = Path.cwd()
    PARENT_DIR = CURRENT_DIR.parent.parent
    DATA_DIR = PARENT_DIR / "P03_data_preprocess"
else:
    CURRENT_DIR = Path.cwd()
    DATA_DIR = CURRENT_DIR / "src/P03_data_preprocess"
print(DATA_DIR)
# df = pd.read_excel("data.xlsx", index_col="exp")
# print(df.head())
# %%

SAVE_DATA = True
