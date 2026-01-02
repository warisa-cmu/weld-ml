import pickle
from datetime import datetime
from pprint import pp

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split

from P04_ml.T00_lib.classes import MyUtil, BaseEstimator, DataHandler


SAVE_DATA = True

df = pd.read_excel("data.xlsx", index_col="exp")
df.head()
