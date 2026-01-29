import sys
import pickle
from datetime import datetime


def check_jupyter():
    isJupyter = "ipykernel" in sys.modules
    if isJupyter:
        print("Code is running in a Jupyter environment.")
    else:
        print("Code is running in a standard Python environment.")

    return isJupyter


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
