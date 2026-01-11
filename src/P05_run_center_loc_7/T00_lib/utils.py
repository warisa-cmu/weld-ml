import sys


def check_jupyter():
    isJupyter = "ipykernel" in sys.modules
    if isJupyter:
        print("Code is running in a Jupyter environment.")
    else:
        print("Code is running in a standard Python environment.")

    return isJupyter
