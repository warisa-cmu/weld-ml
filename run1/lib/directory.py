# %%
from pathlib import Path
from run1.lib.utils import check_jupyter
import re


def get_directory(current_dir, verbose=False):
    if check_jupyter():
        # Use regex to find the weld-ml directory in the current path
        match = re.search(r"(weld-ml)", str(current_dir))
        # If found, set BASE_DIR and ROOT_DIR accordingly
        if match:
            BASE_DIR = Path(str(current_dir)[: match.end()])
            ROOT_DIR = BASE_DIR
        else:
            raise ValueError("Cannot determine BASE_DIR in Jupyter environment.")
    else:
        BASE_DIR = Path.cwd()  # Base directory of the project
        ROOT_DIR = BASE_DIR

    if "P02_MF_1" in str(current_dir):
        DATA_DIR = ROOT_DIR / "run1" / "P02_MF_1" / "T01_af_features"
        DATA_PATH = DATA_DIR / "S01_combined_data.xlsx"
        STUDY_ML_DIR = ROOT_DIR / "run1" / "P02_MF_1" / "T02_optuna"
        STUDY_ML_PATH = STUDY_ML_DIR / "S02_combine_study.xlsx"
        STUDY_TABPFN_DIR = ROOT_DIR / "run1" / "P02_MF_1" / "T11_tabPFN"
        STUDY_TABPFN_PATH = STUDY_TABPFN_DIR / "S01_calculate_performance.xlsx"

        directory = dict(
            ROOT_DIR=ROOT_DIR,
            DATA_DIR=DATA_DIR,
            DATA_PATH=DATA_PATH,
            STUDY_ML_DIR=STUDY_ML_DIR,
            STUDY_ML_PATH=STUDY_ML_PATH,
            STUDY_TABPFN_DIR=STUDY_TABPFN_DIR,
            STUDY_TABPFN_PATH=STUDY_TABPFN_PATH,
        )
    else:
        raise ValueError(f"Cannot determine data directory for {current_dir}")

    if verbose:
        for key, value in directory.items():
            print(f"{key}: {value}")

    return directory
