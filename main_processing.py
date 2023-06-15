"""
AlloEye data processing master script
-pools features and metrics from individual participant data processing
"""

# 1. lib imports and housekeeping
# 2. collate individual participant data processing into master tables
# 3. save/export master csv(s)
# 4. reshape master data for different analyses e.g. by condition, by obj shift distance etc.

# 1. lib imports and housekeeping
# 1a. lib imports

import pandas as pd
from pathlib import Path
from os import path
import numpy as np
import logging
# 1b. logging
logging.basicConfig(level=logging.INFO)

# 1c. directories and filenames

from_home = True    # False if at work

work_data_dir = "C:\\Users\\Luke Emrich-Mills\\Documents\\AlloEye\\MainDataOutput\\"    # work pc
home_data_dir = "C:\\Users\\Luke\\Documents\\AlloEye\\data\\" # home laptop
data_dir = home_data_dir if from_home else work_data_dir

onedrive_folder = "~\\OneDrive\\Documents\\PhD\\AlloEye\\data"
save_dir = path.expanduser(onedrive_folder)

# 2. collate individual ppt data
# 2a. define ppts by code
# full ppt IDs
pIDs_young = [
    "5", "6", "10", "12", "13", "14", "15", "16", "17", "18",    # 10
    "21", "22", "23", "24", "25", "26", "27", "28", "29", "30",    # 20
    "31", "32", "33", "34", "35", "36", "41"]
pIDs_old = [
    "1", "2", "3", "4", "7", "8", "11", "19", "20", "37",   # 10
    "38", "39", "40"],
# pIDs_patients = ["50", "51"]    # placeholder
# pIDs = pIDs_old + pIDs_young + pIDs_patients    # eventually use this
pIDs = pIDs_old + pIDs_young

# test pIDs - overwrite if not commented out
pIDs_old = ["2", "4"]
pIDs_young = ["24", "26"]
pIDs_patients = ["50", "51"]
pIDs = pIDs_old + pIDs_young
