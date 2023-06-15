import platform

import pandas as pd

from src.d00_utils.Conditions import Conditions
from src.d01_data.database.Errors import UnknownComputer
from src.d03_processing.feature_extract.trial_to_ppt import trial_to_ppt
# from plotnine import *

# for saving files
icn_pc = "DESKTOP-IBAG161"
personal_laptop = "LAPTOP-OOBPQ1A8"
if platform.uname().node == icn_pc:
    save_dir = "C:\\Users\\Luke Emrich-Mills\\Documents\\AlloEye\\MainDataOutput\\feature_saves\\"
elif platform.uname().node == personal_laptop:
    save_dir = "C:\\Users\\Luke\\Documents\\AlloEye\\data\\feature_saves\\"
else:
    raise UnknownComputer

ppts = "all"
# ppts = ['50', '44', '34']
# ppts = '44'
# ppts = ""
# headers = TimepointCsvToSQL.timepoint_headers()
# timepoints = pd.read_csv(f"{save_dir}all_timepoints.csv")
timepoints = None
ppts = ['52']
all_df, cond_df = trial_to_ppt(ppts, timepoints=timepoints)
all_df.to_csv(f"{save_dir}all_df.csv", index=False)
cond_df.to_csv(f"{save_dir}cond_df.csv", index=False)
# # plot with plotnine
# (ggplot(cond_df, aes('condition', 'Hn_enc', fill='group'))
#  + geom_boxplot())
Conditions.list

print("end")
