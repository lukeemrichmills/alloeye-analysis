import os

import numpy as np
import pandas as pd

from data import d01_raw
from src.d00_utils.file_dir_utils import get_data_dir
from src.d01_data.fetch.group_conditions import group_conditions
from src.d02_intermediate.df_reshaping import add_col_by_lookup

df = group_conditions()

# add group and NP testing columns
drop_cols = ['Notes', 'dob', 'occupation']
pid_info = pd.read_csv(f"{os.path.abspath(d01_raw.__path__[0])}\ppt_info_alloeye.csv")
pid_info['pid'] = pid_info.pid.apply(lambda s: 'alloeye_' + str(s))
info_cols = pid_info.columns
for col in info_cols:
    if col != 'pid':
        df = add_col_by_lookup(df, col, 'ppt_id', pid_info, 'pid', col)
df = df.drop(drop_cols, axis=1)


# replace ppt_id with random integers to avoid any effect of id number
pid_rand = np.random.default_rng().choice(len(df), size=len(df), replace=False)
df.ppt_id = pid_rand

# save to csv
dir = get_data_dir(folder='feature_saves')
file_name = "all_conds"
file_path = dir + f"{file_name}.csv"
df.to_csv(file_path, index=False)


print("end")