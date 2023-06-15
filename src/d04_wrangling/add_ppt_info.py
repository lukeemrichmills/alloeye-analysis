import os

import numpy as np
import pandas as pd

from data import d01_raw
from src.d02_intermediate.df_reshaping import add_col_by_lookup



def add_ppt_info(df, add_cols=[], drop_cols=['Notes', 'dob', 'occupation']):
    pid_info = pd.read_csv(f"{os.path.abspath(d01_raw.__path__[0])}\ppt_info_alloeye.csv")
    pid_info['pid'] = pid_info.pid.apply(lambda s: 'alloeye_' + str(s))
    add_cols = pid_info.columns if add_cols == [] else add_cols
    pd.options.mode.chained_assignment = None  # default='warn'
    for col in add_cols:
        if col != 'pid':
            df = add_col_by_lookup(df, col, 'ppt_id', pid_info, 'pid', col)
    if len(drop_cols) > 0:
        df = df.drop(drop_cols, axis=1)
    pd.options.mode.chained_assignment = 'warn'  # default='warn'
    # replace ppt_id with random integers to avoid any effect of id number
    # pid_rand = np.random.default_rng().choice(len(df), size=len(df), replace=False)
    # df.ppt_id = pid_rand

    return df


def get_ppts_by_group():
    pid_info = pd.read_csv(f"{os.path.abspath(d01_raw.__path__[0])}\ppt_info_alloeye.csv")
    pid_info['pid'] = pid_info.pid.apply(lambda s: 'alloeye_' + str(s))
    pid_info = pid_info[(pid_info.biomarkers != "") & (pid_info["VR test date"] != 'DECLINED')]

    out_dict = {'Younger': pid_info.pid[pid_info.group == 'Y'].to_list(),
                'Older': pid_info.pid[pid_info.group == 'O'].to_list(),
                'MCI+': pid_info.pid[pid_info.group == 'MCI+'].to_list(),
                'MCI-': pid_info.pid[pid_info.group == 'MCI-'].to_list(),
                'MCIu': pid_info.pid[(pid_info.group == 'P') | (pid_info.group == 'MCI')].to_list()
                }

    return out_dict

def group_plot_order():
    return ['Y', 'O', 'MCI-', 'MCI+', 'MCIu']