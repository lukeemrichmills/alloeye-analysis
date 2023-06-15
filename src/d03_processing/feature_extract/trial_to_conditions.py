import numpy as np
import pandas as pd

from src.d03_processing.Features import Features
from src.d03_processing.feature_extract.general_feature_functions import get_feature


def trial_to_conditions(cond_df, trial_df, features, practice=False, fix_algo='VR_IDT'):
    # per condition

    for i in range(len(cond_df)):
        nullify = False
        condition_id = cond_df.loc[i, 'condition_id']
        cond_df_row = cond_df.loc[cond_df['condition_id'] == condition_id]
        t_df = trial_df.loc[trial_df.condition_id == condition_id].reset_index(drop=True)
        if len(t_df) < 5:
            print(f"not enough trials, setting null {condition_id}")
            nullify = True

        for feature in features:
            args = [(t_df, feature)]
            feat_tuple = Features.conditions_dict[feature]

            pd.options.mode.chained_assignment = None  # default='warn'
            try:
                cond_df_row[feature] = get_feature(feat_tuple, args, nullify)
            except TypeError as e:
                raise e
            pd.options.mode.chained_assignment = 'warn'  # default='warn'

        cond_df.loc[cond_df['condition_id'] == condition_id, :] = cond_df_row
        if (i+1) % 20 == 0 or (i+1) == len(cond_df):
            print(f"Features added to df for {i+1} of {len(cond_df)} conditions")

    return cond_df
