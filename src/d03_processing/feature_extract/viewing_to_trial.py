import warnings

import numpy as np
import pandas as pd

from src.d01_data.fetch.fetch_fixations import fetch_fixations
from src.d01_data.fetch.fetch_viewings import fetch_viewings
from src.d03_processing.Features import Features
from src.d03_processing.feature_extract.general_feature_functions import get_features, get_feature
from src.d03_processing.feature_extract.timepoint_to_viewing import timepoint_to_viewing


def viewing_to_trial(trial_df, viewing_df, features, bc=False, practice=False, fix_algo="VR_IDT"):
    trial_df = trial_df.copy()
    # # remove the following when made into a function
    # viewing_df = timepoint_to_viewing(['25'])
    # bc = False
    # practice = 'exclude'

    if bc is False:
        viewing_df = viewing_df.loc[viewing_df.viewing_type != 'bc']
    else:
        viewing_df = viewing_df.loc[viewing_df.viewing_type == 'bc']

    if practice is False:
        viewing_df = viewing_df.loc[np.invert(viewing_df.trial_id.str.contains('p'))]
    else:
        pass    # will include automatically

    viewing_feats = []
    [viewing_feats.append(feat) for feat in features if feat in Features.trial_from_viewing_dict.keys()]



    compare_feats = []
    [compare_feats.append(feat) for feat in features if feat in Features.trial_view_compare_dict.keys()]
    if len(compare_feats) > 0:
        viewing_list = []
        [viewing_list.append(f"{trial_id}_enc") for trial_id in trial_df.trial_id.to_list()]
        [viewing_list.append(f"{trial_id}_ret") for trial_id in trial_df.trial_id.to_list()]
        fixation_df = fetch_fixations("all", viewing_id=viewing_list, algorithm=fix_algo)
    else:
        fixation_df = None

    for i in range(len(trial_df)):
        nullify_viewing = False
        nullify_compare = False
        trial_id = trial_df.loc[i, 'trial_id']
        trial_df_row = trial_df.loc[trial_df['trial_id'] == trial_id].copy()
        view_df = viewing_df.loc[viewing_df.trial_id == trial_id].reset_index(drop=True)
        if fixation_df is None:
            fix_df = None
        else:
            fix_df = fixation_df.loc[fixation_df.trial_id == trial_id].reset_index(drop=True)

        if len(viewing_feats) > 0 and len(view_df) < 2:
            print(f"not enough viewings, setting null for {trial_id} viewing features")
            nullify_viewing = True

        if len(compare_feats) > 0 and fix_df is None:
            print(f"not enough fixations, setting null for {trial_id} compare features")
            nullify_compare = True
        elif len(compare_feats) > 0 and len(fix_df) < 2 and nullify_compare is False:
            print(f"not enough fixations, setting null for {trial_id} compare features")
            nullify_compare = True

        func_list = [(view_df, viewing_trial_features, nullify_viewing),
                     (fix_df, viewing_comparison_features, nullify_compare)]

        # if nullify_viewing :
        #     print("") # debug
        #
        # if nullify_compare:
        #     print("")  # debug

        for tuple_ in func_list:
            df = tuple_[0]
            get_feature_func = tuple_[1]
            nullify = tuple_[2]
            try:
                pd.options.mode.chained_assignment = None  # default='warn'
                trial_df.loc[trial_df['trial_id'] == trial_id, :] = get_feature_func(trial_df_row, df, features, nullify)
                pd.options.mode.chained_assignment = 'warn'  # default='warn'
            except:
                print("catch")
        if (i + 1) % 50 == 0 or (i+1) == len(trial_df):
            print(f"Features added to df for {i + 1} of {len(trial_df)} trials")

    return trial_df


def viewing_trial_features(trial_df_row, view_df, select_features, nullify):
    trial_id = trial_df_row.loc[trial_df_row.index[0], 'trial_id']
    for feature, _tuple in Features.trial_from_viewing_dict.items():
        if feature in select_features:
            feat_args = (feature, trial_id, view_df)
            pd.options.mode.chained_assignment = None  # default='warn'
            trial_df_row[feature] = get_feature(_tuple, feat_args, nullify)
            pd.options.mode.chained_assignment = 'warn'  # default='warn'
    return trial_df_row


def viewing_comparison_features(trial_df_row, fix_df, select_features, nullify):
    if fix_df is None:
        return trial_df_row
    fix_df = fix_df.sort_values(by='start_time').reset_index(drop=True)
    return get_features(trial_df_row, Features.trial_view_compare_dict, select_features, (trial_df_row, fix_df), nullify)

