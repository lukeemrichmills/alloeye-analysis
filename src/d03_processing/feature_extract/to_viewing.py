import numpy as np
import pandas as pd
from pandas import DataFrame

from src.d00_utils.TaskObjects import TaskObjects
from src.d01_data.fetch.fetch_trials import fetch_trials
from src.d03_processing import aoi
from src.d03_processing.BlinkProcessor import BlinkProcessor
from src.d03_processing.feature_extract.general_feature_functions import get_features, get_feature, \
    select_features_extract
from src.d03_processing.fixations.FixAlgos import FixAlgo
from src.d03_processing.Features import Features
from src.d03_processing.feature_calculate.transition_calculations \
    import external_fixations, transition_matrix
from src.d03_processing.feature_calculate.PupilProcessor import PupilProcessor
from src.d03_processing.fixations.SignalProcessor import SignalProcessor
from src.d03_processing.preprocess import preprocess_timepoints

"""
Module to convert fixation/saccade data and raw timepoints into viewing-level features. Feature calculations saved in 
relevant .py files
"""


def to_viewing(viewing_df: DataFrame, fix_sac_df: DataFrame, all_timepoints: DataFrame = None,
               fix_method: FixAlgo = FixAlgo.GazeCollision, features: list = Features.viewing,
               eye_trackloss_threshold=0.25, fix_method_alt_string: str = None):

    # isolate fixation features required
    fix_features = []
    [fix_features.append(feat) for feat in features if feat in Features.viewing_fix_sacc_derived]

    # check for invalid input, filter by target fixation algorithm
    if len(fix_features) > 0:
        if fix_sac_df is None:
            print("not enough fixations or saccades, skipping")
            return None
        else:
            if fix_method_alt_string is None:
                fix_sac_df = fix_sac_df.loc[fix_sac_df.algorithm == fix_method.name]
            else:
                fix_sac_df = fix_sac_df.loc[fix_sac_df.algorithm == fix_method_alt_string]
            if len(fix_sac_df) < 2:
                print("not enough fixations or saccades, skipping")
                return None

    # isolate timepoint features required
    timepoint_features = []
    [timepoint_features.append(feat) for feat in features if feat in Features.viewing_from_timepoints]

    # check valid input
    if len(timepoint_features) > 0 and all_timepoints is None:
        print("not enough timepoints, skipping")
        return None

    # loop through viewings
    skipped=[]
    for i in range(len(viewing_df)):
        # isolate relevant
        viewing_id = viewing_df.loc[i, 'viewing_id']

        # filter timepoints for this viewing, or return none
        if all_timepoints is None:
            timepoints = None
        else:
            timepoints = all_timepoints.loc[all_timepoints.viewing_id == viewing_id].reset_index(drop=True)

        # check for missing timepoints, else preprocess timepoints
        if len(timepoint_features) > 0:
            # preprocess timepoints, skip if empty or not enough timepoints
            timepoints, skip = preprocess_timepoints(timepoints, eye_trackloss_threshold)
            if skip:
                skipped.append(viewing_id)
                continue

        # validate fixations for this viewing
        if fix_sac_df is None:
            fix_df = None
        else:
            fix_df = fix_sac_df.loc[fix_sac_df.viewing_id == viewing_id].sort_values(by=['start_time']).reset_index(drop=True)

        # check if missing
        if len(fix_features) > 0:
            if fix_df is None:
                print(f"not enough fixations or saccades for viewing {viewing_id}, skipping")
                continue
            elif len(fix_df) < 2:
                print(f"not enough fixations or saccades for viewing {viewing_id}, skipping")
                continue

        # for each dataframe, get features
        dfs = {'Fixation': (fix_df, fixation_derived_features),
               'Timepoint': (timepoints, timepoint_derived_features)}
        viewing_df_row = viewing_df.loc[viewing_df['viewing_id'] == viewing_id]   # get row from output dataframe
        for name, tup in dfs.items():
            df = tup[0]
            get_feat_func = tup[1]
            if tup[0] is not None:
                viewing_df.loc[viewing_df['viewing_id'] == viewing_id, :] = get_feat_func(viewing_df_row, df, features)
            else:
                # print(f"{name} df empty for viewing {viewing_id}, not added to viewing df")
                pass
        if (i + 1) % 100 == 0 or (i + 1) == len(viewing_df):
            print(f"Features added to df for {i + 1} of {len(viewing_df)} trials")

    return viewing_df


def fixation_derived_features(viewing_df, fixation_df, features):
    # check if columns added, otherwise add
    func_dict = {
        "fix_sacc": (Features.viewing_fix_sacc, fix_sacc_features),
        "transition": (Features.viewing_transition, transition_features),
    }
    return select_features_extract(viewing_df, func_dict, fixation_df, features)


def timepoint_derived_features(viewing_df, timepoints, features):
    func_dict = {
        "pupil": (Features.viewing_pupil, pupil_diameter_features),
        "headset_loc": (Features.viewing_headset_loc, headset_location_features),
        "selection_delay": (Features.viewing_selection_delay, selection_delay_features),
        'blinks': (Features.viewing_blinks, blink_features),
        'gaze_distance': (Features.viewing_gaze_distance, gaze_distance_features)
    }
    return select_features_extract(viewing_df, func_dict, timepoints, features)


def fix_sacc_features(row, fix_df, features):
    # input validation
    if fix_df is None:
        return None

    # objects
    object_names = [row.obj1_name.values[0], row.obj2_name.values[0], row.obj3_name.values[0], row.obj4_name.values[0]]
    full_objects = np.unique(fix_df.object).tolist()
    incl_table = object_names
    incl_table.append('Table')
    incl_table.append(TaskObjects.invisible_object)
    other_objects = list(set(full_objects) - set(incl_table))
    refix = False
    pd.options.mode.chained_assignment = None  # default='warn'
    for feature, _tuple in Features.viewing_fix_sacc_dict.items():  # for each feature and its function
        if feature in features:  # if not skipping this feature
            _object = feature.split('_')[-1]  # get object(s) of fixations
            if _object[0:3] == 'obj':
                _object = object_names[int(_object[3]) - 1]  # get array object name
            if _object == 'array':
                _object = object_names  # get array objects list
            if _object == 'other':
                _object = other_objects  # get objects not array or table
            if _object == 'table':
                _object = 'Table'
            if _object == 'dome':
                _object = 'OcclusionDome'
            if _object == 'pp':
                _object = TaskObjects.invisible_object

            refix = 'refix' in feature or 'redwell' in feature

            fdf = fix_df.copy()
            if refix:
                fdf = external_fixations(fdf)


            row[feature] = get_feature(_tuple, (fdf, _object))

    pd.options.mode.chained_assignment = 'warn'  # default='warn'

    return row


def transition_features(viewing_row, fixation_df, features):
    # create transition matrix
    # only external fixations
    df_list = [group for _, group in fixation_df.groupby('missing_split_group') if not group.empty]
    for i in range(len(df_list)):
        fix_df = df_list[i].reset_index(drop=True)
        ext_fix_df = external_fixations(fix_df)
        external_df = ext_fix_df if i == 0 else pd.concat([external_df, ext_fix_df]).reset_index(drop=True)

    try:
        external_df = aoi.split_table_fixations_into_areas_of_interest(external_df)
        prob_matrix, objects = transition_matrix(external_df, prob=True, split_by_missing_group=False)
    except UnboundLocalError as e:
        # raise e
        print(e)
        print(f'resorting to full df for {viewing_row.viewing_id}')
        external_df = external_fixations(fix_df)
        external_df = aoi.split_table_fixations_into_areas_of_interest(external_df)
        prob_matrix, objects = transition_matrix(external_df, prob=True, split_by_missing_group=False)

    args = (prob_matrix, objects, external_df)
    return get_features(viewing_row, Features.viewing_transition, features, args)


def pupil_diameter_features(viewing_df, timepoints, features):
    pp = PupilProcessor(timepoints)
    feat_args = (pp.left_diameter, pp.right_diameter)
    return get_features(viewing_df, Features.viewing_pupil, features, feat_args)


def blink_features(viewing_df, timepoints, features):
    return get_features(viewing_df, Features.viewing_blinks, features, [timepoints])


def gaze_distance_features(viewing_df, timepoints, features):
    trial_id = viewing_df.trial_id[viewing_df.index[0]]
    trial = fetch_trials("all", trial_ids=[trial_id])
    return get_features(viewing_df, Features.viewing_gaze_distance, features, [timepoints, trial])


def headset_location_features(viewing_df, timepoints, features):
    return viewing_df


def selection_delay_features(viewing_df, timepoints, features):
    return viewing_df


