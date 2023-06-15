import numpy as np
import pandas as pd
from Levenshtein import distance, ratio

from src.d00_utils.TaskObjects import TaskObjects
from src.d01_data.database.Errors import InvalidValue
from src.d03_processing import aoi
from src.d03_processing.feature_calculate.transition_calculations import external_fixations
from src.d03_processing.feature_calculate.viewing_compare_calcs import fixation_df_comparison_setup


def lev_dist_xfix_s(trial_row, fix_df):
    return lev_feature_standard_aoi(trial_row, fix_df, levenshtein_distance)


def lev_ratio_xfix_s(trial_row, fix_df):
    return lev_feature_standard_aoi(trial_row, fix_df, levenshtein_ratio)


def lev_feature_standard_aoi(trial_row, fix_df, distance_function):
    if fix_df is None:
        return np.nan

    view1_fix, view2_fix = fixation_df_comparison_setup(fix_df, trial_row)
    if view1_fix is None:
        return np.nan
    view1_fix = aoi.convert_AOIs(view1_fix)
    view2_fix = aoi.convert_AOIs(view2_fix)

    string1 = standard_aoi_to_string(view1_fix)
    string2 = standard_aoi_to_string(view2_fix)

    return distance_function(string1, string2)


def levenshtein_ratio(string1, string2):
    return ratio(string1, string2)


def levenshtein_distance(string1, string2):
    return distance(string1, string2)


def standard_aoi_to_string(fix_df):

    objects = fix_df.sort_values(by="start_time").object.to_list()
    if not any(np.isin(objects, TaskObjects.standard_aois)):
        raise InvalidValue(objects, TaskObjects.standard_aois,
                           message=f"fixation objects not in standard aois: {objects}")
    string = ""
    for i in range(len(objects)):
        string = f"{string}{TaskObjects.standard_aoi_chars[objects[i]]}"

    return string



def aoi_to_string():
    pass
