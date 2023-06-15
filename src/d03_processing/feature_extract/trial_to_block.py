# get data
import os
import platform

import numpy as np
import pandas as pd

from data import d01_raw
from src.d00_utils.Conditions import Conditions
from src.d01_data.database.Errors import UnmatchingValues, UnknownComputer
from src.d01_data.fetch.fetch_trials import fetch_trials
from src.d02_intermediate.df_reshaping import convert_condition_cols, feature_per_condition, add_col_by_lookup
from src.d03_processing.feature_extract.timepoint_to_viewing import timepoint_to_viewing
from src.d03_processing.feature_extract.viewing_to_trial import viewing_to_trial

def trial_to_block(ppts, via_timepoints=False, timepoints=None):
    """"""
    # for saving files
    icn_pc = "DESKTOP-IBAG161"
    personal_laptop = "LAPTOP-OOBPQ1A8"
    if platform.uname().node == icn_pc:
        save_dir = "C:\\Users\\Luke Emrich-Mills\\Documents\\AlloEye\\MainDataOutput\\feature_saves\\"
    elif platform.uname().node == personal_laptop:
        save_dir = "C:\\Users\\Luke\\Documents\\AlloEye\\data\\feature_saves\\"
    else:
        raise UnknownComputer

    # get ppt info for adding group variable
    pid_info = pd.read_csv(f"{os.path.abspath(d01_raw.__path__[0])}\ppt_info_alloeye.csv")
    pid_info['alloeye_id'] = pid_info.pid.apply(lambda s: 'alloeye_' + str(s))

    # get trial info from db
    print("fetching trials from db...")
    df = fetch_trials(ppts, conditions=Conditions.all, trials="all")

    # add correct column by trial - don't move this
    df['correct'] = (df['object_shifted'] == df['selected_object'])
    # add condition columns dummy variables
    df = convert_condition_cols(df)  # gets 6 condition columns, value is true if that condition AND correct trial

    # if indicated, get eye features from feature_extract via viewings
    if via_timepoints:
        print("fetching viewing features from feature_extract...")
        viewing_df = timepoint_to_viewing(ppts, timepoints)
        try:
            viewing_df.to_csv(f"{save_dir}viewing_df_block.csv", index=False)
        except:
            print("catch - don't lose!")
        print("converting viewing features into trial features")
        df_2 = viewing_to_trial(viewing_df)
        try:
            df_2.to_csv(f"{save_dir}trial_df2_block.csv", index=False)
        except:
            print("catch - don't lose!")
        if len(df_2) != len(df):
            raise UnmatchingValues
        if all(df.trial_id == df_2.trial_id):
            df = pd.merge(df, df_2, how="outer", on="trial_id")
        else:
            raise UnmatchingValues

    print("rearranging data")
    pd.options.mode.chained_assignment = None  # default='warn'
    # big catch - save data first
    # by group overall -
    # save stem df - columns not summed or meaned etc.
    col_index = np.where(df.columns.values == 'correct')[0][0]
    stem_df = df.iloc[:, :col_index]

    # trial_id_col = stem_df.loc[:, 'trial_id']
    pid_col = stem_df.loc[:, 'ppt_id']
    # split correct and conditions (and other summed features) into one df, and eye (or other meaned features) into another
    summing_df = df.loc[:, 'correct':Conditions.list[-1]]
    summing_df = pd.concat([pid_col, summing_df], axis=1)   # add ppt_id
    av_df = df.loc[:, 'Hn_enc':]
    av_df = pd.concat([pid_col, av_df], axis=1)

    # group each by ppts (as_index = False) and convert to df by applying sum or mean
    summed_df = summing_df.groupby(['block_id'], as_index=False).sum()
    summed_df['n_trials'] = (summed_df[Conditions.list[0]] + summed_df[Conditions.list[1]] + summed_df[Conditions.list[2]] +
                             summed_df[Conditions.list[3]] + summed_df[Conditions.list[4]] + summed_df[Conditions.list[5]])
    summed_df['p_correct'] = summed_df['correct'] / summed_df['n_trials']

    aved_df = av_df.groupby(['block_id'], as_index=False).mean()

    all_df = pd.merge(summed_df, aved_df, how="outer", on="ppt_id")

    # add group column by lookup
    all_df = add_col_by_lookup(all_df, 'group', 'ppt_id', pid_info, 'alloeye_id', 'group')

    # per condition
    # save stem df with correct - remove eye features (anything not proportioned per condition)
    col_index = np.where(df.columns.values == 'Hn_enc')[0][0]
    corr_df = df.iloc[:, :col_index]
    # save stem df without correct - only features to mean
    # for correct only, use feature_per_condition with 'proportion' groupfunc then melt with var_name 'condition' and value_name = 'correct'
    cond_corr_df, new_cols = feature_per_condition(corr_df, 'correct', groupby='block_id', groupfunc='proportion')
    cond_corr_df_long = cond_corr_df.reset_index().melt(id_vars='block_id', value_vars=new_cols,
                                                        var_name='condition', value_name='p_correct')
    # repeat above per feature, using group_func 'mean' instead
    full_cond_df = cond_corr_df_long
    for feature in av_df.columns.values:
        aving_df = av_df
        if feature != 'block_id':
            aving_df = pd.concat([aving_df, df.loc[:, Conditions.list[0]:Conditions.list[-1]]], axis=1)
            cond_df, new_cols = feature_per_condition(aving_df, feature, groupby='block_id', groupfunc='mean')
            cond_df = cond_df.drop([feature], axis=1)
            new_name = f'{feature}'
            cond_df_long = cond_df.reset_index().melt(id_vars='block_id', value_vars=new_cols,
                                                      var_name='condition', value_name=new_name)
            full_cond_df = pd.concat([full_cond_df, cond_df_long.drop(['block_id', 'condition'], axis=1)], axis=1)
    full_cond_df = add_col_by_lookup(full_cond_df, 'group', 'ppt_id', pid_info, 'alloeye_id', 'group')

    pd.options.mode.chained_assignment = 'warn'  # default='warn'
    return all_df, full_cond_df
