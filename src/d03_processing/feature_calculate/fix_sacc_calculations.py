import numpy as np

from src.d01_data.database.Errors import InvalidValue
from src.d03_processing.FeatureBounds import FeatureBounds as fb


def refix_duration_mean(fix_df, obj):
    df = refixations_only(fix_df)
    return fix_duration_mean(df, obj)


def fix_duration_mean(fix_df, objects):
    obj = 'total'
    fix_df = fix_df_adjust(fix_df, obj)
    fix_df.duration_time.fillna(value=np.nan, inplace=True)
    fix_df = fix_df.reset_index(drop=True)

    if fix_df is None:
        return np.nan
    elif len(fix_df) == 0:
        return 0.0
    elif len(fix_df) < 2:
        return fix_df.duration_time[0]
    else:
        return np.nanmean(fix_df.duration_time)


def n_refix(xdf, obj):
    refixations = refixations_only(xdf)
    return n_fix(refixations, objects=obj)


def redwell(xdf, obj):
    refixations = refixations_only(xdf)
    return dwell_time(refixations, objects=obj)


def refixations_only(fix_df):
    fix_df['PrevFixations'] = fix_df.groupby(['object']).cumcount()
    df_refixations = fix_df[fix_df['PrevFixations'] > 0]
    return df_refixations


def df_adjust(df, objects):
    if isinstance(objects, list):
        return df.iloc[np.isin(df.object, objects), :]
    elif objects == 'total':
        return df
    elif isinstance(objects, str):
        return df[df['object'] == objects]
    else:
        print('NA')


def fix_df_adjust(fix_df, objects):
    try:
        fix_df = fix_df[fix_df['fixation_or_saccade'] == 'fixation']
    except KeyError:
        print("catch")
        raise KeyError
    return df_adjust(fix_df, objects)


def sacc_df_adjust(fix_df, objects):
    sacc_df = fix_df[fix_df['fixation_or_saccade'] == 'saccade']
    return df_adjust(sacc_df, objects)


def n_fix(fix_df, objects):
    fix_df = fix_df_adjust(fix_df, objects)

    return int(len(fix_df))
    

def dwell_time(fix_df, objects, fix_or_sacc='fix'):
    if fix_or_sacc == 'fix':
        fix_df = fix_df_adjust(fix_df, objects)
    elif fix_or_sacc == 'sacc':
        fix_df = sacc_df_adjust(fix_df, objects)
    else:
        df_adjust(fix_df, objects)

    if len(fix_df) < 1:
        return float(0)
    return float(np.sum(fix_df.duration_time))


def t_first(fix_df, objects):
    start = fix_df.start_time[0]
    fix_df = fix_df_adjust(fix_df, objects)
    if len(fix_df) < 1:
        return float(fb.t_first_obj_upper)
    first = min(fix_df.start_time)
    t_first = first - start

    if t_first < 0:
        raise InvalidValue

    return float(t_first)


def n_sacc(fix_df, objects):
    sacc_df = sacc_df_adjust(fix_df, objects)
    return int(len(sacc_df))


def velocity_mean(fix_df, objects):
    return velocity_func(fix_df, objects, 2, np.nanmean)


def velocity_std(fix_df, objects):
    return velocity_func(fix_df, objects, 2, np.nanstd)


def velocity_func(df, objects, min_len, func):
    v_df = velocity(df, objects)
    return min_len_df_func(v_df, min_len, func)


def velocity(fix_df, objects):
    sacc_df = sacc_df_adjust(fix_df, objects)
    return sacc_df.mean_velocity


def min_len_df_func(df, min_len, func):

    if df is not None and len(df) > min_len:
        try:
            return func(df)
        except TypeError as e:
            raise e
    else:
        return np.nan


def dispersion_mean(fix_df, objects):
    fix_df = fix_df_adjust(fix_df, 'total')
    fix_df.dispersion.fillna(value=np.nan, inplace=True)
    fix_df = fix_df.reset_index(drop=True)
    return min_len_df_func(fix_df.dispersion, 2, np.nanmean)


def total_invalid(fix_df, objects):
    fix_df.invalid_duration.fillna(value=np.nan, inplace=True)
    return np.nansum(fix_df.invalid_duration)
