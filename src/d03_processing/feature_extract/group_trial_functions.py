import numpy as np


def n_trials(args):
    trials_df = args[0]
    return int(len(trials_df))


def n_correct(args):
    trials_df = args[0]
    correct_df = trials_df[trials_df['object_shifted'] == trials_df['selected_object']]
    return n_trials([correct_df])


def p_correct(args):
    trials_df = args[0]
    return n_correct([trials_df]) / n_trials([trials_df])


def trial_mean(args):
    trials_df = args[0]
    feature = args[1]
    feature_col = trials_df[feature]
    feature_col.fillna(value=np.nan, inplace=True)
    try:
        out = np.nanmean(feature_col)
    except TypeError as e:
        raise e
    return out


def trial_median(args):
    trials_df = args[0]
    feature = args[1]
    feature_col = trials_df[feature]
    feature_col.fillna(value=np.nan, inplace=True)
    try:
        out = np.nanmedian(feature_col)
    except TypeError as e:
        raise e
    return out

def confidence_mean(args):
    trials_df = args[0]
    c = np.where(trials_df.confidence_rating == -999, np.nan, trials_df.confidence_rating)
    return np.nanmean(c)


def mm_vector(args):
    trials_df = args[0]
