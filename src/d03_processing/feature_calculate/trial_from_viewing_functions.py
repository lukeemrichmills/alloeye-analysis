import pandas as pd


def return_diff(feature, trial, df):
    ret = return_ret(feature, trial, df)
    enc = return_enc(feature, trial, df)
    if ret is None or enc is None:
        return None
    return ret - enc


def return_enc(viewing_feature, trial, viewing_df):
    return return_view_type(viewing_feature, trial, viewing_df, 'enc')


def return_ret(viewing_feature, trial, viewing_df):
    return return_view_type(viewing_feature, trial, viewing_df, 'ret')


def return_view_type(feature, trial, df, view_type):
    feat_split = feature.split('_')
    feat = '_'.join(feat_split[:-1])
    return df[feat][(df.trial_id == trial) &
                    (df.viewing_type == view_type)].values[0]


