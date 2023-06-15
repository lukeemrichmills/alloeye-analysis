import pandas as pd

from src.d03_processing.FeatureWrapper import FeatureWrapper


def get_features(row, full_features, selected_features, feat_args, nullify=False):
    for feature, _tuple in full_features.items():
        if feature in selected_features:
            pd.options.mode.chained_assignment = None  # default='warn'
            row[feature] = get_feature(_tuple, feat_args, nullify)
            pd.options.mode.chained_assignment = 'warn'  # default='warn'
    return row


def get_feature(feature_tuple, feat_args, nullify=False):
    if nullify is True:
        return None
    calculation, out_dtype, low_bound, upp_bound = feature_tuple
    return FeatureWrapper(calculation, feat_args, out_dtype, (low_bound, upp_bound)).out


def select_features_extract(df, func_dict, data_df, features):
    for name, tup in func_dict.items():
        full_features, function = tup
        select_features = [f for f in features if f in full_features]
        df = function(df, data_df, select_features)
    return df
