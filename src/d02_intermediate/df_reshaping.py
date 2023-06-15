from src.d00_utils.Conditions import Conditions
from src.d01_data.database.Errors import InvalidValue

import numpy as np


def convert_condition_cols(df, shape="wide"):
    """
    takes and df with move_type and table_rotates columns and converts into 6 separate conditions
    either in wide or long format
    :param df: dataframe with move_type and table_rotates columns
    :param shape: str either "wide" or "long"
    :return:
    """
    if shape == "wide":
        new_col_names = Conditions.list
        move_type_names = 2 * ['Stay', 'Walk', 'Teleport']
        table_rotate_bools = np.repeat([False, True], 3)

        for i in range(6):
            df = convert_condition_col(df, new_col_names[i], move_type_names[i], table_rotate_bools[i])

        df = df.drop(columns=['move_type', 'table_rotates'])
    elif shape == "long":
        new_col_name = 'condition'
        new_vals = Conditions.list
        print("long format not yet written")
    return df


def convert_condition_col(df, new_col, move_type, table_rotates):
    df[new_col] = (df['move_type'] == move_type) & (df['table_rotates'] == table_rotates)

    return df

def add_col_by_lookup(df, new_col_name, by_col, lookup_df, lookup_by_col, return_col):
    """
    adds column to df with new_col_name that searches in df[by_col] for matching in lookup_df[lookup_by_col] and
    returns return_col in df[new_col_name]
    :param df:
    :param new_col_name:
    :param by_col:
    :param lookup_df:
    :param lookup_by_col:
    :param return_col:
    :return:
    """
    df[new_col_name] = np.repeat('NA', len(df))
    for i in range(len(df)):
        lookup_df_rows = lookup_df[lookup_df[lookup_by_col] == df[by_col][i]]
        if len(lookup_df_rows) == 1:
            df[new_col_name][i] = lookup_df_rows[return_col].iloc[0]
        elif len(lookup_df_rows) == 0:
            print("value missing from lookup df")
        elif len(lookup_df_rows) > 1:
            raise InvalidValue(True, False, message="multiple values returned")

    return df


def feature_per_condition(df, feature, groupby="none", groupfunc="mean"):
    """
    return df with 6 condition columns with values as features, grouped by groupby.
    conditions must be named according to convert_condition_cols function.
    default drop all other columns
    """
    if isinstance(df[feature][0], int) or isinstance(df[feature][0], bool):
        df[feature] = df[feature].astype(int)

    condition_cols = Conditions.list

    def cond_col_name(condition_col, suffix):
        return f'{condition_col}_{suffix}'

    new_cols = []
    for i in range(len(condition_cols)):
        new_col = np.repeat(np.nan, len(df))
        for j in range(len(df)):
            if df[condition_cols[i]][j] == 1:
                new_col[j] = df[feature][j]
        new_col_name = cond_col_name(condition_cols[i], feature)
        df[new_col_name] = new_col
        new_cols.append(new_col_name)

    if groupby != "none":
        if groupfunc == "sum":
            df = df.groupby([groupby], as_index=False).sum()
        elif groupfunc == "mean":
            df = df.groupby([groupby], as_index=False).mean()
        elif groupfunc == "std":
            df = df.groupby([groupby], as_index=False).std()
        elif groupfunc == "median":
            df = df.groupby([groupby], as_index=False).median()
        elif groupfunc == "count":
            df = df.groupby([groupby], as_index=False).count()
        elif groupfunc == "proportion":
            df = df.groupby([groupby], as_index=False).sum()
            for i in range(len(condition_cols)):
                new_col = cond_col_name(condition_cols[i], feature)
                df[new_col] = df[new_col] / df[condition_cols[i]]   # divide each new column by trial count
        elif groupfunc == "none":
            df = df.groupby([groupby], as_index=False)
        else:
            raise InvalidValue

    return df, new_cols

