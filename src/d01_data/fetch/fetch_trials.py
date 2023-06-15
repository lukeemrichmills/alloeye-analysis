import numpy as np
import pandas as pd

from src.d01_data.database.db_connect import db_connect
from src.d00_utils.Conditions import Conditions
from src.d00_utils.type_checks import type_or_list
from .sql_convert import *
from src.d01_data.database.Errors import InvalidValue
import pandas.io.sql as sqlio
import warnings


def fetch_trials(pid, conditions=Conditions.all, trial_ids=[], condition_ids=[],
                 trial_numbers="all", blocks="all", practice=False, cols="all",
                 study_id="alloeye", db='local', table='alloeye_trial',
                 suppress_print=True, remove_training_trials=True):
    all_pids = False
    if pid == "all":
        all_pids = True
        block_query_2 = ""
    else:
        if isinstance(pid, str):
            if study_id not in pid:
                print("study_id not in pid, altering pid")
                pid = f"{study_id}_{pid}"
        elif isinstance(pid, list):
            for i in range(len(pid)):
                if study_id not in pid[i]:
                    print("study_id not in pid, altering pid")
                    pid[i] = f"{study_id}_{pid[i]}"
        else:
            raise InvalidValue(True, False, "pid input invalid")
        ppt_str = CsvToSQL.cat_values(type_or_list(str, pid))
        block_query_2 = f"""and block.ppt_id in {ppt_str}"""

    # connect to database - use separate connection class
    conn = db_connect(db, suppress_print=True)

    # check and format inputs
    # base query
    cols_str = '*' if cols == "all" else CsvToSQL.cat_values(type_or_list(str, cols), True)

    # trial query
    move_type_string, rot_type_string = condition_to_sql(conditions)
    trial_numbers = [i for i in range(18)] if trial_numbers == "all" else trial_numbers
    trial_numbers_string = CsvToSQL.cat_values(type_or_list(int, trial_numbers))


    # block query
    blocks = [1, 2, 3] if blocks == "all" else blocks
    blocks = type_or_list(int, blocks)
    practice = type_or_list(bool, practice)
    blocks_str = CsvToSQL.cat_values(blocks)
    practice_str = CsvToSQL.cat_values(practice)

    root_query = f"""select {cols_str} from {table} """
    if len(trial_ids) > 0:
        trial_ids_str = CsvToSQL.cat_values(type_or_list(str, trial_ids))
        base_query = f"""where trial_id in {trial_ids_str} """
        query = f'{root_query}{base_query}'
    elif len(condition_ids) > 0:
        condition_ids_str = CsvToSQL.cat_values(type_or_list(str, condition_ids))
        base_query = f"""where condition_id in {condition_ids_str} """ \
                     f"""and practice in {practice_str} """
        query = f'{root_query}{base_query}'
    else:
        trial_col_name = 'trial_number'
        # construct postgres queries
        base_query =  f"""where move_type in {move_type_string} """ \
                      f"""and table_rotates in {rot_type_string} """ \
                      f"""and trial_number in {trial_numbers_string} """ \
                      f"""and block_id in ("""

        block_query_1 = f"""select block_id from "block" """ \
                        f"""where block_order in {blocks_str} """ \
                        f"""and practice in {practice_str} """

        block_query = block_query_1 + block_query_2

        query = f'{root_query}{base_query}{block_query})'

    # execute and fetch query as df
    warnings.filterwarnings("ignore")
    if suppress_print is False:
        print(query)
    df = sqlio.read_sql_query(query, conn)
    if remove_training_trials:
        df = training_trial_removal(df)
    df = multimatch_col_split(df)   # split multimatch string into respective columns
    warnings.filterwarnings("default")
    conn.close()
    return df


def multimatch_col_split(df):
    import json
    from json.decoder import JSONDecodeError
    new_cols = ['vector', 'direction', 'length', 'position', 'duration']
    new_cols = [f'mm_{i}' for i in new_cols]
    new_dict = {}
    for col in new_cols:
        new_dict[col] = []
    # print(new_dict)
    df = df.reset_index(drop=True)

    for i in range(len(df)):
        try:
            mm_values = json.loads(df.multimatch[i])
        except (TypeError, JSONDecodeError) as e:
            mm_values = [None] * 5
        if len(mm_values) < 5:
            mm_values = [None] * 5
        for j, col in enumerate(new_cols):
            new_dict[col].append(mm_values[j])

    df[new_cols] = pd.DataFrame(new_dict)
    return df


def training_trial_removal(trial_df):
    training_configurations = [i for i in range(55, 64)]
    shape_before = trial_df.shape[0]
    trial_df = trial_df.loc[np.invert(np.isin(trial_df.configuration_number, training_configurations))]
    return trial_df.reset_index(drop=True)
