import time
import warnings

from src.d01_data.database.db_connect import db_connect
from src.d00_utils.Conditions import Conditions
from . import fetch_tools
from .sql_convert import *
from src.d01_data.database.Errors import InvalidValue
import pandas.io.sql as sqlio

from ...d00_utils.type_checks import type_or_list


def fetch_timepoints(pid, conditions=Conditions.all, viewing_id="conditional", viewing_type="trial",
                     trials="all", blocks="all", practice=False, cols="all",
                     table="viewing", study_id="alloeye", ret_epochs=['na', 'view'],
                     configurations="all",
                     db='local', suppress_print=False):
    pid, skip_pid = fetch_tools.pid_parse(pid, study_id)

    # connect to database - use separate connection class
    conn = db_connect(db, suppress_print=True)

    # check and format inputs
    # base query
    cols_str = '*' if cols == "all" else CsvToSQL.cat_values(type_or_list(str, cols), True)
    if not (table == 'viewing' or table == 'all' or
            table == "alloeye_timepoint_viewing" or
            table == "alloeye_timepoint_all"):
        raise InvalidValue(True, False, "table input invalid")
    if table == 'viewing' or table == 'all':
        table_name = f"alloeye_timepoint_{table}"
    else:
        table_name = table

    ppt_str = "" if skip_pid else CsvToSQL.cat_values(type_or_list(str, pid))
    ret_epoch_str = CsvToSQL.cat_values(type_or_list(str, ret_epochs))



    base_query = f"""SELECT {cols_str} """ \
                 f"""FROM "{table_name}" """\
                 f"""WHERE retrieval_epoch in {ret_epoch_str} """

    if viewing_id == "conditional":
        viewing_string, move_type_string, rot_type_string, trials_string, config_string, blocks_str, practice_str = \
            get_conditional_strings(viewing_type, conditions, trials, configurations, blocks, practice)

        viewing_query, trial_query, block_query = build_conditional_subqueries(viewing_string, move_type_string,
                                                                               rot_type_string, trials_string, config_string,
                                                                               blocks_str, practice_str, ppt_str)
    else:
        viewing_id_string = CsvToSQL.cat_values(type_or_list(str, viewing_id))
        viewing_query, trial_query, block_query = build_viewing_query(viewing_id_string, ppt_str)

    query = f'{base_query}{viewing_query}{trial_query}{block_query}'

    # execute and fetch query
    if suppress_print is False:
        print(f"executing query\n{query}")
    start_time = time.time()
    warnings.filterwarnings("ignore")
    df = sqlio.read_sql_query(query, conn)
    df = df.sort_values(by='eye_timestamp_ms').reset_index(drop=True)  # sort by time by default
    warnings.filterwarnings("default")
    time_elapsed = time.time() - start_time
    print(f"dataframe returned in {time_elapsed}")

    conn.close()
    return df


def get_conditional_strings(viewing_type, conditions, trials, configurations, blocks, practice):

    # viewing type query strings
    viewing_type = ['enc', 'ret'] if viewing_type == "trial" else viewing_type
    viewing_type = type_or_list(str, viewing_type)
    viewing_string = CsvToSQL.cat_values(viewing_type)

    # trial query
    move_type_string, rot_type_string = condition_to_sql(conditions)
    trials = [i for i in range(18)] if trials == "all" else trials
    trials = type_or_list(int, trials)
    trials_string = CsvToSQL.cat_values(trials)
    configurations = [i for i in range(100)] if configurations == "all" else configurations
    configurations = type_or_list(int, configurations)
    config_string = CsvToSQL.cat_values(configurations)

    # block query
    blocks = [1, 2, 3] if blocks == "all" else blocks
    blocks = type_or_list(int, blocks)
    practice = type_or_list(bool, practice)
    blocks_str = CsvToSQL.cat_values(blocks)
    practice_str = CsvToSQL.cat_values(practice)

    return viewing_string, move_type_string, rot_type_string, trials_string, config_string, blocks_str, practice_str


def build_conditional_subqueries(viewing_string, move_type_string, rot_type_string, trials_string, config_string,
                                 blocks_str, practice_str, ppt_str):
    # construct postgres queries
    viewing_base_query = f"""and viewing_id in ("""

    viewing_query = f"""select viewing_id from "alloeye_viewing" """ \
                    f"""where viewing_type in {viewing_string} """ \
                    f"""and trial_id in ("""

    trial_query = f"""select trial_id from "alloeye_trial" """ \
                  f"""where move_type in {move_type_string} """ \
                  f"""and table_rotates in {rot_type_string} """ \
                  f"""and trial_number in {trials_string} """ \
                  f"""and configuration_number in {config_string}""" \
                  f"""and block_id in ("""

    block_query = f"""select block_id from "block" """ \
                  f"""where block_order in {blocks_str} """ \
                  f"""and practice in {practice_str} """

    ppt_query = "" if ppt_str == "" else f"""and block.ppt_id in {ppt_str})))"""
    block_query = block_query + ppt_query

    return f'{viewing_base_query}{viewing_query}', trial_query, block_query


def build_viewing_query(viewing_id_string, ppt_str):
    if viewing_id_string == "all":
        viewing_query = ""
    else:
        viewing_query = f"""and viewing_id in {viewing_id_string} """
    trial_query = ""
    block_query = "" if ppt_str == "" else f"""and ppt_id in {ppt_str}"""
    return viewing_query, trial_query, block_query
