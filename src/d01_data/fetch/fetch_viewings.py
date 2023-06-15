import warnings

from src.d01_data.database.db_connect import db_connect
from src.d00_utils.Conditions import Conditions
from .sql_convert import *
from src.d01_data.database.Errors import InvalidValue
import pandas.io.sql as sqlio

from ...d00_utils.type_checks import type_or_list


def fetch_viewings(pid, conditions=Conditions.all, viewing_type="trial", viewing_list=[],
                   trial_numbers="all", trial_ids=[], blocks="all", practice=False, cols="all",
                   table="viewing", study_id="alloeye", db='local'):

    print("- fetching viewings from db...")

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
    # if not (table == 'viewing' or table == 'all'):
    #     raise InvalidValue(True, False, "study_id input invalid")

    # viewing type query
    viewing_type = ['enc', 'ret'] if viewing_type == "trial" else viewing_type
    viewing_type = type_or_list(str, viewing_type)
    viewing_string = CsvToSQL.cat_values(viewing_type)

    # trial query
    move_type_string, rot_type_string = condition_to_sql(conditions)
    trial_numbers = [i for i in range(18)] if trial_numbers == "all" else trial_numbers
    trial_numbers = type_or_list(int, trial_numbers)
    trials_string = CsvToSQL.cat_values(trial_numbers)

    # block query
    blocks = [1, 2, 3] if blocks == "all" else blocks
    blocks = type_or_list(int, blocks)
    practice = type_or_list(bool, practice)
    blocks_str = CsvToSQL.cat_values(blocks)
    practice_str = CsvToSQL.cat_values(practice)
    ppt_str = CsvToSQL.cat_values(type_or_list(str, pid))

    # construct postgres queries
    root_query = f"""select {cols_str} from "alloeye_viewing" """
    if len(viewing_list) > 0:
        viewing_id_string = CsvToSQL.cat_values(type_or_list(str, viewing_list))
        base_query = f"""where viewing_id in {viewing_id_string}"""
        query = f'{root_query}{base_query}'
    elif len(trial_ids) > 0:
        trial_ids_str = CsvToSQL.cat_values(type_or_list(str, trial_ids))
        base_query = f"""where trial_id in {trial_ids_str}"""
        query = f'{root_query}{base_query}'
    else:
        base_query = f"""where viewing_type in {viewing_string} """ \
                     f"""and trial_id in ("""

        trial_query = f"""select trial_id from "alloeye_trial" """ \
                      f"""where move_type in {move_type_string} """ \
                      f"""and table_rotates in {rot_type_string} """ \
                      f"""and trial_number in {trials_string} """ \
                      f"""and block_id in ("""

        block_query_1 = f"""select block_id from "block" """ \
                      f"""where block_order in {blocks_str} """ \
                      f"""and practice in {practice_str} """

        block_query = block_query_1 + block_query_2

        query = f'{root_query}{base_query}{trial_query}{block_query}));'

    # execute and fetch query
    warnings.filterwarnings("ignore")
    df = sqlio.read_sql_query(query, conn)
    warnings.filterwarnings("default")
    print("viewings fetched from db")

    conn.close()
    return df

