import time
import warnings

from src.d00_utils.Conditions import Conditions
from src.d00_utils.type_checks import type_or_list
from src.d01_data.database.Errors import InvalidValue
from src.d01_data.database.ToSQL.CsvToSQL.CsvToSQL import CsvToSQL
from src.d01_data.database.ToSQL.ToSQL import ToSQL
from src.d01_data.database.db_connect import db_connect
from src.d01_data.fetch import fetch_tools
import pandas.io.sql as sqlio


def fetch_fixations(pid, conditions=Conditions.all, viewing_id="all", viewing_type="trial",
                    trials="all", blocks="all", practice=False, cols="all",
                    table="fixations_saccades", study_id="alloeye", algorithm=["GazeCollision"],
                    conn=None, db='local',
                    suppress_print=True,
                    random=False, random_n=1):

    pid, skip_pid = fetch_tools.pid_parse(pid, study_id)

    # connect to database - use separate connection class
    if conn is None:
        conn = db_connect(db, suppress_print=True)

    # check and format inputs
    # base query
    cols_str = '*' if cols == "all" else ToSQL.cat_values(type_or_list(str, cols), True)

    ppt_str = "" if skip_pid else ToSQL.cat_values(type_or_list(str, pid))

    algo_str = ToSQL.cat_values(type_or_list(str, algorithm))

    base_query = f"""SELECT {cols_str} """ \
                 f"""FROM "{table}" """ \
                 f"""WHERE algorithm in {algo_str} """

    viewing_id_string = CsvToSQL.cat_values(type_or_list(str, viewing_id))
    viewing_query, ppt_query = build_viewing_query(viewing_id_string, ppt_str)
    # print(viewing_query)
    # print(ppt_query)
    query = f'{base_query}{viewing_query}{ppt_query}'


    # execute and fetch query
    if suppress_print is False:
        print(f"executing query\n{query}")
    start_time = time.time()
    warnings.filterwarnings("ignore")
    df = sqlio.read_sql_query(query, conn)
    df = df.sort_values(by='start_time').reset_index(drop=True)   # put in time order by default
    warnings.filterwarnings("default")
    time_elapsed = time.time() - start_time
    print(f"- dataframe returned in {time_elapsed}")

    conn.close()
    return df


def build_viewing_query(viewing_id_string, ppt_str):
    # print(viewing_id_string)
    if viewing_id_string == "('all')":
        viewing_query = ""
    else:
        viewing_query = f"""and viewing_id in {viewing_id_string} """
    ppt_query = "" if ppt_str == "" else f"""and ppt_id in {ppt_str}"""
    return viewing_query, ppt_query


