from src.d00_utils.type_checks import type_or_list
from src.d01_data.database.ToSQL.ToSQL import ToSQL
from src.d01_data.database.db_connect import db_connect
from src.d01_data.fetch import fetch_tools
import pandas.io.sql as sqlio

def fetch_conditions(pid, condition_ids=[],
                     cols='all', study_id='alloeye',  db='local', table='alloeye_conditions'):
    ppts, skip_pid = fetch_tools.pid_parse(pid, study_id)

    # connect to database - use separate connection class
    conn = db_connect(db, suppress_print=True)

    cols_str = '*' if cols == "all" else ToSQL.cat_values(type_or_list(str, cols), True)

    root_query = f"""select {cols_str} from {table} """
    if len(condition_ids) > 0:
        condition_ids_str = ToSQL.cat_values(type_or_list(str, condition_ids))
        base_query = f"""where condition_id in {condition_ids_str}"""
        query = f'{root_query}{base_query}'
    else:
        print("haven't written this function yet!")
        query = root_query


    # execute and fetch query as df
    df = sqlio.read_sql_query(query, conn)
    conn.close()
    return df
