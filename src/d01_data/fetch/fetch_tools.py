from src.d00_utils.type_checks import type_or_list
from src.d01_data.database.Errors import InvalidValue
from src.d01_data.database.PsqlCommander import PsqlCommander
from src.d01_data.database.ToSQL.ToSQL import ToSQL
import pandas.io.sql as sqlio


def pid_parse(pid, study_id):
    if pid == "all":
        skip_pid = True
    else:
        skip_pid = False
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

    return pid, skip_pid


def fetch_col_cmd(entry_return_col, table):
    return f"""select {entry_return_col} from {table}"""


def fetch_col(entry_return_col, table, connection):
    return PsqlCommander(fetch_col_cmd(entry_return_col, table)).fetch_query(connection)


def fetch_col_cond_cmd(entry_return_col, table, id_col, row_id):
    return fetch_col_cmd(entry_return_col, table) + f""" where {id_col} = {ToSQL.format_value(row_id)};"""



def fetch_col_cond_list_cmd(entry_return_col, table, id_col, row_id_list):
    return fetch_col_cmd(entry_return_col, table) \
           + f""" where {id_col} in {ToSQL.cat_values(type_or_list(str, row_id_list))};"""


def fetch_col_cond(row_id, id_col, check_col, table, entry_return_col, connection):
    command = fetch_col_cond_cmd(entry_return_col, table, id_col, row_id)
    return PsqlCommander(command).fetch_query(connection)


def fetch_null(row_id, id_col, check_col, table, entry_return_col, connection):
    command = fetch_col_cond_cmd(entry_return_col, table, id_col, row_id)[:-1] +\
              f""" and {check_col} is null;"""

    return PsqlCommander(command).fetch_query(connection)


def fetch_all_null(row_id_list, id_col, check_col, table, entry_return_col, connection):
    command = fetch_col_cond_list_cmd(entry_return_col, table, id_col, row_id_list)[:-1] +\
              f""" and {check_col} is null;"""
    return PsqlCommander(command).fetch_query(connection)


def fetch_not_null(row_id, id_col, check_col, table, entry_return_col, connection):
    command = fetch_col_cond_cmd(entry_return_col, table, id_col, row_id)[:-1] + \
              f""" and {check_col} is not null;"""

    return PsqlCommander(command).fetch_query(connection)


def fetch_practice(table, entry_return_col, connection):
    command = fetch_col_cmd(entry_return_col, table) + \
              f""" where trial_id in (select trial_id from alloeye_trial where practice = true);"""
    return PsqlCommander(command).fetch_query(connection)


def fetch_baseline_calibration(table, entry_return_col, connection):
    command = fetch_col_cmd(entry_return_col, table) + \
              f""" where trial_id in (select trial_id from alloeye_trial where practice = true""" + \
              f""" and trial_number < 0);"""
    return PsqlCommander(command).fetch_query(connection)
