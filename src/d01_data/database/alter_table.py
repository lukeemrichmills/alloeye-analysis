from src.d00_utils.type_checks import type_or_list
from src.d01_data.database.PsqlCommander import PsqlCommander
from src.d01_data.database.ToSQL.ToSQL import ToSQL


def add_column(col_name, data_type, table, connection):
    command = f"""
    ALTER TABLE {table}
    ADD COLUMN {col_name} {data_type};
    """
    return PsqlCommander(command).execute_query(connection)


def drop_rows(table, condition_col, condition_id, connection):
    command = f"""delete from {table} where {condition_col} in"""\
              f"""{ToSQL.format_value(type_or_list(condition_id))}"""
    return PsqlCommander(command).execute_query(connection)
