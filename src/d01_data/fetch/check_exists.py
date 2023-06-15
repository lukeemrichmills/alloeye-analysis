from src.d00_utils.type_checks import type_or_list
from src.d01_data.database.PsqlCommander import PsqlCommander
from src.d01_data.database.ToSQL.ToSQL import ToSQL
from src.d01_data.fetch import fetch_tools


def entry(row_id, id_col, check_col, check_value, check_dtype, table, connection) -> bool:
    """actually checks for null entry (assumes table column exists)"""
    command = f"""
    select exists (select {check_col} 
			       from {table}
			       where {id_col} = {ToSQL.format_value(row_id)}
			       and {check_col} in {ToSQL.cat_values(type_or_list(check_dtype, check_value))});
    """
    return PsqlCommander(command).fetch_bool_query(connection)


def entries_not_null(row_id, id_col, check_col, table, entry_return_col, connection) -> bool:
    """actually checks for null entry (assumes table column exists)"""
    entries = fetch_tools.fetch_null(row_id, id_col, check_col, table, entry_return_col, connection)
    return len(entries) == 0, entries


def column(col, table, connection) -> bool:
    command = f"""
    SELECT EXISTS (SELECT 1 
                   FROM information_schema.columns 
                   WHERE table_name='{table}' AND column_name='{col}');
    """
    return PsqlCommander(command).fetch_bool_query(connection)

