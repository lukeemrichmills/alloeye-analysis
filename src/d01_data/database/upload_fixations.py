import numpy as np

from src.d01_data.database.PsqlCommander import PsqlCommander
from src.d01_data.database.ToSQL.FixationsToSQL import FixationsToSQL
from src.d01_data.database.alter_table import drop_rows
from src.d01_data.database.db_connect import db_connect


def upload_fixations(connection, fix_sac_df, algos, db='local', overwrite=False):

    # get connection
    if connection is None:
        connection = db_connect(db, suppress_print=True)
    if fix_sac_df is None:
        print("nothing to upload")
        return None
    fix_sac_df = fix_sac_df.loc[np.isin(fix_sac_df.algorithm, list(algos.keys()))]
    psql_commander = PsqlCommander()
    fix_col_names = psql_commander.fetch_table_columns(connection, 'fixations_saccades')
    if overwrite:
        viewings_list = fix_sac_df.viewing_id.values.tolist()
        drop_rows('fixations_saccades', 'viewing_id', viewings_list, connection)

    fix_to_sql_commands = FixationsToSQL(fix_sac_df, fix_col_names).convert_to_insert_commands()
    commands_one = "; ".join(fix_to_sql_commands) if type(fix_to_sql_commands) is list else fix_to_sql_commands
    psql_commander.execute_query(connection, commands_one, None)
    print("fixations_uploaded")
