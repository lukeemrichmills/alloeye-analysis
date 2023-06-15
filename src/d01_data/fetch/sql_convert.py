from src.d01_data.database.ToSQL.CsvToSQL.CsvToSQL import CsvToSQL


def condition_to_sql(condition_dict):
    move_out_string = CsvToSQL.cat_values(condition_dict['move_type'])
    rot_out_string = CsvToSQL.cat_values(condition_dict['table_rotates'])
    return move_out_string, rot_out_string




