import re

import pandas as pd
import numpy as np

from src.d00_utils.type_checks import type_or_list
from src.d01_data.database.Errors import InvalidMoveCode, InvalidValue, UnmatchingValues


class ToSQL:
    """class with methods to convert into SQL commands"""

    # constructor
    def __init__(self, data_df, study_id, col_names):
        self.study_id = study_id
        self.output_commands = ""
        self.df = data_df
        self.col_names = col_names
        self.table_name = ""

    def convert_to_insert_commands(self, df=None, col_names="", table_name="", file_data_type=""):
        """iterate through each df row, converting values to sql insert values and appending to one command string"""
        # convert defaults
        df = ToSQL.default_convert(self.df, None, df)
        col_names = ToSQL.default_convert(self.col_names, "", col_names)
        table_name = ToSQL.default_convert(self.table_name, "", table_name)

        commands = []

        for index, row in df.iterrows():
            list_of_row_values, skip = self.get_row_values(row, index, table_name, file_data_type)
            if skip is False:
                commands.append(self.insert_command_output(list_of_row_values, col_names, table_name))

        return commands

    def convert_to_update_commands(self, condition_col, df=None, col_names="", table_name=""):
        # convert defaults
        df = ToSQL.default_convert(self.df, None, df)
        col_names = ToSQL.default_convert(self.col_names, "", col_names)
        table_name = ToSQL.default_convert(self.table_name, "", table_name)
        keep_cols = [condition_col]
        keep_cols.extend(col_names)
        keep_df = df[keep_cols]
        cond_index = keep_df.columns.to_list().index(condition_col)

        # list comprehension for each row in df
        # DUPLICATING COLUMN UPDATE!!
        commands = [self.update_command_output(row, keep_cols, table_name, condition_col, row[cond_index])
                    for row in zip(*[keep_df[col] for col in keep_df])]

        return commands

    def get_row_values(self, row, index, table_name, file_data_type):
       pass

    @staticmethod
    def update_command_output(row_values, col_names, table_name, condition_col, condition_ids):
        commands = f"""UPDATE {table_name} """
        if col_names[0] != condition_col:
            raise InvalidValue(col_names[0], condition_col, message="first col must be condition col")
        commands += f"""SET {col_names[1]} = {ToSQL.format_value(row_values[1])}"""
        if len(col_names) > 2:
            for i in range(2, len(col_names)-1):
                commands += f""", {col_names[i]} = {ToSQL.format_value(row_values[i])}"""
            commands += f""", {col_names[-1]} = {ToSQL.format_value(row_values[-1])} """
        commands += f""" WHERE {condition_col} in {ToSQL.cat_values(type_or_list(str, condition_ids))}"""
        return commands

    @staticmethod
    def insert_command_output(list_of_row_values, col_names, table_name):
        """formats row values into one insert query"""

        if type(col_names) is str:
            col_insert = col_names
            row_values = list_of_row_values
        elif type(col_names) is list:
            if len(col_names) != len(list_of_row_values):
                raise UnmatchingValues
            else:
                col_insert = ToSQL.cat_values(col_names, True)
                row_values = ToSQL.cat_values(list_of_row_values)
        else:
            print("debug")

        output_commands = f'INSERT INTO {table_name} {col_insert} ' \
                          f'VALUES {row_values} '

        return output_commands

    @staticmethod
    def upsert_command_output(list_of_row_values, col_names, table_name, condition_col, condition_ids):
        out = f"{ToSQL.insert_command_output(list_of_row_values, col_names, table_name)} " \
              f"ON CONFLICT ({condition_col}) DO " \
              f"{ToSQL.update_command_output(list_of_row_values, col_names, table_name, condition_col, condition_ids)}"
        return

    @staticmethod
    def cat_values(list_of_values, is_col_name=False):
        """takes list of values and converts into comma separated format with
        braces before and after"""
        if len(list_of_values) > 1:
            values_string = f"({ToSQL.format_value(list_of_values[0], is_col_name)}, "
            iterate_list = list_of_values[1:-1]
            for value in iterate_list:
                values_string = values_string + f"{ToSQL.format_value(value, is_col_name)}, "

            values_string = values_string + f"{ToSQL.format_value(list_of_values[-1], is_col_name)})"
        elif len(list_of_values) > 0:
            values_string = f"({ToSQL.format_value(list_of_values[0], is_col_name)})"
        else:
            raise InvalidValue(len(list_of_values), 1, message="no values")
        return values_string

    @staticmethod
    def format_value(value, is_col_name=False):
        if is_col_name is True:
            output = f'{value}'
        elif isinstance(value, bool):
            val_string = str(value).upper()
            output = val_string
        elif pd.isna(value):
            output = 'NULL'
        else:
            output = f'{value!r}'
        return output


    @staticmethod
    def default_convert(default, default_input, value):
        if default_input is None:
            if value is default_input:
                return default
            else:
                return value
        else:
            if value == default_input:
                return default
            else:
                return value

    @staticmethod
    def int_nancheck(value):
        return np.nan if pd.isna(value) else int(value)

    @staticmethod
    def add_rep_col(df, rep, col_name):
        col = [rep] * len(df)
        df[col_name] = col
        return df

    @staticmethod
    def get_ids_from_viewing(viewing_id):
        vid_split = viewing_id.split('_')
        study_id = vid_split[0]

        pid_block_split = re.split('(\d+)', vid_split[1])
        ppt_no = pid_block_split[1]  # splits by number
        ppt_id = f'{vid_split[0]}_{ppt_no}'
        block_id = f'{vid_split[0]}_{vid_split[1]}'
        trial_id = f'{block_id}_{vid_split[2]}'

        return study_id, ppt_id, block_id, trial_id

    @staticmethod
    def convert_dtype_return_str(data_type):
        """takes python or numpy datatype and converts to psql dtype"""
        if data_type is float or np.issubdtype(type(data_type), np.floating):
            return 'float8'
        elif data_type is int or np.issubdtype(type(data_type), np.integer):
            return 'int'
        elif data_type is str:
            return 'varchar'
