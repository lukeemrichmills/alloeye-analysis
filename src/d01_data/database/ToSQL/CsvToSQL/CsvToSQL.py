import pandas.errors
import psycopg2
import pandas as pd
import numpy as np
from src.d01_data.database.Errors import InvalidMoveCode, InvalidValue, UnmatchingValues
from src.d01_data.database.ToSQL.ToSQL import ToSQL


class CsvToSQL(ToSQL):
    """class with methods to extract SQL commands from csv files"""

    # constructor
    def __init__(self, file, study_id, directory, col_names):
        self.file = file
        self.study_id = study_id
        self.output_commands = ""
        self.df = self.load_file(directory)
        self.col_names = col_names
        self.table_name = ""

    def load_file(self, directory, filename=""):
        """loads file csv into dataframe"""
        if filename == "":
            filename = self.file['filename']
        df = pd.read_csv(directory + "\\" + filename + ".csv")
        df = df.reset_index()
        return df

    def convert_to_insert_commands(self, df=None, col_names="", table_name="", file_data_type=""):
        """iterate through each df row, converting values to sql insert values and appending to one command string"""
        # convert defaults
        data_type = CsvToSQL.default_convert(self.file.data_type, "", file_data_type)
        return super().convert_to_insert_commands(df, col_names, table_name, data_type)

    def get_row_values(self, row, index):
        pass

    def get_ids(self, block=False, trial_no="", viewing_type=""):
        class Output:
            def __init__(self, block_id, trial_id, viewing_id):
                self.block_id = block_id
                self.trial_id = trial_id
                self.viewing_id = viewing_id

        r_or_p = self.get_r_p()
        block_id = f"{self.get_pid()}{r_or_p}{self.file['block']}" if block is True else ""
        trial_id = f"{block_id}_{trial_no}" if trial_no != "" else ""
        viewing_id = f"{trial_id}_{viewing_type}" if viewing_type != "" else ""

        return Output(block_id, trial_id, viewing_id)

    def get_pid(self):
        pid = self.file['pID']
        return f'{self.study_id}_{pid}'

    def get_r_p(self):
        return 'p' if self.file['practice'] else 'r'

    def reconstruct_file_string(self, data_type=""):
        if data_type == "":
            data_type = self.data_type
        return f"{self.file.pID}{self.get_r_p()}{self.file.block}{data_type}"



