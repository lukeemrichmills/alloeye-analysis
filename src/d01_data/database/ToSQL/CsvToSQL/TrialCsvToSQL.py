import pandas.errors
import psycopg2
import pandas as pd
import numpy as np
from src.d01_data.database.Errors import InvalidMoveCode, InvalidValue, UnmatchingValues
from src.d01_data.database.ToSQL.CsvToSQL.CsvToSQL import CsvToSQL


class TrialCsvToSQL(CsvToSQL):

    def __init__(self, file, study_id, directory, col_names):
        super().__init__(file, study_id, directory, col_names)
        self.table_name = "alloeye_trial"
        self.block_id = ""
        self.first_eight_skipped = False
        self.trial_configs = self.load_file(directory, self.reconstruct_file_string('TrialConfigurations'))

    def viewing_trial_block_commands(self, block_column_names, viewing_col_names):
        self.add_missing_bc_trial()
        n_trials, n_bc_trials = self.get_n_trials()
        if n_trials < len(self.trial_configs):
            self.add_missing_trials()

        trial_commands = super(CsvToSQL, self).convert_to_insert_commands()
        block_commands = self.block_to_sql(block_column_names, n_trials)
        viewing_commands = self.viewing_to_sql(viewing_col_names)

        return trial_commands, block_commands, viewing_commands

    def add_missing_bc_trial(self):
        """need to add final baseline calibration trial in"""
        if self.file.practice is True:
            # add new -1 trial and insert into dataframe at correct location
            trial_row = self.df.iloc[0, :]
            pd.options.mode.chained_assignment = None  # default='warn'
            trial_row['TrialNumber'] = -1
            col_index = self.df.columns.get_loc('TrialType')
            trial_row[col_index:] = np.nan
            trial_row['index'] = 8
            self.df.loc[8:, 'index'] += 1
            self.df = self.df.append(trial_row)
            self.df = self.df.sort_values(by='index')
            self.df = self.df.reset_index(drop=True)
            pd.options.mode.chained_assignment = 'warn'  # default='warn'
            pass
        else:
            pass

    def add_missing_trials(self):
        bool_array = np.invert(np.isin(self.trial_configs['TrialNumber'], self.df['TrialNumber']))
        try:
            missing_trials = self.trial_configs.loc[bool_array, :]
        except:
            print("debug")
        self.df = self.df.append(missing_trials)
        pass

    def block_to_sql(self, block_column_names, n_trials):
        list_of_row_values = [self.block_id,  # block id
                              int(self.file['block']),  # block order
                              n_trials,  # n trials
                              self.get_pid(),  # ppt ID
                              self.study_id,  # study_id
                              self.file['practice']]  # practice bool

        return super(CsvToSQL, self).insert_command_output(list_of_row_values, block_column_names, 'block')

    def viewing_to_sql(self, viewing_col_names):
        commands = []
        first_eight_skipped = False
        for index, row in self.df.iterrows():
            if row['TrialNumber'] == -8 and first_eight_skipped is False:
                first_eight_skipped = True
            else:
                row_values_1, row_values_2 = self.viewing_row_values(row, index)
                commands.append(self.insert_command_output(row_values_1, viewing_col_names, "alloeye_viewing"))
                if row_values_2 != []:
                    commands.append(self.insert_command_output(row_values_2, viewing_col_names, "alloeye_viewing"))

        return commands

    def viewing_row_values(self, row, index):
        # get identifier column values
        trial_no = row['TrialNumber']
        is_baseline_calibration = trial_no < 0

        def add_ids_to_list(ids):
            return [ids.viewing_id, self.get_pid(), self.study_id, ids.block_id,
                    ids.trial_id]

        if is_baseline_calibration:
            # get some values
            bc_code = 'bc'
            bc_ids = self.get_ids(True, trial_no, bc_code)  # get ids

            self.block_id = bc_ids.block_id
            obj_loc_pre = self.get_obj_location_cols(row, True)  # get object locations

            # construct list
            bc_row_values = add_ids_to_list(bc_ids)  # add ids
            bc_row_values.append(bc_code)  # add baseline calibration identifier
            bc_row_values.extend([np.nan for _ in range(4)])  # nans for most columns
            bc_row_values.extend([self.row_return(row, obj_loc_pre[i]) for i in range(0, 15)])

            # # pre-bc?
            # prebc_code = 'pre-bc'
            # prebc_ids = self.get_ids(True, trial_no, prebc_code)
            # prebc_row_values = add_ids_to_list(prebc_ids)
            # prebc_row_values.append(prebc_code)
            # prebc_row_values.extend([np.nan for _ in range(19)])

            return bc_row_values, []
        else:
            # get some values
            enc_code = 'enc'
            ret_code = 'ret'
            ids_enc = self.get_ids(True, trial_no, enc_code)
            ids_ret = self.get_ids(True, trial_no, ret_code)
            self.block_id = ids_enc.block_id
            obj_loc_pre = self.get_obj_location_cols(row, True)
            obj_loc_post = self.get_obj_location_cols(row, False)

            # construct encoding list
            enc_row_values = add_ids_to_list(ids_enc)
            enc_row_values.append(enc_code)
            enc_row_values.extend([row['preShift_X'], row['preShift_Z'],
                                   row['preShiftNoRot_X'], row['preShiftNoRot_Z']])
            try:
                enc_row_values.extend([self.row_return(row, obj_loc_pre[i]) for i in range(0, 15)])  # some will be nans
            except:
                print("debug")

            # construct retrieval list
            ret_row_values = add_ids_to_list(ids_ret)
            ret_row_values.append(ret_code)
            ret_row_values.extend([row['postShift_X'], row['postShift_Z'],
                                   row['postShiftNoRot_X'], row['postShiftNoRot_Z']])
            try:
                ret_row_values.extend([self.row_return(row, obj_loc_post[i]) for i in range(0, 15)])  #
            except:
                print("debug")

            return enc_row_values, ret_row_values

    def get_view_type(self, row):
        if row['TrialNumber'] < 0:
            return 'bc'

    def get_n_trials(self):
        trial_count = 0
        baseline_count = 0
        for index, row in self.df.iterrows():
            if row['TrialNumber'] < 0:
                baseline_count += 1
            else:
                trial_count += 1
        return trial_count, baseline_count - 1

    def get_row_values(self, row, index, table_name, file_data_type):
        skip = False    # don't delete

        if row['TrialNumber'] == -8 and self.first_eight_skipped is False:
            skip = True
            self.first_eight_skipped = True

        # get identifier column values
        ids = self.get_ids(True, row['TrialNumber'])
        self.block_id = ids.block_id
        obj_loc_pre = self.get_obj_location_cols(row, True)
        obj_loc_post = self.get_obj_location_cols(row, False)
        move_type = self.get_move_type(row)
        condition_id = self.get_condition_id(row, move_type)


        # list of all values
        trial_row_values = [ids.trial_id, self.study_id, self.get_pid(), ids.block_id, condition_id,
                            self.file['practice'], self.file['block'], row['TrialNumber'],
                            self.get_config_number(row),  # configuration number
                            move_type,
                            row['TableRotates'], row['Anticlockwise'], row['ViewingAngle'], row['ObjectShifted'],
                            row['Q1Answer'], self.q2_parse(row['Q2Answer']),
                            row['preShift_X'], row['preShift_Z'], row['preShiftNoRot_X'], row['preShiftNoRot_Z'],
                            row['postShift_X'], row['postShift_Z'], row['postShiftNoRot_X'], row['postShiftNoRot_Z'],
                            row['objShiftDist'],
                            self.row_return(row, obj_loc_pre[0]),
                            self.row_return(row, obj_loc_pre[1]), self.row_return(row, obj_loc_pre[2]),
                            self.row_return(row, obj_loc_post[1]), self.row_return(row, obj_loc_post[2]),
                            self.row_return(row, obj_loc_pre[3]),
                            self.row_return(row, obj_loc_pre[4]), self.row_return(row, obj_loc_pre[5]),
                            self.row_return(row, obj_loc_post[4]), self.row_return(row, obj_loc_post[5]),
                            self.row_return(row, obj_loc_pre[6]),
                            self.row_return(row, obj_loc_pre[7]), self.row_return(row, obj_loc_pre[8]),
                            self.row_return(row, obj_loc_post[7]), self.row_return(row, obj_loc_post[8]),
                            self.row_return(row, obj_loc_pre[9]),
                            self.row_return(row, obj_loc_pre[10]), self.row_return(row, obj_loc_pre[11]),
                            self.row_return(row, obj_loc_post[10]), self.row_return(row, obj_loc_post[11]),
                            self.row_return(row, obj_loc_pre[12]),
                            self.row_return(row, obj_loc_pre[13]), self.row_return(row, obj_loc_pre[14]),
                            self.row_return(row, obj_loc_post[13]), self.row_return(row, obj_loc_post[14]),
                            row[self.get_table_loc_col(row, 'X')], row[self.get_table_loc_col(row, 'Z')]
                            ]

        return trial_row_values, skip

    def get_condition_id(self, row, move_type):
        row_rotate = row['TableRotates']
        if row_rotate is False or row_rotate == 'FALSE':
            rot_string = 'Stay'
        elif row_rotate is True or row_rotate == 'TRUE':
            rot_string = 'Rotate'
        else:
            rot_string = ''
        if pd.isna(move_type):
            return move_type    # i.e. nan
        else:
            return f"{self.get_pid()}_{move_type}{rot_string}"

    @staticmethod
    def get_move_type(row):

        move_code = CsvToSQL.int_nancheck(row['MoveCode'])

        if move_code == 1:
            output = "Walk"
        elif move_code == 2:
            output = "Teleport"
        elif move_code == 3:
            output = "Stay"
        elif pd.isna(move_code):
            output = np.nan
        else:
            raise InvalidMoveCode

        return output

    def get_config_number(self, row):
        config_no = CsvToSQL.int_nancheck(row[45])
        return config_no if self.is_four_object(row) else 0

    def get_table_loc_col(self, row, X_or_Z):
        return f"arr2_obj4_{X_or_Z}" if self.is_four_object(row) else f"table{X_or_Z}"

    def is_four_object(self, row):
        return pd.isna(row[48]) & pd.isna(row[49])

    def row_return(self, row, col_string):
        if col_string == '':
            return np.nan
        else:
            return row[col_string]

    def q2_parse(self, button_string):
        if button_string == 'TIMED':
            return 0
        elif pd.isna(button_string):
            return button_string
        elif button_string == 'INVALID':
            return -999
        else:
            return int(button_string.split('Button')[0])

    def get_obj_location_cols(self, row, preshift):
        output = []
        for index in range(1, 6):
            output.append(self.get_obj_location_col(row, preshift, index, 'name'))
            output.append(self.get_obj_location_col(row, preshift, index, 'X'))
            output.append(self.get_obj_location_col(row, preshift, index, 'Z'))
        return output

    def get_obj_location_col(self, row, preshift, object_number, name_X_or_Z):
        four_object = self.is_four_object(row)
        arr_number = '1' if preshift else '2'
        if preshift or not four_object:
            output = f"arr{arr_number}_obj{object_number}_{name_X_or_Z}"
        elif object_number == 1:
            output = f"arr1_obj5_{name_X_or_Z}"
        elif object_number < 5:
            output = f"arr2_obj{object_number - 1}_{name_X_or_Z}"
        elif object_number == 5:
            output = ""
        else:
            raise InvalidValue(object_number, "<= 5", f'Value should be <= 5')

        return output
