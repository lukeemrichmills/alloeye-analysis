import pandas.errors
import pandas as pd
import numpy as np

from src.d01_data.database.ToSQL.CsvToSQL.CsvToSQL import CsvToSQL
from src.d01_data.database.Errors import InvalidValue


class TimepointCsvToSQL(CsvToSQL):

    def __init__(self, file, study_id, directory, col_names):
        super().__init__(file, study_id, directory, col_names)
        self.all_df = self.load_file(directory, self.reconstruct_file_string('AllGazeData'))
        self.table_name = self.get_table_name()
        self.event_log = self.load_event_log(directory)
        self.bc_start, self.bc_end = self.get_bc_timings()  # bc = baseline calibration
        self.df = self.remove_pre_bc()

    def convert_to_insert_commands(self):
        output = []
        trial_commands = super(CsvToSQL, self).convert_to_insert_commands(file_data_type='TrialGazeData')
        output.extend(trial_commands)

        new_df = self.get_trial_from_all()
        if len(new_df) > 0:
            trial_from_all = super(CsvToSQL, self).convert_to_insert_commands(new_df, "", "", 'AllGazeData')
            output.extend(trial_from_all)
            print(f"trial from all: {len(trial_from_all)} from ppt {self.file.pID}")

        all_commands = super(CsvToSQL, self).convert_to_insert_commands(self.all_df, "",
                                                                        self.get_table_name('AllGazeData'),
                                                                        'AllGazeData')
        output.extend(all_commands)
        return output

    def get_trial_from_all(self):
        """find retrieval values from allgaze df"""
        # loop through allgaze df
        # if specific rows are within certain frames as determined by event log
        # remove any rows already in trial df
        # add new rows to new dataframe (or trial dataframe and run sql commands after this method?)
        # make sure these rows follow immediately from trial rows
        # return df
        retrieval_frames = self.get_retrieval_frames()
        retrieval_rows = self.all_df[np.isin(self.all_df['FrameCount'], retrieval_frames)]
        if len(retrieval_rows) < 1:
            return retrieval_rows
        else:
            trial_frames = self.df.loc[:, 'FrameCount']
            output = retrieval_rows[np.invert(np.isin(retrieval_rows['FrameCount'], trial_frames))]
            pd.options.mode.chained_assignment = None  # default='warn'
            output['ViewNo'] = 3
            pd.options.mode.chained_assignment = 'warn'  # default='warn'
            return output

    def get_retrieval_frames(self):
        retrieval_frames = []
        df = self.event_log
        start_string = 'Dome up start'      # beginning of retrieval
        end_string = 'q1 answered'      # end of retrieval
        last_index = 0
        loop_end_index = len(df) - 1
        start_indices = df[df['event'] == start_string].index
        while last_index < loop_end_index:

            if loop_end_index == len(df) - 1:
                loop_end_index = start_indices[-1]

            start_index = start_indices[start_indices > last_index][0]

            if start_index < len(df) - 1:   # cannot be last row, needs to be a next row
                try:
                    next_string = df.loc[start_index + 1, 'event']
                except:
                    print("debug")
                if next_string == end_string:
                    start_frame, end_frame = \
                        self.get_two_frames_event_log(start_string, end_string, df.iloc[start_index:start_index + 2, :])
                    retrieval_frames.extend([i for i in range(int(start_frame), int(end_frame))])

            last_index = start_index + 1
        return retrieval_frames

    def get_table_name(self, data_type=""):
        trial_or_all = self.file.data_type if data_type == "" else data_type
        if trial_or_all == 'TrialGazeData':
            return 'alloeye_timepoint_viewing'
        elif trial_or_all == 'AllGazeData':
            return 'alloeye_timepoint_all'
        else:
            raise InvalidValue

    def load_file(self, directory, filename=""):
        """loads file csv into dataframe"""
        # print(f'debug {self.file.filename} TimepointCsvToSQL load_file')
        if filename == "":
            filename = self.file['filename']
        filestring = directory + "\\" + filename + ".csv"
        headers = TimepointCsvToSQL.timepoint_headers()
        try:
            df = pd.read_csv(filestring, header=None, skiprows=1)
            df.columns = headers.split(',')
        except pandas.errors.EmptyDataError:
            df = pd.DataFrame(columns=headers.split(','))


        df = df.reset_index()
        return df

    def load_event_log(self, directory):
        file_string = self.reconstruct_file_string("EventLog")
        filename = f"{directory}{file_string}.csv"
        try:
            df = pd.read_csv(filename, header=None, skiprows=1)
            df = df.iloc[:, :3]
            df.columns = "Unity time,frame,event".split(',')
        except:
            print("event log load failed")
        df = df.reset_index()

        return df

    def get_bc_timings(self):
        """returns unity frame start and end for baseline calibration phase"""
        if self.file.practice is True:
            df = self.event_log
            bc_start_string = "Baseline Calibration button pressed (or skipped)"
            bc_end_string = "dome down training trial canvas showing"

            return self.get_two_frames_event_log(bc_start_string, bc_end_string)
        else:
            return 0, 0

    def get_two_frames_event_log(self, first, second, df=None):
        if df is None:
            df = self.event_log
        # df = df.reset_index()
        start_indices = df[df['event'] == first].index
        if len(start_indices) == 0:
            start_frame, end_frame = 0, 0
        else:
            start_index = start_indices[0]  #
            start_frame = df.loc[start_index, 'frame']  #
            ind_range = [i for i in range(start_index, start_index + len(df))]
            df = df[np.isin(df['index'], ind_range)]
            try:
                end_index = df[df['event'] == second].index[0]
            except IndexError:
                print("index error")
            end_frame = df.loc[end_index, 'frame']
        return start_frame, end_frame

    def remove_pre_bc(self):
        if len(self.df) == 0 or self.file.practice is False:
            return self.df
        else:
            df = self.df
            bc_start_indices = df[df.iloc[:, 3] == self.bc_start].index

            if len(bc_start_indices) == 0:
                bc_start_index = 0
            else:
                bc_start_index = bc_start_indices[0]

            return self.df.iloc[bc_start_index:, :]

    def get_row_values(self, row, index, table_name, file_data_type):
        # get identifier column values
        skip = False

        try:
            data_type_code = file_data_type[0]
        except IndexError as e:
            raise e
        timepoint_id = self.file['filename'] + "_" + data_type_code + str(index)

        view_identifier = self.get_view_type(row, table_name)
        if view_identifier == "ret_view":
            retrieval_epoch = 'view'
            view_identifier = 'ret'
        elif view_identifier == "ret_full":
            retrieval_epoch = 'full'
            view_identifier = 'ret'
        elif view_identifier == "pre-bc":
            skip = True
        else:
            retrieval_epoch = 'na'

        if view_identifier == 'bc':
            trial_no = row['Trial'] - 8
        else:
            trial_no = row['Trial']

        ids = self.get_ids(True, trial_no, view_identifier)

        # list of all values
        try:
            list_of_row_values = [timepoint_id, file_data_type, self.get_pid(), self.study_id,
                                  ids.block_id, ids.trial_id, ids.viewing_id,
                                  retrieval_epoch,
                                  row[1],  # Eyetimestamp
                                  row[2],  # UnityTime
                                  row[3],  # EyeFrameSequence
                                  row[4],  # FrameCount
                                  row[5],  # FPS
                                  row[6],  # Object
                                  row[7], row[8], row[9],  # PosX-Z
                                  row[13], row[14], row[15],  # ColX-Z
                                  row[29], row[30],  # pupil diameter left and right
                                  row[31], row[32],  # left and right eye openness
                                  row[23], row[24], row[25],  # camera positions x-z
                                  row[26], row[27], row[28],  # camera euler rotation x-z
                                  row[33], row[34], row[35],  # right controller location x-z
                                  row[39], row[40], row[41],  # right controller rotation x-z
                                  row[36], row[37], row[38],  # left controller location x-z
                                  row[42], row[43], row[44],  # left controller rotation x-z
                                  row[45], row[46], row[47],  # l gaze origin
                                  row[48], row[49], row[50],  # r gaze origin
                                  row[51], row[52], row[53],  # l gaze direction
                                  row[54], row[55], row[56],  # r gaze direction
                                  row[59],  # gaze object no table
                                  row[66], row[67], row[68]  # gaze collision xyz no table
                                  ]
        except IndexError:
            print("Index Error raised")
            raise IndexError
        return list_of_row_values, skip

    def get_view_type(self, row, table_name):

        if table_name == 'alloeye_timepoint_viewing' and self.file.practice and \
                self.bc_start < row['FrameCount'] < self.bc_end:
            return 'bc'
        elif table_name == 'alloeye_timepoint_viewing' and self.file.practice and \
                row['FrameCount'] < self.bc_start:
            return 'pre-bc'
        elif row['ViewNo'] == 0:
            if table_name == 'alloeye_timepoint_viewing':
                raise InvalidValue(value='Trial', expected='All', message="cannot be na here")
            else:
                return 'na'
        elif row['ViewNo'] == 1:
            return 'enc'
        elif row['ViewNo'] == 2:
            return 'ret_view'
        elif row['ViewNo'] == 3:
            return 'ret_full'
        else:
            raise InvalidValue

    @staticmethod
    def timepoint_headers():
        headers = 'EyeTimestamp,UnityTime,EyeFrameSequence,FrameCount,FPS,Object,PosX,PosY,PosZ,ScaleX,' \
                  'ScaleY,ScaleZ,ColX,ColY,ColZ,Trial,ViewNo,ViewPos,ViewingAngle,Rot?,ObjShifted,MoveCode,' \
                  'cameraX,cameraY,cameraZ,camRotX,camRotY,camRotZ,leftPupilDiameter,rightPupilDiameter,' \
                  'leftEyeOpenness,rightEyeOpenness,RcontrollerX,RcontrollerY,RcontrollerZ,LcontrollerX,' \
                  'LcontrollerY,LcontrollerZ,RcontrRotX,RcontrRotY,RcontrRotZ,LcontrRotX,LcontrRotY,LcontrRotZ,' \
                  'lGazeOrigin_x,lGazeOrigin_y,lGazeOrigin_z,rGazeOrigin_x,rGazeOrigin_y,rGazeOrigin_z,' \
                  'lGazeDirection_x,lGazeDirection_y,lGazeDirection_z,rGazeDirection_x,rGazeDirection_y,' \
                  'rGazeDirection_z,gazeConvergenceDist,gazeConvDistValidity,Object_notable,posX_notable,' \
                  'posY_notable,posZ_notable,scaleX_notable,scaleY_notable,scaleZ_notable,colX_notable,' \
                  'colY_notable,colZ_notable'
        return headers
