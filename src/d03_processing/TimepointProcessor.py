import numpy as np
import pandas as pd

from src.d00_utils.TaskObjects import TaskObjects
from src.d01_data.database.Errors import InvalidValue, UnmatchingValues



class TimepointProcessor:
    def __init__(self, timepoints):
        self.none = False
        self.timepoints = self.check_timepoints(timepoints)
        self.timepoints = self.remove_unwanted(self.timepoints)

    def check_timepoints(self, timepoints):
        # make sure only one viewing
        if len(pd.unique(timepoints.viewing_id)) > 1:
            raise InvalidValue(len(pd.unique(timepoints.viewing_id)), 1, "should only be one viewing")
        elif timepoints is None:
            self.none = True
        elif len(timepoints) == 0:
            print(f"no timepoints")
            self.none = True
        else:
            # make sure timestamps are in order
            timepoints = timepoints.sort_values(by=['eye_timestamp_ms']).reset_index(drop=True)

            # remove duplicate timestamps or problems caused later
            n_dups = np.sum(timepoints.duplicated(subset=['eye_timestamp_ms'], keep='first'))
            # if n_dups > 0:
            #     print(f"{n_dups} duplicate timepoints dropped")
            timepoints = timepoints.drop_duplicates(subset=['eye_timestamp_ms'], keep='first')

        return timepoints


    @staticmethod
    def create_gaze_point_matrix(tps):
        gp_mat = TimepointProcessor.create_matrix(tps, ['gaze_collision_x',
                                                        'gaze_collision_y',
                                                        'gaze_collision_z'], axis=1)
        return gp_mat

    @staticmethod
    def create_head_loc_matrix(tps):
        hl_mat = TimepointProcessor.create_matrix(tps, ['camera_x',
                                                        'camera_y',
                                                        'camera_z'], axis=1)
        return hl_mat

    @staticmethod
    def create_matrix(tps, cols, axis=1):
        out = tps[cols[0]].to_numpy().reshape(-1, 1)
        for i in range(1, len(cols)):
            out = np.concatenate([out,  tps[cols[i]].to_numpy().reshape(-1, 1)], axis=axis)

        return out

    @staticmethod
    def get_start_end(bools):
        start = np.zeros(bools.shape, dtype=int)
        end = np.zeros(bools.shape, dtype=int)
        for i in range(1, len(bools)):
            this_tp = bools[i]
            prev_tp = bools[i - 1]
            if this_tp == 1 and prev_tp == 0:
                start[i] = 1
            elif this_tp == 0 and prev_tp == 1:
                end[i - 1] = 1

            # first and last
            if prev_tp == 1 and i == 1:  # start at index 0 if index 0 is 1
                start[0] = 1

            if this_tp == 1 and i == (len(bools) - 1):  # end at last index if last index is 1
                end[-1] = 1

        if int(np.sum(end)) != int(np.sum(start)):
            raise UnmatchingValues(message=f"start ({np.sum(start)}) and end ({np.sum(end)}) should be same length")
        return start, end


    def remove_unwanted(self, timepoints):
        if self.none is True:
            return timepoints

        # adjust negative time error in timestamp
        t0 = timepoints.eye_timestamp_ms[0]
        if t0 < 0:
            timepoints.eye_timestamp_ms = timepoints.eye_timestamp_ms + np.absolute(t0) + 1

        # remove dome if starts within the first second (lerp time)
        early_dome = (timepoints.gaze_object == TaskObjects.dome) & ((timepoints.eye_timestamp_ms - t0) < 1000)
        timepoints = timepoints[np.invert(np.array(early_dome))]

        # remove anything on canvas
        not_canvas = (timepoints.gaze_object != 'FirstInstructionsCanvas') \
                     & (timepoints.gaze_object != 'SecondInstructionsCanvas')
        timepoints = timepoints[not_canvas]

        return timepoints.sort_values(by=['eye_timestamp_ms']).reset_index(drop=True)


