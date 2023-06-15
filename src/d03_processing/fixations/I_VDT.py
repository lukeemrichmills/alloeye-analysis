import numpy as np
import pandas as pd

from src.d01_data.database.Errors import InvalidValue
from src.d01_data.database.Tables import Tables
from src.d03_processing.TimepointProcessor import TimepointProcessor
from src.d03_processing.fixations.FixationProcessor import FixationProcessor
from src.d03_processing.fixations.GazeCollision import GazeCollision


class I_VDT(FixationProcessor):
    """
    implementation of a modified velocity and dispersion threshold identification algorithm. Adapted to use vectorized
    functions instead of loops - also inspired by Tobii algorithm?? need to find this - difference being that the time-=
    based threshold is based on merging short saccades with fixations. Might need to also include merging fixations
    together if under dispersion threshold
    """

    def __init__(self, timepoints, angular_v_threshold=20.0, max_angle=0.5,
                 max_time_between_fix=75):
        super().__init__(timepoints)
        self.method_name = 'I_VDT'
        self.angular_v_threshold = angular_v_threshold    # degrees per second
        self.max_angle_between_fixations = max_angle   # degrees
        self.max_time_between_fixations = max_time_between_fix    # ms
        self.timestamp_units_per_second = 1000  # i.e. ms
        self.fix_df, self.timepoints = self.get_fixations_missing_split(self.timepoints)

    def get_fixations(self, timepoints=None, missing_split_group_id=0):
        super().get_fixations(timepoints)
        tps = self.timepoints
        if self.skip_algo:
            return None

        # calculate velocity
        t = timepoints.eye_timestamp_ms.to_numpy()
        # velocity (angular) - also in a function in FixationProcessor but need vectors and time_diff
        point_matrix = TimepointProcessor.create_gaze_point_matrix(tps)
        head_loc_mat = TimepointProcessor.create_head_loc_matrix(tps)
        vectors = point_matrix - head_loc_mat
        angles = []
        for i in range(1, len(vectors)):
            angles.append(FixationProcessor.angle_between(vectors[i - 1], vectors[i]))
        angles = np.array(angles)
        time_diff = np.diff(t)
        # velocity (angular)
        v_ang = FixationProcessor.angular_velocity_vec(tps)

        # velocity threshold
        ang_threshold = (self.angular_v_threshold / self.timestamp_units_per_second)
        fixation = v_ang < ang_threshold
        fixation = np.array(np.hstack([fixation, np.zeros([1, ])]), dtype=int)

        # saccade time threshold
        saccade = np.where(fixation == 1, 0, 1)
        saccade_over_thresh = FixationProcessor.threshold_fixboolarray_duration(saccade, self.max_time_between_fixations, t)
        saccade_under_thresh = (saccade == 1) & (saccade_over_thresh == 0)
        sacc_thresh_start, sacc_thresh_end = TimepointProcessor.get_start_end(saccade_under_thresh)


        # saccade dispersion threshold
        where_fix_end = np.where(sacc_thresh_start == 1)[0] - 1
        where_fix_end = np.where(where_fix_end < 0, 0, where_fix_end)
        where_fix_start = np.where(sacc_thresh_end == 1)[0] + 1
        where_fix_start = np.where(where_fix_start >= len(saccade), len(saccade) - 1, where_fix_start)
        vectors_fix_end = vectors[where_fix_end]
        vectors_fix_start = vectors[where_fix_start]
        time_fix_end = t[where_fix_end]
        time_fix_start = t[where_fix_start]
        time_diff_disp = []
        angles_disp = []
        for i in range(len(vectors_fix_end)):
            angles_disp.append(FixationProcessor.angle_between(vectors_fix_end[i], vectors_fix_start[i]))
            time_diff_disp.append(time_fix_start[i] - time_fix_end[i])
        angles_disp = np.array(angles_disp)
        time_diff_disp = np.array(time_diff_disp)
        under_thresh = angles_disp < self.max_angle_between_fixations
        where_under_thresh_start = where_fix_end[under_thresh]
        where_under_thresh_end = where_fix_start[under_thresh]
        for i in range(len(where_under_thresh_start)):
            fixation[where_under_thresh_start[i]:where_under_thresh_end[i] + 1] = 1

        # time threshold
        fixation = FixationProcessor.threshold_fixboolarray_duration(fixation, self.fixation_threshold, t)
        fix_start, fix_end = TimepointProcessor.get_start_end(fixation)

        # add to original tp df
        timepoints['fixation'] = np.array(fixation, dtype=int)
        timepoints['fixation_start'] = np.array(fix_start, dtype=int)
        timepoints['fixation_end'] = np.array(fix_end, dtype=int)

        return self.convert_fix_df_format(timepoints, missing_split_group_id), timepoints





