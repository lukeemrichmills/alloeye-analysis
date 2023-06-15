import numpy as np
import pandas as pd

from src.d00_utils.generic_tools import del_multiple, shift_list_down, shift_list_up
from src.d01_data.database.Errors import UnmatchingValues
from src.d01_data.database.ToSQL.ToSQL import ToSQL
from src.d03_processing.TimepointProcessor import TimepointProcessor
from src.d03_processing.fixations.FixationProcessor import FixationProcessor


class GazeCollision(FixationProcessor):
    """
        Class for implementing fixation algorithm and related methods based on in-built gaze collision recognition
        from vive eye pro. Algorithm loops through sorted feature_extract, registering end of fixation when object changes
        with some extra adjustments.
        """

    def __init__(self, timepoints, angular_v_threshold=60.0):
        super().__init__(timepoints)
        self.method_name = 'GazeCollision'
        self.max_gap_length = 75  # ms for blinks Komogortsev et al Standardization of Automated Analyses of Oculomotor Fixation and Saccadic Behaviors
        self.fix_removal_threshold = 40
        self.angular_v_threshold = angular_v_threshold   # degree(s)/s
        self.timestamp_units_per_second = 1000     # i.e. ms
        self.fix_df, self.timepoints = self.get_fixations_missing_split(self.timepoints)

    def get_fixations(self, timepoints=None, missing_split_group_id=0):
        super().get_fixations(timepoints)
        tps = self.timepoints

        if self.skip_algo:
            return None

        # adjust negative time error in timestamp
        first_time = tps.eye_timestamp_ms[0]
        if first_time < 0:
            tps.eye_timestamp_ms = tps.eye_timestamp_ms + np.absolute(first_time) + 1

        t = tps.eye_timestamp_ms.to_numpy()

        # velocity (angular)
        point_matrix = np.concatenate([tps.gaze_collision_x.to_numpy().reshape(len(tps), 1),
                                       tps.gaze_collision_z.to_numpy().reshape(len(tps), 1),
                                       tps.gaze_collision_y.to_numpy().reshape(len(tps), 1)], axis=1)
        head_loc_mat = np.concatenate([tps.camera_x.to_numpy().reshape(len(tps), 1),
                                       tps.camera_z.to_numpy().reshape(len(tps), 1),
                                       tps.camera_y.to_numpy().reshape(len(tps), 1)], axis=1)
        vectors = point_matrix - head_loc_mat
        angles = []
        for i in range(1, len(vectors)):
            angles.append(FixationProcessor.angle_between(vectors[i - 1], vectors[i]))
        angles = np.array(angles)
        time_diff = np.diff(t)
        v_ang = angles / time_diff
        # v = FixationProcessor.velocity_vector(tps)

        # prev object array
        gaze_object = tps.gaze_object.to_list()
        prev_object = np.array(shift_list_down(gaze_object, 'None'))
        next_object = np.array(shift_list_up(gaze_object, 'None'))
        gaze_object = np.array(gaze_object)

        # start and end defined by object change
        fix_end = (gaze_object != next_object)
        # fix_end[-1] = 0
        fix_start = gaze_object != prev_object
        # fix_start[0] = 0

        # fixation array - 1s for fixations
        fixation = np.ones(len(gaze_object), dtype=bool)    # start with everything a fixation and adjust/threshold down

        # velocity threshold
        ang_thresholds = (self.angular_v_threshold / self.timestamp_units_per_second)
        over_threshold = v_ang > ang_thresholds
        fixation[np.insert(over_threshold, 0, 0)] = 0
        fix_start, fix_end = GazeCollision.fix_start_end_adjust(fixation, fix_start, fix_end)

        # time threshold
        fix_start, fix_end = GazeCollision.threshold_fixstartend_duration(fixation, fix_start, fix_end, self.fixation_threshold, t)

        # get duration and object arrays for adjustment
        start_time = t[fix_start == 1]
        end_time = t[fix_end == 1]
        fix_duration = list(end_time - start_time)
        fix_objects = tps.gaze_object[fix_start == 1].to_list()
        prev_fix_objects = np.array(shift_list_down(fix_objects, 'None'))
        next_fix_objects = np.array(shift_list_up(fix_objects, 'None'))
        prev_fix_duration = np.array(shift_list_down(fix_duration, np.nan))
        next_fix_duration = np.array(shift_list_up(fix_duration, np.nan))
        fix_objects = np.array(fix_objects)
        fix_duration = np.array(fix_duration)

        # adjustment - Table sandwiched
        is_sandwiched_table_fix = (prev_fix_objects == next_fix_objects) & \
                                 (fix_objects == "Table") & \
                                 (prev_fix_duration > 0) & \
                                 (next_fix_duration > 0) & \
                                 (prev_fix_objects != "Table") & \
                                 (fix_duration < self.fix_removal_threshold)
        sandwiched_table_fix = start_time[is_sandwiched_table_fix]
        sandwiched_table_fix_end = end_time[is_sandwiched_table_fix]

        if len(sandwiched_table_fix) > 0:
            print("test")

        for i in range(len(sandwiched_table_fix)):
            remove_fix_start = sandwiched_table_fix[i]
            remove_fix_end = sandwiched_table_fix_end[i]
            fix_start[tps.eye_timestamp_ms.tonumpy() == remove_fix_start] = 0
            fix_end[tps.eye_timestamp_ms.tonumpy() == remove_fix_end] = 0

        # time threshold
        fixation = FixationProcessor.threshold_fixboolarray_duration(fixation, self.fixation_threshold, t)

        # get new start and end
        fix_start, fix_end = TimepointProcessor.get_start_end(fixation)

        # add to original tp df
        tps['fixation'] = fixation
        tps['fixation_start'] = fix_start
        tps['fixation_end'] = fix_end
        # self.timepoints = tps   # save tps with labels

        return self.convert_fix_df_format(tps, missing_split_group_id), tps

    @staticmethod
    def threshold_fixstartend_duration(fixation, fix_start, fix_end, fix_threshold, t):
        fix_start_ind = np.where(fix_start)[0]
        fix_end_ind = np.where(fix_end)[0]
        dur = t[fix_end_ind] - t[fix_start_ind]
        for i in range(len(dur)):
            if dur[i] < fix_threshold:
                fixation[fix_start_ind[i]:fix_end_ind[i] + 1] = 0
        return GazeCollision.fix_start_end_adjust(fixation, fix_start, fix_end)


    @staticmethod
    def fix_start_end_adjust(fixation, fix_start, fix_end):
        try:
            new_fix_start, new_fix_end = FixationProcessor.get_start_end(fixation)
        except Exception as e:
            raise e

        fix_start = (fix_start == 1) & (fixation == 1)
        fix_end = (fix_end == 1) & (fixation == 1)
        fix_start = fixation & (fix_start | new_fix_start)
        fix_end = fixation & (fix_end | new_fix_end)
        return fix_start, fix_end