import numpy as np
import pandas as pd
from vr_idt.vr_idt import classify_fixations

from src.d01_data.database.Errors import InvalidValue
from src.d01_data.database.Tables import Tables
from src.d03_processing.fixations.FixationProcessor import FixationProcessor


class VR_IDT(FixationProcessor):
    """
    class for implementing I-DT fixation algorithm adapted for VR eye tracking as developed by Jose Llanes-Jurado et al.
    here https://pypi.org/project/vr-idt/ - wrapped in FixationProcessor class
    """

    def __init__(self, timepoints, max_angle=1.0, min_freq=0.03):
        super().__init__(timepoints)
        self.method_name = 'VR_IDT'
        self.max_gap_length = 75  # ms for blinks Komogortsev et al Standardization of Automated Analyses of Oculomotor Fixation and Saccadic Behaviors
        self.min_freq = min_freq # 30 ms converted to sec
        self.max_angle = max_angle  # dispersion threshold in degrees (angular shift)
        self.fix_df, self.timepoints = self.get_fixations_missing_split(self.timepoints)

    def get_fixations(self, timepoints=None, missing_split_group_id=0):
        super().get_fixations(timepoints)
        tps = self.timepoints
        if self.skip_algo:
            return None

        col_map = {
            "time": "eye_timestamp_ms",
            "gaze_world_x": "gaze_collision_x",
            "gaze_world_y": "gaze_collision_y",
            "gaze_world_z": "gaze_collision_z",
            "head_pos_x": "camera_x",
            "head_pos_y": "camera_y",
            "head_pos_z": "camera_z"
            }

        fix_tps = classify_fixations(tps, self.fixation_threshold, self.max_angle, self.min_freq, **col_map)
        # self.timepoints = fix_tps
        return self.convert_fix_df_format(fix_tps, missing_split_group_id), fix_tps





