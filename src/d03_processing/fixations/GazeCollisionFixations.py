import numpy as np
import pandas as pd

from src.d00_utils.generic_tools import del_multiple
from src.d01_data.database.ToSQL.ToSQL import ToSQL
from src.d03_processing.fixations.FixationProcessor import FixationProcessor


class GazeCollisionFixations(FixationProcessor):
    """
    THIS HAS BEEN REPLACED BY GazeCollision.py WHICH IS THE SAME BUT MUCH FASTER AND WITH VELOCITY THRESHOLDING
    Class for implementing fixation algorithm and related methods based on in-built gaze collision recognition
    from vive eye pro. Algorithm loops through sorted feature_extract, registering end of fixation when object changes
    with some extra adjustments.
    """

    def __init__(self, timepoints):
        super().__init__(timepoints)
        self.method_name = 'GazeCollision'
        self.fixation_threshold = 100  # ms
        self.max_gap_length = 75  # ms for blinks Komogortsev et al Standardization of Automated Analyses of Oculomotor Fixation and Saccadic Behaviors
        self.fix_removal_threshold = 40
        self.fix_df, self.excl_fix_df, self.gap_df = self.get_fixations(timepoints)

    def get_fixations(self, timepoints=None):
        tp = self.timepoints if timepoints is None else FixationProcessor.check_timepoints(timepoints)

        # fixation table columns
        viewing_id, fix_start, fix_end, fix_object, fix_duration, fix_start_frame, fix_end_frame, fix_frames, \
        centroid_x, centroid_y, centroid_z, invalid_duration, dispersion, mean_velocity, \
        max_velocity, mean_acceleration = ([] for i in range(16))

        # gap table columns
        gap_start, gap_end, gap_object, gap_duration, gap_start_frame, gap_end_frame, gap_frames \
            = ([] for i in range(7))

        if len(timepoints) < 5:
            print("no feature_extract")
            return None, None, None

        first_time = tp.eye_timestamp_ms[0]
        if first_time < 0:
            tp.eye_timestamp_ms = tp.eye_timestamp_ms + np.absolute(first_time) + 1

        col_x, col_y, col_z = ([] for i in range(3))

        fix_start, fix_start_frame, fix_object, col_x, col_y, col_z = \
            GazeCollisionFixations.start_next_fix(tp.iloc[0, :], fix_start, fix_start_frame,
                                                  fix_object, col_x, col_y, col_z)
        v = []
        a = []
        for tp_i in range(len(tp) - 1):
            row = tp.iloc[tp_i, :]  # current timepoint
            next_row = tp.iloc[tp_i + 1, :]  # next timepoint
            if tp_i > 0:
                last_row = tp.iloc[tp_i - 1, :]
            is_lastline = tp_i + 1 == len(tp) - 1  # if next index is last index

            # velocity and acceleration

            if tp_i == 0:
                v.append(np.nan)
                a.append(np.nan)
            elif tp_i == 1 or len(v) < 2:
                v.append(FixationProcessor.velocity(last_row, row))
                a.append(np.nan)
            else:
                v.append(FixationProcessor.velocity(last_row, row))
                try:
                    v1, v2, t1, t2 = (v[-2], v[-1], last_row.eye_timestamp_ms, row.eye_timestamp_ms)
                except IndexError:
                    print("catch")
                    raise IndexError
                a.append(FixationProcessor.acceleration(v1, v2, t1, t2))

            # determine validity of eye reading timepoint
            current_valid = FixationProcessor.valid_eye_open(row)
            next_valid = FixationProcessor.valid_eye_open(next_row)

            # add gaze collision points to fixation
            col_x.append(row.gaze_collision_x)
            col_y.append(row.gaze_collision_y)
            col_z.append(row.gaze_collision_z)

            # define start and end of gaps
            if current_valid & next_valid:
                pass
            elif current_valid and not next_valid:
                gap_start.append(next_row.eye_timestamp_ms)
                gap_start_frame.append(next_row.eye_frame_number)
                gap_object.append(next_row.gaze_object)
            elif not current_valid and not next_valid:
                pass
            elif not current_valid and next_valid and len(gap_start) > 0:  # avoid the first one
                gap_end, gap_end_frame, gap_duration, gap_start, gap_start_frame, gap_frames = \
                    GazeCollisionFixations.end_duration(next_row, gap_end, gap_end_frame, gap_duration,
                                                        gap_start, gap_start_frame, gap_frames)

            # if end of fixation
            end_of_fixation = row.gaze_object != next_row.gaze_object or is_lastline
            if end_of_fixation:
                # viewing_id
                viewing_id.append(row.viewing_id)

                # fill end and duration variables
                fix_end, fix_end_frame, fix_duration, fix_start, fix_start_frame, fix_frames = \
                    GazeCollisionFixations.end_duration(next_row, fix_end, fix_end_frame, fix_duration,
                                                        fix_start, fix_start_frame, fix_frames)

                # fill disp, vel and acc variables
                dispersion.append(np.nan)
                if len([i for i in v if not pd.isna(i)]) > 1:
                    mean_velocity.append(np.nanmean(v))
                    max_velocity.append(np.nanmax(v))
                else:
                    mean_velocity.append(np.nan)
                    max_velocity.append(np.nan)

                if len([i for i in a if not pd.isna(i)]) > 1:
                    mean_acceleration.append(np.nanmean(a))
                else:
                    mean_acceleration.append(np.nan)

                v = []
                a = []

                # calculate centroid per coordinate
                centroid_x.append(np.mean(col_x))
                centroid_y.append(np.mean(col_y))
                centroid_z.append(np.mean(col_z))

                # find how much invalid
                invalid_duration = GazeCollisionFixations.get_invalid_duration(invalid_duration, gap_start, gap_end,
                                                                               fix_start, fix_end, fix_duration)

                # start next fixation if not the last  one
                if not is_lastline:
                    fix_start, fix_start_frame, fix_object, col_x, col_y, col_z = \
                        GazeCollisionFixations.start_next_fix(next_row, fix_start, fix_start_frame,
                                                              fix_object, col_x, col_y, col_z)
                elif len(gap_start) > len(gap_end):
                    gap_end, gap_end_frame, gap_duration, gap_start, gap_start_frame, gap_frames = \
                        GazeCollisionFixations.end_duration(next_row, gap_end, gap_end_frame, gap_duration,
                                                            gap_start, gap_start_frame, gap_frames)

        # adjustment - if two array object fixations sandwich a table gaze point with 0 duration, then combine them
        end_no = len(fix_start) - 2
        i = 1
        while i < end_no:
            current = i
            previous = i - 1
            next = i + 1

            is_sandwiched_table_fix = fix_object[previous] == fix_object[next] and \
                                      fix_object[current] == "Table" and \
                                      fix_duration[previous] > 0 and fix_duration[next] > 0 and \
                                      fix_object[previous] != "Table" and \
                                      fix_duration[current] < self.fix_removal_threshold

            if is_sandwiched_table_fix:
                var_list = [viewing_id, fix_object, fix_start, fix_start_frame, fix_end, fix_end_frame, fix_duration,
                            fix_frames, centroid_x, centroid_y, centroid_z, invalid_duration, dispersion,
                            mean_velocity, max_velocity, mean_acceleration]
                new = FixationProcessor.combine_fixations(previous, next, list_of_variables=var_list)
                new = del_multiple(current, new)
                viewing_id = new[0]
                fix_object, fix_start, fix_start_frame, fix_end, fix_end_frame = new[1:6]
                fix_duration, fix_frames, centroid_x, centroid_y, centroid_z, = new[6:11]
                invalid_duration = new[11]
                i -= 2  # have removed current and next
                end_no = len(fix_start) - 2
            i += 1

        # combine lists in output dataframes
        fix_df = pd.DataFrame(
            {
                'viewing_id': viewing_id,
                'object': fix_object,
                'start_time': fix_start,
                'start_frame': fix_start_frame,
                'end_time': fix_end,
                'end_frame': fix_end_frame,
                'duration_time': fix_duration,
                'duration_frame': fix_frames,
                'centroid_x': centroid_x,
                'centroid_y': centroid_y,
                'centroid_z': centroid_z,
                'invalid_duration': invalid_duration,
                'dispersion': dispersion,
                'mean_velocity': mean_velocity,
                'max_velocity': max_velocity,
                'mean_acceleration': mean_acceleration
            })
        try:
            gap_df = pd.DataFrame(
                {
                    'start_time': gap_start,
                    'start_frame': gap_start_frame,
                    'end_time': gap_end,
                    'end_frame': gap_end_frame,
                    'duration_time': gap_duration,
                    'duration_frame': gap_frames,
                })
        except:
            print("debug")

        # apply fixation threshold
        excl_fix_df = fix_df.loc[fix_df['duration'] - fix_df['invalid_duration'] < self.fixation_threshold, :]
        excl_fix_df = excl_fix_df.reset_index(drop=True)
        fix_df = fix_df.loc[fix_df['duration'] - fix_df['invalid_duration'] >= self.fixation_threshold, :]
        fix_df = fix_df.reset_index(drop=True)
        if len(fix_df) == 0:
            fix_df = None
        else:
            fix_df = ToSQL.add_rep_col(fix_df, self.method_name, 'algorithm')
            fix_df = ToSQL.add_rep_col(fix_df, 'fixation', 'fixation_or_saccade')

        return fix_df, excl_fix_df, gap_df


    @staticmethod
    def start_next_fix(row, start, start_frame, _object, colx, coly, colz):
        start.append(row.eye_timestamp_ms)  # start of first fix = start of first gaze point
        start_frame.append(row.eye_frame_number)
        _object.append(row.gaze_object)
        colx, coly, colz = ([] for i in range(3))  # temporary collision arrays
        return start, start_frame, _object, colx, coly, colz

    @staticmethod
    def end_duration(next_row, end, end_frame, duration, start, start_frame, frames):
        end.append(next_row.eye_timestamp_ms)
        end_frame.append(next_row.eye_frame_number)
        try:
            duration.append(end[-1] - start[-1])
        except:
            print("debug")
        frames.append(end_frame[-1] - start_frame[-1])
        return end, end_frame, duration, start, start_frame, frames

    @staticmethod
    def add_invalid_dur(invalid_duration, latest_length, invalid_start, invalid_end):
        if len(invalid_duration) < latest_length:  # if first invalid within fixation...
            invalid_duration.append(invalid_end - invalid_start)  # ... append new
        else:  # else if multiple gaps within fixation...
            invalid_duration[latest_length - 1] += invalid_end - invalid_start  # ...add to
        return invalid_duration

    @staticmethod
    def get_invalid_duration(invalid_duration, gap_start, gap_end, fix_start, fix_end, fix_duration):
        latest_length = len(fix_end)
        if gap_start == []:  # if no gaps
            invalid_duration.append(0)  # no invalid duration
        else:
            for g_i in range(len(gap_start)):  # for each gap
                if len(gap_end) == len(gap_start) - 1 & g_i == len(gap_start) - 1:  # if latest gap still open...
                    if fix_start[-1] <= gap_start[g_i] < fix_end[-1]:  # ...and any gaps start inside this fixation
                        invalid_start = gap_start[g_i]  # invalid start at gap start
                        invalid_end = fix_end[-1]  # invalid end at fixation end
                        invalid_duration = \
                            GazeCollisionFixations.add_invalid_dur(invalid_duration, latest_length,
                                                                   invalid_start, invalid_end)

                    elif gap_start[g_i] <= fix_start[-1]:  # ...and whole fixation inside gap
                        invalid_duration = \
                            GazeCollisionFixations.add_invalid_dur(invalid_duration, latest_length,
                                                                   0, fix_duration[-1])  # whole fixation is invalid
                    else:
                        invalid_duration = \
                            GazeCollisionFixations.add_invalid_dur(invalid_duration, latest_length, 0, 0)  # no invalid
                else:  # if not latest, open gap...
                    if fix_start[-1] <= gap_start[g_i] < fix_end[-1]:  # ... and if any gaps start inside this fixation
                        invalid_start = gap_start[g_i]  # invalid start at gap start
                        if fix_start[-1] <= gap_end[g_i] <= fix_end[-1]:  # and if gap ends within fixation
                            invalid_end = gap_end[g_i]
                        else:  # or if gap doesn't end before fixation
                            invalid_end = fix_end[g_i]
                        invalid_duration = \
                            GazeCollisionFixations.add_invalid_dur(invalid_duration, latest_length,
                                                                   invalid_start, invalid_end)
                    elif fix_start[-1] <= gap_end[g_i] <= fix_end[
                        -1]:  # else if any gaps end within latest fixation (but don't start within)
                        invalid_start = fix_start[-1]
                        invalid_end = gap_end[g_i]
                        invalid_duration = \
                            GazeCollisionFixations.add_invalid_dur(invalid_duration, latest_length,
                                                                   invalid_start, invalid_end)

                    elif gap_start[g_i] <= fix_start[-1] & fix_end[-1] <= gap_end[
                        g_i]:  # else if whole fixation invalid
                        invalid_duration = \
                            GazeCollisionFixations.add_invalid_dur(invalid_duration, latest_length,
                                                                   0, fix_duration[-1])  # whole fixation is invalid
                    else:
                        invalid_duration = \
                            GazeCollisionFixations.add_invalid_dur(invalid_duration, latest_length, 0, 0)  # no invalid
        return invalid_duration
