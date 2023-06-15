import warnings

import numpy as np
import pandas as pd

from src.d00_utils.TaskObjects import TaskObjects
from src.d00_utils.generic_tools import shift_list_up, shift_list_down
from src.d03_processing.TimepointProcessor import TimepointProcessor
from src.d03_processing.fixations.FixationProcessor import FixationProcessor
from src.d03_processing.fixations.GazeCollision import GazeCollision
from src.d03_processing.fixations.SignalProcessor import SignalProcessor


class BlinkProcessor(TimepointProcessor):
    def __init__(self, timepoints, max_blink_duration=1000, d_impute_threshold=0.1, impute_buffer_duration=8):
        super().__init__(timepoints)
        self.pre_blink = 1  # ms
        self.post_blink = 1  # ms
        self.lerp_buffer = 100  #ms
        self.openness_threshold = 0.01
        self.trackloss_threshold = 0.25  # % of viewing period
        self.technical_threshold = 100  # ms
        self.max_blink_duration = max_blink_duration    # if 7000 (length of viewing), won't impute blinks
        self.d_impute_threshold = d_impute_threshold    # if 0, won't impute any blinks
        self.impute_buffer_duration = impute_buffer_duration
        self.timepoints = self.blinks(self.timepoints)

    def blinks(self, timepoints=None):
        if self.none:
            return None
        timepoints = self.timepoints if timepoints is None else timepoints
        og_tps = timepoints.copy(deep=True)
        tps = timepoints.eye_timestamp_ms.to_numpy()

        left_diameter = timepoints.left_pupil_diameter.to_numpy()
        right_diameter = timepoints.right_pupil_diameter.to_numpy()
        left_openness = timepoints.left_eye_openness.to_numpy()
        right_openness = timepoints.right_eye_openness.to_numpy()
        ld_blink = left_diameter == -1
        rd_blink = right_diameter == -1
        lo_blink = left_openness < self.openness_threshold
        ro_blink = right_openness < self.openness_threshold

        # raw_trackloss = ld_blink | rd_blink | lo_blink | ro_blink
        raw_trackloss = (ld_blink & rd_blink) | (ld_blink & ro_blink) | (lo_blink & rd_blink) | (lo_blink & ro_blink)

        start_blink, end_blink = TimepointProcessor.get_start_end(raw_trackloss)

        # shift start and end of blinks by 1 timepoint
        shift_start = np.array(shift_list_up(list(start_blink), 0))
        try:
            if shift_start[0] == 0 and start_blink[0] == 1:
                shift_start[0] = start_blink[0]
        except IndexError as e:
            print(e)
            print('returning None')
            return None
        shift_end = np.array(shift_list_down(list(end_blink), 0))
        if shift_end[-1] == 0 and end_blink[-1] == 1:
            shift_end[-1] = end_blink[-1]
        # set blink window before start and after blink end
        t_high = SignalProcessor(start_blink, tps).up_t(1)

        # note - this approach only works if you upsample to unit i.e. 1 ms here
        z_tps = tps - tps[0]
        minus_pre = z_tps[shift_start == 1] - self.pre_blink
        minus_pre = np.where(minus_pre < 0, 0, minus_pre)
        plus_post = z_tps[shift_end == 1] + self.post_blink
        plus_post = np.where(plus_post > len(t_high) - 1, len(t_high) - 1, plus_post)

        # check if another blink window starts within each window
        # get index of next end for each start
        # if sum more than 0 between that in start, then flag up
        # append and cut start indices and subtract plus_post - any negative will be start before next end
        shunt_start_ind = np.append(minus_pre, len(t_high))[1:]
        try:
            inside_start_check = shunt_start_ind - plus_post
        except ValueError as e:
            raise e
        av_start = np.where(inside_start_check < 0, shunt_start_ind, -1)
        av_end = np.where(inside_start_check < 0, plus_post, -1)

        # upsample relevant
        eye_cols = ['gaze_collision_x', 'gaze_collision_y', 'gaze_collision_z',
                    'left_pupil_diameter', 'right_pupil_diameter']
        # eye_sigs = []
        # for col in eye_cols:
        #     eye_sigs.append()

        # do for all these if relevant:  lGazeOrigin_x	lGazeOrigin_y	lGazeOrigin_z	rGazeOrigin_x	rGazeOrigin_y	rGazeOrigin_z	 lGazeDirection_x	lGazeDirection_y	lGazeDirection_z	 rGazeDirection_x	rGazeDirection_y	rGazeDirection_z

        max_buffer = int(np.ceil(self.lerp_buffer / (np.mean(np.diff(t_high)))))
        new_blinks = np.zeros(len(t_high), dtype=int)
        for j in range(len(eye_cols)):
            sig = SignalProcessor(timepoints[eye_cols[j]].to_numpy(), tps).upsample_1d(1)
            i = 0
            add_i = 0
            _from = 0
            while i < len(minus_pre):
                blinks_set = False
                r = np.array([ix for ix in range(len(t_high))])
                av_bools = np.where((r >= av_start[i]) & (r <= av_end[i]), True, False)
                interp_sig = []
                interp_time = []
                start = minus_pre[i]
                _from = 0 if i == 0 else plus_post[i+add_i - 1]
                n_tps = len(t_high[_from:start])
                start_buffer = int(np.minimum(n_tps, max_buffer))
                for k in range(start_buffer):
                    interp_time.append(k)
                    interp_sig.append(sig[start - start_buffer + k])
                interp_sig.append(sig[start])
                interp_time.append(start_buffer)

                if av_start[i] != -1:
                    interp_sig.append(np.mean(sig[av_bools]))
                    interp_time.append(int(((av_end[i] + av_start[i])/2) - start + start_buffer))
                    interp_by = 'linear'
                    add_i=1
                else:
                    interp_by = 'linear'
                    add_i=0
                end = plus_post[i+add_i]
                interp_len = (end - start)
                interp_sig.append(sig[end])
                interp_time.append(start_buffer + interp_len)
                _to = len(t_high) - 1 if i == len(minus_pre) - 1 else minus_pre[i + 1]
                n_tps = len(t_high[end:_to])
                end_buffer = int(np.minimum(n_tps, max_buffer))
                for k in range(end_buffer):
                    interp_time.append(start_buffer + interp_len + 1 + k)
                    interp_sig.append(sig[end + 1 + k])

                new_blinks[start:end] = 1

                blink_bools = np.where((r >= (start - start_buffer)) & (r <= (end + end_buffer)))[0]
                try:
                    sig[blink_bools] = SignalProcessor(np.array(interp_sig), np.array(interp_time)).upsample_1d(1, type=interp_by)
                except ValueError as e:
                    raise e

                i+=1+add_i

            timepoints[eye_cols[j]] = SignalProcessor.downsample_1d_signal(t_high, tps, sig)


        blinks_adj = SignalProcessor.downsample_1d_threshold(t_high, tps, new_blinks)
        # start_adj, end_adj = TimepointProcessor.get_start_end(blinks_adj)

        # time threshold
        blinks_thresh = FixationProcessor.threshold_fixboolarray_duration(np.array(blinks_adj), self.technical_threshold, tps)
        start_thresh, end_thresh = TimepointProcessor.get_start_end(blinks_thresh)

        # impute some blinks
        timepoints, blinks_impute = BlinkProcessor.impute_blinks(timepoints, blinks=blinks_thresh,
                                                                 max_duration=self.max_blink_duration,
                                                                 d_threshold=self.d_impute_threshold,
                                                                 impute_saccades=False,
                                                                 impute_external_aois=True,
                                                                 impute_buffer_duration=self.impute_buffer_duration,
                                                                 impute_with_raw=True, raw_tps=og_tps)

        missing = blinks_thresh & np.invert(blinks_impute)
        timepoints['trackloss'] = raw_trackloss
        timepoints['tech_trackloss'] = raw_trackloss & np.invert(blinks_thresh)
        timepoints['blink'] = blinks_thresh
        timepoints['blink_imputed'] = blinks_impute
        timepoints['start_blink'] = start_thresh
        timepoints['end_blink'] = end_thresh
        timepoints['missing'] = missing
        return timepoints

    @classmethod
    def remove_missing(cls, timepoints, cols=('gaze_collision_x', 'gaze_collision_y', 'gaze_collision_z',
                                              'left_pupil_diameter', 'right_pupil_diameter')):
        pd.options.mode.chained_assignment = None  # default='warn'
        for col in cols:
            timepoints[col][timepoints.missing == 1] = np.nan
        pd.options.mode.chained_assignment = 'warn'  # default='warn'
        return timepoints

    @staticmethod
    def n_blinks(timepoints):
        if timepoints is None:
            return np.nan
        elif any([len(timepoints) < 1, 'start_blink' not in list(timepoints.columns)]):
            return np.nan
        else:
            return np.sum(timepoints.start_blink)

    @staticmethod
    def total_p_trackloss(timepoints, trackloss_column='missing'):
        """total proportion of trackloss for viewing period - used to exclude viewings with e.g. > 25%"""
        if timepoints is None:
            # print('None')
            return np.nan
        elif any([len(timepoints) < 1, 'trackloss' not in list(timepoints.columns)]):
            tl = 'trackloss' not in list(timepoints.columns)
            print(f"len: {len(timepoints)}, trackloss?: {tl}")
            return np.nan
        else:
            t_diff = np.append(np.array([0]), np.diff(timepoints.eye_timestamp_ms))
            bools = np.array(timepoints[trackloss_column].to_numpy(), dtype=bool)
            n_trackloss = np.sum(t_diff[bools])
            total = np.sum(t_diff)
            p_trackloss = float(n_trackloss / total)
            return p_trackloss

    @classmethod
    def impute_blinks(cls, timepoints, blinks, max_duration, d_threshold, impute_saccades, impute_external_aois,
                      impute_buffer_duration, impute_with_raw, raw_tps):

        # impute candidates
        too_high = FixationProcessor.threshold_fixboolarray_duration(np.array(blinks), max_duration,
                                                                     timepoints.eye_timestamp_ms.to_numpy())
        blink_imps = blinks & np.invert(too_high)

        # get before and after blinks - n_tps is based on average tp rate
        max_tps_buffer = int(np.ceil(impute_buffer_duration / (np.mean(np.diff(timepoints.eye_timestamp_ms.to_numpy())))))
        start_blinks, end_blinks = TimepointProcessor.get_start_end(blink_imps)
        where_starts = np.where(start_blinks == 1)[0]
        where_ends = np.where(end_blinks == 1)[0]
        _from = 0
        prev_tps_buffers = []
        it_len = int(np.sum(start_blinks))
        for i in range(it_len):
            blink_start = where_starts[i]
            n_tps = len(timepoints[_from:blink_start])
            n_tps_buffer = int(np.minimum(n_tps, max_tps_buffer))
            prev_tps_buffers.append(n_tps_buffer)
            _from = where_ends[i]

        next_tps_buffers = []
        it_len = int(np.sum(end_blinks))
        for i in range(it_len):
            blink_end = where_ends[i]
            _to = where_starts[i + 1] if (i < it_len - 1) else len(timepoints) - 1
            n_tps = len(timepoints[blink_end:_to])
            n_tps_buffer = int(np.minimum(n_tps, max_tps_buffer))
            next_tps_buffers.append(n_tps_buffer)

        #
        cols = ('gaze_collision_x', 'gaze_collision_y', 'gaze_collision_z')
        point_matrix = TimepointProcessor.create_gaze_point_matrix(timepoints)
        head_loc_matrix = TimepointProcessor.create_head_loc_matrix(timepoints)
        timepoints_2d = FixationProcessor.head_project(point_matrix, head_loc_matrix)
        d_list = []
        for i in range(it_len):
            prev_from = np.maximum(where_starts[i] - prev_tps_buffers[i], 0)
            prev_points = timepoints_2d[prev_from:where_starts[i] + 1]
            next_to = np.minimum(where_ends[i] + next_tps_buffers[i], len(timepoints)-1)
            next_points = timepoints_2d[where_ends[i]:next_to + 1]
            with warnings.catch_warnings():  # catch nan warnings
                warnings.simplefilter("ignore", category=RuntimeWarning)
                before_mean = np.nanmean(prev_points, axis=0)
                after_mean = np.nanmean(next_points, axis=0)

            # displacement squared for speed
            displacement_squared = (after_mean[0] - before_mean[0]) ** 2 +\
                                   (after_mean[1] - before_mean[1]) ** 2 +\
                                   (after_mean[2] - before_mean[2]) ** 2
            d_list.append(np.sqrt(displacement_squared))
            objs = np.unique(timepoints.gaze_object[where_starts[i]:where_ends[i] + 1])

            # DIFFERENT THRESHOLD FOR ext objs
            ext_obj_bool = (TaskObjects.floor in objs) or (TaskObjects.dome in objs)\
                           or (any(np.isin(TaskObjects.distant_objects, objs))) \
                           or (any(np.isin(TaskObjects.ext_cues, objs)))

            if displacement_squared < d_threshold**2 or (len(objs) == 1 and ext_obj_bool and impute_external_aois):

                 # only if registered within object i.e. before and after trackloss same object
                if impute_with_raw:
                    # don't actually change gaze points, keep original
                    timepoints[where_starts[i]:where_ends[i] + 1] = raw_tps[where_starts[i]:where_ends[i] + 1]
                else:
                    blink_imps[where_starts[i]:where_ends[i] + 1] = 0
                    # impute across before and after
                    # not completed
                    pass
            else:
                blink_imps[where_starts[i]:where_ends[i] + 1] = 0

        # need to consider start - tps before blink start < n_tps_buffer - same with end
        # need to consider blinks in quick succession - tps between last end and this start < n_buffer
        # for these - just take the mean - find max of n_tps_buffer and n tps before blink
            # if 0, can't impute - won't be any 0s if coded right
            # if 1, take this instead of mean to avoid warnings


        if impute_saccades:
            # not complete - find blinks that start and end a certain distance apart and e.g. distance between two aois
            # add saccade in middle based on amplitude and known saccade speed, remaining time split between before and end
            pass


        return timepoints, np.array(blink_imps, dtype=bool)



