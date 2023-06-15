import numpy as np

from src.d03_processing.TimepointProcessor import TimepointProcessor
from src.d03_processing.fixations.SignalProcessor import SignalProcessor


class BlinkProcessor(TimepointProcessor):
    def __init__(self, timepoints):
        super().__init__(timepoints)
        self.pre_blink = 60  # ms
        self.post_blink = 150  # ms
        self.openness_threshold = 0.3
        self.trackloss_threshold = 0.25  # % of viewing period
        self.timepoints = self.blinks(self.timepoints)

    def blinks(self, timepoints=None):
        if self.none:
            return None
        timepoints = self.timepoints if timepoints is None else timepoints

        left_diameter = timepoints.left_pupil_diameter.to_numpy()
        right_diameter = timepoints.right_pupil_diameter.to_numpy()
        left_openness = timepoints.left_eye_openness.to_numpy()
        right_openness = timepoints.right_eye_openness.to_numpy()
        ld_blink = left_diameter == -1
        rd_blink = right_diameter == -1
        lo_blink = left_openness < self.openness_threshold
        ro_blink = right_openness < self.openness_threshold

        raw_blinks = ld_blink | rd_blink | lo_blink | ro_blink


        start_blink, end_blink = TimepointProcessor.get_start_end(raw_blinks)

        # set blink window 60ms before start and 150ms after blink end
        tps = timepoints.eye_timestamp_ms.to_numpy()
        t_high = SignalProcessor(start_blink, tps).up_t(1)

        # note - this approach only works if you upsample to unit i.e. 1 ms here
        z_tps = tps - tps[0]
        minus_pre = z_tps[start_blink == 1] - self.pre_blink
        minus_pre = np.where(minus_pre < 0, 0, minus_pre)
        plus_post = z_tps[end_blink == 1] + self.post_blink
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
        eye_sigs = []
        for col in eye_cols:
            eye_sigs.append(SignalProcessor(timepoints[col].to_numpy(), tps).upsample_1d(1))

        # do for all these if relevant:  lGazeOrigin_x	lGazeOrigin_y	lGazeOrigin_z	rGazeOrigin_x	rGazeOrigin_y	rGazeOrigin_z	 lGazeDirection_x	lGazeDirection_y	lGazeDirection_z	 rGazeDirection_x	rGazeDirection_y	rGazeDirection_z

        new_blinks = np.zeros(len(t_high), dtype=int)
        i=0
        while i < len(minus_pre):
            blinks_set = False
            r = np.array([ix for ix in range(len(t_high))])
            av_bools = np.where((r >= av_start[i]) & (r <= av_end[i]), True, False)

            for j in range(len(eye_sigs)):
                sig = eye_sigs[j]
                interp_sig = []
                interp_time = []
                start = minus_pre[i]
                interp_sig.append(sig[start])
                interp_time.append(0)
                if av_start[i] != -1:
                    interp_sig.append(np.mean(sig[av_bools]))
                    interp_time.append(((av_end[i] + av_start[i])/2) - start)
                    add_i=1
                else:
                    add_i=0
                end = plus_post[i+add_i]
                interp_len = (end - start)
                interp_sig.append(sig[end])
                interp_time.append(interp_len)
                if blinks_set is False:
                    new_blinks[start:end] = 1
                    blinks_set = True
                blink_bools = np.where((r >= start) & (r <= end))
                try:
                    sig[blink_bools] = SignalProcessor(np.array(interp_sig), np.array(interp_time)).upsample_1d(1)
                except ValueError as e:
                    raise e
                eye_sigs[j] = sig
            i+=1+add_i

        for j in range(len(eye_sigs)):
            timepoints[eye_cols[j]] = SignalProcessor.downsample_1d_signal(t_high, tps, eye_sigs[j])

        blinks_adj = SignalProcessor.downsample_1d_threshold(t_high, tps, new_blinks)
        start_adj, end_adj = TimepointProcessor.get_start_end(blinks_adj)

        timepoints['trackloss'] = raw_blinks
        timepoints['blink'] = blinks_adj
        timepoints['start_blink'] = start_adj
        timepoints['end_blink'] = end_adj
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
    def total_p_trackloss(timepoints):
        """total proportion of trackloss for viewing period - used to exclude viewings with e.g. > 25%"""
        if timepoints is None:
            return np.nan
        elif any([len(timepoints) < 1, 'trackloss' not in list(timepoints.columns)]):
            return np.nan
        else:
            t_diff = np.append(np.array([0]), np.diff(timepoints.eye_timestamp_ms))
            n_trackloss = np.sum(t_diff[timepoints.trackloss])
            total = np.sum(t_diff)
            # print(f"n trackloss: {n_trackloss}, total t: {total}")
            return float(n_trackloss/total)
