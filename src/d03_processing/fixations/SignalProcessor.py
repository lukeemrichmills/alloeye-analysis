import numpy as np
import pandas as pd
import scipy.signal as ssg
import scipy.interpolate as interp

from src.d00_utils.generic_tools import shift_list_down, shift_list_up
from src.d01_data.database.Errors import InvalidValue
from src.d03_processing import aoi

from src.d03_processing.TimepointProcessor import TimepointProcessor


class SignalProcessor:

    def __init__(self, signal, time):
        self.signal = signal
        self.time = time

    def up_t(self, new_interval):
        return np.arange(np.min(self.time), np.max(self.time) + 1, new_interval)

    def upsample_multi(self, new_interval, signal=None):
        t_new = self.up_t(new_interval)
        signal = self.parse_signal_multi(signal)
        out = []
        width = len(signal)
        width_ax = np.argmin(np.shape(signal))
        for i in range(width):
            sig = signal[i]
            interpf = interp.interp1d(self.time.reshape(-1, ), sig.reshape(-1, ))
            out.append(interpf(t_new))
        return out

    def parse_signal_multi(self, signal):
        signal = self.signal if signal is None else signal
        shape = np.shape(signal)
        width = np.min(shape)
        if width > 1:
            ax = np.argmin(shape)
            return np.split(self.signal, width, axis=ax)
        elif width == 1:
            return self.signal
        else:
            return self.signal.reshape(len(self.signal), 1)

    def upsample_1d(self, new_interval, signal=None, type='linear'):
        t_new = self.up_t(new_interval)
        sig = self.parse_signal_1d(signal)
        interpolater = interp.interp1d
        interpf = interpolater(self.time.reshape(-1, ), sig.reshape(-1, ), kind=type)
        out = interpf(t_new)
        return out

    def parse_signal_1d(self, signal):
        signal = self.signal if signal is None else signal
        return signal.reshape(len(self.signal), 1)

    def butter_lowpass(self, cutoff, fs, order=5):
        return ssg.butter(order, cutoff, fs=fs, btype='low', analog=False)

    def butter_lowpass_filter(self, data, cutoff, fs, order=5):
        b, a = self.butter_lowpass(cutoff, fs, order=order)
        y = ssg.lfilter(b, a, data)
        return y

    def zero_lag_butter(self, data, cutoff, fs, order=5):
        padding_front = np.repeat(data[0], fs)
        padding_back = np.repeat(data[-1], fs)
        data = np.concatenate((padding_front, data, padding_back))
        forward = self.butter_lowpass_filter(data, cutoff, fs, order)
        reverse = np.flip(forward)
        backward = self.butter_lowpass_filter(reverse, cutoff, fs, order)
        rereverse = np.flip(backward)
        return rereverse[int(fs):-int(fs)]  # remove zero padding

    def zero_moving_average(self, x, w):
        x = x.reshape(-1, )
        filt = np.convolve(x, np.ones(w), 'valid') / w
        len_diff = len(x) - len(filt)
        shift = int(np.ceil(len_diff / 2))
        out = np.concatenate((x[:shift], filt, x[
                                               -shift:]))  # appends first few of raw signal to beginning to keep length same - could ma filter this until
        return out

    def filter_signal(self, data, butter_order, butter_fs, butter_cutoff, mf_window, ma_window):
        butter = self.zero_lag_butter(data, butter_cutoff, butter_fs, butter_order)
        mf = ssg.medfilt(butter, mf_window)
        ma = self.zero_moving_average(mf, ma_window)
        return ma

    def velocity_filter(self, tps):
        # get angular velocity

        # mark any over 1000degrees per second

        # remove these points and interpolate
        pass

    @staticmethod
    def sandwiched_gp_filter(tps):
        # remove any gp on an object where two before and after are the same

        tps = tps.copy(deep=True).sort_values(by='eye_timestamp_ms').reset_index(drop=True)
        # get object arrays - shift up and down to get previous and next objects
        fix_objects = tps.gaze_object.to_list()
        prev_fix_objects = np.array(shift_list_down(fix_objects, 'None'))
        prevprev_fix_objects = np.array(shift_list_down(prev_fix_objects.tolist(), 'None'))
        next_fix_objects = np.array(shift_list_up(fix_objects, 'None'))
        nextnext_fix_objects = np.array(shift_list_up(next_fix_objects.tolist(), 'None'))
        fix_objects = np.array(fix_objects)


        # sandwiched points
        sandwiched_gps = (fix_objects != next_fix_objects) & \
                           (fix_objects != nextnext_fix_objects) & \
                           (fix_objects != prev_fix_objects) & \
                           (fix_objects != prevprev_fix_objects) & \
                           (prevprev_fix_objects == prev_fix_objects) & \
                           (prev_fix_objects == next_fix_objects) & \
                           (next_fix_objects == nextnext_fix_objects)

        fix_objects[sandwiched_gps] = prev_fix_objects[sandwiched_gps]
        tps['gaze_object'] = fix_objects
        cols = ('gaze_collision_x', 'gaze_collision_y', 'gaze_collision_z',
                'left_pupil_diameter', 'right_pupil_diameter')
        pd.options.mode.chained_assignment = None  # default='warn'
        for col in cols:
            prev = np.array(shift_list_down(tps[col].to_list(), np.nan), dtype=float)[sandwiched_gps]
            next = np.array(shift_list_up(tps[col].to_list(), np.nan), dtype=float)[sandwiched_gps]
            imp = np.mean([prev, next], axis=0)
            tps[col][sandwiched_gps] = imp

        cols = ('object_position_x', 'object_position_y', 'object_position_z')
        for col in cols:
            prev = np.array(shift_list_down(tps[col].to_list(), 'None'))[sandwiched_gps]
            tps[col][sandwiched_gps] = prev
        pd.options.mode.chained_assignment = 'warn'  # default='warn'
        # if np.sum(sandwiched_gps) > 0:
        #     print("test")

        return tps

    @staticmethod
    def filter_timepoints(tps, cols=('gaze_collision_x', 'gaze_collision_y', 'gaze_collision_z',
                                     'left_pupil_diameter', 'right_pupil_diameter',
                                     'camera_x', 'camera_y', 'camera_z',
                                     'cam_rotation_x', 'cam_rotation_y', 'cam_rotation_z'),
                          gaze_specs={'butter_order': 1, 'butter_fs': 1000, 'butter_cutoff': 40,
                                      'mf_window': 41, 'ma_window': 41},  # 25
                          cam_specs={'butter_order': 1, 'butter_fs': 1000, 'butter_cutoff': 20,
                                      'mf_window': 75, 'ma_window': 41},
                          custom_specs=None, downsample_to='original', aoi_adjust=True):
        if tps is None:
            print("none catch")

        tps = tps.sort_values(by=['eye_timestamp_ms']).reset_index(drop=True)
        # og_tps = tps.copy(deep=True)
        # test_tps = aoi.gaze_object_adjust(og_tps)
        t = tps.eye_timestamp_ms.to_numpy()

        # tps = SignalProcessor.sandwiched_gp_filter(tps)

        for col in cols:
            # time
            sig = tps[col].to_numpy()
            sp = SignalProcessor(sig, t)       # signal processor class instance with signal and time
            up = sp.upsample_1d(1)                  # up sample signal
            if custom_specs is None:            # define specifications for filter
                specs = cam_specs if 'cam' in col else gaze_specs
            else:
                specs = custom_specs
            filt_sig = sp.filter_signal(up, **specs)        # filter signal
            tps[col] = SignalProcessor.downsample_1d_signal(sp.up_t(1), t,
                                                            filt_sig)  # downsample back to original t

        if aoi_adjust:
            tps = aoi.gaze_object_adjust(tps)
        # changes = tps.gaze_collision_x != og_tps.gaze_collision_x
        # if any(changes):
        #     print(f"{np.sum(np.array(changes))} gaze points changed")
        return tps.reset_index(drop=True)

    @staticmethod
    def upsample_1d_bool(inputs, timestamps, sample_freq, output_size):
        output = np.zeros(output_size)
        new_timestamps = np.arange(0, output_size, sample_freq)
        scalar = len(new_timestamps) / len(inputs)
        z_tps = timestamps - timestamps[0]
        input_idxs = z_tps[inputs == 1]
        scaled_idxs = input_idxs * scalar
        for i in range(len(new_timestamps)):
            start = timestamps[i]
            end = timestamps[i + 1]
            start_idx = np.argmin(np.abs(new_timestamps - start))
            end_idx = np.argmin(np.abs(new_timestamps - end))
            output[start_idx:end_idx] = inputs[i]
        output[np.argmin(np.abs(new_timestamps - timestamps[-1])):] = inputs[-1]
        return output

    @staticmethod
    def upsample_1d_string(inputs, timestamps, sample_freq, output_size='max'):
        output_size = timestamps[-1] - timestamps[0] if output_size == 'max' else output_size
        output = np.zeros(output_size, dtype=str)
        new_timestamps = np.arange(0, output_size, sample_freq)
        for i in range(len(new_timestamps)-1):
            start = timestamps[i]
            end = timestamps[i + 1]
            start_idx = np.argmin(np.abs(new_timestamps - start))
            end_idx = np.argmin(np.abs(new_timestamps - end))
            mid_idx = int(np.ceil((start_idx + end_idx) / 2))
            output[start_idx:mid_idx] = inputs[i]
            output[mid_idx:end_idx] = inputs[i + 1]
        output[np.argmin(np.abs(new_timestamps - timestamps[-1])):] = inputs[-1]
        return output

    @staticmethod
    def downsample_1d_threshold(t_high, t_low, up_signal, threshold=0.5):
        down = np.zeros(len(t_low), dtype=int)
        t0 = t_low[0]
        j = 0
        start = 0
        for i in range(len(t_high)):
            # if i == (426981 - t0):
            #     print("catch")
            if t0 + i == t_low[j]:
                # downsampled timepoint defined as fixation if >50% upsampled are fixations
                if len(up_signal[start:i]) > 0:
                    p_points = np.sum(up_signal[start:i]) / len(up_signal[start:i])
                    last_point = up_signal[i-1]
                else:
                    p_points = up_signal[i]
                    last_point = up_signal[i]

                if p_points > threshold or last_point == 1:
                    down[j] = 1
                else:
                    down[j] = 0
                j += 1
                start = i

        return down

    @staticmethod
    def downsample_1d_signal(t_high, t_low, up_signal):
        unique, counts = np.unique(t_low, return_counts=True)
        dups = unique[counts > 1]
        if len(dups) > 0:
            raise InvalidValue(0, len(dups), message=f"duplicate timestamps {dups}, need to remove this when processing tps")

        down = np.zeros(len(t_low))
        t0 = t_low[0]
        j = 0
        for i in range(len(t_high)):
            if t0 + i == t_low[j]:
                down[j] = up_signal[i]
                j += 1

        return down

    def butter_lowpass(self, cutoff, fs, order=5):
        return ssg.butter(order, cutoff, fs=fs, btype='low', analog=False)

    def butter_lowpass_filter(self, data, cutoff, fs, order=5):
        b, a = self.butter_lowpass(cutoff, fs, order=order)
        y = ssg.lfilter(b, a, data)
        return y

    def wiener_filter(self, sig, window=5, noise=0.0002):
        sig = sig.reshape(len(sig), 1)
        return ssg.wiener(sig, (window, window), noise)

