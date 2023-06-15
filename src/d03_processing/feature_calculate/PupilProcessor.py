import numpy as np
import pandas as pd
import scipy.interpolate as interp

from src.d03_processing.TimepointProcessor import TimepointProcessor


class PupilProcessor(TimepointProcessor):
    def __init__(self, timepoints):
        super().__init__(timepoints)
        self.left_diameter, self.right_diameter = self.parse_pupils(timepoints)



    def summarise(self):
        # profile data e.g. summarise blinks and other low-quality readings
        # remove blink data
        # cross-sectional summary
            # test for normality
            # return mean/std
        pass

    def parse_pupils(self, timepoints=None, interp=True):
        if self.none:
            return None, None
        timepoints = self.timepoints if timepoints is None else timepoints
        left_diameter = timepoints.left_pupil_diameter
        right_diameter = timepoints.right_pupil_diameter
        pd.options.mode.chained_assignment = None  # default='warn'
        left_diameter.loc[left_diameter < 0] = np.nan
        right_diameter.loc[right_diameter < 0] = np.nan
        pd.options.mode.chained_assignment = 'warn'  # default='warn'

        return left_diameter, right_diameter

    @staticmethod
    def combined_diameter_median(left_diameter, right_diameter):
        return np.mean([np.nanmedian(left_diameter), np.nanmedian(right_diameter)])

    @staticmethod
    def combined_diameter_iqr(left_diameter, right_diameter):
        if len(left_diameter) < 2 or len(left_diameter) < 2:
            return np.nan
        else:
            left_quantiles = np.nanquantile(left_diameter, [0.25, 0.75])
            right_quantiles = np.nanquantile(right_diameter, [0.25, 0.75])
            return np.mean([left_quantiles[1] - left_quantiles[0], right_quantiles[1] - right_quantiles[0]])
