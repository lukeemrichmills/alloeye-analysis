from src.d01_data.database.timestamp_split_correction import timestamp_split_correction
from src.d03_processing.BlinkProcessor import BlinkProcessor
from src.d03_processing.fixations.SignalProcessor import SignalProcessor


def preprocess_timepoints(timepoints, eye_trackloss_threshold=0.25, offline=False):
    skip = False

    if timepoints is None:
        skip = True
        return timepoints, skip

    if len(timepoints) < 10:
        skip = True
        return timepoints, skip

    # check for time split error (see function for details)
    timepoints = timestamp_split_correction(timepoints)

    # remove random points
    timepoints = SignalProcessor.sandwiched_gp_filter(timepoints)

    # deal with trackloss - interpolate or mark as missing
    timepoints = BlinkProcessor(timepoints).timepoints
    missingness = BlinkProcessor.total_p_trackloss(timepoints, 'missing')

    # deal with nones
    if timepoints is None:
        skip = True
        return timepoints, skip

    # filter timepoint data using range of filters
    timepoints = SignalProcessor.filter_timepoints(timepoints, aoi_adjust=not offline)

    # missingness removal - e.g. mark as NaNs
    timepoints = BlinkProcessor.remove_missing(timepoints)

    # if missingness above threshold, skip
    if missingness > eye_trackloss_threshold:
        skip = True
        return timepoints, skip

    return timepoints, skip
