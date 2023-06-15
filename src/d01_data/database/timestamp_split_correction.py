import numpy as np
from scipy.stats import pearsonr
from src.d01_data.database.Errors import InvalidValue


def timestamp_split_correction(viewing_tps, t_diff_threshold=20000):
    """function to correct error with timestamp recordings: occasionally and inexplicably drops timestamps to very low negative number in middle of viewing. The following checks for this and outputs a corrected dataframe. """

    # none checks
    if viewing_tps is None:
        return viewing_tps
    if len(viewing_tps) < 2:
        return viewing_tps

    # copy and sort
    tps = viewing_tps.copy(deep=True).sort_values(by='eye_timestamp_ms').reset_index(drop=True)

    # check for one viewing, otherwise will combine timestamps for both viewings
    if len(np.unique(tps.viewing_id)) > 1:
        raise InvalidValue(len(np.unique(tps.viewing_id)), 1, message="too many viewings in timepoints")

    # calculate correlation between timestamp and frame, should be almost 1
    timestamp = tps.eye_timestamp_ms.to_numpy()
    frame = tps.eye_frame_number.to_numpy()
    r, p = pearsonr(timestamp, frame)

    # if correlation too low (will be negative if timestamp split)
    if r < 0.999:   # should be almost perfectly positively correlated

        # sort by frame (which never seems to have the problem)
        tps = tps.sort_values(by='eye_frame_number').reset_index(drop=True)
        frames = tps.eye_frame_number.to_numpy()
        timestamps = tps.eye_timestamp_ms.to_numpy()

        print(f"correcting timestamp split error for {tps.viewing_id[0]}")

        # find where split happens (accounts for more than 1 although haven't seen yet)
        split_idxs = []
        for i in range(len(frames)-1):
            this_timestamp = timestamps[i]
            next_timestamp = timestamps[i+1]
            if np.abs(next_timestamp - this_timestamp) > t_diff_threshold:
                split_idxs.append(i+1)

        # correction: split into chunks, assume mean timestep, carry on from previous chunk
        for i in range(len(split_idxs)):
            split_idx = split_idxs[i]
            chunk_1 = tps.iloc[:split_idx, :].eye_timestamp_ms.to_numpy()
            chunk_2 = tps.iloc[split_idx:, :].eye_timestamp_ms.to_numpy()
            chunk_1_diff_mean = np.mean(np.diff(chunk_1))
            chunk_2_diff_mean = np.mean(np.diff(chunk_2))
            diff_mean = np.mean([chunk_1_diff_mean, chunk_2_diff_mean]).astype(int)
            chunk_1_end = chunk_1[-1]
            chunk_2_start = chunk_2[0]
            chunk_2 -= chunk_2_start
            chunk_2 += chunk_1_end + diff_mean
            tps.loc[split_idx:, 'eye_timestamp_ms'] = chunk_2

    return tps
