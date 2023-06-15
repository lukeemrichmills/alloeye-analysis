import pandas as pd

from src.d03_processing.BlinkProcessor import BlinkProcessor
from src.d03_processing.fixations.FixAlgos import FixAlgo, fix_algo_dict
from src.d03_processing.fixations.SignalProcessor import SignalProcessor
from src.d03_processing.preprocess import preprocess_timepoints


def timepoint_to_fixations(viewing_list, all_timepoints, fix_algo_dict=fix_algo_dict(),
                           eye_trackloss_threshold=0.25):
    full_fix_df = None
    skipped = []
    for i in range(len(viewing_list)):    # per viewing

        # isolate relevant
        viewing_id = viewing_list[i]
        timepoints = all_timepoints.loc[all_timepoints.viewing_id == viewing_id].reset_index(drop=True)
        # preprocess timepoints
        timepoints, skip = preprocess_timepoints(timepoints, eye_trackloss_threshold)
        if skip:
            skipped.append(viewing_id)
            continue

        # process into fixations if indicated
        if (i + 1) % 20 == 0 or (i + 1) == len(fix_algo_dict.items()):
            print(f"Processing fixations using {list(fix_algo_dict.keys())} algorithm(s) for {viewing_id} ({i + 1} of {len(viewing_list)} viewings)")   # for each fixation algorithm
        for name, fix_class in fix_algo_dict.items():
            fix_processor = fix_class(timepoints)    # instantiate processor
            fix_df = fix_processor.fix_df           # get fixations

            # concatenate into full fixation dataframe
            if full_fix_df is None:
                full_fix_df = fix_df
            else:
                full_fix_df = pd.concat([full_fix_df, fix_df], ignore_index=True)
    print(f"Fixations processed for {len(viewing_list)} ({len(skipped)} skipped)")
    return full_fix_df
