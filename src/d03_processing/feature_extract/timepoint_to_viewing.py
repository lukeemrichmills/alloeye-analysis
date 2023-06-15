from src.d00_utils.DataLevel import DataLevel
from src.d00_utils.file_dir_utils import get_data_dir
from src.d01_data.fetch.batch_fetch import batch_fetch
from src.d01_data.fetch.fetch_viewings import fetch_viewings
from src.d03_processing.Features import Features
from src.d03_processing.feature_extract.timepoint_to_fixations import timepoint_to_fixations
from src.d03_processing.feature_extract.to_viewing import to_viewing
from src.d03_processing.fixations.FixationProcessor import *


def timepoint_to_viewing(pid, timepoints=None, save_timepoints=False, get_fixations=True,
                         fix_method='GazeCollision', fix_id_upload_method=('GazeCollision', 'IDT', 'IVT'),
                         viewing_features="all", timepoint_table="alloeye_timepoint_viewing"):
    """function to convert alloeye timepoints to viewing features. Optionally fetch timepoints or upload
    depending on timepoints input"""

    # for saving files
    save_dir = get_data_dir(folder='feature_saves')

    # fixation dict for different algos


    # fetch dataframe of viewings table for dataframe and batch fetching
    viewing_df = fetch_viewings(pid)

    # add new feature columns
    viewing_features = Features.viewing if viewing_features == "all" else viewing_features
    viewing_df = add_feature_columns(viewing_df, viewing_features)

    # check fixation contradiction
    if get_fixations is False and any([i in Features.viewing_fix_sacc_derived for i in viewing_features]):
        raise InvalidValue(message="get_fixations must be True if getting fixation/saccade-derived features!")

    # get viewing ids for batching
    viewing_list = viewing_df.viewing_id.values.tolist()

    if timepoints is None:  # if not manually supplying timepoints

        all_timepoints = batch_fetch(data_level=DataLevel.timepoint, table=timepoint_table,
                                     batch_col="viewing_id",batch_list=viewing_list)
        # save to csv
        if save_timepoints is True:
            all_timepoints.to_csv(f"{save_dir}all_timepoints.csv", index=False)
    else:
        # if uploaded timepoints
        all_timepoints = timepoints

    # process timepoints into fixations/saccades
    full_fix_df = None
    viewing_df = viewing_df.head(2)     # MUST REMOVE THIS
    if get_fixations is True:
        full_fix_df = timepoint_to_fixations(viewing_df, all_timepoints)


    # loop over each viewing and populate new features from timepoint data
    viewing_df = to_viewing(viewing_df, full_fix_df, all_timepoints, fix_method, viewing_features)

    return viewing_df, full_fix_df


def add_feature_columns(df, features):
    for feat in features:
        if feat not in df.columns:
            df[feat] = np.repeat(np.nan, len(df))

    return df


if __name__ == '__main__':
    timepoint_to_viewing()
