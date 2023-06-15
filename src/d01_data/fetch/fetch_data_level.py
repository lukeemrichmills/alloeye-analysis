from src.d00_utils.DataLevel import DataLevel
from src.d01_data.fetch.fetch_timepoints import fetch_timepoints
from src.d01_data.fetch.fetch_trials import fetch_trials
from src.d01_data.fetch.fetch_viewings import fetch_viewings
from src.d01_data.fetch.fetch_fixations import fetch_fixations

fetch_data_level = {
    DataLevel.timepoint: fetch_timepoints,
    DataLevel.fix_sacc: fetch_fixations,
    DataLevel.viewing: fetch_viewings,
    DataLevel.trial: fetch_trials
}