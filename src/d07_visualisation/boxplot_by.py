import os.path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.d01_data.fetch.fetch_trials import fetch_trials
from src.d01_data.database.Errors import InvalidValue, UnmatchingValues

from src.d02_intermediate.df_reshaping import *
from data import d01_raw
from plotnine import *

from src.d03_processing.feature_extract.viewing_to_trial import viewing_to_trial


def boxplot_by(features, by='condition', group='group', ppts="all", plot="boxplot", via_timepoints=False):
    """
    produces p plots for p features, converting trial-level into per-participant data.
    boxplots produced by default but can have violin etc. Must be variant on the boxplot
    x-axis (each box) defined under 'by', groups (including colours) defined by group
    """
    # if by == "condition":
    #     # plot with plotnine
    #     (ggplot(df_long, aes('condition', 'score', fill='group'))
    #      + geom_boxplot()
    #      + facet_wrap('~correct'))
    pass