## active information storage
# %%
# Import classes
from idtxl.active_information_storage import ActiveInformationStorage
from idtxl.data import Data

# imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation
import itertools
import random
import seaborn as sns
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.d03_processing.BlinkProcessor import BlinkProcessor
from src.d03_processing.fixations.SignalProcessor import SignalProcessor
from src.d03_processing.fixations.GazeCollision import GazeCollision
from src.d03_processing.fixations.VR_IDT import VR_IDT
from src.d03_processing.fixations.I_VDT import I_VDT
from src.d03_processing.fixations.I_HMM import I_HMM
from src.d03_processing.BlinkProcessor import BlinkProcessor
from src.d03_processing.fixations.SignalProcessor import SignalProcessor
from src.d03_processing.fixations.I_VDT import I_VDT
from src.d00_utils.TaskObjects import *
from src.d03_processing.fixations.FixationProcessor import FixationProcessor
from src.d03_processing.TimepointProcessor import TimepointProcessor
from src.d01_data.fetch.fetch_timepoints import fetch_timepoints
from src.d01_data.fetch.fetch_viewings import fetch_viewings
from src.d01_data.fetch.fetch_trials import fetch_trials
from src.d01_data.fetch.fetch_timepoints import fetch_timepoints
from src.d03_processing.aoi import collision_sphere_radius
from src.d03_processing.feature_extract.to_viewing import to_viewing
from src.d03_processing.fixations.FixAlgos import *
from src.d03_processing.feature_calculate.transition_calculations import external_fixations

# %%
# a) Generate test data
# data = Data()
# data.generate_mute_data(n_samples=1000, n_replications=5)
# print(data)

# # b) Initialise analysis object and define settings
# network_analysis = ActiveInformationStorage()
# settings = {'cmi_estimator':  'JidtGaussianCMI',
#             'max_lag': 5}

# # c) Run analysis
# results = network_analysis.analyse_network(settings=settings, data=data)

# # d) Plot list of processes with significant AIS to console
# print(results.get_significant_processes(fdr=False))
# %%
# get viewings
n = 10
all_trials = pd.read_csv("C:\\Users\\Luke\\Documents\\AlloEye\\data\\feature_saves\\all_real_trials.csv")
n_trials = len(all_trials)
print(n_trials)
r_inds = np.random.randint(0, n_trials, n)
rand_trials = list(all_trials.trial_id.to_numpy()[r_inds])

# # temp while bug fixed
# for t in rand_trials:
#     if '21r2' in t:
#         rand_trials.remove(t)

viewings = []
for t in rand_trials:
    viewings.append(f"{t}_enc")
    viewings.append(f"{t}_ret")

# viewings = ['alloeye_22r3_11_enc']
timepoints = fetch_timepoints("all", viewing_id=viewings)
print(timepoints.shape)

p_tps = []
n_viewings = len(viewings)
# preprocess
for i in range(n_viewings):
    # ind = random.randint(0, len(viewings)-1)
    viewing = viewings[i]
    # viewing = "alloeye_52r2_17_ret"

    # print(viewing)
    tps = timepoints[timepoints.viewing_id == viewing].reset_index(drop=True)
    if tps is None or len(tps) < 2:
        p_tps.append(None)
        continue
    # print(viewings[i])
    # print(tps.shape)
    s_tps = SignalProcessor.sandwiched_gp_filter(tps.copy(deep=True))
    b_tps = BlinkProcessor(s_tps.copy(deep=True), max_blink_duration=1000, d_impute_threshold=0.16,
                           impute_buffer_duration=8).timepoints
    if b_tps is None:
        print(f"{viewing} all blinks?")
        p_tps.append(None)
        continue
    f_tps = SignalProcessor.filter_timepoints(b_tps.copy(deep=True))
    p_tps.append(f_tps)

# fixations

algos = {'ivdt': I_VDT,
         'gc': GazeCollision,
         'idt': VR_IDT,
         'ihmm': I_HMM
         }
algo_names = {}
fix_dfs = {}
for name in algos.keys():
    fix_dfs.update({name: []})

fix_tps = []
for i in range(len(p_tps)):
    tps = p_tps[i]
    if tps is None:
        fix_tps.append(None)
        for name, _class in algos.items():
            fix_dfs[name].append(None)
        continue

    tps = BlinkProcessor.remove_missing(tps.copy(deep=True))

    for name, _class in algos.items():
        instance = _class(tps)
        algo_names.update({name: instance.method_name})
        tps[f'{name}_fixation'] = instance.timepoints.fixation
        fix_dfs[name].append(instance.fix_df)

    fix_tps.append(tps.copy(deep=True))

# %%
algo = 'ivdt'
dfs = fix_dfs[algo]
n_sig = 0
n_nones = 0
for i in range(len(dfs)):
    df = dfs[i]
    if df is None:
        n_nones += 1
        continue
    df = df[df.fixation_or_saccade == 'fixation'].reset_index(drop=True)
    df = external_fixations(df)
    df.head(20)

    # define data for one fixation sequence
    objects = df.object.to_numpy()
    uniques = np.unique(objects)  # get number of AOIs
    print(len(uniques))

    # convert to unique integers
    for i, o in enumerate(uniques):
        objects = np.where(objects == o, i, objects)
    # objects = np.array(objects, dtype=np.int_)  # THIS SEEMS TO YIELD NANS ALWAYS
    print(objects)

    # define Data class instance
    data = Data(objects, dim_order='s', normalise=False)
    print(data.data.dtype)

    # Initialise analysis object and define settings
    # NOTE: this finally worked for me after defining 'n_discrete_bins' as the number of AOIs
    max_lag = 3
    network_analysis = ActiveInformationStorage()
    settings = {'cmi_estimator': 'JidtDiscreteCMI',
                'max_lag': max_lag,
                'discretise_method': 'equal',
                'n_discrete_bins': len(uniques)
                }

    # c) Run analysis
    results = network_analysis.analyse_network(settings=settings, data=data)

    # d) Plot list of processes with significant AIS to console
    print('is significant?', results.get_significant_processes(fdr=False))
    results_dict = results.get_single_process(0, fdr=False)
    if not pd.isna(results_dict['ais']):
        print(results_dict)
        n_sig += 1

print('n_sig', n_sig)
print('n_nones', n_nones)
print("end")
