"""
Create tables, import data to postgres
"""
import pandas as pd
import psycopg2
import platform

from src.d00_utils.EnumFuncWrapper import FuncWrapper
from src.d00_utils.file_dir_utils import get_data_dir
from src.d01_data.database.db_connect import db_connect
from src.d01_data.database.sql_commands import delete_tables, create_tables, fill_tables, add_features
from src.d01_data.database.Errors import UnknownComputer

# change directory location depending on pc identifier PUT THIS IN TXT OR JSON FILE

# db = 'gcloud_psql'
from src.d03_processing.Features import Features
from src.d03_processing.fixations.FixAlgos import fix_algo_dict
from src.d03_processing.fixations.VR_IDT import VR_IDT

db = 'local'
raw_data_dir = get_data_dir(from_ehd=True)

# define sql db connection
conn = db_connect(db)

# delete tables (testing only)
# delete_tables(conn)

# # create tables (if not exist)
# create_tables(conn)    # returns list of bools to skip metadata check if not exist

# # fill tables with data (if necessary)
# fill_tables(conn, raw_data_dir, "alloeye")

# add features to higher level tables
# rerun_fix = {'alloeye_3': ['alloeye_3r2_5_ret']}

rerun_viewing_features = ["gauss_dwell_pp", "gauss_dwell_centroid"]

rerun_trial_features = [*[f"{feat}_enc" for feat in rerun_viewing_features],
                        *[f"{feat}_ret" for feat in rerun_viewing_features],
                        *[f"{feat}_diff" for feat in rerun_viewing_features]]
# rerun_trial_features = [f"{feat}_diff" for feat in rerun_viewing_features]
rerun_trial_features = ["lev_ratio_xfix_s"]
# rerun_trial_features = []
# rerun_features = ['p_trackloss']
fix_algos = ['GazeCollision']
fix_algos_upload = {i: fix_algo_dict()[i] for i in fix_algos}
fix_algo_features = 'GazeCollision'

rerun_these_viewings=[]
# rerun_these_viewings = ['alloeye_40r3_5_enc']


# rerun_these_viewings = pd.read_csv("C:\\Users\\Luke\\Documents\\AlloEye\\data\\feature_saves\\vlist.csv").list.tolist()

rerun_these_trials = []
# rerun_these_trials = ['alloeye_55r1_5']

rerun_fixations_for = []
# rerun_fixations_for = rerun_these_viewings


# GETTING TOO COMPLICATED - SEPARATE INTO DIFFERENT FUNCTIONS FOR DIFFERENT LEVELS? OR IMPROVE SOMEHOW
add_features(conn, rerun_all_ppts=False,
             rerun_all_viewing_features=False, rerun_viewing_features=rerun_viewing_features,
             rerun_all_trial_features=False, rerun_trial_features=rerun_trial_features,
             rerun_all_fixations=False, rerun_fixations_for=rerun_fixations_for,
             rerun_all_viewings=False, rerun_viewings=rerun_these_viewings,
             rerun_all_trials=True, rerun_trials=rerun_these_trials,
             rerun_all_conditions=True, rerun_conditions=[],
             skip_viewing=True, skip_trial=False,
             skip_practice=False,
             fix_algos_upload=fix_algos_upload, fix_algo_features=fix_algo_features,
             rerun_everything=False)

if conn is not None:
    conn.close()
else:
    print("connection is None")

