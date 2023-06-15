import json
import os
import warnings
from collections.abc import Iterable

import numpy as np
import pandas as pd
import math
import multimatch_gaze as mm
from sklearn.preprocessing import StandardScaler

from src.d00_utils.TaskObjects import TaskObjects
from src.d01_data.database.Errors import InvalidValue, InvalidInput
from src.d01_data.fetch.fetch_fixations import fetch_fixations
from src.d03_processing.FeatureWrapper import FeatureWrapper
from src.d03_processing.feature_calculate.transition_calculations import conditional_prob_matrix, symmetric_KL, \
    external_fixations
from src.d07_visualisation.plots import *


def enc_ret_split(trial_row):
    enc_df = trial_row[trial_row.viewing_type == 'enc']
    ret_df = trial_row[trial_row.viewing_type == 'ret']
    return enc_df, ret_df


def D_KL(trial_row, fix_df):
    trial_row = trial_row.reset_index(drop=True)
    if fix_df is None:
        return np.nan

    # replace previous position object with table at retrieval


    # checks
    enc_objs = trial_row.p_matrix_objects_enc[0]
    enc_objs_copy = trial_row.p_matrix_objects_enc[0]
    ret_objs = trial_row.p_matrix_objects_ret[0]
    ret_objs_copy = trial_row.p_matrix_objects_ret[0]
    enc_mat = trial_row.p_matrix_enc[0]
    ret_mat = trial_row.p_matrix_ret[0]

    def is_invalid(input):
        return any([input is None, input == FeatureWrapper.str_too_short, input==FeatureWrapper.str_too_long])

    if any([is_invalid(i) for i in [enc_objs, ret_objs, enc_mat, ret_mat]]):
        return np.nan

    # get matrix keys:
    enc_objs = np.array(json.loads(trial_row.p_matrix_objects_enc[0]))
    ret_objs = np.array(json.loads(trial_row.p_matrix_objects_ret[0]))



    # if TaskObjects.invisible_object in ret_objs:
    #     if 'Table'



    # get matrices
    c = 0.00001
    enc_c = c / trial_row.n_fix_total_enc[0]
    ret_c = c / trial_row.n_fix_total_ret[0]
    enc_matrix = np.array(json.loads(trial_row.p_matrix_enc[0]))
    ret_matrix = np.array(json.loads(trial_row.p_matrix_ret[0]))

    # Table is all the same
    obj_lists = [enc_objs, ret_objs]
    for i, objs in enumerate(obj_lists):
        for j, obj in enumerate(objs):
            if 'Table' in obj or obj == TaskObjects.invisible_object:
                obj_lists[i][j] = 'Table'

    # Combine all Tables
    mat_lists = [enc_matrix, ret_matrix]

    for i, objs in enumerate(obj_lists):
        remove_js = []
        table_j = None
        for j, obj in enumerate(objs):
            if 'Table' in obj:
                if table_j is None:
                    table_j = j
                else:
                    mat_lists[i][table_j, :] += mat_lists[i][j, :]
                    mat_lists[i][:, table_j] += mat_lists[i][:, j]
                    remove_js.append(j)
        # remove other columns
        inds = [i for i in range(len(obj_lists[i])) if i not in remove_js]
        mat_lists[i] = mat_lists[i][inds, :][:, inds]
        obj_lists[i] = obj_lists[i][inds]

    enc_objs, ret_objs = obj_lists
    enc_matrix, ret_matrix = mat_lists

    enc_matrix = enc_matrix + enc_c
    ret_matrix = ret_matrix + ret_c
    cond_enc = conditional_prob_matrix(enc_matrix)
    cond_ret = conditional_prob_matrix(ret_matrix)

    all_objs = np.unique(np.concatenate((enc_objs, ret_objs)))
    enc_missing = set(all_objs) - set(enc_objs)
    ret_missing = set(all_objs) - set(ret_objs)

    # append missing to get same state space but out of order
    def append_missing(missing_set, objs, matrix):
        for obj_ in missing_set:
            objs = np.append(objs, obj_)
            matrix = np.concatenate([matrix, np.zeros([1, matrix.shape[1]])], axis=0)
            matrix = np.concatenate([matrix, np.zeros([matrix.shape[0], 1])], axis=1)
        return objs, matrix

    enc_objs, enc_matrix = append_missing(enc_missing, enc_objs, enc_matrix)
    ret_objs, ret_matrix = append_missing(ret_missing, ret_objs, ret_matrix)

    # reorder to match state spaces
    s = len(all_objs)

    enc_perm = []
    [enc_perm.append(j) for i in range(s) for j in range(s) if all_objs[i] == enc_objs[j]]
    ret_perm = []
    [ret_perm.append(j) for i in range(s) for j in range(s) if all_objs[i] == ret_objs[j]]
    # ret_perm = []
    # for i in range(s):
    #     all_obj = all_objs[i]
    #     for j in range(s):
    #         ret_obj = ret_objs[j]
    #         if all_obj == ret_obj:
    #             ret_perm.append(j)



    enc = enc_matrix[enc_perm, :][:, enc_perm]
    try:
        ret = ret_matrix[ret_perm, :][:, ret_perm]
    except IndexError as e:
        raise e

    return symmetric_KL(enc, ret)


def centroid_to_roi(fix, trial_row):
    fix = fix.reset_index(drop=True)
    pd.options.mode.chained_assignment = None  # default='warn'
    with warnings.catch_warnings():  # catch nan warnings
        warnings.simplefilter("ignore")
        for i in range(len(fix)):
            object_ = fix['object'][i]
            fix.centroid_y = 0.8    # normalise to table height for now
            if object_ == TaskObjects.invisible_object:
                fix.centroid_x[i] = trial_row.obj1_preshift_x[0]
                fix.centroid_z[i] = trial_row.obj1_preshift_z[0]
            elif object_ == 'Table':
                # table_roi = get_table_roi(fix.centroid_x, fix.centroid_z,
                #                           trial_row.table_location_x, trial_row.table_location_z)
                # fix.centroid_x, fix.centroid_z = table_roi.x, table_roi.z

                # table fixations normalised to centre, need to implement rotation around centre where necessary before
                # using fixation centroids i.e. if ret has rotated, reverse rotation on each fix centroid on table
                # to normalise fixations to encoding view
                fix.centroid_x[i], fix.centroid_z[i] = trial_row.table_location_x[0], trial_row.table_location_z[0]
                # pass    # keep Table fix locations the same
            elif object_ in TaskObjects.array_objects:
                for i in range(1, 5):
                    obj = f"obj{i}"
                    if object_ == trial_row[f"{obj}_name"][0]:
                        fix.centroid_x[i], fix.centroid_z[i] = trial_row[f"{obj}_preshift_x"][0], trial_row[f"{obj}_preshift_z"][0]

            else:
                print(f"object {object_} not accounted for")
                raise InvalidValue(object_, 'array_object',
                                   'invalid task object - check off-table objects are properly excluded')
    pd.options.mode.chained_assignment = 'warn'  # default='warn'
    return fix


def rotate_around_point(cx, cz, px, pz, angle_degrees):
    # Convert the angle to radians
    angle = math.radians(angle_degrees)

    # Translate the centroid point to the origin
    T1 = np.array([[1, 0, -cx],
                   [0, 1, -cz],
                   [0, 0, 1]])

    # Rotate around the point
    R = np.array([[math.cos(angle), -math.sin(angle), 0],
                  [math.sin(angle), math.cos(angle), 0],
                  [0, 0, 1]])
    P = np.array([[1, 0, px],
                  [0, 1, pz],
                  [0, 0, 1]])
    T2 = np.array([[1, 0, cx],
                   [0, 1, cz],
                   [0, 0, 1]])

    M = T2 @ P @ R @ T1

    # Apply the transformation to the centroid point
    centroid = np.array([[cx, cz, 1]])
    centroid_rotated = centroid @ M

    # Extract the rotated centroid point
    cx_rotated, cz_rotated, _ = centroid_rotated[0]

    return cx_rotated, cz_rotated


def centroid_rotate_norm(fix_df, trial_row):
    """fix_df should be retrieval/Presentation viewing"""
    fix_df = fix_df.reset_index(drop=True)
    pd.options.mode.chained_assignment = None  # default='warn'
    with warnings.catch_warnings():  # catch nan warnings
        warnings.simplefilter("ignore")
        anticlockwise = bool(trial_row.anticlockwise_move[0])
        table_rotates = bool(trial_row.table_rotates[0])
        table_xz = trial_row.table_location_x[0], trial_row.table_location_z[0]
        if not table_rotates:
            return fix_df
        rotation_angle = trial_row.viewing_angle[0]
        if not anticlockwise:
            rotation_angle = -rotation_angle
        for i in range(len(fix_df)):
            fix_df.centroid_x[i], fix_df.centroid_z[i] = rotate_around_point(fix_df.centroid_x[i],
                                                                             fix_df.centroid_z[i],
                                                                             *table_xz, rotation_angle)
    pd.options.mode.chained_assignment = 'warn'  # default='warn'
    return fix_df


def fixation_df_comparison_setup(fix_df, trial_row=None, comparison='trial_viewings'):

    if comparison == 'trial_viewings':
        trial_row = trial_row.reset_index(drop=True)
        trial_id = trial_row.trial_id[0]
        enc_id = f"{trial_id}_enc"
        ret_id = f"{trial_id}_ret"
        viewings = [enc_id, ret_id]
        # all_fix = fetch_fixations("all", viewing_id=viewings)
        view1_fix = fix_df[fix_df.viewing_id == enc_id]
        view2_fix = fix_df[fix_df.viewing_id == ret_id]
    elif comparison == 'algorithm':
        algorithms = np.unique(fix_df.algorithm)
        alg1, alg2 = algorithms[0:2]
        view1_fix = fix_df[fix_df.algorithm == alg1]
        view2_fix = fix_df[fix_df.algorithm == alg2]
    else:
        viewings = np.unique(fix_df.viewing_id)
        view1, view2 = viewings[0:2]
        view1_fix = fix_df[fix_df.viewing_id == view1]
        view1_fix = fix_df[fix_df.viewing_id == view2]

    if len(view1_fix) < 1 or len(view2_fix) < 1:
        return None, None

    view1_fix = external_fixations(view1_fix.reset_index(drop=True))
    view2_fix = external_fixations(view2_fix.reset_index(drop=True))

    return view1_fix, view2_fix


def ea_td(trial_row, fix_df, comparison='trial_viewings', algorithms=(None, None)):

    if fix_df is None:
        return np.nan
    if comparison == 'trial_viewings':
        trial_row = trial_row.reset_index(drop=True)
        trial_id = trial_row.trial_id[0]
        enc_id = f"{trial_id}_enc"
        ret_id = f"{trial_id}_ret"
        viewings = [enc_id, ret_id]
        # all_fix = fetch_fixations("all", viewing_id=viewings)
        view1_fix = external_fixations(fix_df[fix_df.viewing_id == enc_id].reset_index(drop=True))
        view2_fix = external_fixations(fix_df[fix_df.viewing_id == ret_id].reset_index(drop=True))
    elif comparison == 'algorithm':
        alg1, alg2 = algorithms
        view1_fix = external_fixations(fix_df[fix_df.algorithm == alg1].reset_index(drop=True))
        view2_fix = external_fixations(fix_df[fix_df.algorithm == alg2].reset_index(drop=True))


    if len(view1_fix) < 1 or len(view2_fix) < 1:
        return np.nan

    view1_start_time0 = view1_fix.start_time[0]
    view2_start_time0 = view2_fix.start_time[0]

    dimensions = ['centroid_x', 'centroid_z',  # 'centroid_y',
                  'duration_time', 'start_time']




    # scale dimensions
    # NOTE: IF CHANGING VIEWPOINT, SPACE DIMENSIONS MAY CHANGE VARIANCE, AND COLLISION IN 3D SPACE WILL BE MORE DIFFERENT
    # THAN IF SEEING OBJECTS FROM THE SAME VIEW
    # POTENTIAL SOLUTION: COULD NORMALISE FIXATIONS ON OBJECTS TO CENTRE OF OBJECT I.E. THE CENTROID OF A FIXATION ON AN OBJECT
    # IS REPLACED WITH THE CENTROID OF THAT OBJECT
    # TABLE FIXATION CENTROIDS REMAIN UNCHANGED
    # REMOVE FIXATIONS ON OBJECTS OUTSIDE OF TABLE - MAYBE LOG/ROCK CAN BE COUNTED AS THE SAME

    # PROBLEM: TABLE ROTATION WILL MEAN GREATER DISTANCE FOR ROTATION CONDITIONS
    # SOLUTION: X, Y, Z WILL BE DEFINED AS THE OBJECT'S view1ODING LOCATION FOR BOTH VIEWINGS
    # WILL NEED TO DEFINE ROI SYSTEM FOR TABLE - hexagonal grid probably best, perhaps the size of array object ROI
    # for hexagon: R = distance from centre to farthest point (corner), r = distance from centre to nearest edge point (midpoint of edge)
    # make r = radius of middle AOI

    # PROBLEM: THE OBJECT HAS MOVED! INVISIBLE OBJECT AT PREVIOUS POSITION MEANS THERE'S ANOTHER OBJECT IN THE MIX
    # SOLUTION1: FIXATIONS ON MOVED OBJECT AND INVISIBLE OBJECT (PREVIOUS POSITION) ARE EQUIVALENT TO MOVED OBJECT AT view1ODING
    #

    # remove fixations outside the table
    view1_fix = view1_fix[np.isin(view1_fix.object, TaskObjects.on_table)]
    view2_fix = view2_fix[np.isin(view2_fix.object, TaskObjects.on_table)]

    # replace array object fixation centroid with array object location at encoding
    view1_fix_norm = centroid_to_roi(view1_fix, trial_row)
    view1_fix_norm = view1_fix_norm.loc[:, dimensions]
    view2_fix_norm = centroid_to_roi(view2_fix, trial_row)
    view2_fix_norm = view2_fix_norm.loc[:, dimensions]

    if len(view1_fix_norm) < 1 or len(view2_fix_norm) < 1:
        return np.nan



    # min scale start time
    cat = np.concatenate([view1_fix_norm.start_time.to_numpy(), view2_fix_norm.start_time.to_numpy()])
    grand_std = np.std(cat)
    view1_start_time = (view1_fix_norm.start_time - view1_start_time0) / grand_std
    view2_start_time = (view2_fix_norm.start_time - view2_start_time0) / grand_std

    # z-scale duration
    cols = view1_fix_norm.columns
    view1_fix_duration = view1_fix_norm.duration_time.to_numpy().reshape(-1, 1)
    view1_fix_duration = StandardScaler().fit(view1_fix_duration).transform(view1_fix_duration)
    view1_fix_norm.duration_time = view1_fix_duration
    view2_fix_duration = view2_fix_norm.duration_time.to_numpy().reshape(-1, 1)
    view2_fix_duration = StandardScaler().fit(view2_fix_duration).transform(view2_fix_duration)
    view2_fix_norm.duration_time = view2_fix_duration

    # scale centroids by grand mean and grand std - keep same locations the same
    cat = np.concatenate([view1_fix_norm.centroid_x.to_numpy(), view2_fix_norm.centroid_x.to_numpy()])
    grand_std = np.std(cat)
    grand_mean = np.mean(cat)
    view1_fix_norm.centroid_x = (view1_fix_norm.centroid_x.to_numpy() - grand_mean) / grand_std
    view1_fix_norm.centroid_z = (view1_fix_norm.centroid_z.to_numpy() - grand_mean) / grand_std
    view2_fix_norm.centroid_x = (view2_fix_norm.centroid_x.to_numpy() - grand_mean) / grand_std
    view2_fix_norm.centroid_z = (view2_fix_norm.centroid_z.to_numpy() - grand_mean) / grand_std

    view1_fix_norm.start_time = view1_start_time
    view2_fix_norm.start_time = view2_start_time
    return float(eye_analysis(view1_fix_norm, view2_fix_norm))


def eye_analysis(s_fix, t_fix):
    """
    Finds nearest neighbour for each s_fix in t_fix (point-mapping) and same in reverse (double-mapping).
    Calculates this with summed euclidian distance along each column specified by dimensions.

    Eyeanalysis algorithm taken from original publication:

    Mathôt, S., Cristino, F., Gilchrist, I. D., & Theeuwes, J. (2012).
    A simple way to estimate similarity between pairs of eye movement sequences.
    Journal of Eye Movement Research, 5(1).
    :param s_fix:
    :param t_fix:
    :param dimensions:
    :return:
    """
    d=0
    for p in range(len(s_fix)):
        d += nearest_distance(s_fix.iloc[p, :], t_fix)
    for q in range(len(t_fix)):
        d += nearest_distance(t_fix.iloc[q, :], s_fix)
    return d / np.max([len(s_fix), len(t_fix)])


def nearest_distance(p, T):
    ds = []
    for i in range(len(T)):
        ds.append(euc_distance(p, T.iloc[i, :]))
    nd = np.min(ds)
    return nd


def euc_distance(p, q):
    d_sq = 0
    dimensions = q.axes[0]
    for d in dimensions:
        d_sq += (p[d] - q[d])**2
    return np.sqrt(d_sq)


def mm_string(trial_row, fix_df, comparison='trial_viewings', algorithms=(None, None)):
    mm_out = multimatch(trial_row, fix_df)
    # check if mm_out is iterable
    if isinstance(mm_out, Iterable):
        return json.dumps(list(mm_out))
    else:
        return json.dumps([mm_out])


def multimatch(trial_row, fix_df, comparison='trial_viewings', algorithms=(None, None)):
    trial_row = trial_row.reset_index(drop=True)
    if fix_df is None:
        return np.nan

    # get table x and z for normalisation
    table_x, table_z = trial_row.table_location_x[0], trial_row.table_location_z[0]

    # get fixation sequences to compare
    if comparison == 'trial_viewings':
        viewings = np.unique(fix_df.viewing_id)
        if len(viewings) < 2:
            return np.nan
        elif len(viewings) > 2:
            raise InvalidInput(message="should be exactly 2 viewings to compare fixations")
        view1_fix = external_fixations(fix_df[fix_df.viewing_id == viewings[0]].reset_index(drop=True))
        view2_fix = external_fixations(fix_df[fix_df.viewing_id == viewings[1]].reset_index(drop=True))

        # norm fixations to encoding - rotate retrieval fixations
        view2_fix = centroid_rotate_norm(view2_fix, trial_row)

    elif comparison == 'algorithm':
        algos = np.unique(fix_df.algorithm)
        if len(algos) < 2:
            raise InvalidInput(message="should be at least 2 algorithms to compare fixations")
        alg1, alg2 = algorithms
        view1_fix = external_fixations(fix_df[fix_df.algorithm == alg1].reset_index(drop=True))
        view2_fix = external_fixations(fix_df[fix_df.algorithm == alg2].reset_index(drop=True))

    # use x and z for now
    dimensions = ['centroid_x', 'centroid_z', 'duration_time']
    new_names = ['start_x', 'start_y', 'duration']

    # remove fixations outside of table
    view1_fix = view1_fix[np.isin(view1_fix.object, TaskObjects.on_table)]
    view2_fix = view2_fix[np.isin(view2_fix.object, TaskObjects.on_table)]

    # normalise position to 0
    view1_fix['centroid_x'] -= table_x
    view2_fix['centroid_x'] -= table_x
    view1_fix['centroid_z'] -= table_z
    view2_fix['centroid_z'] -= table_z

    # must be at least 2 fixations
    if len(view1_fix) < 2 or len(view2_fix) < 2:
        return np.nan

    # isolate relevant columns
    view1_fix = view1_fix[dimensions]
    view1_fix.columns = new_names
    view2_fix = view2_fix[dimensions]
    view2_fix.columns = new_names

    # use table as 'screen'
    pixel_scalar = 1000     # convert m to mm
    screen_size = [TaskObjects.lossy_scale_dict['Table'][idx]*pixel_scalar for idx in [0, 2]]
    view1_fix[new_names[0:2]] *= pixel_scalar
    view2_fix[new_names[0:2]] *= pixel_scalar

    # export to csvs
    cwd = os.getcwd()
    path1 = f"{cwd}\\view1.csv"
    path2 = f"{cwd}\\view2.csv"
    view1_fix.to_csv(path1, index=False)
    view2_fix.to_csv(path2, index=False)

    # multimatch
    # read in data
    # fix_vector1 = np.recfromcsv(path1, delimiter=',', dtype={'names': ('start_x', 'start_y', 'duration'),
    #                                                          'formats': ('f8', 'f8', 'f8')})
    df1 = pd.read_csv(path1, delimiter=',')
    df1 = df1.astype({'start_x': 'float64', 'start_y': 'float64', 'duration': 'float64'})
    fix_vector1 = df1.to_records(index=False)


    # fix_vector2 = np.recfromcsv(path2, delimiter=',', dtype={'names': ('start_x', 'start_y', 'duration'),
    #                                                          'formats': ('f8', 'f8', 'f8')})

    df2 = pd.read_csv(path2, delimiter=',')
    df2 = df2.astype({'start_x': 'float64', 'start_y': 'float64', 'duration': 'float64'})
    fix_vector2 = df2.to_records(index=False)
    try:
        mm_out = mm.docomparison(fix_vector1, fix_vector2, screensize=screen_size)
    except Exception as e:
        raise e


    return mm_out





