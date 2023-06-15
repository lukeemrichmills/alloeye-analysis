import numpy as np
import pandas as pd

from src.d00_utils.TaskObjects import TaskObjects
from src.d01_data.fetch.fetch_trials import fetch_trials
from src.d03_processing.TimepointProcessor import TimepointProcessor
from src.d03_processing.fixations.FixAlgos import fix_algo_dict
from src.d03_processing.fixations.FixationProcessor import FixationProcessor

def input_invalid(timepoints, trial):

    if timepoints is None:
        return True
    if trial is None:
        return True
    if len(trial) < 1 or len(timepoints) < 1:
        return True

    return False


def average_distance_obj1(timepoints, trial):

    if input_invalid(timepoints, trial):
        return np.nan
    timepoints = timepoints.reset_index(drop=True)
    trial = trial.reset_index(drop=True)
    viewing_id = timepoints.viewing_id[0]

    prepostshift = 'preshift' if 'enc' in viewing_id else 'postshift'
    # print([col for col in trial.columns.to_list()])
    obj1_name = trial.obj1_name[0]
    obj1_point = np.array([trial[f"obj1_{prepostshift}_x"][0],
                           TaskObjects.collider_y_offset[obj1_name] + 0.7,
                           trial[f"obj1_{prepostshift}_x"][0]])

    distances_obj1 = gaze_distances_from_point(obj1_point, timepoints, cutoff=1.0)
    av_dist = np.nanmedian(distances_obj1)
    return av_dist


def average_distance_pp(timepoints, trial):
    if input_invalid(timepoints, trial):
        return np.nan
    timepoints = timepoints.reset_index(drop=True)
    trial = trial.reset_index(drop=True)
    viewing_id = timepoints.viewing_id[0]

    if 'enc' in viewing_id:
        return np.nan
    # print([col for col in trial.columns.to_list()])
    obj1_name = trial.obj1_name[0]
    pp_point = np.array([trial[f"obj1_preshift_x"][0],
                         TaskObjects.collider_y_offset[obj1_name] + 0.7,
                         trial[f"obj1_preshift_x"][0]])

    distances_pp = gaze_distances_from_point(pp_point, timepoints, cutoff=0.1)
    av_dist = np.nanmedian(distances_pp)
    return av_dist


def average_distance_border(timepoints, trial):
    if input_invalid(timepoints, trial):
        return np.nan

    timepoints = timepoints.reset_index(drop=True)
    trial = trial.reset_index(drop=True)
    distances_border = get_distances_from_border(timepoints, trial)
    return np.nanmedian(distances_border)


def gaze_distances_from_point(from_this_point, timepoints, cutoff=None):
    # filter and sort timepoints
    tps = timepoints.sort_values(by='eye_timestamp_ms').reset_index(drop=True)

    # convert gaze point and head points to numpy
    point_mat = TimepointProcessor.create_gaze_point_matrix(tps)
    head_mat = TimepointProcessor.create_head_loc_matrix(tps)

    # get mean head point for projecting to 2d
    mean_head = np.mean(head_mat, axis=0)

    # project gaze and mean head point to 2d
    # single points don't project? workaround
    stacked_points = np.concatenate([from_this_point.reshape(1, 3), point_mat], axis=0)
    proj_points = FixationProcessor.head_project(stacked_points, mean_head, use_headlocmat_as_mean=True)
    proj_point = proj_points[0, :]
    proj_tps = proj_points[1:, :]
    proj_point_old = FixationProcessor.head_project(from_this_point.reshape(1, 3), mean_head, use_headlocmat_as_mean=True)
    proj_tps_old = FixationProcessor.head_project(point_mat, mean_head, use_headlocmat_as_mean=True)

    # calculated distance between projected gaze points and projected reference point
    distances = np.sqrt(np.sum((proj_tps - proj_point) ** 2, axis=1))
    if cutoff is not None:
        distances = distances[distances < cutoff]  # apply cutoff


    return distances


def get_distances_from_border(timepoints, trial, tabletop_radius=0.55, border_thickness=0.12, cutoff=1.0):
    # Adjust the tabletop_radius
    tabletop_radius_adjusted_in = tabletop_radius - border_thickness
    tabletop_radius_adjusted_out = tabletop_radius

    # get the tabletop centre from trial info
    trial = trial.reset_index(drop=True)
    table_top_center = np.array([trial.table_location_x[0], 0.7, trial.table_location_z[0]])

    # filter and sort timepoints
    tps = timepoints.sort_values(by='eye_timestamp_ms').reset_index(drop=True)

    # convert gaze point and head points to numpy
    point_mat = TimepointProcessor.create_gaze_point_matrix(tps)
    not_nans = ~np.isnan(point_mat).any(axis=1)
    point_mat = point_mat[not_nans]
    head_mat = TimepointProcessor.create_head_loc_matrix(tps)
    not_nans = ~np.isnan(head_mat).any(axis=1)
    head_mat = head_mat[not_nans]
    # print(head_mat)

    # get mean head point for projecting to 2d
    mean_head = np.mean(head_mat, axis=0)

    # project gaze and mean head point to 2d
    proj_table = FixationProcessor.head_project(table_top_center.reshape(1, -1), mean_head, use_headlocmat_as_mean=True)
    proj_tps = FixationProcessor.head_project(point_mat, mean_head, use_headlocmat_as_mean=True)
    # print(proj_tps)
    # # Calculate the projection of gaze points on tabletop plane
    # projected_points = np.copy(gaze_points)
    # projected_points[:, 2] = tabletop_center[2] # The z-coordinate of the projected points is the same as the tabletop center

    # Calculate the distance from the projected points to the center of the table
    distance_to_center = np.linalg.norm(proj_tps - proj_table, axis=1)
    # print(distance_to_center)

    # Create the array for closest points on the tabletop border
    closest_points_on_border = np.copy(proj_tps)

    # If the distance is less than the inner adjusted tabletop's radius
    mask_in = distance_to_center < tabletop_radius_adjusted_in
    closest_points_on_border[mask_in] = proj_table + (tabletop_radius_adjusted_in / distance_to_center[mask_in])[:,
                                                     None] * (proj_tps[mask_in] - proj_table)

    # If the distance is greater than the outer adjusted tabletop's radius
    mask_out = distance_to_center > tabletop_radius_adjusted_out
    closest_points_on_border[mask_out] = proj_table + (tabletop_radius_adjusted_out / distance_to_center[mask_out])[:,
                                                      None] * (proj_tps[mask_out] - proj_table)

    # If the distance is between the inner and outer adjusted tabletop's radius, the gaze point is already within the border
    mask_border = ~(mask_in | mask_out)
    closest_points_on_border[mask_border] = proj_tps[mask_border]
    # print(proj_tps)
    # print(closest_points_on_border)
    # Calculate the distance from the gaze point to the closest point on the tabletop border
    distances = np.linalg.norm(proj_tps - closest_points_on_border, axis=1)
    distances = distances[distances < cutoff]  # apply cutoff

    return distances


def get_projected_centroid(timepoints, trial, prepostshift):

    # get mean head point for projecting to 2d
    head_mat = TimepointProcessor.create_head_loc_matrix(timepoints)
    mean_head = np.mean(head_mat, axis=0)

    # get object positions and project to 2d
    proj_points = []
    for i in [1, 2, 3, 4]:
        obj = trial[f"obj{i}_name"].values[0]
        y_offset = TaskObjects.collider_y_offset[obj]
        point = np.array([trial[f"obj{i}_{prepostshift}_x"].values[0], 0.7 + y_offset, trial[f"obj{i}_{prepostshift}_z"].values[0]])
        proj_point = FixationProcessor.head_project(point.reshape(1, -1), mean_head, use_headlocmat_as_mean=True)
        proj_points.append(proj_point)

    # calculate centroid
    proj_points = np.array(proj_points)
    centroid_2d = np.mean(proj_points, axis=0)
    # print("centroid", centroid_2d)

    return centroid_2d


def get_centroid(timepoints, trial, prepostshift):

    # get mean head point for projecting to 2d
    head_mat = TimepointProcessor.create_head_loc_matrix(timepoints)
    mean_head = np.mean(head_mat, axis=0)

    # get object positions and project to 2d
    points = []
    for i in [1, 2, 3, 4]:
        obj = trial[f"obj{i}_name"].values[0]
        y_offset = TaskObjects.collider_y_offset[obj]
        point = np.array([trial[f"obj{i}_{prepostshift}_x"].values[0], 0.7 + y_offset, trial[f"obj{i}_{prepostshift}_z"].values[0]])
        points.append(point)

    # calculate centroid
    centroid_3d = np.mean(points, axis=0)
    # print("centroid", centroid_2d)

    return centroid_3d


def sum_gauss_duration_centroid(timepoints, trial):
    if input_invalid(timepoints, trial):
        return np.nan
    timepoints = timepoints.reset_index(drop=True)
    trial = trial.reset_index(drop=True)

    return np.nansum(gaze_duration_gaussian_centroid(timepoints, trial))


def sum_gauss_duration_pp(timepoints, trial):
    if input_invalid(timepoints, trial):
        return np.nan
    timepoints = timepoints.reset_index(drop=True)
    trial = trial.reset_index(drop=True)
    viewing_id = timepoints.viewing_id.values[0]
    if 'enc' in viewing_id:
        return np.nan

    return np.nansum(gaze_duration_gaussian_pp(timepoints, trial))


def gaze_duration_gaussian_setup(timepoints, trial=None):
    # get trial row if none
    if trial is None:
        trial_id = timepoints.trial_id.values[0]
        trial = fetch_trials("all", trial_ids=[trial_id], suppress_print=True, remove_training_trials=False,
                             practice=[False, True])
    viewing_id = timepoints.viewing_id.values[0]
    prepostshift = 'preshift' if 'enc' in viewing_id else 'postshift'

    return timepoints, trial, prepostshift


def gaze_duration_gaussian_centroid(timepoints, trial=None, c=0.1, fix_algo='GazeCollision'):
    timepoints, trial, prepostshift = gaze_duration_gaussian_setup(timepoints, trial)
    # centroid = get_projected_centroid(timepoints, trial, prepostshift)
    centroid = get_centroid(timepoints, trial, prepostshift)
    return gaze_duration_gaussian_point(timepoints, centroid, c=c, fix_algo=fix_algo)


def gaze_duration_gaussian_pp(timepoints, trial=None, c=0.1, fix_algo='GazeCollision'):
    timepoints, trial, prepostshift = gaze_duration_gaussian_setup(timepoints, trial)

    if prepostshift == 'preshift':
        return None

    # get previous position location
    obj1_name = trial.obj1_name[0]
    point_pp = np.array([trial.obj1_preshift_x[0],
                         TaskObjects.collider_y_offset[obj1_name] + 0.7,
                         trial.obj1_preshift_z[0]])

    return gaze_duration_gaussian_point(timepoints, point_pp, c=c, fix_algo=fix_algo)


def gaze_duration_gaussian_point(timepoints, point, c=0.1, fix_algo='GazeCollision'):
    if 'fixation' not in timepoints.columns:
        timepoints['fixation'] = fix_algo_dict()[fix_algo](timepoints).timepoints.fixation
    tps = timepoints.copy()[timepoints.fixation == 1].sort_values(by='eye_timestamp_ms')
    return gaze_duration_gaussian(tps, point, c=c)


def gaze_duration_gaussian(tps, point, c=0.1):
    tps = tps.copy()

    distances = gaze_distances_from_point(point, tps, cutoff=None)  # set cutoff to None to avoid using
    t_diffs = np.append(0, np.diff(tps.eye_timestamp_ms.to_numpy()))

    mask = pd.isna(distances)
    distances = distances[~mask]
    t_diffs = t_diffs[~mask]

    gauss_ = gaussian_score(distances, a=t_diffs, c=c)
    return gauss_


def gaussian_score(x, a=1, b=0, c=1.0):
    return a * np.exp(-((x-b)**2) / (2 * c**2))


def distances_from_point(tps, point):
    distances = np.sqrt(np.sum((tps - point)**2, axis=1))
    return distances
