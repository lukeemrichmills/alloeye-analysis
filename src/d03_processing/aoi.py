import numpy as np
import pandas as pd
import math

from src.d00_utils.TaskObjects import TaskObjects
from src.d01_data.database.Errors import InvalidInput
from src.d01_data.database.PsqlCommander import PsqlCommander
from src.d01_data.database.ToSQL.ToSQL import ToSQL
from src.d01_data.database.db_connect import db_connect
from src.d01_data.fetch.fetch_timepoints import fetch_timepoints
from src.d01_data.fetch.fetch_trials import fetch_trials
from src.d01_data.fetch.fetch_viewings import fetch_viewings

from src.d03_processing.TimepointProcessor import TimepointProcessor

"""a series of functions to calculate or re-calculate areas of interests for gaze points and fixations in alloeye"""


def gaze_object_adjust(filtered_timepoints, invisible_obj_resize=0.12):
    """check if filtered timepoints would collide with a different object from unfiltered timepoints"""
    # shorten name
    f_tps = filtered_timepoints
    original_gaze_object = f_tps.gaze_object

    # get gaze objects from timepoints
    gaze_objects = np.unique(f_tps.gaze_object)
    table_objects = gaze_objects[np.isin(gaze_objects, TaskObjects.on_table)]
    array_objects = table_objects[table_objects != 'Table']     # without Table
    visible_array = array_objects[array_objects != TaskObjects.invisible_object]    # without invisible object

    # define dictionary for positions of objects
    position_dict = {}

    # is retrieval
    is_ret = f_tps.retrieval_epoch[0] != 'na'

    # bool for below
    invis_obj_accounted_for = TaskObjects.invisible_object in table_objects if is_ret else True

    # if all objects not accounted for, then get from trial info
    if not all(['Table' in table_objects, invis_obj_accounted_for, len(visible_array) == 4]):
        # fetch object locations from trial data
        trial_id = f_tps.trial_id[0]
        trial = fetch_trials("all", trial_ids=trial_id, remove_training_trials=False)
        trial = trial.reset_index(drop=True)

        # table position
        position_dict['Table'] = (trial.table_location_x[0], TaskObjects.tabletop, trial.table_location_z[0])

        # reset object arrays
        table_objects = ['Table']
        array_objects = []
        visible_array = []

        # if retrieval, set invisible object
        if is_ret:
            position_dict[TaskObjects.invisible_object] = (trial.co_preshift_x_raw[0], TaskObjects.tabletop,
                                                           trial.co_preshift_z_raw[0])
            table_objects.append(TaskObjects.invisible_object)
            array_objects.append(TaskObjects.invisible_object)
            pre_post = "post"
        else:
            pre_post = "pre"

        # set each visible array object
        for i in range(1, 5):
            obj = trial[f"obj{i}_name"][0]
            table_objects.append(obj)
            array_objects.append(obj)
            visible_array.append(obj)
            y = 0.73 if obj == '5Ball' else TaskObjects.tabletop
            position_dict[obj] = (trial[f"obj{i}_{pre_post}shift_x"][0], y,
                                  trial[f"obj{i}_{pre_post}shift_z"][0])
    else:
        # if all objects accounted for in gaze points, then use position data form timepoints
        for obj in table_objects:
            x = f_tps.object_position_x[f_tps.gaze_object == obj].reset_index(drop=True)[0]
            y = f_tps.object_position_y[f_tps.gaze_object == obj].reset_index(drop=True)[0]
            z = f_tps.object_position_z[f_tps.gaze_object == obj].reset_index(drop=True)[0]
            position_dict[obj] = (x, y, z)


    # refit aoi for visible and invisible aois (i.e. objects on table)
    new_misses = np.zeros(len(f_tps), dtype=int)   # initialise array to register new points that don't hit any aoi
    for obj in array_objects:
        if obj == TaskObjects.invisible_object:
            radius_override = invisible_obj_resize
        else:
            radius_override = []
        f_tps = retrofit_aoi(f_tps, obj, position_dict[obj],
                             only_if_missing=False, hit_bools=True, vector_scalar=2,
                             radius_override=radius_override)
        new_aoi_hit = f_tps.aoi_hit.to_numpy()
        new_miss = np.invert(new_aoi_hit) & np.array(f_tps.gaze_object == obj)
        new_misses = new_misses | new_miss
    f_tps['new_misses'] = new_misses

    # check if unaccounted for gaze points are on table
    x_bounds = (position_dict['Table'][0] - TaskObjects.lossy_scale_dict['Table'][0],
                position_dict['Table'][0] + TaskObjects.lossy_scale_dict['Table'][0])
    y_bounds = (TaskObjects.tabletop - 0.01, TaskObjects.tabletop + 0.5)
    z_bounds = (position_dict['Table'][2] - TaskObjects.lossy_scale_dict['Table'][2],
                position_dict['Table'][2] + TaskObjects.lossy_scale_dict['Table'][2])
    in_x_bounds = (x_bounds[0] < f_tps.gaze_collision_x) & (f_tps.gaze_collision_x < x_bounds[1])
    in_y_bounds = (y_bounds[0] < f_tps.gaze_collision_y) & (f_tps.gaze_collision_y < y_bounds[1])
    in_z_bounds = (z_bounds[0] < f_tps.gaze_collision_z) & (f_tps.gaze_collision_z < z_bounds[1])
    in_table_bounds = in_x_bounds & in_y_bounds & in_z_bounds
    new_table = (f_tps.new_misses == 1) & in_table_bounds
    pd.options.mode.chained_assignment = None
    f_tps.gaze_object[new_table] = 'Table'
    f_tps.object_position_x[new_table] = position_dict['Table'][0]
    f_tps.object_position_y[new_table] = position_dict['Table'][1]
    f_tps.object_position_z[new_table] = position_dict['Table'][2]

    # sort remaining into dome and floor, or leave
    new_off_table = (f_tps.new_misses == 1) & (np.invert(in_table_bounds))
    if np.sum(new_off_table) > 0:
        is_dome = (f_tps.gaze_collision_y > 1) & in_x_bounds & in_z_bounds & new_off_table
        f_tps.gaze_object[is_dome] = TaskObjects.dome
        f_tps.object_position_x[is_dome] = position_dict['Table'][0]
        f_tps.object_position_y[is_dome] = 3.0
        f_tps.object_position_z[is_dome] = position_dict['Table'][2]
        is_floor = (f_tps.gaze_collision_y < 0.01) & new_off_table
        f_tps.gaze_object[is_floor] = TaskObjects.floor
        f_tps.object_position_x[is_floor] = position_dict['Table'][0]
        f_tps.object_position_y[is_floor] = 0.0
        f_tps.object_position_z[is_floor] = position_dict['Table'][2]

    changes = f_tps.gaze_object != original_gaze_object
    # if any(changes):
    #     print(f"{np.sum(np.array(changes))} gaze objects changed")

    pd.options.mode.chained_assignment = 'warn'
    return f_tps


def retrofit_aoi_multi_trials(aoi_object, trial_list="all"):
    # download trials from trial list
    if trial_list == "all":
        trials = fetch_trials("all", remove_training_trials=False)
    else:
        trials = fetch_trials("all", trial_ids=trial_list, remove_training_trials=False)

    # filter by those where aoi_object in obj1-4 name
    bools = []
    for i in range(1, 5):
        b = trials[f"obj{i}_name"] == aoi_object
        bools.append(np.array(b))
    full_bool = bools[0] | bools[1] | bools[2] | bools[3]
    trials = trials[full_bool].reset_index(drop=True)



    # for each trial
    for i in range(len(trials)):
        trial = trials.trial_id[i]

        # get trial timepoints
        enc = f"{trial}_enc"
        ret = f"{trial}_ret"
        enc_ret = [enc, ret]
        timepoints = fetch_timepoints("all", viewing_id=enc_ret)

        # filter no timepoints
        if timepoints is None:
            print(f"no timepoints for {trial}")
            continue

        if len(timepoints) < 1:
            print(f"no timepoints for {trial}")
            continue

        # get aoi object position per viewing
        aoi_positions = []
        for j in range(1, 5):
            if trials[f"obj{j}_name"][i] == aoi_object:
                enc_x = trials[f"obj{j}_preshift_x"][i]
                enc_z = trials[f"obj{j}_preshift_z"][i]
                ret_x = trials[f"obj{j}_postshift_x"][i]
                ret_z = trials[f"obj{j}_postshift_z"][i]
                aoi_positions.append((enc_x, TaskObjects.tabletop, enc_z))
                aoi_positions.append((ret_x, TaskObjects.tabletop, ret_z))

        # full df for uploading
        all_tps = None

        # for each viewing
        for j in range(2):
            # isolate timepoints
            tps = timepoints[timepoints.viewing_id == enc_ret[j]]

            # filter no timepoints
            if timepoints is None:
                print(f"no timepoints for {trial}_{enc_ret[j]}")
                continue

            if len(timepoints) < 1:
                print(f"no timepoints for {trial}_{enc_ret[j]}")
                continue

            # retrofit aoi
            timepoints[timepoints.viewing_id == enc_ret[j]] = retrofit_aoi(tps, aoi_object, aoi_positions[j])

            # isolate aoi timepoints
            new_tps = timepoints[timepoints.gaze_object == aoi_object].reset_index(drop=True)

            # concatenate
            if j == 0:
                all_tps = new_tps
            else:
                all_tps = pd.concat([all_tps, new_tps], ignore_index=True)

        # Update relevant columns in db
        if all_tps is not None:
            update_cols = ['gaze_object', 'gaze_collision_x', 'gaze_collision_y', 'gaze_collision_z',
                           'object_position_x', 'object_position_y', 'object_position_z']
            to_sql = ToSQL(all_tps, "alloeye", update_cols)
            query = to_sql.convert_to_update_commands('timepoint_id', table_name='alloeye_timepoint_viewing')
            commands_one = "; ".join(query) if type(query) is list else query
            commander = PsqlCommander()
            connection = db_connect(suppress_print=True)
            commander.execute_query(connection, commands_one, None)

        # log progress
        if (i + 1) % 10 == 0 or (i + 1) == len(trials):
            print(f"Features added to df for {i + 1} of {len(trials)} trials")

    print("all new timepoints uploaded")
    pass


def retrofit_aoi(timepoints, aoi_object, aoi_object_position,
                 only_if_missing=True, hit_bools=False, vector_scalar=1, radius_override=[]):
    pd.options.mode.chained_assignment = None  # default='warn'

    #
    collision_bools = [] if hit_bools else None

    # create copy of df for debugging and testing
    tp_copy = timepoints.copy(deep=True)

    # check aoi_object not already there
    gaze_objects = timepoints.gaze_object.to_numpy()
    if only_if_missing and np.isin(aoi_object, gaze_objects):
        print(f"{aoi_object} already registered in gaze_object column, set only_if_missing=False if resetting")
        return timepoints

    # get centre of collision sphere by halving lossy scale y and adding to y position
    aoi_centre = (aoi_object_position[0],
                  aoi_object_position[1] + TaskObjects.collider_y_offset[aoi_object],   # variable offsets
                  aoi_object_position[2])

    # convert to numpy
    tps = timepoints.eye_timestamp_ms.to_numpy()
    point_matrix = TimepointProcessor.create_gaze_point_matrix(timepoints)
    cam_matrix = TimepointProcessor.create_matrix(timepoints, ['camera_x', 'camera_y', 'camera_z'], axis=1)
    obj_pos_mat = TimepointProcessor.create_matrix(timepoints, ['object_position_x', 'object_position_y', 'object_position_z'], axis=1)

    # get viewpoint position
    view_pos = viewpoint_position(cam_matrix)

    # loop through timepoints and convert gaze object to aoi if intersects
    for i in range(len(tps)):
        cam = cam_matrix[i, :]
        gp = point_matrix[i, :]
        radius = collision_sphere_radius(aoi_object_position, view_pos, radius_override)
        new_point = closest_intersection(cam, gp, aoi_centre, radius, extension_scalar=vector_scalar)
        if new_point is None:
            if hit_bools:
                collision_bools.append(0)
            continue
        else:
            if hit_bools:
                collision_bools.append(1)
            point_matrix[i, :] = new_point
            gaze_objects[i] = aoi_object
            obj_pos_mat[i, :] = aoi_object_position

    collision_bools = np.array(collision_bools)
    timepoints.gaze_object = gaze_objects
    timepoints.gaze_collision_x, timepoints.gaze_collision_y, timepoints.gaze_collision_z \
        = [point_matrix[:, col] for col in range(3)]
    timepoints.object_position_x, timepoints.object_position_y, timepoints.object_position_z \
        = [obj_pos_mat[:, col] for col in range(3)]

    if hit_bools:
        timepoints['aoi_hit'] = collision_bools

    pd.options.mode.chained_assignment = 'warn'  # default='warn'

    return timepoints


def viewpoint_position(cam_matrix=None, trial_id=None):
    if cam_matrix is not None:
        vp = np.mean(cam_matrix, axis=0)
    elif trial_id is not None:
        # get from trial info somewhere
        pass
    else:
        raise ValueError()
    return vp


def collision_sphere_radius(object_position, viewpoint_position, rad_override=[]):
    p1 = np.array([object_position[0], object_position[2]])
    p2 = np.array([viewpoint_position[0], viewpoint_position[2]])
    d = np.sqrt(np.sum((p1-p2)**2, axis=0)) #np.linalg.norm(p1 - p2)
    dc = TaskObjects.viewpoint_from_table_centre
    max_d = TaskObjects.array_obj_max_distance_from_table
    if rad_override != []:
        min_rad = rad_override
        max_rad = rad_override
    else:
        min_rad = TaskObjects.min_aoi_radius
        max_rad = TaskObjects.max_aoi_radius

    if d <= dc:
        return min_rad
    elif d >= max_d:
        return max_rad
    else:
        return min_rad + ((max_rad - min_rad) * ((d - dc) / max_d))


def closest_intersection(p1, p2, center, radius, extension_scalar=1):
    # Calculate the direction vector of the line
    direction_vector = p2 - p1

    # Extend the line by the specified distance
    p2_extended = p2 + extension_scalar * direction_vector

    # Calculate the intersections of the line and the sphere
    intersections = line_sphere_intersection(p1, p2_extended, center, radius)

    # If there are no intersections, return None
    if len(intersections) == 0:
        return None

    # Calculate the Euclidean distances between each intersection and p1
    distances = [np.linalg.norm(p1 - p) for p in intersections]

    # Find the index of the intersection closest to p1
    closest_index = np.argmin(distances)

    # Return the closest intersection
    return intersections[closest_index]


def line_sphere_intersection(p1, p2, center, radius):
    # Convert the points p1 and p2 to numpy arrays
    p1 = np.array(p1)
    p2 = np.array(p2)

    # Convert the center of the sphere to a numpy array
    center = np.array(center)

    # Calculate the direction vector of the line
    d = p2 - p1

    # Calculate the displacement vector from p1 to the center of the sphere
    f = p1 - center

    # Calculate the coefficients of the quadratic equation
    a = np.dot(d, d)
    b = 2 * np.dot(f, d)
    try:
        c = np.dot(f, f) - radius ** 2
    except TypeError as e:
        raise e


    # Calculate the discriminant of the quadratic equation
    discriminant = b ** 2 - 4 * a * c

    # If the discriminant is negative, there are no real solutions
    if discriminant < 0:
        return []

    # If the discriminant is zero, there is one real solution
    elif discriminant == 0:
        t = -b / (2 * a)
        return [p1 + t * d]

    # If the discriminant is positive, there are two real solutions
    else:
        sqrt_discriminant = np.sqrt(discriminant)
        t1 = (-b + sqrt_discriminant) / (2 * a)
        t2 = (-b - sqrt_discriminant) / (2 * a)

        # Check if both solutions are within the range of 0 <= t <= 1
        if 0 <= t1 <= 1 and 0 <= t2 <= 1:
            return [p1 + t1 * d, p1 + t2 * d]

        # Check if only one solution is within the range of 0 <= t <= 1
        elif 0 <= t1 <= 1:
            return [p1 + t1 * d]
        elif 0 <= t2 <= 1:
            return [p1 + t2 * d]

        # If both solutions are outside the range of 0 <= t <= 1, return an empty list
        else:
            return []


def calculate_distance(point1, point2):
    # Calculate Euclidean distance between two points
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def split_table_fixations_into_areas_of_interest(df, radius=0.2):
    centroid_x_col, centroid_y_col, fixation_col = 'centroid_x', 'centroid_z', 'object'
    # print(fixation_col)
    table_fixations = df[df[fixation_col] == 'Table'].copy(deep=True)
    # print(table_fixations.shape)
    areas_of_interest = []

    if len(table_fixations) == 0:
        return df

    current_area = {
        "centroid": (table_fixations.loc[table_fixations.index[0], centroid_x_col],
                     table_fixations.loc[table_fixations.index[0], centroid_y_col]),
        "fixations": [table_fixations.loc[table_fixations.index[0], fixation_col]],
        "indexes": [table_fixations.index[0]]
    }

    areas_of_interest.append(current_area)

    for i in table_fixations.index[1:]:
        is_inside_existing_area = False

        for area in areas_of_interest:
            centroid = area["centroid"]
            fixation = (table_fixations.loc[i, centroid_x_col], table_fixations.loc[i, centroid_y_col])

            if calculate_distance(fixation, centroid) <= radius:
                area["fixations"].append(table_fixations.loc[i, fixation_col])
                area["indexes"].append(i)
                is_inside_existing_area = True
                break

        if not is_inside_existing_area:
            # print(i)
            # print(table_fixations.index)
            new_area = {
                "centroid": (table_fixations.loc[i, centroid_x_col], table_fixations.loc[i, centroid_y_col]),
                "fixations": [table_fixations.loc[i, fixation_col]],
                "indexes": [i]
            }
            areas_of_interest.append(new_area)

    # Overwrite the original column with the new subregion area of interest
    # df[fixation_col] = df[fixation_col].where(df[fixation_col] != 'Table', "")
    for i in range(len(areas_of_interest) - 1, -1, -1):
        area = areas_of_interest[i]
        new_obj_name = f"Table_{i + 1}"

        # print(i)
        df.loc[df.index.isin(area["indexes"]), fixation_col] = new_obj_name

    return df


def convert_AOIs(fix_df, trial=None, prepostshift=None, encoder=False, object_column='object'):

    if len(fix_df) < 1:
        return fix_df

    # get trial row if none
    if trial is None:
        trial_id = fix_df.trial_id.values[0]
        trial = fetch_trials("all", trial_ids=[trial_id], suppress_print=True, remove_training_trials=False,
                             practice=[False, True])

    # get viewing type
    if prepostshift is None:
        prepostshift = 'preshift' if 'enc' in fix_df.viewing_id.values[0] else 'postshift'

    new_objects = fix_df[object_column].copy()
    pd.options.mode.chained_assignment = None  # default='warn'

    # change names of objects to set AOI names
    new_objects[new_objects.isin(TaskObjects.off_table)] = 'External'
    new_objects[new_objects.str.contains('Table')] = 'Table'
    new_objects[new_objects == TaskObjects.invisible_object] = 'Previous'
    new_objects[new_objects == trial.obj1_name.values[0]] = 'Moved'

    # for objs 2 to 4, change name according to distance from moved object
    obj1_loc = np.array([trial[f'obj1_{prepostshift}_x'].values[0], trial[f'obj1_{prepostshift}_z'].values[0]])
    objs2_to_4 = np.unique(new_objects[(new_objects != 'Moved') & new_objects.isin(TaskObjects.array_objects)])
    objs2_to_4_distance_dict = {'object': [], 'distance': []}
    for obj in objs2_to_4:
        objs2_to_4_distance_dict['object'].append(obj)
        obj_loc = get_object_position(obj, trial, prepostshift)
        objs2_to_4_distance_dict['distance'].append(np.linalg.norm(obj_loc - obj1_loc))
    objs_df = pd.DataFrame(objs2_to_4_distance_dict)
    objs_df = objs_df.sort_values(by='distance').reset_index(drop=True)
    for obj in objs2_to_4:
        new_objects[new_objects == obj] = f"Obj{objs_df.index.values[objs_df.object == obj][0] + 2}"

    # optionally convert to integer labels
    if encoder:
        new_objects = aoi_label_encoder(new_objects)
    fix_df[object_column] = new_objects
    return fix_df


def get_object_position(obj, trial, prepostshift):
    for i in [1, 2, 3, 4]:
        # If any matches were found, return the corresponding x and z values
        if obj == trial[f'obj{i}_name'].values[0]:
            return np.array([trial[f'obj{i}_{prepostshift}_x'].values[0], trial[f'obj{i}_{prepostshift}_z'].values[0]])


def aoi_label_encoder(fixation_sequence):
    # Define your custom mapping
    label_mapping = {"Previous": 0, "Moved": 1, "Obj2": 2, "Obj3": 3, "Obj4": 4, "Table": 5, "External": 6}

    # Encode your labels using the mapping
    encoded_labels = [label_mapping[label] for label in fixation_sequence]

    return encoded_labels