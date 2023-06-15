import warnings

import numpy as np
import pandas
import pandas as pd

from src.d00_utils.TaskObjects import TaskObjects
from src.d00_utils.generic_tools import del_multiple
from src.d01_data.database.Errors import InvalidValue, UnmatchingValues
from src.d01_data.database.Tables import Tables
from src.d01_data.database.ToSQL.ToSQL import ToSQL
from src.d03_processing.TimepointProcessor import TimepointProcessor
from sklearn.decomposition import PCA


class FixationProcessor(TimepointProcessor):
    def __init__(self, timepoints):
        self.fixation_threshold = 75  # ms
        super().__init__(timepoints)
        self.skip_algo = False

    def get_fixations_missing_split(self, timepoints):
        timepoints[timepoints.missing == 1] = np.nan
        timepoints['group_no'] = timepoints.isnull().all(axis=1).cumsum()
        timepoints = timepoints[np.invert(pd.isna(timepoints.missing))]
        df_list = [group for _, group in timepoints.groupby('group_no') if not group.empty]
        fixation_df = None
        for i in range(len(df_list)):
            tps = df_list[i].reset_index(drop=True)
            fix_df, tps = self.get_fixations(tps, missing_split_group_id=i)
            if i == 0:
                fixation_df = fix_df
                timepoints = tps
            else:
                fixation_df = pd.concat([fixation_df, fix_df]).reset_index(drop=True)
                timepoints = pd.concat([timepoints, tps]).reset_index(drop=True)
        fixation_df = fixation_df[fixation_df.duration_time > 0].reset_index(drop=True)
        if any(pd.isna(fixation_df.missing_split_group)):
            print("catch")
        return fixation_df, timepoints

    def get_fixations(self, timepoints, missing_split_group_id=0):
        self.timepoints = self.timepoints if timepoints is None else self.check_timepoints(timepoints)

    def post_process_fixations(self, fix_df):
        """remove fixations"""

    def get_saccades(self):
        pass

    def convert_fix_df_format(self, tp_fix_df, missing_split_group_id):
        """ converts vr_idt output to correct format for this including processing velocity, dispersion etc."""

        fix_df_dict = {}
        for col_name in Tables.table_columns['fixations_saccades']:
            fix_df_dict[col_name] = []

        # remove id column names (except viewing), these are added later
        for col_name in Tables.table_columns['fixations_saccades']:
            if '_id' in col_name and col_name != 'viewing_id':
                del fix_df_dict[col_name]

        # fixations
        fix_df_dict['start_time'] = tp_fix_df.eye_timestamp_ms[tp_fix_df.fixation_start == 1].tolist()
        fix_df_dict['start_frame'] = tp_fix_df.eye_frame_number[tp_fix_df.fixation_start == 1].tolist()
        fix_df_dict['end_time'] = tp_fix_df.eye_timestamp_ms[tp_fix_df.fixation_end == 1].tolist()
        fix_df_dict['end_frame'] = tp_fix_df.eye_frame_number[tp_fix_df.fixation_end == 1].tolist()
        n_fix = len(fix_df_dict['start_time'])
        fix_df_dict['fixation_or_saccade'] = list(np.repeat('fixation', n_fix))


        # saccades

        def sacc_start_bools(tp_fix_df):
            """fixation end + 1 defines saccade start, need to account for beginning and end"""
            bools = list((tp_fix_df.fixation_end == 1) & (tp_fix_df.fixation_start != 1))
            start_bool = not bools[0]
            bools_out = bools
            bools_out[0] = start_bool
            return bools_out

        def sacc_end_bools(tp_fix_df):
            bools = list((tp_fix_df.fixation_start == 1) & (tp_fix_df.fixation_end != 1))
            end_bool = not bools[-1]
            bools_out = bools
            bools_out[-1] = end_bool
            return bools_out
                # [a and b for a, b in zip(bools_out, list(tp_fix_df.fixation_end != 1))]

        start_time_sacc = tp_fix_df.eye_timestamp_ms[sacc_start_bools(tp_fix_df)].tolist()
        n_sacc = len(start_time_sacc)
        fix_df_dict['fixation_or_saccade'].extend(np.repeat('saccade', n_sacc))
        fix_df_dict['start_time'].extend(list(start_time_sacc))
        fix_df_dict['start_frame'].extend(tp_fix_df.eye_frame_number[sacc_start_bools(tp_fix_df)].tolist())
        fix_df_dict['end_time'].extend(tp_fix_df.eye_timestamp_ms[sacc_end_bools(tp_fix_df)].tolist())
        fix_df_dict['end_frame'].extend(tp_fix_df.eye_frame_number[sacc_end_bools(tp_fix_df)].tolist())
        fix_df_dict['duration_time'] = list(np.array(fix_df_dict['end_time']) - np.array(fix_df_dict['start_time']))
        fix_df_dict['duration_frame'] = list(np.array(fix_df_dict['end_frame']) - np.array(fix_df_dict['start_frame']))

        # sames
        fix_df_dict['viewing_id'] = np.repeat(tp_fix_df.viewing_id[0], len(fix_df_dict['start_time']))
        fix_df_dict['algorithm'] = np.repeat(self.method_name, len(fix_df_dict['start_time']))


        for i in range(len(fix_df_dict['start_time'])):
            start_time = fix_df_dict['start_time'][i]
            end_time = fix_df_dict['end_time'][i]
            tps = tp_fix_df.loc[(tp_fix_df['eye_timestamp_ms'] >= start_time) &
                                (tp_fix_df['eye_timestamp_ms'] <= end_time), :]


            # gaze objects and proportions
            output = FixationProcessor.extract_gaze_objects(tps)
            if output is None:
                print("catch")
            for key, value in output.items():
                fix_df_dict[key].append(value)


            if len(tps) < 2:
                fix_df_dict['centroid_x'].append(np.nan)
                fix_df_dict['centroid_y'].append(np.nan)
                fix_df_dict['centroid_z'].append(np.nan)
                fix_df_dict['invalid_duration'].append(np.nan)
                fix_df_dict['dispersion'].append(np.nan)
                fix_df_dict['mean_velocity'].append(np.nan)
                fix_df_dict['max_velocity'].append(np.nan)
                fix_df_dict['mean_acceleration'].append(np.nan)
                continue


            # centroid coords
            centroid_x = np.mean(tps.gaze_collision_x)
            centroid_y = np.mean(tps.gaze_collision_y)
            centroid_z = np.mean(tps.gaze_collision_z)
            fix_df_dict['centroid_x'].append(centroid_x)
            fix_df_dict['centroid_y'].append(centroid_y)
            fix_df_dict['centroid_z'].append(centroid_z)

            # invalid duration
            fix_df_dict['invalid_duration'].append(FixationProcessor.invalid_duration(tps))

            # dispersion

            with warnings.catch_warnings():  # catch nan warnings
                warnings.simplefilter("ignore", category=RuntimeWarning)
                fix_df_dict['dispersion'].append(FixationProcessor.dispersion(tps))

            # velocity
            v = FixationProcessor.velocity_vector(tps)
            fix_df_dict['mean_velocity'].append(np.mean(v))
            fix_df_dict['max_velocity'].append(np.max(v))
            # acceleration
            if len(tps) < 3:
                fix_df_dict['mean_acceleration'].append(np.nan)
            else:
                fix_df_dict['mean_acceleration'].append(np.mean(FixationProcessor.acceleration_vector(tps)))
        fix_df_dict['missing_split_group'] = list(np.repeat(missing_split_group_id, len(fix_df_dict['viewing_id'])))
        fix_df = pd.DataFrame(fix_df_dict).sort_values(by='start_time').reset_index(drop=True)

        return fix_df

    def check_fixation_df(self, df=None):
        df = self.df if df is None else df

        column_names = Tables.table_columns['fixations_saccades']
        pass


    @staticmethod
    def acceleration(v1, v2, t1, t2):
        return (v2 - v1) / (t2 - t1)

    @staticmethod
    def velocity(timepoint_1, timepoint_2):
        s = FixationProcessor.displacement(timepoint_1, timepoint_2)
        v = s / (timepoint_2.eye_timestamp_ms - timepoint_1.eye_timestamp_ms)
        return v

    @staticmethod
    def displacement(t1, t2):
        return np.sqrt((t2.gaze_collision_x - t1.gaze_collision_x) ** 2 +
                       (t2.gaze_collision_y - t1.gaze_collision_y) ** 2 +
                       (t2.gaze_collision_z - t1.gaze_collision_z) ** 2)

    @staticmethod
    def displacement_vector(point_matrix):
        return np.sqrt(np.apply_over_axes(np.sum, np.diff(point_matrix, axis=0)**2, [1])).flatten()

    @staticmethod
    def angular_velocity_vec(tps):
        point_matrix = TimepointProcessor.create_gaze_point_matrix(tps)
        head_loc_mat = TimepointProcessor.create_head_loc_matrix(tps)
        vectors = point_matrix - head_loc_mat
        angles = []
        for i in range(1, len(vectors)):
            angles.append(FixationProcessor.angle_between(vectors[i - 1], vectors[i]))
        angles = np.array(angles)
        time_diff = np.diff(tps.eye_timestamp_ms.to_numpy())
        v_ang = angles / time_diff
        return v_ang

    @staticmethod
    def velocity_vector(tps):
        point_matrix = tps.loc[:, ['gaze_collision_x', 'gaze_collision_y', 'gaze_collision_z']].to_numpy()
        time_vector = np.diff(tps.eye_timestamp_ms)
        displacement_vector = FixationProcessor.displacement_vector(point_matrix)
        out = list(displacement_vector / time_vector)
        if np.sum(np.isinf(out) > 0):
            out[np.isinf(out)] = 0
            print("infinite values converted to 0")
        return out

    @staticmethod
    def acceleration_vector(tps):
        time_vector = np.diff(tps.eye_timestamp_ms)[1:]
        delta_v_vector = np.diff(FixationProcessor.velocity_vector(tps))
        out = list(delta_v_vector / time_vector)
        if np.sum(np.isinf(out) > 0):
            out[np.isinf(out)] = 0
            print("infinite values converted to 0")

        return out

    @staticmethod
    def dispersion(tps) -> float:
        """ root mean square deviation from mean (centroid) from 2d projected points"""
        points_3d = TimepointProcessor.create_gaze_point_matrix(tps)
        cam_loc_matrix = TimepointProcessor.create_matrix(tps, ['camera_x', 'camera_y', 'camera_z'], axis=1)
        cam_rot_matrix = TimepointProcessor.create_matrix(tps, ['cam_rotation_x', 'cam_rotation_y', 'cam_rotation_z'], axis=1)

        point_matrix = FixationProcessor.head_project(points_3d, cam_loc_matrix)
        centroid = np.apply_over_axes(np.mean, point_matrix, [0])
        centered = point_matrix - centroid
        cov_matrix = np.cov(centered, rowvar=False)
        stddev = np.sqrt(np.diag(cov_matrix))
        rmsd = np.linalg.norm(stddev)
        return rmsd

    @staticmethod   # taken from vr_idt
    def angle_between(v1: np.array, v2: np.array) -> float:
        """Compute the angle theta between vectors v1 and v2.

        The scalar product of v1 and v2 is defined as:
          dot(v1,v2) = mag(v1) * mag(v2) * cos(theta)

        where dot() is a function which computes the dot product and mag()
        is a function which computes the magnitude of the given vector.

        Args:
            v1: vector with dim (m x n)
            v2: with dim (m x n)

        Returns:
            theta: angle between vectors v1 and v2 in degrees.
        """
        norms = np.linalg.norm(v1) * np.linalg.norm(v2)
        cos_theta = np.dot(v1, v2) / norms
        with warnings.catch_warnings():  # catch nan warnings
            warnings.simplefilter("ignore", category=DeprecationWarning)
            theta = np.arccos(np.clip(cos_theta, -1, 1))
        return np.rad2deg(theta)

    @staticmethod
    def invalid_duration(tps):
        diff = [0]
        diff.extend(list(np.diff(tps.eye_timestamp_ms)))
        diff = np.array(diff)
        out = np.sum(diff[tps.missing.to_numpy() == 1])
        return out


    @staticmethod
    def valid_eye_open(row):
        return row.left_pupil_diameter != -1 and row.right_pupil_diameter != -1

    @staticmethod
    def extract_gaze_objects(tps):

        gaze_objects = np.unique(tps.gaze_object)

        if len(gaze_objects) == 1:
            output = {
                'object': gaze_objects[0],
                'gaze_object_proportion': 1,
                'second_gaze_object': "",
                'second_gaze_object_proportion': 0,
                'other_gaze_object_proportion': 0
            }
            return output
        elif len(gaze_objects) > 1:
            gaze_obj_counts = np.zeros(len(gaze_objects)).tolist()
            for i in range(len(gaze_objects)):
                gaze_obj_counts[i] = np.count_nonzero(tps.gaze_object[tps.gaze_object == gaze_objects[i]])
            max_count = np.max(gaze_obj_counts)
            count_sum = np.sum(gaze_obj_counts)
            n_max = [1 if i == max_count else 0 for i in gaze_obj_counts]
            if np.sum(n_max) > 1:  # tied for max
                indices = np.where(gaze_obj_counts == max_count)[0]
                gaze_object_1 = gaze_objects[indices[0]]
                gaze_object_2 = gaze_objects[indices[1]]
                proportion = max_count / count_sum
                second_proportion = proportion
                without_second = list(filter(lambda obj_count: obj_count < max_count, gaze_obj_counts))
            else:
                gaze_object_1 = gaze_objects[np.argmax(gaze_obj_counts)]
                without_max = list(filter(lambda obj_count: obj_count < max_count, gaze_obj_counts))
                second_max = np.max(without_max)
                gaze_object_2 = gaze_objects[np.argmax(without_max)]
                proportion = max_count / count_sum
                second_proportion = second_max / count_sum
                without_second = list(filter(lambda obj_count: obj_count < max_count, gaze_obj_counts))

            output = {'object': gaze_object_1,
                      'gaze_object_proportion': proportion,
                      'second_gaze_object': gaze_object_2,
                      'second_gaze_object_proportion': second_proportion}

            if len(without_second) > 0:
                output['other_gaze_object_proportion'] = np.sum(without_second) / count_sum
            else:
                output['other_gaze_object_proportion'] = 0

            return output

        elif len(gaze_objects == 0):
            raise InvalidValue(0, 1, "should be at least one gaze object")

    @staticmethod
    def head_project(point_matrix, head_loc_matrix, use_head_rot=False, head_rot_matrix=None,
                     output_dim='3d', use_headlocmat_as_mean=False, hard_mean_point= None, hard_norm=None):
        """
        BUG IN THIS CODE - HEAD PROJECT DOESN'T WORK FOR SINGLE POINT INPUT
        method to project 3d world gaze point to 2d plane with origin at head (camera) location
        and normal as head direction (camera rotation). Can use the output to calculate distance,
        velocity etc.
        :param point_matrix: numpy nd array of xzy gaze world locations n x 3
        :param head_loc_matrix: numpy nd array of xzy head world locations n x 3
        :param head_rot_matrix: numpy nd array of xzy head rotation euler angles as captured by unity n x 3
        :return: numpy nd array of xzy projected points
        """
        point_matrix = point_matrix.copy()
        proj_mat = np.zeros(np.shape(point_matrix))
        proj_mat_2d = np.zeros((np.shape(point_matrix)[0], 2))
        length = np.shape(point_matrix)[0]
        if np.max(head_loc_matrix.shape) == 3 and use_headlocmat_as_mean is True:
            mean_head_loc = head_loc_matrix.copy()
        elif use_headlocmat_as_mean is True:
            message = f"if using head_loc_matrix as mean needs to b 3x1 array input instead of {head_loc_matrix.shape}"
            raise InvalidValue(head_loc_matrix.shape, 3, message)
        else:
            mean_head_loc = np.mean(head_loc_matrix, axis=0)

        # define norm
        if use_head_rot:
            if head_rot_matrix is None:
                raise Exception
            else:
                mean_head_rot = np.nanmean(head_rot_matrix, axis=0)
                head_rot_x = mean_head_rot[0] - 90
                head_rot_y = 90 - mean_head_rot[1]
                head_v_x, head_v_y, head_v_z = FixationProcessor.spher2cart(1, head_rot_y, head_rot_x)
                norm = np.array([head_v_x, head_v_y, head_v_z]) * -1
        elif hard_norm is None:   # define norm by mean of points
            trunc_points = point_matrix.copy()
            n_pts = trunc_points.shape[1]
            if hard_mean_point is None:
                if length > 1:
                    for i in range(n_pts):
                        pts = trunc_points[:, i]
                        sd3 = 1 * np.nanstd(pts)
                        trunc_up = np.nanmean(pts) + sd3
                        trunc_down = np.nanmean(pts) - sd3
                        trunc_up = np.nanmax(pts) if pd.isna(trunc_up) else trunc_up
                        trunc_down = np.nanmax(pts) if pd.isna(trunc_down) else trunc_down
                        with warnings.catch_warnings():  # catch nan warnings
                            warnings.simplefilter("ignore", category=DeprecationWarning)
                            trunc_points[:, i] = np.clip(pts, trunc_down, trunc_up)
                    mean_point = np.nanmean(trunc_points, axis=0)
                else:
                    mean_point = trunc_points[0, :]
            else:
                mean_point = hard_mean_point
            # mean_test = np.nanmean(point_matrix, axis=0)
            v = mean_point - mean_head_loc
            # print("mean_point", mean_point)
            # print("mean_head_loc", mean_head_loc)
            norm = v / np.linalg.norm(v)
        else:
            norm = hard_norm

        # print("norm", norm)
        for i in range(length):
            # print("iter")
            # print("point:", point_matrix[i, :])
            x, y, z = FixationProcessor.project_3d_to_2d(norm, mean_head_loc, point_matrix[i, :])
            # print(f"x: {x}, y: {y}, z: {z}")
            proj_mat[i, :] = (x[0], y[0], z[0])

        if output_dim == '2d':
            # pca = PCA(n_components=2)
            # proj_mat_2d = pca.fit_transform(proj_mat)
            proj_mat_2d = FixationProcessor.convert_3d_to_2d(norm, mean_head_loc, proj_mat)
            # for i in range(length):
            #     x2, y2 = FixationProcessor.convert_3d_to_2d(norm, mean_head_loc, proj_mat[i, :])
            #     proj_mat_2d[i, :] = (x2, y2)
            return proj_mat_2d
        else:
            return proj_mat

    @staticmethod
    def project_3d_to_2d(normal, plane_point, point_3d):
        # print("normal", normal)
        a, b, c = normal
        x0, y0, z0 = plane_point
        x, y, z = point_3d
        # a*x - a*x0 + b*y - b*y0 + c*z - c*z0 = 0
        # a*x +b*y + c*z - (a*x0 + b*y0 + c*z0) = 0
        d = [(a * x0) + (b * y0) + (c * z0)]
        k = (d - (a * x) - (b * y) - (c * z)) / ((a ** 2) + (b ** 2) + (c ** 2))
        out = x + (k * a), y + (k * b), z + (k * c)
        # print("out", out)
        return out

    @staticmethod
    def convert_3d_to_2d(normal, plane_point, points_3d):
        # calculate projection matrix
        R = np.linalg.qr(np.vstack((normal, np.random.randn(2, 3))))[0].T
        print("R.T before", R.T)
        T = -np.dot(R, plane_point)
        M = np.vstack((R.T, T.reshape(1, 3)))
        print("R.T", R.T.shape)
        print("M before", M.shape)
        M = np.vstack((M.T, np.array([0, 0, 0, 1])))
        invM = np.linalg.inv(M)

        # project points onto plane
        projected_points = []
        for point_3d in points_3d:
            point_3d = np.append(point_3d, 1)
            point_2d = np.dot(invM, point_3d)
            projected_points.append(point_2d[:2])

        return np.array(projected_points)



    @staticmethod
    def spher2cart(r, theta, phi, degrees=True):
        """converts spherical coordinates to cartesian"""
        if degrees:
            theta = theta * (np.pi / 180)
            phi = phi * (np.pi / 180)
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        return x, y, z

    @staticmethod
    def cart2spher(x, y, z, degrees=True):
        r = np.sqrt(np.sum([x ** 2, y ** 2, z ** 2]))
        theta = np.arctan(y / x)
        phi = np.arctan((np.sqrt(x ** 2 + y ** 2)) / z)
        if degrees:
            theta *= (180 / np.pi)
            phi *= (180 / np.pi)
        return r, theta, phi

    @staticmethod
    def combine_fixations(first_index, second_index, df=None, list_of_variables=[]):
        """

        :param df:
        :param first_index:
        :param second_index:
        :param list_of_variables: must be in same order as column order in df
        :return:
        """
        if df is None:
            viewing_id, fix_object, fix_start, fix_start_frame, \
            fix_end, fix_end_frame, fix_duration, fix_frames, \
            centroid_x, centroid_y, centroid_z, invalid_duration, dispersion, \
            mean_velocity, max_velocity, mean_acceleration = tuple(list_of_variables)
        else:
            cols = ['viewing_id', 'object', 'start_time', 'start_frame',
                    'end_time', 'end_frame', 'duration_time', 'duration_frame',
                    'centroid_x', 'centroid_y', 'centroid_z', 'invalid_duration',
                    'dispersion', 'mean_velocity', 'max_velocity', 'mean_acceleration']
            # convert none to na
            df = df.fillna(value=np.nan)
            viewing_id, fix_object, fix_start, fix_start_frame, \
            fix_end, fix_end_frame, fix_duration, fix_frames, \
            centroid_x, centroid_y, centroid_z, invalid_duration, dispersion, \
            mean_velocity, max_velocity, mean_acceleration = (df.loc[:, col].values.tolist() for col in cols)

        fix_end[first_index] = fix_end[second_index]
        fix_end_frame[first_index] = fix_end_frame[second_index]
        try:
            total_duration = fix_end[second_index] - fix_start[first_index]
            combined_duration = fix_duration[first_index] + fix_duration[second_index]
        except TypeError as e:
            print("error")
            raise e
        try:
            p_first = fix_duration[first_index] / combined_duration
        except ZeroDivisionError as e:
            raise e
        p_second = fix_duration[second_index] / combined_duration
        p_av_list = (centroid_x, centroid_y, centroid_z, dispersion, mean_velocity, max_velocity,
                     mean_acceleration)
        for _list in p_av_list:
            _list[first_index] = (_list[first_index] * p_first) + (_list[second_index] * p_second)

        fix_frames[first_index] = fix_end_frame[second_index] - fix_start_frame[first_index]
        fix_duration[first_index] = total_duration
        invalid_duration[first_index] = invalid_duration[first_index] + invalid_duration[second_index]
        var_list = [viewing_id, fix_object, fix_start, fix_start_frame, fix_end, fix_end_frame, fix_duration,
                    fix_frames, centroid_x, centroid_y, centroid_z, invalid_duration, dispersion,
                    mean_velocity, max_velocity, mean_acceleration]
        var_list = del_multiple(second_index, var_list)

        if df is None:
            return var_list
        else:
            var_list_df = pd.DataFrame(var_list).transpose()
            var_list_df.columns = cols
            if len(df.columns) > len(cols):     # if extra columns unaccounted for (ids etc)
                other_cols = tuple(set(df.columns) - set(cols))     # get these cols
                other_cols_lists = [df.loc[:, i].values.tolist() for i in other_cols]   # turn into list of lists
                other_cols_lists = del_multiple(second_index, other_cols_lists)     # delete at index
                other_cols_df = pd.DataFrame(other_cols_lists).transpose()      # convert back to df
                other_cols_df.columns = other_cols              # add column names back
                return_df = pd.concat([var_list_df, other_cols_df], axis=1)     # add two dfs together
                return_df = return_df.loc[:, df.columns]        # reorder columns to original
            else:
                return_df = var_list_df

            return return_df

    @staticmethod
    def threshold_fixboolarray_duration(fix_bools, less_than_threshold, t, inverse=False):
        fix_bools = np.array(fix_bools, dtype=int)

        try:
            t_diff = np.diff(np.array(t))
        except TypeError as e:
            raise e

        if inverse:     # invert true and false
            T = 0
            F = 1
        else:
            T = 1
            F = 0

        fix_length = 0
        fix_start_ind = 0
        this_label = None
        prev_label = None
        for i in range(1, len(fix_bools)):      # loop through each point
            this_label = fix_bools[i]
            prev_label = fix_bools[i - 1]

            if this_label == T and prev_label == F:
                fix_start_ind = i
                fix_length += t_diff[i - 1]
            elif this_label == F and prev_label == F:
                pass
            elif this_label == F and prev_label == T:
                if fix_length < less_than_threshold:
                    fix_bools[fix_start_ind:i] = F
                fix_length = 0
            elif this_label == T and prev_label == T:
                fix_length += t_diff[i - 1]
                if i == 1:
                    fix_start_ind = i-1

        # if end
        if this_label is not None:
            if this_label == T and fix_length < less_than_threshold:
                fix_bools[fix_start_ind:] = F
        else:
            print("")

        return np.array(fix_bools, dtype=bool)



