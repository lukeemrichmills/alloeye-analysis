from src.d01_data.database.ToSQL.ToSQL import ToSQL


class FixationsToSQL(ToSQL):

    def __init__(self, fix_sac_df, col_names, study_id="alloeye"):
        self.file = None
        self.study_id = study_id
        self.output_commands = ""
        self.df = fix_sac_df
        self.col_names = col_names
        self.table_name = "fixations_saccades"


    def get_row_values(self, row, index, table_name, file_data_type):

        # get ids
        viewing_id = row.viewing_id
        study_id, ppt_id, block_id, trial_id = ToSQL.get_ids_from_viewing(viewing_id)

        # create fixation id - viewing_id+method+_index
        method = row.algorithm
        trunc_method = method if len(method) < 6 else method[0:6]
        fix_id = f'{viewing_id}_{trunc_method}_{str(index)}'
        
        fix_or_sac = row.fixation_or_saccade

        # add row values to list
        list_of_row_values = [fix_id, study_id, ppt_id, block_id, trial_id, viewing_id,
                              row.algorithm, fix_or_sac,
                              row.object, row.start_time, row.end_time, row.start_frame, row.end_frame,
                              row.duration_time, row.duration_frame, row.invalid_duration,
                              row.centroid_x, row.centroid_y, row.centroid_z,
                              row.dispersion, row.mean_velocity, row.max_velocity, row.mean_acceleration,
                              row.gaze_object_proportion, row.second_gaze_object, row.second_gaze_object_proportion,
                              row.other_gaze_object_proportion, row.missing_split_group]
        return list_of_row_values, False


