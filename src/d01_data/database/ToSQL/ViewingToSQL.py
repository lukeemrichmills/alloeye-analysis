from src.d01_data.database.ToSQL.ToSQL import ToSQL


class ViewingToSQL(ToSQL):

    def __init__(self, viewing_df, col_names, study_id="alloeye"):
        self.file = None
        self.study_id = study_id
        self.output_commands = ""
        self.df = viewing_df
        self.col_names = col_names
        self.table_name = "alloeye_viewing"

    # def get_row_values(self, row, index, table_name, file_data_type):
    #     # get ids
    #     viewing_id = row.viewing_id
    #     study_id, ppt_id, block_id, trial_id = ToSQL.get_ids_from_viewing(viewing_id)
    #
    #
    #     fix_or_sac = row.fixation_or_saccade
    #
    #     # add row values to list
    #     list_of_row_values = [fix_id, study_id, ppt_id, block_id, trial_id, viewing_id,
    #                           row.algorithm, fix_or_sac,
    #                           row.object, row.start_time, row.end_time, row.start_frame, row.end_frame,
    #                           row.duration, row.frames, row.invalid_duration,
    #                           row.centroid_x, row.centroid_y, row.centroid_z,
    #                           row.dispersion, row.mean_velocity, row.max_velocity, row.mean_acceleration]
    #     return list_of_row_values, False
