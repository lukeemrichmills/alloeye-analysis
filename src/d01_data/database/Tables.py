def extract_columns(create_table_command):
    split = create_table_command.split('"')
    col_names = [split[i] for i in range(len(split)) if i % 2 != 0]
    return col_names[1:]

class Tables:
    participants = """
            CREATE TABLE IF NOT EXISTS "participants" (
                "pid" varchar PRIMARY KEY,
                "dob" date,
                "created_at" timestamp,
                "years_of_education" int,
                "sex" varchar,
                "handedness" varchar,
                "occupation" varchar
            )
            """
    study = """
    CREATE TABLE IF NOT EXISTS "study" (
        "study_id" varchar PRIMARY KEY,
        "name" varchar,
        "date_started" date,
        "date_ended" date,
        "recruitment_status" varchar,
        "total_participants" int
    )
    """
    participants_by_study = """
    CREATE TABLE IF NOT EXISTS "participants_by_study" (
      "row_id" SERIAL PRIMARY KEY,
      "study_pid" varchar,
      "study_id" varchar references study(study_id),
      "ppt_id" varchar references participants(pid) ON DELETE CASCADE
    )
    """
    study_data_files = """
    CREATE TABLE IF NOT EXISTS "study_data_files" (
        "row_id" SERIAL PRIMARY KEY,
        "file_id" varchar,
        "ppt_id" varchar,
        "study_id" varchar,
        "study_pid" varchar,
        "test_data" bool,
        "data_type" varchar,
        "file_name" varchar,
        "file_type" varchar,
        "created_at" timestamp
    )
    """
    block = """
    CREATE TABLE IF NOT EXISTS "block" (
        "block_id" varchar PRIMARY KEY,
        "block_order" int,
        "n_trials" int,
        "ppt_id" varchar references participants(pid) ON DELETE CASCADE,
        "study_id" varchar references study(study_id),
        "practice" bool
    )
    """
    alloeye_conditions = """
    CREATE TABLE IF NOT EXISTS "alloeye_conditions" (
        "condition_id" varchar primary key,
        "ppt_id" varchar references participants(pid) ON DELETE CASCADE,
        "study_id" varchar references study(study_id),
        "condition" varchar,
        "n_trials" int,
        "n_correct" int,
        "p_correct" float8
    )
    """
    alloeye_trial = """
    CREATE TABLE IF NOT EXISTS "alloeye_trial" (
        "trial_id" varchar PRIMARY KEY,
        "study_id" varchar references study(study_id),
        "ppt_id" varchar references participants(pid) ON DELETE CASCADE,
        "block_id" varchar references block(block_id),
        "condition_id" varchar references alloeye_conditions(condition_id),
        "practice" bool,
        "block_number" int,
        "trial_number" int,
        "configuration_number" int,
        "move_type" varchar,
        "table_rotates" bool,
        "anticlockwise_move" bool,
        "viewing_angle" int,
        "object_shifted" varchar,
        "selected_object" varchar,
        "confidence_rating" int,
        "co_preshift_x_raw" float8,
        "co_preshift_z_raw" float8,
        "co_preshift_x_rot_adj" float8,
        "co_preshift_z_rot_adj" float8,
        "co_postshift_x_raw" float8,
        "co_postshift_z_raw" float8,
        "co_postshift_x_rot_adj" float8,
        "co_postshift_z_rot_adj" float8,
        "co_shift_distance" float8,
        "obj1_name" varchar,
        "obj1_preshift_x" float8,
        "obj1_preshift_z" float8,
        "obj1_postshift_x" float8,
        "obj1_postshift_z" float8,
        "obj2_name" varchar,
        "obj2_preshift_x" float8,
        "obj2_preshift_z" float8,
        "obj2_postshift_x" float8,
        "obj2_postshift_z" float8,
        "obj3_name" varchar,
        "obj3_preshift_x" float8,
        "obj3_preshift_z" float8,
        "obj3_postshift_x" float8,
        "obj3_postshift_z" float8,
        "obj4_name" varchar,
        "obj4_preshift_x" float8,
        "obj4_preshift_z" float8,
        "obj4_postshift_x" float8,
        "obj4_postshift_z" float8,
        "obj5_name" varchar,
        "obj5_preshift_x" float8,
        "obj5_preshift_z" float8,
        "obj5_postshift_x" float8,
        "obj5_postshift_z" float8,
        "table_location_x" float8,
        "table_location_z" float8
    )
    """
    alloeye_viewing = """       
    CREATE TABLE IF NOT EXISTS "alloeye_viewing" (
        "viewing_id" varchar primary key,
        "ppt_id" varchar references participants(pid) ON DELETE CASCADE,
        "study_id" varchar references study(study_id),
        "block_id" varchar references block(block_id),
        "trial_id" varchar references alloeye_trial(trial_id),
        "viewing_type" varchar,
        "co_x_raw" float8,
        "co_z_raw" float8,
        "co_x_rot_adj" float8,
        "co_z_rot_adj" float8,
        "obj1_name" varchar,
        "obj1_x" float8,
        "obj1_z" float8,
        "obj2_name" varchar,
        "obj2_x" float8,
        "obj2_z" float8,
        "obj3_name" varchar,
        "obj3_x" float8,
        "obj3_z" float8,
        "obj4_name" varchar,
        "obj4_x" float8,
        "obj4_z" float8,
        "obj5_name" varchar,
        "obj5_x" float8,
        "obj5_z" float8
    )
    """
    fixations_saccades = """
    CREATE TABLE IF NOT EXISTS "fixations_saccades" (
        "fixation_id" varchar PRIMARY KEY,
        "study_id" varchar references study(study_id),
        "ppt_id" varchar references participants(pid) ON DELETE CASCADE,
        "block_id" varchar references block(block_id),
        "trial_id" varchar references alloeye_trial(trial_id),
        "viewing_id" varchar references  alloeye_viewing(viewing_id),
        "algorithm" varchar,
        "fixation_or_saccade" varchar,
        "object" varchar,
        "start_time" float8,
        "end_time" float8,
        "start_frame" int,
        "end_frame" int,
        "duration_time" float8,
        "duration_frame" int,
        "invalid_duration" float8,
        "centroid_x" float8,
        "centroid_y" float8,
        "centroid_z" float8,
        "dispersion" float8,
        "mean_velocity" float8,
        "max_velocity" float8,
        "mean_acceleration" float8,            
        "gaze_object_proportion" float8,
        "second_gaze_object" varchar,
        "second_gaze_object_proportion" float8,
        "other_gaze_object_proportion" float8,
        "missing_split_group" int
        )
    """
    alloeye_timepoint_viewing = """  
    CREATE TABLE IF NOT EXISTS "alloeye_timepoint_viewing" (
        "timepoint_id" varchar PRIMARY KEY,
        "trial_or_all" varchar,
        "ppt_id" varchar references participants(pid) ON DELETE CASCADE,
        "study_id" varchar references study(study_id),
        "block_id" varchar references block(block_id),
        "trial_id" varchar references alloeye_trial(trial_id),
        "viewing_id" varchar references alloeye_viewing(viewing_id),
        "retrieval_epoch" varchar, 
        "eye_timestamp_ms" int,
        "unity_timestamp" float,
        "eye_frame_number" int,
        "unity_frame_number" int,
        "fps" float8,
        "gaze_object" varchar,
        "object_position_x" float8,
        "object_position_y" float8,
        "object_position_z" float8,
        "gaze_collision_x" float8,
        "gaze_collision_y" float8,
        "gaze_collision_z" float8,
        "left_pupil_diameter" float8,
        "right_pupil_diameter" float8,
        "left_eye_openness" float8,
        "right_eye_openness" float8,
        "camera_x" float8,
        "camera_y" float8,
        "camera_z" float8,
        "cam_rotation_x" float8,
        "cam_rotation_y" float8,
        "cam_rotation_z" float8,
        "right_controller_x" float8,
        "right_controller_y" float8,
        "right_controller_z" float8,
        "right_cntrllr_rot_x" float8,
        "right_cntrllr_rot_y" float8,
        "right_cntrllr_rot_z" float8,
        "left_controller_x" float8,
        "left_controller_y" float8,
        "left_controller_z" float8,
        "left_cntrllr_rot_x" float8,
        "left_cntrllr_rot_y" float8,
        "left_cntrllr_rot_z" float8,
        "left_gaze_origin_x" float8,
        "left_gaze_origin_y" float8,
        "left_gaze_origin_z" float8,
        "right_gaze_origin_x" float8,
        "right_gaze_origin_y" float8,
        "right_gaze_origin_z" float8,
        "left_gaze_direction_x" float8,
        "left_gaze_direction_y" float8,
        "left_gaze_direction_z" float8,
        "right_gaze_direction_x" float8,
        "right_gaze_direction_y" float8,
        "right_gaze_direction_z" float8,
        "gaze_object_no_table" varchar,
        "gaze_collision_no_table_x" float8,
        "gaze_collision_no_table_y" float8,
        "gaze_collision_no_table_z" float8
    )
    """
    alloeye_timepoint_all = """  
    CREATE TABLE IF NOT EXISTS "alloeye_timepoint_all" (
        "timepoint_id" varchar PRIMARY KEY,
        "trial_or_all" varchar,
        "ppt_id" varchar references participants(pid) ON DELETE CASCADE,
        "study_id" varchar references study(study_id),
        "block_id" varchar references block(block_id),
        "trial_id" varchar references alloeye_trial(trial_id),
        "viewing_id" varchar,
        "retrieval_epoch" varchar, 
        "eye_timestamp_ms" int,
        "unity_timestamp" float,
        "eye_frame_number" int,
        "unity_frame_number" int,
        "fps" float8,
        "gaze_object" varchar,
        "object_position_x" float8,
        "object_position_y" float8,
        "object_position_z" float8,
        "gaze_collision_x" float8,
        "gaze_collision_y" float8,
        "gaze_collision_z" float8,
        "left_pupil_diameter" float8,
        "right_pupil_diameter" float8,
        "left_eye_openness" float8,
        "right_eye_openness" float8,
        "camera_x" float8,
        "camera_y" float8,
        "camera_z" float8,
        "cam_rotation_x" float8,
        "cam_rotation_y" float8,
        "cam_rotation_z" float8,
        "right_controller_x" float8,
        "right_controller_y" float8,
        "right_controller_z" float8,
        "right_cntrllr_rot_x" float8,
        "right_cntrllr_rot_y" float8,
        "right_cntrllr_rot_z" float8,
        "left_controller_x" float8,
        "left_controller_y" float8,
        "left_controller_z" float8,
        "left_cntrllr_rot_x" float8,
        "left_cntrllr_rot_y" float8,
        "left_cntrllr_rot_z" float8,
        "left_gaze_origin_x" float8,
        "left_gaze_origin_y" float8,
        "left_gaze_origin_z" float8,
        "right_gaze_origin_x" float8,
        "right_gaze_origin_y" float8,
        "right_gaze_origin_z" float8,
        "left_gaze_direction_x" float8,
        "left_gaze_direction_y" float8,
        "left_gaze_direction_z" float8,
        "right_gaze_direction_x" float8,
        "right_gaze_direction_y" float8,
        "right_gaze_direction_z" float8,
        "gaze_object_no_table" varchar,
        "gaze_collision_no_table_x" float8,
        "gaze_collision_no_table_y" float8,
        "gaze_collision_no_table_z" float8
    )
    """
    alloeye_feature_score = """
    CREATE TABLE IF NOT EXISTS "alloeye_feature_score" (
        "row_id" SERIAL PRIMARY KEY,
        "feature_id" varchar,
        "ppt_id" varchar references participants(pid),
        "study_id" varchar references study(study_id),
        "trial_id" varchar references alloeye_trial(trial_id),
        "viewing_id" varchar references alloeye_viewing(viewing_id),
        "encoding_retrieval" varchar,
        "feature_name" varchar,
        "score_float" float8,
        "score_int" int,
        "score_bool" bool

    )
    """
    neuropsych_test_score = """
    CREATE TABLE IF NOT EXISTS "neuropsych_test_score" (
      "row_id" SERIAL PRIMARY KEY,
      "score_id" varchar,
      "ppt_id" varchar references participants(pid),
      "study_id" varchar references study(study_id),
      "test_datetime" timestamp,
      "test_name" varchar,
      "score_name" varchar,
      "score" float8
    )
    """
    all_dict = {
        'participants': participants,
        'study': study,
        'participants_by_study': participants_by_study,
        'study_data_files': study_data_files,
        'block': block,
        'alloeye_conditions': alloeye_conditions,
        'alloeye_trial': alloeye_trial,
        'alloeye_viewing': alloeye_viewing,
        'fixations_saccades': fixations_saccades,
        'alloeye_timepoint_viewing': alloeye_timepoint_viewing,
        'alloeye_timepoint_all': alloeye_timepoint_all,
        'alloeye_feature_score': alloeye_feature_score,
        'neuropsych_test_score': neuropsych_test_score
    }
    all = tuple(all_dict.values())

    table_columns = {}
    for name, command in all_dict.items():
        table_columns[name] = extract_columns(command)

