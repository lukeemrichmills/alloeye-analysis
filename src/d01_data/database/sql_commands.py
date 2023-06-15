import math

from src.d00_utils.DataLevel import DataLevel
from src.d00_utils.generic_tools import input_to_continue
from src.d01_data.database import Tables, alter_table
from src.d01_data.database.PsqlCommander import *
from src.d01_data.database.Errors import *
from src.d01_data.database.ToSQL.CsvToSQL.TimepointCsvToSQL import TimepointCsvToSQL
from src.d01_data.database.ToSQL.CsvToSQL.TrialCsvToSQL import TrialCsvToSQL
from src.d01_data.database.ToSQL.FixationsToSQL import FixationsToSQL
from src.d01_data.database.parse_file_name import parse_file_names
from src.d01_data.database.ToSQL.CsvToSQL.CsvToSQL import *
from src.d01_data.database.upload_fixations import upload_fixations
from src.d01_data.fetch import check_exists, fetch_tools
from src.d01_data.fetch.batch_fetch import batch_fetch
from src.d01_data.fetch.fetch_conditions import fetch_conditions
from src.d01_data.fetch.fetch_trials import fetch_trials
from src.d01_data.fetch.fetch_viewings import fetch_viewings
from src.d03_processing.Features import Features
from src.d03_processing.feature_extract.timepoint_to_fixations import timepoint_to_fixations
from src.d03_processing.feature_extract.timepoint_to_viewing import timepoint_to_viewing, add_feature_columns
from src.d03_processing.feature_extract.to_viewing import to_viewing
from src.d03_processing.feature_extract.viewing_to_trial import viewing_to_trial
from src.d03_processing.feature_extract.trial_to_conditions import trial_to_conditions
from src.d03_processing.feature_extract.trial_to_block import trial_to_block
from src.d03_processing.feature_extract.block_to_ppt import block_to_ppt
from src.d03_processing.fixations.FixAlgos import FixAlgo, fix_algo_dict


def delete_tables(connection):
    delete_table_commander = PsqlCommander()
    get_table_name_query = "SELECT table_name " \
                           "FROM information_schema.tables " \
                           "WHERE table_type = 'BASE TABLE' " \
                           "AND table_schema = 'public'"
    table_names = delete_table_commander.fetch_query(connection, get_table_name_query, None)

    for table in table_names:
        delete_table_commander.execute_query(connection, f"DROP TABLE {table} CASCADE", None)
    pass


def create_tables(connection):
    """
    create tables
    :param connection:
    :return:
    """
    table_names = ["participants", "study", "participants_by_study",
                   "study_data_files", "block", "alloeye_trial", "alloeye_viewing", "fixations_saccades",
                   "alloeye_timepoint_viewing", "alloeye_timepoint_all", "alloeye_feature_score",
                   "neuropsych_test_score", "alloeye_conditions"]
    commands = Tables.Tables.all

    if len(commands) != len(table_names):
        raise ListsNotEqualLengthError(commands, table_names)
    create_table = PsqlCommander(commands)
    skip_bools = create_table.check_tables_exist(connection, table_names)
    create_table.run_commands(connection, skip_bools)


def fill_tables(connection, raw_data_dir, study_id):
    """Uploads data for participants not yet uploaded"""
    # get list of pids from files
    df_csvs = parse_file_names(raw_data_dir)
    unique_pIDs = df_csvs.pID.unique()

    # check which pids already uploaded...
    fill_table_commander = PsqlCommander()
    table_metadata = fill_table_commander.get_table_metadata(connection, "alloeye_timepoint_viewing")
    uploaded_pIDs = table_metadata[1]
    uploaded_pIDs.sort()
    unique_pIDs = [f'{study_id}_{str(i)}' for i in unique_pIDs]  # add study_id to front
    unique_pIDs.sort()

    # ... only upload the ones who haven't
    all_pIDs_uploaded = True
    if uploaded_pIDs != unique_pIDs:  # NEED BETTER WAY OF CHECKING
        all_pIDs_uploaded = False
        not_uploaded_pIDs = list(set(unique_pIDs) - set(uploaded_pIDs))
        not_uploaded_pIDs = [i.split(f'{study_id}_')[1] for i in
                             not_uploaded_pIDs]  # need to study_id because gets added again!

    # # insert into study table
    # col_names = ["study_id", "name", "recruitment_status", "total_participants"]
    # row_values = ["alloeye", "AlloEye", "ongoing", 0]
    # study_command = CsvToSQL.command_output(row_values, col_names, "study")
    # fill_table_commander.execute_query(connection, study_command, None)

    # upload all data for pids that haven't been uploaded
    if all_pIDs_uploaded is False:
        print("uploading from the following ppts:")
        print(not_uploaded_pIDs)
        for pid in not_uploaded_pIDs:
            file_uploads = df_csvs[df_csvs.pID == pid]
            table_col_names = Tables.table_columns
            print(f"converting to table insert commands for ppt {pid}")
            full_commands = get_tableinsert_commands(file_uploads, raw_data_dir, table_col_names, pid)
            print(f"executing table insert commands for ppt {pid}")
            for index in range(len(full_commands)):
                count = 0
                command_list = full_commands[index]
                # if index < 4:
                #     command_list = []
                # else:
                #     pass
                if len(command_list) == 0:
                    continue
                for index_2 in range(len(command_list)):
                    commands = command_list[index_2]
                    try:
                        commands_one = "; ".join(commands) if type(commands) is list else commands
                    except:
                        print("TypeError")
                    fill_table_commander.execute_query(connection, commands_one, None)
                    count += len(commands) if type(commands) is list else 1
                table_string = commands_one[12:32].split(" ")[0]
                print(f"{count} rows inserted for participant '{pid}' into table '{table_string}'")
            # NEED TO CHECK IF ROWS ALREADY EXIST - CHECK ROW ID
    else:
        print("All participants already uploaded")
    # if some rows but not full amount, print summary of missing data (e.g. ppt IDs, number of rows etc. not imported)
    # and require input ('continue? y/n') to upload any new data
    # import from csv (should have imported from csv and hold as FileType) and check import worked with logging
    # use

    # Upload participants data

    # Upload study data
    print("fill_tables ended")


def get_tableinsert_commands(files, raw_data_dir, table_col_names, pID):
    """matches file to data type, appends commands per type, returns full commands"""
    study_id = "alloeye"
    timepoint_insert_commands = []
    timepoint_insert_commands_2 = []
    trial_insert_commands = []
    block_insert_commands = []
    viewing_insert_commands = []
    participants_insert_commands = get_participant_insert_commands(table_col_names, pID)
    study_insert_commands = add_ppt_study_table_command("alloeye", pID)
    conditions_insert_commands = get_condition_insert_commands(pID)

    for index, row in files.iterrows():
        filename = row['filename']

        if row['data_type'] == 'TrialGazeData':  # AllGazeData will be included
            # get column names
            col_names = table_col_names['alloeye_timepoint_viewing']

            # get command strings
            sql_commands = TimepointCsvToSQL(row, study_id, raw_data_dir, col_names).convert_to_insert_commands()

            # append strings

            # batch algo
            batch_limit = 5000
            next_batch_no = 0
            last_batch_no = 0
            batch_no = 1
            if len(sql_commands) > batch_limit:
                next_batch_no += batch_limit
                timepoint_insert_commands.append(sql_commands[0:batch_limit])
                last_batch_no = next_batch_no
                for i in range(1, math.ceil(len(sql_commands) / batch_limit)):
                    batch_no += 1
                    next_batch_no += batch_limit
                    if next_batch_no > len(sql_commands):
                        next_batch_no = len(sql_commands)
                    timepoint_insert_commands.append(sql_commands[last_batch_no:next_batch_no])
                    last_batch_no = next_batch_no
            else:
                timepoint_insert_commands.append(sql_commands)

        elif row['data_type'] == 'TrialInfo':
            # get both block and trial table column names
            col_names = table_col_names['alloeye_trial']
            block_cols = table_col_names['block']
            viewing_cols = table_col_names['alloeye_viewing']
            cond_cols = table_col_names['alloeye_conditions']

            # get command strings
            trial_block_commander = TrialCsvToSQL(row, study_id, raw_data_dir, col_names)
            trial_sql_commands, block_sql_commands, viewing_sql_commands \
                = trial_block_commander.viewing_trial_block_commands(block_cols, viewing_cols)

            # append strings
            trial_insert_commands.append(trial_sql_commands)
            block_insert_commands.append(block_sql_commands)
            viewing_insert_commands.append(viewing_sql_commands)

    commands = (#study_insert_commands,   # something wrong with this - database corrupt somehow? never gets granted
                participants_insert_commands,
                block_insert_commands,
                conditions_insert_commands,
                trial_insert_commands,
                viewing_insert_commands,
                timepoint_insert_commands,
                timepoint_insert_commands_2)
    return commands


def get_participant_insert_commands(col_names, pid):
    commands = []
    col_names = "(pid)"
    row_values = f"('alloeye_{pid}')"
    # date of birth
    # years of education
    # sex
    # handedness
    # occupation]
    commands.append(CsvToSQL.insert_command_output(row_values, col_names, "participants"))
    return commands


def get_condition_insert_commands(pid):
    commands = []
    col_names = ["condition_id", "ppt_id", "study_id", "condition"]
    conditions = ('StayStay', 'WalkStay', 'TeleportStay', 'StayRotate', 'WalkRotate', 'TeleportRotate')
    for cond in conditions:
        commands.append(ToSQL.insert_command_output([f'alloeye_{pid}_{cond}', f'alloeye_{pid}', 'alloeye', cond],
                                                    col_names, "alloeye_conditions"))
    return commands


def add_ppt_study_table_command(study_id, pID):
    commands = []
    update_pids = f"UPDATE study SET total_participants = study.total_participants + 1 " \
                  f"where study_id='{study_id}'"
    commands.append(update_pids)
    return commands


def add_features(connection, fix_algo_features='GazeCollision', fix_algos_upload=fix_algo_dict(),
                 rerun_all_ppts=False, rerun_ppts=[],
                 rerun_all_viewing_features=True, rerun_viewing_features=[],
                 rerun_all_trial_features=True, rerun_trial_features=[],
                 rerun_all_viewings=False, rerun_viewings=[], rerun_all_fixations=False, rerun_fixations_for=[],
                 rerun_all_trials=False, rerun_trials=[],
                 rerun_all_conditions=False, rerun_conditions=[],
                 skip_practice=True, skip_viewing=False, skip_trial=False, skip_condition=False,
                 rerun_everything=False):
    """analyses timepoints -> viewing -> trial -> block -> participants, adding features to subsequent tables
    """
    if rerun_everything:
        rerun_all_ppts, rerun_all_viewing_features, rerun_all_trial_features, \
        rerun_all_viewings, rerun_all_trials, rerun_all_conditions = (True for i in range(6))
        skip_viewing, skip_trial, skip_condition = (False for i in range(3))

    # get all uploaded viewings from viewings
    fill_table_commander = PsqlCommander()
    table_metadata = fill_table_commander.get_table_metadata(connection, "alloeye_timepoint_viewing")
    ppts = table_metadata[1]

    # # TIMEPOINTS/FIXATIONS >> VIEWING

    if not skip_viewing:
        batches, df, features = get_batches('viewing_id', 'alloeye_viewing', Features.viewing, Features.viewing_dict,
                                            rerun_all_viewing_features,
                                            rerun_viewing_features, rerun_all_viewings, rerun_viewings, 100, connection,
                                            skip_practice)

        # add viewing features per batch
        for i in range(len(batches)):

            viewing_list, viewing_df, features_to_add, skip = batch_setup(batches, df, features, i, 'viewing',
                                                                          fetch_viewings, 'viewing_list')
            if skip:
                continue

            # check if features require fixations, timepoints or both)
            get_timepoints_all = False
            get_fixations_all = False
            for feature in features_to_add:
                if feature in Features.viewing_from_timepoints:
                    get_timepoints_all = True
                else:
                    get_fixations_all = True

            if rerun_all_fixations:
                rerun_fixations_for = viewing_list

            rerun_fix = any([True for v in viewing_list if v in rerun_fixations_for])

            full_fix_df, all_timepoints = get_fix_tp_dfs(viewing_list, get_timepoints_all, get_fixations_all,
                                                         rerun_fix, rerun_fixations_for, fix_algo_features,
                                                         fix_algos_upload, connection, skip_practice=skip_practice)

            viewing_df = to_viewing(viewing_df=viewing_df, fix_sac_df=full_fix_df,
                                    features=features_to_add, all_timepoints=all_timepoints,
                                    fix_method_alt_string=fix_algo_features)

            # upload viewing features
            upload_updates(viewing_df, features_to_add, 'viewing_id', 'alloeye_viewing', i + 1, connection)

    # # VIEWING >> TRIAL
    if not skip_trial:
        standard_add_features('alloeye_trial', 'trial_id', Features.trial, Features.trial_dict, 'trial', fetch_trials,
                              'trial_ids', DataLevel.viewing, 'alloeye_viewing', rerun_all_trial_features,
                              rerun_trial_features, rerun_all_trials,
                              rerun_trials, viewing_to_trial, connection, skip_practice,
                              fix_algo_features=fix_algo_features)

    # # TRIAL >> CONDITION (ppt, block, condition, group)
    # convert all trial-level features including encoding, even though doesn't make sense to have encoding by condition
    # can then group-by and sum/average for participant-level data
    # CAN CREATE A DEFAULT VIEW OF CONDITION TABLE TO EXCLUDE ENC DATA
    if not skip_condition:
        standard_add_features(table='alloeye_conditions', id_col_name='condition_id', feature_list=Features.conditions,
                              feature_dict=Features.conditions_dict, list_name='condition',
                              fetch_func=fetch_conditions, fetch_kw='condition_ids',
                              lower_data_level=DataLevel.trial, lower_table_name='alloeye_trial',
                              rerun_all_features=rerun_all_trial_features, rerun_these_features=rerun_trial_features,
                              rerun_all_ids=rerun_all_conditions, rerun_these_ids=rerun_conditions,
                              feature_processing_func=trial_to_conditions,
                              connection=connection, skip_practice=False, batch_limit=50000,   # need full trial batch?
                              fix_algo_features=fix_algo_features)
        # note: skip_practice is False here because not applicable

    # # CONDITION >> PARTICIPANT

    # convert trial_df to block

    print("end")
    return None


def standard_add_features(table, id_col_name, feature_list, feature_dict, list_name, fetch_func, fetch_kw,
                          lower_data_level, lower_table_name, rerun_all_features, rerun_these_features, rerun_all_ids,
                          rerun_these_ids,
                          feature_processing_func,
                          connection, skip_practice, batch_limit=500, fix_algo_features='VR_IDT'):

    # get batches
    batches, to_add_df, features = get_batches(id_col_name, table, feature_list,
                                               feature_dict, rerun_all_features, rerun_these_features,
                                               rerun_all_ids, rerun_these_ids, batch_limit, connection,
                                               skip_practice)
    # add trial features per batch
    for i in range(len(batches)):
        id_list, higher_level_df, features_to_add, skip = batch_setup(batches, to_add_df, features,
                                                                      i, list_name, fetch_func, fetch_kw)

        if skip:
            continue

        practice_values = [False] if skip_practice else [True, False]

        lower_level_df = batch_fetch(data_level=lower_data_level, table=lower_table_name,
                                     batch_col=fetch_kw, batch_list=id_list,
                                     skip_practice=skip_practice)

        higher_level_df = feature_processing_func(higher_level_df, lower_level_df, features_to_add,
                                                  practice=practice_values,
                                                  fix_algo=fix_algo_features)

        upload_updates(higher_level_df, features_to_add, id_col_name, table, i + 1, connection)


def upload_updates(df, features_to_add, id_col, table, batch_no, connection):
    if df is not None:
        df_to_sql = ToSQL(df, "alloeye", features_to_add)
        commands = df_to_sql.convert_to_update_commands(id_col, table_name=table)
        commands_one = "; ".join(commands) if type(commands) is list else commands
        commander = PsqlCommander()
        commander.execute_query(connection, commands_one, None)
        print(f"""features uploaded for batch {batch_no}""")
    else:
        print(f"""nothing to upload""")


def get_fix_tp_dfs(viewing_list, get_timepoints_all, get_fixations_all,
                   rerun_fix, rerun_fixations_for, fix_algo_features, fix_algos_upload, connection, skip_practice=True):
    """algorithm for determining whether or not to get timepoints and whether to fetch or
       process fixations from timepoints. Moved here for readability further up"""

    if get_timepoints_all or rerun_fix:
        if rerun_fix:
            if len(rerun_fixations_for) > 0:
                viewing_list = rerun_fixations_for

        # fetch timepoints for all viewings
        all_timepoints = batch_fetch(data_level=DataLevel.timepoint, table="alloeye_timepoint_viewing",
                                     batch_col="viewing_id", batch_list=viewing_list, skip_practice=skip_practice)
        if not get_fixations_all:  # if no fixation-derived features...
            full_fix_df = None
    else:
        all_timepoints = None  # default no timepoints if no required features

    if get_fixations_all:  # if at least one fixation-derived feature

        # separate viewings with fixations uploaded already from not
        fetch_list = []
        process_list = []
        process_fixations = False
        if rerun_fix:
            process_list = viewing_list
        else:
            for viewing in viewing_list:
                exists = check_exists.entry(viewing, 'viewing_id', 'algorithm', fix_algo_features, str,
                                            'fixations_saccades', connection)
                if exists:
                    fetch_list.append(viewing)
                else:
                    process_list.append(viewing)

        def process_upload_fixations(process_list, all_timepoints, fix_algos_upload, connection):
            # ... then fetch timepoints if necessary ...
            if all_timepoints is None:
                # batch fetch timepoints for all viewings
                all_timepoints = batch_fetch(data_level=DataLevel.timepoint, table="alloeye_timepoint_viewing",
                                             batch_col="viewing_id", batch_list=process_list)

            # ...process timepoints into fixations...
            processed_fix_df = timepoint_to_fixations(process_list, all_timepoints, fix_algos_upload)

            # ...upload processed fixations...
            upload_fixations(connection, processed_fix_df, fix_algos_upload)

            # ...fetch again with chosen algo...
            fetched_processed = batch_fetch(DataLevel.fix_sacc, "fixations_saccades", batch_list=process_list,
                                            batch_col="viewing_id", pid="all",
                                            fetch_kwargs={'algorithm': fix_algo_features})
            return fetched_processed

        # if any fixations uploaded
        if len(fetch_list) > 0:
            # ... then fetch fixations from list
            fetched_fix_df = batch_fetch(DataLevel.fix_sacc, "fixations_saccades", batch_list=fetch_list,
                                         batch_col="viewing_id", pid="all",
                                         fetch_kwargs={'algorithm': fix_algo_features})

            # if all fixations uploaded
            if len(process_list) == 0:
                # then we're done
                full_fix_df = fetched_fix_df

            # otherwise if some needed processing from timepoints...
            elif len(process_list) > 0:
                processed_df = process_upload_fixations(process_list, all_timepoints, fix_algos_upload,
                                                        connection)
                # ...and combine fix dfs
                full_fix_df = pd.concat([fetched_fix_df, processed_df], ignore_index=True)
        elif len(process_list) > 0:
            processed_df = process_upload_fixations(process_list, all_timepoints, fix_algos_upload, connection)
            # ...and combine fix dfs
            full_fix_df = processed_df

    return full_fix_df, all_timepoints


def get_features_to_add(df, full_features, check_col):
    try:
        _1 = df[check_col].to_list()[0]
    except IndexError as e:
        raise e

    row = df.loc[df[check_col] == _1, :]
    bools = row.drop([check_col], axis=1).to_numpy()[0]
    features_to_add = []
    for j in range(len(full_features)):
        if bools[j] == True:
            features_to_add.append(full_features[j])

    return features_to_add


def batch_setup(batches, list_to_add_df, features, i, list_name, fetch_func, fetch_kw):
    list_to_add = batches[i]
    id_str = list_name + '_id'

    # get features_to_add
    df = list_to_add_df.loc[np.isin(list_to_add_df[id_str], list_to_add), :]
    features_to_add = get_features_to_add(df, features, id_str)

    print(
        f"processing {list_name} features for {list_name} batch {i + 1} of {len(batches)} ({len(list_to_add)} {list_name}s)\n ")

    # fetch dataframe of table for dataframe and batch fetching
    kwargs = {fetch_kw: list_to_add}
    df = fetch_func(pid="all", **kwargs)

    skip = False
    if len(df) == 0:
        print(f"no {list_name}s, skipping")
        skip = True

    # add new feature columns
    df = add_feature_columns(df, features=features_to_add)

    # get trial ids for batching

    out_list = df[id_str].values.tolist()

    return out_list, df, features_to_add, skip


def get_batches(list_id, table, all_features_list, all_features_dict, rerun_all_features,
                rerun_features_list, rerun_all_list, rerun_list, batch_limit, connection, skip_practice):
    # check columns exist, add any missing columns
    check_list = fetch_tools.fetch_col(list_id, table, connection)
    features = all_features_list if rerun_all_features and len(rerun_features_list) == 0 else rerun_features_list
    add_missing_features(features, all_features_dict, table, connection)

    # get features to add per viewing
    # rerun_all = rerun_all_list is True or len(rerun_list) > 0
    to_add_df = check_entries(list_id, check_list, features, table, list_id,
                              connection, rerun_all=rerun_all_list, rerun_list=rerun_list,
                              all_check_list=check_list, skip_practice=skip_practice)
    bools_only = to_add_df.iloc[:, 1:]
    temp_df = to_add_df.loc[bools_only.any(axis=1), :]  # remove viewings not adding at all
    if len(temp_df) > 1:
        batches = batch_by_duplicate_rows(temp_df, list_id, batch_limit)
    else:
        batches = [rerun_list]

    return batches, temp_df, features


def add_missing_features(features, features_dict, table, connection):
    # check if feature columns exist
    feature_exists_dict = {}
    for feature in features:
        exists = check_exists.column(feature, table, connection)
        feature_exists_dict[feature] = exists

    # add any feature columns that need adding
    columns_to_add = list({k: v for (k, v) in feature_exists_dict.items() if v == False}.keys())
    if len(columns_to_add) > 0:
        for feature in columns_to_add:
            try:
                feat_dtype_str = ToSQL.convert_dtype_return_str(features_dict[feature][1])
            except KeyError as e:
                raise e
            alter_table.add_column(feature, feat_dtype_str, table, connection)

    # features_exist = list({k: v for (k, v) in feature_exists_dict.items() if v == True}.keys())
    # return features_exist


def check_entries(check_col, check_list, features, table, entry_return_col, connection, skip_practice,
                  rerun_all=False, rerun_list=[], all_check_list=[]):
    null_dict = {}
    print(f"checking null {check_col} entries")

    if skip_practice is True:
        practice_entries = fetch_tools.fetch_practice(table, entry_return_col, connection)
        check_list = [i for i in check_list if i not in practice_entries]
        all_check_list = [i for i in check_list if i not in practice_entries]
    else:
        if table != 'alloeye_conditions':
            baseline_calibration_entries = fetch_tools.fetch_baseline_calibration(table, entry_return_col, connection)
            check_list = [i for i in check_list if i not in baseline_calibration_entries]
            all_check_list = [i for i in check_list if i not in baseline_calibration_entries]

    df_dict = {check_col: check_list}

    if rerun_all:
        print("\nrerunning all\n")
        check_list = check_list if len(all_check_list) == 0 else all_check_list
        df_dict = {f: [True for i in check_list] for f in features}
        df_dict[check_col] = check_list
    elif len(rerun_list) > 0:
        check_list = rerun_list
        temp_dict = {f: [True for i in check_list] for f in features}
        df_dict = {check_col: rerun_list}
        df_dict = {**df_dict, **temp_dict}
    else:
        for feature in features:
            add_list = []
            df_dict[feature] = []
            null_entries = fetch_tools.fetch_all_null(check_list, check_col, feature, table, check_col, connection)
            for check in check_list:
                if check in null_entries:
                    df_dict[feature].append(True)
                else:
                    df_dict[feature].append(False)

    # df of whether to add feature by checked
    to_add_df = pd.DataFrame(df_dict)
    # bools_only = to_add_df.iloc[:, 1:]
    # df = to_add_df.loc[bools_only.all(axis=1), :]
    # checked_to_add = df.ppt_id.values.tolist()
    # features_to_add = list(bools_only.columns)

    return to_add_df


def batch_by_duplicate_rows(df, id_col, batch_limit):
    # if adding one feature to all or many, get these viewings (duplicates)
    # this is one batch, split into multiple if over 500 or something
    # do same for other duplicate batches
    # for unique entries, batch of one each
    sub_cols = df.columns.difference([id_col])
    subset_df = df[sub_cols]
    dups = df.duplicated(subset=sub_cols, keep=False)
    unique_dups = df.loc[dups, :].drop_duplicates(subset=sub_cols)
    batches = []
    batch_limit = batch_limit
    for i in range(len(unique_dups)):
        unique_dup = unique_dups.drop([id_col], axis=1).iloc[i, :]
        bools = [tuple([unique_dup[col] for col in sub_cols])  # does this unique duplicate row (converted into tuple)
                 == row for row in  # equal this row
                 zip(*[subset_df[col] for col in subset_df])]  # of the subset df
        full_batch = df.iloc[bools, :][id_col].to_list()
        if len(full_batch) < batch_limit:
            batches.append(full_batch)
        else:
            batches.append(full_batch[:batch_limit])
            batch_start = batch_limit
            for j in range(1, math.ceil(len(full_batch) / batch_limit)):
                batch_end = np.min([batch_start + batch_limit, len(full_batch)])
                batches.append(full_batch[batch_start:batch_end])
                batch_start = batch_end
    unique_entries = df.loc[~dups, :][id_col].to_list()
    for i in range(len(unique_entries)):
        batches.append([unique_entries[i]])

    return batches
