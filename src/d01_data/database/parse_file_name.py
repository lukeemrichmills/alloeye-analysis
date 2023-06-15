from src.d00_utils.generic_tools import input_to_continue
from src.d01_data.database.File import File
import pandas
from src.d01_data.database.Errors import InvalidValue, CannotParseError
import warnings
from os import listdir


def parse_file_names(raw_data_dir):
    # get all csv files from directory
    only_files = listdir(raw_data_dir)

    # run parse_file_name on each
    csv_files = []  # define empty list to populate
    for f in only_files:    # for each item in directory...
        if f[-4:] == '.csv':    # filter by csv ...
            f_parsed = parse_file_name(f)   # parse through File type...
            if f_parsed is not None:    # if parse succeeded...
                csv_files.append(f_parsed)  # add to list.

    ## could replace with this but less debuggable
    # csv_files = [parse_file_name(f) for f in only_files if f[-4:] == '.csv']

    # convert to dataframe with columns as File attributes
    df_csvs = files_to_df(csv_files)

    # check if duplicates
    # check if duplicates with different appendices
    # - 1. isolate duplicated rows

    # check for pure duplicates i.e. duplicates in all fields
    pure_duplicate_bools = df_csvs.duplicated(keep="first")
    if any(row is True for row in pure_duplicate_bools):
        df_csvs = df_csvs[not pure_duplicate_bools]
        warnings.warn("Pure duplicates found in csv files, first duplicate kept")

    # check for block duplicates i.e. duplicates except for appendix
    block_duplicate_bools = df_csvs.duplicated(subset=df_csvs.columns.difference(['filename', 'appendix']), keep=False)
    block_duplicates = df_csvs[block_duplicate_bools]

    for pID in block_duplicates.pID.unique():
        current_pID = block_duplicates[block_duplicates.pID == pID]
        for block in current_pID.block.unique():
            current_block_duplicates = current_pID[current_pID.block == block]
            filename = current_block_duplicates.filename.iloc[0]

            message = f'{filename} has multiple versions. \n' \
                      f'key f: only first will be uploaded. \n'\
                      f'key s: skip this participant. \n'\
                      f'key b: skip this block'

            while True:
                m = input_to_continue(message)
                if m == 'f':
                    df_csvs = df_csvs.drop(current_block_duplicates.index[1:].values.tolist())
                    break
                elif m == 's':
                    df_csvs = df_csvs[df_csvs.pID != pID]
                    break
                elif m == 'b':
                    df_csvs = df_csvs.drop(current_block_duplicates.index[:].values.tolist())
                    break
                else:
                    print("invalid entry, please select either 'f', 's' or 'b'")

    return df_csvs


def parse_file_name(filename):
    """reads filename and returns File class instance"""
    practice = False
    data_type_list = ("AllGazeData",
                      "TrialGazeData",
                      "TrialInfo",
                      "ObjectPositions",
                      "EventLog")
    data_type = ""
    for d_type in data_type_list:
        if d_type in filename:
            data_type = d_type
            continue

    if data_type == "":
        return None

    name = filename.split('.csv')[0]
    first_three_chars = name[0:3]   # pID will be in first 3 chars
    pID = name.split('r')[0] # test to be number, parse as counterbalanced or not based on pID
    if len(pID) > 3:
        practice = True
        pID = name.split('p')[0]

    no_pID = name.split(pID, maxsplit=1)[1]
    real_or_practice = no_pID[0]    # test to be either r or p, nothing else
    if practice and real_or_practice == 'r':
        raise InvalidValue(real_or_practice, 'p')
    if not practice and real_or_practice == 'p':
        raise InvalidValue(real_or_practice, 'r')

    block = no_pID.split(real_or_practice, maxsplit=1)[1][0]    # test to be 1, 2 or 3, if counterbalanced, otherwise 1 digit

    remaining_name = no_pID.split(block, maxsplit=1)[1]

    appendix = ""
    if remaining_name != data_type:
        appendix = remaining_name.split(data_type)[1]

    valid = True
    if valid:
        output = File(name, pID, practice, block, data_type, appendix)
    else:
        output = 'invalid'

    return output


def files_to_df(files):
    '''
    converts list of custom File type into dataframe line
    :param files: list of alloeye File class
    :return: dataframe where each row is a file and each column a file attribute
    '''
    df = pandas.DataFrame(columns=["name", "pID", "practice", "block", "data_type", "appendix", "filename"])
    for f in files:
        new_row = pandas.DataFrame({'filename': [f.filename],
                                    'pID': [f.pID],
                                    'practice': [f.practice],
                                    'block': [f.block],
                                    'data_type': [f.data_type],
                                    'appendix': [f.appendix]})
        df = pandas.concat([df, new_row], ignore_index=True)
    #df.to_csv("C:\\Users\\Luke Emrich-Mills\\OneDrive\\Documents\\PhD\\AlloEye\\data_pipeline\\tests\\test_outputs\\file_list.csv")
    return df


def parse_appendix_number(file_appendix):
    try:
        output = int(file_appendix)
    except:
        try:
            output = int(file_appendix.split('_')[1])
        except:
            raise CannotParseError(file_appendix)


