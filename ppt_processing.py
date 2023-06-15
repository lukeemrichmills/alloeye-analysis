def ppt_processing(pID, dir):
    '''
    Function to process individual participant AlloEye datasets.
    Searches for and combines blocks automatically.
    :param pID: participant ID
    :param dir: file directory
    :return: 1-by-m dataframe with m participant-wide features/metrics,
        t-by-m dataframe with t trials and m trial-wide features/metrics
    '''

    # 1. imports and housekeeping

    # 2. default values and fixed variables
    pIDs_5_objects = ["A1", "A2", "A3", "A4", "A5", "A7", "A8", "A9", "A10"]
    pIDs_4obj_not_counterbalanced = ["001", "002", "003",  "004",  "005"]
    pIDs_not_counterbalanced = pIDs_5_objects + pIDs_4obj_not_counterbalanced
    counterbalanced = pID in pIDs_not_counterbalanced
    five_objects = pID in pIDs_5_objects

    # 3. file import(s) (with unit testing)

    # 4. raw data cleaning
    # 5. calculate eye movement metrics by viewing, trial and participant
    # call methods from different calculation classes, split by category e.g. fixation calcs, markov calcs ??
    # 6. export and return