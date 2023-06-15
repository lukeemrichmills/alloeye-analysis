import math

import pandas as pd
from src.d00_utils.DataLevel import *
from src.d01_data.fetch.fetch_data_level import fetch_data_level


def batch_fetch(data_level: DataLevel, table: str, batch_col: str, batch_list: list,
                batch_limit: int = 500, pid="all", fetch_kwargs=None, skip_practice=True):
    """
    fetch data_level entries from table in batches of batch_limit
    use **kwargs (keyword arguments) for specifying DataLevel-specific arguments for fetching.
    First kwarg key MUST be the column for batching and first kwarg value must be batch_list e.g.
    viewing_id=viewing_list if timepoints

    :param data_level: DataLevel enum
    :param table: string matching appropriate table
    :param batch_limit: int size of batches
    :param batch_col: str column to fetch from in batches
    :param batch_list: list of ids from batch_col for batching
    :param pid: list or str of pids to fetch from
    :return: pandas dataframe of entries
    """


    print("- fetching data from db")
    practice_values = [False] if skip_practice else [True, False]
    # batch fetch algo
    next_batch_no = 0
    last_batch_no = 0
    batch_kwarg = fetch_kwargs if fetch_kwargs is not None else {}
    batch_no = 1
    if len(batch_list) > batch_limit:
        print(f"fetching {data_level} data in batches of {batch_limit}")
        print(f"fetching batch {batch_no}...")
        next_batch_no += batch_limit
        batch_kwarg[batch_col] = batch_list[last_batch_no:next_batch_no]
        all_fetched = fetch_data_level[data_level](pid=pid,
                                                   table=table,
                                                   practice=practice_values,
                                                   **batch_kwarg)
        last_batch_no = next_batch_no
        for i in range(1, math.ceil(len(batch_list) / batch_limit)):
            batch_no += 1
            print(f"fetching batch {batch_no}...")
            next_batch_no += batch_limit
            if next_batch_no > len(batch_list):
                next_batch_no = len(batch_list)

            batch_kwarg[batch_col] = batch_list[last_batch_no:next_batch_no]
            next_fetched = fetch_data_level[data_level](pid=pid,
                                                        table=table,
                                                        practice=practice_values,
                                                        **batch_kwarg)

            all_fetched = pd.concat([all_fetched, next_fetched]).reset_index(drop=True)
            last_batch_no = next_batch_no
    else:
        batch_kwarg[batch_col] = batch_list
        all_fetched = fetch_data_level[data_level](pid=pid,
                                                   table=table,
                                                   practice=practice_values,
                                                   **batch_kwarg)
    print(f"{data_level}s retrieved.")

    return all_fetched


