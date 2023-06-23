import pandas as pd
from os import listdir
import re

from src.d00_utils.file_dir_utils import get_doc_dir
from src.d02_intermediate.df_reshaping import add_col_by_lookup

n_trials = 15

def add_4MT_cols(df, dir_4MT):
    files = listdir(dir_4MT)
    pids, scores, rts, selections = parse_4MT_data(files, dir_4MT)
    score_cols, rt_cols, sel_cols = {}, {}, {}
    for i in range(n_trials):
        score_cols[f"4MT_T{i+1}"] = scores[i]
        rt_cols[f"4MT_T{i+1}_RT"] = rts[i]
        sel_cols[f"4MT_T{i+1}_loc"] = selections[i]

    new_cols = {"pid_4mt": pids, **score_cols, **rt_cols, **sel_cols}
    df_4mt = pd.DataFrame(new_cols)
    df['pid'] = df['pid'].astype(str)
    for col_name in df_4mt.columns:
        if col_name != "pid_4mt":
            df = add_col_by_lookup(df, col_name, "pid", df_4mt, "pid_4mt", col_name)

    return df


def parse_4MT_data(filenames, dir):

    scores = [[] for i in range(n_trials)]
    rts = [[] for i in range(n_trials)]
    selections = [[] for i in range(n_trials)]
    pids = []
    for file in filenames:
        if 'B' in file:
            pids.append(file.split('B')[-1].split('.')[0])
            file_scores, file_rts, file_selections = parse_4MT_file(file, dir)
            for i in range(n_trials):
                scores[i].append(file_scores[i])
                rts[i].append(file_rts[i])
                selections[i].append(file_selections[i])

    return pids, scores, rts, selections


def parse_4MT_file(file, dir):
    scores = []
    rts = []
    locs = []
    with open(dir+'\\'+file) as f:
        lines = f.readlines()
        lines = lines[14:]
        for line in lines:
            try:
                line_split = re.split("\t+", line)
                if line_split[2] == '\n':   # this means timed out
                    selection, rt, score = ['TIMED', line_split[1], 'INCORRECT']
                    scores.append(score)
                    locs.append(selection)
                else:
                    selection, rt, score = line_split[1:4]
                    scores.append(re.split("\n+", score)[0])
                    locs.append(selection)
            except Exception as e:
                raise e

            rts.append(rt)

    return scores, rts, locs

fourMT_dir = get_doc_dir("data\\4MT_data")
p_info_dir = '../../data/d01_raw'
p_info_df = pd.read_csv(f'{p_info_dir}/ppt_info_alloeye.csv')
p_info_df = add_4MT_cols(p_info_df, fourMT_dir)

p_info_df.to_csv(f'{p_info_dir}/ppt_info_alloeye.csv', index=False)
print("end")


