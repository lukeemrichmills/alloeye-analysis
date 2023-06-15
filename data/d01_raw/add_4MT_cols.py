import pandas as pd
from os import listdir
import re

from src.d00_utils.file_dir_utils import get_doc_dir
from src.d02_intermediate.df_reshaping import add_col_by_lookup


def add_4MT_cols(df, dir_4MT):
    files = listdir(dir_4MT)
    pids, scores, rts = parse_4MT_data(files, dir_4MT)
    score_cols = {}
    rt_cols = {}
    for i in range(15):
        score_cols[f"4MT_T{i+1}"] = scores[i]
        rt_cols[f"4MT_T{i+1}_RT"] = rts[i]

    new_cols = {"pid_4mt": pids, **score_cols, **rt_cols}
    df_4mt = pd.DataFrame(new_cols)
    df['pid'] = df['pid'].astype(str)
    for col_name in df_4mt.columns:
        if col_name != "pid_4mt":
            df = add_col_by_lookup(df, col_name, "pid", df_4mt, "pid_4mt", col_name)

    return df


def parse_4MT_data(filenames, dir):
    scores = [[] for i in range(15)]
    rts = [[] for i in range(15)]
    pids = []
    for file in filenames:
        if 'B' in file:
            pids.append(file.split('B')[-1].split('.')[0])
            file_scores, file_rts = parse_4MT_file(file, dir)
            for i in range(15):
                scores[i].append(file_scores[i])
            for i in range(15):
                rts[i].append(file_rts[i])

    return pids, scores, rts


def parse_4MT_file(file, dir):
    scores = []
    rts = []
    with open(dir+'\\'+file) as f:
        lines = f.readlines()
        lines = lines[14:]
        for line in lines:
            try:
                line_split = re.split("\t+", line)
                if line_split[2] == '\n':   # this means timed out
                    rt, score = [line_split[1], 'INCORRECT']
                    scores.append(score)
                else:
                    rt, score = line_split[2:4]
                    scores.append(re.split("\n+", score)[0])
            except:
                print("catch")

            rts.append(rt)
    return scores, rts

fourMT_dir = get_doc_dir("data\\4MT_data")
p_info_df = pd.read_csv('ppt_info_alloeye.csv')
p_info_df = add_4MT_cols(p_info_df, fourMT_dir)

p_info_df.to_csv('ppt_info_alloeye.csv', index=False)
print("end")


