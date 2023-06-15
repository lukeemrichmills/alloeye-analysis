import os.path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.d01_data.fetch.fetch_trials import fetch_trials
from src.d01_data.database.Errors import InvalidValue

from src.d02_intermediate.df_reshaping import *
from data import d01_raw
from plotnine import *

# get data
pid_info = pd.read_csv(f"{os.path.abspath(d01_raw.__path__[0])}\ppt_info_alloeye.csv")
# pid = ['25', '26', '3', '4']
pid_info['alloeye_id'] = pid_info.pid.apply(lambda s: 'alloeye_' + str(s))
pid = 'all'
cols = ['ppt_id', 'object_shifted', 'selected_object', 'move_type', 'table_rotates']
df = fetch_trials(pid, conditions=Conditions.all, trials="all")

# reduce columns, add condition columns dummy variables
df = df[cols]
df = convert_condition_cols(df)  # gets 6 condition columns, value is true if that condition

# add correct column by trial
df['correct'] = (df['object_shifted'] == df['selected_object'])

# by group overall
df_sum = df.groupby(['ppt_id'], as_index=False).sum()
df_sum['n_trials'] = (df_sum[Conditions.list[0]] + df_sum[Conditions.list[1]] + df_sum[Conditions.list[2]] +
                      df_sum[Conditions.list[3]] + df_sum[Conditions.list[4]] + df_sum[Conditions.list[5]])
df_sum['p_correct'] = df_sum['correct'] / df_sum['n_trials']
df_sum = add_col_by_lookup(df_sum, 'group', 'ppt_id', pid_info, 'alloeye_id', 'group')
p1 = (ggplot(df_sum, aes('group', 'p_correct', fill='group'))
      + geom_boxplot(outlier_shape="")
      + xlab(['Older', 'Younger'])
      + geom_point(position=position_jitterdodge()))

# by condition per group
df_sum, new_cols = feature_per_condition(df, 'correct', groupby='ppt_id', groupfunc='proportion')
df_long = df_sum.reset_index().melt(id_vars='ppt_id', value_vars=new_cols,
                                    var_name='condition', value_name='score')
df_long = add_col_by_lookup(df_long, 'group', 'ppt_id', pid_info, 'alloeye_id', 'group')

# plot with plotnine
p = (ggplot(df_long, aes('condition', 'score', fill='group'))
     + geom_boxplot(outlier_shape="")
     + xlab(Conditions.list)
     + geom_point(position=position_jitterdodge()))
# fig = p.draw()
# fig.show()
# grouped_group = conds_sum.groupby(['group'], as_index=False)
# groups_sum = grouped_group.sum()

# plt.bar(groups_sum.index, groups_sum)
plt.show()
