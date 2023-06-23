import pandas as pd
import numpy as np
from statsmodels.regression.linear_model import GLS
from scipy.stats import ttest_ind

# define a function that fits GLS and return intercept and slope
from src.d01_data.database.Errors import InvalidInput

from statsmodels.regression.linear_model import GLS

from src.d05_modelling import stats


def participant_ols(formula, df_p, ws_method):

    # RNORM VERSION
    if ws_method == 'rnorm':
        model = GLS.from_formula(formula, df_p)
        results = model.fit()
        res_sd = results.resid.std()
        intcpt = results.params[0]
        slope = results.params[1]
        seven_sec = intcpt + 7000 * slope
        random_intcpt = np.random.normal(loc=intcpt, scale=res_sd)
        random_seven_sec = np.random.normal(loc=seven_sec, scale=res_sd)
        random_slope = (random_seven_sec - random_intcpt) / 7000

    # FULL DATA VERSION
    elif ws_method == 'none':
        model = GLS.from_formula(formula, df_p)
        results = model.fit()
        random_slope = results.params[1]
        random_intcpt = results.params[0]

    else:
        raise InvalidInput(f"ws_method input {ws_method} is invalid")

    return {'slope': random_slope, 'intercept': random_intcpt}


def sample_ols(formula, ppts, df_g, ws_method):
    n_ppts = len(ppts)

    ppt_slopes = np.zeros(n_ppts)
    ppt_intcpt = np.zeros(n_ppts)

    # for loop participant
    for p in range(n_ppts):
        df_p = df_g[df_g['participant'] == ppts[p]]

        ppt_out = participant_ols(formula, df_p, ws_method)

        ppt_slopes[p] = ppt_out['slope']
        ppt_intcpt[p] = ppt_out['intercept']

    mean_slope = ppt_slopes.mean()
    mean_int = ppt_intcpt.mean()

    return mean_int, mean_slope


def bootline(x, y, df, groups='from_data', ws_method='rnorm',
             B_outer=100, B_middle='from_data', B_inner=5, seed=123,
             suppress_print=False):
    # define linear function
    formula_str = y + " ~ " + x

    # set random seed for consistency
    np.random.seed(seed)

    # define groups
    if groups == 'from_data':
        groups = df['group'].unique()
    n_groups = len(groups)

    # define within-group repetitions. If from_data, use size of smallest group
    if B_middle == 'from_data':
        n_min = df['participant'].nunique() + 1  # start high

        for group in groups:
            df_g = df[df['group'] == group]
            n = df_g['participant'].nunique()
            if n < n_min:
                n_min = n
        B_middle = n_min


    # predefine group data list
    group_slopes = []
    group_intrcpt = []

    # for loop group
    for g in range(len(groups)):
        if not suppress_print:
            print('#####', groups[g], 'group', g, 'of', len(groups), '#####')
        boot_slopes = np.zeros(B_outer)
        boot_intrcpts = np.zeros(B_outer)

        for i in range(B_outer):
            if not suppress_print:
                print('Outer loop', i, 'of', B_outer)
            df_g = df[df['group'] == groups[g]]
            ppts = df_g['participant'].unique()

            ppt_sample = np.random.choice(ppts, size=B_middle, replace=True)
            sample_results = sample_ols(formula_str, ppt_sample, df_g, ws_method)

            boot_slopes[i] = sample_results[1]
            boot_intrcpts[i] = sample_results[0]

        group_slopes.append(boot_slopes)
        group_intrcpt.append(boot_intrcpts)

    # return some grouped statistics such as AUC, first and last difference
    t = np.sort(np.unique(df['time']))
    group_mats = []
    group_onesec = [None] * len(group_slopes)
    group_sevensec = [None] * len(group_slopes)
    group_auc = [None] * len(group_slopes)


    for g in range(len(groups)):
        slopes = group_slopes[g]
        ints = group_intrcpt[g]
        result = np.empty((len(t), len(slopes)))
        result[:] = np.nan

        for i in range(len(slopes)):
            result[:, i] = linfun(t, slopes[i], ints[i])

        group_onesec[g] = result[0, :]
        group_sevensec[g] = result[-1, :]
        group_auc[g] = np.sum(result, axis=0)
        group_mats.append(result)

    np.random.seed(None)

    return {"group_mats": group_mats, "group_slopes": group_slopes, "group_intrcpt": group_intrcpt,
            "group_first": group_onesec, "group_last": group_sevensec, 'group_auc': group_auc, "x": t}


def linfun(t, slope, intercept):
    return (t * slope) + intercept



def get_tstat_clusters(t, group1_mat, group2_mat, t_threshold, alpha=0.05):
    alpha_bonf = 0.05  # / len(t)
    clusters = []
    cluster_ind = 0
    cluster = {'times': [], 'tstats': [], 'cohensD': [], 'pvals': []}
    empty_cluster = cluster.copy()

    all_ts = np.zeros(len(t))
    all_ds = np.zeros(len(t))
    all_ps = np.zeros(len(t))

    for ti in range(len(t)):
        g1 = group1_mat[ti, :]
        g2 = group2_mat[ti, :]
        t_stat, p_val = ttest_ind(g1, g2)
        coh_d = stats.cohensD(g1, g2)

        all_ts[ti] = t_stat
        all_ds[ti] = coh_d
        all_ps[ti] = p_val

        if abs(t_stat) > t_threshold and p_val < alpha_bonf:
            cluster['tstats'].append(t_stat)
            cluster['times'].append(t[ti])
            cluster['cohensD'].append(coh_d)
            cluster['pvals'].append(p_val)
        else:
            if len(cluster['tstats']) > 0:
                clusters.append(cluster)
            cluster = empty_cluster.copy()
            cluster_ind += 1
    if len(cluster['tstats']) > 0:
        clusters.append(cluster)

    return {'clusters': clusters, 'all_tstats': all_ts, 'all_ds': all_ds, 'all_ps': all_ps}
