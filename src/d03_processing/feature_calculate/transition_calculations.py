import math
import numpy as np
import json

from src.d00_utils.TaskObjects import TaskObjects
from src.d01_data.database.Errors import UnmatchingValues
from src.d03_processing import aoi
from src.d03_processing.feature_calculate.fix_sacc_calculations import n_fix, dwell_time
from src.d03_processing.fixations.FixationProcessor import FixationProcessor


def n_transitions(prob_matrix, objects, ext_fix_df):
    """return the number of transitions between areas of interest"""

    return n_fix(ext_fix_df, objects='total') - 1


def gini_fix(prob_matrix, objects, ext_fix_df):
    return gini_fix_df(ext_fix_df, objects, n_fix)


def gini_dwell(prob_matrix, objects, ext_fix_df):
    return gini_fix_df(ext_fix_df, objects, dwell_time)


def gini_refix(prob_matrix, objects, ext_fix_df):
    return gini_refixations(ext_fix_df, objects, n_fix)


def gini_redwell(prob_matrix, objects, ext_fix_df):
    return gini_refixations(ext_fix_df, objects, dwell_time)


def gini_refixations(ext_fix, objects, pi_func):
    refix = refixations_only(ext_fix)
    return gini_fix_df(refix, objects, pi_func)


def gini_fix_df(fix_df, objects, pi_func):
    fix_df = aoi.convert_AOIs(fix_df.copy(deep=True))
    standard_objects = TaskObjects.standard_aois
    pi = stationary_prob(fix_df, standard_objects, prob_func=pi_func, adjust_missing_array_objects=True)
    return gini(pi)


def gini(array):
    """Calculate the Gini coefficient of a numpy array. Used here to quantify the uniformity of a fixation
    probability distribution"""
    array = np.abs(array)   # Values must be sorted:
    array.sort()   # Index per array element:
    index = np.arange(1, array.shape[0]+1)   # Number of array elements:
    n = array.shape[0]
    # Gini coefficient calculation:
    gini = ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array)))
    return gini


def Hn(prob_matrix, objects, ext_fix_df):
    """
    calls transition entropy function where staionary probability is defined by number of fixations on each object
    :param fix_df:
    :return:
    """
    return transition_entropy(prob_matrix, objects, ext_fix_df, n_fix)


def Hd(prob_matrix, objects, ext_fix_df):
    """
    calls transition entropy function where staionary probability is defined by dwell time on each object
    :param prob_matrix:
    :return:
    """
    return transition_entropy(prob_matrix, objects, ext_fix_df, dwell_time)


def Ht(prob_matrix, objects, ext_fix_df):
    """
    calls transition entropy function where stationary distribution is defined by probability of making that transition
    - this will usually end up as n_fix - 1 for number of transitions
    :param prob_matrix:
    :param args:
    :return:
    """
    return transition_entropy_2(prob_matrix)


def Hn_s(prob_matrix, objects, ext_fix_df):
    """
    same as Hn but using standardise area of interest names
    :param fix_df:
    :return:
    """
    return H_standardised(prob_matrix, objects, ext_fix_df, Hn)


def Hd_s(prob_matrix, objects, ext_fix_df):
    """
    same as Hd but using standardise area of interest names
    :param fix_df:
    :return:
    """
    return H_standardised(prob_matrix, objects, ext_fix_df, Hd)


def Ht_s(prob_matrix, objects, ext_fix_df):
    """
    same as Ht but using standardise area of interest names
    :param fix_df:
    :return:
    """
    return H_standardised(prob_matrix, objects, ext_fix_df, Ht)


def H_standardised(prob_matrix, objects, ext_fix_df, H_func):
    fix_df = aoi.convert_AOIs(ext_fix_df.copy(deep=True))
    s_prob_matrix, s_objects = transition_matrix(fix_df, prob=True, split_by_missing_group=False)
    return H_func(s_prob_matrix, s_objects, fix_df)


def p_matrix(prob_matrix, *args) -> str:
    """
    returns string of 2-d p_matrix for SQL storage
    NOTE: to parse back, can do np.array(json.loads(matrix_str))
    :param prob_matrix:
    :param args:
    :return:
    """

    return json.dumps(prob_matrix.tolist())


def p_matrix_objects(prob_matrix, objects, *args) -> str:
    """
    returns string of 1-d objects corresponding to p_matrix row/column order
    NOTE: to parse back, can do np.array(json.loads(array_str))
    :param prob_matrix: ignore this
    :param objects:
    :param args:
    :return:
    """
    return json.dumps(objects.tolist())




def refixations_only(ext_fix_df):
    ext_fix_df['PrevFixations'] = ext_fix_df.groupby(['object']).cumcount()
    df_refixations = ext_fix_df[ext_fix_df['PrevFixations'] > 0]
    return df_refixations


def external_fixations(fix_df):
    # make sure external fixations i.e. transitions between fixations not within
    ext_fix_df = fix_df[fix_df['fixation_or_saccade'] == 'fixation'].sort_values(by='start_time').reset_index(drop=True)
    if len(ext_fix_df) <= 1:
        return ext_fix_df
    i = 1
    end_no = len(ext_fix_df)
    while i < end_no:
        previous = i - 1
        current = i
        if ext_fix_df.object[current] == ext_fix_df.object[previous]:
            ext_fix_df = FixationProcessor.combine_fixations(previous, current, ext_fix_df)
            end_no = len(ext_fix_df)
            i -= 1
        i += 1

    return ext_fix_df


def transition_matrix(ext_fix_df, prob=False, split_by_missing_group=True, missing_group_threshold=2000):
    objects = np.unique(ext_fix_df.object)
    n_objects = len(objects)
    matrix_length = n_objects
    transition_matrix = np.array(np.zeros([matrix_length, matrix_length]))
    transitions_from = []
    transitions_to = []
    if split_by_missing_group:
        df_list = [group for _, group in ext_fix_df.groupby('missing_split_group') if not group.empty]
    else:
        df_list = [ext_fix_df]
    for j in range(len(df_list)):
        df = df_list[j].reset_index(drop=True)
        for i in range(1, len(df)):
            previous_object = df.object[i - 1]
            current_object = df.object[i]
            if current_object != previous_object:
                transitions_from.append(previous_object)
                transitions_to.append(current_object)

    n_transitions = len(transitions_to)
    for i in range(n_transitions):
        from_object = transitions_from[i]
        to_object = transitions_to[i]
        from_index = np.where(objects == from_object)[0][0]
        to_index = np.where(objects == to_object)[0][0]
        try:
            transition_matrix[from_index, to_index] += 1
        except IndexError:
            raise IndexError

    if prob is True:        # probability matrix - divides by number of transitions
        transition_matrix = transition_matrix / n_transitions

    return transition_matrix, objects


def stationary_prob(ext_fix_df, objects, prob_func, adjust_missing_array_objects=False):
    # areas of interest
    n_objects = len(objects)
    if adjust_missing_array_objects:
        n_array_objects = len(np.unique(ext_fix_df.object[np.isin(ext_fix_df.object, TaskObjects.array_objects)]))
        # print('n_array_objects', n_array_objects)
        n_objects += np.max(4 - n_array_objects, 0)   # add space for missing array objects
    # print('n_objects', n_objects)
    # print('len objects', len(objects))
    stationary_prob = np.zeros(n_objects, dtype=float)

    # if no fixations, assume uniform distribution
    if len(ext_fix_df) == 0:
        stationary_prob += 1 / len(stationary_prob)
        return stationary_prob

    total_denom = prob_func(ext_fix_df, objects='total')

    for i in range(len(objects)):
        # print(i)
        numerator_prob = prob_func(ext_fix_df, objects[i])
        try:
            stationary_prob[i] = numerator_prob / total_denom
        except ZeroDivisionError as e:
            raise e

    return stationary_prob


def transition_entropy(prob_matrix, objects, ext_fix_df, pi_func=n_fix):
    """
    calculates transition entropy by converting fixation/saccade dataframe into external fixation df.
    :param fix_df:
    :param pi_func: stationary probability distribution based on either number of fixations or duration of fixations on
    objects
    :return:
    """
    # get stationary probability
    pi = stationary_prob(ext_fix_df, objects, pi_func)

    # get conditional probability matrix
    cond_matrix = conditional_prob_matrix(prob_matrix)

    # get H from n fixations
    return shannon_entropy(pi, cond_matrix)


def transition_entropy_2(prob_matrix):
    """
    calculates transition entropy by converting fixation/saccade dataframe into external fixation df.
    :param fix_df:
    :param pi_func:
    objects
    :return:
    """

    # get conditional probability matrix
    cond_matrix = conditional_prob_matrix(prob_matrix)

    # get H from n fixations
    return shannon_entropy_alt(prob_matrix, cond_matrix)


def conditional_prob_matrix(transition_matrix, prob=False):
    """

    :param transition_matrix:
    :param prob:
    :return:
    """
    cond_matrix = np.array(np.zeros(transition_matrix.shape))
    for i in range(len(transition_matrix)):
        for j in range(len(transition_matrix)):
            p_i = np.sum(transition_matrix[i, :])  # P(A and B)
            p_ij = transition_matrix[i, j]          # P(B)

            cond_matrix[i, j] = 0 if p_i == 0 else p_ij / p_i

    return cond_matrix


def transition_entropy_correction(markov_matrix, h):
    # for ii = 1:size(markovMatrix, 1)
    # rowTotals(ii, 1) = calculateEntropy(sum(markovMatrix(ii,:)));
    # columnTotals(ii, 1) = calculateEntropy(sum(markovMatrix(:, ii)));
    # end
    #
    # rowAndColTotals = sum(rowTotals(~isnan(rowTotals))) + sum(columnTotals(~isnan(columnTotals)));
    # H1 = 1 - ((rowAndColTotals - H) / (rowAndColTotals / 2));
    row_totals = np.zeros(len(markov_matrix))
    col_totals = np.zeros(len(markov_matrix))
    for i in range(len(markov_matrix)):
        row_totals[i] = p_log_p(np.sum(markov_matrix[i, :]))
        col_totals[i] = p_log_p(np.sum(markov_matrix[:, i]))

    row_and_col_totals = np.nansum(row_totals) + np.nansum(col_totals)
    return 1 - ((row_and_col_totals - h) / (row_and_col_totals / 2))


def p_log_p(p, log=2):
    if log == 2:
        return p * np.log2(p)
    elif log == "ln":
        return p * np.log(p)


def p1_log_p2(p1, p2, log=2):
    if log == 2:
        return p1 * np.log2(p2)
    elif log == "ln":
        return p1 * np.log(p2)


def shannon_entropy_alt(joint_matrix, conditional_matrix):
    """
    shannon entropy based on joint probability multiplied by log of conditional probability
    :param joint_matrix: joint probability matrix n x n
    :param conditional_matrix: conditional probability matrix n x n
    :return: scalar h
    """
    n = joint_matrix.shape[0]
    assert joint_matrix.shape[1] == n
    assert conditional_matrix.shape[0] == n
    assert conditional_matrix.shape[1] == n

    h = 0
    for i in range(n):
        for j in range(n):
            try:
                if joint_matrix[i, j] != 0:
                    h += p1_log_p2(joint_matrix[i, j], conditional_matrix[i, j])
            except IndexError:
                raise IndexError

    return -1 * h


def shannon_entropy(pi, p):
    """
    calculation that sums the conditional probability of each transition by the log of that conditional probability. Varies from
    other method ('transfer_entropy()') which multiplies joint prob by log of conditional prob.
    :param pi: stationary distribution - vector of length n where n is number if objects
    and each element is stationary probability of dwelling on that object
    :param p: conditional probability matrix - matrix of size n x n
    :return: - sum i, j (pi_i * p_ij * log(p*ij))
    """
    n = len(pi)

    h = 0
    for i in range(n):
        for j in range(n):
            try:
                if p[i, j] != 0:
                    h += pi[i] * p_log_p(p[i, j])
                    if h > 1:
                        print("error?")
            except IndexError:
                raise IndexError

    return -1 * h


def p_log_pq(p, q, log="ln"):
    if log == "ln":
        return p * np.log(p/q)
    else:
        return p * math.log(p/q, log)


def relative_entropy(p, q):
    n_p = p.shape[0]
    n_q = q.shape[0]
    c = 0.00001
    if n_p != n_q:
        raise UnmatchingValues(message="matrices should be same size")
    dkl = 0
    for i in range(n_p):
        for j in range(n_p):
            try:
                if i != j:
                    dkl += p_log_pq(p[i, j] + c, q[i, j] + c)
            except IndexError:
                raise IndexError

    return dkl


def symmetric_KL(p, q):
    dkl_pq = relative_entropy(p, q)
    dkl_qp = relative_entropy(q, p)
    return float(dkl_qp + dkl_pq)



