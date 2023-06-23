import numpy as np


def cohensD(group1, group2):
    """
    Calculate Cohen's d.
    """
    n1 = len(group1)
    n2 = len(group2)
    df = n1 + n2 - 2
    s_pooled = np.sqrt(((n1 - 1) * np.var(group1, ddof=1) + (n2 - 1) * np.var(group2, ddof=1)) / df)
    d = (np.mean(group1) - np.mean(group2)) / s_pooled
    return d