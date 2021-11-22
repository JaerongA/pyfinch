"""
Functions for statistical testing
"""


def get_sig(pval):
    """Returns asterisk depending on the magnitude of significance"""
    if pval < 0.001:
        sig = '***'
    elif pval < 0.01:
        sig = '**'
    elif pval < 0.05:
        sig = '*'
    else:
        sig = 'ns'  # non-significant
    return sig


def z_test(count1, nbos1, count2, nobs2):
    """Z-test for proportion"""
    import numpy as np
    from statsmodels.stats.proportion import proportions_ztest
    count = np.array([count1, nbos1])
    nobs = np.array([count2, nobs2])
    stat, pval = proportions_ztest(count, nobs)
    return stat, pval


def paired_ttest(arr1, arr2):
    """Performs paired t-test between two arrays"""
    from scipy.stats import ttest_rel
    stat = ttest_rel(arr1, arr2)
    degree_of_freedom = len(arr1) + len(arr2) - 2
    msg1 = f"t({degree_of_freedom}) = {stat.statistic :.2f}"
    if stat.pvalue < 0.001:  # mark significance
        msg2 = "p < 0.001"
    else:
        msg2 = f"p = {stat.pvalue :.3f}"
    msg = msg1 + ', ' + msg2
    return stat, msg


def two_sample_ttest(group_var, dependent_var):
    """Performs independent two-sample t-test between two arrays"""
    from scipy import stats
    group1 = dependent_var[group_var == list(set(group_var))[0]].dropna()
    group2 = dependent_var[group_var == list(set(group_var))[1]].dropna()
    tval, pval = stats.ttest_ind(group2, group1, nan_policy='omit')
    degree_of_freedom = len(group1) + len(group2) - 2
    msg1 = ('t({:.0f})'.format(degree_of_freedom) + ' = {:.2f}'.format(tval))
    if pval < 0.001:  # mark significance
        msg2 = 'p < 0.001'
    else:
        msg2 = ('p = {:.3f}'.format(pval))
    msg = msg1 + ', ' + msg2
    return msg