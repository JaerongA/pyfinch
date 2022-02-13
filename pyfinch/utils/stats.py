"""
Utility functions for statistical testing
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
    sig = get_sig(stat.pvalue)  # print out asterisk
    return stat, pval, sig


def paired_ttest(arr1, arr2):
    """Performs paired t-test between two arrays"""
    from scipy.stats import ttest_rel
    stat = ttest_rel(arr1, arr2, nan_policy='omit')
    sig = get_sig(stat.pvalue)  # print out asterisk
    degree_of_freedom = len(arr1) - 1
    msg1 = f"t({degree_of_freedom}) = {stat.statistic :.4f}"
    if stat.pvalue < 0.001:  # mark significance
        msg2 = "p < 0.001"
    else:
        msg2 = f"p = {stat.pvalue :.3f}"
    msg = msg1 + ', ' + msg2
    return stat.statistic, stat.pvalue, msg, sig


def two_sample_ttest(arr1, arr2):
    """Performs independent two-sample t-test between two arrays"""
    from scipy import stats
    import numpy as np

    # Remove nan if any
    arr1 = arr1[~np.isnan(arr1)]
    arr2 = arr2[~np.isnan(arr2)]

    tval, pval = stats.ttest_ind(arr1, arr2, nan_policy='omit')
    sig = get_sig(pval)  # print out asterisk
    degree_of_freedom = len(arr1) + len(arr2) - 2
    msg1= f"t({degree_of_freedom}) = {tval : 3f}"
    if pval < 0.001:  # mark significance
        msg2 = 'p < 0.001'
    else:
        msg2 = (f"p={pval :.3f}")
    msg = msg1 + ', ' + msg2
    return tval, pval, msg, sig


def rank_sum_test(arr1, arr2):
    """Performs rank-sum test (non-parametric independent 2-sample test)"""
    from scipy.stats import ranksums
    import numpy as np

    # Remove nan if any
    arr1 = arr1[~np.isnan(arr1)]
    arr2 = arr2[~np.isnan(arr2)]
    z, pval = ranksums(arr1, arr2)
    sig = get_sig(pval)  # print out asterisk
    msg1 = f"Z = {z : .3f}"
    if pval < 0.001:  # mark significance
        msg2 = 'p < 0.001'
    else:
        msg2 = (f"p={pval :.4f}")
    msg = msg1 + ', ' + msg2
    return z, pval, msg, sig


def signed_rank_test(arr1, arr2):
    """Wilcoxon signed-rank test (non-parametric paired test)"""
    from scipy.stats import wilcoxon
    ## Todo: the output slightly different from statview. This needs to be checked.
    z, pval = wilcoxon(arr1, arr2)
    sig = get_sig(pval)  # print out asterisk
    msg1 = f"Z = {z : .3f}"
    if pval < 0.001:  # mark significance
        msg2 = 'p < 0.001'
    else:
        msg2 = (f"p={pval :.4f}")
    msg = msg1 + ', ' + msg2
    return z, pval, msg, sig


def two_sample_ks_test(arr1, arr2, alternative='two-sided'):
    """Performs Kolmogorov-Smirnov test to compare two distributions"""
    from scipy import stats
    import numpy as np

    # Remove nan if any
    arr1 = arr1[~np.isnan(arr1)]
    arr2 = arr2[~np.isnan(arr2)]

    stat, pval = stats.ks_2samp(arr1, arr2, alternative=alternative)
    sig = get_sig(pval)  # print out asterisk
    if pval < 0.001:  # mark significance
        msg = 'p < 0.001'
    else:
        msg = (f"p={pval :.3f}")
    return pval, msg, sig