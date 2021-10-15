

def z_test(count1, nbos1, count2, nobs2):
    """Z-test for proportion"""
    import numpy as np
    from statsmodels.stats.proportion import proportions_ztest
    count = np.array([count1, nbos1])
    nobs = np.array([count2, nobs2])
    stat, pval = proportions_ztest(count, nobs)
    return stat, pval

