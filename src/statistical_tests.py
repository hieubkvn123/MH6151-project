import numpy as np
from scipy import stats

def twoSampleZTest(X1, X2, two_tail=True):
    # Sample sizes
    N1, N2 = len(X1), len(X2)
    
    # Calculate mean
    X1_bar = np.mean(X1)
    X2_bar = np.mean(X2)
    
    # Calculate population variance
    sigma1 = np.var(X1)
    sigma2 = np.var(X2)
    
    # Z-statistic + pvalue
    Z = (X1_bar - X2_bar) / np.sqrt(sigma1/N1 + sigma2/N2)
    if(two_tail):
        pvalue = 2 * stats.norm.sf(abs(Z))
    else:
        pvalue = stats.norm.sf(abs(Z))
    return Z, pvalue

def oneSampleTTest(X, mu, two_tail=True):
    # Sample size
    N = len(X)

    # Calculate mean
    Xbar = np.mean(X)

    # Sample variance
    S = (1 / (N-1)) ** np.sum((X - mu)**2)

    # T-statistic + pvalue
    T = (Xbar - mu) / (S / np.sqrt(N))
    if(two_tail):
        pvalue = 2 * stats.t.sf(abs(T), N-1)
    else:
        pvalue = stats.t.sf(abs(T), N-1)
    return T, pvalue

def proportionDiffTest(p0, p1, p2, n1, n2):
    # Z-statistic + pvalue
    Z = (p1 - p2) / np.sqrt(p0 * (1 - p0) * (1/n1 + 1/n2))
    pvalue = 2 * stats.norm.sf(abs(Z))
    return Z, pvalue