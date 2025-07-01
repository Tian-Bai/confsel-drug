import numpy as np
import pandas as pd 
import math
import time

# A map of constants for the target regions of each dataset
# for each dataset `d`, the target region is (-infinity, thresholds_map[d]).
thresholds_map = {'NK1': 6.5, 'PGP': -0.3, 'LOGD': 1.5, '3A4': 4.35, 'CB1': 6.5, 'DPP4': 6, 'HIVINT': 6, 'HIVPROT': 4.5, 'METAB': 40, 'OX1': 5, 'OX2': 6, 'PPB': 1, 'RAT_F': 0.3, 'TDI': 0, 'THROMBIN': 6}
offsets_map = {'NK1': 0.2, 'PGP': 0.1, 'LOGD': 0.2, '3A4': 0.1, 'CB1': 0.2, 'DPP4': 0.3, 'HIVINT': 0.1, 'HIVPROT': 0.2, 'METAB': 10, 'OX1': 0.5, 'OX2': 0.5, 'PPB': 0.3, 'RAT_F': 0.1, 'TDI': 0.1, 'THROMBIN': 1}

def conf_pval(calib_scores, test_scores):
    r"""
    Compute the "clipped" conformal p-values given the calibration and test nonconformity scores.

    Args:
        calib_scores (np.ndarray): 1-d array of calibration nonconformity scores.
        test_scores (np.ndarray): 1-d array of the test nonconformity scores.

    Returns:
        np.ndarray: 1-d array of conformal p-values, with size equal to that of `test_scores`.
    """
    ntest, ncalib = len(test_scores), len(calib_scores)
    pvals = np.zeros(ntest)
    
    for j in range(ntest):
        pvals[j] = (np.sum(calib_scores < test_scores[j]) + np.random.uniform() * (np.sum(calib_scores == test_scores[j]) + 1)) / (ncalib + 1)
    return pvals

def BH(pvals, q):
    r""" 
    Compute the BH rejection set given p-values and a nominal FDR level `q`.

    Args:
        pvals (np.ndarray): 1-d array of p-values to consider.
        q (float): nominal FDR level, in (0,1). 

    Returns:
        np.ndarray: 1-d array of the indices of the rejected p-values.
    """  
    ntest = len(pvals)
    df_test = pd.DataFrame({"id": range(ntest), "pval": pvals}).sort_values(by='pval')
    
    df_test['threshold'] = q * np.linspace(1, ntest, num=ntest) / ntest 
    idx_smaller = [j for j in range(ntest) if df_test.iloc[j,1] <= df_test.iloc[j,2]]
    
    if len(idx_smaller) == 0:
        return np.array([])
    else:
        idx_sel = np.array(df_test.index[range(np.max(idx_smaller) + 1)])
        return idx_sel
    
def eval(Y, rejected, lower, higher):
    r""" 
    Evaluate the selection correctness given the true values, the selected subsets, and the target region (lower,higher).

    Args:
        Y (np.ndarray): 1-d array of true responses (activity levels).
        rejected (np.ndarray): 1-d array of the indices of rejected hypotheses.
        lower, higher (float, float): The boundaries defining the desirable values of Y, which is (lower,higher).

    Returns:
        float, float, float: The empirical FDP, PCER and Power of the selection.
    """
    true_reject = np.sum((lower < Y) & (Y < higher))
    if len(rejected) == 0:
        fdp = 0
        pcer = 0
        power = 0
    else:
        if np.isscalar(lower) or np.ndim(lower) == 0:
            lower = np.full_like(Y, lower)
        if np.isscalar(higher) or np.ndim(higher) == 0:
            higher = np.full_like(Y, higher)
        fdp = np.sum((lower[rejected] >= Y[rejected]) | (Y[rejected] >= higher[rejected])) / len(rejected)
        pcer = np.sum((lower[rejected] >= Y[rejected]) | (Y[rejected] >= higher[rejected])) / len(Y)
        power = np.sum((lower[rejected] < Y[rejected]) & (Y[rejected] < higher[rejected])) / true_reject if true_reject != 0 else 0
    return fdp, pcer, power

def dice_sim(f, set_of_f):
    r"""
    Compute the dice similarity between a data point `f` and a batch of data `set_of_f`.

    Args:
        f: (np.ndarray): 1-d array of size (feature, ) representing the data point.
        set_of_f: (np.ndarray): 2-d array of size (N, feature) representing the batch of data.

    Returns:
        float: The computed dice similarity.
    """
    f = np.array(f)
    set_of_f = np.array(set_of_f) 

    min_count = np.minimum(f, set_of_f).sum(axis=1) * 2
    sum_count = f.sum() + set_of_f.sum(axis=1)
    sum_count[sum_count == 0] = 1

    return np.max(min_count / sum_count)

class Timer:
    r"""
    A context manager class for timing the execution of a block of code.
    """
    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.end_time = time.time()
        self.runtime = self.end_time - self.start_time