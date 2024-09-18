import numpy as np
import pandas as pd 
import math

rf_param = ['n_estim', 'max_depth', 'max_features', 'max_leaf_nodes']
mlp_param = ['hidden', 'layers']

'''
Calculate the conformal p-values and then apply Benjamini-Hochberg procedure to do selection while controlling FDR.
'''
def BH(calib_scores, test_scores, q = 0.1, extra_info=None):
    ntest = len(test_scores)
    ncalib = len(calib_scores)
    pvals = np.zeros(ntest)
    
    for j in range(ntest):
        pvals[j] = (np.sum(calib_scores < test_scores[j]) + np.random.uniform(size=1)[0] * (np.sum(calib_scores == test_scores[j]) + 1)) / (ncalib+1)
         
    # BH(q) 
    df_test = pd.DataFrame({"id": range(ntest), "score": test_scores, "pval": pvals}).sort_values(by='pval')
    
    df_test['threshold'] = q * np.linspace(1, ntest, num=ntest) / ntest 
    idx_smaller = [j for j in range(ntest) if df_test.iloc[j,2] <= df_test.iloc[j,3]]
    
    if len(idx_smaller) == 0:
        if not extra_info:
            return (np.array([]))
        elif extra_info == 'pval':
            return np.array([]), pvals
        else:
            return np.array([]), df_test
    else:
        idx_sel = np.array(df_test.index[range(np.max(idx_smaller)+1)])
        if not extra_info:
            return (idx_sel)
        elif extra_info == 'pval':
            return idx_sel, pvals
        else:
            return idx_sel, df_test

'''
Calculate the conformal p-values and then apply Bonferroni correction to select.
'''
def Bonferroni(calib_scores, test_scores, q = 0.1, extra_info=None):
    return SingleSel(calib_scores, test_scores, q / len(test_scores), extra_info)

'''
Calculate the conformal p-values and go single selection.
'''
def SingleSel(calib_scores, test_scores, q = 0.1, extra_info=None):
    ntest = len(test_scores)
    ncalib = len(calib_scores)
    pvals = np.zeros(ntest)
    
    for j in range(ntest):
        pvals[j] = (np.sum(calib_scores < test_scores[j]) + np.random.uniform(size=1)[0] * (np.sum(calib_scores == test_scores[j]) + 1)) / (ncalib+1)

    df_test = pd.DataFrame({"id": range(ntest), "score": test_scores, "pval": pvals}).sort_values(by='pval')
    
    idxs = [j for j in range(ntest) if df_test.iloc[j,2] <= q]
    if len(idxs) == 0:
        if not extra_info:
            return np.array([])
        elif extra_info == 'pval':
            return np.array([]), pvals
        else:
            return np.array([]), df_test
    else:
        idx_sel = np.array(df_test.index[idxs])
        if not extra_info:
            return idx_sel
        elif extra_info == 'pval':
            return idx_sel, pvals
        else:
            return idx_sel, df_test

def RegionSel(calib_scores, test_scores, q = 0.1):
    ntest = len(test_scores)
    ncalib = len(calib_scores)

    calib_scores = np.sort(calib_scores)
    d = ncalib - math.ceil((1 - q) * (1 + ncalib))
    if d >= 0:
        threshold = calib_scores[d]
    else:
        threshold = -math.inf
    return [j for j in range(len(test_scores)) if 1 + test_scores[j] > threshold and test_scores[j] <= threshold]
