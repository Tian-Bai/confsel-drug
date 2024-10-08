import pandas as pd
from DeepPurpose import utils, CompoundPred
from tdc.utils import create_fold
import numpy as np
from sklearn.model_selection import train_test_split

'''
Evaluate the selection performace: power and FDP. The region (lower, higher) corresponds to the null hypothesis
'''
def eval(Y, rejected, lower, higher):
    true_reject = np.sum((lower < Y) & (Y < higher))
    if len(rejected) == 0:
        fdp = 0
        power = 0
    else:
        fdp = np.sum((lower >= Y[rejected]) | (Y[rejected] >= higher)) / len(rejected)
        power = np.sum((lower < Y[rejected]) & (Y[rejected] < higher)) / true_reject if true_reject != 0 else 0
    return fdp, power

''' 
Given a list of p-values and nominal FDR level q, apply BH procedure to get a rejection set.
'''
def BH(pvals, q):
    ntest = len(pvals)
         
    df_test = pd.DataFrame({"id": range(ntest), "pval": pvals}).sort_values(by='pval')
    
    df_test['threshold'] = q * np.linspace(1, ntest, num=ntest) / ntest 
    idx_smaller = [j for j in range(ntest) if df_test.iloc[j,1] <= df_test.iloc[j,2]]
    
    if len(idx_smaller) == 0:
        return np.array([])
    else:
        idx_sel = np.array(df_test.index[range(np.max(idx_smaller) + 1)])
        return idx_sel

df = pd.read_csv(f"data/public_admet_data_all.csv")
df = df[['smiles', 'Pgp_human']].dropna()

threshold = 0

encodings = ['DGL_AttentiveFP', 'Morgan', 'CNN', 'rdkit_2d_normalized', 'DGL_GCN', 'DGL_NeuralFP', 'DGL_GIN_AttrMasking', 'DGL_GIN_ContextPred']
drug_encoding = encodings[2]

df = utils.data_process(X_drug = df.smiles.values, y = df.Pgp_human.values, 
                            drug_encoding = drug_encoding,
                            split_method='no_split')

df_split = create_fold(df, 0, [0.5, 0.1, 0.4])
train = df_split['train']
val = df_split['valid']
calibtest = df_split['test'] # calib + test

train['Label'] = train['Label'] < threshold
calibtest['Label'] = calibtest['Label'] < threshold
val['Label'] = val['Label'] < threshold

config = utils.generate_config(drug_encoding = drug_encoding, 
                               train_epoch = 100, 
                               batch_size = 128)

model = CompoundPred.model_initialize(**config)
model.train(train, val)    

Ycalibtest = calibtest.Label.values
Ycalibtest_pred = np.array(model.predict(calibtest))

n = len(Ycalibtest)

q_list = [0.1, 0.2, 0.3, 0.4, 0.5]
fdps, powers = np.zeros(5), np.zeros(5)

for itr in range(100):
    Ycalib, Ytest, Ycalibpred, Ytestpred = train_test_split(Ycalibtest, Ycalibtest_pred, train_size=0.5)
    ntest = len(Ytest)

    for i, q in enumerate([0.1, 0.2, 0.3, 0.4, 0.5]):
        calib_scores = 1000 * (Ycalib > 0) - Ycalibpred
        test_scores = -Ytestpred
        
        pvals = np.zeros(ntest)
        for j in range(ntest):
            pvals[j] = (np.sum(calib_scores < test_scores[j]) + np.random.uniform(size=1)[0] * (np.sum(calib_scores == test_scores[j]) + 1)) / (len(calib_scores) + 1)
        sel = BH(pvals, q)
        fdp, power = eval(Ytest, sel, 0, np.inf)
        fdps[i] += fdp / 100
        powers[i] += power / 100

print(f"conformal - FDP: {fdps}, Power: {powers}")