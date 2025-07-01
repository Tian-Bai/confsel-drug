from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import numpy as np
import pandas as pd
import sys
import os
import random
from sklearn.model_selection import train_test_split
import argparse
from utility import conf_pval, BH, eval, dice_sim, thresholds_map, Timer

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str) # the name of the dataset to use
parser.add_argument('sample', type=float) # percentage of the dataset to consider
parser.add_argument('seed', type=int)
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)

dataset_name = args.dataset
dataset_path = os.path.join('data', f'{dataset_name}_training_disguised.csv')

dataset = pd.read_csv(dataset_path)

assert 0 < args.sample and args.sample <= 1
if args.sample < 1:
    dataset = dataset.sample(frac=args.sample)

threshold = thresholds_map[dataset_name]

total_Y = dataset['Act'].to_numpy()
total_X = dataset.drop(columns=['MOLECULE', 'Act']).to_numpy()

Xtc, Xtest, Ytc, Ytest = train_test_split(total_X, total_Y, test_size=15/100, shuffle=True) # split 15% as the test data, and the rest for train and calib (tc)

# fdp_nominals = np.round(np.linspace(0.1, 0.5, 9), 2) # nominal FDR levels
fdp_nominals = np.round(np.linspace(0.05, 0.95, 19), 2) # nominal FDR levels
all_res = pd.DataFrame() # results

all_res['fdp_nominals'] = fdp_nominals

''' with binarization + regression '''

# clipped score
rf = RandomForestRegressor(n_estimators=100, max_depth=20, max_features='sqrt')
Xtrain, Xcalib, Ytrain, Ycalib = train_test_split(Xtc, Ytc, train_size=50/85, shuffle=True)
rf.fit(Xtrain, Ytrain < threshold)

fdps_bin_r, pcers_bin_r, powers_bin_r = [], [], []

for i, fdp_nominal in enumerate(fdp_nominals):
    calib_scores = 1000 * (Ycalib < threshold) - rf.predict(Xcalib)
    test_scores = -rf.predict(Xtest)

    pvals = conf_pval(calib_scores, test_scores)
    sel = BH(pvals, fdp_nominal)
    fdp, pcer, power = eval(Ytest, sel, -np.inf, threshold)
    fdps_bin_r.append(fdp)
    pcers_bin_r.append(pcer)
    powers_bin_r.append(power)

all_res['fdps_bin_r'] = fdps_bin_r
all_res['pcers_bin_r'] = pcers_bin_r
all_res['powers_bin_r'] = powers_bin_r

''' with binarization + classification '''

rf = RandomForestClassifier(n_estimators=100, max_depth=20, max_features='sqrt')
Xtrain, Xcalib, Ytrain, Ycalib = train_test_split(Xtc, Ytc, train_size=50/85, shuffle=True)
rf.fit(Xtrain, Ytrain < threshold)

fdps_bin_c, pcers_bin_c, powers_bin_c = [], [], []

for i, fdp_nominal in enumerate(fdp_nominals):
    calib_scores = 1000 * (Ycalib < threshold) - rf.predict_proba(Xcalib)[:, 1] # probability of 1
    test_scores = -rf.predict_proba(Xtest)[:, 1]

    pvals = conf_pval(calib_scores, test_scores)
    sel = BH(pvals, fdp_nominal)
    fdp, pcer, power = eval(Ytest, sel, -np.inf, threshold)
    fdps_bin_c.append(fdp)
    pcers_bin_c.append(pcer)
    powers_bin_c.append(power)

all_res['fdps_bin_c'] = fdps_bin_c
all_res['pcers_bin_c'] = pcers_bin_c
all_res['powers_bin_c'] = powers_bin_c

''' with regression '''

# clipped score

rf = RandomForestRegressor(n_estimators=100, max_depth=20, max_features='sqrt')
Xtrain, Xcalib, Ytrain, Ycalib = train_test_split(Xtc, Ytc, train_size=50/85, shuffle=True)
rf.fit(Xtrain, Ytrain)

fdps_neg, pcers_neg, powers_neg = [], [], []

for i, fdp_nominal in enumerate(fdp_nominals):
    calib_scores = 1000 * (Ycalib < threshold) + rf.predict(Xcalib)
    test_scores = rf.predict(Xtest)

    pvals = conf_pval(calib_scores, test_scores)
    sel = BH(pvals, fdp_nominal)
    fdp, pcer, power = eval(Ytest, sel, -np.inf, threshold)
    fdps_neg.append(fdp)
    pcers_neg.append(pcer)
    powers_neg.append(power)

all_res['fdps_neg'] = fdps_neg
all_res['pcers_neg'] = pcers_neg
all_res['powers_neg'] = powers_neg

# save the results
out_dir = os.path.join('result_ts', f'{dataset_name} {args.sample:.2f}')
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

all_res.to_csv(os.path.join('result_ts', f'{dataset_name} {args.sample:.2f}', f'{dataset_name} {args.sample:.2f} {args.seed}.csv'))
