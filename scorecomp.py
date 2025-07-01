from sklearn.ensemble import RandomForestRegressor
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

''' without sigma '''

# clipped score
rf = RandomForestRegressor(n_estimators=100, max_depth=20, max_features='sqrt')
Xtrain, Xcalib, Ytrain, Ycalib = train_test_split(Xtc, Ytc, train_size=50/85, shuffle=True)
rf.fit(Xtrain, Ytrain < threshold)

fdps_clip, pcers_clip, powers_clip = [], [], []

for i, fdp_nominal in enumerate(fdp_nominals):
    calib_scores = 1000 * (Ycalib < threshold) - rf.predict(Xcalib)
    test_scores = -rf.predict(Xtest)

    pvals = conf_pval(calib_scores, test_scores)
    sel = BH(pvals, fdp_nominal)
    fdp, pcer, power = eval(Ytest, sel, -np.inf, threshold)
    fdps_clip.append(fdp)
    pcers_clip.append(pcer)
    powers_clip.append(power)

all_res['fdps_clip'] = fdps_clip
all_res['pcers_clip'] = pcers_clip
all_res['powers_clip'] = powers_clip

# signed error score
rf = RandomForestRegressor(n_estimators=100, max_depth=20, max_features='sqrt')
Xtrain, Xcalib, Ytrain, Ycalib = train_test_split(Xtc, Ytc, train_size=50/85, shuffle=True)
rf.fit(Xtrain, Ytrain)

fdps_res, pcers_res, powers_res = [], [], []

for i, fdp_nominal in enumerate(fdp_nominals):
    calib_scores = -Ycalib - rf.predict(Xcalib)
    test_scores = -threshold - rf.predict(Xtest)

    pvals = conf_pval(calib_scores, test_scores)
    sel = BH(pvals, fdp_nominal)
    fdp, pcer, power = eval(Ytest, sel, -np.inf, threshold)
    fdps_res.append(fdp)
    pcers_res.append(pcer)
    powers_res.append(power)

all_res['fdps_res'] = fdps_res
all_res['pcers_res'] = pcers_res
all_res['powers_res'] = powers_res

''' with sigma '''

# clipped score

rf = RandomForestRegressor(n_estimators=100, max_depth=20, max_features='sqrt')
Xtrain, Xcalib, Ytrain, Ycalib = train_test_split(Xtc, Ytc, train_size=70/85, shuffle=True)
Xtrain, Xtrain_rmse, Ytrain, Ytrain_rmse = train_test_split(Xtrain, Ytrain, train_size=50/70, shuffle=True)
rf.fit(Xtrain, Ytrain < threshold)

rf_rmse = RandomForestRegressor(n_estimators=100, max_depth=20, max_features='sqrt')
Ytrain_rmse_pred = rf.predict(Xtrain_rmse)
rf_rmse.fit(Xtrain_rmse, np.abs((Ytrain_rmse < threshold) - Ytrain_rmse_pred))

fdps_clip_s, pcers_clip_s, powers_clip_s = [], [], []

for i, fdp_nominal in enumerate(fdp_nominals):
    calib_scores = 1000 * (Ycalib < threshold) - rf.predict(Xcalib) / rf_rmse.predict(Xcalib)
    test_scores = -rf.predict(Xtest) / rf_rmse.predict(Xtest)

    pvals = conf_pval(calib_scores, test_scores)
    sel = BH(pvals, fdp_nominal)
    fdp, pcer, power = eval(Ytest, sel, -np.inf, threshold)
    fdps_clip_s.append(fdp)
    pcers_clip_s.append(pcer)
    powers_clip_s.append(power)

all_res['fdps_clip_s'] = fdps_clip_s
all_res['pcers_clip_s'] = pcers_clip_s
all_res['powers_clip_s'] = powers_clip_s

# signed error score
rf = RandomForestRegressor(n_estimators=100, max_depth=20, max_features='sqrt')
Xtrain, Xcalib, Ytrain, Ycalib = train_test_split(Xtc, Ytc, train_size=70/85, shuffle=True)
Xtrain, Xtrain_rmse, Ytrain, Ytrain_rmse = train_test_split(Xtrain, Ytrain, train_size=50/70, shuffle=True)
rf.fit(Xtrain, Ytrain)

rf_rmse = RandomForestRegressor(n_estimators=100, max_depth=20, max_features='sqrt')
Ytrain_rmse_pred = rf.predict(Xtrain_rmse)
rf_rmse.fit(Xtrain_rmse, np.abs((Ytrain_rmse < threshold) - Ytrain_rmse_pred))

fdps_res_s, pcers_res_s, powers_res_s = [], [], []

for i, fdp_nominal in enumerate(fdp_nominals):
    calib_scores = (-Ycalib - rf.predict(Xcalib)) / rf_rmse.predict(Xcalib)
    test_scores = (-threshold - rf.predict(Xtest)) / rf_rmse.predict(Xtest)

    pvals = conf_pval(calib_scores, test_scores)
    sel = BH(pvals, fdp_nominal)
    fdp, pcer, power = eval(Ytest, sel, -np.inf, threshold)
    fdps_res_s.append(fdp)
    pcers_res_s.append(pcer)
    powers_res_s.append(power)

all_res['fdps_res_s'] = fdps_res_s
all_res['pcers_res_s'] = pcers_res_s
all_res['powers_res_s'] = powers_res_s

# save the results
out_dir = os.path.join('result_sc', f'{dataset_name} {args.sample:.2f}')
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

all_res.to_csv(os.path.join('result_sc', f'{dataset_name} {args.sample:.2f}', f'{dataset_name} {args.sample:.2f} {args.seed}.csv'))
