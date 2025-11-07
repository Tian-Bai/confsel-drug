from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import sys
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
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

fdp_nominals = np.round(np.linspace(0.1, 0.5, 9), 2) # nominal FDR levels
# fdp_nominals = np.round(np.linspace(0.05, 0.95, 19), 2) # nominal FDR levels
fop_nominals = np.array([0.05, 0.1])

all_res = pd.DataFrame() # results

''' conformal selection method '''
# 35% calibration, 50% train

# fit random forest regressor
rf = RandomForestRegressor(n_estimators=100, max_depth=20, max_features='sqrt')
Xtrain, Xcalib, Ytrain, Ycalib = train_test_split(Xtc, Ytc, train_size=50/85, shuffle=True)
rf.fit(Xtrain, Ytrain < threshold)

for fdp_nominal in fdp_nominals:
    for fop_nominal in fop_nominals:
        # forward sel
        calib_scores = 1000 * (Ycalib < threshold) - rf.predict(Xcalib)
        test_scores = -rf.predict(Xtest)

        pvals = conf_pval(calib_scores, test_scores)
        sel_forward = BH(pvals, fdp_nominal)

        # backward sel
        calib_scores = 1000 * (-Ycalib < -threshold) - (-rf.predict(Xcalib))
        test_scores = -(-rf.predict(Xtest))

        pvals = conf_pval(calib_scores, test_scores)
        sel_backward = BH(pvals, fop_nominal)

        # get the 3 zones
        sel_backward_c = set(range(0, len(Ytest))) - set(sel_backward)

        green = set(sel_forward) & set(sel_backward_c)
        grey = (set(sel_forward) - set(sel_backward_c)) | (set(sel_backward_c) - set(sel_forward)) 
        red = set(sel_backward) - set(sel_forward)

        green_fdp, _, green_power = eval(Ytest, list(green), -np.inf, threshold)
        gg_fdp, _, gg_power = eval(Ytest, list(green | grey), -np.inf, threshold)
        red_fdp, _, red_power = eval(Ytest, list(red), threshold, np.inf) # red_fdp is exactly FOP, red_power is the deselection power
        # deselection power: fraction of candidates that I didn't select, among those I should not select. The higher the better

        all_res = pd.concat((all_res, pd.DataFrame({
            'fdp_nominal': [fdp_nominal],
            'fop_nominal': [fop_nominal],
            'green_fdp': [green_fdp],
            'green_power': [green_power],
            'gg_fdp': [gg_fdp],
            'gg_power': [gg_power],
            'red_fdp': [red_fdp],
            'red_power': [red_power],
        })))

# save the results
out_dir = os.path.join('result', f'3zone {dataset_name} {args.sample:.2f}')
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

all_res.to_csv(os.path.join('result', f'3zone {dataset_name} {args.sample:.2f}', f'3zone {dataset_name} {args.sample:.2f} {args.seed}.csv'))
