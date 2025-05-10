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

fop_nominals = np.round(np.linspace(0.02, 0.2, 10), 2)

all_res = pd.DataFrame() # results

''' conformal selection method '''
# 35% calibration, 50% train

# fit random forest regressor
rf = RandomForestRegressor(n_estimators=100, max_depth=20, max_features='sqrt')
Xtrain, Xcalib, Ytrain, Ycalib = train_test_split(Xtc, Ytc, train_size=50/85, shuffle=True)
rf.fit(Xtrain, Ytrain < threshold)

fops, powers = [], []

for fop_nominal in fop_nominals:
    # backward sel
    calib_scores = 1000 * (-Ycalib < -threshold) - (-rf.predict(Xcalib))
    test_scores = -(-rf.predict(Xtest))

    pvals = conf_pval(calib_scores, test_scores)
    sel = BH(pvals, fop_nominal)

    sel_backward_c = set(range(0, len(Ytest))) - set(sel)

    fop, _, power = eval(Ytest, sel, threshold, np.inf)
    fops.append(fop)
    powers.append(power)

all_res['fop_nominal'] = fop_nominals
all_res['fops'] = fops
all_res['powers'] = powers

# save the results
out_dir = os.path.join('result', f'fop {dataset_name} {args.sample:.2f}')
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

all_res.to_csv(os.path.join('result', f'fop {dataset_name} {args.sample:.2f}', f'fop {dataset_name} {args.sample:.2f} {args.seed}.csv'))
