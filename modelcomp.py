from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import sys
import os
import random
from tqdm import tqdm
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

def get_model(depth):
    return RandomForestRegressor(n_estimators=100, max_depth=depth, max_features='sqrt')

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

''' conformal selection method '''
# 35% calibration, 50% train

# fit random forest regressor
for depth in tqdm(range(1, 21)):
    mdl = get_model(depth)
    Xtrain, Xcalib, Ytrain, Ycalib = train_test_split(Xtc, Ytc, train_size=50/85, shuffle=True)
    mdl.fit(Xtrain, Ytrain < threshold)

    df_tmp = pd.DataFrame()
    fdps_cs, pcers_cs, powers_cs = [], [], []

    for i, fdp_nominal in enumerate(fdp_nominals):
        calib_scores = 1000 * (Ycalib < threshold) - mdl.predict(Xcalib)
        test_scores = -mdl.predict(Xtest)

        pvals = conf_pval(calib_scores, test_scores)
        sel = BH(pvals, fdp_nominal)
        fdp, pcer, power = eval(Ytest, sel, -np.inf, threshold)
        fdps_cs.append(fdp)
        pcers_cs.append(pcer)
        powers_cs.append(power)

        df_tmp[f'fdps_{fdp_nominal:.2f}'] = [fdp]
        df_tmp[f'pcers_{fdp_nominal:.2f}'] = [pcer]
        df_tmp[f'powers_{fdp_nominal:.2f}'] = [power]

    r2 = r2_score(Ytest < threshold, mdl.predict(Xtest))
    df_tmp['r2'] = [r2]
    all_res = pd.concat((all_res, df_tmp))

# save the results
out_dir = os.path.join(f'result_mdl', f'{dataset_name} {args.sample:.2f}')
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

all_res.to_csv(os.path.join(f'result_mdl', f'{dataset_name} {args.sample:.2f}', f'{dataset_name} {args.sample:.2f} {args.seed}.csv'))
