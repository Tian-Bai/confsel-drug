from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import sys
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import argparse
from utility import BH

def dice_sim(f, set_of_f):
    f = np.array(f)
    set_of_f = np.array(set_of_f) 

    min_count = np.minimum(f, set_of_f).sum(axis=1) * 2
    sum_count = f.sum() + set_of_f.sum(axis=1)
    sum_count[sum_count == 0] = 1
    return np.max(min_count / sum_count)

def eval(Y, rejected, lower, higher):
    true_reject = np.sum((lower < Y) & (Y < higher))
    if len(rejected) == 0:
        fdp = 0
        pcer = 0
        power = 0
    else:
        fdp = np.sum((lower >= Y[rejected]) | (Y[rejected] >= higher)) / len(rejected)
        pcer = np.sum((lower >= Y[rejected]) | (Y[rejected] >= higher)) / len(Y)
        power = np.sum((lower < Y[rejected]) & (Y[rejected] < higher)) / true_reject if true_reject != 0 else 0
    return fdp, pcer, power

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str)
parser.add_argument('sample', type=float)
parser.add_argument('seed', type=int)
args = parser.parse_args()

random.seed(args.seed)

dataset_name = args.dataset
dataset_path = f'data/{dataset_name}_training_disguised.csv'

dataset = pd.read_csv(dataset_path)

if args.sample < 1:
    dataset = dataset.sample(frac=args.sample)

thresholds_map = {'NK1': 6.5, 'PGP': -0.3, 'LOGD': 1.5, '3A4': 4.35, 'CB1': 6.5, 'DPP4': 6, 'HIVINT': 6, 'HIVPROT': 4.5, 'METAB': 40, 'OX1': 5, 'OX2': 6, 'PPB': 1, 'RAT_F': 0.3, 'TDI': 0, 'THROMBIN': 6}
threshold = thresholds_map[dataset_name]

total_Y = dataset['Act'].to_numpy()
total_X = dataset.drop(columns=['MOLECULE', 'Act']).to_numpy()

Xtc, Xtest, Ytc, Ytest = train_test_split(total_X, total_Y, test_size=0.3, shuffle=True) # tc: train and calib

fdp_nominals = np.linspace(0.01, 0.99, 200)
all_res = pd.DataFrame()

all_res['fdp_nominals'] = fdp_nominals

''' RMSE-bin (Sheridan 2012) '''

# since data is limited in size in our case, we will only use 1d binning

Xtrain, Xcalib, Ytrain, Ycalib = train_test_split(Xtc, Ytc, train_size=0.7, shuffle=True)

# now only use Xtrain to do RMSE binning

rmse_df_list = []
for k in range(5):
    Xtrain_1, Xtrain_2, Ytrain_1, Ytrain_2 = train_test_split(Xtrain, Ytrain, train_size=0.7, shuffle=True)
    rf = RandomForestRegressor(n_estimators=100, max_depth=20, max_features='sqrt')
    rf.fit(Xtrain_1, Ytrain_1)

    sim_nearest = np.array([dice_sim(x, Xtrain_1) for x in Xtrain_2])
    RMSE = np.absolute(Ytrain_2 - rf.predict(Xtrain_2))
    rmse_df_one = pd.DataFrame({'dice': sim_nearest, 'RMSE': RMSE})
    rmse_df_list.append(rmse_df_one)

RMSE_df = pd.concat(rmse_df_list, ignore_index=True)

# now use 70% of Xtrain to train rf
rf = RandomForestRegressor(n_estimators=100, max_depth=20, max_features='sqrt')
Xtrain, _, Ytrain, _ = train_test_split(Xtrain, Ytrain, train_size=0.7, shuffle=True)
rf.fit(Xtrain, Ytrain)

R_list = np.zeros(200) # which threshold to use?
fdps_15_rb, pcers_15_rb, powers_15_rb = [], [], []

Ypred_calib = rf.predict(Xcalib)
sim_calib = np.array([dice_sim(x, Xtrain) for x in Xcalib])
rmse_calib = []
for s in sim_calib:
    filtered_RMSE_df = RMSE_df[(s - 0.05 <= RMSE_df["dice"]) & (RMSE_df["dice"] <= s + 0.05)]
    rmse = filtered_RMSE_df["RMSE"].mean()
    rmse_calib.append(rmse)
rmse_calib = np.array(rmse_calib)

z_calib = (threshold - Ypred_calib) / rmse_calib

for R in np.linspace(0.5, -2, 200):
    try_r_sel = [j for j in range(len(z_calib)) if z_calib[j] >= R]
    try_fdp, _, _ = eval(Ycalib, try_r_sel, -100, threshold)
    R_list[fdp_nominals >= try_fdp] = R

Ypred_test = rf.predict(Xtest)
sim_test = np.array([dice_sim(x, Xtrain) for x in Xtest])
rmse_test = []
for s in sim_test:
    filtered_RMSE_df = RMSE_df[(s - 0.05 <= RMSE_df["dice"]) & (RMSE_df["dice"] <= s + 0.05)]
    rmse = filtered_RMSE_df["RMSE"].mean()
    rmse_test.append(rmse)
rmse_test = np.array(rmse_test)

z_test = (threshold - Ypred_test) / rmse_test

for i, R in enumerate(R_list):
    sheridan_15 = [j for j in range(len(z_test)) if z_test[j] >= R]
    fdp, pcer, power = eval(Ytest, sheridan_15, -100, threshold)
    fdps_15_rb.append(fdp)
    pcers_15_rb.append(pcer)
    powers_15_rb.append(power)

all_res['fdps_15_rb'] = fdps_15_rb
all_res['pcers_15_rb'] = pcers_15_rb
all_res['powers_15_rb'] = powers_15_rb

''' RMSE-pred (Sheridan 2013) '''

rf = RandomForestRegressor(n_estimators=100, max_depth=20, max_features='sqrt')
rf_rmse = RandomForestRegressor(n_estimators=100, max_depth=20, max_features='sqrt')
Xtrain, Xcalib, Ytrain, Ycalib = train_test_split(Xtc, Ytc, train_size=0.7, shuffle=True)
Xtrain, Xtrain_rmse, Ytrain, Ytrain_rmse = train_test_split(Xtrain, Ytrain, train_size=0.5, shuffle=True)
rf.fit(Xtrain, Ytrain)

Ytrain_rmse_pred = rf.predict(Xtrain_rmse)
all_Ytrain_rmse_pred = np.column_stack([tree.predict(Xtrain_rmse) for tree in rf.estimators_])
var_train_rmse = np.var(all_Ytrain_rmse_pred, axis=1)

rf_rmse.fit(np.column_stack((Ytrain_rmse_pred, var_train_rmse)), np.abs(Ytrain_rmse - Ytrain_rmse_pred))

R_list = np.zeros(200) # which threshold to use?
fdps_15_rp, pcers_15_rp, powers_15_rp = [], [], []

Ypred_calib = rf.predict(Xcalib)
all_Ypred = np.column_stack([tree.predict(Xcalib) for tree in rf.estimators_])
var_calib = np.var(all_Ypred, axis=1)

rmse_calib = rf_rmse.predict(np.column_stack((Ypred_calib, var_calib)))

z_calib = (threshold - Ypred_calib) / rmse_calib

for R in np.linspace(0.5, -2, 200):
    try_r_sel = [j for j in range(len(z_calib)) if z_calib[j] >= R]
    try_fdp, _, _ = eval(Ycalib, try_r_sel, -100, threshold)
    R_list[fdp_nominals >= try_fdp] = R

Ypred_test = rf.predict(Xtest)
all_Ypred = np.column_stack([tree.predict(Xtest) for tree in rf.estimators_])
var_test = np.var(all_Ypred, axis=1)

rmse_test = rf_rmse.predict(np.column_stack((Ypred_test, var_test)))
z_test = (threshold - Ypred_test) / rmse_test

for i, R in enumerate(R_list):
    sheridan_15 = [j for j in range(len(z_test)) if z_test[j] >= R]
    fdp, pcer, power = eval(Ytest, sheridan_15, -100, threshold)
    fdps_15_rp.append(fdp)
    pcers_15_rp.append(pcer)
    powers_15_rp.append(power)

all_res['fdps_15_rp'] = fdps_15_rp
all_res['pcers_15_rp'] = pcers_15_rp
all_res['powers_15_rp'] = powers_15_rp

''' conformal selection method '''

rf = RandomForestRegressor(n_estimators=100, max_depth=20, max_features='sqrt')
Xtrain, Xcalib, Ytrain, Ycalib = train_test_split(Xtc, Ytc, train_size=0.7, shuffle=True)
rf.fit(Xtrain, Ytrain < threshold)

fdps_cs, pcers_cs, powers_cs = [], [], []

for i, fdp_nominal in enumerate(fdp_nominals):
    calib_scores_2clip = 1000 * (Ycalib < threshold) - rf.predict(Xcalib)   # Ycalib_cs > 0.5 <=> original Ycalib_cs < threshold
    test_scores = -rf.predict(Xtest)
    BH_2clip = BH(calib_scores_2clip, test_scores, fdp_nominal)
    fdp, pcer, power = eval(Ytest, BH_2clip, -100, threshold)
    fdps_cs.append(fdp)
    pcers_cs.append(pcer)
    powers_cs.append(power)

all_res['fdps_cs'] = fdps_cs
all_res['pcers_cs'] = pcers_cs
all_res['powers_cs'] = powers_cs

out_dir = f'result/{dataset_name} {args.sample:.2f}'

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

all_res.to_csv(f'result/{dataset_name} {args.sample:.2f}/{dataset_name} {args.sample:.2f} {args.seed}.csv')
