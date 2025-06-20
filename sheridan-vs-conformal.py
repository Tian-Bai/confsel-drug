from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
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
parser.add_argument('model', type=str, default='rf', choices=['rf', 'lin', 'nn']) # prediction model used
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)

def get_model(mdl_str):
    if mdl_str == 'rf':
        return RandomForestRegressor(n_estimators=100, max_depth=20, max_features='sqrt')
    if mdl_str == 'lin':
        return LinearRegression()
    if mdl_str == 'nn':
        return MLPRegressor(hidden_layer_sizes=[64, 32])

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

''' RMSE-bin (Sheridan 2004) '''
# 15% calib, and for each RMSE accumulation, 50% train, 20% test
# since data is limited in size in our case, we will only use 1d binning

with Timer() as timer:
    Xtrain, Xcalib, Ytrain, Ycalib = train_test_split(Xtc, Ytc, train_size=70/85, shuffle=True)

    # accumulate RMSE dataset
    rmse_df_list = []
    for k in range(5):
        Xtrain_1, Xtrain_2, Ytrain_1, Ytrain_2 = train_test_split(Xtrain, Ytrain, train_size=50/70, shuffle=True)
        mdl = get_model(args.model)
        mdl.fit(Xtrain_1, Ytrain_1)

        sim_nearest = np.array([dice_sim(x, Xtrain_1) for x in Xtrain_2])
        RMSE = np.absolute(Ytrain_2 - mdl.predict(Xtrain_2))
        rmse_df_one = pd.DataFrame({'dice': sim_nearest, 'RMSE': RMSE})
        rmse_df_list.append(rmse_df_one)

    RMSE_df = pd.concat(rmse_df_list, ignore_index=True)

    # Use 50% of data to train random forest
    mdl = get_model(args.model)
    Xtrain, _, Ytrain, _ = train_test_split(Xtrain, Ytrain, train_size=50/70, shuffle=True)
    mdl.fit(Xtrain, Ytrain)

    R_list = np.zeros(len(fdp_nominals)) # the threshold to use for each FDR level
    fdps_15_rb, pcers_15_rb, powers_15_rb = [], [], []

    # get z scores for calibration
    Ypred_calib = mdl.predict(Xcalib)
    sim_calib = np.array([dice_sim(x, Xtrain) for x in Xcalib])
    rmse_calib = []
    for s in sim_calib:
        filtered_RMSE_df = RMSE_df[(s - 0.05 <= RMSE_df["dice"]) & (RMSE_df["dice"] <= s + 0.05)]
        # rmse = filtered_RMSE_df["RMSE"].mean()
        rmse = np.mean(filtered_RMSE_df["RMSE"].to_numpy() ** 2)
        rmse = np.sqrt(rmse)
        rmse_calib.append(rmse)
    rmse_calib = np.array(rmse_calib)

    z_calib = (threshold - Ypred_calib) / rmse_calib

    # search for proper R levels
    for R in np.linspace(0.5, -2, 200):
        try_r_sel = [j for j in range(len(z_calib)) if z_calib[j] >= R]
        try_fdp, _, _ = eval(Ycalib, try_r_sel, -np.inf, threshold)
        R_list[fdp_nominals >= try_fdp] = R

    # get z scores for test
    Ypred_test = mdl.predict(Xtest)
    sim_test = np.array([dice_sim(x, Xtrain) for x in Xtest])
    rmse_test = []
    for s in sim_test:
        filtered_RMSE_df = RMSE_df[(s - 0.05 <= RMSE_df["dice"]) & (RMSE_df["dice"] <= s + 0.05)]
        # rmse = filtered_RMSE_df["RMSE"].mean()
        rmse = np.mean(filtered_RMSE_df["RMSE"].to_numpy() ** 2)
        rmse = np.sqrt(rmse)
        rmse_test.append(rmse)
    rmse_test = np.array(rmse_test)

    z_test = (threshold - Ypred_test) / rmse_test

    # conduct selection, and evaluate performance
    for i, R in enumerate(R_list):
        sheridan_15 = [j for j in range(len(z_test)) if z_test[j] >= R]
        fdp, pcer, power = eval(Ytest, sheridan_15, -np.inf, threshold)
        fdps_15_rb.append(fdp)
        pcers_15_rb.append(pcer)
        powers_15_rb.append(power)

all_res['fdps_15_rb'] = fdps_15_rb
all_res['pcers_15_rb'] = pcers_15_rb
all_res['powers_15_rb'] = powers_15_rb
all_res['time_15_rb'] = [timer.runtime] * len(fdp_nominals)

''' RMSE-pred (Sheridan 2013) '''
# Similar to RMSE-bin, but use error models instead of RMSE accumulation

with Timer() as timer:
    # fit random forest regressor and error model
    mdl = get_model(args.model)
    mdl_rmse = get_model(args.model)
    Xtrain, Xcalib, Ytrain, Ycalib = train_test_split(Xtc, Ytc, train_size=70/85, shuffle=True)
    Xtrain, Xtrain_rmse, Ytrain, Ytrain_rmse = train_test_split(Xtrain, Ytrain, train_size=50/70, shuffle=True)
    mdl.fit(Xtrain, Ytrain)

    if args.model == 'rf':
        # for ensemble models such as rf, we use the predicted value and prediction variance as features for the error model 
        # recommended by Sheridan et al. (2013)
        Ytrain_rmse_pred = mdl.predict(Xtrain_rmse)
        all_Ytrain_rmse_pred = np.column_stack([tree.predict(Xtrain_rmse) for tree in mdl.estimators_])
        var_train_rmse = np.var(all_Ytrain_rmse_pred, axis=1)

        mdl_rmse.fit(np.column_stack((Ytrain_rmse_pred, var_train_rmse)), np.abs(Ytrain_rmse - Ytrain_rmse_pred))
    else:
        # otherwise, above option is impossible and we simply use the original feature X as feature for the error model
        Ytrain_rmse_pred = mdl.predict(Xtrain_rmse)
        mdl_rmse.fit(Xtrain_rmse, np.abs(Ytrain_rmse - Ytrain_rmse_pred))

    R_list = np.zeros(len(fdp_nominals))
    fdps_15_rp, pcers_15_rp, powers_15_rp = [], [], []

    # get z scores for calibration
    if args.model == 'rf':
        Ypred_calib = mdl.predict(Xcalib)
        all_Ypred = np.column_stack([tree.predict(Xcalib) for tree in mdl.estimators_])
        var_calib = np.var(all_Ypred, axis=1)
        rmse_calib = mdl_rmse.predict(np.column_stack((Ypred_calib, var_calib)))
    else:
        rmse_calib = mdl_rmse.predict(Xcalib)

    z_calib = (threshold - Ypred_calib) / rmse_calib

    # search for proper R levels
    for R in np.linspace(0.5, -2, 200):
        try_r_sel = [j for j in range(len(z_calib)) if z_calib[j] >= R]
        try_fdp, _, _ = eval(Ycalib, try_r_sel, -np.inf, threshold)
        R_list[fdp_nominals >= try_fdp] = R

    # get z scores for test
    if args.model == 'rf':
        Ypred_test = mdl.predict(Xtest)
        all_Ypred = np.column_stack([tree.predict(Xtest) for tree in mdl.estimators_])
        var_test = np.var(all_Ypred, axis=1)
        rmse_test = mdl_rmse.predict(np.column_stack((Ypred_test, var_test)))
    else:
        rmse_test = mdl_rmse.predict(Xtest)

    z_test = (threshold - Ypred_test) / rmse_test

    for i, R in enumerate(R_list):
        sheridan_15 = [j for j in range(len(z_test)) if z_test[j] >= R]
        fdp, pcer, power = eval(Ytest, sheridan_15, -np.inf, threshold)
        fdps_15_rp.append(fdp)
        pcers_15_rp.append(pcer)
        powers_15_rp.append(power)

all_res['fdps_15_rp'] = fdps_15_rp
all_res['pcers_15_rp'] = pcers_15_rp
all_res['powers_15_rp'] = powers_15_rp
all_res['time_15_rp'] = [timer.runtime] * len(fdp_nominals) # time them together

''' conformal selection method '''
# 35% calibration, 50% train

with Timer() as timer:
    # fit random forest regressor
    mdl = get_model(args.model)
    Xtrain, Xcalib, Ytrain, Ycalib = train_test_split(Xtc, Ytc, train_size=50/85, shuffle=True)
    mdl.fit(Xtrain, Ytrain < threshold)

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

all_res['fdps_cs'] = fdps_cs
all_res['pcers_cs'] = pcers_cs
all_res['powers_cs'] = powers_cs
all_res['time_cs'] = [timer.runtime] * len(fdp_nominals)

# save the results
out_dir = os.path.join(f'result', args.model, f'{dataset_name} {args.sample:.2f}')
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

all_res.to_csv(os.path.join(f'result', args.model, f'{dataset_name} {args.sample:.2f}', f'{dataset_name} {args.sample:.2f} {args.seed}.csv'))
