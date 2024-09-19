import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str)
parser.add_argument('sample', type=float)
parser.add_argument('seednum', type=int)
args = parser.parse_args()

df_list = []

for i in range(1, args.seednum + 1):
    path = os.path.join('result', f'{args.dataset} {args.sample:.2f}', f'{args.dataset} {args.sample:.2f} {i}.csv')
    one_df = pd.read_csv(path)
    df_list.append(one_df)

df = pd.concat(df_list).groupby(level=0).mean()

fdp_nominals = df['fdp_nominals']

powers_15_rb = df['powers_15_rb']
fdps_15_rb = df['fdps_15_rb']
pcers_15_rb = df['pcers_15_rb']

powers_15_rp = df['powers_15_rp']
fdps_15_rp = df['fdps_15_rp']
pcers_15_rp = df['pcers_15_rp']

powers_cs = df['powers_cs']
fdps_cs = df['fdps_cs']
pcers_cs = df['pcers_cs']

# plot treevar

out_dir = os.path.join('indiv_pic', f'{args.dataset} {args.sample:.2f}')

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# plot FDP comparison

idx_list = [np.searchsorted(fdp_nominals, i, side='right') for i in np.linspace(0.1, 0.5, 10)]

fig, axs = plt.subplots(figsize=(8, 6))

fig.suptitle(f"FDP control for Sheridan's method and Conformal Selection method, \n {args.dataset} dataset, averaged over {args.seednum} times")
axs.plot(fdp_nominals[idx_list], fdps_15_rb[idx_list], label='Sheridan (2015), bin', marker='o')
axs.plot(fdp_nominals[idx_list], fdps_15_rp[idx_list], label='Sheridan (2015), pred', marker='o')
axs.plot(fdp_nominals[idx_list], fdps_cs[idx_list], label='Conformal Selection', marker='o')
axs.plot([0.05, 0.55], [0.05, 0.55], color='grey', alpha=0.7, linestyle='-.')
axs.set_xlabel("Nominal level")
axs.set_ylabel("Risk ")
plt.legend()
plt.savefig(os.path.join('indiv_pic', f'{args.dataset} {args.sample:.2f}', f'FDP control.png'))

# plot power comparison

idx_list_15_rb = [np.searchsorted(fdps_15_rb, i, side='right') for i in np.linspace(0.1, 0.5, 10)]
idx_list_15_rp = [np.searchsorted(fdps_15_rp, i, side='right') for i in np.linspace(0.1, 0.5, 10)]
idx_list_cs = [np.searchsorted(fdps_cs, i, side='right') for i in np.linspace(0.1, 0.5, 10)]

fig, axs = plt.subplots(figsize=(8, 6))

fig.suptitle(f"Power vs FDP for Sheridan's method and Conformal Selection method, \n {args.dataset} dataset, averaged over {args.seednum} times")
axs.plot(fdps_15_rb[idx_list_15_rb], powers_15_rb[idx_list_15_rb], label='Sheridan (2015), bin', marker='o')
axs.plot(fdps_15_rp[idx_list_15_rp], powers_15_rp[idx_list_15_rp], label='Sheridan (2015), pred', marker='o')
axs.plot(fdps_cs[idx_list_cs], powers_cs[idx_list_cs], label='Conformal Selection', marker='o')
axs.set_xlabel("FDP")
axs.set_ylabel("Power")
plt.legend()
plt.savefig(os.path.join('indiv_pic', f'{args.dataset} {args.sample:.2f}', f'Power vs FDP.png'))
