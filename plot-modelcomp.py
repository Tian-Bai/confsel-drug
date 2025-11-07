import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
import seaborn as sns
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('sample', type=float)
parser.add_argument('n_itr', type=int)
args = parser.parse_args()

sample = args.sample
n_itr = args.n_itr

# Set ggplot style for the plots
plt.style.use('ggplot')

df_list = []
dataset_list = ['3A4', 'CB1', 'DPP4', 'HIVINT', 'HIVPROT', 'LOGD', 'METAB', 'NK1', 'OX1', 'OX2', 'PGP', 'PPB', 'RAT_F', 'TDI', 'THROMBIN']

for name in dataset_list:
    df_ones = []
    for j in range(1, 1+n_itr):
        try:
            df = pd.read_csv(os.path.join("result_mdl", f"{name} {sample:.2f}", f"{name} {sample:.2f} {j}.csv"))
            df['depth'] = np.arange(1, 21)
        except FileNotFoundError as e:
            print(e)
        df_ones.append(df)
    df = pd.concat(df_ones).groupby("depth", as_index=False).mean()

    # if only to q=0.5
    # df = df[df['fdp_nominals'] <= 0.5]
    df_list.append(df)

# Create a grid for subplots
fig, axs = plt.subplots(nrows=3, ncols=5, figsize=(18, 12))
axs = axs.flatten()

# Loop through datasets and plot the data on each subplot
for i, name in enumerate(dataset_list):
    ax = axs[i]

    if i == 0:
        # Plot data for each model and conformal method
        # print(df_list[i]['fdp_nominals'])
        # line1, = ax.plot(df_list[i]["depth"], df_list[i]["powers_0.20"],
        #             marker="o", alpha=0.8, label="Power", color='steelblue')
        line2, = ax.plot(df_list[i]["depth"], df_list[i]["powers_0.30"],
                    marker="o", alpha=0.8, label="Power", color='steelblue')
        line3, = ax.plot(df_list[i]["depth"], df_list[i]["r2"],
                    marker="o", alpha=0.8, label="Out-of-sample $R^2$", color='orange')
        
        ax.legend(loc='best', bbox_to_anchor=(4, -2.9), frameon=True, shadow=False, ncol=4, fontsize=21)
    else:
        # line1, = ax.plot(df_list[i]["depth"], df_list[i]["powers_0.20"],
        #             marker="o", alpha=0.8, color='steelblue')
        line2, = ax.plot(df_list[i]["depth"], df_list[i]["powers_0.30"],
                    marker="o", alpha=0.8, label="Power", color='steelblue')
        line3, = ax.plot(df_list[i]["depth"], df_list[i]["r2"],
                    marker="o", alpha=0.8, color='orange')

    # Set axis labels
    ax.set_title(f'{name}', fontsize=18)
    ax.tick_params(axis='both', labelsize=11)

    # Add grid lines
    ax.grid(True)

for ax in axs:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
        spine.set_edgecolor('gray')

# Adjust spacing between subplots
fig.subplots_adjust(wspace=0.2, hspace=0.3, top=0.9, bottom=0.13, left=0.07, right=0.96)

# Add global x and y labels, move them slightly outward
fig.text(0.5, 0.07, 'Depth', ha='center', fontsize=22)  # Moved down slightly
fig.text(0.025, 0.5, 'Power / $R^2$', va='center', rotation='vertical', fontsize=22)  # Moved left slightly

# Title for the entire plot
# fig.suptitle("FDP Control for all 15 Datasets", fontsize=16)

# Display the plot
plt.savefig(os.path.join("pic", "mdl.pdf"))
# plt.show()