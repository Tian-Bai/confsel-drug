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
            df = pd.read_csv(os.path.join("result_fop", f"{sample:.2f}", f"fop {name} {sample:.2f}", f"fop {name} {sample:.2f} {j}.csv"))
        except FileNotFoundError as e:
            print(e)
        df_ones.append(df)
    df = pd.concat(df_ones).groupby("fop_nominal", as_index=False).mean()
    df_list.append(df)

# Create a grid for subplots
fig, axs = plt.subplots(nrows=3, ncols=5, figsize=(18, 12))
axs = axs.flatten()

# Loop through datasets and plot the data on each subplot
for i, name in enumerate(dataset_list):
    ax = axs[i]

    if i == 0:
        # Plot data for each model and conformal method
        # print(df_list[i]['fop_nominal'])
        line1, = ax.plot(df_list[i]['fop_nominal'], df_list[i]['fops'], 
                        marker='o', color='steelblue', alpha=0.8)
        
        # ax.legend(loc='best', bbox_to_anchor=(4.2, -3), frameon=True, shadow=False, ncol=3, fontsize=12)
    else:
        ax.plot(df_list[i]['fop_nominal'], df_list[i]['fops'], 
                marker='o', color='steelblue', alpha=0.8)

    # Reference line for y=x
    ax.plot([0.02, 0.2], [0.02, 0.2], color='grey', alpha=0.7, linestyle='-.')

    # Set axis labels
    ax.set_title(f'{name}', fontsize=12)

    # Add grid lines
    ax.grid(True)

# Adjust spacing between subplots
fig.subplots_adjust(wspace=0.2, hspace=0.3, top=0.9, bottom=0.13, left=0.07, right=0.96)

# Add global x and y labels, move them slightly outward
fig.text(0.5, 0.07, 'Nominal FOR', ha='center', fontsize=14)  # Moved down slightly
fig.text(0.03, 0.5, 'Observed FOR', va='center', rotation='vertical', fontsize=14)  # Moved left slightly

# Title for the entire plot
# fig.suptitle("FDP Control for all 15 Datasets", fontsize=16)

# Display the plot
plt.savefig(os.path.join("pic", "fop.png"))
# plt.show()

''''''''''''

# Create a grid for subplots
fig, axs = plt.subplots(nrows=3, ncols=5, figsize=(18, 12))
axs = axs.flatten()

# Loop through datasets and plot the data on each subplot
for i, name in enumerate(dataset_list):
    ax = axs[i]

    if i == 0:
        # Plot data for each model and conformal method
        line1, = ax.plot(df_list[i]['fops'], df_list[i]['powers'], 
                        marker='o', color='steelblue', alpha=0.8)
        
        # ax.legend(loc='best', bbox_to_anchor=(4.2, -3), frameon=True, shadow=False, ncol=3, fontsize=12)
    else:
        ax.plot(df_list[i]['fops'], df_list[i]['powers'], 
                marker='o', color='steelblue', alpha=0.8)
    
    # Set axis labels
    ax.set_title(f'{name}', fontsize=12)

    # Add grid lines
    ax.grid(True)

# Adjust spacing between subplots
fig.subplots_adjust(wspace=0.2, hspace=0.3, top=0.9, bottom=0.13, left=0.07, right=0.96)

# Add global x and y labels, move them slightly outward
fig.text(0.5, 0.07, 'Observed FOR', ha='center', fontsize=14)  # Moved down slightly
fig.text(0.03, 0.5, 'Observed Removal Power', va='center', rotation='vertical', fontsize=14)  # Moved left slightly

# Title for the entire plot
# fig.suptitle("Power for all 15 Datasets", fontsize=16)

# Display the plot
plt.savefig(os.path.join("pic", "fop-power.png"))
# plt.show()