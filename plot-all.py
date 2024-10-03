import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
import seaborn as sns
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('sample', type=float)
parser.add_argument('seed', type=int)
args = parser.parse_args()

sample = args.sample
n_itr = args.seed

# Set ggplot style for the plots
plt.style.use('ggplot')

df_list = []
dataset_list = ['3A4', 'CB1', 'DPP4', 'HIVINT', 'HIVPROT', 'LOGD', 'METAB', 'NK1', 'OX1', 'OX2', 'PGP', 'PPB', 'RAT_F', 'TDI', 'THROMBIN']

for name in dataset_list:
    df_ones = []
    for j in range(1, 1+n_itr):
        df = pd.read_csv(os.path.join("result", f"{sample:.2f}", f"{name} {sample:.2f}", f"{name} {sample:.2f} {j}.csv"))
        df_ones.append(df)
    df = pd.concat(df_ones).groupby("fdp_nominals", as_index=False).mean()
    df_list.append(df)

# Create a grid for subplots
fig, axs = plt.subplots(nrows=3, ncols=5, figsize=(18, 12))
axs = axs.flatten()

# Loop through datasets and plot the data on each subplot
for i, name in enumerate(dataset_list):
    ax = axs[i]

    if i == 0:
        # Plot data for each model and conformal method
        line1, = ax.plot(df_list[i]['fdp_nominals'], df_list[i]['fdps_15_rb'], 
                        label='Sheridan-bin', marker='o', color='steelblue', alpha=0.8)
        
        line2, = ax.plot(df_list[i]['fdp_nominals'], df_list[i]['fdps_15_rp'], 
                        label='Sheridan-pred', marker='o', color='orange', alpha=0.8)
        
        line3, = ax.plot(df_list[i]['fdp_nominals'], df_list[i]['fdps_cs'], 
                        label='Conformal Selection', marker='o', color='darkgreen', alpha=0.8)
        
        ax.legend(loc='best', bbox_to_anchor=(4, -3), frameon=True, shadow=False, ncol=3, fontsize=12)
    else:
        ax.plot(df_list[i]['fdp_nominals'], df_list[i]['fdps_15_rb'], 
                marker='o', color='steelblue', alpha=0.8)
        
        ax.plot(df_list[i]['fdp_nominals'], df_list[i]['fdps_15_rp'], 
                marker='o', color='orange', alpha=0.8)
        
        ax.plot(df_list[i]['fdp_nominals'], df_list[i]['fdps_cs'], 
                marker='o', color='darkgreen', alpha=0.8)

    # Reference line for y=x
    ax.plot([0.05, 0.55], [0.05, 0.55], color='grey', alpha=0.7, linestyle='-.')

    # Set axis labels
    ax.set_title(f'{name}', fontsize=12)

    # Add grid lines
    ax.grid(True)

# Adjust spacing between subplots
fig.subplots_adjust(wspace=0.2, hspace=0.3, top=0.9, bottom=0.13, left=0.07, right=0.96)

# Add global x and y labels, move them slightly outward
fig.text(0.5, 0.07, 'Nominal FDP', ha='center', fontsize=14)  # Moved down slightly
fig.text(0.03, 0.5, 'Observed FDP', va='center', rotation='vertical', fontsize=14)  # Moved left slightly

# Title for the entire plot
# fig.suptitle("FDP Control for all 15 Datasets", fontsize=16)

# Display the plot
plt.savefig(os.path.join("pic", "fdp.png"))
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
        line1, = ax.plot(df_list[i]['fdps_15_rb'], df_list[i]['powers_15_rb'], 
                        label='Sheridan-bin', marker='o', color='steelblue', alpha=0.8)
        
        line2, = ax.plot(df_list[i]['fdps_15_rp'], df_list[i]['powers_15_rp'], 
                        label='Sheridan-pred', marker='o', color='orange', alpha=0.8)
        
        line3, = ax.plot(df_list[i]['fdps_cs'], df_list[i]['powers_cs'], 
                        label='Conformal Selection', marker='o', color='darkgreen', alpha=0.8)
        
        ax.legend(loc='best', bbox_to_anchor=(4, -3), frameon=True, shadow=False, ncol=3, fontsize=12)
    else:
        ax.plot(df_list[i]['fdps_15_rb'], df_list[i]['powers_15_rb'], 
                marker='o', color='steelblue', alpha=0.8)
        
        ax.plot(df_list[i]['fdps_15_rp'], df_list[i]['powers_15_rp'], 
                marker='o', color='orange', alpha=0.8)
        
        ax.plot(df_list[i]['fdps_cs'], df_list[i]['powers_cs'], 
                marker='o', color='darkgreen', alpha=0.8)

    # Set axis labels
    ax.set_title(f'{name}', fontsize=12)

    # Add grid lines
    ax.grid(True)

# Adjust spacing between subplots
fig.subplots_adjust(wspace=0.2, hspace=0.3, top=0.9, bottom=0.13, left=0.07, right=0.96)

# Add global x and y labels, move them slightly outward
fig.text(0.5, 0.07, 'Observed FDP', ha='center', fontsize=14)  # Moved down slightly
fig.text(0.03, 0.5, 'Observed Power', va='center', rotation='vertical', fontsize=14)  # Moved left slightly

# Title for the entire plot
# fig.suptitle("Power for all 15 Datasets", fontsize=16)

# Display the plot
plt.savefig(os.path.join("pic", "power.png"))
# plt.show()

# ''''''''''''
# # Assume dataset_list is already defined
# df_list = []

# for name in dataset_list:
#     df_ones = []
#     for j in range(1, 1+n_itr):
#         df = pd.read_csv(os.path.join("result", f"{sample:.2f}", f"{name} {sample:.2f}", f"{name} {sample:.2f} {j}.csv"))
#         df = df[df['fdp_nominals'] == 0.2]
#         df_ones.append(df)
#     df = pd.concat(df_ones)
#     df_list.append(df)

# fig, axs = plt.subplots(nrows=3, ncols=5, figsize=(18, 12))
# axs = axs.flatten()

# # Define colors for the three boxplots
# colors = ['steelblue', 'orange', 'darkgreen']

# for i, name in enumerate(dataset_list):
#     ax = axs[i]

#     # Prepare data for the 3 boxplots (each array separately)
#     data = pd.DataFrame({
#         'bin': df_list[i]['fdps_15_rb'],
#         'pred': df_list[i]['fdps_15_rp'],
#         'cs': df_list[i]['fdps_cs']
#     })

#     # Create boxplot using seaborn
#     sns.boxplot(data=data, ax=ax, palette=colors, showfliers=True, width=0.5, showmeans=True, 
#                 meanprops={'markerfacecolor':'red', 
#                           'markeredgecolor':'red'})
    
#     ax.axhline(y=0.2, color='grey', linestyle='-.')

#     # Set title
#     ax.set_title(f'{name}', fontsize=12)

#     # Customize grid
#     ax.grid(True, linestyle='--', alpha=0.7)

# # Adjust spacing between subplots
# fig.subplots_adjust(wspace=0.2, hspace=0.3, top=0.9, bottom=0.13, left=0.07, right=0.96)

# # Add global x and y labels, move them slightly outward
# fig.text(0.5, 0.07, 'Observed FDP', ha='center', fontsize=14)
# fig.text(0.03, 0.5, 'Selection Methods', va='center', rotation='vertical', fontsize=14)

# # Save the plot to a file
# plt.savefig(os.path.join("pic", "fdp-box.png"))

# ''''''''''''
# df_list = []

# for name in dataset_list:
#     df_ones = []
#     for j in range(1, 1+n_itr):
#         df = pd.read_csv(os.path.join("result", f"{sample:.2f}", f"{name} {sample:.2f}", f"{name} {sample:.2f} {j}.csv"))
#         filtered_15_rb = df[df['fdps_15_rb'] <= 0.2]
#         filtered_15_rp = df[df['fdps_15_rp'] <= 0.2]
#         filtered_cs = df[df['fdps_cs'] <= 0.2]

#         # Find the maximum powers where the condition holds
#         df = pd.DataFrame({
#             'powers_15_rb': [filtered_15_rb['powers_15_rb'].max()],
#             'powers_15_rp': [filtered_15_rp['powers_15_rp'].max()],
#             'powers_cs': [filtered_cs['powers_cs'].max()]
#         }).fillna(0)
#         df_ones.append(df)
#     df = pd.concat(df_ones)
#     df_list.append(df)

# fig, axs = plt.subplots(nrows=3, ncols=5, figsize=(18, 12))
# axs = axs.flatten()

# for i, name in enumerate(dataset_list):
#     ax = axs[i]

#     # Prepare data for the 3 boxplots (each array separately)
#     data = pd.DataFrame({
#         'bin': df_list[i]['powers_15_rb'],
#         'pred': df_list[i]['powers_15_rp'],
#         'cs': df_list[i]['powers_cs']
#     })

#     # Create boxplot using seaborn
#     sns.boxplot(data=data, ax=ax, palette=colors, showfliers=True, width=0.5, showmeans=True, 
#                 meanprops={'markerfacecolor':'red', 
#                           'markeredgecolor':'red'})

#     # Set title
#     ax.set_title(f'{name}', fontsize=12)

#     # Customize grid
#     ax.grid(True, linestyle='--', alpha=0.7)

# # Adjust spacing between subplots
# fig.subplots_adjust(wspace=0.2, hspace=0.3, top=0.9, bottom=0.13, left=0.07, right=0.96)

# # Add global x and y labels, move them slightly outward
# fig.text(0.5, 0.07, 'Observed Power', ha='center', fontsize=14)
# fig.text(0.03, 0.5, 'Selection Methods', va='center', rotation='vertical', fontsize=14)

# # Save the plot to a file
# plt.savefig(os.path.join("pic", "power-box.png"))
