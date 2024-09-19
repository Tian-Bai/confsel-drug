import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os

# Set ggplot style for the plots
plt.style.use('ggplot')

df_list = []
dataset_list = ['3A4', 'CB1', 'DPP4', 'HIVINT', 'HIVPROT', 'LOGD', 'METAB', 'NK1', 'OX1', 'OX2', 'PGP', 'PPB', 'RAT_F', 'TDI', 'THROMBIN']

for name in dataset_list:
    df = pd.read_csv(os.path.join("result_avg", f"{name} 0.10.csv"))
    df_list.append(df)

# Create a grid for subplots
fig, axs = plt.subplots(nrows=3, ncols=5, figsize=(18, 12))
axs = axs.flatten()

# Define the search index for the x-axis
idx_list = [np.searchsorted(np.linspace(0.01, 0.99, 200), j, side='right') for j in np.linspace(0.1, 0.5, 10)]

# Loop through datasets and plot the data on each subplot
for i, name in enumerate(dataset_list):
    ax = axs[i]

    if i == 0:
        # Plot data for each model and conformal method
        line1, = ax.plot(df_list[i]['fdp_nominals'][idx_list], df_list[i]['fdps_15_rb'][idx_list], 
                        label='Sheridan (2015), bin', marker='o', color='steelblue', alpha=0.8)
        
        line2, = ax.plot(df_list[i]['fdp_nominals'][idx_list], df_list[i]['fdps_15_rp'][idx_list], 
                        label='Sheridan (2015), pred', marker='o', color='orange', alpha=0.8)
        
        line3, = ax.plot(df_list[i]['fdp_nominals'][idx_list], df_list[i]['fdps_cs'][idx_list], 
                        label='Conformal Selection', marker='o', color='darkgreen', alpha=0.8)
        
        ax.legend(loc='best', bbox_to_anchor=(4.2, -3), frameon=True, shadow=False, ncol=3, fontsize=12)
    else:
        ax.plot(df_list[i]['fdp_nominals'][idx_list], df_list[i]['fdps_15_rb'][idx_list], 
                marker='o', color='steelblue', alpha=0.8)
        
        ax.plot(df_list[i]['fdp_nominals'][idx_list], df_list[i]['fdps_15_rp'][idx_list], 
                marker='o', color='orange', alpha=0.8)
        
        ax.plot(df_list[i]['fdp_nominals'][idx_list], df_list[i]['fdps_cs'][idx_list], 
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
fig.suptitle("FDP Control for all 15 Datasets", fontsize=16)

# Display the plot
plt.savefig(os.path.join("pic", "fdp.png"))
# plt.show()

''''''''''''

# Create a grid for subplots
fig, axs = plt.subplots(nrows=3, ncols=5, figsize=(18, 12))
axs = axs.flatten()

# Define the search index for the x-axis
idx_list = [np.searchsorted(np.linspace(0.01, 0.99, 200), j, side='right') for j in np.linspace(0.1, 0.5, 10)]

# Loop through datasets and plot the data on each subplot
for i, name in enumerate(dataset_list):
    ax = axs[i]
    idx_list_15_rb = [min(np.searchsorted(df_list[i]['fdps_15_rb'], j, side='right'), 199) for j in np.linspace(0.1, 0.5, 10)]
    idx_list_15_rp = [min(np.searchsorted(df_list[i]['fdps_15_rp'], j, side='right'), 199) for j in np.linspace(0.1, 0.5, 10)]
    idx_list_cs = [min(np.searchsorted(df_list[i]['fdps_cs'], j, side='right'), 199) for j in np.linspace(0.1, 0.5, 10)]

    if i == 0:
        # Plot data for each model and conformal method
        line1, = ax.plot(df_list[i]['fdps_15_rb'][idx_list_15_rb], df_list[i]['powers_15_rb'][idx_list_15_rb], 
                        label='Sheridan (2015), bin', marker='o', color='steelblue', alpha=0.8)
        
        line2, = ax.plot(df_list[i]['fdps_15_rp'][idx_list_15_rp], df_list[i]['powers_15_rp'][idx_list], 
                        label='Sheridan (2015), pred', marker='o', color='orange', alpha=0.8)
        
        line3, = ax.plot(df_list[i]['fdps_cs'][idx_list_cs], df_list[i]['powers_cs'][idx_list_cs], 
                        label='Conformal Selection', marker='o', color='darkgreen', alpha=0.8)
        
        ax.legend(loc='best', bbox_to_anchor=(4.2, -3), frameon=True, shadow=False, ncol=3, fontsize=12)
    else:
        ax.plot(df_list[i]['fdps_15_rb'][idx_list_15_rb], df_list[i]['powers_15_rb'][idx_list_15_rb], 
                marker='o', color='steelblue', alpha=0.8)
        
        ax.plot(df_list[i]['fdps_15_rp'][idx_list_15_rp], df_list[i]['powers_15_rp'][idx_list_15_rp], 
                marker='o', color='orange', alpha=0.8)
        
        ax.plot(df_list[i]['fdps_cs'][idx_list_cs], df_list[i]['powers_cs'][idx_list_cs], 
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
fig.suptitle("Power for all 15 Datasets", fontsize=16)

# Display the plot
plt.savefig(os.path.join("pic", "power.png"))
# plt.show()

''''''''''''