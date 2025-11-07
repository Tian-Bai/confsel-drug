import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
import seaborn as sns
import argparse
from matplotlib.colors import LinearSegmentedColormap, to_rgba

parser = argparse.ArgumentParser()
parser.add_argument('sample', type=float)
parser.add_argument('n_itr', type=int)
parser.add_argument('fop_nominal', type=float)
args = parser.parse_args()

sample = args.sample
n_itr = args.n_itr
fop_nominal = args.fop_nominal

# Set ggplot style for the plots
plt.style.use('ggplot')

df_list = []
dataset_list = ['3A4', 'CB1', 'DPP4', 'HIVINT', 'HIVPROT', 'LOGD', 'METAB', 'NK1', 'OX1', 'OX2', 'PGP', 'PPB', 'RAT_F', 'TDI', 'THROMBIN']

for name in dataset_list:
    df_ones = []
    for j in range(1, 1+n_itr):
        try:
            df = pd.read_csv(os.path.join("result_3z_new", f"{sample:.2f}", f"3zone {name} {sample:.2f}", f"3zone {name} {sample:.2f} {j}.csv"))
            df = df[df['fop_nominal'] == fop_nominal]
        except FileNotFoundError as e:
            print(e)
        df_ones.append(df)
    df = pd.concat(df_ones).groupby("fdp_nominal", as_index=False).mean()
    df_list.append(df)

# Create a grid for subplots
fig, axs = plt.subplots(nrows=3, ncols=5, figsize=(18, 12))
axs = axs.flatten()

# Loop through datasets and plot the data on each subplot
for i, name in enumerate(dataset_list):
    ax = axs[i]

    if i == 0:
        # Plot data for each model and conformal method
        # print(df_list[i]['fdp_nominal'])
        line1, = ax.plot(df_list[i]['fdp_nominal'], df_list[i]['green_fdp'], 
                        label='Green Zone FDR', marker='o', color='steelblue', alpha=0.8)
        
        line2, = ax.plot(df_list[i]['fdp_nominal'], df_list[i]['red_fdp'], 
                        label='Red Zone FOR', marker='o', color='orange', alpha=0.8)
        
        ax.legend(loc='best', bbox_to_anchor=(4.2, -2.9), frameon=True, shadow=False, ncol=3, fontsize=21)
    else:
        ax.plot(df_list[i]['fdp_nominal'], df_list[i]['green_fdp'], 
                marker='o', color='steelblue', alpha=0.8)
        
        ax.plot(df_list[i]['fdp_nominal'], df_list[i]['red_fdp'], 
                marker='o', color='orange', alpha=0.8)
        

    # Reference line for y=x
    ax.plot([0.05, 0.55], [0.05, 0.55], color='grey', alpha=0.7, linestyle='-.')
    
    ax.axhline(fop_nominal, color='grey', alpha=0.7, linestyle='-.')

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
fig.text(0.5, 0.07, 'Nominal FDR', ha='center', fontsize=22)  # Moved down slightly
fig.text(0.03, 0.5, 'Observed FDR/FOR', va='center', rotation='vertical', fontsize=22)  # Moved left slightly

# Title for the entire plot
# fig.suptitle("FDR Control for all 15 Datasets", fontsize=16)

# Display the plot
plt.savefig(os.path.join("pic", "3zfdp.pdf"))
# plt.show()

''''''''''''

# Create a grid for subplots
fig, axs = plt.subplots(nrows=3, ncols=5, figsize=(18, 12))
axs = axs.flatten()

# Loop through datasets and plot the data on each subplot
for i, name in enumerate(dataset_list):
    ax = axs[i]

    y1 = df_list[i]['green_power']
    y2 = df_list[i]['gg_power']
    x = df_list[i]['fdp_nominal']

    if i == 0:
        # Plot data for each model and conformal method
        line1, = ax.plot(df_list[i]['fdp_nominal'], df_list[i]['green_power'], 
                        label='Green Zone Power', marker='o', color='steelblue', alpha=0.8)
        
        line2, = ax.plot(df_list[i]['fdp_nominal'], df_list[i]['gg_power'], 
                        label='Green Zone + Grey Zone Power', marker='o', color='darkviolet', alpha=0.8)
        
        ax.legend(loc='best', bbox_to_anchor=(4.8, -2.9), frameon=True, shadow=False, ncol=3, fontsize=21)
    else:
        ax.plot(df_list[i]['fdp_nominal'], df_list[i]['green_power'], 
                marker='o', color='steelblue', alpha=0.8)
        
        ax.plot(df_list[i]['fdp_nominal'], df_list[i]['gg_power'], 
                marker='o', color='darkviolet', alpha=0.8)

    num_layers = 50
    
    # Define start and end colors in RGBA format
    start_color = to_rgba('lightsteelblue')
    end_color = to_rgba('thistle') # 'thistle' is a nice light purple
    
    # Create an array of colors that smoothly transition from start to end
    all_colors = [
        (
            start_color[0] + (end_color[0] - start_color[0]) * j / num_layers,
            start_color[1] + (end_color[1] - start_color[1]) * j / num_layers,
            start_color[2] + (end_color[2] - start_color[2]) * j / num_layers,
            0.7 # Set a constant alpha for the fill
        )
        for j in range(num_layers)
    ]
    
    # Stack thin horizontal strips, each with a slightly different color
    for j in range(num_layers):
        # Calculate the y-boundaries for this thin strip
        y_bottom = y1 + (y2 - y1) * (j / num_layers)
        y_top = y1 + (y2 - y1) * ((j + 1) / num_layers)
        
        # Fill the strip with its corresponding color
        # Use linewidth=0 to avoid drawing borders between the strips
        ax.fill_between(x, y_bottom, y_top, color=all_colors[j], linewidth=0, alpha=0.4)

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
fig.text(0.5, 0.07, 'Nominal FDR', ha='center', fontsize=22)  # Moved down slightly
fig.text(0.03, 0.5, 'Observed Power', va='center', rotation='vertical', fontsize=22)  # Moved left slightly

# Title for the entire plot
# fig.suptitle("Power for all 15 Datasets", fontsize=16)

# Display the plot
plt.savefig(os.path.join("pic", "3zpower.pdf"))
# plt.show()