import pandas as pd

# Load the Excel file
file_path = './draw_figure/reconstruct_analysis/RQ3_v4_mean_std.xlsx'
data = pd.read_excel(file_path)

# Display the first few rows of the dataframe to understand its structure
data.head()

import matplotlib.pyplot as plt

# Define supervised method results for comparison
supervised_acc = 0.5219
supervised_auc = 0.8042
supervised_f1 = 0.4667

import matplotlib as mpl

mpl.rcParams.update(mpl.rcParamsDefault)  # Reset to default parameters

# Set plot background color
# mpl.rcParams['axes.facecolor'] = 'whitesmoke'
# mpl.rcParams['axes.edgecolor'] = 'lightgrey'
# mpl.rcParams['axes.grid'] = True
# mpl.rcParams['grid.color'] = 'white'
# mpl.rcParams['grid.linestyle'] = '-'
# mpl.rcParams['grid.linewidth'] = 2
# mpl.rcParams['axes.axisbelow'] = True
# mpl.rcParams['axes.spines.right'] = False
# mpl.rcParams['axes.spines.top'] = False
# mpl.rcParams['axes.spines.left'] = True
# mpl.rcParams['axes.spines.bottom'] = True
# mpl.rcParams['axes.spines.left'] = False
# mpl.rcParams['axes.spines.bottom'] = False
# mpl.rcParams['axes.linewidth'] = 0.5
# mpl.rcParams['xtick.major.width'] = 0.5
# mpl.rcParams['ytick.major.width'] = 0.5
# mpl.rcParams['xtick.color'] = 'grey'
# mpl.rcParams['ytick.color'] = 'grey'
# mpl.rcParams['xtick.labelsize'] = 'medium'
# mpl.rcParams['ytick.labelsize'] = 'medium'

plt.style.use('default')  # Reset to default
plt.rcParams['axes.facecolor'] = '#f0f0f0'  # Background color to a light grey as in the example
plt.rcParams['axes.edgecolor'] = '#f0f0f0'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.5
plt.rcParams['grid.color'] = "white"  # Grid color to white
plt.rcParams['grid.linestyle'] = '-'  # Solid grid line
plt.rcParams['grid.linewidth'] = 1.5
plt.rcParams['axes.axisbelow'] = True  # Grid below the plots
plt.rcParams['axes.linewidth'] = 0  # No axes border
plt.rcParams['axes.spines.top'] = False  # Hide top spine
plt.rcParams['axes.spines.right'] = False  # Hide right spine
plt.rcParams['axes.spines.left'] = True  # Show left spine
plt.rcParams['axes.spines.bottom'] = True  # Show bottom spine
plt.rcParams['axes.spines.left'] = False  # But make them invisible
plt.rcParams['axes.spines.bottom'] = False
plt.rcParams['xtick.bottom'] = False  # Hide xtick marks
plt.rcParams['ytick.left'] = False  # Hide ytick marks
plt.rcParams['font.family'] = 'Times New Roman'

# Plotting
fig, axs = plt.subplots(1, 3, figsize=(21, 4), dpi=100)
blue_rgba = (0.67843137, 0.84705882, 0.90196078, 0.5)
# fig.suptitle('Node classification performance of varying reconstruction loss coefficients (λ)', fontsize=18)
# for ax in axs:
#     ax.grid(True, which='both', linestyle='-', linewidth=1, color='lightgrey', zorder=0)
#     ax.set_facecolor('white')
#     # Hide the right and top spines
#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)
#     ax.spines['left'].set_color('lightgrey')
#     ax.spines['bottom'].set_color('lightgrey')
#     ax.spines['left'].set_linewidth(1)
#     ax.spines['bottom'].set_linewidth(1)
#     # Set zorder for lines to be above the grid
#     ax.set_axisbelow(True)    

# Acc
# 设置网格
# axs[0].grid(True, which='both', linestyle='--', linewidth=1.0)
# axs[0].plot(data['λ'], data['Acc'], label='GCOPE', linestyle='-', color='green', marker='o', linewidth=2.5, markersize=6)
axs[0].plot(data['λ'], data['Acc'], label='GCOPE', linestyle='-', marker='o', linewidth=2.5, markersize=6)
axs[0].axhline(y=supervised_acc, label='Supervised', linestyle='--', color='red', linewidth=2.5,xmax=0.95, xmin=0.05)
axs[0].fill_between(data['λ'], data['Acc'] - data['Acc std'], data['Acc'] + data['Acc std'], color='lightblue', alpha=0.3)
# axs[0].set_title('Acc', fontsize=14, fontweight='bold')
axs[0].set_xlabel('λ', fontsize=14, fontweight='bold')
axs[0].set_ylabel('Acc', fontsize=14, fontweight='bold')
axs[0].legend(prop={'weight':'bold'})

# AUC
# 设置网格
# axs[1].grid(True, which='both', linestyle='--', linewidth=1.0)
# axs[1].plot(data['λ'], data['AUC'], label='GCOPE', linestyle='-', color='green', marker='o', linewidth=2.5, markersize=6)
axs[1].plot(data['λ'], data['AUC'], label='GCOPE', linestyle='-', marker='o', linewidth=2.5, markersize=6)
axs[1].axhline(y=supervised_auc, label='Supervised', linestyle='--', color='red', linewidth=2.5,xmax=0.95, xmin=0.05)
axs[1].fill_between(data['λ'], data['AUC'] - data['AUC std'], data['AUC'] + data['AUC std'], color='lightblue', alpha=0.3)
# axs[1].set_title('AUC', fontsize=14, fontweight='bold')
axs[1].set_xlabel('λ', fontsize=14, fontweight='bold')
axs[1].set_ylabel('AUC', fontsize=14, fontweight='bold')
axs[1].legend(prop={'weight':'bold'})

# F1
# 设置网格
# axs[2].grid(True, which='both', linestyle='--', linewidth=1.0)
# axs[2].plot(data['λ'], data['F1'], label='GCOPE', linestyle='-', color='green', marker='o', linewidth=2.5, markersize=6)
axs[2].plot(data['λ'], data['F1'], label='GCOPE', linestyle='-', marker='o', linewidth=2.5, markersize=6)
axs[2].axhline(y=supervised_f1, label='Supervised', linestyle='--', color='red', linewidth=2.5,xmax=0.95, xmin=0.05)
axs[2].fill_between(data['λ'], data['F1'] - data['F1 std'], data['F1'] + data['F1 std'], color='lightblue', alpha=0.3)
# axs[2].set_title('F1 Score', fontweight='bold')
axs[2].set_xlabel('λ', fontsize=14, fontweight='bold')
axs[2].set_ylabel('F1', fontsize=14, fontweight='bold')
axs[2].legend(prop={'weight':'bold'})

plt.tight_layout()

# Save the plot
plt.savefig('./draw_figure/reconstruct_analysis/reconstruct_analysis.pdf')

plt.show()