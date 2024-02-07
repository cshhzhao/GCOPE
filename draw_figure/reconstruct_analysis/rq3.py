import pandas as pd

# Load the Excel file
file_path = './draw_figure/reconstruct_analysis/RQ3_v5_mean_std.xlsx'
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
plt.rcParams['font.size'] = 14

# Plotting
fig, axs = plt.subplots(1, 3, figsize=(21, 4), dpi=100)
blue_rgba = (0.67843137, 0.84705882, 0.90196078, 0.5)

# Acc
# 设置网格
# axs[0].grid(True, which='both', linestyle='--', linewidth=1.0)
# axs[0].plot(data['λ'], data['Acc'], label='GCOPE', linestyle='-', color='green', marker='o', linewidth=2.5, markersize=6)
axs[0].plot(data['λ'], data['Acc'], label='GCOPE', linestyle='-', marker='o', linewidth=2.5, markersize=6)
axs[0].axhline(y=supervised_acc, label='Supervised', linestyle='--', color='red', linewidth=2.5,xmax=0.95, xmin=0.05)
axs[0].fill_between(data['λ'], data['Acc'] - data['Acc std'], data['Acc'] + data['Acc std'], color='lightblue', alpha=0.3)
# axs[0].set_title('Acc', fontsize=22, fontweight='bold')
# axs[0].set_xlabel('λ', fontsize=22, fontweight='bold')
# axs[0].set_ylabel('Acc', fontsize=22, fontweight='bold')
axs[0].set_xlabel('λ', fontsize=22)
axs[0].set_ylabel('Acc', fontsize=22)
axs[0].legend(prop={'size':16}) # .legend(prop={'weight':'bold', 'size':14})

# AUC
# 设置网格
# axs[1].grid(True, which='both', linestyle='--', linewidth=1.0)
# axs[1].plot(data['λ'], data['AUC'], label='GCOPE', linestyle='-', color='green', marker='o', linewidth=2.5, markersize=6)
axs[1].plot(data['λ'], data['AUC'], label='GCOPE', linestyle='-', marker='o', linewidth=2.5, markersize=6)
axs[1].axhline(y=supervised_auc, label='Supervised', linestyle='--', color='red', linewidth=2.5,xmax=0.95, xmin=0.05)
axs[1].fill_between(data['λ'], data['AUC'] - data['AUC std'], data['AUC'] + data['AUC std'], color='lightblue', alpha=0.3)
# axs[1].set_title('AUC', fontsize=22, fontweight='bold')
# axs[1].set_xlabel('λ', fontsize=22, fontweight='bold')
# axs[1].set_ylabel('AUC', fontsize=22, fontweight='bold')
axs[1].set_xlabel('λ', fontsize=22)
axs[1].set_ylabel('AUC', fontsize=22)
axs[1].legend(prop={'size':16}) # .legend(prop={'weight':'bold', 'size':14})

# F1
# 设置网格
# axs[2].grid(True, which='both', linestyle='--', linewidth=1.0)
# axs[2].plot(data['λ'], data['F1'], label='GCOPE', linestyle='-', color='green', marker='o', linewidth=2.5, markersize=6)
axs[2].plot(data['λ'], data['F1'], label='GCOPE', linestyle='-', marker='o', linewidth=2.5, markersize=6)
axs[2].axhline(y=supervised_f1, label='Supervised', linestyle='--', color='red', linewidth=2.5,xmax=0.95, xmin=0.05)
axs[2].fill_between(data['λ'], data['F1'] - data['F1 std'], data['F1'] + data['F1 std'], color='lightblue', alpha=0.3)
# axs[2].set_title('F1 Score', fontweight='bold')
# axs[2].set_xlabel('λ', fontsize=22, fontweight='bold')
# axs[2].set_ylabel('F1', fontsize=22, fontweight='bold')
axs[2].set_xlabel('λ', fontsize=22)
axs[2].set_ylabel('F1', fontsize=22)
axs[2].legend(prop={'size':16}) # .legend(prop={'weight':'bold', 'size':14})

# for ax in axs:
#     for label in ax.get_yticklabels():
#         label.set_fontweight('bold')
#     for label in ax.get_xticklabels():
#         label.set_fontweight('bold')        

plt.tight_layout()

# Save the plot
plt.savefig('./draw_figure/reconstruct_analysis/reconstruct_analysis_subplots.pdf')

plt.show()