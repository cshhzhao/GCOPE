# Re-importing necessary libraries and reloading the data after a reset
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def to_percent(y, position):
    # 将数值转换为百分比的字符串
    return "{:.0%}".format(y)

# Load the Excel file again
file_path = './draw_figure/negative_transfer_analysis/overall_negative_transfer.xlsx'
data = pd.read_excel(file_path)

# Adjusting the DataFrame columns due to reset

# plt.rcParams.update({'font.weight': 'normal', 'axes.labelweight': 'normal', 'axes.titleweight': 'normal'})
plt.rcParams.update({'font.size': 22})  # Resetting font size to default for clarity

data.columns = ['Dataset', 'Acc_Supervised', 'AUC_Supervised', 'F1_Supervised', 'Acc_Pubmed', 'AUC_Pubmed', 'F1_Pubmed' , 'Acc_Photo', 'AUC_Photo', 'F1_Photo']

df_plot = data

df_diff = pd.DataFrame({
    'Dataset': df_plot['Dataset'],
    'Acc Difference (Photos - Supervised)': (df_plot['Acc_Photo'][1:] - df_plot['Acc_Supervised'][1:])/df_plot['Acc_Supervised'][1:],
    'Acc Difference (Pubmed - Supervised)': (df_plot['Acc_Pubmed'][1:] - df_plot['Acc_Supervised'][1:])/df_plot['Acc_Supervised'][1:],
})

df_diff['Dataset'][1]='Wisconsin'
# dataset_name = ['Wis.', 'Tex.', 'Cor.','Cha.', 'Squ.']
# df_diff['Dataset'][1:]=dataset_name

# Plotting the differences as trend lines
fig, axs = plt.subplots(1, 1, figsize=(12, 8))
# fig, axs = plt.subplots(1, 1, figsize=(12, 6))

# fig.suptitle('Transfer Curve', x=0.5, y=1.0, fontsize=30, fontweight='bold')

# Plot settings
# plot_settings = {
#     'markersize': 10,
#     'linewidth': 5,
# }

plot_settings = {
    'markersize': 12,
    'linewidth': 6,
}

for i, metric in enumerate(['Acc']):

    photo_diff_col = f'{metric} Difference (Photos - Supervised)'
    pubmed_diff_col = f'{metric} Difference (Pubmed - Supervised)'    

    # Plotting

    axs.plot(df_diff['Dataset'][1:], df_diff[photo_diff_col][1:], marker='o', label='Pretrained on Photos', **plot_settings)
    axs.plot(df_diff['Dataset'][1:], df_diff[pubmed_diff_col][1:], marker='s', label='Pretrained on Pubmed', **plot_settings)

    # axs.plot(df_diff['Dataset'][1:], df_diff[photo_diff_col][1:], marker='s', label='Pretraned on Photos', **plot_settings)
    # axs.plot(df_diff['Dataset'][1:], df_diff[pubmed_diff_col][1:], marker='p', label='Pretraned on Pubmed', **plot_settings)    

    # Titles and labels
    # axs.set_title('Negative Transfer Percentage on Acc',x=0.5, y=1.0, fontweight='bold')
    # axs.set_title(f'{metric} Metric',x=0.5, y=1.0, fontweight='bold')
    axs.set_xlabel('')
    # axs.axhline(0, color='black', linestyle='-',xmax=0.99, xmin=0.01, linewidth=4.0, label='Supervised')  # Add a horizontal line at 0 for reference
    axs.axhline(0, color='red', linestyle='--',xmax=0.99, xmin=0.01, linewidth=5.0, label='Supervised')  # Add a horizontal line at 0 for reference
    axs.tick_params(axis='x', rotation=0)
    # axs.grid(True, which='both', linestyle='--', linewidth=2.0)

    for label in axs.get_xticklabels():
        label.set_fontweight('bold')   

    yticks = [0,-0.1,-0.2,-0.3]
    axs.set_yticks(yticks)
    axs.tick_params(axis='y')

    for label in axs.get_yticklabels():
        if('−'==label._text[0]):
            label.set_fontweight('bold')
            label.set_color('red')
        else:
            label.set_fontweight('bold')
            label.set_color('black')

    axs.spines['left'].set_linewidth(1.5)
    axs.spines['bottom'].set_linewidth(1.5)
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)

    axs.set_ylabel('Negative Transfer Percentage',fontsize=26, fontweight='bold')
    axs.yaxis.set_label_coords(-0.12, 0.5)

    # axs.legend(title='', title_fontsize='20',
    #       loc='bottom right',
    #       bbox_to_anchor=(0.5, 1.17),
    #       ncol=3,
    #       prop={'size': 20, 'weight': 'bold'})

    axs.legend(title='', title_fontsize='20',
               loc='lower right',
               bbox_to_anchor=(1.014, 0.0),
               ncol=1,
               prop={'size': 20, 'weight': 'bold'})    

    from matplotlib.ticker import FuncFormatter
    formatter = FuncFormatter(to_percent)
    axs.yaxis.set_major_formatter(formatter)    

plt.tight_layout()

plt.savefig('./draw_figure/negative_transfer_analysis/overall_negative_transfer.pdf')

plt.show()
