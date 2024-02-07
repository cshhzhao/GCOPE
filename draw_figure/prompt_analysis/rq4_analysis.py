import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

# Load the Excel file to check its content
file_path = './draw_figure/prompt_analysis/prompt_analysis.xlsx'
data = pd.read_excel(file_path)

# Create a new DataFrame to hold just the mean values for each method and dataset combination
mean_data = data.copy()
# We already have the mean values in the dataframe, so we can directly use them for plotting

# Extracting unique combinations of training schemes and methods
mean_data['Training schemes'].fillna('', inplace=True)  # Fill NaNs for proper grouping
mean_data['Method + Training'] = mean_data['Training schemes'] + ' ' + mean_data['Methods']
mean_data['Method + Training'] = mean_data['Method + Training'].str.strip()

unique_combinations = mean_data['Method + Training'].unique()
colors = plt.cm.tab20(np.linspace(0, 1, len(unique_combinations)))

# Plotting
plt.figure(figsize=(18, 10))


# Data for plotting
datasets = data.columns[2:]
methods = data['Methods']
methods[1:3] = 'fine-tune+' + methods[1:3]
methods[3:] = 'prompt+' + methods[3:]
unique_methods = methods.unique()
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_methods)))

bar_width = 0.15
indices = np.arange(len(datasets))

for column in data.columns[2:]:  # Skip the first two columns
    data[column] = data[column].str.split(' ± ').str[0].astype(float)

for i, combination in enumerate(unique_combinations):
    # Extract mean performance for the current method + training scheme combination
    combination_data = mean_data[mean_data['Method + Training'] == combination].mean(numeric_only=True).values
    plt.bar(indices + i * bar_width, combination_data, width=bar_width, label=combination, color=colors[i])

plt.xlabel('Datasets', fontsize=14)
plt.ylabel('Mean Performance', fontsize=14)
plt.title('Mean Performance Comparison Across Datasets (Fine-tune vs Prompt)', fontsize=16)
plt.xticks(indices + bar_width * len(unique_combinations) / 2, datasets, rotation=45, ha="right")
plt.legend(title='Method + Training', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# Save the updated plot
updated_plot_path = 'draw_figure/prompt_analysis/mean_performance_comparison_chart_finetune_prompt.png'
plt.savefig(updated_plot_path)
plt.show()
