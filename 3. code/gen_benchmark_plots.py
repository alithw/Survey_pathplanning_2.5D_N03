"""
gen_benchmark_plots.py
----------------------
Reads the large-scale benchmark results (509 maps) from '../4. logs/benchmark_results.csv'
and generates beautiful publication-quality boxplots and statistical comparison charts.

Saved directly to '../1. Report/figures/benchmark_boxplots.png'.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.dpi': 200,
})

csv_path = os.path.join(os.path.dirname(__file__), '..', '4. logs', 'benchmark_results.csv')
output_dir = os.path.join(os.path.dirname(__file__), '..', '1. Report', 'figures')
os.makedirs(output_dir, exist_ok=True)

if not os.path.exists(csv_path):
    print(f"[ERROR] Could not find benchmark results at: {csv_path}")
    exit(1)

df = pd.read_csv(csv_path)
print(f"Loaded benchmark results with {len(df)} samples.")

# We will create a 2x2 plot for the 4 key metrics:
# 1. Planning Time (ms)
# 2. Energy consumption (J)
# 3. Path Length (m)
# 4. Smoothness

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Large-Scale Statistical Evaluation on 509 Maps (Weighted A* vs. MCTD-A*)', 
             fontsize=15, fontweight='bold', y=0.98)

metrics = [
    {
        'title': 'Planning Execution Time',
        'ylabel': 'Execution Time (ms)',
        'astar_col': 'a_star_time_ms',
        'mctd_col': 'mctd_time_ms',
        'color_astar': '#2980b9',
        'color_mctd': '#27ae60',
    },
    {
        'title': 'Energy Consumption (Minetti + Bekker)',
        'ylabel': 'Energy Cost (J)',
        'astar_col': 'a_star_energy_j',
        'mctd_col': 'mctd_energy_j',
        'color_astar': '#2980b9',
        'color_mctd': '#27ae60',
    },
    {
        'title': 'Geometric Path Length',
        'ylabel': 'Length (m)',
        'astar_col': 'a_star_length_m',
        'mctd_col': 'mctd_length_m',
        'color_astar': '#2980b9',
        'color_mctd': '#27ae60',
    },
    {
        'title': 'Path Smoothness (Steering Variation)',
        'ylabel': 'Smoothness Score (lower is better)',
        'astar_col': 'a_star_smoothness',
        'mctd_col': 'mctd_smoothness',
        'color_astar': '#2980b9',
        'color_mctd': '#27ae60',
    }
]

for idx, metric in enumerate(metrics):
    row = idx // 2
    col = idx % 2
    ax = axes[row, col]
    
    # Prepare data for plotting
    plot_data = pd.DataFrame({
        'Weighted A*': df[metric['astar_col']],
        'MCTD-A*': df[metric['mctd_col']]
    })
    
    # Create boxplot
    sns.boxplot(data=plot_data, ax=ax, palette=[metric['color_astar'], metric['color_mctd']], width=0.5,
                showmeans=True, meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"6"})
    
    # Add labels and formatting
    ax.set_title(metric['title'], fontweight='bold', pad=10)
    ax.set_ylabel(metric['ylabel'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Print individual means
    astar_mean = df[metric['astar_col']].mean()
    mctd_mean = df[metric['mctd_col']].mean()
    diff_pct = (astar_mean - mctd_mean) / astar_mean * 100
    
    ax.text(0.1, 0.9, f"Mean: {astar_mean:.2f}", transform=ax.transAxes, ha='left', va='center', fontweight='semibold', color=metric['color_astar'])
    ax.text(0.9, 0.9, f"Mean: {mctd_mean:.2f}", transform=ax.transAxes, ha='right', va='center', fontweight='semibold', color=metric['color_mctd'])
    
    # For Planning Time, we show that it's slightly higher due to System 1 overhead & fallback.
    # For others, they are nearly identical because of A* fallback safety.
    
plt.tight_layout(rect=[0, 0, 1, 0.95])
save_path = os.path.join(output_dir, 'benchmark_boxplots.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"[OK] Generated benchmark boxplots at: {save_path}")
plt.close()
