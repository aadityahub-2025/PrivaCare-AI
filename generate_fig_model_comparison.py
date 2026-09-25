"""
PrivaCare-AI - generate_fig_model_comparison.py
Generate Bonus Figure: Model Comparison Bar Chart at ε=1.0 with Gaussian NB highlighted.
Loads data directly from results/benchmark_results.json without hardcoded numbers.
Compact, elegant proportions (reduced width, sleek narrow bars).
"""

import os
import json
import math
import matplotlib.pyplot as plt
import numpy as np

# Styling
plt.style.use('bmh')

os.makedirs("visualizations", exist_ok=True)

# 1. Load benchmark results
json_path = "results/benchmark_results.json"
if not os.path.exists(json_path):
    raise FileNotFoundError(f"Benchmark results JSON not found at {json_path}. Run compare_all.py first!")

with open(json_path, "r", encoding="utf-8") as f:
    benchmark_data = json.load(f)

res_list = benchmark_data["results"]
res_map = {r["Model"]: r for r in res_list}

models_keys = [
    "Gaussian Naive Bayes",
    "Random Forest",
    "Logistic Reg (Gaussian)",
    "Logistic Reg (Laplace)"
]

models_display = [
    "Gaussian NB\n(Stats DP)\n★ DEPLOYED",
    "Random Forest\n(Tree-DP)",
    "Logistic Reg\n(Output Gauss)",
    "Logistic Reg\n(Objective Lap)"
]

# Extract accuracies and std deviations at epsilon = 1.0
accuracies_e10 = [res_map[m]["e_1.0_acc_mean"] for m in models_keys]
std_e10 = [res_map[m]["e_1.0_acc_std"] for m in models_keys]
f1_e10 = [float(res_map[m]["e=1.0 F1"]) * 100.0 for m in models_keys]

# 2. Plot Grouped Bar Chart with Sleeker Width & Proportions
x = np.arange(len(models_display))
width = 0.26  # Sleek, narrow bar width

# Proportions: Compact width (7.2), taller height (6.8) for slim, aesthetic bars
fig, ax = plt.subplots(figsize=(7.2, 6.8), dpi=300)

# Colors: Highlight Gaussian NB with distinct deep navy/blue and gold border
colors_acc = ['#1b4f72', '#566573', '#797d7f', '#99a3a4']
edge_colors = ['#d4ac0d', 'none', 'none', 'none']
edge_widths = [2.2, 0, 0, 0]

rects1 = ax.bar(
    x - width/2,
    accuracies_e10,
    width,
    yerr=std_e10,
    capsize=3.5,
    label='Accuracy (%) ± 1σ',
    color=colors_acc,
    edgecolor=edge_colors,
    linewidth=edge_widths,
    alpha=0.95,
    zorder=3
)

colors_f1 = ['#d35400', '#eb984e', '#edbb99', '#f5cba7']
rects2 = ax.bar(
    x + width/2,
    f1_e10,
    width,
    label='F1-Score (Macro %)',
    color=colors_f1,
    edgecolor=edge_colors,
    linewidth=edge_widths,
    alpha=0.9,
    zorder=3
)

ax.set_title(
    r"DP-ML Mechanism Comparison at $\epsilon = 1.0$" "\n" r"(Selected Model Highlighted)",
    fontsize=12.5,
    fontweight='bold',
    pad=14
)
ax.set_ylabel('Score (%)', fontsize=11.5, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models_display, fontsize=9.5, fontweight='bold')

min_val = min(accuracies_e10 + f1_e10)
ax.set_ylim(math.floor(min_val - 6.0), 92.0)
ax.grid(True, linestyle='-', alpha=0.35, color='#999999', zorder=0)

ax.legend(loc='upper right', fontsize=9.5, frameon=True, facecolor='white', framealpha=0.9)

# Annotations on top of bars
def autolabel(rects, is_acc=True):
    for idx, rect in enumerate(rects):
        height = rect.get_height()
        font_weight = 'bold' if idx == 0 else 'normal'
        ax.annotate(
            f'{height:.1f}%',
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha='center', va='bottom',
            fontsize=8.5,
            fontweight=font_weight
        )

autolabel(rects1, is_acc=True)
autolabel(rects2, is_acc=False)

plt.tight_layout()
output_path = "visualizations/fig_model_selection_comparison.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"[+] Successfully generated updated compact model comparison plot at: {output_path}")
