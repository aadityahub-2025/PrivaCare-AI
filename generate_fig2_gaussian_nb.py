"""
PrivaCare-AI - generate_fig2_gaussian_nb.py
Generate Fig. 2: Accuracy vs Privacy Budget (ε) Trade-off for Gaussian Naive Bayes.
Reads directly from results/benchmark_results.json without any hardcoding.
"""

import os
import json
import math
import matplotlib.pyplot as plt
import numpy as np

# Apply styling
plt.style.use('bmh')

os.makedirs("visualizations", exist_ok=True)

# 1. Load benchmark results
json_path = "results/benchmark_results.json"
if not os.path.exists(json_path):
    raise FileNotFoundError(f"Benchmark results JSON not found at {json_path}. Run compare_all.py first!")

with open(json_path, "r", encoding="utf-8") as f:
    benchmark_data = json.load(f)

epsilons = benchmark_data["epsilons"]
res_list = benchmark_data["results"]

# Filter for Gaussian Naive Bayes only
res_map = {r["Model"]: r for r in res_list}
if "Gaussian Naive Bayes" not in res_map:
    raise KeyError("Gaussian Naive Bayes model not found in benchmark_results.json")

nb_data = res_map["Gaussian Naive Bayes"]

# Extract values dynamically
mean_accs = [nb_data[f"e_{e}_acc_mean"] for e in epsilons]
std_accs = [nb_data[f"e_{e}_acc_std"] for e in epsilons]
baseline_acc = nb_data.get("base_acc_raw", 0.0) * 100.0 if "base_acc_raw" in nb_data else float(nb_data["Baseline"].replace("%", ""))

# 2. Plot Privacy vs Utility Curve
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

# Plot DP Model Curve with Error Bars (N=30 trials)
ax.errorbar(
    epsilons,
    mean_accs,
    yerr=std_accs,
    fmt='-o',
    color='#1f77b4',
    ecolor='#1f77b4',
    elinewidth=1.8,
    capsize=5,
    capthick=1.5,
    linewidth=2.5,
    markersize=8,
    label=r'Gaussian NB DP ($\pm 1\sigma$, $N=30$ trials)'
)

# Annotate each point with mean accuracy value
for eps, mean_val, std_val in zip(epsilons, mean_accs, std_accs):
    ax.annotate(
        f'{mean_val:.2f}% (±{std_val:.2f}%)',
        xy=(eps, mean_val),
        xytext=(0, 10),
        textcoords="offset points",
        ha='center',
        fontsize=10,
        fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#1f77b4", alpha=0.9, lw=1)
    )

# Horizontal dashed line for Non-Private Baseline
ax.axhline(
    y=baseline_acc,
    color='#d62728',
    linestyle='--',
    linewidth=2.0,
    label=f'Non-Private Baseline ({baseline_acc:.2f}%)'
)

# Plot styling and labels
ax.set_title(
    "Privacy-Utility Trade-off: Gaussian Naive Bayes\n(Sufficient Statistics Perturbation)",
    fontsize=13,
    fontweight='bold',
    pad=15
)
ax.set_xlabel(r'Privacy Budget ($\epsilon$)', fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')

# Set appropriate axis bounds
y_min = math.floor(min(mean_accs) - max(std_accs) - 2.0)
y_max = math.ceil(baseline_acc + 2.0)
ax.set_ylim(y_min, y_max)
ax.set_xticks(epsilons)
ax.set_xticklabels([f"$\epsilon={e}$" for e in epsilons], fontsize=11)

ax.legend(loc='lower right', frameon=True, shadow=True, fontsize=10)

plt.tight_layout()
output_path = "visualizations/fig2_gaussian_nb_privacy_utility.png"
plt.savefig(output_path, dpi=300)
plt.close()

print(f"[+] Successfully generated Fig. 2: {output_path}")
