import matplotlib.pyplot as plt
import numpy as np
import os
import json
import math

# Set style for research papers
plt.style.use('bmh')
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

os.makedirs("visualizations", exist_ok=True)

# Delete old graphs (except fig4 which is confusion matrix, or delete all to rebuild)
old_graphs = [f for f in os.listdir("visualizations") if f.endswith(".png") and not f.startswith("fig4")]
for g in old_graphs:
    os.remove(os.path.join("visualizations", g))

# Load benchmark results from results/benchmark_results.json
json_path = "results/benchmark_results.json"
if not os.path.exists(json_path):
    raise FileNotFoundError(f"Benchmark results JSON not found at {json_path}. Run compare_all.py first!")

with open(json_path, "r", encoding="utf-8") as f:
    benchmark_data = json.load(f)

epsilons = benchmark_data["epsilons"]
res_list = benchmark_data["results"]

# Map model name to result dict
res_map = {r["Model"]: r for r in res_list}

nb_gaussian_acc = [res_map["Gaussian Naive Bayes"][f"e_{e}_acc_mean"] for e in epsilons]
rf_laplace_acc  = [res_map["Random Forest"][f"e_{e}_acc_mean"] for e in epsilons]
lr_gaussian_acc = [res_map["Logistic Reg (Gaussian)"][f"e_{e}_acc_mean"] for e in epsilons]
lr_laplace_acc  = [res_map["Logistic Reg (Laplace)"][f"e_{e}_acc_mean"] for e in epsilons]

# ==============================================================================
# PLOT 1: Privacy vs Utility Tradeoff Curve (N=30 Trials)
# ==============================================================================
fig, ax = plt.subplots(figsize=(8.5, 6))

ax.plot(epsilons, rf_laplace_acc, marker='o', linewidth=2.5, color=colors[0], label='Random Forest (Tree-DP, pure e-DP)')
ax.plot(epsilons, lr_laplace_acc, marker='s', linewidth=2.5, color=colors[2], label='Logistic Reg (Objective Perturbation, pure e-DP)')
ax.plot(epsilons, lr_gaussian_acc, marker='^', linewidth=2.5, color=colors[1], label='Logistic Reg (Output Gauss, (e,d)-DP)')
ax.plot(epsilons, nb_gaussian_acc, marker='D', linewidth=2.5, color=colors[3], label='Gaussian NB (Sufficient Stats, pure e-DP)')

ax.set_title(r"Privacy vs Utility Tradeoff Across DP Mechanisms ($N=30$ Trials)", fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel(r'Privacy Budget ($\epsilon$)', fontsize=13)
ax.set_ylabel('Model Accuracy (%)', fontsize=13)

min_y = min(nb_gaussian_acc + rf_laplace_acc + lr_gaussian_acc + lr_laplace_acc)
ax.set_ylim(math.floor(min_y - 4.0), 92.0)
ax.set_xticks(epsilons)
ax.legend(loc='lower right', frameon=True, shadow=True, fontsize=10)

plt.tight_layout()
plt.savefig("visualizations/fig1_privacy_utility_curve.png", dpi=300)
plt.close()

# ==============================================================================
# PLOT 2: Model Comparison at Strict Privacy (e=0.5, N=30 Trials)
# ==============================================================================
models_display = [
    'Gaussian NB\n(Stats DP)',
    'Random Forest\n(Tree-DP)',
    'Logistic Reg\n(Output Gauss)',
    'Logistic Reg\n(Objective Lap)'
]

accuracies_e05 = [
    res_map["Gaussian Naive Bayes"]["e_0.5_acc_mean"],
    res_map["Random Forest"]["e_0.5_acc_mean"],
    res_map["Logistic Reg (Gaussian)"]["e_0.5_acc_mean"],
    res_map["Logistic Reg (Laplace)"]["e_0.5_acc_mean"]
]

f1_scores_e05 = [
    float(res_map["Gaussian Naive Bayes"]["e=0.5 F1"]),
    float(res_map["Random Forest"]["e=0.5 F1"]),
    float(res_map["Logistic Reg (Gaussian)"]["e=0.5 F1"]),
    float(res_map["Logistic Reg (Laplace)"]["e=0.5 F1"])
]

x = np.arange(len(models_display))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, accuracies_e05, width, label='Accuracy (%)', color='#2b5c8f', alpha=0.9)
rects2 = ax.bar(x + width/2, [f*100 for f in f1_scores_e05], width, label='F1 Score (x100)', color='#e27c3e', alpha=0.9)

ax.set_title(r"Performance at Strict Privacy Budget ($\epsilon=0.5, N=30$ Trials)", fontsize=14, fontweight='bold', pad=15)
ax.set_ylabel('Score (%)', fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(models_display, fontsize=11)
min_score = min(accuracies_e05 + [f*100 for f in f1_scores_e05])
ax.set_ylim(math.floor(min_score - 8.0), 95.0)
ax.legend(loc='upper right', fontsize=11)

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.savefig("visualizations/fig2_model_comparison.png", dpi=300)
plt.close()

# ==============================================================================
# PLOT 3: Accuracy Drop from Baseline at Strict Privacy (e=0.5)
# ==============================================================================
baselines = [
    res_map["Gaussian Naive Bayes"]["base_acc_raw"] * 100.0,
    res_map["Random Forest"]["base_acc_raw"] * 100.0,
    res_map["Logistic Reg (Gaussian)"]["base_acc_raw"] * 100.0,
    res_map["Logistic Reg (Laplace)"]["base_acc_raw"] * 100.0
]

drops = [b - a for b, a in zip(baselines, accuracies_e05)]

fig, ax = plt.subplots(figsize=(9, 5.5))
bar_colors = ['#d95f02', '#1b9e77', '#7570b3', '#e7298a']
bars = ax.bar(models_display, drops, color=bar_colors, width=0.55, alpha=0.9)

ax.set_title(r"Privacy Cost: Accuracy Drop from Non-DP Baseline ($\epsilon=0.5, N=30$)", fontsize=14, fontweight='bold', pad=15)
ax.set_ylabel('Accuracy Drop (Percentage Points)', fontsize=13)
ax.set_ylim(0, max(drops) * 1.25)

for bar in bars:
    height = bar.get_height()
    ax.annotate(f'-{height:.2f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig("visualizations/fig3_privacy_cost.png", dpi=300)
plt.close()

print("[+] Successfully regenerated research plots dynamically from results/benchmark_results.json in visualizations/")
