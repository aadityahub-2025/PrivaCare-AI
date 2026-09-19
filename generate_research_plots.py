import matplotlib.pyplot as plt
import numpy as np
import os

# Set style for research papers
plt.style.use('bmh')
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

os.makedirs("visualizations", exist_ok=True)

# Delete old graphs
old_graphs = [f for f in os.listdir("visualizations") if f.endswith(".png") and not f.startswith("fig4")]
for g in old_graphs:
    os.remove(os.path.join("visualizations", g))

# ==============================================================================
# PLOT 1: Privacy vs Utility Tradeoff Curve (N=30 Trials)
# ==============================================================================
epsilons = [0.5, 0.7, 1.0]

# Accuracies based on 30-trial benchmark (compare_all.py)
nb_gaussian_acc = [97.26, 98.21, 98.49]  # Sufficient Stats DP (pure e-DP)
rf_laplace_acc  = [99.12, 99.13, 99.13]  # Tree-based DP (pure e-DP)
lr_gaussian_acc = [94.76, 97.24, 98.34]  # Output Perturbation ((e, d)-DP)
lr_laplace_acc  = [97.82, 98.22, 98.53]  # Objective Perturbation (pure e-DP)

fig, ax = plt.subplots(figsize=(8.5, 6))

ax.plot(epsilons, rf_laplace_acc, marker='o', linewidth=2.5, color=colors[0], label='Random Forest (Tree-DP, pure e-DP)')
ax.plot(epsilons, lr_laplace_acc, marker='s', linewidth=2.5, color=colors[2], label='Logistic Reg (Objective Perturbation, pure e-DP)')
ax.plot(epsilons, lr_gaussian_acc, marker='^', linewidth=2.5, color=colors[1], label='Logistic Reg (Output Gauss, (e,d)-DP)')
ax.plot(epsilons, nb_gaussian_acc, marker='D', linewidth=2.5, color=colors[3], label='Gaussian NB (Sufficient Stats, pure e-DP)')

ax.set_title(r"Privacy vs Utility Tradeoff Across DP Mechanisms ($N=30$ Trials)", fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel(r'Privacy Budget ($\epsilon$)', fontsize=13)
ax.set_ylabel('Model Accuracy (%)', fontsize=13)
ax.set_ylim(92, 100.5)
ax.set_xticks(epsilons)
ax.legend(loc='lower right', frameon=True, shadow=True, fontsize=10)

plt.tight_layout()
plt.savefig("visualizations/fig1_privacy_utility_curve.png", dpi=300)
plt.close()

# ==============================================================================
# PLOT 2: Model Comparison at Strict Privacy (e=0.5, N=30 Trials)
# ==============================================================================
models = [
    'Gaussian NB\n(Stats DP)',
    'Random Forest\n(Tree-DP)',
    'Logistic Reg\n(Output Gauss)',
    'Logistic Reg\n(Objective Lap)'
]
accuracies = [97.26, 99.12, 94.76, 97.82]
f1_scores  = [0.9713, 0.9912, 0.9460, 0.9781]

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, accuracies, width, label='Accuracy (%)', color='#2b5c8f', alpha=0.9)
rects2 = ax.bar(x + width/2, [f*100 for f in f1_scores], width, label='F1 Score (x100)', color='#e27c3e', alpha=0.9)

ax.set_title(r"Performance at Strict Privacy Budget ($\epsilon=0.5, N=30$ Trials)", fontsize=14, fontweight='bold', pad=15)
ax.set_ylabel('Score (%)', fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)
ax.set_ylim(88, 103)
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
baselines = [99.88, 99.66, 99.19, 99.83]
drops = [b - a for b, a in zip(baselines, accuracies)]

fig, ax = plt.subplots(figsize=(9, 5.5))
bar_colors = ['#d95f02', '#1b9e77', '#7570b3', '#e7298a']
bars = ax.bar(models, drops, color=bar_colors, width=0.55, alpha=0.9)

ax.set_title(r"Privacy Cost: Accuracy Drop from Non-DP Baseline ($\epsilon=0.5, N=30$)", fontsize=14, fontweight='bold', pad=15)
ax.set_ylabel('Accuracy Drop (Percentage Points)', fontsize=13)
ax.set_ylim(0, 6.0)

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

print("[+] Successfully regenerated research plots with N=30 benchmark numbers in visualizations/")
