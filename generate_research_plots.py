import matplotlib.pyplot as plt
import numpy as np
import os

# Set style for research papers
plt.style.use('bmh')
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

os.makedirs("visualizations", exist_ok=True)

# Delete old graphs
old_graphs = [f for f in os.listdir("visualizations") if f.endswith(".png")]
for g in old_graphs:
    os.remove(os.path.join("visualizations", g))

# ==============================================================================
# PLOT 1: Privacy vs Utility Tradeoff Curve
# ==============================================================================
epsilons = [0.5, 0.7, 1.0]

# Accuracies based on standardized benchmark runs (compare_all.py)
nb_gaussian_acc = [93.12, 97.30, 99.77]  # Sufficient Stats DP (pure e-DP)
rf_laplace_acc  = [99.21, 99.22, 99.19]  # Tree-based DP (pure e-DP)
lr_gaussian_acc = [98.66, 99.15, 99.32]  # Output Perturbation ((e, d)-DP)
lr_laplace_acc  = [97.67, 98.02, 98.32]  # Objective Perturbation (pure e-DP)

fig, ax = plt.subplots(figsize=(8.5, 6))

ax.plot(epsilons, rf_laplace_acc, marker='o', linewidth=2.5, color=colors[0], label='Random Forest (Tree-DP, pure e-DP)')
ax.plot(epsilons, lr_gaussian_acc, marker='^', linewidth=2.5, color=colors[1], label='Logistic Reg (Output Gauss, (e,d)-DP)')
ax.plot(epsilons, lr_laplace_acc, marker='s', linewidth=2.5, color=colors[2], label='Logistic Reg (Objective Lap, pure e-DP)')
ax.plot(epsilons, nb_gaussian_acc, marker='D', linewidth=2.5, color=colors[3], label='Gaussian NB (Sufficient Stats, pure e-DP)')

ax.set_title("Privacy vs Utility Tradeoff Across DP Mechanisms", fontsize=15, fontweight='bold', pad=15)
ax.set_xlabel(r'Privacy Budget ($\epsilon$)', fontsize=13)
ax.set_ylabel('Model Accuracy (%)', fontsize=13)
ax.set_ylim(90, 100.5)
ax.set_xticks(epsilons)
ax.legend(loc='lower right', frameon=True, shadow=True, fontsize=10)

plt.tight_layout()
plt.savefig("visualizations/fig1_privacy_utility_curve.png", dpi=300)
plt.close()

# ==============================================================================
# PLOT 2: Model Comparison at Strict Privacy (e=0.5)
# ==============================================================================
models = [
    'Gaussian NB\n(Stats DP)',
    'Random Forest\n(Tree-DP)',
    'Logistic Reg\n(Output Gauss)',
    'Logistic Reg\n(Objective Lap)'
]
accuracies = [93.12, 99.21, 98.66, 97.67]
f1_scores  = [0.9253, 0.9920, 0.9866, 0.9767]

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, accuracies, width, label='Accuracy (%)', color='#2b5c8f', alpha=0.9)
rects2 = ax.bar(x + width/2, [f*100 for f in f1_scores], width, label='F1 Score (x100)', color='#e27c3e', alpha=0.9)

ax.set_title(r"Performance at Strict Privacy Budget ($\epsilon=0.5$)", fontsize=15, fontweight='bold', pad=15)
ax.set_ylabel('Score (%)', fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)
ax.set_ylim(85, 103)
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

ax.set_title(r"Privacy Cost: Accuracy Drop from Non-DP Baseline ($\epsilon=0.5$)", fontsize=15, fontweight='bold', pad=15)
ax.set_ylabel('Accuracy Drop (Percentage Points)', fontsize=13)
ax.set_ylim(0, 8.0)

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

print("[+] Successfully generated 3 updated research plots in 'visualizations/'")
