import matplotlib.pyplot as plt
import numpy as np
import os

# Set style for research papers
plt.style.use('bmh')  # built-in matplotlib style
colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3']

# Create visualizations directory if it doesn't exist
os.makedirs("visualizations", exist_ok=True)

# Delete old graphs to keep directory clean
old_graphs = [f for f in os.listdir("visualizations") if f.endswith(".png")]
for g in old_graphs:
    os.remove(os.path.join("visualizations", g))

# ==============================================================================
# PLOT 1: Privacy vs Utility Tradeoff Curve
# ==============================================================================
epsilons = [0.5, 0.7, 1.0]

# Accuracies based on compare_all.py outputs
rf_laplace_acc = [89.3, 91.1, 93.4]   # Tree-DP
lr_laplace_acc = [85.4, 87.2, 90.5]   # Objective DP
lr_gaussian_acc = [80.3, 83.1, 86.0]  # Input DP
rf_gaussian_acc = [23.6, 24.1, 25.2]  # Failing Input DP

fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(epsilons, rf_laplace_acc, marker='o', linewidth=2.5, color=colors[0], label='RF Laplace (Tree-DP)')
ax.plot(epsilons, lr_laplace_acc, marker='s', linewidth=2.5, color=colors[1], label='LR Laplace (Objective DP)')
ax.plot(epsilons, lr_gaussian_acc, marker='^', linewidth=2.5, color=colors[2], label='LR Gaussian (Input DP)')

ax.set_title("Privacy vs Utility Tradeoff", fontsize=16, fontweight='bold', pad=15)
ax.set_xlabel(r'Privacy Budget ($\epsilon$)', fontsize=14)
ax.set_ylabel('Model Accuracy (%)', fontsize=14)
ax.set_ylim(75, 100)
ax.set_xticks(epsilons)
ax.legend(loc='lower right', frameon=True, shadow=True)

plt.tight_layout()
plt.savefig("visualizations/fig1_privacy_utility_curve.png", dpi=300)
plt.close()

# ==============================================================================
# PLOT 2: Model Comparison at Strict Privacy (e=0.5)
# ==============================================================================
models = ['Random Forest\n(Laplace)', 'Logistic Reg\n(Laplace)', 'Logistic Reg\n(Gaussian)']
accuracies = [89.3, 85.4, 80.3]
f1_scores = [0.89, 0.85, 0.79]

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 6))
rects1 = ax.bar(x - width/2, accuracies, width, label='Accuracy (%)', color=colors[0], alpha=0.9)
rects2 = ax.bar(x + width/2, [f*100 for f in f1_scores], width, label='F1 Score (x100)', color=colors[1], alpha=0.9)

ax.set_title(r"Performance at Strict Privacy ($\epsilon=0.5$)", fontsize=16, fontweight='bold', pad=15)
ax.set_ylabel('Score', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylim(0, 105)
ax.legend(loc='upper right')

# Add value labels on top of bars
def autolabel(rects, is_pct=False):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}{"%" if is_pct else ""}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11)

autolabel(rects1, True)
autolabel(rects2)

plt.tight_layout()
plt.savefig("visualizations/fig2_model_comparison.png", dpi=300)
plt.close()

# ==============================================================================
# PLOT 3: The Impact of Input Perturbation on Trees
# ==============================================================================
categories = ['Baseline\n(No DP)', 'RF Laplace\n(Tree-DP)', 'RF Gaussian\n(Input-DP)']
acc_vals = [98.9, 89.3, 23.6]
bar_colors = ['#bdbdbd', colors[0], '#fb6a4a']

fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(categories, acc_vals, color=bar_colors, width=0.6, alpha=0.9)

ax.set_title("Vulnerability of Decision Trees to Input Perturbation", fontsize=16, fontweight='bold', pad=15)
ax.set_ylabel('Accuracy (%)', fontsize=14)
ax.set_ylim(0, 110)

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), 
                textcoords="offset points",
                ha='center', va='bottom', fontsize=12, fontweight='bold')

# Add a text box explaining the drop
textstr = "Input DP adds static noise to features.\nTrees split on pure noise, dropping\naccuracy to random guessing (23%).\nTree-DP rescues utility by perturbing\nthe internal splits instead."
props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
ax.text(1.2, 70, textstr, fontsize=11, bbox=props, verticalalignment='top')

plt.tight_layout()
plt.savefig("visualizations/fig3_input_perturbation_failure.png", dpi=300)
plt.close()

print("[+] Successfully generated 3 research-grade graphs in 'visualizations/' directory.")
