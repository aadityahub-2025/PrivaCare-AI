"""
PrivaCare-AI - generate_fig_data_safety_bars.py
Generate Figure: Data Safety with Differential Privacy (DP) - Before DP vs After DP for Gaussian Naive Bayes.
Demonstrates the classic privacy-utility trade-off across varying privacy budgets (epsilon = 0.1 to 5.0).
"""

import os
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import diffprivlib.models as dp

# 1. Dataset Configuration & Loading
DATASET_PATH = "datasets/dataset_1_rf_gaussian.json"
FEATURE_COLS = ["glucose_level", "stress_level", "heart_rate", "blood_pressure_systolic"]
DOMAIN_BOUNDS = {
    "glucose_level": (30.0, 300.0),
    "stress_level": (0.0, 1.0),
    "heart_rate": (30, 220),
    "blood_pressure_systolic": (60, 250)
}
SEED = 42

def normalize(X):
    X_norm = X.copy().astype(float)
    for i, col in enumerate(FEATURE_COLS):
        lo, hi = DOMAIN_BOUNDS[col]
        X_norm[:, i] = np.clip(X_norm[:, i], lo, hi)
        X_norm[:, i] = (X_norm[:, i] - lo) / (hi - lo + 1e-12)
    return X_norm

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

df = pd.read_json(DATASET_PATH)

# Using a standard edge clinic evaluation cohort (N=500) to demonstrate realistic local sensitivity
# under pure Differential Privacy
N_SAMPLES = 500
df_sample = df.iloc[:N_SAMPLES].copy()

X = normalize(df_sample[FEATURE_COLS].values.astype(float))
y = df_sample["health_event"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

# 2. Train Non-Private Baseline (Before DP)
nb_baseline = GaussianNB()
nb_baseline.fit(X_train, y_train)
y_base_pred = nb_baseline.predict(X_test)
base_accuracy = accuracy_score(y_test, y_base_pred) * 100.0

# 3. Evaluate DP Model Across Specified Epsilon Values (After DP)
epsilons = [0.1, 0.5, 1.0, 2.0, 5.0]
dp_accuracies = []

for eps in epsilons:
    accs = []
    # 20 Monte Carlo trials for smooth empirical mean
    for trial in range(20):
        model = dp.GaussianNB(
            epsilon=eps,
            bounds=([0.0] * len(FEATURE_COLS), [1.0] * len(FEATURE_COLS))
        )
        if hasattr(model, 'random_state'):
            model.random_state = SEED + trial
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        accs.append(accuracy_score(y_test, pred) * 100.0)
    dp_accuracies.append(float(np.mean(accs)))

print(f"Non-Private Baseline Accuracy (Before DP): {base_accuracy:.2f}%")
for eps, dp_acc in zip(epsilons, dp_accuracies):
    print(f"Epsilon = {eps:3.1f} | After DP Accuracy: {dp_acc:.2f}%")

# 4. Matplotlib Plotting — Matching Reference Paper Style
plt.rcParams['font.family'] = 'serif'
fig, ax = plt.subplots(figsize=(7.5, 5.0), dpi=300)

x = np.arange(len(epsilons))
width = 0.28  # Bar width

# Before DP bars (constant baseline across epsilons)
before_dp_vals = [base_accuracy] * len(epsilons)
rects1 = ax.bar(
    x - width/2,
    before_dp_vals,
    width,
    label='Before DP,',
    color='#dca578',
    edgecolor='#6b2d5c',
    linewidth=1.4,
    zorder=3
)

# After DP bars (empirical DP accuracy at each epsilon)
rects2 = ax.bar(
    x + width/2,
    dp_accuracies,
    width,
    label='After DP',
    color='#1ea5de',
    edgecolor='#006680',
    linewidth=1.4,
    zorder=3
)

# Styling details matching reference
ax.set_ylim(0, 100)
ax.set_yticks([0, 20, 40, 60, 80, 100])
ax.set_xticks(x)
ax.set_xticklabels([f"{e:.1f}" for e in epsilons], fontsize=11)

ax.set_xlabel('Epsilon Values', fontsize=12, labelpad=8)
ax.set_ylabel('Accuracy (In%)', fontsize=12, labelpad=8)

# Grid styling
ax.grid(True, linestyle='-', alpha=0.35, color='#999999', zorder=0)

# Ticks on all sides pointing inward
ax.tick_params(direction='in', top=True, right=True, length=4, width=1.0)

# Legend Box (Top Right, framed like the reference paper)
legend = ax.legend(
    loc='upper right',
    frameon=True,
    edgecolor='#444444',
    facecolor='white',
    framealpha=0.9,
    ncol=2,
    fontsize=10.5,
    handletextpad=0.4,
    columnspacing=0.8
)
legend.get_frame().set_linewidth(1.0)

# Caption below the plot
plt.figtext(
    0.5, -0.04,
    "Figure 2.    Data Safety with Differential Privacy (DP).",
    wrap=True, horizontalalignment='center', fontsize=12
)

plt.tight_layout()

os.makedirs("visualizations", exist_ok=True)
output_path = "visualizations/fig_data_safety_before_after_dp.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"[+] Successfully generated Data Safety bar chart at: {output_path}")
