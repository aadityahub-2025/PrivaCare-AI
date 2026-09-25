"""
PrivaCare-AI - generate_fig3_confusion_matrix.py
Generate Fig. 3: Confusion Matrix for Deployed Model (Gaussian Naive Bayes at ε=1.0).
Trains on actual test set and computes real confusion matrix via sklearn.metrics.
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import diffprivlib.models as dp
import warnings
warnings.filterwarnings("once")

plt.style.use('bmh')

# Configuration
DATASET_PATH = "datasets/dataset_1_rf_gaussian.json"
FEATURE_COLS = ["glucose_level", "stress_level", "heart_rate", "blood_pressure_systolic"]
DOMAIN_BOUNDS = {
    "glucose_level": (30.0, 300.0),
    "stress_level": (0.0, 1.0),
    "heart_rate": (30, 220),
    "blood_pressure_systolic": (60, 250)
}
CLASS_NAMES = ["Healthy", "Pre-diabetic", "Hypertensive", "Metabolic Risk"]
EPSILON = 1.0
SEED = 42

def normalize(X):
    X_norm = X.copy().astype(float)
    for i, col in enumerate(FEATURE_COLS):
        lo, hi = DOMAIN_BOUNDS[col]
        X_norm[:, i] = np.clip(X_norm[:, i], lo, hi)
        X_norm[:, i] = (X_norm[:, i] - lo) / (hi - lo + 1e-12)
    return X_norm

# 1. Load Data
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

df = pd.read_json(DATASET_PATH)
X = normalize(df[FEATURE_COLS].values.astype(float))
y = df["health_event"].values

# 2. Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

# 3. Train Gaussian NB at epsilon=1.0
nb_model = dp.GaussianNB(
    epsilon=EPSILON,
    bounds=([0.0] * len(FEATURE_COLS), [1.0] * len(FEATURE_COLS))
)
if hasattr(nb_model, 'random_state'):
    nb_model.random_state = SEED

nb_model.fit(X_train, y_train)

# 4. Predict & Compute Confusion Matrix
y_pred = nb_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3])

print("Computed Confusion Matrix (epsilon=1.0):")
print(cm)

# 5. Plot Heatmap
fig, ax = plt.subplots(figsize=(7, 6), dpi=300)

im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=10)

ax.set_title(r"Confusion Matrix — Gaussian Naive Bayes ($\epsilon = 1.0$)", fontsize=13, fontweight='bold', pad=15)
ax.set_xticks(np.arange(len(CLASS_NAMES)))
ax.set_yticks(np.arange(len(CLASS_NAMES)))
ax.set_xticklabels(CLASS_NAMES, fontsize=10, rotation=25, ha="right", fontweight='bold')
ax.set_yticklabels(CLASS_NAMES, fontsize=10, fontweight='bold')

thresh = cm.max() / 2.0
for i in range(len(CLASS_NAMES)):
    for j in range(len(CLASS_NAMES)):
        ax.text(
            j, i, format(cm[i, j], 'd'),
            ha="center", va="center",
            color="white" if cm[i, j] > thresh else "black",
            fontsize=12, fontweight='bold'
        )

ax.set_ylabel('True Class', fontsize=11, fontweight='bold', labelpad=8)
ax.set_xlabel('Predicted Class', fontsize=11, fontweight='bold', labelpad=8)
ax.grid(False)

plt.tight_layout()
os.makedirs("visualizations", exist_ok=True)
output_path = "visualizations/fig3_gaussian_nb_confusion_matrix.png"
plt.savefig(output_path, dpi=300)
plt.close()

print(f"[+] Successfully generated Fig. 3: {output_path}")
