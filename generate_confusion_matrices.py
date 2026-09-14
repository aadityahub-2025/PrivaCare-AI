import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import diffprivlib.models as dp
import math

# Set up matplotlib style
plt.style.use('bmh')

# 1. Load Data
df = pd.read_csv("data/dataset.csv")
target_col = "health_event"
feature_cols = [
    "glucose_level",
    "stress_level",
    "heart_rate",
    "blood_pressure_systolic"
]

X = df[feature_cols].values.astype(float)
le = LabelEncoder()
y = le.fit_transform(df[target_col].values)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Normalize
X_train_norm = X_train.copy()
X_test_norm = X_test.copy()

DOMAIN_BOUNDS = {
    "glucose_level": (30.0, 300.0),
    "stress_level": (0.0, 1.0),
    "heart_rate": (30, 220),
    "blood_pressure_systolic": (60, 250)
}

for i, col in enumerate(feature_cols):
    lo, hi = DOMAIN_BOUNDS[col]
    X_train_norm[:, i] = np.clip(X_train_norm[:, i], lo, hi)
    X_train_norm[:, i] = (X_train_norm[:, i] - lo) / (hi - lo + 1e-12)
    X_test_norm[:, i] = np.clip(X_test_norm[:, i], lo, hi)
    X_test_norm[:, i] = (X_test_norm[:, i] - lo) / (hi - lo + 1e-12)

# 2. Train Models at Epsilon = 0.5
epsilon = 0.5

# Model A: RF Laplace (Tree-DP)
bounds = ([0.0] * 4, [1.0] * 4)
rf_model = dp.RandomForestClassifier(
    n_estimators=20, max_depth=10, min_samples_leaf=10, 
    epsilon=epsilon, bounds=bounds, random_state=42
)
rf_model.fit(X_train_norm, y_train)
rf_preds = rf_model.predict(X_test_norm)

# Model B: LR Laplace (Objective DP)
data_norm = math.sqrt(4)
lr_model = dp.LogisticRegression(epsilon=epsilon, data_norm=data_norm, random_state=42)
lr_model.fit(X_train_norm, y_train)
lr_preds = lr_model.predict(X_test_norm)

# 3. Generate Confusion Matrices
cm_rf = confusion_matrix(y_test, rf_preds)
cm_lr = confusion_matrix(y_test, lr_preds)
classes = le.classes_

# 4. Plot Confusion Matrices
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

def plot_cm(ax, cm, title):
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    # Show all ticks
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    # Label them
    ax.set_xticklabels(classes, fontsize=10)
    ax.set_yticklabels(classes, fontsize=10)
    
    # Loop over data dimensions and create text annotations.
    thresh = cm.max() / 2.
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=12, fontweight='bold')
            
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    
    # Rotate x tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

plot_cm(ax1, cm_rf, f"Confusion Matrix: Random Forest (Tree-DP, e={epsilon})")
plot_cm(ax2, cm_lr, f"Confusion Matrix: Logistic Regression (e={epsilon})")

plt.tight_layout()
os.makedirs("visualizations", exist_ok=True)
plt.savefig("visualizations/fig4_confusion_matrices.png", dpi=300)
print("[+] Generated fig4_confusion_matrices.png successfully!")
