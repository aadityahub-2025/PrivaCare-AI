"""
PrivaCare-AI
Differential Privacy using diffprivlib (IBM) + Random Forest
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score
import diffprivlib.models as dp
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. Load Data
# ─────────────────────────────────────────────
csv_path = "data/dataset.csv"
df = pd.read_csv(csv_path)

# Auto-detect target column
if "health_event" in df.columns:
    target_col = "health_event"
elif "disease" in df.columns:
    target_col = "disease"
else:
    target_col = df.columns[-1]

# Drop non-feature columns
drop_cols = ["timestamp", "device_id", "patient_id", "is_synthetic", target_col]
feature_cols = [c for c in df.columns if c not in drop_cols
                and df[c].dtype in [np.float64, np.int64, float, int]]

X = df[feature_cols].values.astype(float)
y = LabelEncoder().fit_transform(df[target_col].values)

print(f"\n--> Dataset loaded: {X.shape[0]} rows, {X.shape[1]} features")
print(f"    Target: '{target_col}' | Classes: {len(np.unique(y))}")

# ─────────────────────────────────────────────
# 2. Train / Test Split
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ─────────────────────────────────────────────
# 3. Normalize features to [0, 1]
# ─────────────────────────────────────────────
scaler = MinMaxScaler(feature_range=(0, 1))
X_train_norm = scaler.fit_transform(X_train)
X_test_norm  = scaler.transform(X_test)

# Define bounds for the DP model
bounds = ([0.0] * X_train_norm.shape[1], [1.0] * X_train_norm.shape[1])

# ─────────────────────────────────────────────
# 4. Ask Epsilon (Only Epsilon, Default 0.5)
# ─────────────────────────────────────────────
print("\n------------------------------------------------")
try:
    user_input = input("Enter Epsilon value (e.g. 0.1, 0.5, 1.0) [Default 0.5]: ").strip()
    epsilon = float(user_input) if user_input else 0.5
except ValueError:
    print("Invalid input! Defaulting to 0.5")
    epsilon = 0.5
print(f"--> Epsilon = {epsilon}")
print("------------------------------------------------")

# ─────────────────────────────────────────────
# 5. Baseline Model — No DP Noise
# ─────────────────────────────────────────────
print("\n[1] Training Baseline Random Forest (No DP Noise)...")
rf_baseline = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_baseline.fit(X_train_norm, y_train)
acc_baseline = accuracy_score(y_test, rf_baseline.predict(X_test_norm))

# ─────────────────────────────────────────────
# 6. DP Model — Algorithmic Privacy
# ─────────────────────────────────────────────
print(f"\n[2] Training DP Random Forest (Algorithmic Privacy, epsilon={epsilon})...")
rf_dp = dp.RandomForestClassifier(
    n_estimators=100, 
    epsilon=epsilon, 
    bounds=bounds,
    random_state=42
)
rf_dp.fit(X_train_norm, y_train)
acc_dp = accuracy_score(y_test, rf_dp.predict(X_test_norm))

# ─────────────────────────────────────────────
# 7. Result Summary
# ─────────────────────────────────────────────
print("\n================ RESULT SUMMARY ================")
print(f"Original Model Accuracy (No Privacy) : {acc_baseline * 100:.2f}%")
print(f"DP Model Accuracy (Epsilon={epsilon})      : {acc_dp * 100:.2f}%")
print(f"Accuracy Drop (Privacy Cost)         : {(acc_baseline - acc_dp) * 100:.2f}%")
print("================================================")
