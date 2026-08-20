"""
PrivaCare-AI
Differential Privacy using Gaussian Mechanism + Random Forest
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score
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
scaler = MinMaxScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm  = scaler.transform(X_test)

# ─────────────────────────────────────────────
# 4. Ask Epsilon
# ─────────────────────────────────────────────
print("\n------------------------------------------------")
try:
    user_input = input("Enter Epsilon value (e.g. 0.1, 0.5, 1.0) [Default 0.5]: ").strip()
    epsilon = float(user_input) if user_input else 0.5
except ValueError:
    print("Invalid input! Defaulting to 0.5")
    epsilon = 0.5
delta = 1e-5
print(f"--> Epsilon = {epsilon} | Delta = {delta}")
print("------------------------------------------------")

# ─────────────────────────────────────────────
# 5. Baseline Model — No DP Noise
# ─────────────────────────────────────────────
print("\n[1] Training Baseline Random Forest (No DP Noise)...")
rf_baseline = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_baseline.fit(X_train_norm, y_train)
acc_baseline = accuracy_score(y_test, rf_baseline.predict(X_test_norm))
print(f"--> Baseline Accuracy (Without Privacy): {acc_baseline * 100:.2f}%")

# ─────────────────────────────────────────────
# 6. Apply Gaussian DP Noise
# ─────────────────────────────────────────────
sensitivity = 1.0  # features already in [0,1]
sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
print(f"\n[2] Applying Gaussian DP Noise (epsilon={epsilon}, sigma={sigma:.4f})...")

print(f"    Sample BEFORE noise: {X_train_norm[0].round(4)}")
noise      = np.random.normal(loc=0.0, scale=sigma, size=X_train_norm.shape)
X_train_dp = np.clip(X_train_norm + noise, 0.0, 1.0)
print(f"    Sample AFTER  noise: {X_train_dp[0].round(4)}")

# ─────────────────────────────────────────────
# 7. DP Model — Train on Noisy Data
# ─────────────────────────────────────────────
print(f"\n[3] Training DP Random Forest (With Gaussian Noise)...")
rf_dp = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_dp.fit(X_train_dp, y_train)
acc_dp = accuracy_score(y_test, rf_dp.predict(X_test_norm))
print(f"--> DP Model Accuracy (With Privacy, epsilon={epsilon}): {acc_dp * 100:.2f}%")

# ─────────────────────────────────────────────
# 8. Result Summary
# ─────────────────────────────────────────────
print("\n================ RESULT SUMMARY ================")
print(f"Original Model Accuracy (No Privacy) : {acc_baseline * 100:.2f}%")
print(f"Gaussian DP Model Accuracy           : {acc_dp * 100:.2f}%")
print(f"Accuracy Drop (Privacy Cost)         : {(acc_baseline - acc_dp) * 100:.2f}%")
print("================================================")
