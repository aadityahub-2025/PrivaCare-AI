"""
PrivaCare-AI - models/rf_gaussian.py
Model    : Gaussian Naive Bayes
Mechanism: Sufficient Statistics Perturbation (diffprivlib GaussianNB)
Guarantee: Pure epsilon-DP (delta = 0)

DP Approach (Sufficient Statistics Perturbation — diffprivlib GaussianNB):
  - Gaussian Naive Bayes assumes features follow per-class normal distributions.
  - DP is applied by perturbing the SUFFICIENT STATISTICS (per-class counts, sums,
    and sums of squares / variances) rather than perturbing raw data or model weights.
  - IBM diffprivlib GaussianNB implements this via Laplace mechanism perturbation
    on sufficient statistics, satisfying pure epsilon-DP (delta = 0).
  - Guarantee: Pure epsilon-DP (exact differential privacy).

Dataset  : datasets/dataset_1_rf_gaussian.json (ONLY this file)
Reference: Holohan et al. (2019) "diffprivlib: The IBM Differential Privacy Library"
           Dwork & Roth (2014) "The Algorithmic Foundations of Differential Privacy"
"""

import math
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    classification_report,
)
import diffprivlib.models as dp
import joblib
import os
import warnings
warnings.filterwarnings("once")

# ===========================================================================
#  CONFIGURATION
# ===========================================================================
DATASET_PATH = "datasets/dataset_1_rf_gaussian.json"   # ONLY this dataset
N_TRIALS     = 30
BASE_SEED    = 42

# ===========================================================================
#  DATA-INDEPENDENT FEATURE BOUNDS (for [0,1] normalization)
# ===========================================================================
FEATURE_COLS = [
    "glucose_level",            # [30, 300]
    "stress_level",             # [0, 1]
    "heart_rate",               # [30, 220]
    "blood_pressure_systolic",  # [60, 250]
]
DOMAIN_BOUNDS = [
    (30.0, 300.0),   # glucose_level
    (0.0,  1.0),     # stress_level
    (30,   220),     # heart_rate
    (60,   250),     # blood_pressure_systolic
]

GENDER_MAP = {"male": 1, "m": 1, "female": 0, "f": 0, "woman": 0, "man": 1}


# ===========================================================================
#  DATA-INDEPENDENT NORMALIZATION — MinMax to [0, 1]
# ===========================================================================
def normalize_features(X):
    X_norm = X.copy().astype(float)
    for i, (lo, hi) in enumerate(DOMAIN_BOUNDS):
        X_norm[:, i] = np.clip(X_norm[:, i], lo, hi)
        X_norm[:, i] = (X_norm[:, i] - lo) / (hi - lo + 1e-12)
    return X_norm


# ===========================================================================
#  1. LOAD DATA — ONLY dataset_1_rf_gaussian.json
# ===========================================================================
df = pd.read_json(DATASET_PATH)

if "gender" in df.columns and df["gender"].dtype == object:
    df["gender"] = df["gender"].str.strip().str.lower().map(GENDER_MAP).fillna(0).astype(int)

target_col = "health_event"
X  = df[FEATURE_COLS].values.astype(float)
le = LabelEncoder()
y  = le.fit_transform(df[target_col].values)

n_features  = len(FEATURE_COLS)
n_classes   = len(le.classes_)
n_total     = len(X)

print(f"\n{'='*60}")
print(f"  PrivaCare-AI -- Sufficient Statistics DP Training")
print(f"  Model    : Gaussian Naive Bayes")
print(f"  Mechanism: Sufficient Statistics Perturbation (diffprivlib GaussianNB)")
print(f"  Guarantee: Pure epsilon-DP (delta = 0)")
print(f"  Dataset  : {DATASET_PATH}")
print(f"{'='*60}")
print(f"\n  Dataset : {n_total:,} rows | {n_features} features")
print(f"  Target  : '{target_col}' | {n_classes} classes")
print(f"  Features: {FEATURE_COLS}\n")

# ===========================================================================
#  2. TRAIN / TEST SPLIT
# ===========================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=BASE_SEED, stratify=y
)

# ===========================================================================
#  3. DATA-INDEPENDENT NORMALIZATION
# ===========================================================================
X_train_norm = normalize_features(X_train)
X_test_norm  = normalize_features(X_test)

# ===========================================================================
#  4. EPSILON INPUT
# ===========================================================================
print("  " + "="*58)
print("  GOLDEN RULE:")
print("  e (epsilon) badhao  --> privacy KAM,  accuracy ZYADA")
print("  e (epsilon) ghatao  --> privacy ZYADA, accuracy KAM")
print("  " + "-"*58)
print("  Recommended ranges:")
print("    e <= 0.5   -->  High Privacy   (medical / sensitive data)")
print("    e  = 1.0   -->  Balanced       (research)")
print("    e >= 3.0   -->  High Accuracy  (less sensitive data)")
print("  " + "="*58)

try:
    user_input = input("  Enter Epsilon value (e.g. 0.1, 0.5, 1.0) [Default 0.5]: ").strip()
    epsilon = float(user_input) if user_input else 0.5
except ValueError:
    epsilon = 0.5

total_epsilon_basic = N_TRIALS * epsilon

print(f"\n  --> Epsilon (e, per run)      = {epsilon}")
print(f"      Guarantee                 : Pure epsilon-DP (delta = 0)")
print(f"      Trials                    = {N_TRIALS} runs")
print(f"\n  [!] COMPOSITION NOTE:")
print(f"      {N_TRIALS} independent runs on same training data consume:")
print(f"        Basic Composition: e_total = {total_epsilon_basic:.2f}")
print(f"  " + "-"*58 + "\n")

# ===========================================================================
#  5. BASELINE Naive Bayes — No Privacy (for comparison)
# ===========================================================================
print(f"[1] Training Baseline Gaussian Naive Bayes (No DP)...")
nb_baseline = GaussianNB()
nb_baseline.fit(X_train_norm, y_train)
y_base_pred  = nb_baseline.predict(X_test_norm)
acc_baseline = accuracy_score(y_test, y_base_pred)
f1_baseline  = f1_score(y_test, y_base_pred, average="macro")
rec_baseline = recall_score(y_test, y_base_pred, average="macro")
print(f"    Baseline Accuracy : {acc_baseline * 100:.2f}%")
print(f"    Baseline F1 Score : {f1_baseline:.4f}")
print(f"    Baseline Recall   : {rec_baseline:.4f}\n")

# ===========================================================================
#  6. DP Naive Bayes — SUFFICIENT STATISTICS PERTURBATION
# ===========================================================================
print(f"[2] Training DP Gaussian Naive Bayes ({N_TRIALS} trials | e={epsilon})...")
print(f"    Using diffprivlib GaussianNB | Sufficient Statistics Perturbation\n")

# Feature bounds in normalized space [0, 1]
bounds = (
    [0.0] * n_features,
    [1.0] * n_features
)

trial_accs, trial_f1s, trial_recs = [], [], []
nb_dp = None

for trial in range(N_TRIALS):
    seed = BASE_SEED + trial

    # diffprivlib GaussianNB
    nb_dp = dp.GaussianNB(
        epsilon=epsilon,
        bounds=bounds
    )
    # Set random state for reproducibility if available
    if hasattr(nb_dp, 'random_state'):
        nb_dp.random_state = seed

    nb_dp.fit(X_train_norm, y_train)

    y_pred_t = nb_dp.predict(X_test_norm)
    acc_t    = accuracy_score(y_test, y_pred_t)
    f1_t     = f1_score(y_test, y_pred_t, average="macro")
    rec_t    = recall_score(y_test, y_pred_t, average="macro")

    print(f"    Trial {trial+1:02d}/{N_TRIALS} (seed={seed}): Acc={acc_t*100:.2f}%  F1={f1_t:.4f}  Recall={rec_t:.4f}")
    trial_accs.append(acc_t)
    trial_f1s.append(f1_t)
    trial_recs.append(rec_t)

acc_dp  = float(np.mean(trial_accs))
acc_std = float(np.std(trial_accs))
f1_dp   = float(np.mean(trial_f1s))
rec_dp  = float(np.mean(trial_recs))

y_pred = nb_dp.predict(X_test_norm)
report = classification_report(y_test, y_pred, target_names=[str(c) for c in le.classes_])

# ===========================================================================
#  7. RESULT SUMMARY
# ===========================================================================
print(f"\n{'='*60}")
print(f"  RESULT SUMMARY")
print(f"{'='*60}")
print(f"  Baseline Accuracy (No DP)                  : {acc_baseline * 100:.2f}%")
print(f"  DP Accuracy (Sufficient Stats DP e={epsilon}) : {acc_dp * 100:.2f}% +/- {acc_std * 100:.2f}%")
print(f"  Accuracy Drop (Privacy Cost)               : {(acc_baseline - acc_dp) * 100:.2f}%")
print(f"  DP F1 Score (Macro)                        : {f1_dp:.4f}")
print(f"  DP Recall (Macro)                          : {rec_dp:.4f}")
print(f"  Guarantee                                  : Pure epsilon-DP (delta = 0)")
print(f"  Trials Run                                 : {N_TRIALS}")
print(f"{'='*60}")

print("\nDetailed Classification Report (Last Trial):")
print(report)

# ===========================================================================
#  8. SAVE TRAINED MODEL & METADATA
# ===========================================================================
os.makedirs("saved_models", exist_ok=True)
model_artifact = {
    "model_name": "Gaussian Naive Bayes",
    "mechanism": "Sufficient Statistics Perturbation (diffprivlib GaussianNB)",
    "epsilon": epsilon,
    "delta": 0.0,
    "guarantee": "pure epsilon-DP",
    "model": nb_dp,
    "accuracy_baseline": acc_baseline,
    "accuracy_dp_mean": acc_dp,
    "accuracy_dp_std": acc_std,
    "f1_dp_macro": f1_dp,
    "feature_cols": FEATURE_COLS,
    "domain_bounds": DOMAIN_BOUNDS,
    "dataset_used": DATASET_PATH
}
joblib.dump(model_artifact, "saved_models/rf_gaussian_model.pkl")
print("  Model artifact saved to saved_models/rf_gaussian_model.pkl")
print(f"{'='*60}\n")
