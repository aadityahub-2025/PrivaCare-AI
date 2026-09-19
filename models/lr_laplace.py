"""
PrivaCare-AI - models/lr_laplace.py
Model    : Logistic Regression
Mechanism: Objective Perturbation — Laplace (pure epsilon-DP)
Guarantee: Pure epsilon-DP  (NO delta needed)

DP Approach (Objective Perturbation — Chaudhuri et al., 2011):
  - Uses diffprivlib DPLogisticRegression
  - Perturbs the objective function (loss + noise) during training
  - data_norm = max L2 norm of a normalized input sample
  - Z-score normalization: features clipped to [-2, 2]
    -> For k=4 features: max L2 = sqrt(k * 2^2) = 2*sqrt(4) = 4.0
  - Guarantee: pure epsilon-DP (exact, no delta approximation)

Dataset  : datasets/dataset_4_lr_laplace.json  (ONLY this file — no other dataset)
Reference: Chaudhuri, Monteleoni & Sarwate (2011) "Differentially Private Empirical Risk Minimization"
           diffprivlib: Holohan et al. (2019), IBM
"""

import math
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from diffprivlib.models import LogisticRegression as DPLogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    classification_report,
)
import joblib
import os
import sys
import warnings
warnings.filterwarnings("once")

# ===========================================================================
#  CONFIGURATION
# ===========================================================================
DATASET_PATH = "datasets/dataset_4_lr_laplace.json"   # ONLY this dataset
N_TRIALS     = 30
BASE_SEED    = 42

# ===========================================================================
#  DATA-INDEPENDENT FEATURE BOUNDS (for Z-score normalization)
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
#  DATA-INDEPENDENT Z-SCORE NORMALIZATION
#  Mean and std derived from DOMAIN_BOUNDS (not from data) -> DP compliant
#  Features are clipped to [-2, 2] -> max L2 per sample = 2*sqrt(k) = 4.0
# ===========================================================================
def normalize_features(X):
    X_norm = X.copy().astype(float)
    for i, (lo, hi) in enumerate(DOMAIN_BOUNDS):
        expected_mu  = (lo + hi) / 2.0
        expected_std = (hi - lo) / 4.0 if (hi - lo) > 0 else 1.0
        X_norm[:, i] = (X_norm[:, i] - expected_mu) / expected_std
        X_norm[:, i] = np.clip(X_norm[:, i], -2.0, 2.0)
    return X_norm

# data_norm = max L2 norm of a normalized input sample
# For k=4 features in [-2, 2]: max L2 = sqrt(4 * 2^2) = sqrt(16) = 4.0
DATA_NORM = 2.0 * math.sqrt(len(FEATURE_COLS))   # = 4.0


# ===========================================================================
#  1. LOAD DATA — ONLY dataset_4_lr_laplace.json
# ===========================================================================
df = pd.read_json(DATASET_PATH)

# Encode gender if present
if "gender" in df.columns and df["gender"].dtype == object:
    df["gender"] = df["gender"].str.strip().str.lower().map(GENDER_MAP).fillna(0).astype(int)

target_col = "health_event"
X  = df[FEATURE_COLS].values.astype(float)
le = LabelEncoder()
y  = le.fit_transform(df[target_col].values)

n_features = len(FEATURE_COLS)
n_classes  = len(le.classes_)
n_total    = len(X)

print(f"\n{'='*60}")
print(f"  PrivaCare-AI -- Laplace Mechanism DP Training")
print(f"  Model    : Logistic Regression")
print(f"  Mechanism: Objective Perturbation (pure epsilon-DP)")
print(f"  Dataset  : {DATASET_PATH}")
print(f"{'='*60}")
print(f"\n  Dataset : {n_total:,} rows | {n_features} features")
print(f"  Target  : '{target_col}' | {n_classes} classes")
print(f"  Features: {FEATURE_COLS}\n")

print("  +--[ PRIVACY SCOPE NOTE ]" + "-"*35 + "+")
print(f"  | Target label ('{target_col}') is NOT DP-protected.              |")
print("  | Standard DP-ML design (Chaudhuri et al., 2011).            |")
print("  | Privacy = what the *model* reveals about training records.  |")
print("  +" + "-"*59 + "+\n")

# ===========================================================================
#  2. TRAIN / TEST SPLIT
# ===========================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=BASE_SEED, stratify=y
)

# ===========================================================================
#  3. DATA-INDEPENDENT Z-SCORE NORMALIZATION
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

if len(sys.argv) > 1:
    try:
        epsilon = float(sys.argv[1])
    except ValueError:
        epsilon = 0.5
else:
    try:
        user_input = input("  Enter Epsilon value (e.g. 0.1, 0.5, 1.0) [Default 0.5]: ").strip()
        epsilon = float(user_input) if user_input else 0.5
    except (ValueError, EOFError):
        epsilon = 0.5


total_epsilon_basic = N_TRIALS * epsilon

print(f"\n  --> Epsilon (e, per run)    = {epsilon}")
print(f"      n_train                 = {X_train_norm.shape[0]:,}")
print(f"      K (classes)             = {n_classes}")
print(f"      data_norm (L2 bound)    = {DATA_NORM:.4f}  (2 * sqrt(k) for Z-score [-2,2])")
print(f"      Mechanism               : Objective Perturbation (pure epsilon-DP)")
print(f"      NO delta needed         : Exact DP guarantee")
print(f"      Trials                  = {N_TRIALS} runs")
print(f"\n  [!] PRIVACY ACCOUNTING NOTE:")
print(f"      - Production Release Guarantee: A single deployed release satisfies target epsilon = {epsilon}")
print(f"      - Evaluation Note: These {N_TRIALS} runs are local Monte Carlo simulations to estimate utility distribution.")
print(f"      - If all {N_TRIALS} models were released publicly: Basic Composition e_total = {total_epsilon_basic:.2f}")
print(f"  " + "-"*58 + "\n")

# ===========================================================================
#  5. BASELINE LR — No Privacy (for comparison)
# ===========================================================================
print(f"[1] Training Baseline Logistic Regression (No DP)...")
lr_baseline = LogisticRegression(C=1.0, max_iter=1000, multi_class='multinomial', random_state=BASE_SEED)
lr_baseline.fit(X_train_norm, y_train)
y_base_pred  = lr_baseline.predict(X_test_norm)
acc_baseline = accuracy_score(y_test, y_base_pred)
f1_baseline  = f1_score(y_test, y_base_pred, average="macro")
rec_baseline = recall_score(y_test, y_base_pred, average="macro")
print(f"    Baseline Accuracy : {acc_baseline * 100:.2f}%")
print(f"    Baseline F1 Score : {f1_baseline:.4f}")
print(f"    Baseline Recall   : {rec_baseline:.4f}\n")

# ===========================================================================
#  6. DP LR — OBJECTIVE PERTURBATION (Laplace / pure epsilon-DP)
#
#  Algorithm (diffprivlib DPLogisticRegression):
#    1. Add calibrated noise to the objective function gradient
#    2. Minimize noisy objective -> DP weight vector
#    3. Guarantee: pure epsilon-DP (Chaudhuri et al., 2011)
#    4. Test on clean X_test (post-processing, DP preserved)
# ===========================================================================
print(f"[2] Training DP LR with Objective Perturbation ({N_TRIALS} trials | e={epsilon})...")
print(f"    Using diffprivlib DPLogisticRegression | data_norm={DATA_NORM:.4f} | pure epsilon-DP\n")

trial_accs, trial_f1s, trial_recs = [], [], []
lr_dp = None

for trial in range(N_TRIALS):
    seed = BASE_SEED + trial

    lr_dp = DPLogisticRegression(
        epsilon=epsilon,
        data_norm=DATA_NORM,
        max_iter=1000,
        random_state=seed
    )
    lr_dp.fit(X_train_norm, y_train)
    y_pred_t = lr_dp.predict(X_test_norm)

    acc_t = accuracy_score(y_test, y_pred_t)
    f1_t  = f1_score(y_test, y_pred_t, average="macro")
    rec_t = recall_score(y_test, y_pred_t, average="macro")

    print(f"    Trial {trial+1}/{N_TRIALS}  (seed={seed}): Acc={acc_t*100:.2f}%  F1={f1_t:.4f}  Recall={rec_t:.4f}")
    trial_accs.append(acc_t)
    trial_f1s.append(f1_t)
    trial_recs.append(rec_t)

acc_dp  = float(np.mean(trial_accs))
acc_std = float(np.std(trial_accs))
f1_dp   = float(np.mean(trial_f1s))
rec_dp  = float(np.mean(trial_recs))

y_pred = lr_dp.predict(X_test_norm)
report = classification_report(y_test, y_pred, target_names=[str(c) for c in le.classes_])

# ===========================================================================
#  7. RESULT SUMMARY
# ===========================================================================
print(f"\n{'='*60}")
print(f"  RESULT SUMMARY")
print(f"{'='*60}")
print(f"  Baseline Accuracy (No DP)              : {acc_baseline * 100:.2f}%")
print(f"  Baseline F1 Score                      : {f1_baseline:.4f}")
print(f"  Baseline Recall                        : {rec_baseline:.4f}")
print(f"  " + "-"*58)
print(f"  DP Accuracy ({N_TRIALS} trials, e={epsilon})          : {acc_dp * 100:.2f}% +/- {acc_std * 100:.2f}%")
print(f"  DP Macro F1 Score                      : {f1_dp:.4f}")
print(f"  DP Macro Recall                        : {rec_dp:.4f}")
print(f"  Accuracy Drop (Privacy Cost)           : {(acc_baseline - acc_dp) * 100:.2f}%")
print(f"  data_norm (L2 bound)                   : {DATA_NORM:.4f}")
print(f"  Mechanism                              : Objective Perturbation (Chaudhuri et al., 2011)")
print(f"  " + "-"*58)
print(f"  PRIVACY BUDGET ACCOUNTING:")
print(f"    Mechanism                : Objective Perturbation (diffprivlib)")
print(f"    Per-run guarantee        : pure {epsilon}-DP  (NO delta needed)")
print(f"    Total consumed (basic)   : {total_epsilon_basic:.4f}-DP  <-- {N_TRIALS} runs x e={epsilon}")
print(f"    (Advanced composition not applicable for pure epsilon-DP)")
print(f"  " + "-"*58)
print(f"\n  PER-CLASS REPORT (last trial):\n")
print(report)

# ===========================================================================
#  8. SAVE MODEL
# ===========================================================================
os.makedirs("saved_models", exist_ok=True)
model_path = "saved_models/lr_laplace.pkl"
joblib.dump(lr_dp, model_path)
print(f"  [+] Model successfully saved to: {model_path}")
print(f"{'='*60}\n")
