"""
PrivaCare-AI - models/lr_gaussian.py
Model    : Logistic Regression
Mechanism: Analytic Gaussian Mechanism — Output Perturbation
Guarantee: (epsilon, delta)-DP

DP Approach (Output Perturbation — Chaudhuri & Monteleoni, 2008):
  - Train standard L2-regularized Logistic Regression on clean data
  - Add calibrated Gaussian noise to trained weight vector w*
  - Sensitivity of w* for L2-reg LR: Delta_2 = (2 * C / n) * sqrt(K)
    where C = regularization param (1/lambda), n = train size, K = classes
  - sigma calibrated via Analytic Gaussian Mechanism (Balle & Wang, 2018)
  - Guarantee: (epsilon, delta)-DP

Dataset  : datasets/dataset_3_lr_gaussian.json  (ONLY this file — no other dataset)
Reference: Chaudhuri & Monteleoni (2008) "Privacy-preserving logistic regression"
           Balle & Wang (NeurIPS 2018) "Improving the Gaussian Mechanism for DP"
"""

import math
import numpy as np
import pandas as pd
from scipy.special import erfc
from sklearn.linear_model import LogisticRegression
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
import warnings
warnings.filterwarnings("once")

# ===========================================================================
#  CONFIGURATION
# ===========================================================================
DATASET_PATH = "datasets/dataset_3_lr_gaussian.json"   # ONLY this dataset
DELTA        = 1e-5
N_TRIALS     = 3
BASE_SEED    = 42
C_REG        = 1.0    # L2 regularization strength (lambda = 1/C_REG)

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
#  ANALYTIC GAUSSIAN SIGMA (Balle & Wang, 2018)
#  Finds the minimum sigma such that the mechanism is (epsilon, delta)-DP
# ===========================================================================
def analytic_gaussian_sigma(epsilon, delta, sensitivity=1.0):
    def phi(t):
        return 0.5 * erfc(-t / math.sqrt(2))
    def delta_of_sigma(s):
        a = sensitivity / (2 * s)
        b = epsilon * s / sensitivity
        return phi(a - b) - math.exp(epsilon) * phi(-a - b)
    lo, hi = 1e-9, 1e6
    for _ in range(1000):
        mid = (lo + hi) / 2
        if delta_of_sigma(mid) <= delta:
            hi = mid
        else:
            lo = mid
    return hi


# ===========================================================================
#  DATA-INDEPENDENT NORMALIZATION — MinMax to [0, 1]
#  Uses fixed DOMAIN_BOUNDS so normalization is data-independent (DP-safe)
# ===========================================================================
def normalize_features(X):
    X_norm = X.copy().astype(float)
    for i, (lo, hi) in enumerate(DOMAIN_BOUNDS):
        X_norm[:, i] = np.clip(X_norm[:, i], lo, hi)
        X_norm[:, i] = (X_norm[:, i] - lo) / (hi - lo + 1e-12)
    return X_norm


# ===========================================================================
#  1. LOAD DATA — ONLY dataset_3_lr_gaussian.json
# ===========================================================================
df = pd.read_json(DATASET_PATH)

# Encode gender if present
if "gender" in df.columns and df["gender"].dtype == object:
    df["gender"] = df["gender"].str.strip().str.lower().map(GENDER_MAP).fillna(0).astype(int)

# Target column
target_col = "health_event"
X  = df[FEATURE_COLS].values.astype(float)
le = LabelEncoder()
y  = le.fit_transform(df[target_col].values)

n_features  = len(FEATURE_COLS)
n_classes   = len(le.classes_)
n_total     = len(X)

print(f"\n{'='*60}")
print(f"  PrivaCare-AI -- Gaussian Mechanism DP Training")
print(f"  Model    : Logistic Regression")
print(f"  Mechanism: Output Perturbation — (epsilon, delta)-DP")
print(f"  Dataset  : {DATASET_PATH}")
print(f"{'='*60}")
print(f"\n  Dataset : {n_total:,} rows | {n_features} features")
print(f"  Target  : '{target_col}' | {n_classes} classes")
print(f"  Features: {FEATURE_COLS}\n")

print("  +--[ PRIVACY SCOPE NOTE ]" + "-"*35 + "+")
print(f"  | Target label ('{target_col}') is NOT DP-protected.              |")
print("  | Standard DP-ML design (Chaudhuri & Monteleoni, 2008).      |")
print("  | Privacy = what the *model weights* reveal about training.   |")
print("  +" + "-"*59 + "+\n")

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

# ===========================================================================
#  OUTPUT PERTURBATION SENSITIVITY
#  Chaudhuri & Monteleoni (2008):
#    Sensitivity = (2 * C) / (n * lambda) = 2 * C / n   [for single class]
#    For K-class multiclass: Delta_2 = (2 * C / n) * sqrt(K)
#  This is the L2 sensitivity of the optimal weight vector w*
# ===========================================================================
n_train       = X_train_norm.shape[0]
l2_sensitivity = (2 * C_REG / n_train) * math.sqrt(n_classes)
sigma          = analytic_gaussian_sigma(epsilon, DELTA, sensitivity=l2_sensitivity)

total_epsilon_basic    = N_TRIALS * epsilon
total_epsilon_advanced = math.sqrt(N_TRIALS) * epsilon

print(f"\n  --> Epsilon (e, per run)      = {epsilon}")
print(f"      Delta   (d)               = {DELTA}")
print(f"      n_train                   = {n_train:,}")
print(f"      K (classes)               = {n_classes}")
print(f"      C_reg                     = {C_REG}")
print(f"      L2 Sensitivity            = 2C/(n*sqrt(K)) = {l2_sensitivity:.8f}")
print(f"      Sigma (Analytic GM)       = {sigma:.8f}  [noise std added to weights]")
print(f"      Trials                    = {N_TRIALS} runs")
print(f"\n  [!] COMPOSITION WARNING:")
print(f"      {N_TRIALS} trials on same data --> TOTAL consumed:")
print(f"        Basic composition    : e_total = {total_epsilon_basic:.4f}  (= {N_TRIALS} x {epsilon})")
print(f"        Advanced composition : e_total ~ {total_epsilon_advanced:.4f}  (= sqrt({N_TRIALS}) x {epsilon})")
print(f"  " + "-"*58 + "\n")

# ===========================================================================
#  5. BASELINE LR — No Privacy (for comparison)
# ===========================================================================
print(f"[1] Training Baseline Logistic Regression (No DP)...")
lr_baseline = LogisticRegression(C=C_REG, max_iter=1000, multi_class='multinomial', random_state=BASE_SEED)
lr_baseline.fit(X_train_norm, y_train)
y_base_pred  = lr_baseline.predict(X_test_norm)
acc_baseline = accuracy_score(y_test, y_base_pred)
f1_baseline  = f1_score(y_test, y_base_pred, average="macro")
rec_baseline = recall_score(y_test, y_base_pred, average="macro")
print(f"    Baseline Accuracy : {acc_baseline * 100:.2f}%")
print(f"    Baseline F1 Score : {f1_baseline:.4f}")
print(f"    Baseline Recall   : {rec_baseline:.4f}\n")

# ===========================================================================
#  6. DP LR — OUTPUT PERTURBATION (Gaussian Mechanism)
#
#  Algorithm:
#    1. Train standard L2-regularized LR on clean (normalized) data
#    2. Add Gaussian noise N(0, sigma^2) to each element of w* and b
#    3. Release noisy model — this is (epsilon, delta)-DP by Chaudhuri 2008
#    4. Test on clean X_test (post-processing, DP is preserved)
# ===========================================================================
print(f"[2] Training DP LR with Output Perturbation ({N_TRIALS} trials | e={epsilon})...")
print(f"    Adding Gaussian noise N(0, {sigma:.8f}^2) to trained weights\n")

trial_accs, trial_f1s, trial_recs = [], [], []
lr_dp = None
for trial in range(N_TRIALS):
    seed = BASE_SEED + trial

    # Step 1: Train standard (non-DP) L2-regularized LR
    lr_dp = LogisticRegression(C=C_REG, max_iter=1000, multi_class='multinomial', random_state=seed)
    lr_dp.fit(X_train_norm, y_train)

    # Step 2: Add Gaussian noise to coefficients (Output Perturbation)
    rng = np.random.RandomState(seed)
    lr_dp.coef_      += rng.normal(0, sigma, size=lr_dp.coef_.shape)
    lr_dp.intercept_ += rng.normal(0, sigma, size=lr_dp.intercept_.shape)

    # Step 3: Evaluate on clean test set (DP post-processing — safe)
    y_pred_t = lr_dp.predict(X_test_norm)
    acc_t    = accuracy_score(y_test, y_pred_t)
    f1_t     = f1_score(y_test, y_pred_t, average="macro")
    rec_t    = recall_score(y_test, y_pred_t, average="macro")

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
print(f"  Gaussian Sigma                         : {sigma:.8f}")
print(f"  Mechanism                              : Output Perturbation (Chaudhuri & Monteleoni, 2008)")
print(f"  " + "-"*58)
print(f"  PRIVACY BUDGET ACCOUNTING:")
print(f"    Mechanism                : Analytic Gaussian (Balle & Wang, 2018)")
print(f"    Per-run guarantee        : ({epsilon}, {DELTA})-DP")
print(f"    Total consumed (basic)   : ({total_epsilon_basic:.4f}, {DELTA})-DP  <-- {N_TRIALS} runs x e={epsilon}")
print(f"    Total consumed (advanced): (~{total_epsilon_advanced:.4f}, ...)-DP  <-- sqrt({N_TRIALS}) x e={epsilon}")
print(f"  " + "-"*58)
print(f"\n  PER-CLASS REPORT (last trial):\n")
print(report)

# ===========================================================================
#  8. SAVE MODEL
# ===========================================================================
os.makedirs("saved_models", exist_ok=True)
model_path = "saved_models/lr_gaussian.pkl"
joblib.dump(lr_dp, model_path)
print(f"  [+] Model successfully saved to: {model_path}")
print(f"{'='*60}\n")
