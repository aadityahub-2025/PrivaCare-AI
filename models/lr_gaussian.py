"""
PrivaCare-AI - models/lr_gaussian.py
Model    : Logistic Regression
Mechanism: Analytic Gaussian Mechanism — Output Perturbation
Guarantee: (epsilon, delta)-DP

DP Approach (Output Perturbation — Chaudhuri et al., 2011; Rubinstein et al., 2012):
  - Augment features with constant bias feature: x_aug = [x_norm, 1.0] / sqrt(d + 1)
    ensuring ||x||_2 <= 1 strictly across all samples.
  - Train L2-regularized Logistic Regression with fit_intercept=False on clean data:
      min_W  (1/2) * ||W||_F^2 + C * sum_i ell(W; x_i, y_i)
  - Loss ell is L-Lipschitz with L = sqrt(2) for multinomial cross-entropy.
  - Objective is strongly convex with lambda = 1.
  - Sensitivity of W* under single-sample replacement:
      Delta_2 = 2 * L * C / lambda = 2 * sqrt(2) * C
  - Gaussian noise N(0, sigma^2) added to W* calibrated via Analytic Gaussian Mechanism
    (Balle & Wang, 2018).
  - Guarantee: (epsilon, delta)-DP holds for both feature weights and intercept.

Dataset  : datasets/dataset_3_lr_gaussian.json (ONLY this file)
Reference: Chaudhuri, Monteleoni, Sarwate (JMLR 2011) "Differentially Private ERM"
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
import sys
import warnings
warnings.filterwarnings("once")

# ===========================================================================
#  CONFIGURATION
# ===========================================================================
DATASET_PATH = "datasets/dataset_3_lr_gaussian.json"   # ONLY this dataset
DELTA        = 1e-5
N_TRIALS     = 30
BASE_SEED    = 42
C_REG        = 0.02   # Inverse regularization parameter for L2-regularized ERM

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
#  DATA-INDEPENDENT NORMALIZATION & BIAS AUGMENTATION (||x||_2 <= 1)
# ===========================================================================
def normalize_and_augment(X):
    """
    MinMax scale to [0, 1] using fixed domain bounds, append constant 1.0 bias feature,
    and scale by 1 / sqrt(d + 1) so that ||x||_2 <= 1 strictly holds.
    """
    X_norm = X.copy().astype(float)
    for i, (lo, hi) in enumerate(DOMAIN_BOUNDS):
        X_norm[:, i] = np.clip(X_norm[:, i], lo, hi)
        X_norm[:, i] = (X_norm[:, i] - lo) / (hi - lo + 1e-12)
    
    d = X_norm.shape[1]
    # Append constant bias column 1.0 and scale so Euclidean norm is <= 1
    X_aug = np.hstack([X_norm, np.ones((X_norm.shape[0], 1))]) / math.sqrt(d + 1)
    return X_aug


# ===========================================================================
#  1. LOAD DATA — ONLY dataset_3_lr_gaussian.json
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
print(f"  PrivaCare-AI -- Gaussian Mechanism DP Training")
print(f"  Model    : Logistic Regression")
print(f"  Mechanism: Output Perturbation — (epsilon, delta)-DP")
print(f"  Dataset  : {DATASET_PATH}")
print(f"{'='*60}")
print(f"\n  Dataset : {n_total:,} rows | {n_features} features")
print(f"  Target  : '{target_col}' | {n_classes} classes")
print(f"  Features: {FEATURE_COLS}\n")

# ===========================================================================
#  2. TRAIN / TEST SPLIT & NORMALIZATION
# ===========================================================================
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=BASE_SEED, stratify=y
)

X_train = normalize_and_augment(X_train_raw)
X_test  = normalize_and_augment(X_test_raw)

# ===========================================================================
#  3. EPSILON INPUT
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


# ===========================================================================
#  4. OUTPUT PERTURBATION SENSITIVITY (Chaudhuri et al., 2011)
#  Loss Lipschitz constant L = sqrt(2)
#  Objective: (1/2)||W||_F^2 + C * sum_i ell(W; x_i, y_i)
#  Upper Bound Sensitivity Delta_2 <= 2 * L * C = 2 * sqrt(2) * C
# ===========================================================================
n_train        = X_train.shape[0]
l2_sensitivity = 2.0 * math.sqrt(2) * C_REG
sigma          = analytic_gaussian_sigma(epsilon, DELTA, sensitivity=l2_sensitivity)

# Basic composition over N_TRIALS
total_epsilon_basic = N_TRIALS * epsilon

print(f"\n  --> Epsilon (e, per run)      = {epsilon}")
print(f"      Delta   (d)               = {DELTA}")
print(f"      n_train                   = {n_train:,}")
print(f"      K (classes)               = {n_classes}")
print(f"      C_reg                     = {C_REG}")
print(f"      L2 Sensitivity Bound (2*sqrt(2)*C)= {l2_sensitivity:.6f}")
print(f"      Sigma (Analytic GM)       = {sigma:.6f}  [noise std added to weights]")
print(f"      Trials                    = {N_TRIALS} runs")
print(f"\n  [!] COMPOSITION NOTE:")
print(f"      {N_TRIALS} independent runs on same training data consume:")
print(f"        Basic Composition: e_total = {total_epsilon_basic:.2f}, delta_total = {N_TRIALS * DELTA:.1e}")
print(f"  " + "-"*58 + "\n")

# ===========================================================================
#  5. BASELINES (Tuned Non-Private vs Regularized C=0.02)
# ===========================================================================
print(f"[1] Training Non-Private Baselines...")
# Tuned non-private baseline (unconstrained C=1.0)
lr_tuned = LogisticRegression(C=1.0, max_iter=1000, multi_class='multinomial', random_state=BASE_SEED)
lr_tuned.fit(X_train_raw, y_train)
acc_tuned_base = accuracy_score(y_test, lr_tuned.predict(X_test_raw))

# Regularized baseline (C=0.02, fit_intercept=False on augmented data)
lr_baseline = LogisticRegression(
    C=C_REG, fit_intercept=False, max_iter=1000, multi_class='multinomial', random_state=BASE_SEED
)
lr_baseline.fit(X_train, y_train)
y_base_pred  = lr_baseline.predict(X_test)
acc_baseline = accuracy_score(y_test, y_base_pred)
f1_baseline  = f1_score(y_test, y_base_pred, average="macro")
rec_baseline = recall_score(y_test, y_base_pred, average="macro")

print(f"    Tuned Baseline (C=1.0, unconstrained) : {acc_tuned_base * 100:.2f}%")
print(f"    Regularized Baseline (C={C_REG})       : {acc_baseline * 100:.2f}% (F1={f1_baseline:.4f})\n")

# ===========================================================================
#  6. DP LR — OUTPUT PERTURBATION (Gaussian Mechanism)
# ===========================================================================
print(f"[2] Training DP LR with Output Perturbation ({N_TRIALS} trials | e={epsilon})...")
print(f"    Adding Gaussian noise N(0, {sigma:.6f}^2) to regularized weight matrix\n")

trial_accs, trial_f1s, trial_recs = [], [], []
noisy_W_last = None

for trial in range(N_TRIALS):
    seed = BASE_SEED + trial

    # Train regularized model on training data
    lr_dp = LogisticRegression(
        C=C_REG, fit_intercept=False, max_iter=1000, multi_class='multinomial', random_state=seed
    )
    lr_dp.fit(X_train, y_train)

    # Add calibrated Gaussian noise to all weights (including regularized bias column)
    rng = np.random.RandomState(seed)
    noisy_W = lr_dp.coef_ + rng.normal(0, sigma, size=lr_dp.coef_.shape)
    noisy_W_last = noisy_W

    # Evaluate on clean test set via post-processing
    y_pred_t = np.argmax(X_test @ noisy_W.T, axis=1)
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

# Final evaluation classification report from last trial
y_pred_final = np.argmax(X_test @ noisy_W_last.T, axis=1)
report = classification_report(y_test, y_pred_final, target_names=[str(c) for c in le.classes_])

# ===========================================================================
#  7. RESULT SUMMARY
# ===========================================================================
print(f"\n{'='*60}")
print(f"  RESULT SUMMARY")
print(f"{'='*60}")
print(f"  Baseline Accuracy (No DP)              : {acc_baseline * 100:.2f}%")
print(f"  DP Accuracy (Output Perturbation e={epsilon}): {acc_dp * 100:.2f}% +/- {acc_std * 100:.2f}%")
print(f"  Accuracy Drop (Privacy Cost)           : {(acc_baseline - acc_dp) * 100:.2f}%")
print(f"  DP F1 Score (Macro)                    : {f1_dp:.4f}")
print(f"  DP Recall (Macro)                      : {rec_dp:.4f}")
print(f"  Trials Run                             : {N_TRIALS}")
print(f"{'='*60}")

print("\nDetailed Classification Report (Trial 10):")
print(report)

# ===========================================================================
#  8. SAVE TRAINED MODEL & METADATA
# ===========================================================================
os.makedirs("saved_models", exist_ok=True)
model_artifact = {
    "model_name": "Logistic Regression",
    "mechanism": "Analytic Gaussian Mechanism (Output Perturbation)",
    "epsilon": epsilon,
    "delta": DELTA,
    "C_reg": C_REG,
    "sensitivity": l2_sensitivity,
    "sigma": sigma,
    "noisy_weights": noisy_W_last,
    "accuracy_baseline": acc_baseline,
    "accuracy_dp_mean": acc_dp,
    "accuracy_dp_std": acc_std,
    "f1_dp_macro": f1_dp,
    "feature_cols": FEATURE_COLS,
    "domain_bounds": DOMAIN_BOUNDS,
    "dataset_used": DATASET_PATH
}
joblib.dump(model_artifact, "saved_models/lr_gaussian_model.pkl")
print("  Model artifact saved to saved_models/lr_gaussian_model.pkl")
print(f"{'='*60}\n")
