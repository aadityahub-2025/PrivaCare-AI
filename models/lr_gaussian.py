"""
PrivaCare-AI - models/lr_gaussian.py
Model    : Logistic Regression
Mechanism: Analytic Gaussian Mechanism (Balle & Wang, 2018)
Guarantee: (epsilon, delta)-DP

DP Approach:
  - Features normalized to [0,1] via data-independent DOMAIN_BOUNDS
    L-inf sensitivity = 1.0 per feature (fully data-independent)
  - Gaussian noise N(0, sigma^2) added to TRAINING features
    sigma = analytic_gaussian_sigma(epsilon, delta, sensitivity=1.0)
  - sklearn LogisticRegression trained on noisy features
  - Test features stay CLEAN (standard DP-ML practice)
  - Guarantee: (epsilon, delta)-DP via Analytic Gaussian Mechanism

Compared to rf_gaussian.py:
  - rf_gaussian.py uses diffprivlib RandomForest (tree-based splits)
  - lr_gaussian.py uses sklearn LogisticRegression (linear boundary)
  - LR is faster but may have lower accuracy on non-linear data

Reference: Balle & Wang (NeurIPS 2018), Improving the Gaussian Mechanism for DP
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
import warnings
warnings.filterwarnings("ignore")

# ===========================================================================
#  CONFIGURATION
# ===========================================================================
DELTA        = 1e-5
N_TRIALS     = 3
N_ESTIMATORS = 100   # not used for LR, kept for consistency
BASE_SEED    = 42
STRICT_DOMAIN_BOUNDS = True

# ===========================================================================
#  DATA-INDEPENDENT FEATURE BOUNDS
# ===========================================================================
DOMAIN_BOUNDS = {
    "heart_rate":               (30,    220),
    "blood_oxygen":             (70.0, 100.0),
    "blood_pressure_systolic":  (60,    250),
    "blood_pressure_diastolic": (40,    150),
    "glucose_level":            (50.0,  500.0),
    "body_temperature":         (35.0,  42.0),
    "respiratory_rate":         (8,     40),
    "activity_level":           (0.0,   10.0),
    "sleep_quality":            (0.0,   10.0),
    "stress_level":             (0.0,   10.0),
    "hrv_sdnn":                 (5.0,  200.0),
    "steps_count":              (0,    50000),
    "calories_burned":          (0,     5000),
    "age":                      (0,     120),
    "gender":                   (0,     1),
}

BINARY_FEATURES = {"gender"}
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
#  DATA-INDEPENDENT NORMALIZATION
# ===========================================================================
def normalize_with_domain_bounds(X, feature_names, fallback_bounds=None):
    X_norm = X.copy().astype(float)
    computed_fallbacks = {}
    for i, col in enumerate(feature_names):
        if col in DOMAIN_BOUNDS:
            lo, hi = DOMAIN_BOUNDS[col]
        elif fallback_bounds and col in fallback_bounds:
            lo, hi = fallback_bounds[col]
        else:
            if STRICT_DOMAIN_BOUNDS:
                raise ValueError(
                    f"\n  [DP ERROR] Feature '{col}' not in DOMAIN_BOUNDS.\n"
                    f"  Add '{col}' to DOMAIN_BOUNDS dict."
                )
            lo, hi = float(X[:, i].min()), float(X[:, i].max())
            computed_fallbacks[col] = (lo, hi)
            print(f"  WARNING: No domain bound for '{col}' -- using train min/max.")
        X_norm[:, i] = np.clip(X_norm[:, i], lo, hi)
        X_norm[:, i] = (X_norm[:, i] - lo) / (hi - lo + 1e-12)
    return X_norm, computed_fallbacks


# ===========================================================================
#  GAUSSIAN NOISE INJECTION
# ===========================================================================
def add_gaussian_dp_noise(X_norm, feature_names, sigma, rng):
    X_noisy = X_norm.copy()
    for i, col in enumerate(feature_names):
        if col in BINARY_FEATURES:
            continue
        noise = rng.normal(loc=0.0, scale=sigma, size=X_norm.shape[0])
        X_noisy[:, i] = np.clip(X_noisy[:, i] + noise, 0.0, 1.0)
    return X_noisy


# ===========================================================================
#  1. LOAD DATA
# ===========================================================================
csv_path = "data/dataset.csv"
df = pd.read_csv(csv_path)

if "health_event" in df.columns:
    target_col = "health_event"
elif "disease" in df.columns:
    target_col = "disease"
else:
    target_col = df.columns[-1]

if "gender" in df.columns and df["gender"].dtype == object:
    def encode_gender(val):
        v = str(val).strip().lower()
        if v not in GENDER_MAP:
            raise ValueError(f"Unknown gender value: '{val}'. Expected: {list(GENDER_MAP.keys())}")
        return GENDER_MAP[v]
    df["gender"] = df["gender"].apply(encode_gender)

drop_cols    = ["timestamp", "device_id", "patient_id", "is_synthetic", target_col]
feature_cols = [c for c in df.columns
                if c not in drop_cols
                and df[c].dtype in [np.float64, np.int64, float, int]]

X  = df[feature_cols].values.astype(float)
le = LabelEncoder()
y  = le.fit_transform(df[target_col].values)
n_features = len(feature_cols)

print(f"\n{'='*60}")
print(f"  PrivaCare-AI -- Gaussian Mechanism DP Training")
print(f"  Model    : Logistic Regression")
print(f"  Mechanism: Analytic Gaussian (epsilon, delta)-DP")
print(f"{'='*60}")
print(f"\n  Dataset : {X.shape[0]:,} rows | {n_features} features")
print(f"  Target  : '{target_col}' | {len(np.unique(y))} classes")
print(f"  Features: {feature_cols}\n")

print("  +--[ PRIVACY SCOPE NOTE ]" + "-"*35 + "+")
print(f"  | Target label ('{target_col}') is NOT DP-protected.              |")
print("  | Standard DP-ML design (Abadi et al. 2016 DP-SGD).          |")
print("  | Privacy = what the *model* reveals about training records.  |")
print("  +" + "-"*59 + "+\n")

# ===========================================================================
#  2. TRAIN / TEST SPLIT
# ===========================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=BASE_SEED, stratify=y
)

# ===========================================================================
#  3. NORMALIZATION
# ===========================================================================
X_train_norm, fallback_bounds = normalize_with_domain_bounds(X_train, feature_cols)
X_test_norm, _                = normalize_with_domain_bounds(
    X_test, feature_cols, fallback_bounds=fallback_bounds
)

# ===========================================================================
#  4. EPSILON INPUT
# ===========================================================================
print("  " + "="*58)
print("  GOLDEN RULE:")
print("  e (epsilon) badhao  -->  privacy KAM,  accuracy ZYADA")
print("  e (epsilon) ghatao  -->  privacy ZYADA, accuracy KAM")
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

sigma = analytic_gaussian_sigma(epsilon, DELTA, sensitivity=1.0)
total_epsilon_basic    = N_TRIALS * epsilon
total_epsilon_advanced = math.sqrt(N_TRIALS) * epsilon

print(f"\n  --> Epsilon (e, per run)   = {epsilon}")
print(f"      Delta   (d)             = {DELTA}")
print(f"      Sigma   (s, Analytic GM)= {sigma:.4f}  [Gaussian noise std]")
print(f"      Trials                  = {N_TRIALS} runs")
print(f"\n  [!] COMPOSITION WARNING:")
print(f"      {N_TRIALS} trials on same data -> TOTAL consumed:")
print(f"        Basic composition    : e_total = {total_epsilon_basic:.4f}  (= {N_TRIALS} x {epsilon})")
print(f"        Advanced composition : e_total ~ {total_epsilon_advanced:.4f}  (= sqrt({N_TRIALS}) x {epsilon})")
print(f"  " + "-"*58 + "\n")

# ===========================================================================
#  5. BASELINE LR — No Privacy
# ===========================================================================
print(f"[1] Training Baseline Logistic Regression (No DP)...")
lr_baseline = LogisticRegression(max_iter=1000, random_state=BASE_SEED)
lr_baseline.fit(X_train_norm, y_train)
y_base_pred  = lr_baseline.predict(X_test_norm)
acc_baseline = accuracy_score(y_test, y_base_pred)
f1_baseline  = f1_score(y_test, y_base_pred, average="macro")
rec_baseline = recall_score(y_test, y_base_pred, average="macro")
print(f"    Baseline Accuracy : {acc_baseline * 100:.2f}%")
print(f"    Baseline F1 Score : {f1_baseline:.4f}")
print(f"    Baseline Recall   : {rec_baseline:.4f}\n")

# ===========================================================================
#  6. DP LR — GAUSSIAN MECHANISM, MULTIPLE TRIALS
# ===========================================================================
print(f"[2] Training DP LR with Gaussian Mechanism ({N_TRIALS} trials | e={epsilon})...")
print(f"    Gaussian sigma={sigma:.4f} | sensitivity=1.0 | (e,d)-DP\n")

trial_accs = []
trial_f1s  = []
trial_recs = []
lr_dp = None

for trial in range(N_TRIALS):
    seed = BASE_SEED + trial
    rng  = np.random.RandomState(seed)

    X_train_noisy = add_gaussian_dp_noise(X_train_norm, feature_cols, sigma, rng)

    lr_dp = LogisticRegression(max_iter=1000, random_state=seed)
    lr_dp.fit(X_train_noisy, y_train)
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

y_pred       = lr_dp.predict(X_test_norm)
y_pred_proba = lr_dp.predict_proba(X_test_norm)

class_names = [str(c) for c in le.classes_]
report = classification_report(y_test, y_pred, target_names=class_names)

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
print(f"  Gaussian Sigma (Analytic GM)           : {sigma:.4f}")
print(f"  Normalization                          : Domain-bound clipping (data-independent)")
print(f"  " + "-"*58)
print(f"  PRIVACY BUDGET ACCOUNTING:")
print(f"    Mechanism                : Analytic Gaussian (Balle & Wang, 2018)")
print(f"    Per-run guarantee        : ({epsilon}, {DELTA})-DP")
print(f"    Total consumed (basic)   : ({total_epsilon_basic:.4f}, {DELTA})-DP  <-- {N_TRIALS} runs x e={epsilon}")
print(f"    Total consumed (advanced): (~{total_epsilon_advanced:.4f}, ...)-DP  <-- sqrt({N_TRIALS}) x e={epsilon}")
print(f"  " + "-"*58)
print(f"\n  PER-CLASS REPORT (last trial):\n")
print(report)
print(f"{'='*60}\n")
