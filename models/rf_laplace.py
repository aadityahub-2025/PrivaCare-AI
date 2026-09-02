"""
PrivaCare-AI - models/rf_laplace.py
Model    : Random Forest
Mechanism: Laplace Mechanism (Dwork & Roth, 2014)
Guarantee: Pure epsilon-DP  (NO delta needed -- stronger formal claim)

DP Approach:
  - Features normalized to [0,1] via data-independent DOMAIN_BOUNDS
    L-inf sensitivity = 1.0 per feature (fully data-independent)
  - Laplace noise Lap(0, b) added to TRAINING features only
    b = sensitivity / epsilon = 1.0 / epsilon
  - sklearn RandomForestClassifier trained on noisy features
  - Test features stay CLEAN (standard DP-ML practice)
  - Guarantee: pure epsilon-DP

Compared to rf_gaussian.py:
  - Gaussian: (epsilon, delta)-DP  -- approximate DP, needs delta
  - Laplace:  pure epsilon-DP      -- exact DP, no delta needed
  - Laplace has heavier tails -> may affect accuracy differently

Reference: Dwork & Roth (2014), Algorithmic Foundations of DP, Chap 3
"""

import math
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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
N_TRIALS     = 3      # DP model runs to average (improves reliability)
N_ESTIMATORS = 100
BASE_SEED    = 42     # Trial i gets seed = BASE_SEED + i
STRICT_DOMAIN_BOUNDS = True


# ===========================================================================
#  DATA-INDEPENDENT FEATURE BOUNDS
#  DP RULE: Sensitivity from domain knowledge, NOT from data.
#  After clipping + scaling to [0,1]: L-inf sensitivity = 1.0
# ===========================================================================
DOMAIN_BOUNDS = {
    # -- Vital Signs --------------------------------------------------------
    "heart_rate":               (30,    220),
    "blood_oxygen":             (70.0, 100.0),
    "blood_pressure_systolic":  (60,    250),
    "blood_pressure_diastolic": (40,    150),
    "glucose_level":            (50.0,  500.0),
    "body_temperature":         (35.0,  42.0),
    "respiratory_rate":         (8,     40),
    # -- Wearable / Lifestyle -----------------------------------------------
    "activity_level":           (0.0,   10.0),
    "sleep_quality":            (0.0,   10.0),
    "stress_level":             (0.0,   10.0),
    "hrv_sdnn":                 (5.0,  200.0),
    "steps_count":              (0,    50000),
    "calories_burned":          (0,     5000),
    # -- Demographics -------------------------------------------------------
    "age":                      (0,     120),
    "gender":                   (0,     1),
}

BINARY_FEATURES = {"gender"}
GENDER_MAP = {"male": 1, "m": 1, "female": 0, "f": 0, "woman": 0, "man": 1}


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
                    f"\n  [DP ERROR] Feature '{col}' is NOT in DOMAIN_BOUNDS.\n"
                    f"  ACTION REQUIRED: Add '{col}' to DOMAIN_BOUNDS dict."
                )
            lo, hi = float(X[:, i].min()), float(X[:, i].max())
            computed_fallbacks[col] = (lo, hi)
            print(f"  WARNING: No domain bound for '{col}' -- using train min/max.")
        X_norm[:, i] = np.clip(X_norm[:, i], lo, hi)
        X_norm[:, i] = (X_norm[:, i] - lo) / (hi - lo + 1e-12)
    lower_bounds = [0.0] * X_norm.shape[1]
    upper_bounds = [1.0] * X_norm.shape[1]
    return X_norm, lower_bounds, upper_bounds, computed_fallbacks


# ===========================================================================
#  LAPLACE NOISE INJECTION
#  -------------------------------------------------------------------------
#  Laplace Mechanism: Add Lap(0, b) noise where b = sensitivity / epsilon
#  sensitivity = 1.0 (from [0,1] normalization via DOMAIN_BOUNDS)
#  Guarantee: pure epsilon-DP per feature
#  Under basic composition across N features: N*epsilon total
#
#  Key difference from Gaussian:
#    Gaussian  → (epsilon, delta)-DP  [approximate DP, needs delta]
#    Laplace   → pure epsilon-DP      [exact DP, no delta needed]
# ===========================================================================
def add_laplace_dp_noise(X_norm, feature_names, epsilon, rng):
    """
    Add Laplace noise Lap(0, sensitivity/epsilon) to training features.

    Args:
        X_norm       : normalized training array (values in [0,1])
        feature_names: list of feature column names
        epsilon      : privacy budget per feature (sensitivity=1.0)
        rng          : numpy RandomState for reproducibility

    Returns:
        X_noisy      : noisy training array (clipped back to [0,1])
        b            : Laplace scale parameter used
    """
    b = 1.0 / epsilon  # sensitivity=1.0 / epsilon
    X_noisy = X_norm.copy()
    for i, col in enumerate(feature_names):
        if col in BINARY_FEATURES:
            continue  # Skip binary features (Laplace DP inappropriate for 0/1)
        noise = rng.laplace(loc=0.0, scale=b, size=X_norm.shape[0])
        X_noisy[:, i] = np.clip(X_noisy[:, i] + noise, 0.0, 1.0)
    return X_noisy, b


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

# Explicit gender encoding
if "gender" in df.columns and df["gender"].dtype == object:
    def encode_gender(val):
        v = str(val).strip().lower()
        if v not in GENDER_MAP:
            raise ValueError(
                f"  [DATA ERROR] Unknown gender value: '{val}'.\n"
                f"  Expected: {list(GENDER_MAP.keys())}."
            )
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
print(f"  PrivaCare-AI -- Laplace Mechanism DP Training")
print(f"  Model    : Random Forest")
print(f"  Mechanism: Laplace (pure epsilon-DP, no delta)")
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
#  3. DATA-INDEPENDENT NORMALIZATION
# ===========================================================================
X_train_norm, lower_bounds, upper_bounds, fallback_bounds = normalize_with_domain_bounds(
    X_train, feature_cols
)
X_test_norm, _, _, _ = normalize_with_domain_bounds(
    X_test, feature_cols, fallback_bounds=fallback_bounds
)

# ===========================================================================
#  4. EPSILON INPUT
# ===========================================================================
print("  " + "="*58)
print("  GOLDEN RULE:")
print("  e (epsilon) badhao  -->  privacy KAM,  accuracy ZYADA")
print("  e (epsilon) ghatao  -->  privacy ZYADA, accuracy KAM")
print("  YE TRADEOFF HAI! Choose wisely.")
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
    print("  Invalid input -- defaulting to epsilon = 0.5")
    epsilon = 0.5

b = 1.0 / epsilon  # Laplace scale parameter
total_epsilon_basic    = N_TRIALS * epsilon
total_epsilon_advanced = math.sqrt(N_TRIALS) * epsilon

print(f"\n  --> Epsilon (e, per run)    = {epsilon}")
print(f"      Laplace scale (b=1/e)   = {b:.4f}  << ACTUALLY USED for noise")
print(f"      Mechanism               : Laplace (pure epsilon-DP)")
print(f"      NO delta needed         : Laplace gives exact DP guarantee")
print(f"      Trials                  = {N_TRIALS} runs")
print(f"\n  [!] COMPOSITION WARNING:")
print(f"      {N_TRIALS} trials on same data -> TOTAL consumed:")
print(f"        Basic composition    : e_total = {total_epsilon_basic:.4f}  (= {N_TRIALS} x {epsilon})")
print(f"        Advanced composition : e_total ~ {total_epsilon_advanced:.4f}  (= sqrt({N_TRIALS}) x {epsilon})")
print(f"  " + "-"*58 + "\n")

# ===========================================================================
#  5. BASELINE RF — No Privacy
# ===========================================================================
print(f"[1] Training Baseline Random Forest (No DP)...")
rf_baseline = RandomForestClassifier(
    n_estimators=N_ESTIMATORS, random_state=BASE_SEED, n_jobs=-1
)
rf_baseline.fit(X_train_norm, y_train)
y_base_pred   = rf_baseline.predict(X_test_norm)
acc_baseline  = accuracy_score(y_test, y_base_pred)
f1_baseline   = f1_score(y_test, y_base_pred, average="macro")
rec_baseline  = recall_score(y_test, y_base_pred, average="macro")
print(f"    Baseline Accuracy : {acc_baseline * 100:.2f}%")
print(f"    Baseline F1 Score : {f1_baseline:.4f}")
print(f"    Baseline Recall   : {rec_baseline:.4f}\n")

# ===========================================================================
#  6. DP RF — LAPLACE MECHANISM, MULTIPLE TRIALS
# ===========================================================================
print(f"[2] Training DP RF with Laplace Mechanism ({N_TRIALS} trials | e={epsilon})...")
print(f"    Laplace scale b={b:.4f} | sensitivity=1.0 | pure epsilon-DP\n")

trial_accs = []
trial_f1s  = []
trial_recs = []
rf_dp = None

for trial in range(N_TRIALS):
    seed = BASE_SEED + trial
    rng  = np.random.RandomState(seed)

    X_train_noisy, b_used = add_laplace_dp_noise(
        X_train_norm, feature_cols, epsilon, rng
    )

    rf_dp = RandomForestClassifier(
        n_estimators=N_ESTIMATORS, random_state=seed, n_jobs=-1
    )
    rf_dp.fit(X_train_noisy, y_train)
    y_pred_t = rf_dp.predict(X_test_norm)

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

# Final trial model for report
y_pred       = rf_dp.predict(X_test_norm)
y_pred_proba = rf_dp.predict_proba(X_test_norm)

# ===========================================================================
#  7. CLASSIFICATION REPORT (per-class)
# ===========================================================================
class_names = [str(c) for c in le.classes_]
report = classification_report(y_test, y_pred, target_names=class_names)

# ===========================================================================
#  8. RESULT SUMMARY
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
print(f"  Laplace Scale (b = 1/epsilon)          : {b:.4f}")
print(f"  Normalization                          : Domain-bound clipping (data-independent)")
print(f"  " + "-"*58)
print(f"  PRIVACY BUDGET ACCOUNTING:")
print(f"    Mechanism                : Laplace Mechanism (Dwork & Roth, 2014)")
print(f"    Per-run guarantee        : pure {epsilon}-DP  (NO delta needed)")
print(f"    Total consumed (basic)   : {total_epsilon_basic:.4f}-DP  <-- {N_TRIALS} runs x e={epsilon}")
print(f"    Total consumed (advanced): ~{total_epsilon_advanced:.4f}-DP  <-- sqrt({N_TRIALS}) x e={epsilon}")
print(f"  " + "-"*58)
print(f"\n  PER-CLASS REPORT (last trial):\n")
print(report)
print(f"{'='*60}\n")
