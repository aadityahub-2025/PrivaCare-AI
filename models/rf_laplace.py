"""
PrivaCare-AI - models/rf_laplace.py
Model    : Random Forest
Mechanism: Tree-based DP — Laplace (pure epsilon-DP)
Guarantee: Pure epsilon-DP  (NO delta needed)

DP Approach (Tree-based DP — diffprivlib RandomForestClassifier):
  - Each decision tree is constructed via random splitting criterion over domain bounds
  - The PermuteAndFlip mechanism is applied to determine noisy leaf node labels
  - Guarantee: Pure epsilon-DP (exact, no delta approximation needed)

Why better than naive Laplace Input Perturbation:
  - Input Perturbation on 13 features: Laplace scale b = 13/0.5 = 26 -> destroys signal
  - Tree-DP perturbs splits and leaves directly -> much smaller effective noise
  - diffprivlib handles all DP internals correctly

Dataset  : datasets/dataset_2_rf_laplace.json  (ONLY this file — no other dataset)
Reference: Dwork & Roth (2014) "Algorithmic Foundations of DP"
           Fletcher & Islam (2019) "Decision Tree Classification with Differential Privacy"
           diffprivlib: Holohan et al. (2019), IBM Research
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import diffprivlib.models as dp
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
DATASET_PATH = "datasets/dataset_2_rf_laplace.json"   # ONLY this dataset
N_TRIALS     = 3
N_ESTIMATORS = 20     # DP RF: more trees = more epsilon budget split; 20 is good balance
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
#  diffprivlib RF requires bounds = ([lo,...], [hi,...])
# ===========================================================================
def normalize_features(X):
    X_norm = X.copy().astype(float)
    for i, (lo, hi) in enumerate(DOMAIN_BOUNDS):
        X_norm[:, i] = np.clip(X_norm[:, i], lo, hi)
        X_norm[:, i] = (X_norm[:, i] - lo) / (hi - lo + 1e-12)
    return X_norm

LOWER_BOUNDS = [0.0] * len(FEATURE_COLS)
UPPER_BOUNDS = [1.0] * len(FEATURE_COLS)
BOUNDS = (LOWER_BOUNDS, UPPER_BOUNDS)


# ===========================================================================
#  1. LOAD DATA — ONLY dataset_2_rf_laplace.json
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
print(f"  Model    : Random Forest")
print(f"  Mechanism: Tree-based DP (pure epsilon-DP, no delta)")
print(f"  Dataset  : {DATASET_PATH}")
print(f"{'='*60}")
print(f"\n  Dataset : {n_total:,} rows | {n_features} features")
print(f"  Target  : '{target_col}' | {n_classes} classes")
print(f"  Features: {FEATURE_COLS}\n")

print("  +--[ PRIVACY SCOPE NOTE ]" + "-"*35 + "+")
print(f"  | Target label ('{target_col}') is NOT DP-protected.              |")
print("  | Tree-based DP: Exponential Mech for splits + Laplace for leaves.|")
print("  | Pure epsilon-DP — no delta required.                        |")
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

total_epsilon_basic = N_TRIALS * epsilon

print(f"\n  --> Epsilon (e, per run)    = {epsilon}")
print(f"      n_train                 = {X_train_norm.shape[0]:,}")
print(f"      N_ESTIMATORS            = {N_ESTIMATORS}")
print(f"      Mechanism               : Tree-based DP (diffprivlib)")
print(f"      NO delta needed         : Pure epsilon-DP guarantee")
print(f"      Trials                  = {N_TRIALS} runs")
print(f"\n  [!] COMPOSITION WARNING:")
print(f"      {N_TRIALS} trials on same data --> TOTAL consumed:")
print(f"        Basic composition    : e_total = {total_epsilon_basic:.4f}  (= {N_TRIALS} x {epsilon})")
print(f"        (Advanced composition not applicable for pure epsilon-DP)")
print(f"  " + "-"*58 + "\n")

# ===========================================================================
#  5. BASELINE RF — No Privacy (for comparison)
# ===========================================================================
print(f"[1] Training Baseline Random Forest (No DP)...")
rf_baseline = RandomForestClassifier(
    n_estimators=N_ESTIMATORS, random_state=BASE_SEED, n_jobs=-1
)
rf_baseline.fit(X_train_norm, y_train)
y_base_pred  = rf_baseline.predict(X_test_norm)
acc_baseline = accuracy_score(y_test, y_base_pred)
f1_baseline  = f1_score(y_test, y_base_pred, average="macro")
rec_baseline = recall_score(y_test, y_base_pred, average="macro")
print(f"    Baseline Accuracy : {acc_baseline * 100:.2f}%")
print(f"    Baseline F1 Score : {f1_baseline:.4f}")
print(f"    Baseline Recall   : {rec_baseline:.4f}\n")

# ===========================================================================
#  6. DP RF — TREE-BASED DP (Laplace, pure epsilon-DP)
#
#  Algorithm (diffprivlib RandomForestClassifier):
#    1. Build each tree using Exponential Mechanism for split selection
#    2. Add Laplace noise to leaf class counts
#    3. Aggregate tree predictions for final classification
#    4. Guarantee: pure epsilon-DP (Dwork & Roth, 2014)
# ===========================================================================
print(f"[2] Training DP RF with Tree-based Laplace DP ({N_TRIALS} trials | e={epsilon})...")
print(f"    Using diffprivlib RandomForestClassifier | pure epsilon-DP\n")

trial_accs, trial_f1s, trial_recs = [], [], []
rf_dp = None
classes = np.unique(y_train)

for trial in range(N_TRIALS):
    seed = BASE_SEED + trial

    rf_dp = dp.RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=10,
        epsilon=epsilon,
        bounds=BOUNDS,
        classes=classes,
        random_state=seed
    )
    rf_dp.fit(X_train_norm, y_train)
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

y_pred = rf_dp.predict(X_test_norm)
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
print(f"  Mechanism                              : Tree-based DP (diffprivlib)")
print(f"  " + "-"*58)
print(f"  PRIVACY BUDGET ACCOUNTING:")
print(f"    Mechanism                : Tree-based DP (Exponential + Laplace)")
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
model_path = "saved_models/rf_laplace.pkl"
joblib.dump(rf_dp, model_path)
print(f"  [+] Model successfully saved to: {model_path}")
print(f"{'='*60}\n")
