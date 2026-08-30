"""
PrivaCare-AI - Differential Privacy Training
Comparison: Two DP Approaches

APPROACH 1 - Analytic Gaussian Mechanism (Balle & Wang, 2018):
  - Features normalized to [0,1] via data-independent DOMAIN_BOUNDS
  - Calibrated Gaussian noise injected into training features
  - sigma = analytic_gaussian_sigma(epsilon, delta), sensitivity=1.0
  - Guarantee: (epsilon, delta)-DP per feature
  - Under basic composition: (N*epsilon, N*delta)-DP overall
  - sklearn RF trained on noisy features

APPROACH 2 - diffprivlib DP Random Forest (IBM):
  - Uses Exponential Mechanism for tree splitting
  - Uses Laplace Mechanism for leaf node vote counts
  - Guarantee: pure epsilon-DP (no delta needed)
  - Designed specifically for RF trees -- more efficient than feature noise

Both approaches use data-independent DOMAIN_BOUNDS for sensitivity=1.0.
"""

import math
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import diffprivlib.models as dp
import warnings
warnings.filterwarnings("ignore")

# ===========================================================================
#  CONFIGURATION
# ===========================================================================
N_TRIALS     = 3
DELTA        = 1e-5
N_ESTIMATORS = 100
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
#  ANALYTIC GAUSSIAN MECHANISM  (Balle & Wang, NeurIPS 2018)
# ===========================================================================
def analytic_gaussian_sigma(epsilon, delta, sensitivity=1.0):
    """
    Minimum sigma for (epsilon, delta)-DP via Analytic Gaussian Mechanism.
    Valid for all epsilon > 0. Binary search on normal CDF.
    """
    from scipy.special import erfc

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
#  NORMALIZATION
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
                    f"[DP ERROR] Feature '{col}' not in DOMAIN_BOUNDS. "
                    f"Add it or set STRICT_DOMAIN_BOUNDS=False."
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
    """
    Add Gaussian noise N(0, sigma^2) to each continuous feature.
    Binary features (BINARY_FEATURES) are skipped.
    Output is clipped to [0,1] after noise.
    """
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

print(f"\n{'='*62}")
print(f"  PrivaCare-AI -- DP Comparison: Gaussian vs diffprivlib RF")
print(f"{'='*62}")
print(f"\n  Dataset : {X.shape[0]:,} rows | {n_features} features | {len(np.unique(y))} classes\n")

# Label privacy note
print("  +--[ PRIVACY SCOPE NOTE ]" + "-"*35 + "+")
print(f"  | Labels are NOT DP-protected (standard DP-ML design).        |")
print("  | Privacy = what the trained model reveals about training data. |")
print("  +" + "-"*62 + "+\n")

# ===========================================================================
#  2. TRAIN / TEST SPLIT + NORMALIZATION
# ===========================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=BASE_SEED, stratify=y
)
X_train_norm, fallback_bounds = normalize_with_domain_bounds(X_train, feature_cols)
X_test_norm,  _               = normalize_with_domain_bounds(
    X_test, feature_cols, fallback_bounds=fallback_bounds
)
bounds_dp = ([0.0] * n_features, [1.0] * n_features)

# ===========================================================================
#  3. EPSILON INPUT
# ===========================================================================
print("  " + "="*60)
print("  GOLDEN RULE:")
print("  e (epsilon) badhao  -->  privacy KAM,  accuracy ZYADA")
print("  e (epsilon) ghatao  -->  privacy ZYADA, accuracy KAM")
print("  " + "="*60)

try:
    user_input = input("\n  Enter Epsilon value (e.g. 0.5, 1.0, 5.0) [Default 1.0]: ").strip()
    epsilon = float(user_input) if user_input else 1.0
except ValueError:
    epsilon = 1.0

sigma = analytic_gaussian_sigma(epsilon, DELTA, sensitivity=1.0)
total_epsilon_basic    = N_TRIALS * epsilon
total_epsilon_advanced = math.sqrt(N_TRIALS) * epsilon

print(f"\n  --> Epsilon                     = {epsilon}")
print(f"      Sigma (Analytic GM, e={epsilon}) = {sigma:.4f}")
print(f"      Delta                         = {DELTA}")

# ===========================================================================
#  4. BASELINE RF
# ===========================================================================
print(f"\n{'='*62}")
print(f"  [0] BASELINE -- No Privacy, No Noise")
print(f"{'='*62}")
rf_baseline = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=BASE_SEED, n_jobs=-1)
rf_baseline.fit(X_train_norm, y_train)
acc_baseline = accuracy_score(y_test, rf_baseline.predict(X_test_norm))
print(f"      Accuracy : {acc_baseline * 100:.2f}%")

# ===========================================================================
#  5. APPROACH 1 — ANALYTIC GAUSSIAN MECHANISM + sklearn RF
# ===========================================================================
print(f"\n{'='*62}")
print(f"  [1] APPROACH 1 -- Analytic Gaussian Mechanism (Balle & Wang, 2018)")
print(f"      Guarantee: ({epsilon}, {DELTA})-DP per feature")
print(f"      sigma={sigma:.4f}, sensitivity=1.0, {N_TRIALS} trials")
print(f"{'='*62}")

gauss_accs = []
for trial in range(N_TRIALS):
    seed = BASE_SEED + trial
    rng  = np.random.RandomState(seed)
    X_train_noisy = add_gaussian_dp_noise(X_train_norm, feature_cols, sigma, rng)
    rf_g = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=seed, n_jobs=-1)
    rf_g.fit(X_train_noisy, y_train)
    acc_t = accuracy_score(y_test, rf_g.predict(X_test_norm))
    print(f"      Trial {trial+1}/{N_TRIALS} (seed={seed}): {acc_t*100:.2f}%")
    gauss_accs.append(acc_t)

acc_gauss = float(np.mean(gauss_accs))
std_gauss = float(np.std(gauss_accs))
y_pred_gauss = rf_g.predict(X_test_norm)
print(f"\n      Mean Accuracy : {acc_gauss*100:.2f}% +/- {std_gauss*100:.2f}%")
print(f"      Privacy       : ({epsilon}, {DELTA})-DP per feature")
print(f"      Mechanism     : Analytic Gaussian (Balle & Wang, 2018)")

# ===========================================================================
#  6. APPROACH 2 — diffprivlib DP Random Forest
# ===========================================================================
print(f"\n{'='*62}")
print(f"  [2] APPROACH 2 -- diffprivlib DP Random Forest (IBM)")
print(f"      Guarantee: {epsilon}-DP (pure DP, Exponential + Laplace)")
print(f"      {N_TRIALS} trials")
print(f"{'='*62}")

dp_accs = []
rf_dp = None
for trial in range(N_TRIALS):
    seed = BASE_SEED + trial
    rf_dp = dp.RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        epsilon=epsilon,
        bounds=bounds_dp,
        random_state=seed,
    )
    rf_dp.fit(X_train_norm, y_train)
    acc_t = accuracy_score(y_test, rf_dp.predict(X_test_norm))
    print(f"      Trial {trial+1}/{N_TRIALS} (seed={seed}): {acc_t*100:.2f}%")
    dp_accs.append(acc_t)

acc_dp  = float(np.mean(dp_accs))
std_dp  = float(np.std(dp_accs))
y_pred_dp       = rf_dp.predict(X_test_norm)
y_pred_dp_proba = rf_dp.predict_proba(X_test_norm)
print(f"\n      Mean Accuracy : {acc_dp*100:.2f}% +/- {std_dp*100:.2f}%")
print(f"      Privacy       : {epsilon}-DP (pure DP)")
print(f"      Mechanism     : Exponential (splits) + Laplace (leaf counts)")

# ===========================================================================
#  7. COMPARISON SUMMARY
# ===========================================================================
print(f"\n{'='*62}")
print(f"  FINAL COMPARISON (epsilon={epsilon})")
print(f"{'='*62}")
print(f"  {'Approach':<35} {'Accuracy':>10} {'Privacy Guarantee'}")
print(f"  {'-'*60}")
print(f"  {'Baseline (No DP)':<35} {acc_baseline*100:>9.2f}%  None")
print(f"  {'[1] Gaussian Mechanism (Balle & Wang)':<35} {acc_gauss*100:>9.2f}%  ({epsilon}, {DELTA})-DP")
print(f"  {'[2] diffprivlib RF (Exponential+Laplace)':<35} {acc_dp*100:>9.2f}%  pure {epsilon}-DP")
print(f"  {'-'*60}")
print(f"\n  COMPOSITION (across {N_TRIALS} trials on same dataset):")
print(f"    Basic    : {N_TRIALS} x {epsilon} = {total_epsilon_basic:.2f}-DP total")
print(f"    Advanced : sqrt({N_TRIALS}) x {epsilon} ~ {total_epsilon_advanced:.4f}-DP total")
print(f"{'='*62}\n")
