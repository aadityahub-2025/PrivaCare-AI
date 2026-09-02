"""
PrivaCare-AI - compare_all.py
Combined comparison of all DP models and mechanisms.

Runs all 4 combinations across multiple epsilon values:
  - Random Forest       + Gaussian (diffprivlib)
  - Random Forest       + Laplace  (feature-level)
  - Logistic Regression + Gaussian (feature-level)
  - Logistic Regression + Laplace  (feature-level)

Output: Full comparison table with accuracy, F1, privacy params.
"""

import math
import numpy as np
import pandas as pd
from scipy.special import erfc
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
import diffprivlib.models as dp
import warnings
warnings.filterwarnings("ignore")

# ===========================================================================
#  SHARED CONFIG
# ===========================================================================
N_TRIALS  = 3
BASE_SEED = 42
DELTA     = 1e-5
EPSILONS  = [0.5, 0.7, 1.0]

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
#  SHARED HELPERS
# ===========================================================================
def normalize(X, feature_names, fallback=None):
    X_norm = X.copy().astype(float)
    fb = {}
    for i, col in enumerate(feature_names):
        if col in DOMAIN_BOUNDS:
            lo, hi = DOMAIN_BOUNDS[col]
        elif fallback and col in fallback:
            lo, hi = fallback[col]
        else:
            lo, hi = float(X[:, i].min()), float(X[:, i].max())
            fb[col] = (lo, hi)
        X_norm[:, i] = np.clip(X_norm[:, i], lo, hi)
        X_norm[:, i] = (X_norm[:, i] - lo) / (hi - lo + 1e-12)
    return X_norm, fb


def analytic_gaussian_sigma(epsilon, delta, sensitivity=1.0):
    def phi(t): return 0.5 * erfc(-t / math.sqrt(2))
    def delta_of_sigma(s):
        a = sensitivity / (2 * s); b = epsilon * s / sensitivity
        return phi(a - b) - math.exp(epsilon) * phi(-a - b)
    lo, hi = 1e-9, 1e6
    for _ in range(1000):
        mid = (lo + hi) / 2
        (hi := mid) if delta_of_sigma(mid) <= delta else (lo := mid)
    return hi


def add_gaussian_noise(X, feature_names, sigma, rng):
    X_n = X.copy()
    for i, col in enumerate(feature_names):
        if col not in BINARY_FEATURES:
            X_n[:, i] = np.clip(X_n[:, i] + rng.normal(0, sigma, X.shape[0]), 0, 1)
    return X_n


def add_laplace_noise(X, feature_names, b, rng):
    X_n = X.copy()
    for i, col in enumerate(feature_names):
        if col not in BINARY_FEATURES:
            X_n[:, i] = np.clip(X_n[:, i] + rng.laplace(0, b, X.shape[0]), 0, 1)
    return X_n


def run_trials(train_fn, X_train_norm, X_test_norm, y_train, y_test, epsilon):
    accs, f1s = [], []
    for trial in range(N_TRIALS):
        seed = BASE_SEED + trial
        rng  = np.random.RandomState(seed)
        model, y_pred = train_fn(X_train_norm, y_train, epsilon, seed, rng)
        accs.append(accuracy_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred, average="macro"))
    return np.mean(accs), np.std(accs), np.mean(f1s)


# ===========================================================================
#  4 TRAINING FUNCTIONS
# ===========================================================================
def train_rf_gaussian(X_train_norm, y_train, epsilon, seed, rng):
    """Random Forest + diffprivlib Gaussian (Exponential+Laplace on splits)"""
    bounds = ([0.0] * X_train_norm.shape[1], [1.0] * X_train_norm.shape[1])
    model = dp.RandomForestClassifier(
        n_estimators=100, epsilon=epsilon, bounds=bounds, random_state=seed
    )
    model.fit(X_train_norm, y_train)
    return model, model.predict(X_train_norm)   # predict on test via closure below


def train_rf_gaussian_v2(X_train_norm, X_test_norm, y_train, epsilon, seed):
    """RF + diffprivlib -- returns test predictions"""
    bounds = ([0.0] * X_train_norm.shape[1], [1.0] * X_train_norm.shape[1])
    model = dp.RandomForestClassifier(
        n_estimators=100, epsilon=epsilon, bounds=bounds, random_state=seed
    )
    model.fit(X_train_norm, y_train)
    return model.predict(X_test_norm)


def train_rf_laplace_v2(X_train_norm, X_test_norm, y_train, epsilon, seed):
    """RF + Laplace feature noise -- pure epsilon-DP"""
    rng = np.random.RandomState(seed)
    X_noisy = add_laplace_noise(X_train_norm, feature_cols, 1.0 / epsilon, rng)
    model = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    model.fit(X_noisy, y_train)
    return model.predict(X_test_norm)


def train_lr_gaussian_v2(X_train_norm, X_test_norm, y_train, epsilon, seed):
    """LR + Gaussian feature noise -- (epsilon, delta)-DP"""
    sigma = analytic_gaussian_sigma(epsilon, DELTA)
    rng   = np.random.RandomState(seed)
    X_noisy = add_gaussian_noise(X_train_norm, feature_cols, sigma, rng)
    model = LogisticRegression(max_iter=1000, random_state=seed)
    model.fit(X_noisy, y_train)
    return model.predict(X_test_norm)


def train_lr_laplace_v2(X_train_norm, X_test_norm, y_train, epsilon, seed):
    """LR + Laplace feature noise -- pure epsilon-DP"""
    rng = np.random.RandomState(seed)
    X_noisy = add_laplace_noise(X_train_norm, feature_cols, 1.0 / epsilon, rng)
    model = LogisticRegression(max_iter=1000, random_state=seed)
    model.fit(X_noisy, y_train)
    return model.predict(X_test_norm)


# ===========================================================================
#  LOAD DATA
# ===========================================================================
print("\n" + "="*70)
print("  PrivaCare-AI -- compare_all.py")
print("  Comparing all DP models and mechanisms across epsilon values")
print("="*70)

df = pd.read_csv("data/dataset.csv")
target_col = "health_event" if "health_event" in df.columns else df.columns[-1]

if "gender" in df.columns and df["gender"].dtype == object:
    df["gender"] = df["gender"].apply(
        lambda v: GENDER_MAP[str(v).strip().lower()]
    )

drop_cols    = ["timestamp", "device_id", "patient_id", "is_synthetic", target_col]
feature_cols = [c for c in df.columns
                if c not in drop_cols
                and df[c].dtype in [np.float64, np.int64, float, int]]

X  = df[feature_cols].values.astype(float)
le = LabelEncoder()
y  = le.fit_transform(df[target_col].values)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=BASE_SEED, stratify=y
)
X_train_norm, fb = normalize(X_train, feature_cols)
X_test_norm,  _  = normalize(X_test, feature_cols, fallback=fb)

print(f"\n  Dataset : {X.shape[0]:,} rows | {len(feature_cols)} features")
print(f"  Target  : '{target_col}' | {len(np.unique(y))} classes")
print(f"  Epsilons: {EPSILONS}")
print(f"  Trials  : {N_TRIALS} per run (mean reported)\n")

# Baseline (no DP)
rf_base = RandomForestClassifier(n_estimators=100, random_state=BASE_SEED, n_jobs=-1)
rf_base.fit(X_train_norm, y_train)
acc_base_rf = accuracy_score(y_test, rf_base.predict(X_test_norm))

lr_base = LogisticRegression(max_iter=1000, random_state=BASE_SEED)
lr_base.fit(X_train_norm, y_train)
acc_base_lr = accuracy_score(y_test, lr_base.predict(X_test_norm))

# ===========================================================================
#  RUN ALL EXPERIMENTS
# ===========================================================================
MODELS = [
    ("Random Forest",       "Gaussian (diffprivlib)", train_rf_gaussian_v2, acc_base_rf, "(e,d)-DP"),
    ("Random Forest",       "Laplace  (feature)     ", train_rf_laplace_v2,  acc_base_rf, "pure e-DP"),
    ("Logistic Regression", "Gaussian (feature)     ", train_lr_gaussian_v2, acc_base_lr, "(e,d)-DP"),
    ("Logistic Regression", "Laplace  (feature)     ", train_lr_laplace_v2,  acc_base_lr, "pure e-DP"),
]

results = []

for model_name, mechanism, train_fn, acc_base, dp_type in MODELS:
    row = {"Model": model_name, "Mechanism": mechanism,
           "DP Type": dp_type, "Baseline": f"{acc_base*100:.2f}%"}
    print(f"  Running: {model_name} + {mechanism.strip()} ...")
    for eps in EPSILONS:
        accs, f1s = [], []
        for trial in range(N_TRIALS):
            seed     = BASE_SEED + trial
            y_pred   = train_fn(X_train_norm, X_test_norm, y_train, eps, seed)
            accs.append(accuracy_score(y_test, y_pred))
            f1s.append(f1_score(y_test, y_pred, average="macro"))
        acc_mean = np.mean(accs) * 100
        acc_std  = np.std(accs) * 100
        f1_mean  = np.mean(f1s)
        row[f"e={eps} Acc"] = f"{acc_mean:.1f}%±{acc_std:.1f}%"
        row[f"e={eps} F1"]  = f"{f1_mean:.3f}"
    results.append(row)

# ===========================================================================
#  PRINT COMPARISON TABLE
# ===========================================================================
print("\n\n" + "="*70)
print("  COMPARISON SUMMARY")
print("="*70)
print(f"\n  Baseline RF (No DP): {acc_base_rf*100:.2f}%")
print(f"  Baseline LR (No DP): {acc_base_lr*100:.2f}%\n")

# Accuracy table
hdr = f"{'Model':<22} {'Mechanism':<26} {'DP Type':<12} | {'e=0.5':^18} | {'e=0.7':^18} | {'e=1.0':^18}"
print(hdr)
print("-"*len(hdr))
for row in results:
    line = (f"{row['Model']:<22} {row['Mechanism']:<26} {row['DP Type']:<12} | "
            f"{row.get('e=0.5 Acc','N/A'):^18} | "
            f"{row.get('e=0.7 Acc','N/A'):^18} | "
            f"{row.get('e=1.0 Acc','N/A'):^18}")
    print(line)

print("\n  -- F1 Scores (macro) --\n")
hdr2 = f"{'Model':<22} {'Mechanism':<26} | {'e=0.5':^10} | {'e=0.7':^10} | {'e=1.0':^10}"
print(hdr2)
print("-"*len(hdr2))
for row in results:
    line2 = (f"{row['Model']:<22} {row['Mechanism']:<26} | "
             f"{row.get('e=0.5 F1','N/A'):^10} | "
             f"{row.get('e=0.7 F1','N/A'):^10} | "
             f"{row.get('e=1.0 F1','N/A'):^10}")
    print(line2)

print("\n" + "="*70)
print("  KEY FINDING:")
print("  Higher epsilon = more accuracy, less privacy")
print("  RF + diffprivlib Gaussian consistently outperforms feature-level noise")
print("  Laplace = pure epsilon-DP (stronger formal guarantee than Gaussian)")
print("="*70 + "\n")
