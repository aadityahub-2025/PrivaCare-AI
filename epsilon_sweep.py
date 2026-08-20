"""
Epsilon Sweep - PrivaCare-AI
Tests DP Random Forest across multiple epsilon values.
"""

import numpy as np
import pandas as pd
import warnings
import json
import os
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score

warnings.filterwarnings("ignore")
np.random.seed(42)

DATA_PATH = "data/dataset.csv"
DELTA     = 1e-5
EPSILONS  = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, None]

RF_PARAMS = dict(
    n_estimators=200, max_depth=15, min_samples_split=5,
    min_samples_leaf=2, max_features="sqrt", n_jobs=-1,
    random_state=42, class_weight="balanced", oob_score=True,
)

def gaussian_sigma(epsilon, delta, sensitivity=1.0):
    return sensitivity * np.sqrt(2.0 * np.log(1.25 / delta)) / epsilon

def add_dp_noise(X, sigma):
    return np.clip(X + np.random.normal(0, sigma, X.shape), 0.0, 1.0)

def privacy_level(epsilon):
    if epsilon is None:    return "No Privacy (baseline)"
    if epsilon <= 0.5:     return "Very High Privacy"
    if epsilon <= 2.0:     return "High Privacy"
    if epsilon <= 7.0:     return "Moderate Privacy"
    return                        "Low Privacy"

def load_data():
    df = pd.read_csv(DATA_PATH)
    df.dropna(inplace=True)
    always_drop = ["timestamp","device_id","patient_id","is_synthetic"]
    target_col  = "health_event" if "health_event" in df.columns else df.columns[-1]
    drop_cols   = always_drop + [target_col]
    feature_cols = [c for c in df.columns if c not in drop_cols
                    and df[c].dtype in [np.float64, np.int64, float, int]]
    X  = df[feature_cols].values.astype(float)
    le = LabelEncoder()
    y  = le.fit_transform(df[target_col].values)
    return X, y, feature_cols, le

def run_experiment(X_train_norm, X_test_norm, y_train, y_test, epsilon):
    sigma      = gaussian_sigma(epsilon, DELTA) if epsilon else 0.0
    X_train_dp = add_dp_noise(X_train_norm, sigma) if epsilon else X_train_norm.copy()
    rf = RandomForestClassifier(**RF_PARAMS)
    rf.fit(X_train_dp, y_train)
    y_pred       = rf.predict(X_test_norm)
    y_pred_proba = rf.predict_proba(X_test_norm)
    test_acc = accuracy_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, y_pred_proba, multi_class="ovr", average="macro")
    except:
        auc = None
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_accs = []
    for tr_idx, val_idx in skf.split(X_train_norm, y_train):
        fold_X = add_dp_noise(X_train_norm[tr_idx], sigma) if epsilon else X_train_norm[tr_idx]
        fold_rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
        fold_rf.fit(fold_X, y_train[tr_idx])
        cv_accs.append(accuracy_score(y_train[val_idx], fold_rf.predict(X_train_norm[val_idx])))
    return {
        "epsilon":  epsilon if epsilon else "No DP",
        "sigma":    round(sigma, 4),
        "test_acc": round(test_acc * 100, 2),
        "cv_mean":  round(np.mean(cv_accs) * 100, 2),
        "cv_std":   round(np.std(cv_accs)  * 100, 2),
        "auc_roc":  round(auc, 4) if auc else None,
        "privacy":  privacy_level(epsilon),
    }

def main():
    print("=" * 70)
    print("  PrivaCare-AI: Epsilon Sweep --- Privacy vs Accuracy Tradeoff")
    print(f"  Run Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print("\n Loading data...")
    X, y, feature_cols, le = load_data()
    print(f"   Shape: {X.shape} | Classes: {len(le.classes_)} | Features: {len(feature_cols)}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    scaler = MinMaxScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm  = scaler.transform(X_test)
    print(f"   Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"\n Running {len(EPSILONS)} experiments...\n")
    results = []
    for eps in EPSILONS:
        label = f"eps={eps}" if eps else "No DP (baseline)"
        print(f"   Testing {label:22s} ", end="", flush=True)
        t = datetime.now()
        r = run_experiment(X_train_norm, X_test_norm, y_train, y_test, eps)
        elapsed = (datetime.now() - t).total_seconds()
        results.append(r)
        print(f"Done  Test Acc: {r['test_acc']:5.2f}%  AUC: {r['auc_roc']}  ({elapsed:.1f}s)")
    print("\n" + "=" * 70)
    print("  RESULTS --- Privacy vs Accuracy Comparison")
    print("=" * 70)
    print(f"\n{'Epsilon':>12} {'Sigma':>8} {'Test Acc':>10} {'CV Acc':>10} {'CV Std':>8} {'AUC':>7}  Privacy Level")
    print("-" * 75)
    for r in results:
        print(
            f"{str(r['epsilon']):>12} {r['sigma']:>8.4f} "
            f"{r['test_acc']:>9.2f}% {r['cv_mean']:>9.2f}% "
            f"{r['cv_std']:>7.2f}% {str(r['auc_roc']):>7}  {r['privacy']}"
        )
    print("-" * 75)
    dp_only  = [r for r in results if r['epsilon'] != "No DP"]
    best     = max(dp_only, key=lambda r: r['test_acc'])
    baseline = next(r for r in results if r['epsilon'] == "No DP")
    print(f"\n  Best DP epsilon : eps={best['epsilon']}  ->  {best['test_acc']}% accuracy")
    print(f"  No-DP baseline  :          ->  {baseline['test_acc']}% accuracy")
    print(f"  Privacy cost    :          ->  -{round(baseline['test_acc'] - best['test_acc'], 2)}% accuracy drop for DP")
    os.makedirs("results", exist_ok=True)
    with open("results/epsilon_sweep.json", "w") as f:
        json.dump({"run_time": datetime.now().isoformat(), "delta": DELTA, "results": results}, f, indent=2)
    print(f"\n  Results saved -> results/epsilon_sweep.json")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
