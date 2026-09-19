"""
PrivaCare-AI - scripts/sweep_rf_epsilon.py
Runs a sweep over epsilon values for Random Forest (Tree-based DP via diffprivlib)
across 30 trials to empirically evaluate and verify the ultra-strict privacy tradeoff curve.

Dataset: datasets/dataset_2_rf_laplace.json
Trials : 30 runs per epsilon (seeds = 42 + trial)
"""

import numpy as np
import pandas as pd
import diffprivlib.models as dp
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import time

FEATURE_COLS = [
    "glucose_level",
    "stress_level",
    "heart_rate",
    "blood_pressure_systolic"
]
DOMAIN_BOUNDS = {
    "glucose_level":           (30.0, 300.0),
    "stress_level":            (0.0,  1.0),
    "heart_rate":              (30.0, 220.0),
    "blood_pressure_systolic": (60.0, 250.0),
}
BASE_SEED = 42
N_TRIALS  = 30

def normalize(df, cols):
    X = np.zeros((len(df), len(cols)), dtype=float)
    for i, col in enumerate(cols):
        lo, hi = DOMAIN_BOUNDS[col]
        X[:, i] = np.clip((df[col].values - lo) / (hi - lo), 0.0, 1.0)
    return X

def main():
    print("=" * 70)
    print("  PrivaCare-AI -- Random Forest Epsilon Sweep (N=30 Trials)")
    print("=" * 70)

    df = pd.read_json("datasets/dataset_2_rf_laplace.json")
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=BASE_SEED, stratify=df["health_event"]
    )

    X_train = normalize(train_df, FEATURE_COLS)
    y_train = train_df["health_event"].values.astype(int)
    X_test  = normalize(test_df, FEATURE_COLS)
    y_test  = test_df["health_event"].values.astype(int)

    # Non-private baseline
    base_clf = RandomForestClassifier(n_estimators=20, random_state=BASE_SEED, n_jobs=-1)
    base_clf.fit(X_train, y_train)
    base_acc = accuracy_score(y_test, base_clf.predict(X_test))
    print(f"Non-private Baseline Accuracy: {base_acc*100:.2f}%\n")

    bounds = ([0.0] * len(FEATURE_COLS), [1.0] * len(FEATURE_COLS))
    classes = np.unique(y_train)

    epsilons = [1.0, 0.7, 0.5, 0.1, 0.05, 0.01]
    results = []

    print(f"{'Epsilon':<10} {'Mean Acc':<16} {'Median':<10} {'Min':<10} {'Max':<10} {'Macro F1':<10}")
    print("-" * 70)

    for eps in epsilons:
        t0 = time.time()
        accs, f1s = [], []
        for trial in range(N_TRIALS):
            seed = BASE_SEED + trial
            model = dp.RandomForestClassifier(
                n_estimators=20, max_depth=10, epsilon=eps,
                bounds=bounds, classes=classes, random_state=seed
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            accs.append(accuracy_score(y_test, preds))
            f1s.append(f1_score(y_test, preds, average="macro"))

        mean_acc = np.mean(accs)
        std_acc  = np.std(accs)
        med_acc  = np.median(accs)
        min_acc  = np.min(accs)
        max_acc  = np.max(accs)
        mean_f1  = np.mean(f1s)

        results.append({
            "epsilon": eps,
            "mean": mean_acc,
            "std": std_acc,
            "median": med_acc,
            "min": min_acc,
            "max": max_acc,
            "f1": mean_f1
        })

        print(f"{eps:<10.2f} {mean_acc*100:.2f}% +/- {std_acc*100:.2f}%  {med_acc*100:.2f}%     {min_acc*100:.2f}%     {max_acc*100:.2f}%     {mean_f1:.4f}  ({time.time()-t0:.1f}s)")

    print("=" * 70)
    return results

if __name__ == "__main__":
    main()
