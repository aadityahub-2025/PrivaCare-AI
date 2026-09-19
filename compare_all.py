"""
PrivaCare-AI - compare_all.py
Combined benchmark and comparison across all 4 DP models and mechanisms.

Each model evaluates on its dedicated independent synthetic benchmark replicate:
  - Model 1: Gaussian Naive Bayes (diffprivlib GaussianNB) -> datasets/dataset_1_rf_gaussian.json
  - Model 2: Random Forest (diffprivlib RandomForest)      -> datasets/dataset_2_rf_laplace.json
  - Model 3: Logistic Regression (Output Gaussian DP)      -> datasets/dataset_3_lr_gaussian.json
  - Model 4: Logistic Regression (Objective Laplace DP)    -> datasets/dataset_4_lr_laplace.json

Evaluates accuracy, F1 score, and privacy-utility tradeoff across epsilons [0.5, 0.7, 1.0].
N_TRIALS = 30 for statistically robust means and standard deviations.
"""

import math
import numpy as np
import pandas as pd
from scipy.special import erfc
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
import diffprivlib.models as dp
import os
import warnings
warnings.filterwarnings("once")

# ===========================================================================
#  CONFIG
# ===========================================================================
N_TRIALS  = 30
BASE_SEED = 42
DELTA     = 1e-5
EPSILONS  = [0.5, 0.7, 1.0]

FEATURE_COLS = [
    "glucose_level",
    "stress_level",
    "heart_rate",
    "blood_pressure_systolic"
]

DOMAIN_BOUNDS = {
    "glucose_level":            (30.0, 300.0),
    "stress_level":             (0.0,  1.0),
    "heart_rate":               (30,   220),
    "blood_pressure_systolic":  (60,   250),
}

# ===========================================================================
#  HELPERS
# ===========================================================================
def normalize(X, feature_names):
    X_norm = X.copy().astype(float)
    for i, col in enumerate(feature_names):
        lo, hi = DOMAIN_BOUNDS[col]
        X_norm[:, i] = np.clip(X_norm[:, i], lo, hi)
        X_norm[:, i] = (X_norm[:, i] - lo) / (hi - lo + 1e-12)
    return X_norm

def analytic_gaussian_sigma(epsilon, delta, sensitivity):
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

def load_and_split(filepath):
    df = pd.read_json(filepath)
    X  = df[FEATURE_COLS].values.astype(float)
    le = LabelEncoder()
    y  = le.fit_transform(df["health_event"].values)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=BASE_SEED, stratify=y
    )
    return X_tr, X_te, y_tr, y_te

# ===========================================================================
#  MODEL 1: Gaussian Naive Bayes (Sufficient Statistics Perturbation)
# ===========================================================================
def eval_nb_gaussian():
    X_tr, X_te, y_tr, y_te = load_and_split("datasets/dataset_1_rf_gaussian.json")
    X_tr_norm = normalize(X_tr, FEATURE_COLS)
    X_te_norm = normalize(X_te, FEATURE_COLS)

    base_clf = GaussianNB()
    base_clf.fit(X_tr_norm, y_tr)
    base_acc = accuracy_score(y_te, base_clf.predict(X_te_norm))

    bounds = ([0.0] * len(FEATURE_COLS), [1.0] * len(FEATURE_COLS))

    def train_fn(eps, seed):
        model = dp.GaussianNB(epsilon=eps, bounds=bounds)
        if hasattr(model, 'random_state'):
            model.random_state = seed
        model.fit(X_tr_norm, y_tr)
        return model.predict(X_te_norm)

    return "Gaussian Naive Bayes", "Sufficient Stats DP", "pure e-DP", base_acc, y_te, train_fn

# ===========================================================================
#  MODEL 2: Random Forest (Tree-based DP)
# ===========================================================================
def eval_rf_laplace():
    X_tr, X_te, y_tr, y_te = load_and_split("datasets/dataset_2_rf_laplace.json")
    X_tr_norm = normalize(X_tr, FEATURE_COLS)
    X_te_norm = normalize(X_te, FEATURE_COLS)

    base_clf = RandomForestClassifier(n_estimators=20, random_state=BASE_SEED, n_jobs=-1)
    base_clf.fit(X_tr_norm, y_tr)
    base_acc = accuracy_score(y_te, base_clf.predict(X_te_norm))

    bounds = ([0.0] * len(FEATURE_COLS), [1.0] * len(FEATURE_COLS))
    classes = np.unique(y_tr)

    def train_fn(eps, seed):
        model = dp.RandomForestClassifier(
            n_estimators=20, max_depth=10, epsilon=eps,
            bounds=bounds, classes=classes, random_state=seed
        )
        model.fit(X_tr_norm, y_tr)
        return model.predict(X_te_norm)

    return "Random Forest", "Tree-based DP", "pure e-DP", base_acc, y_te, train_fn

# ===========================================================================
#  MODEL 3: Logistic Regression (Analytic Gaussian Output Perturbation)
# ===========================================================================
def eval_lr_gaussian():
    X_tr, X_te, y_tr, y_te = load_and_split("datasets/dataset_3_lr_gaussian.json")
    X_tr_norm = normalize(X_tr, FEATURE_COLS)
    X_te_norm = normalize(X_te, FEATURE_COLS)

    d = len(FEATURE_COLS)
    X_tr_aug = np.hstack([X_tr_norm, np.ones((X_tr_norm.shape[0], 1))]) / math.sqrt(d + 1)
    X_te_aug = np.hstack([X_te_norm, np.ones((X_te_norm.shape[0], 1))]) / math.sqrt(d + 1)

    C_REG = 0.02
    l2_sens = 2.0 * math.sqrt(2) * C_REG

    # Regularized baseline
    base_clf = LogisticRegression(C=C_REG, fit_intercept=False, max_iter=1000, multi_class='multinomial', random_state=BASE_SEED)
    base_clf.fit(X_tr_aug, y_tr)
    base_acc = accuracy_score(y_te, base_clf.predict(X_te_aug))

    # Tuned non-private baseline (C=1.0)
    tuned_clf = LogisticRegression(C=1.0, max_iter=1000, multi_class='multinomial', random_state=BASE_SEED)
    tuned_clf.fit(X_tr, y_tr)
    tuned_acc = accuracy_score(y_te, tuned_clf.predict(X_te))

    def train_fn(eps, seed):
        sigma = analytic_gaussian_sigma(eps, DELTA, sensitivity=l2_sens)
        clf = LogisticRegression(C=C_REG, fit_intercept=False, max_iter=1000, multi_class='multinomial', random_state=seed)
        clf.fit(X_tr_aug, y_tr)
        rng = np.random.RandomState(seed)
        noisy_W = clf.coef_ + rng.normal(0, sigma, size=clf.coef_.shape)
        preds = np.argmax(X_te_aug @ noisy_W.T, axis=1)
        return preds

    return "Logistic Reg (Gaussian)", "Output Perturbation", "(e, d)-DP", base_acc, y_te, train_fn

# ===========================================================================
#  MODEL 4: Logistic Regression (Objective Perturbation)
# ===========================================================================
def eval_lr_laplace():
    X_tr, X_te, y_tr, y_te = load_and_split("datasets/dataset_4_lr_laplace.json")
    X_tr_norm = X_tr.copy().astype(float)
    X_te_norm = X_te.copy().astype(float)
    for i, col in enumerate(FEATURE_COLS):
        lo, hi = DOMAIN_BOUNDS[col]
        mid = (lo + hi) / 2.0
        scale = (hi - lo) / 4.0
        X_tr_norm[:, i] = np.clip((X_tr[:, i] - mid) / scale, -2.0, 2.0)
        X_te_norm[:, i] = np.clip((X_te[:, i] - mid) / scale, -2.0, 2.0)

    data_norm = 4.0

    base_clf = LogisticRegression(C=1.0, max_iter=1000, multi_class='multinomial', random_state=BASE_SEED)
    base_clf.fit(X_tr_norm, y_tr)
    base_acc = accuracy_score(y_te, base_clf.predict(X_te_norm))

    def train_fn(eps, seed):
        model = dp.LogisticRegression(epsilon=eps, data_norm=data_norm, random_state=seed)
        model.fit(X_tr_norm, y_tr)
        return model.predict(X_te_norm)

    return "Logistic Reg (Laplace)", "Objective Perturbation", "pure e-DP", base_acc, y_te, train_fn


# ===========================================================================
#  MAIN BENCHMARK RUNNER
# ===========================================================================
def main():
    print("\n" + "="*78)
    print("  PrivaCare-AI -- Comprehensive DP Benchmark (compare_all.py)")
    print(f"  Epsilons : {EPSILONS} | Trials: {N_TRIALS} per configuration")
    print(f"  Delta    : {DELTA} (for approximate DP models)")
    print("="*78 + "\n")

    evaluators = [
        eval_nb_gaussian,
        eval_rf_laplace,
        eval_lr_gaussian,
        eval_lr_laplace,
    ]

    results = []

    for fn in evaluators:
        name, mech, dp_type, base_acc, y_te, train_fn = fn()
        print(f"--> Benchmarking: {name} [{mech} | {dp_type}] (Baseline: {base_acc*100:.2f}%)")
        row = {
            "Model": name,
            "Mechanism": mech,
            "DP Type": dp_type,
            "Baseline": f"{base_acc*100:.2f}%",
            "base_acc_raw": base_acc
        }

        for eps in EPSILONS:
            accs, f1s = [], []
            for trial in range(N_TRIALS):
                seed = BASE_SEED + trial
                y_pred = train_fn(eps, seed)
                accs.append(accuracy_score(y_te, y_pred))
                f1s.append(f1_score(y_te, y_pred, average="macro"))

            acc_m = float(np.mean(accs)) * 100
            acc_s = float(np.std(accs)) * 100
            f1_m  = float(np.mean(f1s))

            row[f"e={eps} Acc"] = f"{acc_m:.2f}% +/- {acc_s:.2f}%"
            row[f"e={eps} F1"]  = f"{f1_m:.4f}"
            row[f"e_{eps}_acc_mean"] = acc_m
            row[f"e_{eps}_acc_std"]  = acc_s
            print(f"    eps={eps}: Acc = {acc_m:.2f}% +/- {acc_s:.2f}% | F1 = {f1_m:.4f}")

        results.append(row)
        print()

    # ===========================================================================
    #  FORMAT SUMMARY TABLES
    # ===========================================================================
    output_lines = []
    output_lines.append("="*92)
    output_lines.append("  PRIVACARE-AI: FULL BENCHMARK RESULTS (N=30 Trials, UTF-8 Encoded)")
    output_lines.append("="*92)

    hdr = f"{'Model':<25} {'Mechanism':<24} {'DP Type':<11} {'Baseline':<10} | {'e=0.5':^19} | {'e=0.7':^19} | {'e=1.0':^19}"
    output_lines.append("\n" + hdr)
    output_lines.append("-" * len(hdr))

    for r in results:
        line = (f"{r['Model']:<25} {r['Mechanism']:<24} {r['DP Type']:<11} {r['Baseline']:<10} | "
                f"{r['e=0.5 Acc']:^19} | "
                f"{r['e=0.7 Acc']:^19} | "
                f"{r['e=1.0 Acc']:^19}")
        output_lines.append(line)

    output_lines.append("\n  -- F1 Scores (Macro Average) --\n")
    hdr2 = f"{'Model':<25} {'Mechanism':<24} | {'e=0.5':^12} | {'e=0.7':^12} | {'e=1.0':^12}"
    output_lines.append(hdr2)
    output_lines.append("-" * len(hdr2))
    for r in results:
        line2 = (f"{r['Model']:<25} {r['Mechanism']:<24} | "
                 f"{r['e=0.5 F1']:^12} | "
                 f"{r['e=0.7 F1']:^12} | "
                 f"{r['e=1.0 F1']:^12}")
        output_lines.append(line2)

    # Dynamic Key Empirical Findings
    output_lines.append("\n" + "="*92)
    output_lines.append("  EMPIRICAL RESEARCH FINDINGS (COMPUTED DYNAMICALLY OVER N=30 TRIALS):")
    for r in results:
        base_num = r["base_acc_raw"] * 100
        e05_num  = r["e_0.5_acc_mean"]
        e10_num  = r["e_1.0_acc_mean"]
        drop_05  = base_num - e05_num
        recovery = e10_num - e05_num
        output_lines.append(
            f"  * {r['Model']:<25}: Baseline = {base_num:.2f}%. At strict privacy (e=0.5), accuracy drop is "
            f"{drop_05:.2f}%. Relaxing budget to e=1.0 recovers +{recovery:.2f}% accuracy."
        )
    output_lines.append("="*92 + "\n")

    summary_text = "\n".join(output_lines)
    print(summary_text)

    # Save to results/compare_all.txt with UTF-8 encoding
    os.makedirs("results", exist_ok=True)
    with open("results/compare_all.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)
    print("  Results saved to results/compare_all.txt (UTF-8 encoded)\n")

if __name__ == "__main__":
    main()
