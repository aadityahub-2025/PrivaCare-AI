import matplotlib.pyplot as plt
import numpy as np
import os
import math
import pandas as pd
from scipy.special import erfc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
import diffprivlib.models as dp
import warnings
warnings.filterwarnings("once")

plt.style.use('bmh')

FEATURE_COLS = ["glucose_level", "stress_level", "heart_rate", "blood_pressure_systolic"]
DOMAIN_BOUNDS = {
    "glucose_level": (30.0, 300.0),
    "stress_level": (0.0, 1.0),
    "heart_rate": (30, 220),
    "blood_pressure_systolic": (60, 250)
}
CLASS_NAMES = ["Healthy", "Pre-diabetic", "Hypertensive", "Metabolic Risk"]
epsilon = 0.5
delta = 1e-5

def normalize(X):
    X_norm = X.copy().astype(float)
    for i, col in enumerate(FEATURE_COLS):
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

# Model 1: Gaussian NB (Sufficient Stats DP)
df1 = pd.read_json("datasets/dataset_1_rf_gaussian.json")
X1 = normalize(df1[FEATURE_COLS].values.astype(float))
y1 = df1["health_event"].values
X1_tr, X1_te, y1_tr, y1_te = train_test_split(X1, y1, test_size=0.2, random_state=42, stratify=y1)
nb_model = dp.GaussianNB(epsilon=epsilon, bounds=([0.0]*4, [1.0]*4))
nb_model.fit(X1_tr, y1_tr)
nb_preds = nb_model.predict(X1_te)
cm_nb = confusion_matrix(y1_te, nb_preds, labels=[0, 1, 2, 3])

# Model 2: Random Forest (Tree-DP)
df2 = pd.read_json("datasets/dataset_2_rf_laplace.json")
X2 = normalize(df2[FEATURE_COLS].values.astype(float))
y2 = df2["health_event"].values
X2_tr, X2_te, y2_tr, y2_te = train_test_split(X2, y2, test_size=0.2, random_state=42, stratify=y2)
rf_model = dp.RandomForestClassifier(n_estimators=20, max_depth=10, epsilon=epsilon, bounds=([0.0]*4, [1.0]*4), classes=np.unique(y2_tr), random_state=42)
rf_model.fit(X2_tr, y2_tr)
rf_preds = rf_model.predict(X2_te)
cm_rf = confusion_matrix(y2_te, rf_preds, labels=[0, 1, 2, 3])

# Model 3: Logistic Regression (Output Gaussian DP)
df3 = pd.read_json("datasets/dataset_3_lr_gaussian.json")
X3_norm = normalize(df3[FEATURE_COLS].values.astype(float))
y3 = df3["health_event"].values
X3_aug = np.hstack([X3_norm, np.ones((X3_norm.shape[0], 1))]) / math.sqrt(5)
X3_tr, X3_te, y3_tr, y3_te = train_test_split(X3_aug, y3, test_size=0.2, random_state=42, stratify=y3)
C_REG = 0.02
l2_sens = 2.0 * math.sqrt(2) * C_REG
sigma = analytic_gaussian_sigma(epsilon, delta, l2_sens)
lr_g_clf = LogisticRegression(C=C_REG, fit_intercept=False, max_iter=1000, multi_class='multinomial', random_state=42)
lr_g_clf.fit(X3_tr, y3_tr)
rng = np.random.RandomState(42)
noisy_W = lr_g_clf.coef_ + rng.normal(0, sigma, size=lr_g_clf.coef_.shape)
lr_g_preds = np.argmax(X3_te @ noisy_W.T, axis=1)
cm_lr_g = confusion_matrix(y3_te, lr_g_preds, labels=[0, 1, 2, 3])

# Model 4: Logistic Regression (Objective Laplace DP)
df4 = pd.read_json("datasets/dataset_4_lr_laplace.json")
X4 = df4[FEATURE_COLS].values.astype(float)
y4 = df4["health_event"].values
X4_norm = np.zeros_like(X4)
for i, col in enumerate(FEATURE_COLS):
    lo, hi = DOMAIN_BOUNDS[col]
    mid = (lo + hi) / 2.0
    scale = (hi - lo) / 4.0
    X4_norm[:, i] = np.clip((X4[:, i] - mid) / scale, -2.0, 2.0)
X4_tr, X4_te, y4_tr, y4_te = train_test_split(X4_norm, y4, test_size=0.2, random_state=42, stratify=y4)
lr_l_model = dp.LogisticRegression(epsilon=epsilon, data_norm=4.0, random_state=42)
lr_l_model.fit(X4_tr, y4_tr)
lr_l_preds = lr_l_model.predict(X4_te)
cm_lr_l = confusion_matrix(y4_te, lr_l_preds, labels=[0, 1, 2, 3])

# Plot 2x2 grid
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
plots = [
    (axes[0, 0], cm_nb, "Gaussian NB (Sufficient Stats DP)", "pure e-DP"),
    (axes[0, 1], cm_rf, "Random Forest (Tree-DP)", "pure e-DP"),
    (axes[1, 0], cm_lr_g, "Logistic Reg (Output Gauss)", "(e, d)-DP"),
    (axes[1, 1], cm_lr_l, "Logistic Reg (Objective Lap)", "pure e-DP")
]

for ax, cm, title, dp_type in plots:
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title(f"{title} [{dp_type}, $\epsilon={epsilon}$]", fontsize=12, fontweight='bold', pad=12)
    ax.set_xticks(np.arange(len(CLASS_NAMES)))
    ax.set_yticks(np.arange(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, fontsize=9, rotation=30, ha="right")
    ax.set_yticklabels(CLASS_NAMES, fontsize=9)
    thresh = cm.max() / 2.
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=11, fontweight='bold')
    ax.set_ylabel('True Class', fontsize=11, fontweight='bold')
    ax.set_xlabel('Predicted Class', fontsize=11, fontweight='bold')

plt.tight_layout()
os.makedirs("visualizations", exist_ok=True)
plt.savefig("visualizations/fig4_confusion_matrices.png", dpi=300)
plt.close()
print("[+] Generated 2x2 confusion matrix grid in visualizations/fig4_confusion_matrices.png")
