"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         PrivaCare-AI — Matplotlib Visualizations (Auto-Adaptive)            ║
║         Kisi bhi dataset ke saath kaam karta hai                             ║
╚══════════════════════════════════════════════════════════════════════════════╝

Yeh file chalao AFTER dp_train_test.py
Automatically detect karta hai:
  - Dataset columns
  - Target column
  - Number of classes
  - Feature types
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import joblib
import json
import warnings
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore")
np.random.seed(42)

# ──────────────────────────────────────────────
#  PATHS
# ──────────────────────────────────────────────
DATA_PATH    = "data/dataset.csv"
MODEL_PATH   = "models/dp_rf_model.pkl"
SCALER_PATH  = "models/scaler.pkl"
ENCODER_PATH = "models/label_encoder.pkl"
RESULTS_PATH = "results/training_results.json"
PLOTS_DIR    = "results/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# ──────────────────────────────────────────────
#  DARK THEME SETUP
# ──────────────────────────────────────────────
BG_COLOR    = "#0F1117"
PANEL_COLOR = "#1A1D2E"
GRID_COLOR  = "#2A2D3A"
TEXT_COLOR  = "#E8EAF6"
ACCENT      = "#00E5FF"
COLORS10    = ["#4C72B0","#DD8452","#55A868","#C44E52",
               "#8172B3","#937860","#DA8BC3","#8C8C8C","#CCB974","#64B5CD"]

plt.rcParams.update({
    "figure.facecolor":  BG_COLOR,
    "axes.facecolor":    PANEL_COLOR,
    "axes.edgecolor":    GRID_COLOR,
    "axes.labelcolor":   TEXT_COLOR,
    "axes.titlecolor":   TEXT_COLOR,
    "xtick.color":       TEXT_COLOR,
    "ytick.color":       TEXT_COLOR,
    "text.color":        TEXT_COLOR,
    "grid.color":        GRID_COLOR,
    "grid.alpha":        0.4,
    "legend.facecolor":  PANEL_COLOR,
    "legend.edgecolor":  GRID_COLOR,
    "font.family":       "DejaVu Sans",
    "font.size":         11,
})


# ══════════════════════════════════════════════
#  DATA LOAD (Auto-detect columns)
# ══════════════════════════════════════════════
def load_data():
    df = pd.read_csv(DATA_PATH)

    # Convert numerics
    for col in df.columns:
        try:
            converted = pd.to_numeric(df[col], errors='coerce')
            if converted.notna().mean() > 0.8:
                df[col] = converted
        except Exception:
            pass

    # Auto-detect target
    if "health_event" in df.columns:
        target_col = "health_event"
    elif "disease" in df.columns:
        target_col = "disease"
    else:
        target_col = df.columns[-1]

    # Encode gender if present
    if "gender" in df.columns and df["gender"].dtype == object:
        df["gender"] = (df["gender"].str.strip().str.lower() == "male").astype(int)

    # Drop ID/meta columns
    always_drop = ["timestamp", "device_id", "patient_id", "is_synthetic", target_col]
    drop_extra  = [c for c in df.columns
                   if c not in always_drop and df[c].dtype == object and df[c].nunique() > 50]
    all_drop = list(set(always_drop + drop_extra))

    feature_cols = [c for c in df.columns
                    if c not in all_drop and df[c].dtype in [np.float64, np.int64, float, int]]

    X     = df[feature_cols].values.astype(float)
    le    = joblib.load(ENCODER_PATH)
    y_raw = df[target_col].values
    y     = le.transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler      = joblib.load(SCALER_PATH)
    X_test_norm = scaler.transform(X_test)
    X_train_norm= scaler.transform(X_train)

    rf = joblib.load(MODEL_PATH)
    y_pred       = rf.predict(X_test_norm)
    y_pred_proba = rf.predict_proba(X_test_norm)

    class_names = [str(c) for c in le.classes_]

    return (df, rf, le, scaler, X_train_norm, X_test_norm,
            y_test, y_pred, y_pred_proba, feature_cols, class_names, target_col)


# ══════════════════════════════════════════════
#  PLOT 1: CONFUSION MATRIX
# ══════════════════════════════════════════════
def plot_confusion_matrix(y_test, y_pred, class_names):
    fig, ax = plt.subplots(figsize=(max(7, len(class_names)*2), max(6, len(class_names)*2)))
    fig.patch.set_facecolor(BG_COLOR)

    cm      = confusion_matrix(y_test, y_pred)
    cm_pct  = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    cmap    = LinearSegmentedColormap.from_list("dp", [PANEL_COLOR, "#4C72B0", ACCENT])
    ax.imshow(cm_pct, cmap=cmap, vmin=0, vmax=100)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, fontsize=11)
    ax.set_yticklabels(class_names, fontsize=11)
    ax.set_xlabel("Predicted", fontsize=12, labelpad=8)
    ax.set_ylabel("Actual",    fontsize=12, labelpad=8)
    ax.set_title(f"Confusion Matrix ({len(class_names)} Classes)\nDP + Random Forest",
                 fontsize=14, fontweight="bold", pad=12)

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            color = "white" if cm_pct[i,j] < 55 else BG_COLOR
            ax.text(j, i, f"{cm[i,j]}\n({cm_pct[i,j]:.0f}%)",
                    ha="center", va="center", fontsize=10, fontweight="bold", color=color)

    plt.tight_layout()
    path = f"{PLOTS_DIR}/1_confusion_matrix.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════
#  PLOT 2: FEATURE IMPORTANCE
# ══════════════════════════════════════════════
def plot_feature_importance(rf, feature_names):
    fig, ax = plt.subplots(figsize=(11, max(5, len(feature_names) * 0.55)))
    fig.patch.set_facecolor(BG_COLOR)

    imp = rf.feature_importances_
    idx = np.argsort(imp)
    sf  = [feature_names[i] for i in idx]
    si  = imp[idx]
    clrs= plt.cm.Blues(np.linspace(0.3, 1.0, len(sf)))

    bars = ax.barh(sf, si * 100, color=clrs, edgecolor=GRID_COLOR, height=0.65)
    for bar, val in zip(bars, si * 100):
        ax.text(val + 0.15, bar.get_y() + bar.get_height()/2,
                f"{val:.2f}%", va="center", fontsize=9.5, color=TEXT_COLOR)

    ax.set_xlabel("Importance (%)", fontsize=12)
    ax.set_title("Feature Importance (Random Forest — DP Trained)",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xlim(0, max(si*100) + 3)
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    path = f"{PLOTS_DIR}/2_feature_importance.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════
#  PLOT 3: TARGET CLASS DISTRIBUTION
# ══════════════════════════════════════════════
def plot_class_distribution(df, target_col, class_names):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle(f"Class Distribution — '{target_col}'",
                 fontsize=15, fontweight="bold", y=1.01)

    counts = df[target_col].value_counts().sort_index()
    labels = [str(c) for c in counts.index]
    values = counts.values
    colors = COLORS10[:len(labels)]

    # Donut pie
    wedge_props = dict(width=0.55, edgecolor=BG_COLOR, linewidth=2)
    _, texts, autotexts = ax1.pie(values, labels=labels, autopct="%1.1f%%",
                                   colors=colors, wedgeprops=wedge_props,
                                   textprops={"color": TEXT_COLOR, "fontsize": 11},
                                   startangle=90)
    for at in autotexts:
        at.set_fontsize(12); at.set_fontweight("bold"); at.set_color(BG_COLOR)
    ax1.set_title("Proportion", fontsize=13, pad=10)
    ax1.set_facecolor(BG_COLOR)

    # Bar chart
    bars = ax2.bar(labels, values, color=colors, edgecolor=GRID_COLOR, width=0.6)
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                 f"{val:,}", ha="center", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_xlabel(f"Class ({target_col})", fontsize=12)
    ax2.set_title("Records per Class", fontsize=13, pad=10)
    ax2.set_ylim(0, max(values) * 1.15)
    ax2.grid(axis="y", alpha=0.3)
    ax2.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    path = f"{PLOTS_DIR}/3_class_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════
#  PLOT 4: ROC CURVES (Multiclass)
# ══════════════════════════════════════════════
def plot_roc_curves(y_test, y_pred_proba, class_names):
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor(BG_COLOR)

    y_bin = label_binarize(y_test, classes=range(n))
    for i, (name, color) in enumerate(zip(class_names, COLORS10)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_pred_proba[:, i])
        roc_val     = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2.5,
                label=f"Class {name}  (AUC = {roc_val:.4f})")
        ax.fill_between(fpr, tpr, alpha=0.07, color=color)

    ax.plot([0,1],[0,1],"w--", lw=1.5, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title("ROC Curves — Multiclass (One vs Rest)\nDP + Random Forest",
                 fontsize=14, fontweight="bold", pad=12)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(alpha=0.3)
    ax.set_xlim([-0.01, 1.01]); ax.set_ylim([-0.01, 1.05])
    ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    path = f"{PLOTS_DIR}/4_roc_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════
#  PLOT 5: PRIVACY TRADEOFF (ε vs σ)
# ══════════════════════════════════════════════
def plot_privacy_tradeoff(current_epsilon=5.0):
    # Load actual epsilon from results if available
    try:
        with open(RESULTS_PATH) as f:
            res = json.load(f)
        current_epsilon = res["privacy"]["epsilon"]
        current_sigma   = res["privacy"]["sigma"]
    except Exception:
        current_sigma = np.sqrt(2 * np.log(1.25 / 1e-5)) / current_epsilon

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle("Differential Privacy: Epsilon vs Noise Analysis",
                 fontsize=15, fontweight="bold", y=1.01)

    eps_range = np.linspace(0.1, 20, 500)
    sig_range = np.sqrt(2 * np.log(1.25 / 1e-5)) / eps_range

    # Left: ε vs σ
    ax1.plot(eps_range, sig_range, color=ACCENT, lw=2.5)
    ax1.fill_between(eps_range, sig_range, alpha=0.12, color=ACCENT)
    ax1.scatter([current_epsilon], [current_sigma],
                color="#FF6B6B", s=130, zorder=5,
                label=f"Current Model\nε={current_epsilon}, σ={current_sigma:.3f}")
    ax1.axvline(current_epsilon, color="#FF6B6B", ls="--", alpha=0.5, lw=1.5)
    ax1.axhline(current_sigma,   color="#FF6B6B", ls="--", alpha=0.5, lw=1.5)

    ax1.axvspan(0,  2, alpha=0.06, color="#FF6B6B", label="High Privacy (ε<2)")
    ax1.axvspan(2,  8, alpha=0.06, color="#FFA500", label="Moderate (2–8)")
    ax1.axvspan(8, 20, alpha=0.06, color="#55A868", label="Low Privacy (ε>8)")

    ax1.set_xlabel("Epsilon (ε)", fontsize=12)
    ax1.set_ylabel("Sigma (σ) — Noise Scale", fontsize=12)
    ax1.set_title("ε → σ Relationship (δ = 1e-5)", fontsize=12)
    ax1.legend(fontsize=9, loc="upper right"); ax1.grid(alpha=0.3)
    ax1.set_xlim([0, 20])
    ax1.spines[["top","right"]].set_visible(False)

    # Right: Gaussian noise distributions at different epsilons
    x = np.linspace(-6, 6, 1000)
    eps_show = [1.0, 3.0, 5.0, 10.0]
    clrs     = ["#FF6B6B","#FFA500","#00E5FF","#55A868"]
    for ep, clr in zip(eps_show, clrs):
        sig = np.sqrt(2 * np.log(1.25 / 1e-5)) / ep
        y   = (1/(sig*np.sqrt(2*np.pi))) * np.exp(-0.5*(x/sig)**2)
        ax2.plot(x, y, color=clr, lw=2.5, label=f"ε={ep:.0f}  → σ={sig:.2f}")
        ax2.fill_between(x, y, alpha=0.08, color=clr)

    # Highlight current
    sig_curr = np.sqrt(2 * np.log(1.25 / 1e-5)) / current_epsilon
    y_curr   = (1/(sig_curr*np.sqrt(2*np.pi))) * np.exp(-0.5*(x/sig_curr)**2)
    ax2.plot(x, y_curr, color="#FF6B6B", lw=3, ls="--",
             label=f"Current (ε={current_epsilon})")

    ax2.axvline(0, color="white", lw=1, alpha=0.3)
    ax2.set_xlabel("Noise Added to Data", fontsize=12)
    ax2.set_ylabel("Probability Density",  fontsize=12)
    ax2.set_title("Gaussian Noise Distributions\n(Khili curve = kam noise = better accuracy)",
                  fontsize=12)
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3)
    ax2.set_xlim([-6, 6])
    ax2.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    path = f"{PLOTS_DIR}/5_privacy_tradeoff.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════
#  PLOT 6: DP NOISE EFFECT (Before vs After)
# ══════════════════════════════════════════════
def plot_dp_noise_effect(X_train_norm, feature_names, current_sigma=0.969):
    try:
        with open(RESULTS_PATH) as f:
            res = json.load(f)
        current_sigma = res["privacy"]["sigma"]
    except Exception:
        pass

    n_feat = len(feature_names)
    cols   = min(4, n_feat)
    rows   = (n_feat + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*3.5))
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle(f"DP Gaussian Noise Effect on Features\n(σ={current_sigma:.4f}  — Original vs DP-Noisy)",
                 fontsize=14, fontweight="bold", y=1.01)

    if rows * cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    noise      = np.random.normal(0, current_sigma, X_train_norm[:500].shape)
    X_noisy    = np.clip(X_train_norm[:500] + noise, 0, 1)

    for idx in range(rows * cols):
        r, c = divmod(idx, cols)
        ax   = axes[r][c]
        if idx >= n_feat:
            ax.set_visible(False)
            continue
        orig  = X_train_norm[:500, idx]
        noisy = X_noisy[:, idx]
        ax.hist(orig,  bins=25, alpha=0.65, color="#4C72B0", label="Original", density=True)
        ax.hist(noisy, bins=25, alpha=0.65, color="#FF6B6B", label="DP Noisy", density=True)
        fname = feature_names[idx].replace("_", "\n")
        ax.set_title(fname, fontsize=10, fontweight="bold")
        ax.grid(alpha=0.3)
        ax.spines[["top","right"]].set_visible(False)
        if idx == 0:
            ax.legend(fontsize=8)

    plt.tight_layout()
    path = f"{PLOTS_DIR}/6_dp_noise_effect.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════
#  PLOT 7: FEATURE DISTRIBUTION PER CLASS
# ══════════════════════════════════════════════
def plot_feature_per_class(df, target_col, feature_names, class_names):
    # Pick top 6 most important-looking numeric features
    n_show = min(6, len(feature_names))
    features_show = feature_names[:n_show]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle(f"Feature Distribution per Class ({target_col})",
                 fontsize=14, fontweight="bold", y=1.01)

    for idx, (ax, feat) in enumerate(zip(axes.flat, features_show)):
        for ci, (cls, color) in enumerate(zip(class_names, COLORS10)):
            subset = df[df[target_col].astype(str) == cls][feat]
            subset_num = pd.to_numeric(subset, errors='coerce').dropna()
            if len(subset_num) > 0:
                ax.hist(subset_num, bins=25, alpha=0.55, color=color,
                        label=f"Class {cls}", density=True)
        fname = feat.replace("_", " ").title()
        ax.set_title(fname, fontsize=11, fontweight="bold")
        ax.set_xlabel("Value", fontsize=9)
        ax.grid(alpha=0.3)
        ax.spines[["top","right"]].set_visible(False)
        if idx == 0:
            ax.legend(fontsize=8)

    # Hide unused
    for ax in axes.flat[n_show:]:
        ax.set_visible(False)

    plt.tight_layout()
    path = f"{PLOTS_DIR}/7_feature_per_class.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════
#  PLOT 8: FEATURE CORRELATION HEATMAP
# ══════════════════════════════════════════════
def plot_correlation_heatmap(df, feature_names):
    fig, ax = plt.subplots(figsize=(max(8, len(feature_names)), max(7, len(feature_names))))
    fig.patch.set_facecolor(BG_COLOR)

    feat_df = df[[f for f in feature_names if f in df.columns]].apply(
        pd.to_numeric, errors='coerce'
    ).dropna()
    corr = feat_df.corr()

    cmap = LinearSegmentedColormap.from_list("corr", ["#DD4444", PANEL_COLOR, "#4C72B0"])
    im   = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    labels = [c.replace("_", "\n") for c in corr.columns]
    ax.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold", pad=12)

    for i in range(len(corr)):
        for j in range(len(corr.columns)):
            val = corr.values[i, j]
            color = "white" if abs(val) < 0.5 else BG_COLOR
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7.5, color=color)

    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02).set_label("Correlation", color=TEXT_COLOR)
    plt.tight_layout()
    path = f"{PLOTS_DIR}/8_correlation_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════
#  PLOT 9: MODEL CONFIDENCE
# ══════════════════════════════════════════════
def plot_confidence(y_pred, y_pred_proba, class_names):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle("Prediction Confidence Analysis", fontsize=15, fontweight="bold", y=1.01)

    max_conf = y_pred_proba.max(axis=1) * 100

    ax1.hist(max_conf, bins=40, color="#4C72B0", edgecolor=BG_COLOR, alpha=0.85, density=True)
    ax1.axvline(max_conf.mean(), color="#FF6B6B", lw=2.5, ls="--",
                label=f"Mean: {max_conf.mean():.1f}%")
    ax1.axvline(np.median(max_conf), color="#FFD700", lw=2, ls=":",
                label=f"Median: {np.median(max_conf):.1f}%")
    ax1.set_xlabel("Max Confidence (%)", fontsize=12)
    ax1.set_ylabel("Density", fontsize=12)
    ax1.set_title("Overall Confidence Distribution", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10); ax1.grid(alpha=0.3)
    ax1.spines[["top","right"]].set_visible(False)

    for i, (name, color) in enumerate(zip(class_names, COLORS10)):
        mask   = y_pred == i
        conf_i = max_conf[mask]
        if len(conf_i) > 0:
            ax2.hist(conf_i, bins=25, alpha=0.6, color=color,
                     label=f"Class {name} (avg={conf_i.mean():.0f}%)", density=True)
    ax2.set_xlabel("Confidence (%)", fontsize=12)
    ax2.set_ylabel("Density",        fontsize=12)
    ax2.set_title("Confidence per Class", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3)
    ax2.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    path = f"{PLOTS_DIR}/9_confidence.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════
#  PLOT 0: MASTER DASHBOARD
# ══════════════════════════════════════════════
def plot_dashboard(df, rf, le, X_train_norm, X_test_norm, y_test, y_pred, y_pred_proba,
                   feature_names, class_names, target_col):
    # Load results
    try:
        with open(RESULTS_PATH) as f:
            res = json.load(f)
        test_acc  = res["performance"]["test_accuracy"]
        auc_val   = res["performance"]["auc_roc"]
        epsilon   = res["privacy"]["epsilon"]
        sigma     = res["privacy"]["sigma"]
        cv_mean   = res["performance"]["cv_mean"]
        cv_std    = res["performance"]["cv_std"]
        n_train   = res["run_info"]["train_size"]
        n_test    = res["run_info"]["test_size_n"]
    except Exception:
        from sklearn.metrics import accuracy_score
        test_acc = accuracy_score(y_test, y_pred)
        auc_val  = None; epsilon = 5.0; sigma = 0.969
        cv_mean  = 0.0;  cv_std  = 0.0
        n_train  = len(y_test)*4; n_test = len(y_test)

    n_classes = len(class_names)
    fig = plt.figure(figsize=(22, 15))
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle(
        f"PrivaCare-AI — DP + Random Forest | Dataset: {target_col} ({n_classes} classes)",
        fontsize=17, fontweight="bold", color=TEXT_COLOR, y=0.99
    )
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.35)

    # ── A: Confusion Matrix ──
    ax_cm  = fig.add_subplot(gs[0, 0])
    cm     = confusion_matrix(y_test, y_pred)
    cm_pct = cm.astype(float)/cm.sum(axis=1,keepdims=True)*100
    cmap2  = LinearSegmentedColormap.from_list("dp",[PANEL_COLOR,"#4C72B0",ACCENT])
    ax_cm.imshow(cm_pct, cmap=cmap2)
    ax_cm.set_xticks(range(n_classes)); ax_cm.set_yticks(range(n_classes))
    ax_cm.set_xticklabels(class_names, fontsize=8); ax_cm.set_yticklabels(class_names, fontsize=8)
    ax_cm.set_title("Confusion Matrix", fontsize=12, fontweight="bold", pad=8)
    for i in range(n_classes):
        for j in range(n_classes):
            c = "white" if cm_pct[i,j]<55 else BG_COLOR
            ax_cm.text(j,i,f"{cm[i,j]}\n({cm_pct[i,j]:.0f}%)",
                       ha="center",va="center",fontsize=7,fontweight="bold",color=c)

    # ── B: Feature Importance ──
    ax_fi = fig.add_subplot(gs[0, 1:])
    imp   = rf.feature_importances_; idx = np.argsort(imp)
    sf    = [feature_names[i] for i in idx]; si = imp[idx]
    cfi   = plt.cm.Blues(np.linspace(0.3,1.0,len(sf)))
    ax_fi.barh(sf, si*100, color=cfi, edgecolor=GRID_COLOR, height=0.65)
    for i,(f,v) in enumerate(zip(sf,si*100)):
        ax_fi.text(v+0.1,i,f"{v:.1f}%",va="center",fontsize=8)
    ax_fi.set_title("Feature Importance (%)",fontsize=12,fontweight="bold",pad=8)
    ax_fi.set_xlabel("Importance"); ax_fi.grid(axis="x",alpha=0.3)
    ax_fi.spines[["top","right"]].set_visible(False)

    # ── C: ROC Curves ──
    ax_roc = fig.add_subplot(gs[1, 0])
    y_bin  = label_binarize(y_test, classes=range(n_classes))
    for i,(name,color) in enumerate(zip(class_names,COLORS10)):
        fpr,tpr,_ = roc_curve(y_bin[:,i], y_pred_proba[:,i])
        ax_roc.plot(fpr,tpr,color=color,lw=2,label=f"C{name}({auc(fpr,tpr):.2f})")
    ax_roc.plot([0,1],[0,1],"w--",lw=1,alpha=0.4)
    ax_roc.set_title("ROC Curves",fontsize=12,fontweight="bold",pad=8)
    ax_roc.set_xlabel("FPR",fontsize=9); ax_roc.set_ylabel("TPR",fontsize=9)
    ax_roc.legend(fontsize=7,loc="lower right"); ax_roc.grid(alpha=0.3)
    ax_roc.spines[["top","right"]].set_visible(False)

    # ── D: Privacy Tradeoff ──
    ax_dp = fig.add_subplot(gs[1, 1])
    er    = np.linspace(0.5, 15, 300)
    sr    = np.sqrt(2*np.log(1.25/1e-5))/er
    ax_dp.plot(er, sr, color=ACCENT, lw=2)
    ax_dp.scatter([epsilon],[sigma],color="#FF6B6B",s=90,zorder=5,
                  label=f"Current\nε={epsilon}, σ={sigma:.3f}")
    ax_dp.axvline(epsilon,color="#FF6B6B",ls="--",alpha=0.4,lw=1.5)
    ax_dp.set_title("Privacy Tradeoff (ε vs σ)",fontsize=12,fontweight="bold",pad=8)
    ax_dp.set_xlabel("Epsilon (ε)"); ax_dp.set_ylabel("Sigma (σ)")
    ax_dp.legend(fontsize=8); ax_dp.grid(alpha=0.3)
    ax_dp.spines[["top","right"]].set_visible(False)

    # ── E: Class Distribution ──
    ax_cls = fig.add_subplot(gs[1, 2])
    counts = df[target_col].value_counts().sort_index()
    cls_colors = COLORS10[:len(counts)]
    ax_cls.bar([str(c) for c in counts.index], counts.values,
               color=cls_colors, edgecolor=GRID_COLOR, width=0.6)
    for i,(v) in enumerate(counts.values):
        ax_cls.text(i, v + max(counts)*0.01, str(v), ha="center", fontsize=9, fontweight="bold")
    ax_cls.set_title(f"Class Distribution ({target_col})",fontsize=12,fontweight="bold",pad=8)
    ax_cls.set_xlabel("Class"); ax_cls.set_ylabel("Count")
    ax_cls.grid(axis="y",alpha=0.3); ax_cls.spines[["top","right"]].set_visible(False)

    # ── F: Confidence Distribution ──
    ax_conf = fig.add_subplot(gs[2, 0:2])
    mc = y_pred_proba.max(axis=1)*100
    ax_conf.hist(mc, bins=50, color="#4C72B0", edgecolor=BG_COLOR, alpha=0.85, density=True)
    ax_conf.axvline(mc.mean(),color="#FF6B6B",lw=2,ls="--",
                    label=f"Mean={mc.mean():.1f}%")
    ax_conf.axvline(np.median(mc),color="#FFD700",lw=2,ls=":",
                    label=f"Median={np.median(mc):.1f}%")
    ax_conf.set_title("Prediction Confidence Distribution",fontsize=12,fontweight="bold",pad=8)
    ax_conf.set_xlabel("Confidence (%)"); ax_conf.set_ylabel("Density")
    ax_conf.legend(fontsize=10); ax_conf.grid(alpha=0.3)
    ax_conf.spines[["top","right"]].set_visible(False)

    # ── G: Summary Box ──
    ax_met = fig.add_subplot(gs[2, 2])
    ax_met.set_xlim(0,1); ax_met.set_ylim(0,1); ax_met.axis("off")
    metrics = [
        ("Test Accuracy",  f"{test_acc*100:.2f}%",  "#55A868"),
        ("AUC-ROC",        f"{auc_val:.4f}" if auc_val else "N/A", "#55A868"),
        ("CV Accuracy",    f"{cv_mean*100:.1f}±{cv_std*100:.1f}%", "#4C72B0"),
        ("Epsilon (ε)",    str(epsilon),  "#FFA500"),
        ("Sigma (σ)",      f"{sigma:.4f}", "#4C72B0"),
        ("Classes",        str(n_classes), TEXT_COLOR),
        ("Train Samples",  f"{n_train:,}", TEXT_COLOR),
        ("Test Samples",   f"{n_test:,}",  TEXT_COLOR),
        ("RF Trees",       "200",          TEXT_COLOR),
    ]
    ax_met.text(0.5,0.97,"Model Summary",ha="center",va="top",
                fontsize=12,fontweight="bold",color=TEXT_COLOR,transform=ax_met.transAxes)
    for i,(lbl,val,clr) in enumerate(metrics):
        yp = 0.86 - i*0.096
        ax_met.text(0.04,yp,lbl+":",fontsize=9,color="#AAAAAA",transform=ax_met.transAxes)
        ax_met.text(0.96,yp,val,fontsize=10,fontweight="bold",
                    color=clr,ha="right",transform=ax_met.transAxes)
        if i < len(metrics)-1:
            ly = yp - 0.045
            ax_met.plot([0.02,0.98],[ly,ly],color=GRID_COLOR,
                        lw=0.5,transform=ax_met.transAxes,clip_on=False)

    plt.savefig(f"{PLOTS_DIR}/0_DASHBOARD.png", dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"  Saved: {PLOTS_DIR}/0_DASHBOARD.png")


# ══════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")

    print("=" * 60)
    print("  PrivaCare-AI — Visualizations (Auto-Adaptive)")
    print("=" * 60)

    print("\n  Data aur models load ho rahe hain...")
    (df, rf, le, scaler, X_tr_n, X_te_n,
     y_test, y_pred, y_pred_proba,
     feature_cols, class_names, target_col) = load_data()

    print(f"  Dataset: {len(df):,} rows | Target: '{target_col}' | Classes: {class_names}")
    print(f"  Features ({len(feature_cols)}): {feature_cols}")
    print(f"\n  9 plots + 1 dashboard generate ho rahe hain...\n")

    print("  [1/9] Confusion Matrix...")
    plot_confusion_matrix(y_test, y_pred, class_names)

    print("  [2/9] Feature Importance...")
    plot_feature_importance(rf, feature_cols)

    print("  [3/9] Class Distribution...")
    plot_class_distribution(df, target_col, class_names)

    print("  [4/9] ROC Curves...")
    plot_roc_curves(y_test, y_pred_proba, class_names)

    print("  [5/9] Privacy Tradeoff...")
    plot_privacy_tradeoff()

    print("  [6/9] DP Noise Effect...")
    plot_dp_noise_effect(X_tr_n, feature_cols)

    print("  [7/9] Feature Distribution per Class...")
    plot_feature_per_class(df, target_col, feature_cols, class_names)

    print("  [8/9] Correlation Heatmap...")
    plot_correlation_heatmap(df, feature_cols)

    print("  [9/9] Confidence Distribution...")
    plot_confidence(y_pred, y_pred_proba, class_names)

    print("\n  [DASHBOARD] Master chart...")
    plot_dashboard(df, rf, le, X_tr_n, X_te_n, y_test, y_pred, y_pred_proba,
                   feature_cols, class_names, target_col)

    print(f"\n{'=' * 60}")
    print(f"  Sab 10 plots save ho gaye: {PLOTS_DIR}/")
    print(f"{'=' * 60}")
