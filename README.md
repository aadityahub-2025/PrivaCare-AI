# 🏥 PrivaCare-AI

> **Differential Privacy for Healthcare AI** — Protecting patient data with IBM's `diffprivlib` Analytic Gaussian Mechanism while maintaining predictive accuracy using Random Forest classifiers.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![sklearn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikit-learn&logoColor=white)
![diffprivlib](https://img.shields.io/badge/IBM-diffprivlib-purple)
![Privacy](https://img.shields.io/badge/Privacy-Differential%20Privacy-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Overview

**PrivaCare-AI** is a B.Tech research project that applies **Differential Privacy (DP)** to healthcare machine learning pipelines. It uses IBM's `diffprivlib` library to train a DP-protected Random Forest with rigorous correctness — providing algorithmic, model-level privacy guarantees — and compares it against a baseline (non-private) model. The project quantifies the **Privacy–Accuracy Trade-off** at different epsilon values.

### 🎯 Key Objectives
- Apply **(ε, δ)-Differential Privacy** using IBM `diffprivlib` on wearable health sensor data
- Use **data-independent feature bounds** (clinical domain knowledge) to preserve formal DP guarantee
- Train and compare **baseline vs. DP-protected** Random Forest classifiers across multiple trials
- Analyze the **Privacy-Accuracy Trade-off** using the Analytic Gaussian Mechanism (Balle & Wang, 2018)
- Generate rich visualizations: confusion matrix, ROC curves, class distribution, noise effects, and more

---

## 🗂️ Project Structure

```
PrivaCare-AI/
│
├── data/
│   └── dataset.csv                  # Primary health dataset (6,000 samples, 13 features)
│
├── results/
│   └── plots/                       # All generated visualizations (10 plots)
│       ├── 0_DASHBOARD.png          # Master summary dashboard
│       ├── 1_confusion_matrix.png
│       ├── 2_feature_importance.png
│       ├── 3_class_distribution.png
│       ├── 4_roc_curves.png
│       ├── 5_privacy_tradeoff.png
│       ├── 6_dp_noise_effect.png
│       ├── 7_feature_per_class.png
│       ├── 8_correlation_heatmap.png
│       └── 9_confidence.png
│
├── dp_train_test.py                 # DP Random Forest trainer (interactive, with diffprivlib)
├── visualize_results.py             # Generate all 10 result plots (auto-adaptive)
├── requirements.txt                 # Python dependencies
├── LICENSE                          # MIT License
├── .gitignore
└── README.md
```

---

## ⚙️ How It Works

### Privacy Mechanism — Analytic Gaussian Mechanism (Balle & Wang, 2018)

This project uses **IBM's `diffprivlib`** which implements privacy at the **algorithmic level** inside the Random Forest training process. The noise scale **σ** is computed via the **Analytic Gaussian Mechanism** — valid for **all ε > 0** (unlike the classical formula which is only proven for ε < 1):

$$\sigma^* = \min \left\{ \sigma : \Phi\!\left(\frac{\Delta f}{2\sigma} - \frac{\varepsilon\sigma}{\Delta f}\right) - e^{\varepsilon}\,\Phi\!\left(-\frac{\Delta f}{2\sigma} - \frac{\varepsilon\sigma}{\Delta f}\right) \leq \delta \right\}$$

| Symbol | Meaning |
|--------|---------|
| **ε (epsilon)** | Privacy budget — smaller = more private |
| **δ (delta)** | Failure probability (fixed at `1e-5`) |
| **Δf** | L∞-sensitivity = 1.0 (data-independent, domain-bound normalization) |
| **σ** | Minimum noise standard deviation (Analytic GM) |

This provides a formal **(ε, δ)-DP guarantee**: an adversary cannot identify any individual patient's record from the model output.

### DP Correctness Fixes Applied

| Fix | Issue | Solution |
|-----|-------|----------|
| **1** | Sensitivity from data (MinMaxScaler) | `DOMAIN_BOUNDS` dict with clinical ranges — data-independent |
| **2** | Classical σ formula only valid for ε < 1 | Analytic Gaussian Mechanism (Balle & Wang, 2018) |
| **3** | Epsilon composition not tracked | Per-run composition warning printed at runtime |
| **4** | Single run — unreliable results | `N_TRIALS = 3` runs, mean ± std reported |
| **5** | Binary features getting Gaussian noise | `BINARY_FEATURES` set, correct `(0,1)` bounds passed |
| **6** | No fixed seeds — non-reproducible | `seed = BASE_SEED + trial_idx` for each trial |
| **7** | Label (target) unprotected | Runtime note — standard DP-ML design (Abadi et al. 2016) |

### Model Pipeline

```
Raw CSV Data
    │
    ▼
Feature Selection + Label Encoding
    │
    ▼
Train / Test Split (80 / 20, stratified)
    │
    ▼
Domain-Bound Normalization → [0, 1]   ← data-independent (clinical bounds)
    │
    ├──── Baseline RF (sklearn, no privacy) ──────► Baseline Accuracy
    │
    └──── DP Random Forest (diffprivlib, ε, δ) ───► DP Accuracy (avg of N_TRIALS)
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

> Python **3.10+** recommended.

### Clone the Repository

```bash
git clone https://github.com/aadityahub-2025/PrivaCare-AI.git
cd PrivaCare-AI
```

---

## 🧪 Usage

### Step 1 — Train Models

Trains a baseline RF and a DP-protected RF (3 trials, averaged). Prompts for epsilon at runtime.

```bash
python dp_train_test.py
```

**Example output:**
```
========================================================
  PrivaCare-AI -- Differential Privacy Training
========================================================

  Dataset : 6,000 rows | 13 features
  Target  : 'health_event' | 4 classes

  +--[ PRIVACY SCOPE NOTE ]---...---+
  | Target label is NOT DP-protected (standard DP-ML design) |
  +----------------------------------------------------------+

  Enter Epsilon value (e.g. 0.1, 0.5, 1.0) [Default 0.5]: 0.5
  --> Epsilon (e)           = 0.5
      Delta   (d)           = 1e-05
      Sigma   (s, analytic) = 7.0318
      Trials                = 3 runs (results averaged)

  [!] COMPOSITION WARNING: Each run consumes epsilon = 0.5 ...

[1] Baseline Accuracy : 99.92%

[2] DP Random Forest (3 trials | epsilon=0.5 | diffprivlib)
    Trial 1/3  (seed=42): 80.17%
    Trial 2/3  (seed=43): 88.58%
    Trial 3/3  (seed=44): 84.67%

========================================================
  RESULT SUMMARY
========================================================
  Baseline Accuracy (No DP)           : 99.92%
  DP Accuracy (e=0.5, 3 trials)       : 84.47% +/- 3.44%
  Accuracy Drop (Privacy Cost)        : 15.44%
  Noise Sigma (Analytic GM)           : 7.0318
  Privacy Guarantee                   : (0.5, 1e-05)-DP
  Normalization                       : Domain-bound clipping (data-independent)
========================================================
```

---

### Step 2 — Generate Visualizations

```bash
python visualize_results.py
```

Enter the **same epsilon** as Step 1. Saves **10 plots** to `results/plots/`.

---

## 📊 Results

### Performance Summary (ε = 0.5, δ = 1e-5)

| Metric | Baseline (No DP) | DP Model (ε=0.5, 3 trials) |
|--------|:----------------:|:--------------------------:|
| **Test Accuracy** | 99.92% | **84.47% ± 3.44%** |
| **Noise σ (Analytic GM)** | 0.0 | 7.0318 |
| **Privacy Guarantee** | ❌ None | ✅ (0.5, 1e-5)-DP |
| **Normalization** | MinMaxScaler | Domain-bound clipping |

> **Key insight:** The variance (±3.44%) across 3 trials reveals the *stochastic cost* of DP training. The Analytic GM computes σ=7.03 for ε=0.5 — significantly more accurate than the classical formula (which gives σ=9.69, over-adding noise by ~38%).

### Per-Class Accuracy (Confusion Matrix, ε=0.5)

| Class | Correctly Predicted | Accuracy |
|-------|:-------------------:|:--------:|
| Normal | 299 / 300 | **100%** ✅ |
| Mild Risk | 70 / 300 | **23%** ⚠️ |
| Moderate Risk | 300 / 300 | **100%** ✅ |
| High Risk | 293 / 300 | **98%** ✅ |

> ⚠️ **Mild Risk confusion:** 71% of Mild Risk patients are misclassified as Moderate Risk. This is a known effect of DP noise on borderline/adjacent classes — the added noise blurs the decision boundary between similar classes.

### Privacy Level Classification

| Epsilon (ε) | Privacy Level | σ (Analytic GM) | Typical Use Case |
|:-----------:|:-------------:|:---------------:|:----------------:|
| ≤ 0.5 | 🔒 High Privacy | ~7.03 | Highly sensitive medical data |
| 0.5 – 2.0 | 🔐 Moderate-High | ~1.9–7.0 | Clinical research |
| 2.0 – 7.0 | 🔑 Moderate | ~0.6–1.9 | General health analytics |
| > 7.0 | 🔓 Low Privacy | < 0.6 | Non-sensitive aggregates |

### Feature Importance (Baseline RF — from actual run)

| Rank | Feature | Importance |
|:----:|---------|:----------:|
| 1 | Glucose Level | **18.6%** |
| 2 | Stress Level | 16.0% |
| 3 | Blood Pressure (Systolic) | 15.9% |
| 4 | Sleep Quality | 14.7% |
| 5 | Blood Pressure (Diastolic) | 12.9% |
| 6 | HRV SDNN | 5.9% |
| 7 | Heart Rate | 4.6% |
| 8 | Steps Count | 3.3% |
| 9 | Activity Level | 2.8% |

---

## 🖼️ Visualizations

All plots are saved in `results/plots/` after running `visualize_results.py`.

| Plot File | Description |
|-----------|-------------|
| `0_DASHBOARD.png` | Master summary dashboard (all metrics in one view) |
| `1_confusion_matrix.png` | Class-wise prediction confusion with per-class % |
| `2_feature_importance.png` | Baseline RF feature ranking |
| `3_class_distribution.png` | Target class balance — **Normal / Mild Risk / Moderate Risk / High Risk** |
| `4_roc_curves.png` | ROC curve per class (One vs Rest, with AUC) |
| `5_privacy_tradeoff.png` | ε vs σ (Analytic GM) + Gaussian noise distributions |
| `6_dp_noise_effect.png` | Original vs DP-noisy feature distributions |
| `7_feature_per_class.png` | Feature distributions separated by health class |
| `8_correlation_heatmap.png` | Feature correlation matrix |
| `9_confidence.png` | Model prediction confidence per class |

---

## 🧠 Dataset

- **Source:** Synthetic wearable health sensor data
- **Samples:** 6,000 patient records
- **Features (13):**
  `heart_rate`, `blood_oxygen`, `blood_pressure_systolic`, `blood_pressure_diastolic`,
  `glucose_level`, `body_temperature`, `respiratory_rate`, `activity_level`,
  `sleep_quality`, `stress_level`, `hrv_sdnn`, `steps_count`, `calories_burned`
- **Target:** `health_event` — 4 health classes (balanced, 1,500 each):
  - `0` → **Normal**
  - `1` → **Mild Risk**
  - `2` → **Moderate Risk**
  - `3` → **High Risk**
- **Split:** 80% train / 20% test (stratified)
- **Note:** Dataset is synthetic — no real patient records. For production use, a data ethics review and IRB approval would be required.

---

## 📐 Mathematical Background

### (ε, δ)-Differential Privacy

A randomized mechanism **M** satisfies **(ε, δ)-DP** if for all neighboring datasets **D**, **D'** (differing in one record), and all outputs **S**:

$$\Pr[M(D) \in S] \leq e^{\varepsilon} \cdot \Pr[M(D') \in S] + \delta$$

### Analytic Gaussian Mechanism (Balle & Wang, NeurIPS 2018)

The **Analytic GM** computes the minimum σ satisfying (ε, δ)-DP for **any ε > 0** via binary search on the normal CDF — as opposed to the classical formula which is only valid for ε < 1. This project uses the Analytic GM for both training (diffprivlib internally) and all σ display values in plots.

### Privacy Composition

Each training run on the same dataset consumes ε from the total privacy budget:
- **Basic composition:** k runs → total loss = k × ε
- **Advanced composition** (Dwork et al. 2010): O(√k · ε)

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.10+ |
| ML Model | `scikit-learn` RandomForestClassifier (baseline) |
| DP Model | `diffprivlib` DP RandomForestClassifier (IBM) |
| Analytic GM | `scipy` — binary search on normal CDF |
| Data Processing | `pandas`, `numpy` |
| Visualization | `matplotlib` (dark theme, 10 plots) |
| Evaluation | Accuracy ± std (3 trials), AUC-ROC, Confusion Matrix |

---

## 📁 Key Files

| File | Purpose |
|------|---------|
| [`dp_train_test.py`](dp_train_test.py) | Interactive training — baseline + DP RF, multi-trial, composition warnings |
| [`visualize_results.py`](visualize_results.py) | Auto-adaptive visualization engine (10 plots, Analytic GM sigma) |
| [`requirements.txt`](requirements.txt) | All Python dependencies |

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## 👨‍💻 Authors

Developed as a **B.Tech Project** — PSIT (Pranveer Singh Institute of Technology)

> *"Privacy is not about hiding. It's about protecting the individual's right to control their own data."*

---

<div align="center">
  <strong>⭐ Star this repo if you found it useful!</strong>
</div>
