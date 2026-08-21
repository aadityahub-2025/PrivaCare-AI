# 🏥 PrivaCare-AI

> **Differential Privacy for Healthcare AI** — Protecting patient data with IBM's `diffprivlib` Gaussian Mechanism while maintaining predictive accuracy using Random Forest classifiers.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![sklearn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikit-learn&logoColor=white)
![diffprivlib](https://img.shields.io/badge/IBM-diffprivlib-purple)
![Privacy](https://img.shields.io/badge/Privacy-Differential%20Privacy-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Overview

**PrivaCare-AI** is a B.Tech research project that applies **Differential Privacy (DP)** to healthcare machine learning pipelines. It uses IBM's `diffprivlib` library to train a DP-protected Random Forest directly — providing algorithmic, model-level privacy guarantees — and compares it against a baseline (non-private) model. The project quantifies the **Privacy–Accuracy Trade-off** at different epsilon values.

### 🎯 Key Objectives
- Apply **(ε, δ)-Differential Privacy** using IBM `diffprivlib` on wearable health sensor data
- Train and compare **baseline vs. DP-protected** Random Forest classifiers
- Analyze the **Privacy-Accuracy Trade-off** across epsilon values (ε = 0.1 → 50)
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
├── .gitignore
└── README.md
```

---

## ⚙️ How It Works

### Privacy Mechanism — diffprivlib DP Random Forest

Unlike manual noise injection, this project uses **IBM's `diffprivlib`** which implements privacy at the **algorithmic level** inside the Random Forest training process.

The noise scale **σ** is internally derived from the privacy budget:

$$\sigma = \frac{\Delta f \cdot \sqrt{2 \ln(1.25 / \delta)}}{\varepsilon}$$

| Symbol | Meaning |
|--------|---------|
| **ε (epsilon)** | Privacy budget — smaller = more private |
| **δ (delta)** | Failure probability (fixed at `1e-5`) |
| **Δf** | Sensitivity (= 1.0 for [0,1]-normalized features) |
| **σ** | Standard deviation of injected noise |

This provides a formal **(ε, δ)-DP guarantee**: an adversary cannot identify any individual patient's record from the model output.

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
MinMax Normalization → [0, 1]
    │
    ├──── Baseline RF (sklearn, no privacy) ──────► Baseline Accuracy
    │
    └──── DP Random Forest (diffprivlib, ε, δ) ───► DP Accuracy
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install numpy pandas scikit-learn matplotlib diffprivlib
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

Trains a baseline RF and a DP-protected RF. Prompts for epsilon at runtime.

```bash
python dp_train_test.py
```

**Example output:**
```
--> Dataset loaded: 6000 rows, 13 features
    Target: 'health_event' | Classes: 4

Enter Epsilon value (e.g. 0.1, 0.5, 1.0) [Default 0.5]: 0.5
--> Epsilon = 0.5

[1] Training Baseline Random Forest (No DP Noise)...
[2] Training DP Random Forest (Algorithmic Privacy, epsilon=0.5)...

================ RESULT SUMMARY ================
Original Model Accuracy (No Privacy) : 99.83%
DP Model Accuracy (Epsilon=0.5)      : 96.50%
Accuracy Drop (Privacy Cost)         : 3.33%
================================================
```

---

### Step 2 — Generate Visualizations

```bash
python visualize_results.py
```

Enter the **same epsilon** as Step 1. Saves **10 plots** to `results/plots/`.

---

## 📊 Results

### Performance Summary

| ε (Epsilon) | Baseline Accuracy | DP Accuracy | Accuracy Drop | Privacy Level |
|:-----------:|:-----------------:|:-----------:|:-------------:|:-------------:|
| **0.5** | 99.83% | **96.50%** | 3.33% | 🔒 High |
| **1.0** | 99.83% | **93.58%** | 6.25% | 🔐 Moderate-High |

> **Key insight:** At ε = 0.5, the DP model achieves **96.50% accuracy** — only a 3.33% drop from baseline — while providing strong formal privacy guarantees. This shows diffprivlib's algorithmic DP is far more efficient than manual noise injection.

### Privacy Level Classification

| Epsilon (ε) | Privacy Level | Typical Use Case |
|:-----------:|:-------------:|:----------------:|
| ≤ 0.5 | 🔒 High Privacy | Highly sensitive medical data |
| 0.5 – 2.0 | 🔐 Moderate-High | Clinical research |
| 2.0 – 7.0 | 🔑 Moderate | General health analytics |
| > 7.0 | 🔓 Low Privacy | Non-sensitive aggregates |

### Feature Importance (Baseline RF)

| Feature | Importance |
|---------|:----------:|
| Activity Level | 8.52% |
| HRV SDNN | 8.34% |
| Steps Count | 8.13% |
| Body Temperature | 7.93% |
| Blood Pressure (Diastolic) | 7.85% |
| Heart Rate | 7.64% |
| Glucose Level | 7.62% |
| Blood Oxygen | 7.47% |
| Calories Burned | 7.50% |

---

## 🖼️ Visualizations

All plots are saved in `results/plots/` after running `visualize_results.py`.

| Plot File | Description |
|-----------|-------------|
| `0_DASHBOARD.png` | Master summary dashboard (all metrics in one view) |
| `1_confusion_matrix.png` | Class-wise prediction confusion (with % annotations) |
| `2_feature_importance.png` | Baseline RF feature ranking |
| `3_class_distribution.png` | Target class balance — **Normal / Mild Risk / Moderate Risk / High Risk** |
| `4_roc_curves.png` | ROC curve per class (One vs Rest, with AUC) |
| `5_privacy_tradeoff.png` | ε vs σ relationship + Gaussian noise distributions |
| `6_dp_noise_effect.png` | Original vs DP-noisy feature distributions |
| `7_feature_per_class.png` | Feature distributions separated by health class |
| `8_correlation_heatmap.png` | Feature correlation matrix |
| `9_confidence.png` | Model prediction confidence distribution per class |

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

---

## 📐 Mathematical Background

### (ε, δ)-Differential Privacy

A randomized mechanism **M** satisfies **(ε, δ)-DP** if for all neighboring datasets **D**, **D'** (differing in one record), and all outputs **S**:

$$\Pr[M(D) \in S] \leq e^{\varepsilon} \cdot \Pr[M(D') \in S] + \delta$$

The **Gaussian Mechanism** achieves this by adding noise calibrated to the **L2-sensitivity** of the query function, making it ideal for continuous-valued feature vectors in ML pipelines.

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.10+ |
| ML Model | `scikit-learn` RandomForestClassifier (baseline) |
| DP Model | `diffprivlib` DP RandomForestClassifier (IBM) |
| Data Processing | `pandas`, `numpy` |
| Visualization | `matplotlib` (dark theme, 10 plots) |
| Evaluation | Accuracy, AUC-ROC, Confusion Matrix |

---

## 📁 Key Files

| File | Purpose |
|------|---------|
| [`dp_train_test.py`](dp_train_test.py) | Interactive training — baseline + DP RF with diffprivlib |
| [`visualize_results.py`](visualize_results.py) | Auto-adaptive visualization engine (10 plots) |

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
