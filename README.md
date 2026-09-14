# 🏥 PrivaCare-AI

> **Differential Privacy for Healthcare AI** — Protecting patient data while maintaining predictive accuracy by comparing 4 DP model+mechanism combinations across multiple epsilon values.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![sklearn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikit-learn&logoColor=white)
![diffprivlib](https://img.shields.io/badge/IBM-diffprivlib-purple)
![Privacy](https://img.shields.io/badge/Privacy-Differential%20Privacy-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Overview

**PrivaCare-AI** is a B.Tech research project that applies **Differential Privacy (DP)** to healthcare machine learning. It compares **4 model + mechanism combinations** across 3 epsilon values (ε = 0.5, 0.7, 1.0), quantifying the **Privacy–Accuracy Trade-off** on wearable health sensor data.

### 🎯 Key Objectives
- Compare **Random Forest** vs **Logistic Regression** under differential privacy
- Compare **Gaussian Mechanism** vs **Laplace Mechanism** for DP noise injection
- Use **data-independent domain bounds** to preserve formal DP guarantees
- Run **3 trials per experiment** (mean ± std reported for reliability)
- Generate rich **visualizations** per model: confusion matrix, ROC curves, feature importance, and more

---

## 🗂️ Project Structure

```
PrivaCare-AI/
│
├── data/
│   └── dataset.csv                        # Master dataset (6,000 rows, 13 features) — CSV
│
├── datasets/                              # Model-specific datasets (JSON format)
│   ├── dataset_1_rf_gaussian.json         # Used by rf_gaussian.py
│   ├── dataset_2_rf_laplace.json          # Used by rf_laplace.py
│   ├── dataset_3_lr_gaussian.json         # Used by lr_gaussian.py
│   └── dataset_4_lr_laplace.json          # Used by lr_laplace.py
│
├── models/                                # 4 DP training scripts
│   ├── rf_gaussian.py                     # Random Forest + Gaussian (diffprivlib)
│   ├── rf_laplace.py                      # Random Forest + Laplace (feature-level)
│   ├── lr_gaussian.py                     # Logistic Regression + Gaussian (feature-level)
│   └── lr_laplace.py                      # Logistic Regression + Laplace (feature-level)
│
├── visualizations/                        # 4 visualization scripts (10 plots each)
│   ├── rf_gaussian.py                     # → saves to results/rf_gaussian/
│   ├── rf_laplace.py                      # → saves to results/rf_laplace/
│   ├── lr_gaussian.py                     # → saves to results/lr_gaussian/
│   └── lr_laplace.py                      # → saves to results/lr_laplace/
│
├── results/                               # Experiment result files
│   ├── rf_gaussian.txt                    # Trial-wise results (3 epsilons)
│   ├── rf_laplace.txt                     # Trial-wise results (3 epsilons)
│   ├── lr_gaussian.txt                    # Trial-wise results (3 epsilons)
│   ├── lr_laplace.txt                     # Trial-wise results (3 epsilons)
│   └── compare_all.txt                    # Combined comparison (all 4 models)
│
├── compare_all.py                         # Live comparison script (all models × epsilons)
├── requirements.txt                       # Python dependencies
├── LICENSE                                # MIT License
└── README.md
```

---

## 🔬 Models & Mechanisms

| # | Model | Mechanism | DP Type | File |
|---|-------|-----------|---------|------|
| 1 | Random Forest | Gaussian (diffprivlib) | (ε, δ)-DP | `models/rf_gaussian.py` |
| 2 | Random Forest | Laplace (feature-level) | pure ε-DP | `models/rf_laplace.py` |
| 3 | Logistic Regression | Gaussian (feature-level) | (ε, δ)-DP | `models/lr_gaussian.py` |
| 4 | Logistic Regression | Laplace (feature-level) | pure ε-DP | `models/lr_laplace.py` |

### Key Difference: Where is noise applied?

- **RF Gaussian (diffprivlib):** Noise at tree-level (split selection + leaf values) — more efficient
- **RF/LR Laplace/Gaussian (feature-level):** Noise added to training features before fitting

---

## ⚙️ Privacy Mechanisms

### 1. Gaussian Mechanism — (ε, δ)-DP
Uses **Analytic Gaussian Mechanism** (Balle & Wang, NeurIPS 2018):

$$\sigma^* = \min \left\{ \sigma : \Phi\!\left(\frac{1}{2\sigma} - \varepsilon\sigma\right) - e^{\varepsilon}\,\Phi\!\left(-\frac{1}{2\sigma} - \varepsilon\sigma\right) \leq \delta \right\}$$

Valid for **all ε > 0** (classical formula only valid for ε < 1).

| ε | δ | σ (Analytic GM) |
|---|---|-----------------|
| 0.5 | 1e-5 | 7.0318 |
| 0.7 | 1e-5 | 5.1665 |
| 1.0 | 1e-5 | 3.7306 |

### 2. Laplace Mechanism — pure ε-DP
$$b = \frac{\text{sensitivity}}{\varepsilon} = \frac{1.0}{\varepsilon}$$

Stronger formal guarantee — **no delta needed**.

| ε | b (Laplace scale) |
|---|------------------|
| 0.5 | 2.0000 |
| 0.7 | 1.4286 |
| 1.0 | 1.0000 |

---

## 📊 Results Summary

### Accuracy (mean ± std, 3 trials per ε)

| Model | Mechanism | ε=0.5 | ε=0.7 | ε=1.0 |
|-------|-----------|-------|-------|-------|
| **Random Forest** | **Laplace / Tree-DP (diffprivlib)** | **93.4% ±2.7%** | **93.9% ±1.9%** | **94.6% ±1.8%** |
| Logistic Regression | Laplace (Objective) | 69.5% ±14.8% | 82.6% ±5.1% | 88.2% ±2.7% |
| Logistic Regression | True Gaussian (Input) | 27.2% ±1.9% | 35.3% ±5.2% | 47.9% ±8.2% |
| Random Forest | True Gaussian (Input) | 25.0% ±0.0% | 28.6% ±7.7% | 22.7% ±5.2% |

> **Baselines (No DP):** RF = 98.92% | LR = 96.17%

### Key Findings
1. **RF Laplace (Tree-DP)** is consistently the best (~94%) because it uses Objective Perturbation (noise at the split level), protecting utility.
2. **LR True Gaussian (~47%)** survives slightly better than RF under Input Perturbation because linear boundaries mathematically average out symmetric noise.
3. **RF True Gaussian (~23%) completely fails** because decision trees split on pure random noise added to the features. This proves that Input Perturbation destroys utility for high-dimensional data.
4. **Conclusion:** The failure of Input Perturbation is a genuine privacy-utility tradeoff, highlighting why advanced mechanisms like Tree-DP are required for clinical health datasets.

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

### Run a Single Model

```bash
# Random Forest + Gaussian (best accuracy)
python models/rf_gaussian.py

# Random Forest + Laplace
python models/rf_laplace.py

# Logistic Regression + Gaussian
python models/lr_gaussian.py

# Logistic Regression + Laplace
python models/lr_laplace.py
```

Each model will prompt for an epsilon value:
```
Enter Epsilon value (e.g. 0.1, 0.5, 1.0) [Default 0.5]: 0.5
```

### Run Full Comparison (All Models)

```bash
python compare_all.py
```

Runs all 4 models × 3 epsilons × 3 trials and prints a complete comparison table.

### Generate Visualizations

```bash
# After running a model, generate its plots
python visualizations/rf_gaussian.py
python visualizations/rf_laplace.py
python visualizations/lr_gaussian.py
python visualizations/lr_laplace.py
```

Each generates 10 plots saved to `results/<model_name>/`.

### View Experiment Results

All results are pre-documented in `results/`:

```bash
# Individual model results (trial-wise breakdown)
results/rf_gaussian.txt
results/rf_laplace.txt
results/lr_gaussian.txt
results/lr_laplace.txt

# Combined comparison table
results/compare_all.txt
```

---

## 🖼️ Visualizations (per model)

| Plot | Description |
|------|-------------|
| `0_DASHBOARD.png` | Master summary dashboard |
| `1_confusion_matrix.png` | Class-wise prediction confusion |
| `2_feature_importance.png` | Feature ranking (RF: importances, LR: coefficients) |
| `3_class_distribution.png` | Target class balance |
| `4_roc_curves.png` | ROC curve per class (One vs Rest, AUC) |
| `5_privacy_tradeoff.png` | ε vs noise parameter curve |
| `6_dp_noise_effect.png` | Original vs DP-noisy feature distributions |
| `7_feature_per_class.png` | Feature distributions per health class |
| `8_correlation_heatmap.png` | Feature correlation matrix |
| `9_confidence.png` | Model prediction confidence per class |

---

## 🧠 Dataset

| Property | Value |
|----------|-------|
| **Source** | Synthetic wearable health sensor data |
| **Samples** | 6,000 patient records |
| **Features** | 13 (heart_rate, blood_oxygen, bp_systolic, bp_diastolic, glucose_level, body_temperature, respiratory_rate, activity_level, sleep_quality, stress_level, hrv_sdnn, steps_count, calories_burned) |
| **Target** | `health_event` — 4 classes (Normal, Mild Risk, Moderate Risk, High Risk) |
| **Balance** | 1,500 samples per class (balanced) |
| **Split** | 80% train / 20% test (stratified) |
| **Master file** | `data/dataset.csv` (CSV) |
| **Model datasets** | `datasets/dataset_N_<model>.json` (JSON, same data) |

---

## 📐 Mathematical Background

### (ε, δ)-Differential Privacy

A randomized mechanism **M** satisfies **(ε, δ)-DP** if for all neighboring datasets **D, D'** and all outputs **S**:

$$\Pr[M(D) \in S] \leq e^{\varepsilon} \cdot \Pr[M(D') \in S] + \delta$$

### Privacy Composition (3 trials)

Each run on the same dataset consumes ε from the total privacy budget:
- **Basic composition:** 3 runs → ε_total = 3 × ε
- **Advanced composition:** ε_total ≈ √3 × ε

### Normalization (Data-Independent)

All features are clipped and scaled using clinical **DOMAIN_BOUNDS** (not data statistics), ensuring L∞-sensitivity = 1.0 without leaking any information from the dataset.

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.10+ |
| ML Models | `scikit-learn` (RF, LR) |
| DP RF | `diffprivlib` (IBM) — tree-level DP |
| Analytic GM | `scipy` — binary search on normal CDF |
| Data Processing | `pandas`, `numpy` |
| Visualization | `matplotlib` (dark theme, 10 plots per model) |
| Evaluation | Accuracy ± std, Macro F1, Macro Recall, AUC-ROC |

---

## 🔮 Planned Extensions

| Feature | Status |
|---------|--------|
| IoT Sensor Integration | 🔵 Planned |
| Blockchain Audit Trail | 🔵 Planned |
| PATE Framework | 🔵 Planned |
| Federated Learning | 🔵 Planned |

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
