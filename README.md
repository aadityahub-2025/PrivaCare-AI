# 🏥 PrivaCare-AI

> **Differential Privacy for Healthcare AI** — Protecting patient data with Gaussian Mechanism while maintaining predictive accuracy using Random Forest classifiers.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![sklearn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikit-learn&logoColor=white)
![Privacy](https://img.shields.io/badge/Privacy-Differential%20Privacy-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Overview

**PrivaCare-AI** is a B.Tech research project that applies **Differential Privacy (DP)** to healthcare machine learning pipelines. It demonstrates how Gaussian noise can be injected into training data to provide provable privacy guarantees — preventing adversaries from inferring individual patient records — while quantifying the resulting accuracy trade-off.

### 🎯 Key Objectives
- Apply **(ε, δ)-Differential Privacy** using the **Gaussian Mechanism** to wearable health sensor data
- Train and compare **baseline vs. DP-protected** Random Forest classifiers
- Analyze the **Privacy-Accuracy Trade-off** across a range of epsilon values (ε = 0.1 → 50)
- Generate rich visualizations of model performance, feature importance, and noise effects

---

## 🗂️ Project Structure

```
PrivaCare-AI/
│
├── data/
│   ├── dataset.csv                        # Primary health dataset (6000 samples)
│   ├── dataset_original_backup.csv        # Backup of original dataset
│   └── realistic_dataset.csv              # Realistic synthetic health data
│
├── models/
│   ├── dp_rf_model.pkl                    # Saved DP Random Forest model
│   ├── label_encoder.pkl                  # Fitted label encoder
│   └── scaler.pkl                         # Fitted MinMaxScaler
│
├── results/
│   ├── training_results.json              # Full metrics from last training run
│   ├── epsilon_sweep.json                 # Results across all epsilon values
│   └── plots/                             # Generated visualizations (14 plots)
│       ├── 0_DASHBOARD.png
│       ├── 1_confusion_matrix.png
│       ├── 2_feature_importance.png
│       ├── 4_roc_curves.png
│       ├── 5_privacy_tradeoff.png
│       └── ...
│
├── privacy_engine.py                      # Core Gaussian DP noise function
├── run_experiment.py                      # MLP baseline + DP experiment runner
├── dp_train_test.py                       # Random Forest DP trainer (interactive)
├── epsilon_sweep.py                       # Sweep ε values, save comparison results
├── visualize_results.py                   # Generate all result plots
├── health_data_balanced_after_overfitting.xlsx   # Balanced dataset workbook
├── .gitignore
└── README.md
```

---

## ⚙️ How It Works

### Privacy Mechanism — Gaussian Noise

For each feature in the normalized training set (clipped to **[0, 1]**), Gaussian noise is sampled from:

$$\mathcal{N}(0,\ \sigma^2)$$

where the noise scale **σ** is derived from the privacy budget:

$$\sigma = \frac{\Delta f \cdot \sqrt{2 \ln(1.25 / \delta)}}{\varepsilon}$$

| Symbol | Meaning |
|--------|---------|
| **ε (epsilon)** | Privacy budget — smaller = more private |
| **δ (delta)** | Failure probability (fixed at `1e-5`) |
| **Δf** | Sensitivity (= 1.0 for [0,1]-normalized features) |
| **σ** | Standard deviation of injected noise |

This provides a **formal (ε, δ)-DP guarantee**: an adversary with any side information cannot identify any single patient's record with probability exceeding `e^ε + δ`.

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
    ├──── Baseline RF (no noise) ──────► Baseline Accuracy
    │
    └──── Add Gaussian DP Noise ───────► DP Random Forest ──► DP Accuracy
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install numpy pandas scikit-learn matplotlib seaborn openpyxl
```

> Python **3.10+** recommended.

### Clone the Repository

```bash
git clone https://github.com/aadityahub-2025/PrivaCare-AI.git
cd PrivaCare-AI
```

---

## 🧪 Usage

### 1. Quick DP Training (Interactive)

Trains a baseline RF and a DP-protected RF. Prompts for epsilon at runtime.

```bash
python dp_train_test.py
```

**Example output:**
```
--> Dataset loaded: 6000 rows, 13 features
    Target: 'health_event' | Classes: 4

Enter Epsilon value (e.g. 0.1, 0.5, 1.0) [Default 0.5]: 1.0
--> Epsilon = 1.0 | Delta = 1e-05

[1] Training Baseline Random Forest (No DP Noise)...
--> Baseline Accuracy (Without Privacy): 99.92%

[2] Applying Gaussian DP Noise (epsilon=1.0, sigma=4.8448)...
[3] Training DP Random Forest (With Gaussian Noise)...
--> DP Model Accuracy (With Privacy, epsilon=1.0): XX.XX%

================ RESULT SUMMARY ================
Original Model Accuracy (No Privacy) : 99.92%
Gaussian DP Model Accuracy           : XX.XX%
Accuracy Drop (Privacy Cost)         : XX.XX%
================================================
```

---

### 2. Epsilon Sweep (Privacy vs. Accuracy Analysis)

Tests DP across **8 epsilon values** (`0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, No DP`) with 5-fold cross-validation.

```bash
python epsilon_sweep.py
```

Results are saved to `results/epsilon_sweep.json`.

---

### 3. MLP Baseline Experiment

Runs an MLP neural network experiment with the privacy engine.

```bash
python run_experiment.py
```

---

### 4. Generate All Visualizations

```bash
python visualize_results.py
```

Saves **14 plots** to `results/plots/`.

---

## 📊 Results

### Performance Summary (ε = 0.5, δ = 1e-5)

| Metric | Baseline (No DP) | DP Model (ε=0.5) |
|--------|:----------------:|:----------------:|
| **Test Accuracy** | 99.92% | 23.17% |
| **AUC-ROC** | 1.0000 | 0.5174 |
| **Privacy Guarantee** | ❌ None | ✅ (0.5, 1e-5)-DP |
| **Noise σ** | 0.0 | 9.6896 |

> **Note:** At ε = 0.5 (very high privacy), noise dominates the signal. Higher ε values (e.g. ε = 10–50) yield much better accuracy while still providing meaningful privacy protection.

### Privacy Level Classification

| Epsilon (ε) | Privacy Level | Noise σ |
|:-----------:|:-------------:|:-------:|
| ≤ 0.5 | 🔒 Very High | Large |
| 0.5 – 2.0 | 🔐 High | Moderate |
| 2.0 – 7.0 | 🔑 Moderate | Small |
| > 7.0 | 🔓 Low | Minimal |

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

| Plot | Description |
|------|-------------|
| `0_DASHBOARD.png` | Full summary dashboard |
| `1_confusion_matrix.png` | Class-wise prediction confusion |
| `2_feature_importance.png` | RF feature ranking |
| `3_disease_distribution.png` | Target class balance |
| `4_roc_curves.png` | ROC curve per class |
| `5_privacy_tradeoff.png` | Accuracy vs. Epsilon curve |
| `6_dp_noise_effect.png` | Effect of noise on features |
| `7_symptom_pattern.png` | Feature patterns per disease class |
| `8_correlation_heatmap.png` | Feature correlation matrix |
| `9_confidence_dist.png` | Model prediction confidence |

---

## 🧠 Dataset

- **Source:** Synthetic wearable health sensor data
- **Samples:** 6,000 patient records
- **Features (13):** `heart_rate`, `blood_oxygen`, `blood_pressure_systolic`, `blood_pressure_diastolic`, `glucose_level`, `body_temperature`, `respiratory_rate`, `activity_level`, `sleep_quality`, `stress_level`, `hrv_sdnn`, `steps_count`, `calories_burned`
- **Target:** `health_event` — 4 disease classes (balanced)
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
|-----------|-----------|
| Language | Python 3.10+ |
| ML Model | `scikit-learn` RandomForestClassifier |
| Privacy | Custom Gaussian Mechanism (`privacy_engine.py`) |
| Data Processing | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn` |
| Evaluation | Accuracy, AUC-ROC, 5-fold CV |

---

## 📁 Key Files

| File | Purpose |
|------|---------|
| [`privacy_engine.py`](privacy_engine.py) | Core DP noise injection function |
| [`dp_train_test.py`](dp_train_test.py) | Interactive RF training with DP |
| [`epsilon_sweep.py`](epsilon_sweep.py) | Sweep multiple ε values, compare results |
| [`run_experiment.py`](run_experiment.py) | MLP + DP baseline experiment |
| [`visualize_results.py`](visualize_results.py) | Generate all 14 result plots |

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
