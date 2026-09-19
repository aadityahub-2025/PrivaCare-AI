# 🏥 PrivaCare-AI

> **Differential Privacy for Healthcare AI** — Protecting patient data while maintaining predictive clinical accuracy by comparing 4 DP model+mechanism combinations across multiple epsilon values.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![sklearn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikit-learn&logoColor=white)
![diffprivlib](https://img.shields.io/badge/IBM-diffprivlib%200.6.6-purple)
![Privacy](https://img.shields.io/badge/Privacy-Differential%20Privacy-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Overview

**PrivaCare-AI** is a research framework that applies **Differential Privacy (DP)** to healthcare machine learning classification. It compares **4 distinct model + mechanism combinations** across 3 privacy budget regimes ($\epsilon \in \{0.5, 0.7, 1.0\}$), quantifying the **Privacy–Utility Trade-off** on physiological sensor features.

### 🎯 Key Objectives
- Compare **Tree-based DP** vs **Objective Perturbation** vs **Output Perturbation** vs **Sufficient Statistics Perturbation**.
- Evaluate exact **pure $\epsilon$-DP** (Laplace / PermuteAndFlip) vs approximate $(\epsilon, \delta)$-DP (Analytic Gaussian Mechanism).
- Enforce **data-independent domain bounds** and bias regularization to preserve formal mathematical DP guarantees.
- Conduct multi-trial evaluation across **independent synthetic benchmark replicates** (80,000 samples per replicate).
- Generate publication-ready visualizations: tradeoff curves, model comparisons, privacy cost bars, and confusion matrices.

---

## 🗂️ Project Structure

```
PrivaCare-AI/
│
├── data/
│   └── dataset.csv                        # Original reference dataset (6,000 rows)
│
├── datasets/                              # Independent synthetic benchmark replicates (80k each)
│   ├── dataset_1_rf_gaussian.json         # Replicate 1 (seed=41) -> models/rf_gaussian.py
│   ├── dataset_2_rf_laplace.json          # Replicate 2 (seed=42) -> models/rf_laplace.py
│   ├── dataset_3_lr_gaussian.json         # Replicate 3 (seed=43) -> models/lr_gaussian.py
│   └── dataset_4_lr_laplace.json          # Replicate 4 (seed=44) -> models/lr_laplace.py
│
├── models/                                # 4 Dedicated DP Training Scripts
│   ├── rf_gaussian.py                     # Gaussian Naive Bayes (Sufficient Stats DP, pure ε-DP)
│   ├── rf_laplace.py                      # Random Forest (Tree-based DP, pure ε-DP)
│   ├── lr_gaussian.py                     # Logistic Regression (Analytic Gaussian Output DP, (ε,δ)-DP)
│   └── lr_laplace.py                      # Logistic Regression (Objective Perturbation DP, pure ε-DP)
│
├── visualizations/                        # Research plots & confusion matrices (PNG)
│   ├── fig1_privacy_utility_curve.png     # Privacy vs Utility Tradeoff Curve
│   ├── fig2_model_comparison.png          # Model Comparison at Strict Budget (ε=0.5)
│   ├── fig3_privacy_cost.png              # Accuracy Drop from Baseline
│   └── fig4_confusion_matrices.png        # 2x2 Grid of Confusion Matrices (ε=0.5)
│
├── results/
│   └── compare_all.txt                    # Comprehensive benchmark report
│
├── compare_all.py                         # Live comparative evaluation runner
├── generate_research_plots.py             # Script to generate publication-grade figures
├── generate_confusion_matrices.py         # Script to generate 2x2 confusion matrix grid
├── requirements.txt                       # Python dependencies (pinned diffprivlib)
├── PROJECT_STORY.txt                      # Comprehensive narrative, architecture & research notes
├── LICENSE                                # MIT License
└── README.md
```

---

## 🔬 Models & Privacy Mechanisms

| # | Script | Model Architecture | Privacy Mechanism | Guarantee | Privacy Budget |
|---|--------|-------------------|-------------------|-----------|----------------|
| 1 | `models/rf_gaussian.py` | Gaussian Naive Bayes | Sufficient Statistics Perturbation | Pure $\epsilon$-DP ($\delta=0$) | $\epsilon \in \{0.5, 0.7, 1.0\}$ |
| 2 | `models/rf_laplace.py` | Random Forest (20 trees) | Tree-based DP (Random splits + PermuteAndFlip) | Pure $\epsilon$-DP ($\delta=0$) | $\epsilon \in \{0.5, 0.7, 1.0\}$ |
| 3 | `models/lr_gaussian.py` | Logistic Regression ($C=0.02$) | Analytic Gaussian Output Perturbation ($\Delta_2 = 2\sqrt{2}C$) | $(\epsilon, \delta)$-DP ($\delta=10^{-5}$) | $\epsilon \in \{0.5, 0.7, 1.0\}$ |
| 4 | `models/lr_laplace.py` | Logistic Regression ($C=1.0$) | Objective Perturbation (diffprivlib) | Pure $\epsilon$-DP ($\delta=0$) | $\epsilon \in \{0.5, 0.7, 1.0\}$ |

### Theoretical Mechanism Details:
1. **Gaussian Naive Bayes (`rf_gaussian.py`):**
   - Gaussian Naive Bayes models class-conditional Gaussian distributions.
   - IBM `diffprivlib` perturbs the *sufficient statistics* (class counts, feature sums, and feature variances) with Laplace noise.
   - Formal guarantee is **pure $\epsilon$-DP** ($\delta=0$).
2. **Random Forest (`rf_laplace.py`):**
   - Construct decision trees using data-independent uniform random split candidate thresholds within domain bounds.
   - The `PermuteAndFlip` mechanism is applied to choose noisy leaf class predictions.
   - Pure $\epsilon$-DP guarantee without requiring $\delta$.
3. **Logistic Regression with Output Perturbation (`lr_gaussian.py`):**
   - Input vectors are augmented with a constant bias feature and normalized such that $\|\tilde{x}\|_2 \le 1$ strictly:
     $$\tilde{x} = \frac{[x_{\text{norm}}, 1.0]}{\sqrt{d + 1}}$$
   - Model is trained with $L_2$ regularization parameter $C=0.02$ and `fit_intercept=False`.
   - The loss Lipschitz constant is $L = \sqrt{2}$.
   - Under single-sample replacement, the exact $L_2$ sensitivity of the optimal weight matrix $W^*$ is:
     $$\Delta_2(W^*) = 2 \cdot L \cdot C = 2\sqrt{2} \cdot C \approx 0.056569$$
   - Noise $\sigma$ is calibrated using the exact **Analytic Gaussian Mechanism** (Balle & Wang, NeurIPS 2018) for $(\epsilon, \delta=10^{-5})$.
4. **Logistic Regression with Objective Perturbation (`lr_laplace.py`):**
   - Perturbs the continuous optimization objective directly during gradient descent.
   - Pure $\epsilon$-DP guarantee.

---

## 📊 Benchmark Results

Evaluated across $N=5$ trials per configuration with stratified 80/20 train/test splits on 80,000-sample benchmark cohorts:

### Accuracy Comparison Across Privacy Budgets

| Model | DP Mechanism | Formal Guarantee | Baseline (No DP) | $\epsilon=0.5$ (High Privacy) | $\epsilon=0.7$ (Moderate) | $\epsilon=1.0$ (Balanced) |
|-------|--------------|------------------|------------------|-------------------------------|---------------------------|---------------------------|
| **Gaussian Naive Bayes** | Sufficient Statistics | Pure $\epsilon$-DP | 99.88% | 93.12% ± 5.57% | 97.30% ± 3.69% | 99.77% ± 0.10% |
| **Random Forest** | Tree-based DP | Pure $\epsilon$-DP | 99.66% | 99.21% ± 0.21% | 99.22% ± 0.24% | 99.19% ± 0.27% |
| **Logistic Regression** | Output Perturbation | $(\epsilon, \delta)$-DP | 99.19% | 98.66% ± 0.76% | 99.15% ± 0.26% | 99.32% ± 0.17% |
| **Logistic Regression** | Objective Perturbation | Pure $\epsilon$-DP | 99.83% | 97.67% ± 0.54% | 98.02% ± 0.42% | 98.32% ± 0.35% |

### Macro F1-Scores

| Model | Mechanism | $\epsilon=0.5$ | $\epsilon=0.7$ | $\epsilon=1.0$ |
|-------|-----------|----------------|----------------|----------------|
| **Gaussian Naive Bayes** | Sufficient Statistics | 0.9253 | 0.9715 | 0.9977 |
| **Random Forest** | Tree-based DP | 0.9920 | 0.9922 | 0.9918 |
| **Logistic Regression** | Output Perturbation | 0.9866 | 0.9915 | 0.9932 |
| **Logistic Regression** | Objective Perturbation | 0.9767 | 0.9801 | 0.9832 |

---

## 🔍 Research Integrity, Data Design & Limitations

When evaluating and reporting this work in academic papers or presentations, the following methodology details and limitations should be explicitly disclosed:

1. **Synthetic Benchmark Replicates vs Real Hospital Cohorts:**
   - The 4 JSON datasets (`dataset_1` to `dataset_4`, 80,000 rows each) are **controlled synthetic benchmark replicates** generated with physiological parameters (means and standard deviations derived from clinical literature for Healthy, Pre-diabetic, Hypertensive, and Metabolic Risk classes).
   - They serve to test the mathematical resilience and variance of DP mechanisms under identical underlying distributions without confounding noise levels.
   - Because they are generated synthetically, the empirical DP guarantee applies to the synthetic participant records within the benchmark.
2. **Original Reference Data (`dataset.csv`) Limitations:**
   - In the initial Kaggle wearable reference dataset (6,000 rows), approximately 61% of total records and ~80% of rows for classes 1–3 contain synthetic markers.
   - The original dataset contains 4,233 unique patient IDs across 6,000 rows, meaning some patients contributed up to 12 measurements. In formal DP terms, standard record-level DP protects individual time-slice observations, whereas user-level DP would require grouping or bounding contributions per patient ID.
3. **Hyperparameter Selection Privacy Accounting:**
   - In standard differential privacy literature, tuning hyperparameters (such as $C$, tree depth, and number of estimators) on the training set technically consumes privacy budget unless tuned on a disjoint public validation set or accounted for via private selection algorithms (e.g., Report Noisy Max).
4. **Composition Over Multiple Experiments:**
   - Running $k$ trials on the same training cohort consumes privacy according to DP composition theorems:
     - Basic composition: $\epsilon_{\text{total}} = k \cdot \epsilon$.
     - Advanced composition (Dwork et al.): $\epsilon_{\text{total}} = \sqrt{2k\ln(1/\delta')}\epsilon + k\epsilon(e^\epsilon - 1)$ with total failure probability $k\delta + \delta'$.

---

## 🚀 Getting Started

### Installation

```bash
git clone https://github.com/aadityahub-2025/PrivaCare-AI.git
cd PrivaCare-AI
pip install -r requirements.txt
```

Pinned dependencies:
- `python>=3.10`
- `scikit-learn>=1.2,<1.5.0`
- `diffprivlib>=0.6.0,<=0.6.6`
- `scipy>=1.9`
- `matplotlib>=3.6`
- `pandas>=1.5`
- `numpy>=1.23`

### Running Individual Models

```bash
# 1. Gaussian Naive Bayes (Sufficient Statistics DP)
python models/rf_gaussian.py

# 2. Random Forest (Tree-based DP)
python models/rf_laplace.py

# 3. Logistic Regression (Analytic Gaussian Output DP)
python models/lr_gaussian.py

# 4. Logistic Regression (Objective Perturbation DP)
python models/lr_laplace.py
```

### Running Full Benchmark & Plots

```bash
# Run comprehensive benchmark across all 4 models and epsilons [0.5, 0.7, 1.0]
python compare_all.py

# Generate publication-grade tradeoff curves & bar charts
python generate_research_plots.py

# Generate 2x2 confusion matrix grid
python generate_confusion_matrices.py
```

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](file:///c:/Users/DELL/Desktop/psit/projects%20btech/PrivaCare-AI/LICENSE) file for details.
