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
- Evaluate exact **pure $\epsilon$-DP** (Laplace / PermuteAndFlip / Vector Perturbation) vs approximate $(\epsilon, \delta)$-DP (Analytic Gaussian Mechanism).
- Enforce **data-independent domain bounds** and bias regularization to preserve formal mathematical DP guarantees.
- Conduct multi-trial evaluation across **independent synthetic benchmark replicates** (80,000 samples per replicate, $N=30$ trials).
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
│   ├── fig1_privacy_utility_curve.png     # Privacy vs Utility Tradeoff Curve (N=30)
│   ├── fig2_model_comparison.png          # Model Comparison at Strict Budget (ε=0.5, N=30)
│   ├── fig3_privacy_cost.png              # Accuracy Drop from Baseline (N=30)
│   └── fig4_confusion_matrices.png        # 2x2 Grid of Confusion Matrices (ε=0.5)
│
├── results/
│   └── compare_all.txt                    # Comprehensive benchmark report (N=30, UTF-8)
│
├── generate_datasets.py                   # Deterministic generator for all 4 benchmark replicates
├── compare_all.py                         # Live comparative evaluation runner (N=30 trials)
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
| 3 | `models/lr_gaussian.py` | Logistic Regression ($C=0.02$) | Analytic Gaussian Output Perturbation ($\Delta_2 \le 2\sqrt{2}C$) | $(\epsilon, \delta)$-DP ($\delta=10^{-5}$) | $\epsilon \in \{0.5, 0.7, 1.0\}$ |
| 4 | `models/lr_laplace.py` | Logistic Regression ($C=1.0$) | Objective Perturbation (diffprivlib Vector DP) | Pure $\epsilon$-DP ($\delta=0$) | $\epsilon \in \{0.5, 0.7, 1.0\}$ |

### Theoretical Mechanism Details & Terminology Note:
- **Nomenclature Clarification:** Rather than a simplistic "Gaussian vs Laplace" contrast, the framework evaluates **$(\epsilon, \delta)$ Analytic Gaussian Output Perturbation vs Pure-$\epsilon$ In-Training / Perturbation Mechanisms**.
  - `rf_gaussian.py` is a historical filename; it implements Gaussian Naive Bayes where the name refers to the model's distributional assumption (class features are normal), while the privacy mechanism perturbs sufficient statistics via the Laplace mechanism (pure $\epsilon$-DP, $\delta=0$).
  - `rf_laplace.py` uses IBM diffprivlib's `RandomForestClassifier`, which achieves pure $\epsilon$-DP through random candidate split selection and the `PermuteAndFlip` mechanism on leaf nodes.
  - `lr_laplace.py` uses IBM diffprivlib's `LogisticRegression`, which implements the Objective Perturbation / Vector Perturbation mechanism (Chaudhuri et al., 2011; Zhang et al., 2012).
  - `lr_gaussian.py` implements True Analytic Gaussian Output Perturbation (Chaudhuri et al., 2011; Balle & Wang, 2018).

### Mathematical Sensitivity of Logistic Regression Output Perturbation:
1. Input vectors are augmented with a constant bias feature and normalized such that $\|\tilde{x}\|_2 \le 1$ strictly:
   $$\tilde{x} = \frac{[x_{\text{norm}}, 1.0]}{\sqrt{d + 1}}$$
2. Model is trained with $L_2$ regularizer $C=0.02$ and `fit_intercept=False`:
   $$\min_W \frac{1}{2} \|W\|_F^2 + C \sum_{i=1}^n \ell(W; \tilde{x}_i, y_i)$$
3. The multinomial cross-entropy loss is $L$-Lipschitz with $L = \sqrt{2}$.
4. Under single-sample replacement, the **upper bound on the $L_2$ sensitivity** of the optimal weight matrix $W^*$ is:
   $$\Delta_2(W^*) \le \frac{2 L C}{\lambda} = 2\sqrt{2} \cdot C \approx 0.056569$$
5. Gaussian noise $\sigma$ is calibrated using the exact **Analytic Gaussian Mechanism** (Balle & Wang, NeurIPS 2018) for $(\epsilon, \delta=10^{-5})$.

---

## 📊 Benchmark Results ($N=30$ Trials)

Evaluated across $N=30$ trials per configuration with stratified 80/20 train/test splits on 80,000-sample benchmark replicates:

### Accuracy Comparison Across Privacy Budgets

| Model | DP Mechanism | Formal Guarantee | Baseline (No DP) | $\epsilon=0.5$ (High Privacy) | $\epsilon=0.7$ (Moderate) | $\epsilon=1.0$ (Balanced) |
|-------|--------------|------------------|------------------|-------------------------------|---------------------------|---------------------------|
| **Gaussian Naive Bayes** | Sufficient Statistics | Pure $\epsilon$-DP | 99.73% | 97.37% ± 3.51% | 97.73% ± 2.93% | 98.11% ± 3.71% |
| **Random Forest** | Tree-based DP | Pure $\epsilon$-DP | 99.48% | 98.46% ± 0.79% | 98.48% ± 0.79% | 98.50% ± 0.80% |
| **Logistic Regression** | Output Perturbation | $(\epsilon, \delta)$-DP | 97.99% (Weak Reg: 99.50%) | 91.55% ± 6.56% | 94.83% ± 4.11% | 96.51% ± 2.55% |
| **Logistic Regression** | Objective Perturbation | Pure $\epsilon$-DP | 99.64% | 96.18% ± 1.20% | 96.72% ± 0.97% | 97.18% ± 0.77% |

### Macro F1-Scores

| Model | Mechanism | $\epsilon=0.5$ | $\epsilon=0.7$ | $\epsilon=1.0$ |
|-------|-----------|----------------|----------------|----------------|
| **Gaussian Naive Bayes** | Sufficient Statistics | 0.9726 | 0.9767 | 0.9798 |
| **Random Forest** | Tree-based DP | 0.9846 | 0.9848 | 0.9850 |
| **Logistic Regression** | Output Perturbation | 0.9088 | 0.9462 | 0.9643 |
| **Logistic Regression** | Objective Perturbation | 0.9615 | 0.9670 | 0.9716 |

### Detailed Findings:
1. **Output Perturbation vs Budget:** Logistic Regression under Output Perturbation demonstrates a clear, statistically robust privacy-utility tradeoff over 30 trials: accuracy smoothly ascends from $91.55\% \pm 6.56\%$ at $\epsilon=0.5$ to $94.83\% \pm 4.11\%$ at $\epsilon=0.7$ and $96.51\% \pm 2.55\%$ at $\epsilon=1.0$, remaining strictly bounded beneath the non-private baselines (97.99% regularized $C=0.02$, 99.50% weakly regularized $C=10.0$).
2. **Random Forest Noise Resilience & Ultra-Strict Sweep:** At $n=64,000$ training rows across 4 separable clinical clusters, each tree leaf aggregates thousands of samples. As a result, diffprivlib's `PermuteAndFlip` mechanism selects majority class labels with near-certainty across $\epsilon \in [0.5, 1.0]$. The tradeoff curve for RF becomes apparent under ultra-strict budgets evaluated via [`scripts/sweep_rf_epsilon.py`](scripts/sweep_rf_epsilon.py): $\epsilon=0.10 \to 98.19\% \pm 0.97\%$, $\epsilon=0.05 \to 98.13\% \pm 0.94\%$ (median 98.42%), and $\epsilon=0.01 \to 96.15\% \pm 1.83\%$ (median 96.62%).
3. **Naive Bayes Skewed / Heavy-Tailed Variance:** Sufficient statistics perturbation on per-class feature counts and variances exhibits a heavy-tailed / skewed utility distribution. While median accuracy is consistently high ($98.97\%$ at $\epsilon=0.5$ and $99.49\%$ at $\epsilon=1.0$), Laplace noise on small variance estimates causes a small fraction of trials (3–5 out of 30) to drop below 95% (worst-case min $82.84\% - 85.61\%$), maintaining standard deviation at ~3.5%.

---

## 🔍 Research Integrity, Data Design & Disclosures

When referencing or evaluating this work in research papers, the following methodology details and limitations must be explicitly cited:

### 1. Clinical Distribution Citations & Benchmark Design:
The 4 benchmark replicates (`dataset_1` to `dataset_4`, 80,000 rows each) were generated using [`generate_datasets.py`](generate_datasets.py) with class profiles derived directly from clinical diagnostic guidelines:
- **Glucose Ranges:** American Diabetes Association (ADA). "Classification and Diagnosis of Diabetes: Standards of Care in Diabetes—2024." *Diabetes Care* 47 (Suppl. 1), 2024: S20–S42. (Fasting normal < 100 mg/dL; Prediabetes 100–125 mg/dL, Class 1 centered at 112.0 mg/dL; Diabetes $\ge$ 126 mg/dL).
- **Stress Scale:** Perceived Stress Scale (PSS, Cohen et al., 1983) mapped and normalized to a unit interval $[0.0, 1.0]$ alongside wearable autonomic stress composite indices (Cohen, S., Kamarck, T., & Mermelstein, R., *J Health Soc Behav* 24(4), 1983: 385–396; operational benchmark centers: Low: 0.20, Moderate: 0.44, Elevated: 0.66, High Strain: 0.79).
- **Blood Pressure Ranges:** Whelton, P.K., et al. "2017 ACC/AHA/AAPA/ABC/ACPM/AGS/APhA/ASH/ASPC/NMA/PCNA Guideline for Prevention, Detection, Evaluation, and Management of High Blood Pressure in Adults." *J Am Coll Cardiol* 71(19), 2018: e127–e248. (Normal systolic < 120 mmHg; Stage 1 130–139 mmHg; Stage 2 $\ge$ 140 mmHg).
- **Resting Heart Rate:** Clinical consensus on normal resting heart rate (60–100 BPM).
- **Crucial Disclosure:** Class centers were designed directly from published clinical guidelines and were **not fitted to `data/dataset.csv`**, where raw class feature centers materially deviate from standard clinical reference distributions. The datasets are synthetic benchmark replicates; formal DP guarantees apply to the synthetic participants in the benchmark.

### 2. LR Gaussian vs LR Laplace Model Confounding:
- `lr_gaussian.py` optimizes a single multinomial objective ($C=0.02$, MinMax $[0,1]$ + bias augmentation) consuming budget $\epsilon$ jointly across classes.
- `lr_laplace.py` uses IBM diffprivlib, which implements One-vs-Rest (OvR) binary logistic regressions ($C=1.0$, Z-score clipping to $[-2,2]$) allocating an individual privacy budget of $\epsilon / K = \epsilon / 4$ to each binary classifier.
- Consequently, accuracy differences reflect both the perturbation mechanism (output vs objective) and the architectural setup (multinomial vs OvR composition).

### 3. Record-level DP vs Patient-level DP:
- In the initial Kaggle wearable reference dataset (6,000 rows), 4,233 unique patient IDs exist (up to 12 rows per patient), and ~61% of rows contain synthetic markers.
- Our pipeline provides **Record-level DP**. For multi-row patient monitoring, patient-level DP requires bounding individual participant contributions.

### 4. Hyperparameter Accounting & Composition:
- Hyperparameter tuning ($C=0.02$, tree depth, number of estimators) was conducted on the training cohort; in strict academic DP frameworks, hyperparameter search consumes privacy budget unless performed on disjoint public data.
- Repeated runs on the same training set compose privacy according to the Advanced Composition Theorem:
  $$\epsilon_{\text{total}} = \sqrt{2k\ln(1/\delta')}\epsilon + k\epsilon(e^\epsilon - 1) \quad \text{with total failure probability } k\delta + \delta'$$

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

### Regenerate Benchmark Replicates

```bash
python generate_datasets.py
```

### Running Individual Models

```bash
# 1. Gaussian Naive Bayes (Sufficient Statistics DP, pure ε-DP)
python models/rf_gaussian.py

# 2. Random Forest (Tree-based DP, pure ε-DP)
python models/rf_laplace.py

# 3. Logistic Regression (Analytic Gaussian Output DP, (ε,δ)-DP)
python models/lr_gaussian.py

# 4. Logistic Regression (Objective Perturbation DP, pure ε-DP)
python models/lr_laplace.py
```

### Running Full 30-Trial Benchmark & Plots

```bash
# Run comprehensive benchmark across all 4 models and epsilons [0.5, 0.7, 1.0] over 30 trials
python compare_all.py

# Generate publication-grade tradeoff curves & bar charts
python generate_research_plots.py

# Generate 2x2 confusion matrix grid
python generate_confusion_matrices.py
```

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
