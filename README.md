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
---

## 💍 Smart Ring IoT Telemetry Differential Privacy Architecture (Paper Implementation)

```
 [ Smart Ring Wearable Telemetry ]
   ├─ Optical PPG Sensor  ──────> Heart Rate (BPM)
   ├─ HRV / EDA Sensor    ──────> Autonomic Wearable Stress Index
   ├─ Pulse Wave Velocity ──────> Systolic Blood Pressure (mmHg)
   └─ Interstitial Optical ─────> Glucose Level Estimation (mg/dL)
                    │
                    ▼
 [ Sensitive Data Preprocessing Layer ]
   └─ Data-Independent Domain Clipping & MinMax Scaling to [0.0, 1.0]
                    │
                    ▼
 [ Differential Privacy Perturbation Engine ]
   └─ Sufficient Statistics Laplace Perturbation (diffprivlib GaussianNB)
   └─ Privacy Guarantee: Pure Epsilon-DP (delta = 0.0)
                    │
                    ▼
 [ Privacy-Protected Machine Learning Classifier ]
   └─ Gaussian Naive Bayes DP Model Inference
                    │
                    ▼
 [ Privacy-Safe Clinical Health Output & Alert Engine ]
   └─ Health States: Healthy | Pre-diabetic | Hypertensive | Metabolic Risk
```

### 🔬 Smart Ring Telemetry Paper Workflow:
1. **Telemetry Ingestion**: Continuous sensor readings from wearable smart rings (Oura Ring / Ultrahuman / Galaxy Ring) capture sensitive physiological signals.
2. **Clinical Domain Bounds Clipping**: Ingested signals are clipped to established clinical bounds ($Glucose \in [30, 300]$, $Stress \in [0.0, 1.0]$, $HR \in [30, 220]$, $BP \in [60, 250]$) and normalized to $[0, 1]$.
3. **Sufficient Statistics Perturbation**: Rather than exposing raw sensor streams or unperturbed counts to the cloud server, Laplace noise is added directly to per-class sufficient statistics (sample counts, feature sums, and sums of squares), ensuring mathematical **Pure $\epsilon$-DP ($\delta=0$)**.
4. **Privacy-Safe Inference**: The cloud or edge classifier yields accurate health classifications without storing or revealing any individual patient's raw telemetry stream.
5. **Run the Live Telemetry Pipeline**:
   ```bash
   python models/smart_ring_dp_pipeline.py 0.5
   ```

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
├── models/                                # Dedicated DP Training Scripts & IoT Pipeline
│   ├── smart_ring_dp_pipeline.py          # 💍 Smart Ring Telemetry Differential Privacy Pipeline (PRIMARY)
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

## 🔬 Models & Differential Privacy Mechanisms (Architectural & Mathematical Logic)

| # | Script | Model Architecture | Privacy Mechanism | Guarantee | Privacy Budget | Baseline Acc | DP Acc ($\epsilon=0.5$) | Privacy Cost (Drop) |
|---|--------|-------------------|-------------------|-----------|----------------|:---:|:---:|:---:|
| 1 | [`models/rf_gaussian.py`](models/rf_gaussian.py) | Gaussian Naive Bayes | Sufficient Statistics Perturbation | Pure $\epsilon$-DP ($\delta=0$) | $\epsilon \in \{0.5, 0.7, 1.0\}$ | **86.31%** | **85.28% ± 0.96%** | `-1.02%` |
| 2 | [`models/rf_laplace.py`](models/rf_laplace.py) | Random Forest (20 trees) | Tree-based DP (Permute & Flip) | Pure $\epsilon$-DP ($\delta=0$) | $\epsilon \in \{0.5, 0.7, 1.0\}$ | **85.46%** | **83.23% ± 1.07%** | `-2.23%` |
| 3 | [`models/lr_gaussian.py`](models/lr_gaussian.py) | Logistic Regression ($C=0.02$) | Analytic Gaussian Output Perturbation | $(\epsilon, \delta)$-DP ($\delta=10^{-5}$) | $\epsilon \in \{0.5, 0.7, 1.0\}$ | **82.47%** | **76.76% ± 4.63%** | `-5.72%` |
| 4 | [`models/lr_laplace.py`](models/lr_laplace.py) | Logistic Regression ($C=1.0$) | Objective Perturbation (Vector DP) | Pure $\epsilon$-DP ($\delta=0$) | $\epsilon \in \{0.5, 0.7, 1.0\}$ | **86.79%** | **84.96% ± 0.00%** | `-1.84%` |

---

### 🧠 In-Depth Mechanism Logic Across the 4 Models

#### 1. Gaussian Naive Bayes — Sufficient Statistics Perturbation (`models/rf_gaussian.py`)
- **Core Logic:** In Gaussian Naive Bayes, classification decisions depend solely on empirical class frequencies $N_c$, per-class feature sums $\sum x_i$, and sums of squares $\sum x_i^2$. Instead of perturbing inputs or predictions, Laplace noise calibrated to domain sensitivity is added directly to these sufficient statistics during training.
- **Mathematical Sensitivity:**
  $$\tilde{N}_c = N_c + \text{Lap}\left(0, \frac{1}{\epsilon}\right), \quad \tilde{\mu}_{c,j} = \mu_{c,j} + \text{Lap}\left(0, \frac{\Delta_\mu}{\epsilon}\right)$$
- **Guarantee:** Strict **Pure $\epsilon$-DP ($\delta=0$)**.
- **Utility Performance:** Retains high baseline fidelity (**85.28%** at $\epsilon=0.5$) with minimal drop ($-1.02\%$).

#### 2. Random Forest — Tree-Based Differential Privacy (`models/rf_laplace.py`)
- **Core Logic:** Standard Decision Trees compute split criteria via Gini Impurity or Information Gain, which leaks exact training distributions. The DP Random Forest selects feature split thresholds **randomly** within pre-defined domain bounds $[L_j, U_j]$ (consuming zero privacy budget).
- **Leaf Node Perturbation:** Leaf class counts are perturbed using the **Exponential / Permute-and-Flip mechanism** with Laplace noise to assign class votes without revealing individual training patient paths.
- **Guarantee:** Strict **Pure $\epsilon$-DP ($\delta=0$)**.
- **Utility Performance:** High stability across ensemble averaging (**83.23% ± 1.07%** at $\epsilon=0.5$).

#### 3. Logistic Regression — Analytic Gaussian Output Perturbation (`models/lr_gaussian.py`)
- **Core Logic:** Trains an $L_2$-regularized multinomial Logistic Regression on clean normalized data ($C=0.02$). Input samples are augmented with a constant bias feature and scaled by $1/\sqrt{d+1}$ to guarantee $\|\tilde{x}\|_2 \le 1$.
- **Exact Sensitivity & Noise Calibration:** Under multinomial cross-entropy loss with Lipschitz constant $L=\sqrt{2}$, the $L_2$ weight sensitivity under single-sample replacement is strictly bounded:
  $$\Delta_2(W^*) \le 2\sqrt{2} \cdot C \approx 0.056569$$
  Calibrated Gaussian noise is added directly to the final learned weight matrix using the exact **Balle & Wang (NeurIPS 2018) Analytic Gaussian Mechanism**:
  $$W_{\text{DP}} = W^* + \mathcal{N}(0, \sigma^2 \mathbf{I}), \quad \text{where } \sigma = 0.397780 \text{ for } (\epsilon=0.5, \delta=10^{-5})$$
- **Guarantee:** Approximate **$(\epsilon, \delta)$-DP** with $\delta = 10^{-5}$.
- **Utility Performance:** Displays an exemplary smooth privacy-utility tradeoff: **76.76%** at $\epsilon=0.5 \to$ **79.30%** at $\epsilon=0.7 \to$ **80.81%** at $\epsilon=1.0$.

#### 4. Logistic Regression — Objective Perturbation (`models/lr_laplace.py`)
- **Core Logic:** Rather than perturbing weights post-hoc (output perturbation), noise is injected directly into the **optimization objective function** (loss function) prior to minimization (Chaudhuri et al., JMLR 2011):
  $$J_{\text{DP}}(w) = \frac{1}{n} \sum_{i=1}^n \ell(w; x_i, y_i) + \frac{\lambda}{2} \|w\|^2 + \frac{1}{n} b^T w$$
  where vector $b$ is drawn from a spherical distribution with density $\propto \exp(-\frac{\epsilon n}{2} \|b\|)$.
- **Guarantee:** Strict **Pure $\epsilon$-DP ($\delta=0$)**.
- **Utility Performance:** Consistent convex convergence (**84.96%** accuracy across $\epsilon \ge 0.5$).

---

## 📊 Comprehensive Benchmark Results ($N=30$ Monte Carlo Trials)

Evaluated across $N=30$ independent trials per configuration with stratified 80/20 train/test splits on 80,000-sample benchmark replicates with realistic clinical population variance (class overlap):

### Accuracy Comparison Across Privacy Budgets

| Model | DP Mechanism | Formal Guarantee | Baseline (No DP) | $\epsilon=0.5$ (High Privacy) | $\epsilon=0.7$ (Moderate) | $\epsilon=1.0$ (Balanced) |
|-------|--------------|------------------|:---:|:---:|:---:|:---:|
| **Gaussian Naive Bayes** | Sufficient Statistics | Pure $\epsilon$-DP | **86.31%** | **85.28% ± 0.96%** | **85.90% ± 0.42%** | **86.11% ± 0.34%** |
| **Random Forest** | Tree-based DP | Pure $\epsilon$-DP | **85.46%** | **83.23% ± 1.07%** | **83.32% ± 1.05%** | **83.36% ± 1.07%** |
| **Logistic Regression** | Output Perturbation | $(\epsilon, \delta)$-DP | **82.47%** | **76.76% ± 4.63%** | **79.30% ± 3.30%** | **80.81% ± 2.32%** |
| **Logistic Regression** | Objective Perturbation | Pure $\epsilon$-DP | **86.79%** | **84.96% ± 0.00%** | **84.96% ± 0.00%** | **84.96% ± 0.00%** |

### Macro F1-Scores

| Model | Mechanism | Guarantee | $\epsilon=0.5$ | $\epsilon=0.7$ | $\epsilon=1.0$ |
|-------|-----------|-----------|:---:|:---:|:---:|
| **Gaussian Naive Bayes** | Sufficient Statistics | Pure $\epsilon$-DP | **0.8528** | **0.8591** | **0.8612** |
| **Random Forest** | Tree-based DP | Pure $\epsilon$-DP | **0.8314** | **0.8322** | **0.8327** |
| **Logistic Regression (Gaussian)** | Output Perturbation | $(\epsilon, \delta)$-DP | **0.7492** | **0.7800** | **0.7979** |
| **Logistic Regression (Laplace)** | Objective Perturbation | Pure $\epsilon$-DP | **0.8465** | **0.8465** | **0.8465** |

### Detailed Findings & Research Insights:
1. **Clear Privacy–Utility Tradeoff:** As the privacy budget is relaxed from $\epsilon=0.5 \to \epsilon=1.0$, model accuracies steadily increase towards their non-private baselines (e.g., Logistic Regression Gaussian recovers $+4.05\%$ accuracy from $\epsilon=0.5$ to $\epsilon=1.0$).
2. **Realistic Clinical Baseline Overlap:** By establishing standard clinical diagnostic standard deviations ($\text{stress } \sigma=0.15, \text{glucose } \sigma=22.0, \text{BP } \sigma=18.0, \text{HR } \sigma=15.0$), non-DP baselines reside in the realistic medical diagnostic range ($82\% - 86\%$), avoiding artificial 100% separability.
3. **Robust Distribution Metrics:** With $N=30$ Monte Carlo trials per configuration, variance bounds and confidence intervals reliably reflect differential privacy perturbation effects.

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
