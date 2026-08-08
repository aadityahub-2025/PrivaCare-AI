# PrivaCare-AI 🛡️🩺

**PrivaCare-AI** is a privacy-preserving healthcare machine learning framework designed to protect sensitive clinical patient datasets. It leverages **Gaussian Differential Privacy (GDP)** to inject calibrated mathematical noise into patient symptoms and vitals before training a Multi-Layer Perceptron (MLP) classification model.

This project empirically demonstrates the fundamental **Privacy-Utility Tradeoff** ($\epsilon$-budget vs. Model Accuracy) in modern healthcare AI systems.

---

## 🚀 Key Features

- **Gaussian Differential Privacy Engine:** Mathematical privacy guarantees using Gaussian noise calibrated to specific privacy budgets ($\epsilon, \delta$).
- **Interactive Runtime Configuration:** Dynamically test different privacy budgets ($\epsilon$) at runtime without editing code.
- **Deep MLP Classifier:** Multi-Layer Perceptron (`128 -> 64 -> 32`) architecture built using Scikit-Learn.
- **Empirical Benchmarking:** Automated pipeline calculating Baseline Accuracy vs. Privacy-Preserving Accuracy and evaluating the Privacy Cost.

---

## 📂 Project Structure

```text
PrivaCare-AI/
│
├── data/
│   └── dataset.csv         # Clinical dataset (Symptoms & Vitals)
│
├── privacy_engine.py       # Core Gaussian DP noise engine implementation
├── run_experiment.py       # Main experiment pipeline & benchmark runner
└── README.md               # Project documentation
```
