"""
PrivaCare-AI -- models/smart_ring_dp_pipeline.py
Smart Ring IoT Telemetry Differential Privacy Pipeline & Research Benchmark.

Architecture Flow for Research Paper:
  [ Smart Ring Sensors (PPG, HRV, Temp, PWV) ]
                        │
                        ▼
  [ Sensor Data Collection & Preprocessing Engine ]
    - Heart Rate (BPM)            -> [30, 220]
    - Stress Index (HRV/EDA)      -> [0.0, 1.0]
    - Blood Pressure Systolic     -> [60, 250] mmHg
    - Glucose Level               -> [30.0, 300.0] mg/dL
                        │
                        ▼
  [ Clinical Bounds Clipping & MinMax Normalization [0,1] ]
                        │
                        ▼
  [ Differential Privacy Perturbation Layer ]
    - Sufficient Statistics Noise Injection (diffprivlib GaussianNB)
    - Guarantee: Pure Epsilon-DP (delta = 0)
                        │
                        ▼
  [ DP Machine Learning Classifier (Gaussian Naive Bayes) ]
                        │
                        ▼
  [ Privacy-Safe Clinical Health Output & Alert Engine ]
    (Classes: 0=Healthy, 1=Pre-diabetic, 2=Hypertensive, 3=Metabolic Risk)
"""

import math
import numpy as np
import pandas as pd
import diffprivlib.models as dp
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import sys
import json

# ===========================================================================
# 1. SMART RING SENSOR CONFIGURATION & DOMAIN BOUNDS
# ===========================================================================
FEATURE_COLS = [
    "glucose_level",            # mg/dL  [30, 300]
    "stress_level",             # Ratio  [0.0, 1.0]
    "heart_rate",               # BPM    [30, 220]
    "blood_pressure_systolic",  # mmHg   [60, 250]
]

DOMAIN_BOUNDS = {
    "glucose_level":           (30.0, 300.0),
    "stress_level":            (0.0,  1.0),
    "heart_rate":              (30.0, 220.0),
    "blood_pressure_systolic": (60.0, 250.0),
}

CLASS_NAMES = ["Healthy", "Pre-diabetic", "Hypertensive", "Metabolic Risk"]

# ===========================================================================
# 2. PREPROCESSING & NORMALIZATION ENGINE
# ===========================================================================
def preprocess_ring_telemetry(raw_telemetry):
    """
    Clips raw smart ring telemetry readings to clinical domain bounds
    and normalizes features to the unit hypercube [0, 1].
    """
    features_norm = []
    for col in FEATURE_COLS:
        val = float(raw_telemetry[col])
        lo, hi = DOMAIN_BOUNDS[col]
        clipped_val = np.clip(val, lo, hi)
        norm_val = (clipped_val - lo) / (hi - lo + 1e-12)
        features_norm.append(norm_val)
    return np.array(features_norm).reshape(1, -1)

def batch_normalize(X):
    X_norm = np.zeros_like(X, dtype=float)
    for i, col in enumerate(FEATURE_COLS):
        lo, hi = DOMAIN_BOUNDS[col]
        X_norm[:, i] = np.clip(X[:, i], lo, hi)
        X_norm[:, i] = (X_norm[:, i] - lo) / (hi - lo + 1e-12)
    return X_norm

# ===========================================================================
# 3. SMART RING TELEMETRY SIMULATOR
# ===========================================================================
def simulate_smart_ring_telemetry(num_samples=10, seed=42):
    """
    Simulates real-time telemetry packets streamed from a Smart Ring
    (e.g., PPG sensor for HR, HRV/EDA composite for Stress, PWV for BP, Glucose).
    """
    rng = np.random.RandomState(seed)
    samples = []
    
    profiles = [
        # (Glucose, Stress, HR, BP, Target_Label)
        (92.5, 0.22, 68.0, 110.0, 0),   # Patient A: Healthy
        (115.0, 0.45, 78.0, 126.0, 1),  # Patient B: Pre-diabetic
        (98.0, 0.68, 95.0, 158.0, 2),   # Patient C: Hypertensive
        (178.0, 0.82, 105.0, 165.0, 3)  # Patient D: Metabolic Risk
    ]
    
    for i in range(num_samples):
        prof = profiles[i % len(profiles)]
        g = float(round(rng.normal(prof[0], 5.0), 2))
        s = float(round(np.clip(rng.normal(prof[1], 0.03), 0.0, 1.0), 2))
        h = float(round(rng.normal(prof[2], 4.0), 2))
        b = float(round(rng.normal(prof[3], 5.0), 2))
        samples.append({
            "device_id": f"RING_IOT_{1000 + i}",
            "glucose_level": g,
            "stress_level": s,
            "heart_rate": h,
            "blood_pressure_systolic": b,
            "ground_truth_class": prof[4],
            "ground_truth_label": CLASS_NAMES[prof[4]]
        })
    return samples

# ===========================================================================
# 4. MAIN SMART RING DP PIPELINE EXECUTION
# ===========================================================================
def run_smart_ring_pipeline(epsilon=0.5):
    print("\n" + "="*75)
    print("  PrivaCare-AI -- Smart Ring IoT Telemetry & Differential Privacy Pipeline")
    print("  Architecture: Ring Telemetry -> Sensing -> Norm -> Sufficient Stats DP -> Output")
    print("="*75 + "\n")

    dataset_path = "datasets/dataset_1_rf_gaussian.json"
    if not os.path.exists(dataset_path):
        print(f"[!] Dataset not found at {dataset_path}. Generating benchmark datasets...")
        from generate_datasets import generate_replicates
        generate_replicates()

    # Load dataset for training the DP Gaussian Naive Bayes Classifier
    df = pd.read_json(dataset_path)
    X_raw = df[FEATURE_COLS].values.astype(float)
    y_raw = df["health_event"].values.astype(int)

    X_norm = batch_normalize(X_raw)

    print(f"[1] Ingesting & Normalizing Benchmark Data ({len(df):,} rows)...")
    bounds = ([0.0] * 4, [1.0] * 4)

    print(f"[2] Initializing DP Gaussian Naive Bayes Model (Privacy Budget e={epsilon})...")
    clf_dp = dp.GaussianNB(epsilon=epsilon, bounds=bounds)
    clf_dp.fit(X_norm, y_raw)
    print("    [+] Model successfully trained under Pure Epsilon-DP (delta=0.0).")

    print("\n" + "-"*75)
    print("  LIVE SMART RING TELEMETRY STREAM SIMULATION & PRIVACY-SAFE INFERENCE")
    print("-"*75)

    ring_data = simulate_smart_ring_telemetry(num_samples=6, seed=2026)

    print(f"{'Device ID':<15} | {'Glucose':<8} {'Stress':<7} {'HR':<6} {'BP':<6} | {'True Health Label':<16} | {'DP Predicted Health Output':<22}")
    print("-" * 95)

    correct = 0
    for sample in ring_data:
        x_ring_norm = preprocess_ring_telemetry(sample)
        pred_class = int(clf_dp.predict(x_ring_norm)[0])
        pred_label = CLASS_NAMES[pred_class]
        is_correct = (pred_class == sample["ground_truth_class"])
        if is_correct:
            correct += 1
        
        status_symbol = "[OK]" if is_correct else "[MISMATCH]"
        print(f"{sample['device_id']:<15} | {sample['glucose_level']:<8.1f} {sample['stress_level']:<7.2f} {sample['heart_rate']:<6.1f} {sample['blood_pressure_systolic']:<6.1f} | {sample['ground_truth_label']:<16} | {pred_label:<16} {status_symbol}")

    print("-" * 95)
    acc = (correct / len(ring_data)) * 100
    print(f"\n[+] Smart Ring Live Stream Accuracy (at e={epsilon}): {acc:.1f}%")
    print(f"[+] Privacy Guarantee: Pure Epsilon-DP (e={epsilon}, d=0.0)")
    print("    Raw telemetry data remains strictly protected via Sufficient Statistics Perturbation.")
    print("="*75 + "\n")

if __name__ == "__main__":
    eps = 0.5
    if len(sys.argv) > 1:
        try:
            eps = float(sys.argv[1])
        except ValueError:
            eps = 0.5
    run_smart_ring_pipeline(epsilon=eps)
