# Hum banayenge - Model train karke result check karne
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from privacy_engine import add_gaussian_noise

def main():
    csv_path = "data/dataset.csv"
    
    if not os.path.exists(csv_path):
        print(f"Error: '{csv_path}' file nahi mili! Pehle data folder me dataset.csv daalo.")
        return

    print(f"\n--> Loading Dataset from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Gender column encoding (Male: 1, Female: 0)
    if 'gender' in df.columns:
        df['gender'] = df['gender'].map({'Male': 1, 'Female': 0}).fillna(0)

    target_column = 'disease' if 'disease' in df.columns else df.columns[-1]
    X = df.drop(columns=[target_column])
    y = df[target_column]

    feature_cols = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # -------------------------------------------------------------
    # RUNTIME INPUT: Ab terminal aapse Epsilon poochega
    # -------------------------------------------------------------
    print("\n------------------------------------------------")
    try:
        user_input = input("Enter Epsilon value (e.g. 1.0, 0.1, 0.05, 0.01) [Default 0.05]: ").strip()
        epsilon = float(user_input) if user_input else 0.05
    except ValueError:
        print("Invalid input! Defaulting Epsilon to 0.05")
        epsilon = 0.05
    print(f"--> Running experiment with Epsilon = {epsilon}")
    print("------------------------------------------------")

    # 1. Baseline Model Train
    print("\n[1] Training Baseline Model (Without Noise)...")
    clf_baseline = MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=200, random_state=42)
    clf_baseline.fit(X_train, y_train)
    acc_baseline = accuracy_score(y_test, clf_baseline.predict(X_test))
    print(f"--> Baseline Accuracy: {acc_baseline * 100:.2f}%")

    # 2. Gaussian DP Model Train
    print(f"\n[2] Applying Gaussian DP Noise (Epsilon={epsilon}) & Training Model...")
    X_train_gaussian = X_train.copy()
    delta = 1e-5   # Failure Probability

    for col in feature_cols:
        X_train_gaussian[col] = add_gaussian_noise(X_train_gaussian[col], epsilon=epsilon, delta=delta)

    clf_gaussian = MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=200, random_state=42)
    clf_gaussian.fit(X_train_gaussian, y_train)
    acc_gaussian = accuracy_score(y_test, clf_gaussian.predict(X_test))
    print(f"--> Gaussian DP Accuracy (Epsilon={epsilon}): {acc_gaussian * 100:.2f}%")

    # Final Output Summary
    print("\n================ RESULT SUMMARY ================")
    print(f"Original Model Accuracy      : {acc_baseline * 100:.2f}%")
    print(f"Gaussian DP Model Accuracy   : {acc_gaussian * 100:.2f}%")
    print(f"Accuracy Drop (Privacy Cost) : {(acc_baseline - acc_gaussian) * 100:.2f}%")
    print("================================================")

if __name__ == "__main__":
    main()