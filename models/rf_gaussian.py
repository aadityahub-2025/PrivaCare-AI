"""
PrivaCare-AI - models/rf_gaussian.py
Compatibility alias for models/nb_gaussian.py (Gaussian Naive Bayes DP).
"""
import runpy
import sys
import os

if __name__ == "__main__":
    target = os.path.join(os.path.dirname(__file__), "nb_gaussian.py")
    runpy.run_path(target, run_name="__main__")
