from __future__ import annotations
from typing import Tuple
import numpy as np

def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    # probs: predicted probability of positive class; labels in {0,1}
    probs = np.clip(probs, 1e-6, 1 - 1e-6)
    return float(np.mean((probs - labels) ** 2))

def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    inds = np.digitize(probs, bins) - 1
    ece_val = 0.0
    n = len(probs)
    for b in range(n_bins):
        mask = inds == b
        if not np.any(mask):
            continue
        conf = float(np.mean(probs[mask]))
        acc = float(np.mean(labels[mask]))
        ece_val += (np.sum(mask) / n) * abs(acc - conf)
    return float(ece_val)

class TemperatureScaler:
    def __init__(self, T: float = 1.0):
        self.T = T

    def fit(self, logits: np.ndarray, labels: np.ndarray, lr: float = 0.01, steps: int = 200) -> float:
        T = max(0.1, self.T)
        for _ in range(steps):
            p = 1.0 / (1.0 + np.exp(-logits / T))
            # dNLL/dT
            grad = np.mean((p - labels) * (logits / (T ** 2)) * p * (1 - p))
            T -= lr * grad
            T = float(np.clip(T, 0.1, 10.0))
        self.T = T
        return self.T

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        T = max(0.1, self.T)
        return 1.0 / (1.0 + np.exp(-logits / T))
