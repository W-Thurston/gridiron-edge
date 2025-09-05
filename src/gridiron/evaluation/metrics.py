from __future__ import annotations

import numpy as np
import pandas as pd


def brier_score(y_true: pd.Series, p: pd.Series) -> float:
    return float(np.mean((y_true.astype(float) - p.astype(float)) ** 2))


def log_loss(y_true: pd.Series, p: pd.Series, eps: float = 1e-12) -> float:
    p = np.clip(p.astype(float), eps, 1 - eps)
    y = y_true.astype(int)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def accuracy(y_true: pd.Series, p: pd.Series, thresh: float = 0.5) -> float:
    return float(np.mean((p >= thresh).astype(int) == y_true.astype(int)))
