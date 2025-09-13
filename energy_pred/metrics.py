from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Sequence, Tuple

ArrayLike = Sequence[float]

__all__ = ["mae", "mape", "rmse", "train_test_split_chronological"]

def _to_1d(y: ArrayLike) -> np.ndarray:
    if isinstance(y, (pd.Series, pd.Index)):
        return y.to_numpy(dtype=float)
    arr = np.asarray(y, dtype=float)
    if arr.ndim != 1:
        raise ValueError("Expected 1-D array")
    return arr

def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    a = _to_1d(y_true)
    b = _to_1d(y_pred)
    if a.size != b.size:
        raise ValueError("Size mismatch")
    return float(np.mean(np.abs(a - b)))

def mape(y_true: ArrayLike, y_pred: ArrayLike, epsilon: float = 1e-8) -> float:
    a = _to_1d(y_true)
    b = _to_1d(y_pred)
    if a.size != b.size:
        raise ValueError("Size mismatch")
    denom = np.clip(np.abs(a), epsilon, None)
    return float(np.mean(np.abs((a - b) / denom)))

def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    a = _to_1d(y_true)
    b = _to_1d(y_pred)
    if a.size != b.size:
        raise ValueError("Size mismatch")
    return float(np.sqrt(np.mean((a - b) ** 2)))

def train_test_split_chronological(y: ArrayLike, test_size: int) -> Tuple[np.ndarray, np.ndarray]:
    arr = _to_1d(y)
    if test_size <= 0 or test_size >= arr.size:
        raise ValueError("test_size must be >0 and < len(y)")
    return arr[:-test_size], arr[-test_size:]
