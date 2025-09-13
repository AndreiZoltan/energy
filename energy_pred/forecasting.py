from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from typing import Sequence, Optional

ArrayLike = Sequence[float]

@dataclass
class NaiveForecaster:
    """Forecasts future values as the last observed value.

    If horizon > 1 returns a constant vector equal to last value.
    """
    def fit(self, y: ArrayLike):
        arr = _to_1d_array(y)
        if arr.size == 0:
            raise ValueError("y is empty")
        self.last_ = float(arr[-1])
        return self

    def predict(self, horizon: int):
        if not hasattr(self, "last_"):
            raise RuntimeError("Call fit first")
        return np.full(shape=horizon, fill_value=self.last_, dtype=float)

@dataclass
class SeasonalNaiveForecaster:
    season_length: int

    def fit(self, y: ArrayLike):
        arr = _to_1d_array(y)
        if arr.size < self.season_length:
            raise ValueError("Not enough data for a full season")
        self.history_ = arr
        return self

    def predict(self, horizon: int):
        if not hasattr(self, "history_"):
            raise RuntimeError("Call fit first")
        reps = int(np.ceil(horizon / self.season_length))
        pattern = self.history_[-self.season_length:]
        return np.tile(pattern, reps)[:horizon].astype(float)

@dataclass
class MovingAverageForecaster:
    window: int

    def fit(self, y: ArrayLike):
        arr = _to_1d_array(y)
        if arr.size < self.window:
            raise ValueError("Window larger than series length")
        self.arr_ = arr
        return self

    def predict(self, horizon: int):
        if not hasattr(self, "arr_"):
            raise RuntimeError("Call fit first")
        mean_val = float(self.arr_[-self.window:].mean())
        return np.full(horizon, mean_val, dtype=float)

@dataclass
class ExpSmoothingForecaster:
    alpha: float = 0.3  # smoothing factor in (0,1]
    initial_level: Optional[float] = None

    def fit(self, y: ArrayLike):
        arr = _to_1d_array(y)
        if arr.size == 0:
            raise ValueError("y is empty")
        level = self.initial_level if self.initial_level is not None else arr[0]
        alpha = self.alpha
        if not (0 < alpha <= 1):
            raise ValueError("alpha must be in (0,1]")
        for v in arr:
            level = alpha * v + (1 - alpha) * level
        self.level_ = float(level)
        return self

    def predict(self, horizon: int):
        if not hasattr(self, "level_"):
            raise RuntimeError("Call fit first")
        return np.full(horizon, self.level_, dtype=float)

def _to_1d_array(y: ArrayLike) -> np.ndarray:
    if isinstance(y, (pd.Series, pd.Index)):
        return y.to_numpy(dtype=float)
    arr = np.asarray(y, dtype=float)
    if arr.ndim != 1:
        raise ValueError("Input series must be 1-D")
    return arr
