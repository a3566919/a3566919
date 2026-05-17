"""Reusable technical indicators (MA / MACD / swing structure)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def ma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=1).mean()


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Return (DIF, DEA, HIST). HIST uses the common A-share 2*(DIF-DEA)
    convention; only its sign/slope is used downstream."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    hist = 2 * (dif - dea)
    return dif, dea, hist


def golden_cross(dif: pd.Series, dea: pd.Series, lookback: int = 5) -> bool:
    """DIF crossed above DEA within the last ``lookback`` bars."""
    diff = (dif - dea).to_numpy()
    n = len(diff)
    for i in range(max(1, n - lookback), n):
        if diff[i - 1] <= 0 and diff[i] > 0:
            return True
    return False


def dead_cross(dif: pd.Series, dea: pd.Series, lookback: int = 3) -> bool:
    diff = (dif - dea).to_numpy()
    n = len(diff)
    for i in range(max(1, n - lookback), n):
        if diff[i - 1] >= 0 and diff[i] < 0:
            return True
    return False


def swing_points(series: pd.Series, order: int = 2):
    """Local maxima/minima indices using an ``order``-bar window.

    Returns (highs_idx, lows_idx) as positional integer lists.
    """
    v = series.to_numpy()
    n = len(v)
    highs, lows = [], []
    for i in range(order, n - order):
        window = v[i - order:i + order + 1]
        if v[i] == window.max() and v[i] >= v[i - 1]:
            highs.append(i)
        if v[i] == window.min() and v[i] <= v[i - 1]:
            lows.append(i)
    return highs, lows


def _rising_steps(values) -> int:
    """Number of consecutive strictly-rising adjacent steps in a sequence."""
    steps = 0
    for a, b in zip(values, values[1:]):
        if b > a:
            steps += 1
    return steps


def count_higher_low_groups(high: pd.Series, low: pd.Series, order: int = 2) -> int:
    """Count groups of consecutively-rising local highs/lows (spec 3.1).

    A "group" is an adjacent swing-low step that rises while the surrounding
    swing-high structure also trends up; we approximate it as the smaller of
    the rising swing-high step count and rising swing-low step count, which
    rewards a clean stair-step uptrend and ignores noise.
    """
    highs_idx, _ = swing_points(high, order=order)
    _, lows_idx = swing_points(low, order=order)
    h = high.to_numpy()
    lo = low.to_numpy()
    rising_highs = _rising_steps([h[i] for i in highs_idx])
    rising_lows = _rising_steps([lo[i] for i in lows_idx])
    return min(rising_highs, rising_lows)
