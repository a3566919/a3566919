"""Scoring primitives and date helpers.

The spec (4.2) scores every sub-indicator on a 0/3/6/10 ladder:
    0  = below the base line
    3  = reaches the base line
    6  = approaches the full-mark threshold
    10 = exceeds the full-mark threshold
All helpers below return one of those four values so dimension code stays
declarative.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

LADDER = (0, 3, 6, 10)


def score_ge(value, base, full, mid=None) -> int:
    """Higher-is-better ladder.

    base < full. ``mid`` is the "approaching full" line; when omitted it is
    placed at the mid-point of base..full.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0
    if mid is None:
        mid = (base + full) / 2.0
    if value >= full:
        return 10
    if value >= mid:
        return 6
    if value >= base:
        return 3
    return 0


def score_le(value, base, full, mid=None) -> int:
    """Lower-is-better ladder.

    ``base`` is the loosest still-acceptable value, ``full`` the strictest
    (full < base). e.g. amplitude: base=1.8, full=1.5.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0
    if mid is None:
        mid = (base + full) / 2.0
    if value <= full:
        return 10
    if value <= mid:
        return 6
    if value <= base:
        return 3
    return 0


def score_band(value, ideal_lo, ideal_hi, ok_lo, ok_hi) -> int:
    """Range indicator (e.g. turnover 5-15%, ideal 7-12%).

    inside ideal band -> 10, inside acceptable band -> 6,
    within a 50% widened acceptable band -> 3, otherwise 0.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0
    if ideal_lo <= value <= ideal_hi:
        return 10
    if ok_lo <= value <= ok_hi:
        return 6
    span = ok_hi - ok_lo
    if (ok_lo - 0.5 * span) <= value <= (ok_hi + 0.5 * span):
        return 3
    return 0


def score_bool(flag: bool, true_score: int = 10, false_score: int = 0) -> int:
    return true_score if flag else false_score


def weighted(subscores: dict, weights: dict) -> float:
    """Weighted average of 0/3/6/10 subscores -> value in [0, 10].

    Weights are looked up by the keys present in ``subscores`` and renormalised
    so a missing data feed degrades gracefully instead of zeroing the
    dimension.
    """
    keys = [k for k in subscores if k in weights]
    if not keys:
        return 0.0
    wsum = sum(weights[k] for k in keys)
    if wsum == 0:
        return 0.0
    return sum(subscores[k] * weights[k] for k in keys) / wsum


# --- date helpers ----------------------------------------------------------

def _parse(date_str: str) -> datetime:
    s = str(date_str).replace("-", "").replace("/", "")[:8]
    return datetime.strptime(s, "%Y%m%d")


def offset_date(date_str: str, days: int) -> str:
    """Shift a YYYYMMDD (or YYYY-MM-DD) date by calendar ``days``.

    Returns the same compact ``YYYYMMDD`` form AkShare expects.
    """
    return (_parse(date_str) + timedelta(days=days)).strftime("%Y%m%d")


def to_ts(date_str: str) -> pd.Timestamp:
    return pd.Timestamp(_parse(date_str))
