"""Dimension 1 -- 月线级别长期底部盘整 (spec 3 维度1, weight 25%)."""

from __future__ import annotations

import pandas as pd

from ..config import DEFAULT_THRESHOLDS, D1_SUB_WEIGHTS, DIMENSION_WEIGHTS
from ..types import DimensionResult
from ..utils import score_ge, score_le, weighted


def _consolidation_window(monthly: pd.DataFrame):
    """The slice *after* the all-time-high bar = the base-building range."""
    hi_pos = monthly["high"].to_numpy().argmax()
    consol = monthly.iloc[hi_pos + 1:]
    return hi_pos, consol


def score_d1(monthly: pd.DataFrame, chip: pd.DataFrame | None = None,
             th=DEFAULT_THRESHOLDS) -> DimensionResult:
    """`monthly` = normalised monthly OHLCV. Returns the dimension result and
    stashes the consolidation window in ``metrics`` for Dimension 2 reuse."""
    w = DIMENSION_WEIGHTS["d1"]
    if monthly is None or len(monthly) < 7:
        return DimensionResult("d1", 0.0, w, veto="月线数据不足")

    hist_high = float(monthly["high"].max())
    cur_price = float(monthly["close"].iloc[-1])
    drawdown = 1.0 - cur_price / hist_high if hist_high else 0.0

    hi_pos, consol = _consolidation_window(monthly)
    consol_months = len(consol)

    metrics = {
        "hist_high": hist_high,
        "cur_price": cur_price,
        "drawdown": drawdown,
        "consol_months": consol_months,
        "consol_start_pos": hi_pos + 1,
    }

    # --- one-vote vetoes (spec 4.4.1) ---
    if consol_months < th.d1_duration_min:
        return DimensionResult("d1", 0.0, w, metrics=metrics,
                               veto=f"盘整时长 {consol_months} 月 < 6 月")
    amplitude = float(consol["high"].max() / consol["low"].min())
    metrics["amplitude"] = amplitude
    if amplitude > th.d1_amp_veto:
        return DimensionResult("d1", 0.0, w, metrics=metrics,
                               veto=f"盘整振幅 {amplitude:.2f} > 2.5（仍在下跌）")

    # 1.4 量能收敛: tail-3-month avg vs whole-consolidation avg
    vol_col = "amount" if "amount" in consol and consol["amount"].notna().any() else "volume"
    whole_avg = float(consol[vol_col].mean())
    tail_avg = float(consol[vol_col].iloc[-3:].mean())
    vol_shrink = tail_avg / whole_avg if whole_avg else 1.0
    metrics["vol_shrink_ratio"] = vol_shrink

    sub = {
        "drawdown": score_ge(drawdown, th.d1_drawdown_base, th.d1_drawdown_full),
        "amplitude": score_le(amplitude, th.d1_amp_base, th.d1_amp_full),
        "volume_shrink": score_le(vol_shrink, th.d1_vol_shrink_base,
                                  th.d1_vol_shrink_full),
    }

    # 1.2 盘整时长: ideal 12-36 months -> 10, >=6 -> 3, deep (>36) still good
    if th.d1_duration_lo <= consol_months <= th.d1_duration_hi:
        sub["duration"] = 10
    elif consol_months > th.d1_duration_hi:
        sub["duration"] = 10  # 深度盘整 赔率最高 (spec 1.2)
    else:
        sub["duration"] = 6 if consol_months >= 9 else 3

    # 1.5 筹码集中: 90% chip-range width
    if chip is not None and len(chip):
        width = _chip_width(chip)
        if width is not None:
            metrics["chip_width"] = width
            sub["chip"] = score_le(width, th.d1_chip_width_base,
                                   th.d1_chip_width_full)

    score = weighted(sub, D1_SUB_WEIGHTS)
    return DimensionResult("d1", score, w, subscores=sub, metrics=metrics)


def _chip_width(chip: pd.DataFrame):
    """90% chip concentration band width as a fraction of its mid price.
    AkShare `stock_cyq_em` exposes 90成本-低 / 90成本-高."""
    cols = {c: c for c in chip.columns}
    lo_c = next((c for c in cols if "90" in str(c) and "低" in str(c)), None)
    hi_c = next((c for c in cols if "90" in str(c) and "高" in str(c)), None)
    if lo_c is None or hi_c is None:
        return None
    lo = float(chip[lo_c].iloc[-1])
    hi = float(chip[hi_c].iloc[-1])
    mid = (lo + hi) / 2.0
    return (hi - lo) / mid if mid else None
