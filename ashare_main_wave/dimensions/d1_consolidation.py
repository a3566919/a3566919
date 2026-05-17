"""Dimension 1 -- 月线级别长期底部盘整 (spec 3 维度1, weight 25%)."""

from __future__ import annotations

import pandas as pd

from ..config import DEFAULT_THRESHOLDS, D1_SUB_WEIGHTS, DIMENSION_WEIGHTS
from ..types import DimensionResult
from ..utils import score_ge, score_le, weighted


def _consolidation_window(monthly: pd.DataFrame, breakout_leg: int = 6,
                          escape: float = 0.80):
    """Locate the low base-building range.

    1. The structural peak is taken from history *excluding* the trailing
       ``breakout_leg`` bars, so a fresh break to new highs (precisely the
       主升浪 setup we target) does not collapse the window and self-veto
       (audit M4).
    2. The base then runs from just after that peak up to -- but not
       including -- the first month whose close decisively escapes its own
       running low by ``escape`` (default +80%, the spec's base-amplitude
       ceiling). This keeps the amplitude/volume metrics measuring the *flat
       base* even when scored after the breakout has already run (audit m1).
    """
    highs = monthly["high"].to_numpy()
    closes = monthly["close"].to_numpy()
    lows = monthly["low"].to_numpy()
    n = len(highs)
    cutoff = max(1, n - breakout_leg)
    hi_pos = int(highs[:cutoff].argmax())
    start = hi_pos + 1
    if start >= n:
        return hi_pos, monthly.iloc[start:]

    end = n
    run_min = float("inf")
    for i in range(start, n):
        run_min = min(run_min, lows[i])
        if i > start and closes[i] > run_min * (1 + escape):
            end = i
            break
    consol = monthly.iloc[start:end]
    if len(consol) == 0:
        consol = monthly.iloc[start:start + 1]
    return hi_pos, consol


def score_d1(monthly: pd.DataFrame, chip: pd.DataFrame | None = None,
             th=DEFAULT_THRESHOLDS) -> DimensionResult:
    """`monthly` = normalised monthly OHLCV. Returns the dimension result and
    stashes the consolidation window in ``metrics`` for Dimension 2 reuse."""
    w = DIMENSION_WEIGHTS["d1"]
    if monthly is None or len(monthly) < 7:
        return DimensionResult("d1", 0.0, w, veto="月线数据不足")

    hi_pos, consol = _consolidation_window(monthly)
    consol_months = len(consol)
    cur_price = float(monthly["close"].iloc[-1])

    # 1.1 base depth: structural peak (pre-breakout) vs the base trough.
    # Measuring the trough (not the recovered price) keeps the score stable
    # when evaluated right at the breakout, matching the spec's "底部" intent.
    peak_high = float(monthly["high"].iloc[:hi_pos + 1].max())
    base_low = float(consol["low"].min()) if len(consol) else cur_price
    drawdown = 1.0 - base_low / peak_high if peak_high else 0.0

    metrics = {
        "peak_high": peak_high,
        "base_low": base_low,
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
