"""Dimension 3 -- 日线级别走出上涨趋势 (spec 3 维度3, weight 15%)."""

from __future__ import annotations

import pandas as pd

from ..config import DEFAULT_THRESHOLDS, D3_SUB_WEIGHTS, DIMENSION_WEIGHTS
from ..indicators import count_higher_low_groups, dead_cross, golden_cross, ma, macd
from ..types import DimensionResult
from ..utils import score_band, score_ge, score_le, weighted


def score_d3(daily: pd.DataFrame, d2_metrics: dict,
             th=DEFAULT_THRESHOLDS) -> DimensionResult:
    w = DIMENSION_WEIGHTS["d3"]
    if daily is None or len(daily) < 60:
        return DimensionResult("d3", 0.0, w, veto="日线数据不足")

    close = daily["close"]
    ma5, ma10, ma20, ma60 = (ma(close, n) for n in (5, 10, 20, 60))
    dif, dea, hist = macd(close)

    last_close = float(close.iloc[-1])

    # --- veto (spec 4.4.3): close < MA20 and not recovered within 3 days ---
    rec = th.d3_ma20_recover_days
    below = (close.iloc[-rec - 1:] < ma20.iloc[-rec - 1:])
    metrics = {"last_close": last_close, "ma20": float(ma20.iloc[-1])}
    if bool(below.iloc[-1]) and bool(below.iloc[-rec:].all()):
        return DimensionResult("d3", 0.0, w, metrics=metrics,
                               veto="日线跌破 MA20 且 3 日未收复")
    if dead_cross(dif, dea, lookback=3):
        return DimensionResult("d3", 0.0, w, metrics=metrics,
                               veto="日线 MACD 死叉")

    win = daily.iloc[-20:]
    groups = count_higher_low_groups(win["high"], win["low"])

    ma_stack = (last_close > ma5.iloc[-1] > ma10.iloc[-1] > ma20.iloc[-1]
                and ma60.iloc[-1] >= ma60.iloc[-5])

    macd_gc = golden_cross(dif, dea, lookback=5)
    above_zero = dif.iloc[-1] > 0 and dea.iloc[-1] > 0
    hist_rising = bool((hist.iloc[-3:].diff().iloc[1:] > 0).all()) and hist.iloc[-1] > 0

    plat = d2_metrics.get("prev12_high")
    start_gain = (last_close / plat - 1) if plat else 1.0

    turn = float(daily["turnover"].iloc[-5:].mean()) if "turnover" in daily else None

    metrics.update({
        "higher_low_groups": groups,
        "ma_stack": bool(ma_stack),
        "macd_golden_cross": bool(macd_gc),
        "macd_above_zero": bool(above_zero),
        "start_gain_from_platform": start_gain,
        "turnover5": turn,
    })

    sub = {
        "higher_lows": score_ge(groups, th.d3_higher_lows_base,
                                th.d3_higher_lows_full),
        "ma_stack": 10 if ma_stack else (6 if last_close > ma20.iloc[-1] else 0),
        "start_gain": score_le(start_gain, th.d3_start_gain_base,
                               th.d3_start_gain_full),
    }
    if macd_gc and above_zero:
        sub["macd"] = 10
    elif macd_gc or hist_rising:
        sub["macd"] = 6
    elif above_zero:
        sub["macd"] = 3
    else:
        sub["macd"] = 0
    if turn is not None:
        sub["turnover"] = score_band(turn, th.d3_turnover_ideal_lo,
                                     th.d3_turnover_ideal_hi,
                                     th.d3_turnover_lo, th.d3_turnover_hi)

    score = weighted(sub, D3_SUB_WEIGHTS)
    return DimensionResult("d3", score, w, subscores=sub, metrics=metrics)
