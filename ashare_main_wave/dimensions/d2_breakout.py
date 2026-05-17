"""Dimension 2 -- 月线级别突破平台 (spec 3 维度2, weight 20%)."""

from __future__ import annotations

import pandas as pd

from ..config import DEFAULT_THRESHOLDS, D2_SUB_WEIGHTS, DIMENSION_WEIGHTS
from ..types import DimensionResult
from ..utils import score_ge, weighted


def score_d2(monthly: pd.DataFrame, weekly: pd.DataFrame | None,
             d1_metrics: dict, th=DEFAULT_THRESHOLDS) -> DimensionResult:
    w = DIMENSION_WEIGHTS["d2"]
    if monthly is None or len(monthly) < 13:
        return DimensionResult("d2", 0.0, w, veto="月线数据不足")

    last = monthly.iloc[-1]
    prev12_high = float(monthly["close"].iloc[-13:-1].max())
    excess = last["close"] / prev12_high - 1 if prev12_high else -1.0

    consol_start = d1_metrics.get("consol_start_pos", 0)
    consol = monthly.iloc[consol_start:-1] if consol_start < len(monthly) - 1 \
        else monthly.iloc[:-1]
    vol_col = "amount" if "amount" in consol and consol["amount"].notna().any() else "volume"
    avg_vol = float(consol[vol_col].mean()) if len(consol) else float("nan")
    last_vol = float(last["amount"] if vol_col == "amount" else last["volume"])
    vol_mult = last_vol / avg_vol if avg_vol else 0.0

    rng = float(last["high"] - last["low"]) or 1e-9
    body_ratio = abs(float(last["close"] - last["open"])) / rng
    bullish = last["close"] >= last["open"]

    metrics = {
        "prev12_high": prev12_high,
        "breakout_excess": excess,
        "vol_mult": vol_mult,
        "body_ratio": body_ratio,
        "bullish": bool(bullish),
    }

    # one-vote veto (spec 4.4.2): 突破月量能 < 1.5x 盘整均量
    if excess < 0 or vol_mult < th.d2_vol_mult_min:
        veto = (f"月线突破量能 {vol_mult:.2f}x < 1.5x"
                if excess >= 0 else "月收盘未有效突破前12月")
        return DimensionResult("d2", 0.0, w, metrics=metrics, veto=veto)

    sub = {
        "excess": score_ge(excess, 0.0, th.d2_excess_full, mid=th.d2_excess_full / 2),
        "body": score_ge(body_ratio if bullish else 0.0,
                         th.d2_body_base, th.d2_body_full),
        "volume": score_ge(vol_mult, th.d2_vol_mult_min, th.d2_vol_mult_full),
    }

    # 2.5 周线确认: weeks in the breakout month closing above the platform top
    if weekly is not None and len(weekly):
        bo_month = pd.Timestamp(last["date"]).to_period("M")
        in_month = weekly[weekly["date"].dt.to_period("M") == bo_month]
        weeks_above = int((in_month["close"] > prev12_high).sum())
        metrics["weeks_above_platform"] = weeks_above
        sub["weekly"] = score_ge(weeks_above, th.d2_weekly_base,
                                 th.d2_weekly_full)

    score = weighted(sub, D2_SUB_WEIGHTS)
    return DimensionResult("d2", score, w, subscores=sub, metrics=metrics)
