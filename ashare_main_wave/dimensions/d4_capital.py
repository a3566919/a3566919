"""Dimension 4 -- 主力行为 (spec 3 维度4, weight 25% = 4A 12% + 4B 13%).

Feeds here are frequently missing in offline/back-test runs; every
sub-indicator is optional and :func:`utils.weighted` renormalises around the
feeds that are present so a partial data set never zeroes the dimension
spuriously (the global "<30%" veto in :mod:`scoring` still guards quality).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import (D4A_SUB_WEIGHTS, D4A_WEIGHT, D4B_SUB_WEIGHTS,
                       D4B_WEIGHT, DEFAULT_THRESHOLDS)
from ..indicators import ma
from ..types import DimensionResult
from ..utils import score_band, score_ge, score_le, weighted


def _find_col(df: pd.DataFrame, *keywords):
    for c in df.columns:
        name = str(c)
        if all(k in name for k in keywords):
            return c
    return None


def _num(series) -> float:
    return float(pd.to_numeric(series, errors="coerce").fillna(0).sum())


# --- 4A public-data feeds --------------------------------------------------

def _s_lhb(lhb, float_mktcap, th):
    if lhb is None or len(lhb) == 0:
        return None, {}
    seat_col = _find_col(lhb, "营业部") or _find_col(lhb, "交易营业部")
    net_col = _find_col(lhb, "净额") or _find_col(lhb, "净买入")
    if net_col is None:
        return None, {}
    inst_mask = lhb.astype(str).apply(
        lambda r: r.str.contains("机构专用").any(), axis=1)
    inst = lhb[inst_mask] if inst_mask.any() else lhb
    nets = pd.to_numeric(inst[net_col], errors="coerce").fillna(0)
    buys = nets[nets > 0]
    count = int(len(buys))
    total = float(buys.sum())
    single_ok = (buys >= th.d4a_lhb_min_amount).any() or (
        float_mktcap and (buys >= 0.003 * float_mktcap).any())
    metrics = {"lhb_inst_buy_count": count, "lhb_inst_buy_total": total}
    if count == 0 or not single_ok:
        return 0, metrics
    if count >= th.d4a_lhb_full_count and total >= th.d4a_lhb_full_amount:
        return 10, metrics
    return 6 if count >= th.d4a_lhb_full_count or total >= th.d4a_lhb_full_amount else 3, metrics


def _s_block(block, float_mktcap, th):
    if block is None or len(block) == 0:
        return None, {}
    disc_col = _find_col(block, "折溢率") or _find_col(block, "折价")
    amt_col = _find_col(block, "成交额") or _find_col(block, "成交金额")
    if disc_col is None or amt_col is None:
        return None, {}
    disc = pd.to_numeric(block[disc_col], errors="coerce")
    if disc.abs().median() > 1.5:
        disc = disc / 100.0
    disc = -disc  # store discount as positive number
    amt = pd.to_numeric(block[amt_col], errors="coerce").fillna(0)
    keep = disc <= th.d4a_block_discount_base
    ratio = float(amt[keep].sum()) / float_mktcap if float_mktcap else 0.0
    best_disc = float(disc[keep].min()) if keep.any() else 1.0
    metrics = {"block_float_ratio": ratio, "block_best_discount": best_disc}
    if not keep.any() or ratio < th.d4a_block_ratio_base:
        return 0, metrics
    s_ratio = score_ge(ratio, th.d4a_block_ratio_base, th.d4a_block_ratio_full)
    s_disc = score_le(best_disc, th.d4a_block_discount_base,
                      th.d4a_block_discount_full)
    return int(round((s_ratio + s_disc) / 2)), metrics


def _s_north(north, th):
    if north is None or len(north) == 0:
        return None, {}
    pct_col = _find_col(north, "持股比例") or _find_col(north, "占比")
    if pct_col is None:
        return None, {}
    s = pd.to_numeric(north[pct_col], errors="coerce").dropna()
    if len(s) < 2:
        return None, {}
    if s.abs().median() > 1.5:
        s = s / 100.0
    change = float(s.iloc[-1] - s.iloc[-min(20, len(s))])
    metrics = {"north_pct_change": change}
    return score_ge(change, th.d4a_north_base, th.d4a_north_full), metrics


def _s_survey(survey, th):
    if survey is None or len(survey) == 0:
        return None, {}
    n = len(survey)
    metrics = {"survey_institutions": n}
    return score_ge(n, th.d4a_survey_base, th.d4a_survey_full), metrics


def _s_margin(margin, float_mktcap, th):
    if margin is None or len(margin) == 0:
        return None, {}
    bal_col = _find_col(margin, "融资余额")
    if bal_col is None:
        return None, {}
    s = pd.to_numeric(margin[bal_col], errors="coerce").dropna()
    if len(s) < 2:
        return None, {}
    base = s.iloc[-min(20, len(s))]
    growth = float(s.iloc[-1] / base - 1) if base else 0.0
    cap_ok = (not float_mktcap) or (s.iloc[-1] / float_mktcap <= th.d4a_margin_cap)
    metrics = {"margin_growth": growth}
    if not cap_ok:
        return 3, metrics
    return score_ge(growth, th.d4a_margin_base, th.d4a_margin_full), metrics


# --- 4B price-action feeds -------------------------------------------------

def _s_net_inflow(fund_flow, float_mktcap, th):
    if fund_flow is None or len(fund_flow) == 0 or not float_mktcap:
        return None, {}
    col = _find_col(fund_flow, "主力", "净流入") or _find_col(fund_flow, "主力净流入")
    if col is None:
        return None, {}
    s = pd.to_numeric(fund_flow[col], errors="coerce").fillna(0)
    last5 = float(s.iloc[-5:].sum())
    last10 = float(s.iloc[-10:].sum())
    ratio = last10 / float_mktcap
    metrics = {"inflow5": last5, "inflow10_float_ratio": ratio}
    if last5 < 0:
        return 0, metrics
    return score_ge(ratio, th.d4b_inflow_base, th.d4b_inflow_full), metrics


def _s_key_kline(daily, th):
    if daily is None or len(daily) < 6:
        return None, {}
    last = daily.iloc[-1]
    prev = daily.iloc[-2]
    rng_prev = daily.iloc[-21:-1]
    vol_ma = float(rng_prev["volume"].mean()) or 1.0
    vol_ratio = float(last["volume"]) / vol_ma
    pct = last["close"] / prev["close"] - 1
    amp = (last["high"] - last["low"]) / prev["close"]
    turnover = float(last.get("turnover", 0) or 0)
    gap_up = last["low"] > prev["high"]
    gap_pct = (last["low"] / prev["high"] - 1) if gap_up else 0.0

    limit_up = pct >= 0.095 and vol_ratio >= 2 and turnover >= 0.05
    big_gap = gap_up and gap_pct >= 0.03
    big_bull = amp >= 0.07 and vol_ratio >= 3 and last["close"] > last["open"]
    metrics = {"vol_ratio": vol_ratio, "gap_up_pct": gap_pct,
               "limit_up": bool(limit_up), "big_bottom_bull": bool(big_bull)}
    if limit_up and big_gap:
        return 10, metrics
    if limit_up or big_gap or big_bull:
        return 6, metrics
    return 0, metrics


def _s_gap(daily, d1_metrics, th):
    """4B.3 distinguish breakout gap (+) from exhaustion gap (-)."""
    if daily is None or len(daily) < 6:
        return None, {}
    last = daily.iloc[-1]
    prev = daily.iloc[-2]
    if not (last["low"] > prev["high"]):
        return 3, {"gap": "none"}  # neutral, no gap to classify
    upper_shadow = last["high"] - max(last["close"], last["open"])
    body = abs(last["close"] - last["open"]) or 1e-9
    long_shadow = upper_shadow > 1.5 * body
    overextended = d1_metrics.get("start_gain_from_platform", 0) > 0.80
    if long_shadow and overextended:
        return 0, {"gap": "exhaustion"}  # 衰竭跳空 扣分
    return 10, {"gap": "breakout"}       # 突破跳空 加分


def _s_chip_concentration(chip, th):
    if chip is None or len(chip) < 25:
        return None, {}
    lo_c = _find_col(chip, "90", "低")
    hi_c = _find_col(chip, "90", "高")
    cost_c = _find_col(chip, "平均成本")
    if lo_c is None or hi_c is None:
        return None, {}
    def width(row):
        mid = (row[lo_c] + row[hi_c]) / 2.0
        return (row[hi_c] - row[lo_c]) / mid if mid else np.nan
    w_now = width(chip.iloc[-1])
    w_prev = width(chip.iloc[-22])
    narrow = (w_prev - w_now) / w_prev if w_prev else 0.0
    cost_up = 0.0
    if cost_c is not None:
        c0 = float(chip[cost_c].iloc[-22])
        cost_up = float(chip[cost_c].iloc[-1]) / c0 - 1 if c0 else 0.0
    metrics = {"chip_narrow": narrow, "chip_cost_up": cost_up}
    s_n = score_ge(narrow, th.d4b_chip_narrow_base, th.d4b_chip_narrow_full)
    s_c = 10 if cost_up >= 0.05 else (3 if cost_up > 0 else 0)
    return int(round((s_n + s_c) / 2)), metrics


def _s_vol_turnover(daily, th):
    if daily is None or len(daily) < 21:
        return None, {}
    last = daily.iloc[-1]
    vol_ma = float(daily["volume"].iloc[-21:-1].mean()) or 1.0
    vol_ratio = float(last["volume"]) / vol_ma
    turn = float(last.get("turnover", 0) or 0)
    metrics = {"vol_ratio_5": vol_ratio, "turnover": turn}
    s_v = score_band(vol_ratio, th.d4b_vol_ratio_lo, th.d4b_vol_ratio_hi,
                     th.d4b_vol_ratio_lo, th.d4b_vol_ratio_hi)
    s_t = score_band(turn, 0.07, 0.10, th.d4b_turnover_lo, th.d4b_turnover_hi)
    return int(round((s_v + s_t) / 2)), metrics


def score_d4(*, daily=None, lhb=None, block=None, north=None, survey=None,
             margin=None, fund_flow=None, chip=None, d1_metrics=None,
             float_mktcap=None, th=DEFAULT_THRESHOLDS) -> DimensionResult:
    d1_metrics = d1_metrics or {}
    metrics = {}
    sub_a, sub_b = {}, {}

    for key, (val, m) in {
        "lhb": _s_lhb(lhb, float_mktcap, th),
        "block": _s_block(block, float_mktcap, th),
        "north": _s_north(north, th),
        "survey": _s_survey(survey, th),
        "margin": _s_margin(margin, float_mktcap, th),
    }.items():
        metrics.update(m)
        if val is not None:
            sub_a[key] = val

    for key, (val, m) in {
        "net_inflow": _s_net_inflow(fund_flow, float_mktcap, th),
        "key_kline": _s_key_kline(daily, th),
        "gap": _s_gap(daily, d1_metrics, th),
        "chip": _s_chip_concentration(chip, th),
        "vol_turnover": _s_vol_turnover(daily, th),
    }.items():
        metrics.update(m)
        if val is not None:
            sub_b[key] = val

    score_a = weighted(sub_a, D4A_SUB_WEIGHTS) if sub_a else 0.0
    score_b = weighted(sub_b, D4B_SUB_WEIGHTS) if sub_b else 0.0

    # contribution-equivalent score on the 0..10 scale so DimensionResult.
    # contribution (= score*weight*10) matches d4a*0.12 + d4b*0.13 (spec 4.1).
    total_w = D4A_WEIGHT + D4B_WEIGHT
    blended = (score_a * D4A_WEIGHT + score_b * D4B_WEIGHT) / total_w

    sub = {f"4A.{k}": v for k, v in sub_a.items()}
    sub.update({f"4B.{k}": v for k, v in sub_b.items()})
    metrics["d4a_score"] = round(score_a, 2)
    metrics["d4b_score"] = round(score_b, 2)
    return DimensionResult("d4", blended, total_w, subscores=sub,
                           metrics=metrics)
