"""Composite 5-dimension scoring, grading and the 7 one-vote vetoes.

Total scale (spec 4.1-4.4):
    total = d1c + d2c + d3c + d4c + d5c                   in [0, 100]
where each ``*c`` is ``DimensionResult.contribution`` = score*weight*10, i.e.
points out of 100 (spec 4.1 case study: 17.5+18+12+22.5+15 = 85).

Note on veto rule 6 ("主力行为 < 30%"): the spec's section-10 pseudo-code
checks only the 4B slice (``d4 < 0.13*0.3*10``) -- that is a transcription
bug. We implement the section-4.4 intent: the *whole* Dimension-4 score
< 30% of its max.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from .config import DEFAULT_THRESHOLDS, GRADE_BOUNDS, GRADE_DEFAULT
from .data import DataProvider
from .dimensions import score_d1, score_d2, score_d3, score_d4, score_d5
from .types import StockScore
from .utils import offset_date


def grade_for(total: float) -> str:
    for name, bound in GRADE_BOUNDS:
        if total >= bound:
            return name
    return GRADE_DEFAULT


def _is_restricted(name: Optional[str], basic: Optional[dict]) -> Optional[str]:
    """Veto rule 4: ST / *ST / 退市风险 / 立案调查."""
    nm = name or ""
    if basic:
        nm = str(basic.get("股票简称") or basic.get("简称") or nm)
    # A-share risk-warning names carry ST/*ST/SST as a leading token, not as
    # an arbitrary substring (audit M5: avoid "STARK"-type false positives).
    nm_u = nm.upper().replace(" ", "")
    if nm_u.startswith(("ST", "*ST", "SST", "S*ST")):
        return f"风险警示股（{nm}）"
    blob = " ".join(str(v) for v in basic.values()) if basic else ""
    for bad in ("退市风险", "立案调查", "立案"):
        if bad in blob:
            return f"命中“{bad}”"
    return None


def _veto_momentum_decay(daily: pd.DataFrame, th) -> Optional[str]:
    """Veto rule 7: after breakout, 3 consecutive days with no new high *and*
    volume decaying 50%+."""
    if daily is None or len(daily) < 25:
        return None
    n = th.veto_no_new_high_days
    recent = daily.iloc[-n:]
    prior_high = float(daily["high"].iloc[-(n + 20):-n].max())
    no_new_high = bool((recent["high"] < prior_high).all())
    base_vol = float(daily["volume"].iloc[-(n + 5):-n].mean()) or 1.0
    vol_decay = float(recent["volume"].mean()) / base_vol
    if no_new_high and vol_decay <= (1 - th.veto_vol_decay):
        return "突破后连续3日未创新高且量能萎缩50%+（动能衰竭）"
    return None


def _sector_inputs(provider: DataProvider, symbol: str, end: str,
                   catalysts: int):
    """Distil provider concept data into the structured Dimension-5 inputs.
    Returns kwargs for :func:`score_d5`. Gracefully degrades to catalyst-only
    when concept feeds are unavailable (offline / back-test)."""
    out = {"catalyst_count": catalysts}
    concepts = provider.concepts_of(symbol)
    rank = provider.concept_rank()
    if not concepts or rank is None or len(rank) == 0:
        return out
    name_col = next((c for c in rank.columns if "名称" in str(c)), None)
    if name_col is None:
        return out
    total = len(rank)
    rank = rank.reset_index(drop=True)
    pos = {str(r[name_col]): i for i, r in rank.iterrows()}
    pcts = [pos[c] / total for c in concepts if c in pos]
    if pcts:
        out["sector_percentile"] = min(pcts)
        out["strong_concept_count"] = sum(1 for p in pcts if p <= 0.20)
    return out


def score_stock(symbol: str, end_date: str, provider: DataProvider, *,
                catalysts: int = 0, leader_rank: Optional[int] = None,
                limitups_in_sector: Optional[int] = None,
                stock_name: Optional[str] = None,
                th=DEFAULT_THRESHOLDS) -> StockScore:
    """Score one stock as of ``end_date`` (YYYYMMDD).

    ``catalysts``/``leader_rank``/``limitups_in_sector`` are qualitative
    Dimension-5 inputs the caller supplies from research/news (the spec treats
    5.4/5.5 as analyst-judged).
    """
    monthly = provider.kline(symbol, "monthly", "20180101", end_date, "qfq")
    weekly = provider.kline(symbol, "weekly",
                            offset_date(end_date, -540), end_date, "qfq")
    daily = provider.kline(symbol, "daily",
                           offset_date(end_date, -260), end_date, "qfq")

    basic = provider.basic_info(symbol)
    float_mktcap = _float_mktcap(basic)

    res = StockScore(symbol=symbol, date=end_date, total=0.0,
                     grade=GRADE_DEFAULT)

    d1 = score_d1(monthly, provider.chip(symbol), th=th)
    d2 = score_d2(monthly, weekly, d1.metrics, th=th)
    d3 = score_d3(daily, d2.metrics, th=th)
    d4 = score_d4(daily=daily, lhb=provider.lhb(symbol, offset_date(end_date, -30), end_date),
                  block=provider.block_trade(symbol, offset_date(end_date, -60), end_date),
                  north=provider.northbound(symbol, offset_date(end_date, -30), end_date),
                  survey=provider.institution_survey(symbol, offset_date(end_date, -30), end_date),
                  margin=provider.margin(symbol, offset_date(end_date, -30), end_date),
                  fund_flow=provider.fund_flow(symbol), chip=provider.chip(symbol),
                  d1_metrics={**d1.metrics, **d3.metrics},
                  float_mktcap=float_mktcap, th=th)
    d5 = score_d5(leader_rank=leader_rank,
                  limitups_in_sector=limitups_in_sector,
                  th=th,
                  **_sector_inputs(provider, symbol, end_date, catalysts))

    res.dimensions = {"d1": d1, "d2": d2, "d3": d3, "d4": d4, "d5": d5}

    # --- collect one-vote vetoes (spec 4.4) ---
    vetoes = []
    for dim in (d1, d2, d3, d5):
        if dim.veto:
            vetoes.append(f"[{dim.name}] {dim.veto}")

    restricted = _is_restricted(stock_name, basic)
    if restricted:
        vetoes.append(f"[rule4] {restricted}")

    if d4.score < 10 * th.veto_d4_min_ratio:   # rule 6 (fixed; see module doc)
        vetoes.append(f"[rule6] 主力行为得分 {d4.score:.1f}/10 < 30%")

    decay = _veto_momentum_decay(daily, th)
    if decay:
        vetoes.append(f"[rule7] {decay}")

    if vetoes:
        res.vetoes = vetoes
        res.total = 0.0
        res.grade = "D"
        res.notes.append("一票否决触发，归入 D 级")
        return res

    # contribution == score*weight*10 is already points-out-of-100
    # (spec 4.1 case study: 17.5+18+12+22.5+15 = 85). Do NOT re-scale.
    res.total = round(sum(d.contribution for d in res.dimensions.values()), 2)
    res.grade = grade_for(res.total)
    return res


def _float_mktcap(basic: Optional[dict]) -> Optional[float]:
    if not basic:
        return None
    for k, v in basic.items():
        if "流通市值" in str(k):
            try:
                return float(v)
            except (TypeError, ValueError):
                return None
    return None
