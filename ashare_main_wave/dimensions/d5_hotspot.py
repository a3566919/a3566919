"""Dimension 5 -- 复合热点风口共振 (spec 3 维度5, weight 15%).

Kept pure: the orchestrator distils raw provider data into the structured
inputs below so this stays trivially unit-testable.

Inputs
------
sector_percentile : float | None
    Best (smallest) market-wide percentile across the stock's concepts for
    10-day gain, 0.0 = strongest, 1.0 = weakest.
strong_concept_count : int
    Number of the stock's concepts whose 10-day gain is in the top 20%.
limitups_in_sector : int | None
    Cumulative limit-ups in the leading sector over the last 5 days.
catalyst_count : int
    Distinct policy / event catalysts within 30 days (spec 5.4).
leader_rank : int | None
    The stock's 20-day-gain rank inside its sector (1 = sector leader).
"""

from __future__ import annotations

from ..config import D5_SUB_WEIGHTS, DEFAULT_THRESHOLDS, DIMENSION_WEIGHTS
from ..types import DimensionResult
from ..utils import score_ge, weighted


def score_d5(*, sector_percentile=None, strong_concept_count=0,
             limitups_in_sector=None, catalyst_count=0, leader_rank=None,
             th=DEFAULT_THRESHOLDS) -> DimensionResult:
    w = DIMENSION_WEIGHTS["d5"]
    metrics = {
        "sector_percentile": sector_percentile,
        "strong_concept_count": strong_concept_count,
        "limitups_in_sector": limitups_in_sector,
        "catalyst_count": catalyst_count,
        "leader_rank": leader_rank,
    }

    # one-vote veto (spec 4.4.5): sector 20d gain in bottom 1/3 & no catalyst
    if (sector_percentile is not None
            and sector_percentile > (1 - th.d5_sector_veto_quantile)
            and catalyst_count == 0):
        return DimensionResult("d5", 0.0, w, metrics=metrics,
                               veto="所属板块涨幅后 1/3 且无政策催化")

    sub = {}
    if sector_percentile is not None:
        # 5.1 strength: top 5% -> 10, top 20% -> 3..6
        if sector_percentile <= th.d5_strength_full_pct:
            sub["strength"] = 10
        elif sector_percentile <= th.d5_strength_base_pct / 2:
            sub["strength"] = 6
        elif sector_percentile <= th.d5_strength_base_pct:
            sub["strength"] = 3
        else:
            sub["strength"] = 0

    sub["tags"] = score_ge(strong_concept_count, th.d5_tags_base,
                           th.d5_tags_full)

    if limitups_in_sector is not None:
        sub["limitups"] = score_ge(limitups_in_sector, th.d5_limitups_base,
                                   th.d5_limitups_full)

    # 5.4 is analyst-supplied; when no catalyst is provided, omit the
    # sub-indicator (graceful, like strength/limitups/leader) instead of
    # injecting a hard 0 that depresses every un-annotated stock (audit M7).
    if catalyst_count:
        sub["catalyst"] = score_ge(catalyst_count, th.d5_catalyst_base,
                                   th.d5_catalyst_full)

    if leader_rank is not None:
        if leader_rank == 1:
            sub["leader"] = 10
        elif leader_rank <= th.d5_leader_rank_base:
            sub["leader"] = 6
        else:
            sub["leader"] = 0

    score = weighted(sub, D5_SUB_WEIGHTS)
    return DimensionResult("d5", score, w, subscores=sub, metrics=metrics)
