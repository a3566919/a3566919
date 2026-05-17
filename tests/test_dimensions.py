"""Per-dimension scorers on synthetic frames.

Each dimension gets a high-scoring case and a veto case; we assert the
*contracts* (veto set/unset, subscore & metric keys present, score band)
rather than brittle exact floats, per the spec's "implied band" guidance.
"""

import conftest as c

from ashare_main_wave.dimensions import (score_d1, score_d2, score_d3,
                                         score_d4, score_d5)
from ashare_main_wave.types import DimensionResult


# --- Dimension 1: 月线盘整 -------------------------------------------------
class TestD1:
    def test_high_scoring_textbook_base(self):
        r = score_d1(c.textbook_monthly(), c.strong_chip())
        assert isinstance(r, DimensionResult)
        assert r.name == "d1"
        assert r.veto is None
        assert r.weight == 0.25
        assert r.score >= 6.0
        # metrics + subscores expose the documented keys
        for k in ("drawdown", "consol_months", "amplitude",
                  "vol_shrink_ratio", "peak_high", "base_low"):
            assert k in r.metrics
        for k in ("drawdown", "amplitude", "volume_shrink", "duration"):
            assert k in r.subscores
        assert "chip" in r.subscores            # chip feed supplied
        assert 0 <= r.score <= 10

    def test_veto_short_consolidation(self):
        r = score_d1(c.short_consolidation_monthly(), None)
        assert r.veto is not None
        assert "< 6 月" in r.veto
        assert r.score == 0.0
        assert r.metrics.get("consol_months", 99) < 6

    def test_veto_wide_amplitude_still_falling(self):
        r = score_d1(c.wide_amplitude_monthly(), None)
        assert r.veto is not None
        assert "振幅" in r.veto
        assert r.score == 0.0
        assert r.metrics["amplitude"] > 2.5

    def test_insufficient_data_vetoes(self):
        short = c.textbook_monthly().iloc[:5]
        r = score_d1(short, None)
        assert r.veto is not None
        assert r.score == 0.0


# --- Dimension 2: 月线突破 -------------------------------------------------
class TestD2:
    def _d1m(self, monthly):
        return score_d1(monthly, None).metrics

    def test_high_scoring_breakout(self):
        m = c.textbook_monthly()
        r = score_d2(m, c.textbook_weekly(), self._d1m(m))
        assert r.name == "d2"
        assert r.veto is None
        assert r.weight == 0.20
        assert r.score >= 8.0
        for k in ("breakout_excess", "vol_mult", "body_ratio",
                  "prev12_high"):
            assert k in r.metrics
        for k in ("excess", "body", "volume"):
            assert k in r.subscores
        assert "weekly" in r.subscores          # weekly feed supplied

    def test_veto_low_volume_breakout(self):
        m = c.low_vol_breakout_monthly()
        r = score_d2(m, c.textbook_weekly(breakout=False), self._d1m(m))
        assert r.veto is not None
        assert "1.5x" in r.veto                  # 突破量能 < 1.5x
        assert r.score == 0.0

    def test_veto_no_effective_breakout(self):
        m = c.weak_breakout_monthly()
        r = score_d2(m, None, self._d1m(m))
        assert r.veto is not None
        assert r.score == 0.0


# --- Dimension 3: 日线趋势 -------------------------------------------------
class TestD3:
    D2M = {"prev12_high": 45.3}

    def test_high_scoring_uptrend(self):
        r = score_d3(c.textbook_daily(), self.D2M)
        assert r.name == "d3"
        assert r.veto is None
        assert r.weight == 0.15
        assert r.score >= 6.0
        for k in ("ma_stack", "macd_golden_cross", "macd_above_zero",
                  "higher_low_groups", "start_gain_from_platform"):
            assert k in r.metrics
        for k in ("higher_lows", "ma_stack", "start_gain", "macd"):
            assert k in r.subscores

    def test_veto_below_ma20(self):
        r = score_d3(c.below_ma20_daily(), self.D2M)
        assert r.veto is not None
        assert "MA20" in r.veto
        assert r.score == 0.0

    def test_veto_macd_dead_cross(self):
        r = score_d3(c.macd_dead_daily(), self.D2M)
        assert r.veto is not None
        assert "MACD" in r.veto
        assert r.score == 0.0

    def test_insufficient_data_vetoes(self):
        r = score_d3(c.textbook_daily(n=40), self.D2M)
        assert r.veto is not None
        assert r.score == 0.0


# --- Dimension 4: 主力行为 -------------------------------------------------
class TestD4:
    def test_high_scoring_full_feeds(self):
        r = score_d4(daily=c.textbook_daily(), lhb=c.strong_lhb(),
                     block=c.strong_block(), north=c.strong_north(),
                     survey=c.strong_survey(), margin=c.strong_margin(),
                     fund_flow=c.strong_fund_flow(), chip=c.strong_chip(),
                     d1_metrics={"start_gain_from_platform": 0.1},
                     float_mktcap=8e9)
        assert r.name == "d4"
        assert r.veto is None                    # d4 never sets a hard veto
        assert r.weight == 0.25
        assert r.score >= 6.0
        # 4A and 4B sub-indicators are namespaced
        assert any(k.startswith("4A.") for k in r.subscores)
        assert any(k.startswith("4B.") for k in r.subscores)
        assert "d4a_score" in r.metrics and "d4b_score" in r.metrics
        assert r.metrics["lhb_inst_buy_count"] >= 1

    def test_thin_feeds_score_low_but_no_exception(self):
        # only the daily-derived 4B sub-indicators are available
        r = score_d4(daily=c.textbook_daily(), d1_metrics={},
                     float_mktcap=8e9)
        assert r.veto is None
        assert r.score < 3.0                     # would trip the rule-6 floor
        assert r.metrics["d4a_score"] is None    # 4A half absent

    def test_no_feeds_at_all_scores_zero(self):
        r = score_d4(d1_metrics={})
        assert r.score == 0.0
        assert r.veto is None


# --- Dimension 5: 热点共振 -------------------------------------------------
class TestD5:
    def test_high_scoring_resonance(self):
        r = score_d5(sector_percentile=0.0, strong_concept_count=3,
                     limitups_in_sector=12, catalyst_count=2,
                     leader_rank=1)
        assert r.name == "d5"
        assert r.veto is None
        assert r.weight == 0.15
        assert r.score >= 9.0
        for k in ("sector_percentile", "strong_concept_count",
                  "limitups_in_sector", "catalyst_count", "leader_rank"):
            assert k in r.metrics
        for k in ("strength", "tags", "limitups", "catalyst", "leader"):
            assert k in r.subscores

    def test_veto_weak_sector_no_catalyst(self):
        r = score_d5(sector_percentile=0.90, catalyst_count=0)
        assert r.veto is not None
        assert "后 1/3" in r.veto
        assert r.score == 0.0

    def test_catalyst_omitted_when_zero(self):
        # audit M7: a 0-catalyst stock must not be hard-zeroed on 5.4
        r = score_d5(sector_percentile=0.0, strong_concept_count=3,
                     catalyst_count=0)
        assert r.veto is None
        assert "catalyst" not in r.subscores
        assert r.score > 0.0
