"""Scoring-ladder primitives and date helpers (ashare_main_wave.utils)."""

import numpy as np
import pytest

from ashare_main_wave.utils import (offset_date, score_band, score_ge,
                                    score_le, weighted)


# --- score_ge: higher-is-better 0/3/6/10 ladder ----------------------------
class TestScoreGe:
    def test_ladder_boundaries(self):
        # base=0.5, full=0.6, implied mid=0.55
        assert score_ge(0.49, 0.5, 0.6) == 0
        assert score_ge(0.50, 0.5, 0.6) == 3      # reaches base
        assert score_ge(0.54, 0.5, 0.6) == 3
        assert score_ge(0.55, 0.5, 0.6) == 6      # reaches mid
        assert score_ge(0.59, 0.5, 0.6) == 6
        assert score_ge(0.60, 0.5, 0.6) == 10     # reaches full
        assert score_ge(0.99, 0.5, 0.6) == 10

    def test_explicit_mid(self):
        assert score_ge(7, 0, 10, mid=5) == 6
        assert score_ge(4, 0, 10, mid=5) == 3
        assert score_ge(0, 0, 10, mid=5) == 3

    def test_none_and_nan_are_zero(self):
        assert score_ge(None, 0.5, 0.6) == 0
        assert score_ge(float("nan"), 0.5, 0.6) == 0
        assert score_ge(np.nan, 0.5, 0.6) == 0


# --- score_le: lower-is-better ---------------------------------------------
class TestScoreLe:
    def test_ladder_boundaries(self):
        # base=1.8 (loosest ok), full=1.5 (strictest), implied mid=1.65
        assert score_le(1.81, 1.8, 1.5) == 0
        assert score_le(1.80, 1.8, 1.5) == 3
        assert score_le(1.66, 1.8, 1.5) == 3
        assert score_le(1.65, 1.8, 1.5) == 6
        assert score_le(1.51, 1.8, 1.5) == 6
        assert score_le(1.50, 1.8, 1.5) == 10
        assert score_le(1.10, 1.8, 1.5) == 10

    def test_none_and_nan_are_zero(self):
        assert score_le(None, 1.8, 1.5) == 0
        assert score_le(np.nan, 1.8, 1.5) == 0


# --- score_band: range indicator -------------------------------------------
class TestScoreBand:
    def test_bands(self):
        # ideal 7..12, acceptable 5..15, widened 0..20
        assert score_band(9, 7, 12, 5, 15) == 10      # inside ideal
        assert score_band(7, 7, 12, 5, 15) == 10
        assert score_band(12, 7, 12, 5, 15) == 10
        assert score_band(6, 7, 12, 5, 15) == 6       # acceptable only
        assert score_band(14, 7, 12, 5, 15) == 6
        assert score_band(3, 7, 12, 5, 15) == 3       # widened 50%
        assert score_band(20, 7, 12, 5, 15) == 3
        assert score_band(30, 7, 12, 5, 15) == 0      # outside
        assert score_band(-5, 7, 12, 5, 15) == 0

    def test_none_is_zero(self):
        assert score_band(None, 7, 12, 5, 15) == 0
        assert score_band(np.nan, 7, 12, 5, 15) == 0


# --- weighted: renormalising weighted average ------------------------------
class TestWeighted:
    def test_basic_average(self):
        assert weighted({"a": 10, "b": 0}, {"a": 1, "b": 1}) == 5.0

    def test_missing_key_renormalises(self):
        # only "a" present; its weight is renormalised to 1.0 so the missing
        # feed degrades gracefully instead of zeroing the dimension.
        assert weighted({"a": 10}, {"a": 1, "b": 3}) == 10.0
        assert weighted({"a": 6, "c": 0}, {"a": 0.25, "b": 0.5}) == 6.0

    def test_subscore_not_in_weights_is_ignored(self):
        assert weighted({"a": 10, "zzz": 10}, {"a": 1}) == 10.0

    def test_empty_returns_zero(self):
        assert weighted({}, {"a": 1}) == 0.0
        assert weighted({"x": 10}, {}) == 0.0

    def test_zero_weight_sum_returns_zero(self):
        assert weighted({"a": 10}, {"a": 0}) == 0.0

    def test_weighting_is_proportional(self):
        # 10*0.3 + 0*0.2 over (0.3+0.2) == 6.0
        assert weighted({"x": 10, "y": 0}, {"x": 0.3, "y": 0.2}) == 6.0


# --- offset_date -----------------------------------------------------------
class TestOffsetDate:
    def test_simple_shift(self):
        assert offset_date("20240301", 1) == "20240302"
        assert offset_date("20240301", -1) == "20240229"   # 2024 is a leap yr

    def test_accepts_dashed_form_and_returns_compact(self):
        assert offset_date("2024-03-01", 31) == "20240401"

    def test_year_boundary(self):
        assert offset_date("20240101", -1) == "20231231"
        assert offset_date("20231231", 1) == "20240101"

    def test_non_leap_february(self):
        assert offset_date("20230301", -1) == "20230228"

    @pytest.mark.parametrize("days", [0, 7, -30, 365, -540, -260])
    def test_roundtrip_is_consistent(self, days):
        out = offset_date("20240430", days)
        # shifting back by the same amount returns the original day
        assert offset_date(out, -days) == "20240430"
