"""Technical indicators (ashare_main_wave.indicators)."""

import numpy as np
import pandas as pd

from ashare_main_wave.indicators import (count_higher_low_groups, dead_cross,
                                         golden_cross, ma, macd, swing_points)


# --- ma --------------------------------------------------------------------
class TestMA:
    def test_min_periods_one_no_leading_nans(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0])
        out = ma(s, 3)
        assert not out.isna().any()
        assert out.iloc[0] == 1.0
        assert out.iloc[1] == 1.5
        assert out.iloc[2] == 2.0          # (1+2+3)/3
        assert out.iloc[3] == 3.0          # (2+3+4)/3

    def test_constant_series(self):
        s = pd.Series([5.0] * 10)
        assert (ma(s, 4) == 5.0).all()

    def test_length_preserved(self):
        s = pd.Series(np.arange(50, dtype=float))
        assert len(ma(s, 20)) == 50


# --- macd ------------------------------------------------------------------
class TestMACD:
    def test_shapes_match_input(self):
        close = pd.Series(np.arange(1, 61, dtype=float))
        dif, dea, hist = macd(close)
        assert len(dif) == len(dea) == len(hist) == 60

    def test_rising_ramp_is_bullish(self):
        # On a monotonically rising series the fast EMA leads the slow EMA,
        # so DIF, DEA and HIST are all positive at the end.
        close = pd.Series(np.arange(1, 81, dtype=float))
        dif, dea, hist = macd(close)
        assert float(dif.iloc[-1]) > 0
        assert float(dea.iloc[-1]) > 0
        assert float(hist.iloc[-1]) > 0
        # DIF leads DEA on a steady ramp.
        assert float(dif.iloc[-1]) > float(dea.iloc[-1])

    def test_falling_ramp_is_bearish(self):
        close = pd.Series(np.arange(80, 0, -1, dtype=float))
        dif, dea, hist = macd(close)
        assert float(dif.iloc[-1]) < 0
        assert float(dea.iloc[-1]) < 0

    def test_hist_is_twice_dif_minus_dea(self):
        close = pd.Series(np.linspace(10, 30, 40))
        dif, dea, hist = macd(close)
        np.testing.assert_allclose(hist.to_numpy(),
                                   2.0 * (dif - dea).to_numpy(), rtol=1e-9)


# --- golden_cross / dead_cross ---------------------------------------------
class TestCrosses:
    def test_golden_cross_detected_after_v_bottom(self):
        # falls then rises -> DIF crosses up through DEA near the end
        x = pd.Series([10, 9, 8, 7, 6, 7, 8, 9, 10, 11.0])
        dif, dea, _ = macd(x)
        assert golden_cross(dif, dea, lookback=5) is True
        assert dead_cross(dif, dea, lookback=5) is False

    def test_dead_cross_detected_after_peak(self):
        y = pd.Series([1, 2, 3, 4, 5, 4, 3, 2, 1, 0.0] * 2)
        dif, dea, _ = macd(y)
        assert dead_cross(dif, dea, lookback=3) is True

    def test_no_cross_on_pure_ramp(self):
        s = pd.Series(np.arange(1, 60, dtype=float))
        dif, dea, _ = macd(s)
        # DIF is above DEA the whole way: never crosses up *within* lookback
        assert golden_cross(dif, dea, lookback=5) is False
        assert dead_cross(dif, dea, lookback=5) is False

    def test_lookback_window_is_respected(self):
        # cross happens early; a short lookback at the end should miss it
        x = pd.Series([10, 9, 8, 7, 8, 9, 10, 11, 12, 13, 14, 15.0])
        dif, dea, _ = macd(x)
        assert golden_cross(dif, dea, lookback=2) is False


# --- swing_points ----------------------------------------------------------
class TestSwingPoints:
    def test_hand_built_series(self):
        # peaks at idx 2 (val 3) and idx 6 (val 4); troughs at 4 (1) and 8 (2)
        v = pd.Series([1, 2, 3, 2, 1, 2, 4, 3, 2, 3, 5, 4.0])
        highs, lows = swing_points(v, order=2)
        assert 2 in highs and 6 in highs
        assert 4 in lows and 8 in lows
        # endpoints (within `order` of the edge) are never returned
        assert all(2 <= i <= len(v) - 3 for i in highs + lows)

    def test_monotonic_has_no_interior_swings(self):
        v = pd.Series(np.arange(20, dtype=float))
        highs, lows = swing_points(v, order=2)
        assert highs == [] and lows == []


# --- count_higher_low_groups -----------------------------------------------
class TestHigherLowGroups:
    def test_clean_staircase_counts_groups(self):
        # textbook stair-step: each leg a higher high then a higher low
        high = pd.Series([10, 12, 11, 13, 12, 14, 13, 15, 14, 16, 15,
                          17, 16, 18.0])
        low = pd.Series([9, 11, 10, 12, 11, 13, 12, 14, 13, 15, 14,
                         16, 15, 17.0])
        groups = count_higher_low_groups(high, low, order=1)
        assert groups >= 3

    def test_flat_series_has_zero_groups(self):
        flat = pd.Series([10.0] * 25)
        assert count_higher_low_groups(flat, flat, order=2) == 0

    def test_returns_min_of_rising_highs_and_lows(self):
        # rising highs but flat lows -> bottleneck is the lows -> 0 groups
        high = pd.Series([10, 12, 11, 13, 12, 14, 13, 15.0])
        low = pd.Series([9.0] * 8)
        assert count_higher_low_groups(high, low, order=1) == 0
