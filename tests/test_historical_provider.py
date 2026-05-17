"""Point-in-time guarantees for :class:`HistoricalFrameProvider`.

The crown-jewel property here is the *look-ahead guard*: once ``as_of`` is
set, no returned frame may contain a bar dated after it (and an earlier
``end`` argument tightens the clamp further).  Weekly/monthly frames must be
resampled *from the clipped daily* so they cannot leak future bars either.

All data is closed-form (linspace) -- no RNG, no network, no akshare.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ashare_main_wave.data import HistoricalFrameProvider


def _daily_frame(periods=520, start="2022-01-03"):
    """A 2-year business-day daily OHLCV frame with a monotone close path."""
    dates = pd.bdate_range(start=start, periods=periods)
    closes = np.linspace(10.0, 50.0, periods)
    opens = np.concatenate([[closes[0]], closes[:-1]])
    return pd.DataFrame({
        "date": dates,
        "open": opens,
        "close": closes,
        "high": np.maximum(opens, closes) * 1.01,
        "low": np.minimum(opens, closes) * 0.99,
        "volume": np.linspace(8.0e5, 2.0e6, periods),
        "amount": closes * 1.0e6,
        "turnover": np.full(periods, 0.05),
    })


# ---------------------------------------------------------------------------
# Look-ahead guard on daily frames
# ---------------------------------------------------------------------------
class TestLookAheadGuardDaily:
    def test_as_of_mid_series_hides_all_future_bars(self):
        d = _daily_frame()
        mid = d["date"].iloc[len(d) // 2]
        p = HistoricalFrameProvider({"X": {"daily": d}})
        p.as_of = mid

        out = p.kline("X", "daily", "20000101", "20991231", "qfq")
        assert len(out) > 0
        # NOT a single row may be dated after as_of.
        assert (out["date"] <= mid).all()
        assert out["date"].max() == mid
        # Everything up to and including as_of is retained.
        assert len(out) == int((d["date"] <= mid).sum())

    def test_as_of_none_returns_all_respecting_end(self):
        d = _daily_frame()
        p = HistoricalFrameProvider({"X": {"daily": d}})
        p.as_of = None

        full = p.kline("X", "daily", "20000101", "20991231", "qfq")
        assert len(full) == len(d)
        assert full["date"].max() == d["date"].iloc[-1]

        # `end` still clamps even with as_of=None.
        cut = d["date"].iloc[100]
        clipped = p.kline("X", "daily", "20000101",
                          pd.Timestamp(cut).strftime("%Y%m%d"), "qfq")
        assert clipped["date"].max() == cut
        assert len(clipped) == 101

    def test_end_earlier_than_as_of_wins_min(self):
        d = _daily_frame()
        p = HistoricalFrameProvider({"X": {"daily": d}})
        p.as_of = d["date"].iloc[-1]            # far in the future
        early = d["date"].iloc[100]

        out = p.kline("X", "daily", "20000101",
                      pd.Timestamp(early).strftime("%Y%m%d"), "qfq")
        # min(as_of, end) -> the earlier `end` bound wins.
        assert out["date"].max() == early

    def test_as_of_earlier_than_end_wins_min(self):
        d = _daily_frame()
        p = HistoricalFrameProvider({"X": {"daily": d}})
        as_of = d["date"].iloc[80]
        p.as_of = as_of

        out = p.kline("X", "daily", "20000101", "20991231", "qfq")
        assert out["date"].max() == as_of

    def test_start_lower_bound_applied(self):
        d = _daily_frame()
        p = HistoricalFrameProvider({"X": {"daily": d}})
        p.as_of = None
        lo = d["date"].iloc[200]
        out = p.kline("X", "daily",
                      pd.Timestamp(lo).strftime("%Y%m%d"), "20991231", "qfq")
        assert out["date"].min() >= lo

    def test_missing_symbol_returns_empty_normalised_frame(self):
        p = HistoricalFrameProvider({"X": {"daily": _daily_frame()}})
        out = p.kline("ZZZ", "daily", "20000101", "20991231", "qfq")
        assert len(out) == 0
        assert {"date", "open", "close", "high", "low"} <= set(out.columns)


# ---------------------------------------------------------------------------
# Weekly / monthly resampled from daily only -- no future leakage
# ---------------------------------------------------------------------------
class TestResampleFromDaily:
    @pytest.mark.parametrize("period", ["weekly", "monthly"])
    def test_no_future_bars_vs_as_of(self, period):
        d = _daily_frame()
        mid = d["date"].iloc[len(d) // 2]
        p = HistoricalFrameProvider({"X": {"daily": d}})
        p.as_of = mid

        out = p.kline("X", period, "20000101", "20991231", "qfq")
        assert len(out) > 0
        # The underlying daily rows feeding the resample are clipped to
        # date <= as_of ...
        clipped = d[d["date"] <= mid]
        assert clipped["date"].max() == mid
        # ... and the provider additionally keeps only *fully closed*
        # periods: a resampled bar is labelled by its period END, and any
        # trailing half-formed week/month whose label lands after the as-of
        # cap is dropped.  So NO resampled label may exceed as_of -- this is
        # the strict look-ahead guarantee for resampled frames.
        assert (out["date"] <= mid).all()
        assert out["date"].max() <= mid

    def test_ohlc_aggregation_is_sane(self):
        d = _daily_frame()
        mid = d["date"].iloc[len(d) // 2]
        p = HistoricalFrameProvider({"X": {"daily": d}})
        p.as_of = mid

        wk = p.kline("X", "weekly", "20000101", "20991231", "qfq")

        clipped = d[d["date"] <= mid].copy()
        g = clipped.set_index("date").resample("W-FRI")
        ref = g.agg({"open": "first", "high": "max", "low": "min",
                     "close": "last", "volume": "sum"}).dropna(
                         subset=["close"]).reset_index()
        # The provider keeps only *fully closed* periods: a bucket is labelled
        # by its period END (W-FRI), and a trailing half-formed week whose
        # label is after the as-of cap is dropped (documented "last closed
        # week" semantics).  Replicate that final clip in the reference.
        ref = ref[ref["date"] <= mid].reset_index(drop=True)

        assert len(wk) == len(ref)
        # open == first of bucket, close == last, high == max, low == min,
        # volume == sum.
        assert np.allclose(wk["open"].to_numpy(), ref["open"].to_numpy())
        assert np.allclose(wk["close"].to_numpy(), ref["close"].to_numpy())
        assert np.allclose(wk["high"].to_numpy(), ref["high"].to_numpy())
        assert np.allclose(wk["low"].to_numpy(), ref["low"].to_numpy())
        assert np.allclose(wk["volume"].to_numpy(), ref["volume"].to_numpy())
        # per-bar internal consistency
        assert (wk["high"] >= wk["low"]).all()
        assert (wk["high"] >= wk["close"]).all()
        assert (wk["low"] <= wk["close"]).all()

    def test_supplied_weekly_used_verbatim_not_resampled(self):
        d = _daily_frame()
        # a deliberately distinct weekly frame so we can tell it apart
        wk_dates = pd.date_range("2022-01-07", periods=20, freq="W-FRI")
        supplied = pd.DataFrame({
            "date": wk_dates,
            "open": np.full(20, 1.0), "close": np.full(20, 2.0),
            "high": np.full(20, 3.0), "low": np.full(20, 0.5),
            "volume": np.full(20, 7.0), "amount": np.full(20, 7.0),
        })
        p = HistoricalFrameProvider({"X": {"daily": d, "weekly": supplied}})
        p.as_of = None
        out = p.kline("X", "weekly", "20000101", "20991231", "qfq")
        # values come from the supplied frame, not a daily resample
        assert set(out["close"].unique()) == {2.0}

    def test_resample_still_clipped_by_explicit_end(self):
        d = _daily_frame()
        p = HistoricalFrameProvider({"X": {"daily": d}})
        p.as_of = None
        cut = d["date"].iloc[300]
        out = p.kline("X", "monthly", "20000101",
                      pd.Timestamp(cut).strftime("%Y%m%d"), "qfq")
        # with as_of=None the explicit `end` is the cap; only fully-closed
        # months at/under it survive, so no label exceeds `cut`.
        assert (out["date"] <= cut).all()
        assert out["date"].max().to_period("M") <= cut.to_period("M")


# ---------------------------------------------------------------------------
# Optional feeds / concepts / basic routing
# ---------------------------------------------------------------------------
class TestFeedRouting:
    def _provider(self):
        d = _daily_frame()
        feed_dates = d["date"].iloc[:50]
        feeds = {"X": {
            "lhb": pd.DataFrame({"date": feed_dates, "净额": np.arange(50)}),
            "north": pd.DataFrame({"date": feed_dates,
                                   "持股比例": np.linspace(1, 2, 50)}),
            "fund_flow": pd.DataFrame({"date": feed_dates,
                                       "主力净流入-净额": np.arange(50)}),
        }}
        return HistoricalFrameProvider(
            {"X": {"daily": d}}, feeds=feeds,
            basic={"X": {"股票简称": "演示", "流通市值": 8.0e9}},
            concepts={"X": ["人工智能"]},
            concept_rank=pd.DataFrame({"名称": ["人工智能", "其它"]}))

    def test_present_feeds_return_frames(self):
        p = self._provider()
        assert p.lhb("X", "20000101", "20991231") is not None
        assert p.northbound("X", "20000101", "20991231") is not None
        assert p.fund_flow("X") is not None

    def test_absent_feeds_return_none(self):
        p = self._provider()
        # not supplied for X -> None (graceful degradation contract)
        assert p.block_trade("X", "20000101", "20991231") is None
        assert p.institution_survey("X", "20000101", "20991231") is None
        assert p.margin("X", "20000101", "20991231") is None
        assert p.chip("X") is None
        # unknown symbol -> None across the board
        assert p.lhb("ZZZ", "20000101", "20991231") is None

    def test_feed_respects_as_of_clip(self):
        p = self._provider()
        all_dates = p.lhb("X", "20000101", "20991231")["date"]
        p.as_of = all_dates.iloc[10]
        clipped = p.lhb("X", "20000101", "20991231")
        assert (clipped["date"] <= all_dates.iloc[10]).all()
        assert len(clipped) == 11

    def test_concepts_and_basic_routing(self):
        p = self._provider()
        assert p.concepts_of("X") == ["人工智能"]
        assert p.concepts_of("ZZZ") is None
        assert p.basic_info("X") == {"股票简称": "演示", "流通市值": 8.0e9}
        assert p.basic_info("ZZZ") is None
        assert p.concept_rank() is not None

    def test_concept_rank_none_when_absent(self):
        p = HistoricalFrameProvider({"X": {"daily": _daily_frame()}})
        assert p.concept_rank() is None
        assert p.basic_info("X") is None
        assert p.concepts_of("X") is None
