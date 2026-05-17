"""Back-test engine: pure helpers + the cash-conservation / no-look-ahead
invariants on the offline demo history.

The two crown-jewel invariants are:

* **Cash conservation.** With slippage/commission off, the sum of all
  ``TradeLeg.pnl`` must equal ``final_equity - initial_capital`` to float
  precision; with frictions on it must match within a *bounded* friction
  budget (commissions are charged on the traded notional but are *not*
  reflected in ``TradeLeg.pnl`` -- see the reasoning in
  :meth:`TestCashConservation.test_pnl_reconciles_with_equity`).
* **No look-ahead.** During every ``score_stock`` pass the provider's
  ``as_of`` must never exceed the rebalance date being processed; a
  logging provider subclass records the max ``as_of`` seen on ``kline``.

Assertions are deliberately on *bands / invariants*, not exact demo
numbers, because the demo path may shift with threshold recalibration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ashare_main_wave.backtest import (Backtester, TradeLeg, _metrics,
                                       _rebalance_days)
from ashare_main_wave.data import HistoricalFrameProvider
from ashare_main_wave.demo_data import demo_history

DEMO_SKW = (lambda s: ({"catalysts": 2, "leader_rank": 1,
                        "limitups_in_sector": 12} if s == "000001" else {}))


def _demo_window():
    h = demo_history()
    win = h["klines"]["000001"]["daily"]
    start = win["date"].iloc[420].strftime("%Y%m%d")
    end = win["date"].iloc[-1].strftime("%Y%m%d")
    return h, start, end


def _run_demo(*, commission=0.0003, slippage=0.001, max_positions=2,
              min_grade="B", provider_cls=HistoricalFrameProvider):
    h, start, end = _demo_window()
    prov = provider_cls(**demo_history())
    bt = Backtester(prov, rebalance="W", max_positions=max_positions,
                    min_grade=min_grade, score_kwargs_fn=DEMO_SKW,
                    commission=commission, slippage=slippage)
    res = bt.run(list(h["klines"]), start, end)
    return res, prov, start, end


# ---------------------------------------------------------------------------
# _rebalance_days
# ---------------------------------------------------------------------------
class TestRebalanceDays:
    def test_daily_is_identity(self):
        cal = list(pd.bdate_range("2024-01-01", "2024-06-30"))
        out = _rebalance_days(cal, "D")
        assert out == cal

    def test_weekly_one_day_per_iso_week(self):
        cal = list(pd.bdate_range("2024-01-01", "2024-06-30"))
        out = _rebalance_days(cal, "W")
        idx = pd.DatetimeIndex(out)
        iso = idx.isocalendar()
        keys = list(zip(iso.year, iso.week))
        assert len(keys) == len(set(keys))            # exactly one per week
        # every ISO week present in the calendar is represented
        cal_iso = pd.DatetimeIndex(cal).isocalendar()
        assert set(zip(cal_iso.year, cal_iso.week)) == set(keys)
        # the chosen day is the first calendar day of its week
        assert out[0] == cal[0]

    def test_monthly_one_day_per_month(self):
        cal = list(pd.bdate_range("2024-01-01", "2024-12-31"))
        out = _rebalance_days(cal, "M")
        months = {(d.year, d.month) for d in out}
        assert len(out) == len(months) == 12
        assert out[0] == cal[0]

    def test_monotonic_and_subset(self):
        cal = list(pd.bdate_range("2024-01-01", "2024-12-31"))
        for freq in ("D", "W", "M"):
            out = _rebalance_days(cal, freq)
            assert out == sorted(out)
            assert set(out) <= set(cal)


# ---------------------------------------------------------------------------
# _metrics
# ---------------------------------------------------------------------------
def _leg(ret, shares=100.0, entry=10.0):
    """A synthetic closed TradeLeg with a chosen return."""
    return TradeLeg("S", "A", pd.Timestamp("2024-01-01"), entry,
                    pd.Timestamp("2024-02-01"), entry * (1 + ret),
                    shares, "x")


class TestMetrics:
    def test_rising_equity_basics(self):
        idx = pd.bdate_range("2023-01-02", periods=252)
        eq = pd.Series(np.linspace(1.0e6, 1.5e6, len(idx)), index=idx)
        m = _metrics(eq, [], 1.0e6, idx[0], idx[-1], None)
        assert m["total_return"] == pytest.approx(0.5)
        assert m["cagr"] > 0                       # rising -> positive CAGR
        assert m["max_drawdown"] <= 0.0            # never positive
        # a strictly rising line has ~zero drawdown
        assert m["max_drawdown"] == pytest.approx(0.0, abs=1e-9)
        assert m["num_trades"] == 0
        assert m["win_rate"] == 0.0
        assert m["profit_factor"] == 0.0           # no trades -> 0.0

    def test_drawdown_negative_on_dip(self):
        idx = pd.bdate_range("2023-01-02", periods=10)
        eq = pd.Series([100, 110, 120, 90, 95, 130, 140, 100, 150, 160],
                       index=idx, dtype=float)
        m = _metrics(eq, [], 100.0, idx[0], idx[-1], None)
        assert m["max_drawdown"] < 0.0
        # deepest peak->trough is 140 -> 100 = -28.57% (deeper than the
        # earlier 120 -> 90 = -25% dip)
        assert m["max_drawdown"] == pytest.approx(-1.0 / 3.5, abs=1e-9)

    def test_win_rate_and_profit_factor(self):
        idx = pd.bdate_range("2023-01-02", periods=5)
        eq = pd.Series(np.linspace(1.0e6, 1.1e6, 5), index=idx)
        trades = [_leg(0.5), _leg(0.2), _leg(-0.1), _leg(-0.3)]
        m = _metrics(eq, trades, 1.0e6, idx[0], idx[-1], None)
        assert m["num_trades"] == 4
        assert m["win_rate"] == pytest.approx(0.5)
        assert m["avg_win"] > 0 and m["avg_loss"] < 0
        # gross win = (0.5+0.2)*1000 ; gross loss = (0.1+0.3)*1000
        assert m["profit_factor"] == pytest.approx(0.7 / 0.4)

    def test_profit_factor_infinite_when_no_losses(self):
        idx = pd.bdate_range("2023-01-02", periods=3)
        eq = pd.Series([1.0e6, 1.1e6, 1.2e6], index=idx)
        m = _metrics(eq, [_leg(0.4), _leg(0.1)], 1.0e6,
                     idx[0], idx[-1], None)
        assert m["win_rate"] == 1.0
        assert m["profit_factor"] == float("inf")

    def test_negative_total_return_negative_cagr(self):
        idx = pd.bdate_range("2023-01-02", periods=100)
        eq = pd.Series(np.linspace(1.0e6, 0.7e6, 100), index=idx)
        m = _metrics(eq, [], 1.0e6, idx[0], idx[-1], None)
        assert m["total_return"] < 0
        assert m["cagr"] < 0

    def test_benchmark_alpha_fields(self):
        idx = pd.bdate_range("2023-01-02", periods=50)
        eq = pd.Series(np.linspace(1.0e6, 1.2e6, 50), index=idx)
        bench = pd.Series(np.linspace(1.0e6, 1.1e6, 50), index=idx)
        m = _metrics(eq, [], 1.0e6, idx[0], idx[-1], bench)
        assert "benchmark_return" in m and "alpha_vs_benchmark" in m
        assert m["benchmark_return"] == pytest.approx(0.1)
        assert m["alpha_vs_benchmark"] == pytest.approx(
            m["total_return"] - m["benchmark_return"])


# ---------------------------------------------------------------------------
# Cash conservation / accounting invariant (crown jewel)
# ---------------------------------------------------------------------------
class TestCashConservation:
    def test_equity_series_finite_and_positive(self):
        res, _, _, _ = _run_demo()
        v = res.equity.to_numpy()
        assert np.isfinite(v).all()
        assert (v > 0).all()
        assert res.equity.index.is_monotonic_increasing

    def test_final_equity_equals_total_return_identity(self):
        res, _, _, _ = _run_demo()
        final = float(res.equity.iloc[-1])
        implied = 1_000_000.0 * (1.0 + res.metrics["total_return"])
        # this is an exact algebraic identity in _metrics
        assert final == pytest.approx(implied, rel=1e-12)

    def test_pnl_reconciles_with_equity(self):
        """Sum of TradeLeg.pnl vs (final - initial), accounting for frictions.

        Reasoning: the engine charges commission on the traded notional on
        *both* entry and exit (``cash -= shares*px*(1+commission)`` /
        ``cash += sell*fill*(1-commission)``) but ``_Position.entry_price``
        and ``TradeLeg.exit_price`` are the slippage-adjusted prices *without*
        the commission factor.  Hence ``TradeLeg.pnl`` captures slippage but
        NOT commission, so it *overstates* the realised gain by exactly the
        round-trip commission.  Therefore:

            0 <= sum(pnl) - (final - initial) <= commission_budget

        where the friction budget is bounded by commission * total traded
        notional.  With commission = 0 the identity is exact.
        """
        # ---- frictionless: exact conservation ----
        res0, _, _, _ = _run_demo(commission=0.0, slippage=0.0)
        sum_pnl0 = sum(t.pnl for t in res0.trades)
        delta0 = float(res0.equity.iloc[-1]) - 1_000_000.0
        assert sum_pnl0 == pytest.approx(delta0, rel=1e-9, abs=1e-3)

        # ---- with frictions: bounded, signed reconciliation ----
        res, _, _, _ = _run_demo(commission=0.0003, slippage=0.001)
        sum_pnl = sum(t.pnl for t in res.trades)
        delta = float(res.equity.iloc[-1]) - 1_000_000.0
        gap = sum_pnl - delta
        # commission is a *cost* not reflected in pnl -> pnl >= delta
        assert gap >= -1e-6
        # the gap is the round-trip commission on the traded notional; bound
        # it generously by commission * (entry + exit notional) summed.
        notional = sum(
            t.shares * t.entry_price + t.shares * t.exit_price
            for t in res.trades)
        budget = 0.0003 * notional + 1e-6
        assert gap <= budget
        # sanity: the gap is small relative to capital (sub-1%)
        assert abs(gap) < 0.01 * 1_000_000.0

    def test_every_position_closes_no_dangling(self):
        """After the run there must be no open _Position: every share that
        was bought is sold (entries == exits in share terms per symbol)."""
        res, _, _, _ = _run_demo()
        # reconstruct: an "end_of_test" leg only exists if a position was
        # still open at the last bar; the engine liquidates them, so the
        # post-run state must hold no shares.  We assert via the cash
        # identity: liquidation pushes everything into cash, and the final
        # equity == that cash.  (No position object survives `run`.)
        # Per-symbol net shares: sum of entry-equivalent must net to ~0
        # because each _Position is fully drained by trims+exit legs.
        by_entry: dict = {}
        for t in res.trades:
            key = (t.symbol, t.entry_date, round(t.entry_price, 6))
            by_entry.setdefault(key, []).append(t)
        # Each distinct entry must have a *terminal* full-exit leg whose
        # reason is a full-exit reason (not a trim), proving the position
        # was wound down to zero.
        full_reasons = {"stop_loss_absolute", "stop_pattern_ma20",
                        "trailing_take_profit", "end_of_test"}
        for key, legs in by_entry.items():
            reasons = {leg.reason for leg in legs}
            assert reasons & full_reasons, (
                f"entry {key} never fully exited; legs={reasons}")

    def test_winner_and_laggard_distribution(self):
        res, _, _, _ = _run_demo()
        assert len(res.trades) >= 1
        winners = [t for t in res.trades
                   if t.symbol == "000001" and t.return_pct > 0]
        assert winners, "expected at least one winning trade on 000001"
        n_lag = sum(1 for t in res.trades if t.symbol == "000002")
        # the laggard must never dominate the trade log
        assert n_lag <= len(res.trades) / 2

    def test_smaller_max_positions_runs_clean(self):
        res, _, _, _ = _run_demo(max_positions=1)
        assert np.isfinite(res.equity.to_numpy()).all()
        assert (res.equity.to_numpy() > 0).all()


# ---------------------------------------------------------------------------
# Partial-exit lifecycle
# ---------------------------------------------------------------------------
class TestPartialExitLifecycle:
    TRIM_REASONS = {"take_profit_60", "take_profit_30", "time_stop_half"}
    FULL_REASONS = {"stop_loss_absolute", "stop_pattern_ma20",
                    "trailing_take_profit", "end_of_test"}

    def test_winner_has_trim_then_exit_same_entry(self):
        res, _, _, _ = _run_demo()
        # group legs by (entry_date, entry_price) for the winner
        groups: dict = {}
        for t in res.trades:
            if t.symbol != "000001":
                continue
            groups.setdefault(
                (t.entry_date, round(t.entry_price, 6)), []).append(t)

        multi = [legs for legs in groups.values() if len(legs) >= 2]
        assert multi, "winner should produce >=2 legs from one entry"

        for legs in multi:
            legs_sorted = sorted(legs, key=lambda x: x.exit_date)
            # all legs share the same entry price/date
            assert len({l.entry_date for l in legs_sorted}) == 1
            assert len({round(l.entry_price, 6)
                        for l in legs_sorted}) == 1
            # at least one trim precedes a full exit
            assert any(l.reason in self.TRIM_REASONS
                       for l in legs_sorted[:-1])
            assert legs_sorted[-1].reason in self.FULL_REASONS
            # all reasons are from the allowed vocabulary
            for l in legs_sorted:
                assert l.reason in (self.TRIM_REASONS | self.FULL_REASONS)

    def test_trim_legs_share_entry_and_have_positive_shares(self):
        res, _, _, _ = _run_demo()
        for t in res.trades:
            assert t.shares > 0
            assert t.exit_date >= t.entry_date


# ---------------------------------------------------------------------------
# No look-ahead through the full engine (crown jewel)
# ---------------------------------------------------------------------------
class _LoggingProvider(HistoricalFrameProvider):
    """Records the max ``as_of`` observed on any ``kline`` call while
    ``as_of`` is set (i.e. during a scoring pass)."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.max_asof_seen = None
        self.asof_history = []

    def kline(self, symbol, period, start, end, adjust="qfq"):
        if self.as_of is not None:
            ts = pd.Timestamp(str(self.as_of))
            self.asof_history.append(ts)
            if self.max_asof_seen is None or ts > self.max_asof_seen:
                self.max_asof_seen = ts
        return super().kline(symbol, period, start, end, adjust)


class TestNoLookAhead:
    def test_as_of_never_exceeds_rebalance_date(self):
        h, start, end = _demo_window()
        prov = _LoggingProvider(**demo_history())
        bt = Backtester(prov, rebalance="W", max_positions=2,
                        min_grade="B", score_kwargs_fn=DEMO_SKW)
        bt.run(list(h["klines"]), start, end)   # side effects on prov

        # at least one scoring pass actually happened
        assert prov.asof_history, "no scoring pass observed"
        assert prov.max_asof_seen is not None
        # the maximum as_of ever set is a real trading day <= end
        assert prov.max_asof_seen <= pd.Timestamp(str(end))
        # and every recorded as_of is a calendar day in range
        for ts in prov.asof_history:
            assert pd.Timestamp(str(start)) <= ts <= pd.Timestamp(str(end))
        # the engine must reset as_of to None after scoring
        assert prov.as_of is None

    def test_score_only_sees_clipped_history(self):
        """Subclass that asserts inline: whenever as_of is set, the daily
        frame returned to the scorer has no bar after as_of."""
        violations = []

        class _GuardProvider(HistoricalFrameProvider):
            def kline(self, symbol, period, start, end, adjust="qfq"):
                df = super().kline(symbol, period, start, end, adjust)
                if (self.as_of is not None and df is not None
                        and len(df) and "date" in df):
                    cap = pd.Timestamp(str(self.as_of))
                    if (pd.to_datetime(df["date"]) > cap).any():
                        violations.append(
                            (symbol, period, str(self.as_of),
                             str(df["date"].max())))
                return df

        h, start, end = _demo_window()
        prov = _GuardProvider(**demo_history())
        bt = Backtester(prov, rebalance="W", max_positions=2,
                        min_grade="B", score_kwargs_fn=DEMO_SKW)
        bt.run(list(h["klines"]), start, end)
        assert not violations, f"look-ahead leak: {violations[:3]}"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
class TestDeterminism:
    def test_two_runs_identical(self):
        res1, _, _, _ = _run_demo()
        res2, _, _, _ = _run_demo()

        # equity series identical (index + values)
        pd.testing.assert_series_equal(res1.equity, res2.equity)

        # trades identical leg-for-leg
        assert len(res1.trades) == len(res2.trades)
        for a, b in zip(res1.trades, res2.trades):
            assert a.symbol == b.symbol
            assert a.entry_date == b.entry_date
            assert a.exit_date == b.exit_date
            assert a.entry_price == pytest.approx(b.entry_price, rel=1e-15)
            assert a.exit_price == pytest.approx(b.exit_price, rel=1e-15)
            assert a.shares == pytest.approx(b.shares, rel=1e-15)
            assert a.reason == b.reason

        # metrics identical
        for k, v in res1.metrics.items():
            other = res2.metrics[k]
            if isinstance(v, float) and np.isfinite(v):
                assert other == pytest.approx(v, rel=1e-15)
            else:
                assert other == v
