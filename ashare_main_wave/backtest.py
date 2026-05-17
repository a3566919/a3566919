"""Walk-forward back-test engine for the 5-dimension 主升浪 system.

Design goals
------------
* **No look-ahead.** Signals on rebalance day *D* are produced by
  :func:`score_stock` seeing only data ``date <= D`` (the provider's
  ``as_of`` clamp, set/cleared by the engine). Position management on day
  *D* only ever indexes price rows ``date <= D``. Fills are close-to-close
  on day *D* (a deterministic, slightly conservative simplification).
* **Rules come from the spec, not reinvented.** Sizing uses
  :func:`risk.position_plan`; stops/trims follow spec §7.3/7.4 via
  :func:`risk.stop_loss_price` / :func:`risk.take_profit_action` /
  :func:`risk.time_stop_triggered`.
* **Partial exits** are first-class so the §7.4 ladder (trim 1/3, 1/2,
  trailing) and the §7.3.3 time-stop half-trim are modelled faithfully.

The provider must be point-in-time for a *correct* back-test:
:class:`~ashare_main_wave.data.HistoricalFrameProvider` (or AkShare with a
real historical range) qualify; the fixed-window ``DemoProvider`` does not
(use :func:`ashare_main_wave.demo_data.demo_history` instead).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd

from .config import DEFAULT_RISK, DEFAULT_THRESHOLDS
from .indicators import ma
from .risk import (position_plan, stop_loss_price, take_profit_action,
                   time_stop_triggered)
from .scoring import score_stock


@dataclass
class TradeLeg:
    symbol: str
    grade: str
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: pd.Timestamp
    exit_price: float
    shares: float
    reason: str

    @property
    def pnl(self) -> float:
        return (self.exit_price - self.entry_price) * self.shares

    @property
    def return_pct(self) -> float:
        return self.exit_price / self.entry_price - 1.0

    @property
    def bars_held(self) -> int:
        return max(0, (self.exit_date - self.entry_date).days)


@dataclass
class _Position:
    symbol: str
    grade: str
    entry_date: pd.Timestamp
    entry_price: float
    shares: float
    cap_size: str
    peak: float
    trimmed: set = field(default_factory=set)


@dataclass
class BacktestResult:
    equity: pd.Series                 # daily total equity, indexed by date
    trades: list                      # list[TradeLeg]
    metrics: dict
    params: dict
    benchmark: Optional[pd.Series] = None

    def summary(self) -> dict:
        return {**self.params, **{k: (round(v, 4) if isinstance(v, float)
                                      else v)
                                  for k, v in self.metrics.items()}}

    def trades_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "symbol": t.symbol, "grade": t.grade,
            "entry_date": t.entry_date.date(),
            "entry_price": round(t.entry_price, 3),
            "exit_date": t.exit_date.date(),
            "exit_price": round(t.exit_price, 3),
            "shares": round(t.shares, 1),
            "return_pct": round(t.return_pct, 4),
            "pnl": round(t.pnl, 2), "reason": t.reason,
        } for t in self.trades])


def _rebalance_days(calendar, freq: str):
    cal = pd.DatetimeIndex(calendar)
    if freq == "D":
        return list(cal)
    key = cal.to_period("W" if freq == "W" else "M")
    seen, out = set(), []
    for d, k in zip(cal, key):
        if k not in seen:
            seen.add(k)
            out.append(d)
    return out


def _metrics(equity: pd.Series, trades, initial: float, start, end,
             benchmark: Optional[pd.Series]):
    total_return = float(equity.iloc[-1] / initial - 1.0)
    days = max(1, (pd.Timestamp(end) - pd.Timestamp(start)).days)
    cagr = (1 + total_return) ** (365.0 / days) - 1.0
    ret = equity.pct_change().dropna()
    vol = float(ret.std() * np.sqrt(252)) if len(ret) > 1 else 0.0
    sharpe = (float(ret.mean() / ret.std() * np.sqrt(252))
              if len(ret) > 1 and ret.std() > 0 else 0.0)
    dd = float((equity / equity.cummax() - 1.0).min())

    closed = [t for t in trades]
    wins = [t.return_pct for t in closed if t.return_pct > 0]
    losses = [t.return_pct for t in closed if t.return_pct <= 0]
    gross_win = sum(t.pnl for t in closed if t.pnl > 0)
    gross_loss = -sum(t.pnl for t in closed if t.pnl < 0)
    m = {
        "total_return": total_return,
        "cagr": cagr,
        "annual_vol": vol,
        "sharpe": sharpe,
        "max_drawdown": dd,
        "num_trades": len(closed),
        "win_rate": (len(wins) / len(closed)) if closed else 0.0,
        "avg_win": float(np.mean(wins)) if wins else 0.0,
        "avg_loss": float(np.mean(losses)) if losses else 0.0,
        "profit_factor": (gross_win / gross_loss)
        if gross_loss > 0 else float("inf") if gross_win > 0 else 0.0,
    }
    if benchmark is not None and len(benchmark) > 1:
        b = benchmark.reindex(equity.index).ffill().dropna()
        if len(b) > 1:
            bret = float(b.iloc[-1] / b.iloc[0] - 1.0)
            m["benchmark_return"] = bret
            m["alpha_vs_benchmark"] = total_return - bret
    return m


class Backtester:
    """Run the system over history and report performance.

    Parameters
    ----------
    provider : a *point-in-time* DataProvider (see module docstring).
    initial_capital : starting cash.
    rebalance : ``"W"`` (default, spec §9.1 weekend screen), ``"M"`` or ``"D"``.
    max_positions : concurrent holdings cap.
    min_grade : enter only signals at/above this grade (``"A"`` default).
    score_kwargs_fn : ``symbol -> dict`` extra kwargs for :func:`score_stock`
        (qualitative D5 inputs); optional.
    cap_size_fn : ``symbol -> {"small"|"main"|"default"}`` for the absolute
        stop width (spec §7.3.1); default ``"default"`` (8%).
    commission : per-side cost fraction (round trip applied on entry & exit).
    benchmark : symbol code (priced via the provider) or a price ``Series``.
    """

    def __init__(self, provider, *, initial_capital: float = 1_000_000.0,
                 rebalance: str = "W", max_positions: int = 5,
                 min_grade: str = "A",
                 score_kwargs_fn: Optional[Callable] = None,
                 cap_size_fn: Optional[Callable] = None,
                 commission: float = 0.0003, slippage: float = 0.001,
                 risk=DEFAULT_RISK, thresholds=DEFAULT_THRESHOLDS):
        self.p = provider
        self.cap0 = float(initial_capital)
        self.rebalance = rebalance
        self.max_pos = int(max_positions)
        self.min_grade = min_grade
        self.skw = score_kwargs_fn or (lambda s: {})
        self.cap_size_fn = cap_size_fn or (lambda s: "default")
        self.commission = commission
        self.slippage = slippage
        self.risk = risk
        self.th = thresholds
        self._grade_rank = {"A+": 4, "A": 3, "B": 2, "C": 1, "D": 0}

    # -- internals --------------------------------------------------------
    def _daily(self, symbol, start, end):
        df = self.p.kline(symbol, "daily", start, end, "qfq")
        if df is None or len(df) == 0:
            return None
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date").reset_index(drop=True)

    def _set_asof(self, day):
        if hasattr(self.p, "as_of"):
            self.p.as_of = day

    def _exit_decision(self, pos: _Position, hist: pd.DataFrame):
        """Return (action, fraction, reason). action in
        {"hold","trim","exit"}. ``hist`` = symbol daily rows date<=today."""
        price = float(hist["close"].iloc[-1])
        closes = hist["close"]
        ma10 = float(ma(closes, 10).iloc[-1])
        pnl = price / pos.entry_price - 1.0

        # 1. absolute stop (spec §7.3.1) -- risk module owns the level
        sl = stop_loss_price(pos.entry_price, cap_size=pos.cap_size,
                             risk=self.risk)
        if price <= sl["candidates"]["absolute"]:
            return "exit", 1.0, "stop_loss_absolute"

        # 2. pattern stop: close < MA20 for >= recover_days (spec §7.3.2)
        rec = self.th.d3_ma20_recover_days
        if len(closes) >= rec:
            below = (closes.iloc[-rec:].to_numpy()
                     < ma(closes, 20).iloc[-rec:].to_numpy())
            if below.all():
                return "exit", 1.0, "stop_pattern_ma20"

        # 3. take-profit ladder / trailing (spec §7.4) -- decision delegated
        #    to risk.take_profit_action so the rules live in one place.
        tp = take_profit_action(pnl, below_ma10=price < ma10,
                                weekly_macd_dead=False, risk=self.risk)
        act = tp["action"]
        if pnl >= 1.0:
            if price <= pos.peak * 0.92:           # 8% trailing exit
                return "exit", 1.0, "trailing_take_profit"
        elif "减仓1/2" in act and "tp60" not in pos.trimmed:
            pos.trimmed.add("tp60")
            return "trim", 0.5, "take_profit_60"
        elif "减仓1/3" in act and "tp30" not in pos.trimmed:
            pos.trimmed.add("tp30")
            return "trim", 1 / 3, "take_profit_30"

        # 4. time stop (spec §7.3.3) -- risk.time_stop_triggered owns the rule
        held = hist[hist["date"] >= pos.entry_date]
        if "time" not in pos.trimmed:
            made_new_high = float(held["close"].max()) > pos.entry_price
            if time_stop_triggered(len(held), made_new_high, pnl,
                                   risk=self.risk):
                pos.trimmed.add("time")
                return "trim", 0.5, "time_stop_half"
        return "hold", 0.0, ""

    # -- public -----------------------------------------------------------
    def run(self, universe: Iterable[str], start: str, end: str
            ) -> BacktestResult:
        universe = [str(s) for s in universe]
        daily = {s: self._daily(s, start, end) for s in universe}
        daily = {s: d for s, d in daily.items() if d is not None and len(d)}
        if not daily:
            raise ValueError("no daily data for any symbol in universe")

        cal = sorted(set().union(*[set(d["date"]) for d in daily.values()]))
        cal = [d for d in cal
               if pd.Timestamp(str(start)) <= d <= pd.Timestamp(str(end))]
        rebal = set(_rebalance_days(cal, self.rebalance))

        cash = self.cap0
        positions: dict[str, _Position] = {}
        trades: list[TradeLeg] = []
        eq_dates, eq_vals = [], []

        for day in cal:
            # ---- manage / exit open positions (close of `day`) ----
            for sym in list(positions):
                pos = positions[sym]
                d = daily[sym]
                hist = d[d["date"] <= day]
                if len(hist) == 0:
                    continue
                price = float(hist["close"].iloc[-1])
                pos.peak = max(pos.peak, price)
                action, frac, reason = self._exit_decision(pos, hist)
                if action == "hold":
                    continue
                sell = pos.shares if action == "exit" else pos.shares * frac
                fill = price * (1 - self.slippage)
                cash += sell * fill * (1 - self.commission)
                trades.append(TradeLeg(
                    sym, pos.grade, pos.entry_date, pos.entry_price,
                    day, fill, sell, reason))
                pos.shares -= sell
                if action == "exit" or pos.shares <= 1e-6:
                    del positions[sym]

            # ---- rebalance: open new positions ----
            if day in rebal and len(positions) < self.max_pos:
                equity_now = cash + sum(
                    pos.shares * float(
                        daily[s][daily[s]["date"] <= day]["close"].iloc[-1])
                    for s, pos in positions.items())
                self._set_asof(day)
                ranked = []
                day_str = pd.Timestamp(day).strftime("%Y%m%d")
                for sym in universe:
                    if sym in positions or daily.get(sym) is None:
                        continue
                    sub = daily[sym][daily[sym]["date"] <= day]
                    if len(sub) < 60:
                        continue
                    try:
                        sc = score_stock(sym, day_str, self.p,
                                         **self.skw(sym))
                    except Exception:  # noqa: BLE001
                        continue
                    if (self._grade_rank.get(sc.grade, 0)
                            >= self._grade_rank.get(self.min_grade, 3)):
                        ranked.append((sc.total, sym, sc.grade))
                self._set_asof(None)

                ranked.sort(reverse=True)
                for _, sym, grade in ranked:
                    if len(positions) >= self.max_pos:
                        break
                    plan = position_plan(grade, risk=self.risk)
                    budget = plan.initial * equity_now
                    if budget <= 0 or cash < budget * 0.5:
                        continue
                    budget = min(budget, cash)
                    sub = daily[sym][daily[sym]["date"] <= day]
                    px = float(sub["close"].iloc[-1]) * (1 + self.slippage)
                    shares = budget / (px * (1 + self.commission))
                    if shares <= 0:
                        continue
                    cash -= shares * px * (1 + self.commission)
                    positions[sym] = _Position(
                        sym, grade, pd.Timestamp(day), px, shares,
                        self.cap_size_fn(sym), px)

            # ---- mark to market ----
            mtm = cash
            for s, pos in positions.items():
                h = daily[s][daily[s]["date"] <= day]
                if len(h):
                    mtm += pos.shares * float(h["close"].iloc[-1])
            eq_dates.append(day)
            eq_vals.append(mtm)

        # ---- liquidate remaining at last close ----
        last = cal[-1]
        for sym, pos in list(positions.items()):
            h = daily[sym][daily[sym]["date"] <= last]
            px = float(h["close"].iloc[-1]) * (1 - self.slippage)
            cash += pos.shares * px * (1 - self.commission)
            trades.append(TradeLeg(sym, pos.grade, pos.entry_date,
                                   pos.entry_price, last, px, pos.shares,
                                   "end_of_test"))
        eq_vals[-1] = cash if eq_vals else self.cap0

        equity = pd.Series(eq_vals, index=pd.DatetimeIndex(eq_dates),
                           name="equity")
        bench = self._benchmark_series(daily, start, end)
        metrics = _metrics(equity, trades, self.cap0, start, end, bench)
        params = {"initial_capital": self.cap0, "rebalance": self.rebalance,
                  "max_positions": self.max_pos, "min_grade": self.min_grade,
                  "universe": len(universe), "start": str(start),
                  "end": str(end)}
        return BacktestResult(equity, trades, metrics, params, bench)

    def _benchmark_series(self, daily, start, end):
        """Equal-weight buy-and-hold of the (priced) universe, rebased to the
        engine's initial capital -- a fair "do nothing but diversify"
        baseline for alpha."""
        norm = []
        for d in daily.values():
            s = d.set_index("date")["close"]
            s = s[~s.index.duplicated(keep="last")]
            if len(s) and float(s.iloc[0]) > 0:
                norm.append(s / float(s.iloc[0]))
        if not norm:
            return None
        bench = pd.concat(norm, axis=1).sort_index().ffill().mean(axis=1)
        return (bench * self.cap0).rename("benchmark")
