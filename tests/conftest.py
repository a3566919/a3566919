"""Synthetic-data fixtures for the ashare_main_wave test-suite.

Everything here is deterministic and offline: no network, no akshare.
The :class:`FakeProvider` implements :class:`DataProvider` with crafted
monthly / weekly / daily frames plus the optional feeds, and the ``build_*``
helpers assemble whole scenarios (textbook A+, each of the 7 vetoes, a
B-grade and a D-grade case).

The frames are intentionally over-built (more rows than the minimum the
dimension code requires) and kept internally consistent so every dimension
scorer runs without raising.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ashare_main_wave.data import DataProvider

# All builders are closed-form (linspace / sin), so the suite is fully
# deterministic without any RNG seeding.


# ---------------------------------------------------------------------------
# Low-level OHLCV builders
# ---------------------------------------------------------------------------
def _ohlcv(dates, closes, *, amp=0.015, vol=1.0e6, amount=None,
           turnover=0.06):
    """Build a normalised OHLCV frame from a close path.

    ``closes`` drives open/high/low around it with a small fixed amplitude so
    body / range maths in the dimensions stay well-defined.  ``vol`` and
    ``turnover`` may be scalars or per-row sequences.
    """
    n = len(dates)
    closes = np.asarray(closes, dtype=float)
    opens = np.empty(n)
    opens[0] = closes[0]
    opens[1:] = closes[:-1]
    highs = np.maximum(opens, closes) * (1.0 + amp)
    lows = np.minimum(opens, closes) * (1.0 - amp)
    vol = np.full(n, vol, dtype=float) if np.isscalar(vol) else np.asarray(vol, float)
    if amount is None:
        amount = vol * closes
    else:
        amount = (np.full(n, amount, float) if np.isscalar(amount)
                  else np.asarray(amount, float))
    turn = (np.full(n, turnover, float) if np.isscalar(turnover)
            else np.asarray(turnover, float))
    return pd.DataFrame({
        "date": pd.to_datetime(list(dates)),
        "open": opens,
        "close": closes,
        "high": highs,
        "low": lows,
        "volume": vol,
        "amount": amount,
        "turnover": turn,
    })


def _month_dates(n, end="2024-04-30"):
    return list(pd.date_range(end=end, periods=n, freq="ME"))


def _week_dates(n, end="2024-04-26"):
    return list(pd.date_range(end=end, periods=n, freq="W-FRI"))


def _day_dates(n, end="2024-04-30"):
    return list(pd.date_range(end=end, periods=n, freq="B"))


# ---------------------------------------------------------------------------
# Scenario component builders
# ---------------------------------------------------------------------------
def textbook_monthly(consol_months=24, breakout_mult=3.0, *,
                      drawdown=0.56, breakout_excess=0.06,
                      escape_breakout=True):
    """Long deep base then a strong high-volume bullish breakout month.

    * an early run-up to an all-time high,
    * a long, tight, volume-shrinking consolidation below it,
    * a final big bullish month that clears the prior-12-month high on
      ``breakout_mult`` x the consolidation average volume.

    ``escape_breakout`` (textbook only) lifts the breakout close > base_low x
    1.95 so d1's escape-based window excludes it and the base metrics measure
    the flat platform.  Veto builders disable it so the breakout magnitude
    stays small and the targeted (volume) veto is what trips.
    """
    pre = 8
    decline = 3          # short sharp peak -> base descent
    n = pre + decline + consol_months + 1
    dates = _month_dates(n)

    peak = 100.0
    base = peak * (1.0 - drawdown)
    closes = []
    # run-up into the historic high
    closes += list(np.linspace(40, peak, pre))
    # a short decline off the high into the base
    closes += list(np.linspace(peak * 0.9, base, decline))
    # consolidation: a low, gently undulating, tight platform under the peak
    plateau = base + base * 0.03 * np.sin(np.linspace(0, 4 * np.pi, consol_months))
    closes += list(plateau)
    # breakout month: a decisive close that also escapes the base low by
    # >80% so d1's window finder excludes it and the base metrics measure the
    # *flat platform* (tight amplitude, shrunk tail volume).
    prev12_high = float(np.max(closes[-12:]))
    base_low = float(np.min(plateau))
    if breakout_excess > 0:
        bo_close = prev12_high * (1.0 + breakout_excess) + 2.0
        if escape_breakout:
            bo_close = max(bo_close, base_low * 1.95)
    else:
        bo_close = prev12_high * (1.0 + breakout_excess)
    closes.append(bo_close)
    closes = np.asarray(closes, float)

    # volume: high on run-up, shrinking through the base, exploding on breakout
    vol = np.empty(n)
    vol[:pre + decline] = 3.0e6
    consol_vol = np.linspace(2.0e6, 0.8e6, consol_months)
    vol[pre + decline:pre + decline + consol_months] = consol_vol
    consol_avg = float(consol_vol.mean())
    vol[-1] = consol_avg * breakout_mult

    df = _ohlcv(dates, closes, amp=0.01, vol=vol, amount=None)
    # make the breakout month a clean big bullish body
    df.loc[df.index[-1], "open"] = float(closes[-2]) * 1.005
    df.loc[df.index[-1], "low"] = float(closes[-2]) * 0.995
    df.loc[df.index[-1], "high"] = float(closes[-1]) * 1.002
    return df


def weak_breakout_monthly(consol_months=24):
    """Long base but the final month barely moves on shrunk volume.

    Clears nothing (excess < 0) -> Dimension-2 veto (rule 2 path)."""
    return textbook_monthly(consol_months=consol_months, breakout_mult=0.8,
                            breakout_excess=-0.05, escape_breakout=False)


def low_vol_breakout_monthly(consol_months=24):
    """Price clears the platform but on < 1.5x volume -> rule-2 veto."""
    return textbook_monthly(consol_months=consol_months, breakout_mult=1.1,
                            breakout_excess=0.06, escape_breakout=False)


def wide_amplitude_monthly():
    """A long but still-falling base: amplitude > 2.5 -> Dimension-1
    amplitude veto.

    A slowly-declining sawtooth keeps every bar's close under +80% of its
    running low (so the escape window does NOT truncate it and the duration
    veto does NOT pre-empt), yet the window's high/low spread exceeds 2.5x.
    """
    closes = list(np.linspace(40, 100, 6))      # run-up, structural peak 100
    closes += [88, 80, 84, 72, 76, 64, 68, 56, 60, 48, 52, 40, 44, 34]
    closes.append(38)
    dates = _month_dates(len(closes))
    return _ohlcv(dates, np.asarray(closes, float), amp=0.01, vol=1.5e6)


def short_consolidation_monthly():
    """A too-short base -> Dimension-1 rule-1 veto.

    d1's window runs from just after the structural peak until the first
    month whose close escapes its running base-low by +80%.  We crash off the
    peak then rebound > +80% only 3 months later, so the measured base is
    ~3 months (< the 6-month minimum) and rule-1 fires with "< 6 月".
    """
    n = 22
    dates = _month_dates(n)
    closes = list(np.linspace(50, 150, 10))     # run-up, structural peak ~150
    low = 150 * 0.45                            # crash to the base low
    closes += [low, low * 1.03, low * 0.99]     # 3 flat months at the low
    closes += [low * 1.95]                      # month 4: +95% escape -> end
    # post-escape recovery / breakout continuation
    closes += list(np.linspace(low * 1.95, 150 * 1.05, n - len(closes)))
    closes = np.asarray(closes, float)
    vol = np.full(n, 1.5e6)
    vol[-1] = 5.0e6
    return _ohlcv(dates, closes, amp=0.008, vol=vol)


def textbook_weekly(weeks=120, platform_top=45.3, breakout=True):
    """Weekly path whose final ~4 weeks (the breakout month, 2024-04)
    close above ``platform_top`` so Dimension-2's weekly-confirm scores.

    The d2 check compares weekly closes within the breakout *month* to the
    monthly prior-12-month close high (~45), so the breakout weeks must end
    clearly above that level.
    """
    dates = _week_dates(weeks)
    base = platform_top * 0.62
    closes = base + base * 0.03 * np.sin(np.linspace(0, 8 * np.pi, weeks))
    closes = np.asarray(closes, float)
    if breakout:
        # last 8 weeks ramp through and well above the platform top
        ramp = np.linspace(base, platform_top * 1.20, 8)
        closes[-8:] = ramp
    return _ohlcv(dates, closes, amp=0.01, vol=1.2e6)


def _trend_close_path(n, *, start, end, dip=False):
    base = np.linspace(start, end, n)
    wobble = (end - start) * 0.012 * np.sin(np.linspace(0, 6 * np.pi, n))
    path = base + wobble
    if dip:
        path[-4:] = path[-5] * np.array([0.965, 0.95, 0.94, 0.935])
    return path


def _staircase(n, start, end, *, steps=6, pullback=0.4):
    """A clean rising stair-step path: each leg makes a higher high then a
    shallow higher-low pullback.  Produces rising swing highs *and* rising
    swing lows so :func:`count_higher_low_groups` counts several groups.
    """
    seg = max(4, n // steps)
    path = []
    lvl = start
    inc = (end - start) / steps
    for s in range(steps):
        top = lvl + inc
        up = np.linspace(lvl, top, seg)
        if s == steps - 1:
            # final leg: pure advance so MACD stays in golden cross, price
            # stays above MA20 and the last bar is a fresh high.
            path += list(up) + list(np.linspace(top, top * 1.02, max(3, seg // 3)))
            break
        down = np.linspace(top, top - inc * pullback, max(2, seg // 3))
        path += list(up) + list(down[1:])
        lvl = top - inc * pullback
    path = np.asarray(path, float)
    if len(path) >= n:
        return path[-n:]
    pad = np.linspace(start * 0.98, start, n - len(path))
    return np.concatenate([pad, path])


def textbook_daily(n=180, platform_top=45.3):
    """Clean daily uptrend off the monthly platform (~45): MA stack, MACD
    golden cross above zero, stair-step higher lows, healthy turnover, and a
    modest gain from the platform so 3.4 (start_gain) scores well."""
    dates = _day_dates(n)
    flat = int(n * 0.40)
    closes = np.empty(n)
    # base hugging the platform, then a clean stair-step advance to ~+13%
    closes[:flat] = platform_top * 0.985 + np.linspace(-0.6, 0.4, flat)
    closes[flat:] = _staircase(n - flat, platform_top * 0.99,
                               platform_top * 1.13, steps=6, pullback=0.35)
    vol = np.linspace(8.0e5, 1.6e6, n)
    return _ohlcv(dates, closes, amp=0.012, vol=vol, turnover=0.085)


def below_ma20_daily(n=180, platform_top=45.3):
    """Daily series whose last 4 closes are below MA20 -> rule-3 veto."""
    dates = _day_dates(n)
    closes = np.empty(n)
    rise = int(n * 0.7)
    closes[:rise] = _trend_close_path(rise, start=platform_top * 0.9,
                                      end=platform_top * 1.15)
    # sharp sustained break below the (still-elevated) MA20
    closes[rise:] = np.linspace(platform_top * 1.13, platform_top * 0.80,
                                n - rise)
    return _ohlcv(dates, closes, amp=0.012, vol=np.linspace(1.6e6, 8e5, n),
                  turnover=0.07)


def macd_dead_daily(n=220, platform_top=45.3):
    """Long *accelerating* climb (so the lagging MA20 trails several %% below)
    then a sharp 3-bar dip that flips DIF below DEA (MACD dead cross) while
    price is *not* below MA20 on all of the last 3 bars.  Isolates the
    rule-3 MACD-dead-cross path (the MA20-break veto must not mask it)."""
    dates = _day_dates(n)
    closes = np.empty(n)
    rise = n - 3
    t = np.linspace(0, 1, rise)
    # convex (t**2.2) climb -> recent slope steepest -> MA20 lags more
    closes[:rise] = platform_top * 0.9 + (platform_top * 1.85 - platform_top * 0.9) * (t ** 2.2)
    closes[rise:] = closes[rise - 1] * np.array([0.97, 0.95, 0.935])
    return _ohlcv(dates, closes, amp=0.006, vol=1.0e6, turnover=0.07)


def no_new_high_decay_daily(n=180, platform_top=45.3):
    """Clean uptrend that *passes* Dimension-3 (above MA20, MACD golden
    cross, no dead cross) but whose last 3 bars make no new high and trade on
    volume decayed 50%+ -> the global rule-7 momentum-decay veto fires.

    Rule-7 compares the last 3 highs to the max high of bars[-23:-3] and the
    last-3 mean volume to the mean of bars[-8:-3].
    """
    dates = _day_dates(n)
    closes = np.empty(n)
    rise = n - 3
    # steady climb to the peak at bar -4
    closes[:rise] = np.linspace(platform_top * 0.85, platform_top * 1.28, rise)
    # last 3 closes *exactly flat* at the peak: keeps MACD from dead-crossing
    # and price well above MA20 (passes Dimension-3) -- only the *highs* and
    # *volume* signal exhaustion, which is precisely rule-7's domain.
    peak = float(closes[rise - 1])
    closes[rise:] = peak
    vol = np.full(n, 2.0e6)
    vol[-3:] = 6.0e5          # ~70% decay vs the 2.0e6 base
    df = _ohlcv(dates, closes, amp=0.006, vol=vol, turnover=0.06)
    # cap the last-3 highs just under the prior 20-bar peak high so
    # "no new high" holds (rule-7 compares to high.iloc[-23:-3].max()).
    prior_peak_high = float(df["high"].iloc[-23:-3].max())
    df.loc[df.index[-3:], "high"] = prior_peak_high * 0.99
    return df


def weak_valid_daily(n=180, platform_top=45.3):
    """A weak-but-valid trend that *passes* every Dimension-3 / rule-7 veto
    yet scores Dimension-3 low: a strong early run-up then a long flat,
    slightly-rising top.  Net effect at the last bar:

    * price hugging / just under MA20  -> no MA20-break veto, no MA stack,
    * MACD flat (no dead cross, no fresh golden cross)  -> macd ~3,
    * far above the platform -> start_gain ~0, choppy turnover.

    Used for the B / D middling scenarios so the *grade* (not a veto) is
    what places them.
    """
    dates = _day_dates(n)
    closes = np.empty(n)
    rise = int(n * 0.5)
    closes[:rise] = np.linspace(platform_top * 0.9, platform_top * 1.25, rise)
    top = (platform_top * 1.25
           + platform_top * 0.006 * np.sin(np.linspace(0, 5 * np.pi, n - rise))
           + np.linspace(0, platform_top * 0.01, n - rise))
    closes[rise:] = top
    return _ohlcv(dates, closes, amp=0.02, vol=1.0e6, turnover=0.22)


# ---- optional feeds -------------------------------------------------------
def strong_lhb():
    """Two institutional buy rows, each >5000万, cumulative >1亿."""
    return pd.DataFrame({
        "营业部名称": ["机构专用", "机构专用"],
        "买入净额": [8.0e7, 7.5e7],
    })


def strong_block():
    return pd.DataFrame({
        "折溢率": [-4.0, -2.5],          # discounts (percent form)
        "成交额": [6.0e7, 5.0e7],
    })


def strong_north():
    return pd.DataFrame({
        "持股比例": list(np.linspace(2.0, 3.4, 25)),   # +1.4pct over window
    })


def strong_survey(rows=60):
    return pd.DataFrame({
        "调研机构": [f"机构{i}" for i in range(rows)],
        "日期": ["2024-04-20"] * rows,
    })


def strong_margin():
    return pd.DataFrame({
        "融资余额": list(np.linspace(1.0e8, 1.4e8, 25)),  # +40% growth
    })


def strong_fund_flow(n=30):
    return pd.DataFrame({
        "日期": list(pd.date_range(end="2024-04-30", periods=n, freq="B")),
        "主力净流入-净额": list(np.full(n, 3.0e7)),
    })


def strong_chip(n=40):
    """`stock_cyq_em`-shaped: narrowing 90% band with rising avg cost."""
    lo = np.linspace(80, 95, n)
    hi = np.linspace(130, 112, n)
    cost = np.linspace(95, 108, n)
    return pd.DataFrame({
        "日期": list(pd.date_range(end="2024-04-30", periods=n, freq="B")),
        "90成本-低": lo,
        "90成本-高": hi,
        "平均成本": cost,
    })


def concept_rank_frame(top_names):
    """A concept fund-flow rank where ``top_names`` sit at the very top."""
    names = list(top_names) + [f"板块{i}" for i in range(30)]
    return pd.DataFrame({
        "名称": names,
        "今日主力净流入-净额": list(np.linspace(9.0e8, -3.0e8, len(names))),
    })


def spot_frame(rows):
    """A `stock_zh_a_spot_em`-shaped frame: ``rows`` = [(code, name, cap)]."""
    return pd.DataFrame({
        "代码": [r[0] for r in rows],
        "名称": [r[1] for r in rows],
        "流通市值": [r[2] for r in rows],
    })


def basic_info_normal(name="测试股份", float_cap=8.0e9):
    return {"股票简称": name, "流通市值": float_cap, "行业": "电子"}


def basic_info_st(name="ST测试", float_cap=8.0e9):
    return {"股票简称": name, "流通市值": float_cap, "行业": "电子"}


def basic_info_filed(name="测试股份", float_cap=8.0e9):
    return {"股票简称": name, "流通市值": float_cap,
            "公告": "公司收到证监会立案调查通知书"}


# ---------------------------------------------------------------------------
# The provider
# ---------------------------------------------------------------------------
class FakeProvider(DataProvider):
    """Offline DataProvider returning crafted frames.

    Construct with explicit frames; any feed left ``None`` degrades like a
    missing real feed.  ``raise_for`` makes :meth:`kline` raise for a given
    symbol so screener error-capture can be exercised.
    """

    def __init__(self, *, monthly=None, weekly=None, daily=None,
                 lhb=None, block=None, north=None, survey=None, margin=None,
                 fund_flow=None, chip=None, concepts=None, concept_rank=None,
                 basic=None, raise_for=None, per_symbol=None, spot=None):
        self._monthly = monthly
        self._weekly = weekly
        self._daily = daily
        self._lhb = lhb
        self._block = block
        self._north = north
        self._survey = survey
        self._margin = margin
        self._fund_flow = fund_flow
        self._chip = chip
        self._concepts = concepts
        self._concept_rank = concept_rank
        self._basic = basic
        self._raise_for = set(raise_for or ())
        self._spot = spot
        # per_symbol: {symbol: FakeProvider} routing for the screener tests
        self._per_symbol = per_symbol or {}

    # -- routing helper -----------------------------------------------------
    def _route(self, symbol):
        return self._per_symbol.get(symbol, self)

    def kline(self, symbol, period, start, end, adjust="qfq"):
        if symbol in self._raise_for:
            raise RuntimeError(f"synthetic kline failure for {symbol}")
        tgt = self._per_symbol.get(symbol)
        if tgt is not None:
            return tgt.kline(symbol, period, start, end, adjust)
        frame = {"monthly": self._monthly, "weekly": self._weekly,
                 "daily": self._daily}.get(period)
        if frame is None:
            return pd.DataFrame(columns=["date", "open", "close", "high",
                                         "low", "volume", "amount",
                                         "turnover"])
        return frame.copy()

    def _opt(self, symbol, attr):
        tgt = self._per_symbol.get(symbol)
        if tgt is not None:
            return getattr(tgt, attr)
        return getattr(self, attr)

    def lhb(self, symbol, start, end):
        v = self._opt(symbol, "_lhb")
        return v.copy() if v is not None else None

    def block_trade(self, symbol, start, end):
        v = self._opt(symbol, "_block")
        return v.copy() if v is not None else None

    def northbound(self, symbol, start, end):
        v = self._opt(symbol, "_north")
        return v.copy() if v is not None else None

    def institution_survey(self, symbol, start, end):
        v = self._opt(symbol, "_survey")
        return v.copy() if v is not None else None

    def margin(self, symbol, start, end):
        v = self._opt(symbol, "_margin")
        return v.copy() if v is not None else None

    def fund_flow(self, symbol):
        v = self._opt(symbol, "_fund_flow")
        return v.copy() if v is not None else None

    def chip(self, symbol):
        v = self._opt(symbol, "_chip")
        return v.copy() if v is not None else None

    def concepts_of(self, symbol):
        return self._opt(symbol, "_concepts")

    def concept_rank(self):
        v = self._concept_rank
        if v is None:
            # `concept_rank()` is symbol-less; when this provider is only a
            # per-symbol *router*, surface the first routed rank so d5's
            # sector inputs still resolve.
            for tgt in self._per_symbol.values():
                if tgt._concept_rank is not None:
                    v = tgt._concept_rank
                    break
        return v.copy() if v is not None else None

    def spot(self):
        v = self._spot
        return v.copy() if v is not None else None

    def basic_info(self, symbol):
        v = self._opt(symbol, "_basic")
        return dict(v) if v is not None else None


# ---------------------------------------------------------------------------
# Whole-scenario assembly
# ---------------------------------------------------------------------------
def build_textbook_provider(**overrides):
    """A clean A+/A scenario wired with every optional feed.

    ``overrides`` lets veto builders swap one component while keeping the
    rest textbook-clean (so each veto is isolated).
    """
    kw = dict(
        monthly=textbook_monthly(),
        weekly=textbook_weekly(),
        daily=textbook_daily(),
        lhb=strong_lhb(),
        block=strong_block(),
        north=strong_north(),
        survey=strong_survey(),
        margin=strong_margin(),
        fund_flow=strong_fund_flow(),
        chip=strong_chip(),
        concepts=["人工智能", "半导体", "算力"],
        concept_rank=concept_rank_frame(["人工智能", "半导体", "算力"]),
        basic=basic_info_normal(),
    )
    kw.update(overrides)
    return FakeProvider(**kw)


def textbook_score_kwargs():
    """Strong qualitative Dimension-5 inputs for the textbook case."""
    return dict(catalysts=2, leader_rank=1, limitups_in_sector=12)


# -- the 7 vetoes -----------------------------------------------------------
def build_veto1_provider():
    """Rule 1: monthly consolidation < 6 months."""
    return build_textbook_provider(monthly=short_consolidation_monthly())


def build_veto2_provider():
    """Rule 2: breakout-month volume < 1.5x consolidation average."""
    return build_textbook_provider(monthly=low_vol_breakout_monthly())


def build_veto3_provider():
    """Rule 3: daily close below MA20 not recovered in 3 days."""
    return build_textbook_provider(daily=below_ma20_daily())


def build_veto3b_provider():
    """Rule 3 (alt path): daily MACD dead cross."""
    return build_textbook_provider(daily=macd_dead_daily())


def build_veto4_provider():
    """Rule 4: ST / 立案 etc.  Use an ST name via basic_info."""
    return build_textbook_provider(basic=basic_info_st())


def build_veto4_filed_provider():
    """Rule 4 (alt): 立案调查 surfaced through basic_info text."""
    return build_textbook_provider(basic=basic_info_filed())


def build_veto5_provider():
    """Rule 5: sector 20d gain bottom-1/3 AND no catalyst.

    Put the stock's concepts at the very bottom of the concept rank and pass
    ``catalysts=0``.
    """
    weak_rank = concept_rank_frame([])           # our concepts not at top
    # ensure our concepts exist but rank last
    names = list(weak_rank["名称"]) + ["冷门板块A", "冷门板块B"]
    rank = pd.DataFrame({
        "名称": names,
        "今日主力净流入-净额": list(np.linspace(9e8, -9e8, len(names))),
    })
    return build_textbook_provider(concepts=["冷门板块A", "冷门板块B"],
                                   concept_rank=rank)


def build_veto6_provider():
    """Rule 6: whole Dimension-4 score < 30% of 10.

    Strip every 4A/4B feed so only the daily-derived 4B sub-indicators
    remain, and feed a daily series that scores them ~0 (no key kline,
    no inflow, weak vol/turnover) while still passing d1/d2/d3.
    """
    quiet_daily = textbook_daily()
    # flatten the final bars so key-kline / vol_turnover score ~0 but the
    # trend / MA-stack / MACD that d3 needs are still intact mid-series.
    quiet_daily = quiet_daily.copy()
    tail = quiet_daily.index[-6:]
    last_close = float(quiet_daily.loc[quiet_daily.index[-7], "close"])
    quiet_daily.loc[tail, "close"] = last_close * 1.002
    quiet_daily.loc[tail, "open"] = last_close * 1.001
    quiet_daily.loc[tail, "high"] = last_close * 1.004
    quiet_daily.loc[tail, "low"] = last_close * 0.999
    quiet_daily.loc[tail, "volume"] = 4.0e5      # quiet vs ~1e6+ baseline
    quiet_daily.loc[tail, "turnover"] = 0.005
    return build_textbook_provider(
        daily=quiet_daily,
        lhb=None, block=None, north=None, survey=None, margin=None,
        fund_flow=None, chip=None)


def build_veto7_provider():
    """Rule 7: after breakout, 3 days no new high + volume decayed 50%+."""
    return build_textbook_provider(daily=no_new_high_decay_daily())


# -- B / D grade scenarios --------------------------------------------------
def build_b_grade_provider():
    """A genuine no-veto B-grade case (~mid-60s total).

    Strong price structure + full institutional feeds (so no veto, d1/d2/d3
    score well) but *no hotspot resonance* (no concepts, only 1 catalyst) so
    Dimension-5 is weak and the total lands in the B band [60, 75).
    """
    return FakeProvider(
        monthly=textbook_monthly(),
        weekly=textbook_weekly(),
        daily=textbook_daily(),
        lhb=strong_lhb(), block=strong_block(), north=strong_north(),
        survey=strong_survey(), margin=strong_margin(),
        fund_flow=strong_fund_flow(), chip=strong_chip(),
        concepts=None, concept_rank=None,        # no sector resonance
        basic=basic_info_normal(),
    )


def b_grade_score_kwargs():
    return dict(catalysts=1)


def build_d_grade_provider():
    """A no-veto D-grade weak case (total < 45).

    A weak-but-*valid* breakout (small excess, just-enough volume, short
    base), a flat/exhausted daily that passes every Dimension-3 / rule-7 veto
    but scores low, and strong public 4A feeds *only* to clear the rule-6
    主力 floor while 4B / hotspot stay thin -> a low total, no veto.
    """
    return FakeProvider(
        monthly=textbook_monthly(consol_months=8, breakout_mult=1.7,
                                 breakout_excess=0.006,
                                 escape_breakout=False),
        weekly=textbook_weekly(breakout=False),
        daily=weak_valid_daily(),
        lhb=strong_lhb(), block=strong_block(), north=strong_north(),
        survey=strong_survey(), margin=strong_margin(),
        fund_flow=None, chip=None,               # thin 4B
        concepts=None, concept_rank=None,        # no hotspot
        basic=basic_info_normal(),
    )


# ---------------------------------------------------------------------------
# pytest fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def textbook_provider():
    return build_textbook_provider()


@pytest.fixture
def end_date():
    return "20240430"
