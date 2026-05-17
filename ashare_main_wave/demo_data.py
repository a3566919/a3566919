"""Self-contained synthetic data provider for offline demo / UI.

Lets the dashboard, ``examples`` and quick experiments run the full scoring
pipeline with **no network and no akshare** by serving deterministic,
hand-tuned OHLCV + capital feeds for three canonical scenarios:

    "textbook"  -> a clean A / A+ 主升浪 setup
    "b_grade"   -> strong structure but no hotspot resonance  -> B band
    "d_grade"   -> weak-but-valid breakout, thin feeds         -> D (no veto)

Everything here is closed-form (no RNG) so results are reproducible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import DataProvider

SCENARIOS = ("textbook", "b_grade", "d_grade")


def _dates(n, end, freq):
    return pd.date_range(end=pd.Timestamp(end), periods=n, freq=freq)


def _sync_amount(df):
    """Keep 成交额 consistent with 成交量 after any volume overrides so the
    量能收敛 / volume-multiple metrics measure real shrink, not a stale
    flat amount column."""
    df["amount"] = df["volume"] * df["close"]
    return df


def _ohlcv(dates, closes, *, amp=0.015, vol=1.0e6):
    closes = np.asarray(closes, dtype=float)
    highs = closes * (1 + amp)
    lows = closes * (1 - amp)
    opens = np.concatenate([[closes[0]], closes[:-1]])
    vols = np.full(len(closes), vol, dtype=float)
    return pd.DataFrame({
        "date": dates, "open": opens, "close": closes,
        "high": np.maximum(highs, np.maximum(opens, closes)),
        "low": np.minimum(lows, np.minimum(opens, closes)),
        "volume": vols, "amount": vols * closes,
        "turnover": np.full(len(closes), 0.08),
    })


# --- monthly builders ------------------------------------------------------

def _monthly_textbook():
    pre = np.linspace(60, 100, 6)                 # run-up to structural peak
    base = 50 + 5 * np.sin(np.linspace(0, 6.28, 24))   # 24m flat base ~45..55
    breakout = np.array([82.0])                   # decisive monthly breakout
    closes = np.concatenate([pre, base, breakout])
    df = _ohlcv(_dates(len(closes), "2024-04-30", "ME"), closes, amp=0.025,
                vol=8.0e5)
    df.loc[df.index[:6], "volume"] = 1.2e6
    df.loc[df.index[6:30], "volume"] = np.linspace(9e5, 3e5, 24)  # base shrink
    df.loc[df.index[-1], "volume"] = 3.0e6                        # breakout
    return _sync_amount(df)


def _monthly_b_grade():
    return _monthly_textbook()                    # same strong structure


def _monthly_d_grade():
    # shallow peak so a *small valid* breakout clears prev-12m high (no d2
    # veto); short 8m base + tiny drawdown -> genuinely weak d1, low total.
    pre = np.linspace(44, 52, 6)
    base = np.full(6, 48.0) + 0.8 * np.sin(np.linspace(0, 3, 6))  # short base
    breakout = np.array([53.0])                    # excess ~ +2% only
    closes = np.concatenate([pre, base, breakout])
    df = _ohlcv(_dates(len(closes), "2024-04-30", "ME"), closes, amp=0.03,
                vol=6e5)
    df.loc[df.index[-1], "volume"] = 1.1e6                          # ~1.8x
    return _sync_amount(df)


# --- weekly builders -------------------------------------------------------

def _weekly(platform_top=56.0, breakout=True):
    n = 120
    closes = np.concatenate([
        np.full(n - 4, platform_top - 4.0),
        (np.linspace(platform_top + 2, 82, 4) if breakout
         else np.full(4, platform_top - 3.0)),
    ])
    return _ohlcv(_dates(n, "2024-04-26", "W-FRI"), closes, amp=0.02,
                  vol=9e5)


# --- daily builders --------------------------------------------------------

def _staircase(n, start, end, steps=6, pullback=0.4):
    seg = n // steps
    out, cur = [], start
    inc = (end - start) / steps
    for s in range(steps):
        top = cur + inc
        up = np.linspace(cur, top, seg)
        dn = np.linspace(top, top - inc * pullback, max(2, seg // 3))
        out.extend(up.tolist())
        out.extend(dn.tolist())
        cur = top - inc * pullback
    arr = np.array(out[:n] if len(out) >= n else
                   out + [out[-1]] * (n - len(out)))
    return arr


def _daily_textbook(n=180):
    # fine, frequent stair-steps (5-day cycles: 3 up / 2 shallow pull-back)
    # so the trailing-20d window holds >=4 rising swing groups and a fresh
    # MACD golden cross above zero, ending on a new high (MA stack intact).
    cur, out = 46.0, []
    while len(out) < n:
        out += list(np.linspace(cur, cur + 2.4, 3))
        out += list(np.linspace(cur + 2.4, cur + 1.6, 2))
        cur += 1.6
    closes = np.array(out[:n])
    closes[-1] = closes.max() + 0.5                  # decisive new high
    df = _ohlcv(_dates(n, "2024-04-30", "B"), closes, amp=0.012, vol=1.0e6)
    df.loc[df.index[-40:], "volume"] = np.linspace(1.4e6, 2.6e6, 40)
    df["turnover"] = 0.09
    return _sync_amount(df)


def _daily_weak_valid(n=180):
    # gently up then flat: stays above MA20, no MACD dead cross (passes d3
    # veto) but scores low; volume holds steady so rule-7 (decay) is not hit.
    closes = np.concatenate([
        np.linspace(46, 52, n - 40),
        np.full(40, 52.0) + 0.2 * np.sin(np.linspace(0, 6, 40)),
    ])
    df = _ohlcv(_dates(n, "2024-04-30", "B"), closes, amp=0.01, vol=6e5)
    df["turnover"] = 0.02
    return _sync_amount(df)


# --- capital feeds ---------------------------------------------------------

def _lhb_strong():
    return pd.DataFrame({
        "营业部名称": ["机构专用", "机构专用", "某游资"],
        "净额": [8.0e7, 6.0e7, -1.0e7],
    })


def _block_strong():
    return pd.DataFrame({"成交额": [3.0e7, 4.0e7], "折溢率": [-2.0, -3.0]})


def _north_strong():
    return pd.DataFrame({"持股比例": list(np.linspace(1.0, 2.2, 25))})


def _survey_strong(rows=60):
    return pd.DataFrame({"机构名称": [f"基金{i}" for i in range(rows)]})


def _margin_strong():
    return pd.DataFrame({"融资余额": list(np.linspace(1.0e8, 1.5e8, 25))})


def _fund_flow_strong(n=30):
    return pd.DataFrame({"主力净流入-净额": list(np.linspace(2e7, 9e7, n))})


def _chip_strong(n=40):
    # band tightens to ~16% width and average cost rises >5% -> full marks
    lo = np.linspace(44, 64, n)
    hi = np.linspace(78, 76, n)
    cost = np.linspace(52, 70, n)
    return pd.DataFrame({"90成本-低": lo, "90成本-高": hi,
                         "平均成本": cost})


def _concept_rank(top):
    names = list(top) + [f"冷门{i}" for i in range(40)]
    return pd.DataFrame({
        "名称": names,
        "今日主力净流入-净额": list(np.linspace(9e8, -9e8, len(names))),
    })


def _basic(name="演示股份", cap=8.0e9):
    return {"股票简称": name, "流通市值": cap}


class DemoProvider(DataProvider):
    """Offline provider. ``scenario`` in :data:`SCENARIOS`."""

    def __init__(self, scenario: str = "textbook"):
        if scenario not in SCENARIOS:
            raise ValueError(f"scenario must be one of {SCENARIOS}")
        self.scenario = scenario

    # default Dimension-5 qualitative kwargs for score_stock
    def score_kwargs(self) -> dict:
        return {
            "textbook": dict(catalysts=2, leader_rank=1, limitups_in_sector=12),
            "b_grade": dict(catalysts=1),
            "d_grade": dict(),
        }[self.scenario]

    def kline(self, symbol, period, start, end, adjust="qfq"):
        s = self.scenario
        if period == "monthly":
            return {"textbook": _monthly_textbook, "b_grade": _monthly_b_grade,
                    "d_grade": _monthly_d_grade}[s]()
        if period == "weekly":
            return _weekly(breakout=(s != "d_grade"))
        return _daily_weak_valid() if s == "d_grade" else _daily_textbook()

    def _on(self):  # 4A public feeds present in all demos (clears rule-6
        return True  # floor); d_grade still stays weak via thin 4B/d1/d5

    def lhb(self, *a):
        return _lhb_strong() if self._on() else None

    def block_trade(self, *a):
        return _block_strong() if self._on() else None

    def northbound(self, *a):
        return _north_strong() if self._on() else None

    def institution_survey(self, *a):
        return _survey_strong() if self._on() else None

    def margin(self, *a):
        return _margin_strong() if self._on() else None

    def fund_flow(self, *a):
        return _fund_flow_strong() if self.scenario in ("textbook",
                                                        "b_grade") else None

    def chip(self, *a):
        return _chip_strong() if self.scenario == "textbook" else None

    def concepts_of(self, symbol):
        return ["人工智能", "半导体", "算力"] if self.scenario == "textbook" else None

    def concept_rank(self):
        return (_concept_rank(["人工智能", "半导体", "算力"])
                if self.scenario == "textbook" else None)

    def basic_info(self, symbol):
        return _basic()
