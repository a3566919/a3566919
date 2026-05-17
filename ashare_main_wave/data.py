"""Data access layer.

The scoring engine never calls AkShare directly. It talks to a
:class:`DataProvider`, so the methodology is fully unit-testable offline with
synthetic frames and AkShare stays an *optional* runtime dependency (it does
not even import unless :class:`AkShareProvider` is used).

OHLCV frames are normalised to English columns:
    date, open, close, high, low, volume, amount, turnover
``turnover`` is the daily turnover rate as a fraction (换手率/100).
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd

_OHLCV_MAP = {
    "日期": "date", "开盘": "open", "收盘": "close", "最高": "high",
    "最低": "low", "成交量": "volume", "成交额": "amount",
    "换手率": "turnover", "振幅": "amplitude", "涨跌幅": "pct_change",
}


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Rename AkShare Chinese columns to the canonical schema and coerce
    dtypes. Idempotent: already-normalised frames pass through."""
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["date", "open", "close", "high", "low",
                                     "volume", "amount", "turnover"])
    out = df.rename(columns=_OHLCV_MAP).copy()
    if "date" in out:
        out["date"] = pd.to_datetime(out["date"])
        out = out.sort_values("date").reset_index(drop=True)
    if "turnover" in out and out["turnover"].abs().median() > 1.5:
        out["turnover"] = out["turnover"] / 100.0  # AkShare reports percent
    for col in ("open", "close", "high", "low", "volume", "amount"):
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


class DataProvider(ABC):
    """Interface the scoring engine depends on. Implement the pieces you have
    a data feed for; missing feeds should return ``None`` so dimension scoring
    degrades gracefully (see :func:`utils.weighted`)."""

    @abstractmethod
    def kline(self, symbol: str, period: str, start: str, end: str,
              adjust: str = "qfq") -> pd.DataFrame: ...

    # Optional feeds -- default to "no data".
    def lhb(self, symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
        return None

    def block_trade(self, symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
        return None

    def northbound(self, symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
        return None

    def institution_survey(self, symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
        return None

    def margin(self, symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
        return None

    def fund_flow(self, symbol: str) -> Optional[pd.DataFrame]:
        return None

    def chip(self, symbol: str) -> Optional[pd.DataFrame]:
        return None

    def concepts_of(self, symbol: str) -> Optional[list]:
        return None

    def concept_hist(self, name: str, start: str, end: str) -> Optional[pd.DataFrame]:
        return None

    def concept_rank(self) -> Optional[pd.DataFrame]:
        return None

    def limitup_pool(self, date: str) -> Optional[pd.DataFrame]:
        return None

    def spot(self) -> Optional[pd.DataFrame]:
        return None

    def basic_info(self, symbol: str) -> Optional[dict]:
        return None


def _retry(fn, attempts: int = 4, base_delay: float = 2.0):
    """Exponential backoff (2s, 4s, 8s, 16s) for flaky network calls."""
    last = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - network libs raise broadly
            last = exc
            if i == attempts - 1:
                break
            time.sleep(base_delay * (2 ** i))
    raise last


class AkShareProvider(DataProvider):
    """Live provider backed by AkShare (function names per spec section 6,
    AkShare 1.16-1.18). AkShare is imported lazily so the package works
    without it installed."""

    def __init__(self):
        try:
            import akshare as ak  # noqa: F401
        except Exception as exc:  # pragma: no cover - env dependent
            raise RuntimeError(
                "akshare not installed; `pip install -U akshare` or use a "
                "custom DataProvider for offline/back-test use."
            ) from exc
        self._ak = ak

    def kline(self, symbol, period, start, end, adjust="qfq"):
        df = _retry(lambda: self._ak.stock_zh_a_hist(
            symbol=symbol, period=period, start_date=start,
            end_date=end, adjust=adjust))
        return normalize_ohlcv(df)

    def lhb(self, symbol, start, end):
        return _retry(lambda: self._ak.stock_lhb_stock_detail_em(
            symbol=symbol, date=end, flag="买入"))

    def block_trade(self, symbol, start, end):
        return _retry(lambda: self._ak.stock_dzjy_mrmx(
            symbol="A股", start_date=start, end_date=end))

    def northbound(self, symbol, start, end):
        return _retry(lambda: self._ak.stock_hsgt_stock_statistics_em(
            symbol="北向持股", start_date=start, end_date=end))

    def institution_survey(self, symbol, start, end):
        return _retry(lambda: self._ak.stock_jgdy_detail_em(date=end))

    def margin(self, symbol, start, end):
        return _retry(lambda: self._ak.stock_margin_detail_szse(date=end))

    def fund_flow(self, symbol):
        market = "sz" if symbol.startswith(("0", "3")) else "sh"
        return _retry(lambda: self._ak.stock_individual_fund_flow(
            stock=symbol, market=market))

    def chip(self, symbol):
        return _retry(lambda: self._ak.stock_cyq_em(symbol=symbol, adjust="qfq"))

    def concept_rank(self):
        return _retry(lambda: self._ak.stock_board_concept_fund_flow_rank_em())

    def limitup_pool(self, date):
        return _retry(lambda: self._ak.stock_zt_pool_em(date=date))

    def spot(self):
        return _retry(lambda: self._ak.stock_zh_a_spot_em())

    def basic_info(self, symbol):
        df = _retry(lambda: self._ak.stock_individual_info_em(symbol=symbol))
        try:
            return dict(zip(df["item"], df["value"]))
        except Exception:  # noqa: BLE001
            return None
