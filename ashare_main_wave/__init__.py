"""A 股主升浪起涨点科学识别系统.

Quantitative implementation of the 5-dimension (五维共振) main-rising-wave
model. The scoring engine is decoupled from data access via
:class:`~ashare_main_wave.data.DataProvider`, so it runs fully offline for
back-testing and unit tests; :class:`~ashare_main_wave.data.AkShareProvider`
plugs in live AkShare data.
"""

from .backtest import BacktestResult, Backtester, TradeLeg
from .config import DEFAULT_RISK, DEFAULT_THRESHOLDS, RiskConfig, Thresholds
from .data import (AkShareProvider, DataProvider, HistoricalFrameProvider,
                   normalize_ohlcv)
from .risk import (PositionPlan, position_plan, price_limit_pct,
                   stop_loss_price, take_profit_action, time_stop_triggered)
from .scoring import grade_for, score_stock
from .screener import ScreenResult, screen
from .types import DimensionResult, StockScore

__version__ = "0.1.0"

__all__ = [
    "score_stock", "grade_for", "screen", "ScreenResult",
    "DataProvider", "AkShareProvider", "HistoricalFrameProvider",
    "normalize_ohlcv", "StockScore", "DimensionResult",
    "Thresholds", "RiskConfig", "DEFAULT_THRESHOLDS", "DEFAULT_RISK",
    "position_plan", "PositionPlan", "stop_loss_price",
    "take_profit_action", "time_stop_triggered", "price_limit_pct",
    "Backtester", "BacktestResult", "TradeLeg",
    "__version__",
]
