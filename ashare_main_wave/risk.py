"""Position sizing, stop-loss and take-profit rules (spec section 7)."""

from __future__ import annotations

from dataclasses import dataclass

from .config import DEFAULT_RISK


@dataclass
class PositionPlan:
    grade: str
    initial: float          # fraction of total capital
    max: float
    pyramid: list           # list of (label, add_fraction, condition)


def position_plan(grade: str, risk=DEFAULT_RISK) -> PositionPlan:
    """Signal -> position mapping + pyramid (spec 7.1 / 7.2)."""
    m = risk.position_map.get(grade, {"initial": 0.0, "max": 0.0})
    pyr = []
    if grade in ("A+", "A"):
        pyr = [
            ("首仓", m["initial"], "突破当周缩量回踩 20 日均线"),
            ("2仓", round(m["initial"] * 0.6, 4), "5/10 日线粘合且日线收涨≥3%"),
            ("3仓", round(m["initial"] * 0.4, 4),
             "首仓买点之上15%后强势整理(不破10日线)"),
        ]
    elif grade == "B":
        pyr = [("试探仓", m["initial"], "右侧确认(日线涨停+龙虎榜机构买入)后加5%")]
    return PositionPlan(grade, m["initial"], m["max"], pyr)


def board_type(symbol: str, is_st: bool = False) -> str:
    """Map a stock code to its board for price-limit handling (spec 7.6)."""
    if is_st:
        return "st"
    if symbol.startswith("688"):
        return "star"        # 科创板 ±20%
    if symbol.startswith("3"):
        return "chinext"     # 创业板 ±20%
    if symbol.startswith(("8", "4", "920")):
        return "bse"         # 北交所 ±30%
    return "main"            # 主板 ±10%


def price_limit_pct(symbol: str, is_st: bool = False, risk=DEFAULT_RISK) -> float:
    return risk.price_limit[board_type(symbol, is_st)]


def stop_loss_price(entry: float, *, cap_size: str = "default",
                    platform_top: float | None = None,
                    ma20: float | None = None, risk=DEFAULT_RISK) -> dict:
    """Three-way stop (spec 7.3): absolute / pattern / (time handled separately).
    Returns the candidate stops and the effective (highest) trigger."""
    pct = risk.stop_loss_pct.get(cap_size, risk.stop_loss_pct["default"])
    absolute = entry * (1 - pct)
    candidates = {"absolute": absolute}
    if platform_top is not None:
        candidates["pattern_platform"] = platform_top
    if ma20 is not None:
        candidates["pattern_ma20"] = ma20
    effective = max(candidates.values())
    return {"candidates": candidates, "effective_stop": effective,
            "absolute_pct": pct}


def time_stop_triggered(days_held: int, made_new_high: bool,
                        unrealized_pnl: float, risk=DEFAULT_RISK) -> bool:
    """Spec 7.3.3: >=10 days held, no new high and <5% cushion -> trim half."""
    return (days_held >= risk.time_stop_days and not made_new_high
            and unrealized_pnl < risk.time_stop_cushion)


def take_profit_action(unrealized_pnl: float, *, below_ma10: bool = False,
                        weekly_macd_dead: bool = False,
                        risk=DEFAULT_RISK) -> dict:
    """Segmented trailing take-profit (spec 7.4)."""
    ladder = risk.take_profit_ladder
    band = ladder[0]
    for floor, action, stop_to in ladder:
        if unrealized_pnl >= floor:
            band = (floor, action, stop_to)
    floor, action, stop_to = band

    decision = {"band_floor": floor, "stop_to_pnl": stop_to,
                "action": "持有"}
    if unrealized_pnl >= 1.00:
        decision["action"] = "启用8%移动止盈"
        if weekly_macd_dead:
            decision["action"] = "周线MACD死叉 -> 全清"
    elif unrealized_pnl >= 0.60 and below_ma10:
        decision["action"] = "跌破10日线 -> 减仓1/2"
    elif unrealized_pnl >= 0.30 and below_ma10:
        decision["action"] = "跌破10日线 -> 减仓1/3"
    elif unrealized_pnl >= 0.0:
        decision["action"] = "持有，止损上移保本"
    return decision
