"""Centralised thresholds and weights for the 5-dimension 主升浪 model.

Every number here maps to a row in the methodology spec (sections 3, 4, 7).
Thresholds are deliberately gathered in one place because the spec calls for
re-calibrating them every quarter against fresh samples (section 11.1).
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Dimension weights (spec 4.1) -- must sum to 1.0
# ---------------------------------------------------------------------------
DIMENSION_WEIGHTS = {
    "d1": 0.25,  # 月线盘整
    "d2": 0.20,  # 月线突破
    "d3": 0.15,  # 日线趋势
    "d4": 0.25,  # 主力行为 (4A 0.12 + 4B 0.13)
    "d5": 0.15,  # 热点共振
}
D4A_WEIGHT = 0.12
D4B_WEIGHT = 0.13

# Sub-indicator weights inside each dimension (spec section 3 tables).
D1_SUB_WEIGHTS = {"drawdown": 0.20, "duration": 0.30, "amplitude": 0.20,
                  "volume_shrink": 0.20, "chip": 0.10}
D2_SUB_WEIGHTS = {"excess": 0.25, "body": 0.25, "volume": 0.30, "weekly": 0.20}
D3_SUB_WEIGHTS = {"higher_lows": 0.20, "ma_stack": 0.25, "macd": 0.25,
                  "start_gain": 0.15, "turnover": 0.15}
D4A_SUB_WEIGHTS = {"lhb": 0.25, "block": 0.20, "north": 0.20,
                   "survey": 0.15, "margin": 0.20}
D4B_SUB_WEIGHTS = {"net_inflow": 0.25, "key_kline": 0.20, "gap": 0.15,
                   "chip": 0.20, "vol_turnover": 0.20}
D5_SUB_WEIGHTS = {"strength": 0.25, "tags": 0.20, "limitups": 0.15,
                  "catalyst": 0.20, "leader": 0.20}


# ---------------------------------------------------------------------------
# Grade boundaries (spec 4.3)
# ---------------------------------------------------------------------------
GRADE_BOUNDS = [("A+", 85.0), ("A", 75.0), ("B", 60.0), ("C", 45.0)]
GRADE_DEFAULT = "D"


@dataclass(frozen=True)
class Thresholds:
    """Tunable thresholds. Edit instances rather than the constants when
    back-testing alternative parameter sets (spec 9.5 / 11.1)."""

    # --- Dimension 1: 月线盘整 (spec 3 维度1) ---
    d1_drawdown_base: float = 0.50          # 1.1 回撤 >= 50%
    d1_drawdown_full: float = 0.60          # 1.1 回撤 >= 60% 满分
    d1_duration_min: int = 6                # 1.2 / veto: 盘整 >= 6 月
    d1_duration_lo: int = 12                # 1.2 标准区间下限
    d1_duration_hi: int = 36                # 1.2 标准区间上限
    d1_amp_full: float = 1.5                # 1.3 振幅 <= 1.5 满分
    d1_amp_base: float = 1.8                # 1.3 振幅 <= 1.8 基础
    d1_amp_veto: float = 2.5                # veto: 振幅 > 2.5
    d1_vol_shrink_base: float = 0.70        # 1.4 末期量 <= 70% 全期
    d1_vol_shrink_full: float = 0.60        # 1.4 <= 60% 满分
    d1_chip_width_base: float = 0.25        # 1.5 90% 筹码宽度 <= 25%
    d1_chip_width_full: float = 0.20        # 1.5 <= 20% 满分

    # --- Dimension 2: 月线突破 (spec 3 维度2) ---
    d2_excess_full: float = 0.03            # 2.1 突破幅度 >= 3% 满分
    d2_body_base: float = 0.60              # 2.2 实体占比 >= 60%
    d2_body_full: float = 0.70              # 2.2 >= 70% 满分
    d2_vol_mult_min: float = 1.5            # 2.3 / veto: 突破量 >= 1.5x
    d2_vol_mult_full: float = 2.0           # 2.3 >= 2x 满分
    d2_weekly_base: int = 2                 # 2.5 >= 2 根周线收上平台
    d2_weekly_full: int = 3                 # 2.5 >= 3 根周线 满分

    # --- Dimension 3: 日线趋势 (spec 3 维度3) ---
    d3_higher_lows_base: int = 3            # 3.1 >= 3 组抬高
    d3_higher_lows_full: int = 4            # 3.1 >= 4 组 满分
    d3_start_gain_full: float = 0.15        # 3.4 距突破点涨幅 <= 15% 满分
    d3_start_gain_base: float = 0.25        # 3.4 <= 25% 基础
    d3_turnover_lo: float = 0.05            # 3.5 换手 5%
    d3_turnover_hi: float = 0.15            # 3.5 换手 15%
    d3_turnover_ideal_lo: float = 0.07      # 3.5 理想 7%
    d3_turnover_ideal_hi: float = 0.12      # 3.5 理想 12%
    d3_ma20_recover_days: int = 3           # 3 veto: 跌破 MA20 后 3 日收复

    # --- Dimension 4A: 公开数据 (spec 3 维度4A) ---
    d4a_lhb_min_count: int = 1              # 4A.1 机构净买入 >= 1 次
    d4a_lhb_full_count: int = 2             # 4A.1 >= 2 次 满分
    d4a_lhb_min_amount: float = 5e7         # 4A.1 单次 >= 5000 万
    d4a_lhb_full_amount: float = 1e8        # 4A.1 累计过亿 满分
    d4a_block_discount_base: float = 0.05   # 4A.2 折价率 <= 5%
    d4a_block_discount_full: float = 0.03   # 4A.2 <= 3% 满分
    d4a_block_ratio_base: float = 0.01      # 4A.2 占流通 >= 1%
    d4a_block_ratio_full: float = 0.02      # 4A.2 >= 2% 满分
    d4a_north_base: float = 0.005           # 4A.3 持股比例 +0.5pct
    d4a_north_full: float = 0.01            # 4A.3 +1pct 满分
    d4a_survey_base: int = 20               # 4A.4 调研 >= 20 家
    d4a_survey_full: int = 50               # 4A.4 >= 50 家 满分
    d4a_margin_base: float = 0.15           # 4A.5 融资余额 +15%
    d4a_margin_full: float = 0.30           # 4A.5 +30% 满分
    d4a_margin_cap: float = 0.08            # 4A.5 占流通市值 <= 8%

    # --- Dimension 4B: 量价行为 (spec 3 维度4B) ---
    d4b_inflow_base: float = 0.01           # 4B.1 10 日净流入 >= 流通 1%
    d4b_inflow_full: float = 0.03           # 4B.1 >= 流通 3% 满分
    d4b_chip_narrow_base: float = 0.05      # 4B.4 筹码宽度环比收窄 >= 5%
    d4b_chip_narrow_full: float = 0.10      # 4B.4 >= 10% 满分
    d4b_vol_ratio_lo: float = 1.5           # 4B.5 量比 1.5
    d4b_vol_ratio_hi: float = 3.0           # 4B.5 量比 3
    d4b_turnover_lo: float = 0.05           # 4B.5 换手 5%
    d4b_turnover_hi: float = 0.15           # 4B.5 换手 15%

    # --- Dimension 5: 热点共振 (spec 3 维度5) ---
    d5_strength_full_pct: float = 0.05      # 5.1 板块涨幅前 5% 满分
    d5_strength_base_pct: float = 0.20      # 5.1 前 20% 基础
    d5_tags_base: int = 2                   # 5.2 >= 2 个强势板块
    d5_tags_full: int = 3                   # 5.2 >= 3 个 满分
    d5_limitups_base: int = 5               # 5.3 >= 5 只涨停
    d5_limitups_full: int = 10              # 5.3 >= 10 只 满分
    d5_catalyst_base: int = 1               # 5.4 >= 1 个催化剂
    d5_catalyst_full: int = 2               # 5.4 >= 2 个 满分
    d5_leader_rank_base: int = 3            # 5.5 板块内排名前 3
    d5_sector_veto_quantile: float = 1 / 3  # veto: 板块涨幅后 1/3

    # --- One-vote vetoes (spec 4.4) ---
    veto_d4_min_ratio: float = 0.30         # rule 6: 主力行为 < 30%
    veto_no_new_high_days: int = 3          # rule 7: 连续 3 日未创新高
    veto_vol_decay: float = 0.50            # rule 7: 量能萎缩 50%+


DEFAULT_THRESHOLDS = Thresholds()


@dataclass(frozen=True)
class RiskConfig:
    """Position sizing / stop / take-profit parameters (spec section 7)."""

    # Signal -> position mapping (spec 7.1), values are fraction of total capital
    position_map: dict = field(default_factory=lambda: {
        "A+": {"initial": 0.30, "max": 0.50},
        "A": {"initial": 0.20, "max": 0.30},
        "B": {"initial": 0.10, "max": 0.20},
        "C": {"initial": 0.0, "max": 0.0},
        "D": {"initial": 0.0, "max": 0.0},
    })

    # Absolute stop loss by board type (spec 7.3.1)
    stop_loss_pct: dict = field(default_factory=lambda: {
        "main": 0.06,        # 大盘股 6%
        "small": 0.10,       # 中小盘 10%
        "default": 0.08,     # 默认 8%
    })

    time_stop_days: int = 10          # 7.3.3 进场后 10 日未创新高
    time_stop_cushion: float = 0.05   # 7.3.3 未拉开 5% 安全垫

    # Take-profit ladder (spec 7.4): list of (pnl_floor, action, stop_to)
    take_profit_ladder: tuple = (
        (0.00, "hold", 0.00),
        (0.30, "trim_1_3", 0.15),
        (0.60, "trim_1_2", 0.30),
        (1.00, "trailing_8pct", None),
    )

    # Board price-limit adaptation (spec 7.6)
    price_limit: dict = field(default_factory=lambda: {
        "main": 0.10,
        "star": 0.20,    # 科创板
        "chinext": 0.20, # 创业板
        "bse": 0.30,     # 北交所
        "st": 0.05,
    })


DEFAULT_RISK = RiskConfig()
