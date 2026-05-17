"""Position sizing / stop-loss / take-profit rules (spec section 7)."""

import pytest

from ashare_main_wave import (position_plan, price_limit_pct, stop_loss_price,
                              take_profit_action, time_stop_triggered)
from ashare_main_wave.risk import PositionPlan, board_type


# --- position_plan (spec 7.1 / 7.2) ----------------------------------------
class TestPositionPlan:
    def test_a_plus(self):
        p = position_plan("A+")
        assert isinstance(p, PositionPlan)
        assert p.initial == 0.30 and p.max == 0.50
        assert len(p.pyramid) == 3                # full pyramid for A+/A
        labels = [row[0] for row in p.pyramid]
        assert labels[0] == "首仓"

    def test_a(self):
        p = position_plan("A")
        assert p.initial == 0.20 and p.max == 0.30
        assert len(p.pyramid) == 3

    def test_b_has_single_probe(self):
        p = position_plan("B")
        assert p.initial == 0.10 and p.max == 0.20
        assert len(p.pyramid) == 1

    @pytest.mark.parametrize("grade", ["C", "D"])
    def test_c_and_d_are_no_position(self, grade):
        p = position_plan(grade)
        assert p.initial == 0.0 and p.max == 0.0
        assert p.pyramid == []

    def test_unknown_grade_is_flat(self):
        p = position_plan("???")
        assert p.initial == 0.0 and p.max == 0.0


# --- board_type / price_limit_pct (spec 7.6) -------------------------------
class TestBoard:
    @pytest.mark.parametrize("symbol,board,limit", [
        ("688256", "star", 0.20),       # 科创板
        ("300308", "chinext", 0.20),    # 创业板
        ("600519", "main", 0.10),       # 主板 沪
        ("000001", "main", 0.10),       # 主板 深
        ("830799", "bse", 0.30),        # 北交所 8-prefixed
        ("430139", "bse", 0.30),        # 北交所 4-prefixed
        ("920001", "bse", 0.30),        # 北交所 920-prefixed
    ])
    def test_board_and_limit(self, symbol, board, limit):
        assert board_type(symbol) == board
        assert price_limit_pct(symbol) == limit

    def test_st_overrides_board(self):
        assert board_type("600519", is_st=True) == "st"
        assert price_limit_pct("600519", is_st=True) == 0.05
        # ST flag wins even on a 20%-limit board
        assert board_type("300308", is_st=True) == "st"
        assert price_limit_pct("300308", is_st=True) == 0.05


# --- stop_loss_price (spec 7.3) --------------------------------------------
class TestStopLoss:
    def test_picks_highest_tightest_trigger(self):
        # absolute = 100*(1-0.06) = 94 ; platform 96 ; ma20 97 -> max = 97
        out = stop_loss_price(100.0, cap_size="main",
                              platform_top=96.0, ma20=97.0)
        assert out["absolute_pct"] == 0.06
        assert out["candidates"]["absolute"] == pytest.approx(94.0)
        assert out["effective_stop"] == pytest.approx(97.0)

    def test_absolute_only_when_no_pattern(self):
        out = stop_loss_price(50.0)         # default 8%
        assert out["candidates"] == {"absolute": pytest.approx(46.0)}
        assert out["effective_stop"] == pytest.approx(46.0)

    def test_cap_size_changes_absolute_pct(self):
        small = stop_loss_price(100.0, cap_size="small")
        main = stop_loss_price(100.0, cap_size="main")
        assert small["absolute_pct"] == 0.10
        assert main["absolute_pct"] == 0.06
        assert small["candidates"]["absolute"] < main["candidates"]["absolute"]

    def test_unknown_cap_size_falls_back_to_default(self):
        out = stop_loss_price(100.0, cap_size="nonsense")
        assert out["absolute_pct"] == 0.08


# --- time_stop_triggered (spec 7.3.3) --------------------------------------
class TestTimeStop:
    @pytest.mark.parametrize("days,new_high,pnl,expected", [
        (10, False, 0.04, True),    # 10d, no new high, <5% cushion -> trim
        (15, False, 0.00, True),
        (9, False, 0.00, False),    # not yet 10 days
        (10, True, 0.00, False),    # made a new high -> ok
        (10, False, 0.05, False),   # cushion reached (>=5%) -> ok
        (10, False, 0.06, False),
    ])
    def test_truth_table(self, days, new_high, pnl, expected):
        assert time_stop_triggered(days, new_high, pnl) is expected


# --- take_profit_action (spec 7.4) -----------------------------------------
class TestTakeProfit:
    def test_below_30pct_band_holds(self):
        d = take_profit_action(0.10)
        assert d["band_floor"] == 0.00
        assert d["action"] == "持有，止损上移保本"

    def test_30pct_band_below_ma10_trims_third(self):
        d = take_profit_action(0.35, below_ma10=True)
        assert d["band_floor"] == 0.30
        assert "减仓1/3" in d["action"]

    def test_30pct_band_holding_ma10_holds_and_lifts_stop(self):
        # in profit but holding MA10: hold and trail the stop to break-even
        d = take_profit_action(0.35, below_ma10=False)
        assert d["band_floor"] == 0.30
        assert d["action"] == "持有，止损上移保本"

    def test_negative_pnl_just_holds(self):
        d = take_profit_action(-0.05)
        assert d["band_floor"] == 0.00
        assert d["action"] == "持有"

    def test_60pct_band_below_ma10_trims_half(self):
        d = take_profit_action(0.65, below_ma10=True)
        assert d["band_floor"] == 0.60
        assert "减仓1/2" in d["action"]

    def test_100pct_band_enables_trailing(self):
        d = take_profit_action(1.20)
        assert d["band_floor"] == 1.00
        assert d["action"] == "启用8%移动止盈"

    def test_100pct_band_weekly_macd_dead_full_exit(self):
        d = take_profit_action(1.20, weekly_macd_dead=True)
        assert d["band_floor"] == 1.00
        assert "全清" in d["action"]

    def test_band_floor_monotonic_with_pnl(self):
        floors = [take_profit_action(p)["band_floor"]
                  for p in (-0.1, 0.0, 0.3, 0.6, 1.0, 2.0)]
        assert floors == sorted(floors)
