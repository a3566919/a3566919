"""End-to-end scoring + grading + the 7 one-vote vetoes."""

import conftest as c
import pytest

from ashare_main_wave import grade_for, score_stock
from ashare_main_wave.types import StockScore


END = "20240430"


# --- grade_for boundaries (spec 4.3) ---------------------------------------
class TestGradeFor:
    @pytest.mark.parametrize("total,grade", [
        (100.0, "A+"), (85.0, "A+"), (84.99, "A"),
        (75.0, "A"), (74.99, "B"),
        (60.0, "B"), (59.99, "C"),
        (45.0, "C"), (44.99, "D"),
        (0.0, "D"), (-1.0, "D"),
    ])
    def test_boundaries(self, total, grade):
        assert grade_for(total) == grade


# --- textbook scenario -> A / A+ -------------------------------------------
class TestTextbook:
    def test_textbook_is_top_grade(self):
        sc = score_stock("600519", END, c.build_textbook_provider(),
                          **c.textbook_score_kwargs())
        assert isinstance(sc, StockScore)
        assert sc.vetoes == []
        assert sc.grade in ("A", "A+")
        # total is on the 0..100 scale (spec 4.1) and comfortably high
        assert 75.0 <= sc.total <= 100.0
        assert set(sc.dimensions) == {"d1", "d2", "d3", "d4", "d5"}

    def test_total_equals_sum_of_contributions(self):
        # the real formula (scoring.py): total = round(sum(contribution), 2),
        # contribution = score*weight*10  (NO extra x10 re-scale)
        sc = score_stock("600519", END, c.build_textbook_provider(),
                          **c.textbook_score_kwargs())
        expected = round(sum(d.contribution for d in sc.dimensions.values()),
                         2)
        assert sc.total == expected
        for d in sc.dimensions.values():
            assert d.contribution == pytest.approx(d.score * d.weight * 10.0)

    def test_as_dict_is_serialisable(self):
        sc = score_stock("600519", END, c.build_textbook_provider(),
                          **c.textbook_score_kwargs())
        d = sc.as_dict()
        assert d["grade"] == sc.grade
        assert d["total"] == round(sc.total, 2)
        assert set(d["dimensions"]) == {"d1", "d2", "d3", "d4", "d5"}
        assert "contribution" in d["dimensions"]["d1"]


# --- the 7 one-vote vetoes -------------------------------------------------
# Each tuple: id, provider builder, score kwargs, substring expected in vetoes
VETO_CASES = [
    ("rule1_short_consolidation", c.build_veto1_provider,
     c.textbook_score_kwargs(), "< 6 月"),
    ("rule2_low_breakout_volume", c.build_veto2_provider,
     c.textbook_score_kwargs(), "1.5x"),
    ("rule3_below_ma20", c.build_veto3_provider,
     c.textbook_score_kwargs(), "MA20"),
    ("rule3_macd_dead_cross", c.build_veto3b_provider,
     c.textbook_score_kwargs(), "MACD 死叉"),
    ("rule4_st_name", c.build_veto4_provider,
     c.textbook_score_kwargs(), "风险警示"),
    ("rule4_filed_investigation", c.build_veto4_filed_provider,
     c.textbook_score_kwargs(), "立案"),
    ("rule5_weak_sector_no_catalyst", c.build_veto5_provider,
     dict(leader_rank=1, limitups_in_sector=12, catalysts=0), "后 1/3"),
    ("rule6_capital_below_30pct", c.build_veto6_provider,
     c.textbook_score_kwargs(), "rule6"),
    ("rule7_momentum_decay", c.build_veto7_provider,
     c.textbook_score_kwargs(), "rule7"),
]


class TestVetoes:
    @pytest.mark.parametrize("name,builder,kwargs,substr", VETO_CASES,
                             ids=[t[0] for t in VETO_CASES])
    def test_each_veto_forces_zero_and_D(self, name, builder, kwargs,
                                         substr):
        sc = score_stock("600000", END, builder(), **kwargs)
        assert sc.total == 0.0, f"{name}: expected total 0, got {sc.total}"
        assert sc.grade == "D", f"{name}: expected D, got {sc.grade}"
        assert sc.vetoes, f"{name}: expected at least one veto reason"
        joined = " | ".join(sc.vetoes)
        assert substr in joined, (
            f"{name}: '{substr}' not in fired vetoes {sc.vetoes}")
        assert any("一票否决" in n for n in sc.notes)

    def test_textbook_trips_no_veto(self):
        sc = score_stock("600519", END, c.build_textbook_provider(),
                          **c.textbook_score_kwargs())
        assert sc.vetoes == []


# --- middling B / weak D scenarios -----------------------------------------
class TestGradeBands:
    def test_b_grade_scenario(self):
        sc = score_stock("600001", END, c.build_b_grade_provider(),
                          **c.b_grade_score_kwargs())
        assert sc.vetoes == []
        assert sc.grade == "B"
        assert 60.0 <= sc.total < 75.0

    def test_d_grade_scenario_is_weak_without_veto(self):
        sc = score_stock("600002", END, c.build_d_grade_provider())
        assert sc.vetoes == []          # a genuine low score, not a veto
        assert sc.grade == "D"
        assert sc.total < 45.0

    def test_grade_matches_grade_for_total(self):
        for builder, kw in ((c.build_textbook_provider,
                             c.textbook_score_kwargs()),
                            (c.build_b_grade_provider,
                             c.b_grade_score_kwargs()),
                            (c.build_d_grade_provider, {})):
            sc = score_stock("600000", END, builder(), **kw)
            if not sc.vetoes:
                assert sc.grade == grade_for(sc.total)
