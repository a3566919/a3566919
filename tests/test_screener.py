"""Weekend screening pipeline (ashare_main_wave.screener)."""

import conftest as c

from ashare_main_wave import screen
from ashare_main_wave.screener import ScreenResult


END = "20240430"


def _router(extra_raise=None):
    """A root FakeProvider that routes each symbol to its own scenario."""
    per = {
        "AAA": c.build_textbook_provider(),     # -> A / A+  (sniper)
        "BBB": c.build_b_grade_provider(),      # -> B       (watch)
        "DDD": c.build_d_grade_provider(),      # -> D       (rejected)
        "VET": c.build_veto3_provider(),        # -> veto    (rejected)
    }
    return c.FakeProvider(per_symbol=per, raise_for=extra_raise or set())


class TestExplicitUniverse:
    def test_buckets_by_grade(self):
        root = _router()
        res = screen(root, END, universe=["AAA", "BBB", "DDD", "VET"],
                     catalysts_by_symbol={"AAA": 2, "BBB": 1})
        assert isinstance(res, ScreenResult)
        assert [s.symbol for s in res.sniper_pool] == ["AAA"]
        assert [s.symbol for s in res.watch_pool] == ["BBB"]
        assert res.rejected == 2          # DDD + VET
        assert res.errors == []
        # sniper grades are top-tier
        assert all(s.grade in ("A", "A+") for s in res.sniper_pool)
        assert all(s.grade == "B" for s in res.watch_pool)

    def test_raising_feed_is_captured_not_fatal(self):
        # one symbol's kline blows up; the run must still finish and bucket
        # the others, recording the failure in `errors`.
        root = _router(extra_raise={"BOOM"})
        res = screen(root, END,
                     universe=["AAA", "BOOM", "BBB", "DDD"],
                     catalysts_by_symbol={"AAA": 2, "BBB": 1})
        assert len(res.errors) == 1
        code, msg = res.errors[0]
        assert code == "BOOM"
        assert "RuntimeError" in msg
        # the healthy symbols are still processed
        assert [s.symbol for s in res.sniper_pool] == ["AAA"]
        assert [s.symbol for s in res.watch_pool] == ["BBB"]
        assert res.rejected == 1          # DDD

    def test_summary_shape(self):
        root = _router()
        res = screen(root, END, universe=["AAA", "BBB", "DDD"],
                     catalysts_by_symbol={"AAA": 2, "BBB": 1})
        s = res.summary()
        assert s["sniper_pool"] == ["AAA"]
        assert s["watch_pool"] == ["BBB"]
        assert s["rejected"] == 1
        assert s["errors"] == 0

    def test_pools_sorted_by_total_desc(self):
        # two A-grade names; pool must be ordered best-first
        per = {"A1": c.build_textbook_provider(),
               "A2": c.build_textbook_provider()}
        root = c.FakeProvider(per_symbol=per)
        res = screen(root, END, universe=["A1", "A2"],
                     catalysts_by_symbol={"A1": 2, "A2": 2})
        totals = [s.total for s in res.sniper_pool]
        assert totals == sorted(totals, reverse=True)


class TestSpotDerivedUniverse:
    def test_universe_from_spot_filters_st_and_cap(self):
        per = {"AAA": c.build_textbook_provider()}
        spot = c.spot_frame([
            ("AAA", "测试股份", 8e9),     # in band -> kept
            ("STX", "ST退市", 8e9),       # ST name -> dropped
            ("TINY", "微盘股", 1e8),      # below cap_lo -> dropped
            ("HUGE", "巨无霸", 5e12),     # above cap_hi -> dropped
        ])
        root = c.FakeProvider(per_symbol=per, spot=spot)
        res = screen(root, END, catalysts_by_symbol={"AAA": 2})
        # only AAA survives the spot filter, and it grades into the sniper pool
        assert [s.symbol for s in res.sniper_pool] == ["AAA"]
        assert res.errors == []

    def test_empty_spot_yields_empty_result(self):
        root = c.FakeProvider()       # base spot() -> None
        res = screen(root, END)
        assert res.sniper_pool == []
        assert res.watch_pool == []
        assert res.rejected == 0
