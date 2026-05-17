"""CLI smoke test for the offline demo back-test.

``ashare_main_wave.cli.main(["backtest", "--demo", ...])`` must run fully
offline (no akshare / network), return exit code 0, and print a JSON metrics
block followed by a trade-detail table.  We capture stdout and round-trip the
JSON block so a malformed payload would fail loudly.
"""

from __future__ import annotations

import json

from ashare_main_wave.cli import main


def _extract_json_block(stdout: str) -> str:
    """The CLI prints a pretty JSON object first, then a ``--- 交易明细 ---``
    section.  Slice out the leading ``{ ... }`` object."""
    start = stdout.index("{")
    depth = 0
    for i in range(start, len(stdout)):
        c = stdout[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return stdout[start:i + 1]
    raise AssertionError("no balanced JSON object found in CLI output")


class TestCliBacktest:
    def test_demo_backtest_returns_zero_and_prints_json(self, capsys):
        rc = main(["backtest", "--demo", "--max-positions", "2"])
        assert rc == 0

        out = capsys.readouterr().out
        assert "--- 交易明细 ---" in out

        block = _extract_json_block(out)
        # json.loads accepts the non-strict `Infinity` token that
        # json.dumps emits for an infinite profit_factor.
        metrics = json.loads(block)

        # parameter echo
        assert metrics["max_positions"] == 2
        assert metrics["initial_capital"] == 1_000_000.0
        assert metrics["universe"] == 2
        assert metrics["rebalance"] == "W"

        # metric block present and well-typed
        for key in ("total_return", "cagr", "max_drawdown",
                    "num_trades", "win_rate", "profit_factor"):
            assert key in metrics
        assert isinstance(metrics["num_trades"], int)
        assert metrics["max_drawdown"] <= 0.0
        assert 0.0 <= metrics["win_rate"] <= 1.0
        # profit_factor is a number or +inf (never NaN / negative)
        pf = metrics["profit_factor"]
        assert pf >= 0.0

    def test_demo_default_max_positions(self, capsys):
        rc = main(["backtest", "--demo"])
        assert rc == 0
        out = capsys.readouterr().out
        metrics = json.loads(_extract_json_block(out))
        assert metrics["max_positions"] == 3      # argparse default

    def test_live_backtest_without_universe_errors(self, capsys):
        # no akshare in the env -> provider construction fails with rc 2,
        # and even before that the missing --universe path returns non-zero.
        rc = main(["backtest", "--max-positions", "2"])
        assert rc == 2
