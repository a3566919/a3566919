#!/usr/bin/env python3
"""Runnable, fully-offline demo of the ashare_main_wave scoring engine.

    python examples/demo.py

Builds the *textbook* 主升浪 scenario with the same synthetic data helpers
the test-suite uses (no network, no akshare), scores it, and prints:

  * the full ``StockScore.as_dict()`` (5-dimension breakdown + grade),
  * a position plan for the resulting grade,
  * the three-way stop-loss trigger,
  * a take-profit decision example.

Exits 0 on success.
"""

from __future__ import annotations

import json
import os
import sys

# The synthetic-data builders live in the test-suite's conftest; reuse them
# so the demo and the tests stay in lock-step.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "tests"))

import conftest as scenarios  # noqa: E402

from ashare_main_wave import (position_plan, score_stock,  # noqa: E402
                              stop_loss_price, take_profit_action)


def _hr(title: str) -> None:
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


def main() -> int:
    symbol, end_date = "600519", "20240430"

    _hr(f"Scoring {symbol} @ {end_date}  (textbook 主升浪 scenario)")
    provider = scenarios.build_textbook_provider()
    score = score_stock(symbol, end_date, provider,
                         **scenarios.textbook_score_kwargs())

    print(json.dumps(score.as_dict(), ensure_ascii=False, indent=2))

    print(f"\n-> total = {score.total:.2f} / 100   grade = {score.grade}")
    if score.vetoes:
        print(f"-> vetoes: {score.vetoes}")
    else:
        print("-> no one-vote veto fired")

    _hr(f"Position plan for grade {score.grade} (spec 7.1 / 7.2)")
    plan = position_plan(score.grade)
    print(f"initial = {plan.initial:.0%} of capital, "
          f"max = {plan.max:.0%}")
    for label, frac, condition in plan.pyramid:
        print(f"  + {label}: add {frac:.0%}  when  {condition}")

    _hr("Stop-loss (spec 7.3) -- highest/tightest of the three triggers")
    entry = 50.0
    sl = stop_loss_price(entry, cap_size="main",
                         platform_top=entry * 0.95,
                         ma20=entry * 0.97)
    print(f"entry = {entry:.2f}")
    for name, level in sl["candidates"].items():
        print(f"  {name:>16}: {level:.3f}")
    print(f"  -> effective stop = {sl['effective_stop']:.3f} "
          f"(absolute leg = {sl['absolute_pct']:.0%})")

    _hr("Take-profit ladder (spec 7.4) -- a few P&L states")
    for pnl, below_ma10, wk_dead in [
        (0.10, False, False),
        (0.35, True, False),
        (0.65, True, False),
        (1.20, False, False),
        (1.20, False, True),
    ]:
        d = take_profit_action(pnl, below_ma10=below_ma10,
                               weekly_macd_dead=wk_dead)
        flags = []
        if below_ma10:
            flags.append("跌破10日线")
        if wk_dead:
            flags.append("周线MACD死叉")
        tag = (" [" + ", ".join(flags) + "]") if flags else ""
        print(f"  P&L {pnl:+.0%}{tag}: {d['action']} "
              f"(band floor {d['band_floor']:+.0%})")

    assert score.grade in ("A", "A+"), score.grade
    assert not score.vetoes
    print("\nDemo completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
