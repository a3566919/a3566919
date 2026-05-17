"""Command-line entry point.

    python -m ashare_main_wave score 300308 --date 20230430
    python -m ashare_main_wave screen --date 20230430 --universe 300308,688256
"""

from __future__ import annotations

import argparse
import json
import sys


def _provider():
    from .data import AkShareProvider
    return AkShareProvider()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="ashare_main_wave",
                                description="A股主升浪 5 维共振评分")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("score", help="对单只股票评分")
    sp.add_argument("symbol")
    sp.add_argument("--date", required=True, help="评估日 YYYYMMDD")
    sp.add_argument("--catalysts", type=int, default=0)

    ss = sub.add_parser("screen", help="全市场/指定池筛选")
    ss.add_argument("--date", required=True)
    ss.add_argument("--universe", default=None,
                    help="逗号分隔的代码；缺省则用 spot 全市场")

    args = p.parse_args(argv)

    try:
        provider = _provider()
    except RuntimeError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2

    if args.cmd == "score":
        from .scoring import score_stock
        res = score_stock(args.symbol, args.date, provider,
                           catalysts=args.catalysts)
        print(json.dumps(res.as_dict(), ensure_ascii=False, indent=2))
        return 0

    if args.cmd == "screen":
        from .screener import screen
        uni = args.universe.split(",") if args.universe else None
        res = screen(provider, args.date, universe=uni)
        print(json.dumps(res.summary(), ensure_ascii=False, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
