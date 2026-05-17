"""Command-line entry point.

    python -m ashare_main_wave score 300308 --date 20230430
    python -m ashare_main_wave screen --date 20230430 --universe 300308,688256
    python -m ashare_main_wave backtest --demo
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

    sb = sub.add_parser("backtest", help="走查回测")
    sb.add_argument("--demo", action="store_true",
                    help="使用内置离线合成历史（无需 akshare/网络）")
    sb.add_argument("--universe", default=None,
                    help="逗号分隔代码（实盘模式必填）")
    sb.add_argument("--start", default=None, help="起始 YYYYMMDD")
    sb.add_argument("--end", default=None, help="结束 YYYYMMDD")
    sb.add_argument("--rebalance", default="W", choices=["D", "W", "M"])
    sb.add_argument("--max-positions", type=int, default=3)
    sb.add_argument("--min-grade", default="B",
                    choices=["A+", "A", "B", "C"])

    args = p.parse_args(argv)

    if args.cmd == "backtest" and args.demo:
        return _run_demo_backtest(args)

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

    if args.cmd == "backtest":
        from .backtest import Backtester
        if not args.universe or not args.start or not args.end:
            print("错误: 实盘回测需 --universe/--start/--end，"
                  "或加 --demo 用离线合成历史", file=sys.stderr)
            return 2
        bt = Backtester(provider, rebalance=args.rebalance,
                        max_positions=args.max_positions,
                        min_grade=args.min_grade)
        res = bt.run(args.universe.split(","), args.start, args.end)
        print(json.dumps(res.summary(), ensure_ascii=False, indent=2,
                         default=str))
        return 0

    return 1


def _run_demo_backtest(args) -> int:
    from .backtest import Backtester
    from .data import HistoricalFrameProvider
    from .demo_data import demo_history
    h = demo_history()
    prov = HistoricalFrameProvider(**h)
    win = h["klines"]["000001"]["daily"]
    start = args.start or win["date"].iloc[420].strftime("%Y%m%d")
    end = args.end or win["date"].iloc[-1].strftime("%Y%m%d")
    bt = Backtester(prov, rebalance=args.rebalance,
                    max_positions=args.max_positions,
                    min_grade=args.min_grade,
                    score_kwargs_fn=lambda s: (
                        {"catalysts": 2, "leader_rank": 1,
                         "limitups_in_sector": 12} if s == "000001" else {}))
    res = bt.run(list(h["klines"]), start, end)
    print(json.dumps(res.summary(), ensure_ascii=False, indent=2,
                      default=str))
    tf = res.trades_frame()
    print("\n--- 交易明细 ---")
    print(tf.to_string(index=False) if len(tf) else "（无成交）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
