"""Weekend full-market screening pipeline (spec checklist 9.1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

from .data import DataProvider
from .scoring import score_stock
from .types import StockScore


@dataclass
class ScreenResult:
    sniper_pool: list = field(default_factory=list)   # A+/A
    watch_pool: list = field(default_factory=list)    # B
    rejected: int = 0
    errors: list = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "sniper_pool": [s.symbol for s in self.sniper_pool],
            "watch_pool": [s.symbol for s in self.watch_pool],
            "rejected": self.rejected,
            "errors": len(self.errors),
        }


def _universe_from_spot(provider: DataProvider, cap_lo: float, cap_hi: float):
    spot = provider.spot()
    if spot is None or len(spot) == 0:
        return []
    code_c = next((c for c in spot.columns if "代码" in str(c)), None)
    name_c = next((c for c in spot.columns if "名称" in str(c)), None)
    cap_c = next((c for c in spot.columns if "流通市值" in str(c)), None)
    out = []
    for _, r in spot.iterrows():
        code = str(r[code_c]) if code_c else None
        name = str(r[name_c]) if name_c else ""
        if not code or "ST" in name.upper():
            continue
        if cap_c is not None:
            try:
                cap = float(r[cap_c])
                if not (cap_lo <= cap <= cap_hi):
                    continue
            except (TypeError, ValueError):
                pass
        out.append((code, name))
    return out


def screen(provider: DataProvider, end_date: str, *,
           universe: Optional[Iterable[str]] = None,
           cap_lo: float = 5e9, cap_hi: float = 1e11,
           catalysts_by_symbol: Optional[dict] = None) -> ScreenResult:
    """Run the 6-stage funnel (9.1.1-9.1.8) and bucket results.

    ``universe`` may be an explicit iterable of codes; otherwise it is derived
    from ``provider.spot()`` filtered to the [50亿, 1000亿] cap band.
    """
    catalysts_by_symbol = catalysts_by_symbol or {}
    result = ScreenResult()

    if universe is not None:
        pairs = [(str(s), None) for s in universe]
    else:
        pairs = _universe_from_spot(provider, cap_lo, cap_hi)

    for code, name in pairs:
        try:
            sc: StockScore = score_stock(
                code, end_date, provider,
                catalysts=catalysts_by_symbol.get(code, 0),
                stock_name=name)
        except Exception as exc:  # noqa: BLE001 - never let one bad feed abort the run
            result.errors.append((code, repr(exc)))
            continue
        if sc.grade in ("A+", "A"):
            result.sniper_pool.append(sc)
        elif sc.grade == "B":
            result.watch_pool.append(sc)
        else:
            result.rejected += 1

    result.sniper_pool.sort(key=lambda s: s.total, reverse=True)
    result.watch_pool.sort(key=lambda s: s.total, reverse=True)
    return result
