"""Shared result containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DimensionResult:
    name: str
    score: float                    # weighted sub-indicator avg, 0..10
    weight: float                   # dimension weight (spec 4.1)
    subscores: dict = field(default_factory=dict)   # id -> 0/3/6/10
    metrics: dict = field(default_factory=dict)     # raw computed values
    veto: Optional[str] = None      # non-None => one-vote veto fired

    @property
    def contribution(self) -> float:
        """Points this dimension adds to the 0..100 total (score*weight*10)."""
        return self.score * self.weight * 10.0


@dataclass
class StockScore:
    symbol: str
    date: str
    total: float
    grade: str
    dimensions: dict = field(default_factory=dict)   # name -> DimensionResult
    vetoes: list = field(default_factory=list)        # fired veto reasons
    notes: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "date": self.date,
            "total": round(self.total, 2),
            "grade": self.grade,
            "vetoes": self.vetoes,
            "dimensions": {
                k: {
                    "score": round(v.score, 2),
                    "contribution": round(v.contribution, 2),
                    "subscores": v.subscores,
                    "metrics": {mk: (round(mv, 4)
                                     if isinstance(mv, float) else mv)
                                for mk, mv in v.metrics.items()},
                }
                for k, v in self.dimensions.items()
            },
            "notes": self.notes,
        }
