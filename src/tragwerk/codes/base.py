"""Common structures for traceable code checks.

Every check returns a :class:`CheckResult` carrying the clause it applies,
the governing location and combination, and the calculation steps with the
values substituted, so a checker can follow each number back to the code.

Symbols use a light markup shared with the report and the browser app:
``V_{Rd,c}`` is a subscript, ``x^{1/3}`` a superscript.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

PASS, FAIL, WARN, INFO, NOT_COVERED = "pass", "fail", "warn", "info", "not_covered"


@dataclass
class Step:
    symbol: str          # e.g. "V_{Rd,c}"
    expression: str      # formula in symbols
    substituted: str     # the same formula with numbers
    value: float
    unit: str = ""
    clause: str = ""
    note: str = ""


@dataclass
class CheckResult:
    id: str              # stable id, e.g. "ec2.shear"
    title: str
    clause: str
    member: str
    status: str
    utilisation: float | None = None
    demand: float | None = None
    capacity: float | None = None
    unit: str = ""
    x: float | None = None          # governing position along the member (m)
    combination: str | None = None  # governing combination id
    combination_label: str | None = None
    steps: list[Step] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        if d["utilisation"] is not None and not math.isfinite(d["utilisation"]):
            d["utilisation"] = None
            d["notes"] = d["notes"] + ["utilisation is unbounded (no resistance in this direction)"]
        return d


def status_from_utilisation(u: float, limit: float = 1.0) -> str:
    return PASS if u <= limit + 1e-9 else FAIL


def fmt(v: float, digits: int = 3) -> str:
    """Format a number for a substituted expression."""
    if v == 0:
        return "0"
    if not math.isfinite(v):
        return "∞"
    a = abs(v)
    if 1e5 <= a < 1e12:
        return f"{v:,.0f}".replace(",", " ")
    if a >= 1e12 or a < 1e-3:
        return f"{v:.{digits}e}"
    if a >= 100:
        return f"{v:.1f}"
    if a >= 10:
        return f"{v:.2f}"
    return f"{v:.{digits}g}" if a < 1 else f"{v:.3f}"
