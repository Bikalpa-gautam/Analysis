"""Load combinations to EN 1990:2002 (+A1:2005), recommended values.

ULS STR/GEO set B: Eq. (6.10), or the pair (6.10a)/(6.10b).
SLS: characteristic (6.14b), frequent (6.15b), quasi-permanent (6.16b).

Every variable load case can be on or off ("pattern loading"); cases that
share a ``group`` are mutually exclusive (e.g. wind from the left or the
right). Every active variable case takes a turn as the leading action.
All permanent cases take the same partial factor (single-source principle)
unless ``vary_permanent_individually`` is set.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from .model import Model

# EN 1990 Table A1.1, recommended psi factors (psi0, psi1, psi2)
PSI: dict[str, tuple[float, float, float]] = {
    "A": (0.7, 0.5, 0.3),           # domestic, residential
    "B": (0.7, 0.5, 0.3),           # office
    "C": (0.7, 0.7, 0.6),           # congregation
    "D": (0.7, 0.7, 0.6),           # shopping
    "E": (1.0, 0.9, 0.8),           # storage
    "F": (0.7, 0.7, 0.6),           # traffic, vehicle weight <= 30 kN
    "G": (0.7, 0.5, 0.3),           # traffic, 30 kN < vehicle weight <= 160 kN
    "H": (0.0, 0.0, 0.0),           # roofs
    "snow": (0.5, 0.2, 0.0),        # sites at altitude H <= 1000 m a.s.l.
    "snow_high": (0.7, 0.5, 0.2),   # sites at altitude H > 1000 m a.s.l.
    "wind": (0.6, 0.2, 0.0),
    "temperature": (0.6, 0.5, 0.0), # non-fire
}
CATEGORY_LABEL = {
    "A": "Cat. A, residential", "B": "Cat. B, office", "C": "Cat. C, congregation",
    "D": "Cat. D, shopping", "E": "Cat. E, storage", "F": "Cat. F, light traffic",
    "G": "Cat. G, medium traffic", "H": "Cat. H, roofs", "snow": "Snow, H <= 1000 m",
    "snow_high": "Snow, H > 1000 m", "wind": "Wind", "temperature": "Temperature",
}

KINDS = ("ULS", "SLS_char", "SLS_freq", "SLS_qp")
MAX_VARIABLE_CASES = 12


@dataclass(frozen=True)
class Combination:
    id: str
    kind: str                       # ULS | SLS_char | SLS_freq | SLS_qp
    factors: tuple[float, ...]      # one per load case, in model order
    leading: str | None
    label: str


@dataclass
class CombinationSet:
    case_ids: list[str]
    combos: list[Combination]

    def matrix(self, kind: str | None = None) -> tuple[np.ndarray, list[Combination]]:
        sel = [c for c in self.combos if kind is None or c.kind == kind]
        return np.array([c.factors for c in sel], dtype=float).reshape(len(sel), len(self.case_ids)), sel


@dataclass(frozen=True)
class EN1990Factors:
    gamma_G_sup: float = 1.35
    gamma_G_inf: float = 1.00
    gamma_Q: float = 1.50
    xi: float = 0.85


def _label(case_ids, factors) -> str:
    parts = [f"{f:.2f}·{cid}" for cid, f in zip(case_ids, factors) if abs(f) > 1e-12]
    return " + ".join(parts) if parts else "0"


def _active_subsets(var_ids: list[str], groups: dict[str, str | None]):
    """All subsets of variable cases with at most one case per group."""
    grouped: dict[str, list[str]] = {}
    free: list[str] = []
    for cid in var_ids:
        g = groups[cid]
        (grouped.setdefault(g, []) if g else free).append(cid)
    choices = [[None] + members for members in grouped.values()] + [[None, cid] for cid in free]
    for pick in itertools.product(*choices):
        yield [c for c in pick if c is not None]


def generate(model: Model) -> CombinationSet:
    params = model.settings.params
    f = EN1990Factors(
        gamma_G_sup=float(params.get("gamma_G_sup", 1.35)),
        gamma_G_inf=float(params.get("gamma_G_inf", 1.00)),
        gamma_Q=float(params.get("gamma_Q", 1.50)),
        xi=float(params.get("xi", 0.85)),
    )
    psi = dict(PSI)
    for cat, vals in (params.get("psi") or {}).items():
        psi[cat] = tuple(float(v) for v in vals)
    vary_individually = bool(params.get("vary_permanent_individually", False))
    rule = model.settings.combination_rule

    case_ids = list(model.load_cases)
    idx = {cid: k for k, cid in enumerate(case_ids)}
    perm = [cid for cid in case_ids if model.load_cases[cid].type == "permanent"]
    var = [cid for cid in case_ids if model.load_cases[cid].type == "variable"]
    if len(var) > MAX_VARIABLE_CASES:
        raise ValueError(f"at most {MAX_VARIABLE_CASES} variable load cases are supported, got {len(var)}")
    groups = {cid: model.load_cases[cid].group for cid in var}
    ps = {cid: psi[model.load_cases[cid].category] for cid in var}

    if vary_individually:
        g_sets = list(itertools.product(*[[f.gamma_G_sup, f.gamma_G_inf] for _ in perm]))
    else:
        g_sets = [tuple(f.gamma_G_sup for _ in perm), tuple(f.gamma_G_inf for _ in perm)]

    seen: set[tuple] = set()
    combos: list[Combination] = []

    def add(kind: str, factors: list[float], leading: str | None):
        key = (kind, tuple(round(x, 9) for x in factors))
        if key in seen:
            return
        seen.add(key)
        n = sum(1 for c in combos if c.kind == kind) + 1
        prefix = {"ULS": "ULS", "SLS_char": "CHAR", "SLS_freq": "FREQ", "SLS_qp": "QP"}[kind]
        combos.append(Combination(f"{prefix}{n:03d}", kind, tuple(factors), leading, _label(case_ids, factors)))

    subsets = list(_active_subsets(var, groups))
    # --- ULS
    for gset in g_sets:
        for S in subsets:
            leads = S if S else [None]
            for lead in leads:
                if rule == "6.10":
                    vec = [0.0] * len(case_ids)
                    for cid, g in zip(perm, gset):
                        vec[idx[cid]] = g
                    for cid in S:
                        vec[idx[cid]] = f.gamma_Q * (1.0 if cid == lead else ps[cid][0])
                    add("ULS", vec, lead)
                else:  # 6.10a and 6.10b
                    a = [0.0] * len(case_ids)
                    b = [0.0] * len(case_ids)
                    for cid, g in zip(perm, gset):
                        a[idx[cid]] = g
                        b[idx[cid]] = f.xi * g if g > 1.0 else g
                    for cid in S:
                        a[idx[cid]] = f.gamma_Q * ps[cid][0]
                        b[idx[cid]] = f.gamma_Q * (1.0 if cid == lead else ps[cid][0])
                    add("ULS", a, None)
                    add("ULS", b, lead)
    # --- SLS
    for S in subsets:
        for lead in (S if S else [None]):
            ch = [0.0] * len(case_ids)
            fr = [0.0] * len(case_ids)
            for cid in perm:
                ch[idx[cid]] = fr[idx[cid]] = 1.0
            for cid in S:
                ch[idx[cid]] = 1.0 if cid == lead else ps[cid][0]
                fr[idx[cid]] = ps[cid][1] if cid == lead else ps[cid][2]
            add("SLS_char", ch, lead)
            add("SLS_freq", fr, lead)
        qp = [0.0] * len(case_ids)
        for cid in perm:
            qp[idx[cid]] = 1.0
        for cid in S:
            qp[idx[cid]] = ps[cid][2]
        add("SLS_qp", qp, None)
    return CombinationSet(case_ids, combos)
