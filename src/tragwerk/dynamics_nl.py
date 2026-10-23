"""Earthquake time-history analysis: ground motions, response spectra, a hysteretic oscillator and nonlinear FE dynamics.

Follows the dynamics part of the ETH Method of Finite Elements II course (Prof. E. Chatzi):

* **Ground motion.** Synthetic accelerograms compatible with the EN 1998-1 elastic spectrum (3.2.2.2, 3.2.3.1.2):
  a sum of sinusoids with seeded random phases under a trapezoidal intensity envelope, whose amplitudes are
  corrected iteratively until the 5 % response spectrum matches the target (Gasparini & Vanmarcke 1976, SIMQKE),
  with a baseline correction that brings the final velocity and displacement to zero. The random phases come from
  a 32-bit linear congruential generator, so a seed gives the same record in Python and in the browser.
* **Response spectra** by the exact piecewise-linear (Nigam–Jennings) recurrence for linear SDOF systems.
* **Hysteretic SDOF.** Bouc–Wen oscillator m ü + c u̇ + α k u + (1 − α) k u_y z = −m a_g with
  dz/du = [A − |z|ⁿ (β sgn(du z) + γ)] / u_y, integrated by Newmark's average acceleration method with
  Newton–Raphson iterations; energy balance E_in = E_kin + E_damp + E_hyst + E_strain.
* **Nonlinear FE dynamics.** Implicit Newmark (average acceleration) on the plane FE model of tragwerk.fe2d_nl
  (elastic, J2 or reinforced concrete): consistent element mass plus the gravity loads of the seismic combination
  G + ψ₂Q as lumped mass, Rayleigh damping fitted to the first two modes, gravity applied first by a static
  analysis, then the ground motion in x.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh, splu

from .fe2d import FEError, FEModel, element_mass
from .fe2d_nl import NLOptions, Problem, run_nonlinear

G_ACC = 9.81
SPECTRUM_EC8 = {
    1: {"A": (1.0, 0.15, 0.4, 2.0), "B": (1.2, 0.15, 0.5, 2.0), "C": (1.15, 0.20, 0.6, 2.0), "D": (1.35, 0.20, 0.8, 2.0), "E": (1.4, 0.15, 0.5, 2.0)},
    2: {"A": (1.0, 0.05, 0.25, 1.2), "B": (1.35, 0.05, 0.25, 1.2), "C": (1.5, 0.10, 0.25, 1.2), "D": (1.8, 0.10, 0.30, 1.2), "E": (1.6, 0.05, 0.25, 1.2)},
}


# ============================================================== EN 1998-1 elastic spectrum
def elastic_spectrum(T, ag: float, soil: str = "B", stype: int = 1, xi: float = 0.05) -> np.ndarray:
    """Horizontal elastic response spectrum S_e(T) (EN 1998-1 3.2.2.2, Eqs. 3.2–3.6) in the units of ag."""
    if stype not in SPECTRUM_EC8 or soil not in SPECTRUM_EC8[stype]:
        raise FEError("spectrum: type 1 or 2 and ground type A–E")
    S, TB, TC, TD = SPECTRUM_EC8[stype][soil]
    eta = max(math.sqrt(10.0 / (5.0 + 100.0 * xi)), 0.55)
    T = np.asarray(T, dtype=float)
    out = np.where(T <= TB, ag * S * (1 + T / TB * (eta * 2.5 - 1)),
                   np.where(T <= TC, ag * S * eta * 2.5,
                            np.where(T <= TD, ag * S * eta * 2.5 * TC / np.maximum(T, 1e-12), ag * S * eta * 2.5 * TC * TD / np.maximum(T, 1e-12) ** 2)))
    return out


# ============================================================== random numbers shared with the browser
class LCG:
    """32-bit linear congruential generator (Numerical Recipes constants), identical in Python and JavaScript."""

    def __init__(self, seed: int):
        self.x = int(seed) & 0xFFFFFFFF

    def next(self) -> float:
        self.x = (1664525 * self.x + 1013904223) & 0xFFFFFFFF
        return self.x / 4294967296.0


# ============================================================== response spectrum (Nigam–Jennings)
def _nj_coefficients(T: np.ndarray, xi: float, dt: float):
    w = 2 * math.pi / T
    wd = w * math.sqrt(1 - xi * xi)
    e = np.exp(-xi * w * dt)
    s, c = np.sin(wd * dt), np.cos(wd * dt)
    r = xi / math.sqrt(1 - xi * xi)
    k = w * w
    A = e * (r * s + c)
    B = e * s / wd
    C = (2 * xi / (w * dt) + e * (((1 - 2 * xi * xi) / (wd * dt) - r) * s - (1 + 2 * xi / (w * dt)) * c)) / k
    D = (1 - 2 * xi / (w * dt) + e * ((2 * xi * xi - 1) / (wd * dt) * s + 2 * xi / (w * dt) * c)) / k
    A1 = -e * w / math.sqrt(1 - xi * xi) * s
    B1 = e * (c - r * s)
    C1 = (-1 / dt + e * ((w / math.sqrt(1 - xi * xi) + xi / (dt * math.sqrt(1 - xi * xi))) * s + c / dt)) / k
    D1 = (1 - e * (r * s + c)) / (k * dt)
    return A, B, C, D, A1, B1, C1, D1


def response_spectrum(a: np.ndarray, dt: float, periods, xi: float = 0.05) -> dict:
    """Pseudo-acceleration S_a = ω² S_d, S_d and S_v of linear SDOF oscillators, exact for an acceleration that varies
    linearly within each time step (Chopra, Dynamics of Structures, 5.2)."""
    T = np.asarray(periods, dtype=float)
    A, B, C, D, A1, B1, C1, D1 = _nj_coefficients(T, xi, dt)
    u = np.zeros_like(T)
    v = np.zeros_like(T)
    umax = np.zeros_like(T)
    vmax = np.zeros_like(T)
    p = -np.asarray(a, dtype=float)
    for i in range(len(p) - 1):
        u, v = A * u + B * v + C * p[i] + D * p[i + 1], A1 * u + B1 * v + C1 * p[i] + D1 * p[i + 1]
        umax = np.maximum(umax, np.abs(u))
        vmax = np.maximum(vmax, np.abs(v))
    w = 2 * math.pi / T
    return {"T": T, "Sd": umax, "Sv": vmax, "Sa": w * w * umax}


def sdof_history(a: np.ndarray, dt: float, T: float, xi: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Relative displacement and velocity histories of a linear SDOF oscillator (exact, Nigam–Jennings)."""
    A, B, C, D, A1, B1, C1, D1 = (float(x[0]) for x in _nj_coefficients(np.array([T]), xi, dt))
    p = -np.asarray(a, dtype=float)
    u, v = np.zeros(len(p)), np.zeros(len(p))
    for i in range(len(p) - 1):
        u[i + 1] = A * u[i] + B * v[i] + C * p[i] + D * p[i + 1]
        v[i + 1] = A1 * u[i] + B1 * v[i] + C1 * p[i] + D1 * p[i + 1]
    return u, v


# ============================================================== synthetic accelerogram
def _envelope(t: np.ndarray, t1: float, t2: float, decay: float) -> np.ndarray:
    """Trapezoidal intensity function (Jennings, Housner & Tsai 1968): (t/t1)² rise, strong phase, exponential decay."""
    return np.where(t < t1, (t / t1) ** 2, np.where(t <= t2, 1.0, np.exp(-decay * (t - t2))))


def _baseline(a: np.ndarray, dt: float) -> np.ndarray:
    """Subtract c0 + c1 t so that the final velocity and displacement are zero (piecewise-linear acceleration)."""
    n = len(a)
    T = (n - 1) * dt
    v = np.concatenate([[0.0], np.cumsum(0.5 * (a[1:] + a[:-1]) * dt)])
    d = np.concatenate([[0.0], np.cumsum(v[:-1] * dt + (2 * a[:-1] + a[1:]) * dt * dt / 6)])
    M = np.array([[T, T * T / 2], [T * T / 2, T ** 3 / 6]])
    c0, c1 = np.linalg.solve(M, [v[-1], d[-1]])
    t = np.arange(n) * dt
    return a - c0 - c1 * t


@dataclass
class Record:
    t: np.ndarray
    a: np.ndarray                     # m/s²
    dt: float
    name: str = ""
    target: dict | None = None        # {"ag", "soil", "type", "xi"} for synthetic records
    match: dict | None = None         # periods, target and achieved spectra

    @property
    def pga(self) -> float:
        return float(np.abs(self.a).max())


def synthetic_accelerogram(ag_g: float = 0.25, soil: str = "B", stype: int = 1, duration: float = 20.0, dt: float = 0.01, seed: int = 1,
                           rise: float = 2.0, strong: float = 10.0, iterations: int = 15, xi: float = 0.05) -> Record:
    """Spectrum-compatible artificial accelerogram (EN 1998-1 3.2.3.1.2). ag_g is the design ground acceleration on rock
    in g; the record is matched to S_e(T) from T = 1.5/f_max (0.06 s for Δt = 0.01 s) to 4 s and scaled so that its peak
    reaches a_g S."""
    if not (0 < ag_g <= 2.0) or not (2.0 <= duration <= 120.0) or not (0.002 <= dt <= 0.05):
        raise FEError("record: 0 < a_g ≤ 2 g, 2 s ≤ duration ≤ 120 s, 0.002 s ≤ Δt ≤ 0.05 s")
    if rise + strong >= duration:
        raise FEError("record: rise + strong phase must be shorter than the duration")
    ag = ag_g * G_ACC
    n = int(round(duration / dt)) + 1
    t = np.arange(n) * dt
    env = _envelope(t, rise, rise + strong, -math.log(0.05) / (duration - rise - strong))   # 5 % of the amplitude at the end
    fmax = min(25.0, 0.4 / dt)
    # frequencies 2 % apart (closer than the half-power bandwidth 2ξ of the 5 % oscillators, so the spectrum is smooth
    # between them), but never closer than 1/(2·duration), below which components cannot be told apart in the record
    fl = [0.1]
    while fl[-1] < fmax:
        fl.append(fl[-1] + max(0.02 * fl[-1], 0.5 / duration))
    f = np.array(fl[:-1])
    w = 2 * math.pi * f
    dw = np.gradient(w)
    rng = LCG(seed)
    phi = np.array([2 * math.pi * rng.next() for _ in f])
    Tk = 1.0 / f
    target = elastic_spectrum(Tk, ag, soil, stype, xi)
    # initial amplitudes from the spectral density of a stationary process (Vanmarcke): G(ω) ≈ ξ S_a² / (π ω r²), r = 2.5
    G = xi * target ** 2 / (math.pi * w * 2.5 ** 2)
    amp = np.sqrt(2 * G * dw)
    S = np.sin(np.outer(t, w) + phi[None, :])               # (n, nf)
    match_T = np.geomspace(max(0.05, 1.5 / fmax), 4.0, 60)
    match_target = elastic_spectrum(match_T, ag, soil, stype, xi)
    a = None
    for _ in range(iterations):
        a = _baseline(env * (S @ amp), dt)
        sp = response_spectrum(a, dt, Tk, xi)["Sa"]
        amp = amp * np.clip(target / np.maximum(sp, 1e-12), 0.5, 2.0)
    a = _baseline(env * (S @ amp), dt)
    # the zero-period value (peak ground acceleration) must reach a_g S (3.2.3.1.2(4)a)
    a *= max(1.0, ag * SPECTRUM_EC8[stype][soil][0] / np.abs(a).max())
    achieved = response_spectrum(a, dt, match_T, xi)["Sa"]
    return Record(t=t, a=a, dt=dt, name=f"synthetic, a_g = {ag_g:g} g, ground {soil}, type {stype}, seed {seed}",
                  target={"ag_g": ag_g, "soil": soil, "type": stype, "xi": xi, "duration": duration, "seed": seed},
                  match={"T": match_T, "target": match_target, "achieved": achieved})


def record_from_values(values_g, dt: float, name: str = "record") -> Record:
    a = np.asarray(values_g, dtype=float) * G_ACC
    if a.ndim != 1 or len(a) < 3 or not np.all(np.isfinite(a)):
        raise FEError("record: at least three finite acceleration values (in g) are needed")
    if not (0.0005 <= dt <= 0.1):
        raise FEError("record: 0.0005 s ≤ Δt ≤ 0.1 s")
    return Record(t=np.arange(len(a)) * dt, a=a, dt=float(dt), name=name)


def parse_record(spec: dict) -> Record:
    """{"type": "synthetic", "ag_g", "soil", "spectrum_type", "duration_s", "dt_s", "seed"} or {"type": "values", "dt_s", "values_g"}."""
    kind = spec.get("type", "synthetic")
    if kind == "synthetic":
        return synthetic_accelerogram(float(spec.get("ag_g", 0.25)), str(spec.get("soil", "B")), int(spec.get("spectrum_type", 1)),
                                      float(spec.get("duration_s", 20.0)), float(spec.get("dt_s", 0.01)), int(spec.get("seed", 1)),
                                      float(spec.get("rise_s", 2.0)), float(spec.get("strong_s", 10.0)))
    if kind == "values":
        return record_from_values(spec.get("values_g", []), float(spec.get("dt_s", 0.01)), str(spec.get("name", "record")))
    raise FEError("record.type must be synthetic or values")


# ============================================================== Bouc–Wen oscillator
@dataclass
class BWParams:
    m: float = 100.0          # t
    k: float = 4e4            # kN/m
    uy: float = 0.01          # m
    alpha: float = 0.05       # post-yield stiffness ratio
    n: float = 2.0            # smoothness of the transition
    beta: float = 0.5
    gamma: float = 0.5
    A: float = 1.0
    xi: float = 0.05          # viscous damping ratio (initial stiffness)

    @property
    def period(self) -> float:
        return 2 * math.pi * math.sqrt(self.m / self.k)


def bw_dz(z, du, p: BWParams):
    """dz/du of the rate-independent Bouc–Wen law."""
    return (p.A - np.abs(z) ** p.n * (p.beta * np.sign(du * z) + p.gamma)) / p.uy


def bw_advance(z0: float, du: float, p: BWParams, sub: int | None = None) -> tuple[float, float]:
    """z after a displacement increment du (classical RK4 in u) and dz/du at the end."""
    m = sub or max(1, min(200, int(math.ceil(8 * abs(du) / p.uy))))
    h = du / m
    z = z0
    for _ in range(m):
        k1 = bw_dz(z, du, p)
        k2 = bw_dz(z + 0.5 * h * k1, du, p)
        k3 = bw_dz(z + 0.5 * h * k2, du, p)
        k4 = bw_dz(z + h * k3, du, p)
        z = z + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return float(z), float(bw_dz(z, du if du != 0 else 1.0, p))


def bouc_wen_sdof(p: BWParams, rec: Record, newmark=(0.5, 0.25), tol: float = 1e-10, max_iter: int = 30) -> dict:
    """Newmark (average acceleration) with Newton–Raphson for the Bouc–Wen oscillator under the ground motion."""
    gam, bet = newmark
    dt = rec.dt
    a_g = rec.a
    n = len(a_g)
    c = 2 * p.xi * math.sqrt(p.k * p.m)
    u, v, acc, z, fs = (np.zeros(n) for _ in range(5))
    acc[0] = (-p.m * a_g[0] - c * v[0] - 0.0) / p.m
    E_in, E_d, E_s = np.zeros(n), np.zeros(n), np.zeros(n)
    it_tot = 0
    for i in range(n - 1):
        f_ext = -p.m * a_g[i + 1]
        du = 0.0
        zi, kz = z[i], bw_dz(z[i], 1.0, p)
        for it in range(max_iter):
            u1 = u[i] + du
            v1 = gam / (bet * dt) * du + (1 - gam / bet) * v[i] + dt * (1 - gam / (2 * bet)) * acc[i]
            a1 = du / (bet * dt * dt) - v[i] / (bet * dt) - (1 / (2 * bet) - 1) * acc[i]
            zi, dzdu = bw_advance(z[i], du, p)
            f1 = p.alpha * p.k * u1 + (1 - p.alpha) * p.k * p.uy * zi
            r = f_ext - p.m * a1 - c * v1 - f1
            kt = p.alpha * p.k + (1 - p.alpha) * p.k * p.uy * dzdu
            keff = p.m / (bet * dt * dt) + c * gam / (bet * dt) + kt
            if abs(r) <= tol * max(abs(f_ext), p.k * p.uy * 1e-3) and it > 0:
                break
            du += r / keff
            it_tot += 1
        u[i + 1], v[i + 1], acc[i + 1], z[i + 1], fs[i + 1] = u1, v1, a1, zi, f1
        dU = u[i + 1] - u[i]
        E_in[i + 1] = E_in[i] - p.m * 0.5 * (a_g[i] + a_g[i + 1]) * dU
        E_d[i + 1] = E_d[i] + c * 0.5 * (v[i] + v[i + 1]) * dU
        E_s[i + 1] = E_s[i] + 0.5 * (fs[i] + fs[i + 1]) * dU
    E_k = 0.5 * p.m * v ** 2
    E_strain = 0.5 * p.alpha * p.k * u ** 2 + 0.5 * (1 - p.alpha) * p.k * p.uy ** 2 * z ** 2   # recoverable part (approximation)
    E_h = E_s - E_strain
    return {"t": rec.t, "u": u, "v": v, "a_rel": acc, "a_abs": acc + a_g, "z": z, "fs": fs, "E_in": E_in, "E_kin": E_k, "E_damp": E_d,
            "E_s": E_s, "E_hyst": E_h, "E_strain": E_strain, "u_max": float(np.abs(u).max()), "ductility": float(np.abs(u).max() / p.uy),
            "fs_max": float(np.abs(fs).max()), "iterations": it_tot,
            "energy_error": float(np.abs(E_in - E_k - E_d - E_s).max() / max(np.abs(E_in).max(), 1e-30))}


# ============================================================== nonlinear FE dynamics
@dataclass
class DynOptions:
    combination: str = "qp"                 # gravity of the seismic situation: G + ψ₂ Q (EN 1990 6.4.3.4)
    mass_from_loads: bool = True            # the gravity loads of the combination act as lumped mass (EN 1998-1 3.2.4)
    xi: float = 0.05                        # Rayleigh damping in modes 1 and 2
    newmark: tuple = (0.5, 0.25)            # (γ, β): average acceleration, unconditionally stable, no numerical damping
    tol: float | None = None
    max_iter: int = 30
    direction: int = 0                      # 0: ground motion in x, 1: in y
    snapshots: int = 120                    # stored states for replay
    monitor: tuple | None = None            # (node, component); default: the node of the largest first-mode displacement in x


@dataclass
class DynResult:
    t: np.ndarray
    u_mon: np.ndarray
    v_mon: np.ndarray
    a_mon_abs: np.ndarray
    base_shear: np.ndarray
    periods: list
    rayleigh: tuple
    monitor_dof: int
    frames: list = field(default_factory=list)   # (time index, u, gp stress, extras)
    energy: dict = field(default_factory=dict)
    status: str = "complete"
    message: str = ""
    iterations: list = field(default_factory=list)
    substeps: list = field(default_factory=list)
    mass_total: float = 0.0
    model: FEModel | None = None


def assemble_mass(m: FEModel, opts: DynOptions):
    """Consistent element mass (ρ = γ/g) plus the gravity loads as lumped mass in both directions; dense diagonal blocks
    are kept sparse (COO → CSR)."""
    n2 = 2 * len(m.nodes)
    rows, cols, vals = [], [], []
    rho = m.gamma / G_ACC if m.self_weight else 0.0
    if rho > 0:
        for e in m.elements:
            Me = element_mass(m.etype, m.nodes[e], rho, m.t)
            dofs = np.empty(2 * len(e), dtype=int)
            dofs[0::2], dofs[1::2] = 2 * e, 2 * e + 1
            rows.append(np.repeat(dofs, len(dofs)))
            cols.append(np.tile(dofs, len(dofs)))
            vals.append(Me.ravel())
    if opts.mass_from_loads:
        from .fe2d import combination_factors
        aG, aQ = combination_factors(m, opts.combination)
        fl = aG * (m.forces - m.forces_q) + aQ * m.forces_q          # the self weight is already in the element mass
        ml = np.maximum(-fl[1::2], 0.0) / G_ACC
        for i in np.flatnonzero(ml > 0):
            rows.append(np.array([2 * i, 2 * i + 1]))
            cols.append(np.array([2 * i, 2 * i + 1]))
            vals.append(np.array([ml[i], ml[i]]))
    if not rows:
        raise FEError("the model has no mass: give the material a density (self weight) or gravity loads")
    return coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n2, n2)).tocsr()


def fe_dynamic(m: FEModel, rec: Record, opts: DynOptions | None = None) -> DynResult:
    o = opts or DynOptions()
    gam, bet = o.newmark
    dt = rec.dt
    # 1. gravity: static (nonlinear) analysis of the seismic combination
    st_opts = NLOptions(combination=o.combination, steps=4, lam_max=1.0)
    stat = run_nonlinear(m, st_opts)
    if stat.status != "complete":
        raise FEError(f"the gravity loads of the seismic combination could not be applied: {stat.message}")
    P = Problem(m, NLOptions())
    from .fe2d import load_vector
    f_grav = load_vector(m, o.combination)
    k = P.k
    n2 = k.n2
    fixed = np.array(sorted(m.fixed), dtype=int)
    free = np.setdiff1d(np.arange(n2), fixed)
    M = assemble_mass(m, o)
    _, K0, _, _, _ = P.respond(np.zeros(n2), P.state0)
    Mff, K0ff = M[free][:, free].tocsc(), K0[free][:, free].tocsc()
    # 2. modes 1 and 2 of the initial stiffness for Rayleigh damping (and the monitor)
    nmodes = min(3, len(free) - 1)
    lam_e, vec = eigsh(K0ff, k=nmodes, M=Mff, sigma=0.0, which="LM")
    order = np.argsort(lam_e)
    lam_e, vec = lam_e[order], vec[:, order]
    w = np.sqrt(np.maximum(lam_e, 1e-30))
    w1, w2 = w[0], w[1] if len(w) > 1 else 3 * w[0]
    a0 = o.xi * 2 * w1 * w2 / (w1 + w2)
    a1 = o.xi * 2 / (w1 + w2)
    if o.monitor is None:
        phi = np.zeros(n2)
        phi[free] = vec[:, 0]
        comp = np.abs(phi[o.direction::2])
        mon = 2 * int(np.flatnonzero(comp >= (1 - 1e-6) * comp.max())[0]) + o.direction   # ties: lowest node number
    else:
        mon = 2 * int(o.monitor[0]) + int(o.monitor[1])
    mf = int(np.searchsorted(free, mon))
    r = np.zeros(n2)
    r[o.direction::2] = 1.0
    Mr = (M @ r)[free]
    tol = o.tol if o.tol is not None else (1e-4 if P.kind == "rc" else 1e-8)
    # 3. time stepping from the gravity state
    u = stat.steps[-1].u.copy()
    u_base = u.copy()                         # fixed dofs keep their (zero) values; free dofs are overwritten
    state = stat.state
    nT = len(rec.a)
    uf = u[free].copy()
    vf = np.zeros(len(free))
    fi0, _, _, _, _ = P.respond(u, state, want_K=False)
    af = -r[free] * rec.a[0]
    Cff = a0 * Mff + a1 * K0ff
    t_mon, v_mon, a_mon, Vb = np.zeros(nT), np.zeros(nT), np.zeros(nT), np.zeros(nT)
    a_mon[0] = af[mf] + rec.a[0]
    fx_fixed = fixed[fixed % 2 == o.direction]
    Vb[0] = float(fi0[fx_fixed].sum())
    E_in, E_d, E_s, E_k = np.zeros(nT), np.zeros(nT), np.zeros(nT), np.zeros(nT)
    res = DynResult(t=rec.t, u_mon=t_mon, v_mon=v_mon, a_mon_abs=a_mon, base_shear=Vb, periods=[float(2 * math.pi / x) for x in w],
                    rayleigh=(a0, a1), monitor_dof=mon, model=m, mass_total=float((M @ r).sum()))
    every = max(1, (nT - 1) // max(o.snapshots, 1))
    res.frames.append((0, u.copy(), stat.steps[-1].gp_stress.copy(), {key: np.asarray(v).copy() for key, v in stat.steps[-1].extra.items()}))
    fint_prev = fi0[free].copy()
    c1, c2 = 1 / (bet * dt * dt), gam / (bet * dt)
    linear = P.kind == "elastic" and not P.o.geometric

    def nl_step(uf, vf, af, state, ag1, h):
        """One Newmark step of length h by Newton–Raphson (reinforced concrete: retried with secant matrices)."""
        c1h, c2h = 1 / (bet * h * h), gam / (bet * h)
        fext = f_grav[free] - Mr * ag1
        ref = max(float(np.linalg.norm(fext)), float(np.linalg.norm(Mr * ag1)), 1e-9)
        pv = (1 - gam / bet) * vf + h * (1 - gam / (2 * bet)) * af
        pa = -vf / (bet * h) - (1 / (2 * bet) - 1) * af
        for secant in ((False, True) if P.kind == "rc" else (False,)):
            P.secant = secant
            du = np.zeros(len(free))
            for it in range(1, o.max_iter + 1):
                u1, v1, a1v = uf + du, c2h * du + pv, c1h * du + pa
                ut = u_base.copy()
                ut[free] = u1
                fi, K, st, sig, ex = P.respond(ut, state)
                R = fext - Mff @ a1v - Cff @ v1 - fi[free]
                if float(np.linalg.norm(R)) <= tol * ref and it > 1:
                    P.secant = False
                    return u1, v1, a1v, ut, st, fi, sig, ex, it
                try:
                    du = du + splu((K[free][:, free] + c1h * Mff + c2h * Cff).tocsc()).solve(R)
                except RuntimeError:
                    break
                if not np.all(np.isfinite(du)):
                    break
        P.secant = False
        return None

    def advance(uf, vf, af, state, ag0, ag1, h, depth):
        """The step, or – if it does not converge – two half steps with the ground acceleration interpolated, up to Δt/8."""
        r1 = nl_step(uf, vf, af, state, ag1, h)
        if r1 is not None:
            return [(r1, ag0, ag1)]
        if depth >= 3:
            return None
        agm = 0.5 * (ag0 + ag1)
        first = advance(uf, vf, af, state, ag0, agm, h / 2, depth + 1)
        if first is None:
            return None
        q = first[-1][0]
        second = advance(q[0], q[1], q[2], q[4], agm, ag1, h / 2, depth + 1)
        return None if second is None else first + second

    if linear:                                   # constant effective stiffness: factorise once, one solve per step
        lu_lin = splu((K0ff + c1 * Mff + c2 * Cff).tocsc())
        u_ref = u.copy()
    for i in range(nT - 1):
        if linear:
            fext = f_grav[free] - Mr * rec.a[i + 1]
            pred_v = (1 - gam / bet) * vf + dt * (1 - gam / (2 * bet)) * af
            pred_a = -vf / (bet * dt) - (1 / (2 * bet) - 1) * af
            R = fext - Mff @ pred_a - Cff @ pred_v - K0ff @ uf
            du = lu_lin.solve(R)
            u1, v1, a1v = uf + du, c2 * du + pred_v, c1 * du + pred_a
            fi = np.zeros(n2)
            ut = u.copy()
            ut[free] = u1
            fi = K0 @ ut
            it = 1
            res.iterations.append(1)
            dU = du
            E_in[i + 1] = E_in[i] + float(np.dot(0.5 * (-Mr * rec.a[i] - Mr * rec.a[i + 1]), dU))
            E_d[i + 1] = E_d[i] + float(np.dot(Cff @ (0.5 * (vf + v1)), dU))
            E_s[i + 1] = E_s[i] + float(np.dot(0.5 * (fint_prev + fi[free]) - f_grav[free], dU))
            E_k[i + 1] = 0.5 * float(v1 @ (Mff @ v1))
            fint_prev = fi[free].copy()
            uf, vf, af, u = u1, v1, a1v, ut
            t_mon[i + 1] = u[mon] - u_ref[mon]
            v_mon[i + 1] = vf[mf]
            a_mon[i + 1] = af[mf] + rec.a[i + 1]
            Vb[i + 1] = float(fi[fx_fixed].sum())
            if (i + 1) % every == 0 or i + 1 == nT - 1:
                sig = None
                res.frames.append((i + 1, u.copy(), None, {}))
            continue
        subs = advance(uf, vf, af, state, rec.a[i], rec.a[i + 1], dt, 0)
        if subs is None:
            res.status = "failed"
            res.message = f"no convergence at t = {rec.t[i + 1]:.3f} s, even with the time step divided by 8" + _failure_hint(P, m, u, state)
            nT = i + 1
            break
        E_in[i + 1], E_d[i + 1], E_s[i + 1] = E_in[i], E_d[i], E_s[i]
        its = 0
        for (u1, v1, a1v, ut, st, fi, sig, ex, it), ag0, ag1 in subs:
            dU = u1 - uf
            E_in[i + 1] += float(np.dot(0.5 * (-Mr * ag0 - Mr * ag1), dU))
            E_d[i + 1] += float(np.dot(Cff @ (0.5 * (vf + v1)), dU))
            E_s[i + 1] += float(np.dot(0.5 * (fint_prev + fi[free]) - f_grav[free], dU))
            fint_prev = fi[free].copy()
            uf, vf, af, u, state = u1, v1, a1v, ut, st
            its += it
        res.iterations.append(its)
        res.substeps.append(len(subs))
        E_k[i + 1] = 0.5 * float(vf @ (Mff @ vf))
        t_mon[i + 1] = u[mon] - stat.steps[-1].u[mon]
        v_mon[i + 1] = vf[mf]
        a_mon[i + 1] = af[mf] + rec.a[i + 1]
        Vb[i + 1] = float(fi[fx_fixed].sum())
        if (i + 1) % every == 0 or i + 1 == nT - 1:
            res.frames.append((i + 1, u.copy(), sig.copy(), {key: np.asarray(v).copy() for key, v in ex.items()}))
    res.t, res.u_mon, res.v_mon, res.a_mon_abs, res.base_shear = (x[:nT] for x in (rec.t, t_mon, v_mon, a_mon, Vb))
    res.energy = {"E_in": E_in[:nT], "E_damp": E_d[:nT], "E_internal": E_s[:nT], "E_kin": E_k[:nT]}
    bal = E_in[:nT] - E_d[:nT] - E_s[:nT] - E_k[:nT]
    res.energy["error"] = float(np.abs(bal).max() / max(np.abs(E_in[:nT]).max(), 1e-30))
    return res


def _failure_hint(P: Problem, m: FEModel, u: np.ndarray, state: dict) -> str:
    """What the last converged state shows near the loss of convergence."""
    if P.kind != "rc":
        return ""
    _, _, _, _, ex = P.respond(u, state, want_K=False)
    xy = P.k.xy.reshape(-1, 2)
    e2 = ex["e2"].ravel()
    k = int(np.argmin(e2))
    out = []
    if e2[k] < -0.8 * m.material["eps_c0"]:
        out.append(f"concrete near its compressive strength at ({xy[k, 0]:.2f}, {xy[k, 1]:.2f}) m (ε₂ = {e2[k] * 1000:.2f} ‰): crushing")
    if np.any(ex["rupture"]):
        out.append("reinforcement strain above ε_ud")
    return (": " + "; ".join(out)) if out else ""


def to_dict_dynamic(r: DynResult, rec: Record | None = None, every: int = 1) -> dict:
    """JSON export of an FE time history: histories of the monitor and the base shear, energies and the summary."""
    sl = slice(None, None, max(1, int(every)))
    out = {"status": r.status, "message": r.message, "periods_s": [float(T) for T in r.periods], "rayleigh": [float(v) for v in r.rayleigh],
           "monitor_dof": int(r.monitor_dof), "mass_t": float(r.mass_total),
           "max_monitor_displacement_m": float(np.abs(r.u_mon).max()), "max_base_shear_kN": float(np.abs(r.base_shear).max()),
           "energy_balance_error": float(r.energy.get("error", 0.0)), "iterations": int(sum(r.iterations)),
           "t_s": r.t[sl].tolist(), "u_mon_m": r.u_mon[sl].tolist(), "base_shear_kN": r.base_shear[sl].tolist(),
           "energy_kNm": {k: np.asarray(v)[sl].tolist() for k, v in r.energy.items() if k != "error"}}
    if rec is not None:
        out["record"] = {"name": rec.name, "dt_s": rec.dt, "pga_ms2": float(rec.pga), "duration_s": float(rec.t[-1])}
    return out


__all__ = ["elastic_spectrum", "LCG", "response_spectrum", "sdof_history", "synthetic_accelerogram", "record_from_values", "parse_record", "Record",
           "BWParams", "bouc_wen_sdof", "bw_advance", "DynOptions", "DynResult", "fe_dynamic", "to_dict_dynamic"]
