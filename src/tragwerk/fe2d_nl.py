"""Nonlinear plane finite element analysis (geometric and material nonlinearity).

Follows the structure of the ETH Method of Finite Elements II course (Prof. E. Chatzi):

* **Geometric nonlinearity**, total Lagrangian: F = I + ∂u/∂X, Green–Lagrange
  strain E = ½(FᵀF − I), second Piola–Kirchhoff stress S = C E (St. Venant–
  Kirchhoff), internal forces f_int = ∫ B_Lᵀ S dV₀ and the tangent
  K_T = ∫ B_Lᵀ C B_L dV₀ + ∫ Gᵀ Ŝ G dV₀ (material and geometric parts).
* **Material nonlinearity**: von Mises (J2) plasticity with linear isotropic
  hardening, backward-Euler radial return and the consistent tangent (plane
  strain directly, plane stress by enforcing σ_zz = 0 iteratively); and a
  reinforced concrete membrane model (total strain, rotating smeared cracks:
  softened concrete in compression after Vecchio & Collins or with the EN
  1992 strut factor, tension stiffening, smeared bilinear reinforcement in x
  and y, crack check of the Modified Compression Field Theory).
* **Solution methods**: incremental load control with full Newton–Raphson,
  modified Newton–Raphson or secant iterations; displacement control (the
  monitored displacement is prescribed and the load factor follows); and the
  arc-length method (Crisfield, cylindrical) for limit points, snap-through
  and snap-back. Steps are bisected when an increment
  does not converge.

All quantities in kN, m; stresses reported in MPa. A port of this module runs
in the browser (web/fe2d_nl.js) and is checked against it in the test suite.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

from .fe2d import (FEError, FEModel, G3, Q4_NODES, element_reinforcement, gauss, gauss_point, load_vector, shape)

SQ23 = math.sqrt(2.0 / 3.0)


# ============================================================== precomputed element data
class Kernel:
    """Reference-configuration data of all Gauss points, vectorised over elements."""

    def __init__(self, m: FEModel):
        et = m.etype
        self.rule = gauss(et)
        ne, nn = m.elements.shape
        ng = len(self.rule)
        self.ne, self.nn, self.ng = ne, nn, ng
        self.dNdX = np.zeros((ne, ng, 2, nn))
        self.wdet = np.zeros((ne, ng))
        self.xy = np.zeros((ne, ng, 2))
        self.area = np.zeros(ne)
        for q, e in enumerate(m.elements):
            X = m.nodes[e]
            for g, (xi, eta, w) in enumerate(self.rule):
                gp = gauss_point(et, X, xi, eta, w)
                self.dNdX[q, g] = gp.dN_dx
                self.wdet[q, g] = w * gp.detJ * m.t
                self.xy[q, g] = (gp.x, gp.y)
                self.area[q] += w * gp.detJ
        self.N = np.array([shape(et, xi, eta)[0] for xi, eta, _ in self.rule])
        self.dofs = np.empty((ne, 2 * nn), dtype=int)
        self.dofs[:, 0::2] = 2 * m.elements
        self.dofs[:, 1::2] = 2 * m.elements + 1
        self.rows = np.repeat(self.dofs, 2 * nn, axis=1).ravel()
        self.cols = np.tile(self.dofs, (1, 2 * nn)).ravel()
        self.n2 = 2 * len(m.nodes)
        B = np.zeros((ne, ng, 3, 2 * nn))
        B[:, :, 0, 0::2] = self.dNdX[:, :, 0]
        B[:, :, 1, 1::2] = self.dNdX[:, :, 1]
        B[:, :, 2, 0::2] = self.dNdX[:, :, 1]
        B[:, :, 2, 1::2] = self.dNdX[:, :, 0]
        self.B = B
        self.extrap, self.mids = _nl_extrapolation(et)

    def assemble_K(self, Ke: np.ndarray):
        return coo_matrix((Ke.ravel(), (self.rows, self.cols)), shape=(self.n2, self.n2)).tocsr()

    def assemble_f(self, fe: np.ndarray) -> np.ndarray:
        f = np.zeros(self.n2)
        np.add.at(f, self.dofs.ravel(), fe.ravel())
        return f


def _nl_extrapolation(et: str):
    """Matrix taking integration-point values to the corner nodes, and the midside pairs."""
    if et == "T3":
        E = np.ones((3, 1))
    elif et == "T6":
        E = 2 * np.eye(3) - np.ones((3, 3)) / 3
    elif et in ("Q4", "Q8R"):
        # bilinear through the 2×2 points; the columns follow the order of the integration rule
        Nc = np.array([shape("Q4", a * math.sqrt(3), b * math.sqrt(3))[0] for a, b in Q4_NODES])
        order = [int(np.argmin(np.abs(Q4_NODES - np.sign([xi, eta])).sum(axis=1))) for xi, eta, _ in gauss(et)]
        E = Nc[:, order]
    else:                                        # Q8, 3×3 rule: bilinear through the four corner points
        E = np.zeros((4, 9))
        for i, (a, b) in enumerate(Q4_NODES):
            E[i, [0, 2, 8, 6]] = shape("Q4", a / G3, b / G3)[0]
    mids = {"T6": [(3, 0, 1), (4, 1, 2), (5, 2, 0)], "Q8": [(4, 0, 1), (5, 1, 2), (6, 2, 3), (7, 3, 0)],
            "Q8R": [(4, 0, 1), (5, 1, 2), (6, 2, 3), (7, 3, 0)]}.get(et, [])
    return E, mids


# ============================================================== materials
def elastic_C(E, nu, analysis):
    from .fe2d import constitutive
    return constitutive(E, nu, analysis)


def j2_return(eps4, epsp, alpha, E, nu, sy, H):
    """Radial return (Simo & Hughes) on tensor strains [xx, yy, zz, xy]; returns σ (tensor), D (4×4, engineering
    shear in the last row/column), ε_p, α, Δγ."""
    G, K = E / (2 * (1 + nu)), E / (3 * (1 - 2 * nu))
    ee = eps4 - epsp
    tr = ee[:, 0] + ee[:, 1] + ee[:, 2]
    s = 2 * G * ee
    s[:, :3] -= (2 * G * tr / 3)[:, None]
    ns = np.sqrt(s[:, 0] ** 2 + s[:, 1] ** 2 + s[:, 2] ** 2 + 2 * s[:, 3] ** 2)
    f = ns - SQ23 * (sy + H * alpha)
    plastic = f > 1e-12 * sy
    dg = np.where(plastic, f / (2 * G + 2 * H / 3), 0.0)
    safe = np.where(ns > 0, ns, 1.0)
    n = s / safe[:, None]
    s_new = s - (2 * G * dg)[:, None] * n
    sig = s_new.copy()
    sig[:, :3] += (K * tr)[:, None]
    epsp_new = epsp + dg[:, None] * n
    alpha_new = alpha + SQ23 * dg
    theta = 1 - 2 * G * dg / safe
    thetab = np.where(plastic, 1 / (1 + H / (3 * G)) - (1 - theta), 0.0)
    m = np.array([1.0, 1.0, 1.0, 0.0])
    Ieng = np.diag([1.0, 1.0, 1.0, 0.5])
    D = K * np.einsum("i,j->ij", m, m)[None] + (2 * G * theta)[:, None, None] * (Ieng - np.einsum("i,j->ij", m, m) / 3)[None]
    D = D - (2 * G * thetab)[:, None, None] * np.einsum("ni,nj->nij", n, n)
    return sig, D, epsp_new, alpha_new, dg


def j2_update(eps, st, mat, E, nu, analysis):
    """eps (N, 3) engineering; returns σ (N, 3), D (N, 3, 3), new state, extras."""
    N = len(eps)
    eps4 = np.zeros((N, 4))
    eps4[:, 0], eps4[:, 1], eps4[:, 3] = eps[:, 0], eps[:, 1], eps[:, 2] / 2
    a = [0, 1, 3]
    if analysis == "plane_strain":
        sig, D, ep, al, dg = j2_return(eps4, st["epsp"], st["alpha"], E, nu, mat["fy"], mat["H"])
        ezz = np.zeros(N)
    else:
        ezz = st["ezz"].copy()
        for _ in range(40):
            eps4[:, 2] = ezz
            sig, D, ep, al, dg = j2_return(eps4, st["epsp"], st["alpha"], E, nu, mat["fy"], mat["H"])
            res = sig[:, 2]
            if np.max(np.abs(res)) <= 1e-10 * mat["fy"]:
                break
            ezz = ezz - res / D[:, 2, 2]
        Dc = D[:, a][:, :, a] - np.einsum("ni,nj->nij", D[:, a, 2], D[:, 2, a]) / D[:, 2, 2][:, None, None]
        new = {"epsp": ep, "alpha": al, "ezz": ezz}
        return sig[:, a], Dc, new, {"alpha": al, "yield": dg > 0, "szz": sig[:, 2]}
    return sig[:, a], D[:, a][:, :, a], {"epsp": ep, "alpha": al, "ezz": ezz}, {"alpha": al, "yield": dg > 0, "szz": sig[:, 2]}


def _concrete_env(e, et, mat, Ec, plain, hcb):
    """Envelope of the uniaxial concrete law in a principal direction (tension positive), with the transverse strain et.
    Returns the stress and its slope dσ/dε (the transverse coupling of the compression softening is not included)."""
    fc, fct, e0 = mat["fc"], mat["fct"], mat["eps_c0"]
    ecr = fct / Ec
    sig = np.zeros_like(e)
    dsig = np.zeros_like(e)
    t = e >= 0
    lin = t & (e <= ecr)
    sig[lin] = Ec * e[lin]
    dsig[lin] = Ec
    cr = t & (e > ecr)
    if np.any(cr):
        # reinforced: tension stiffening fct/(1 + √(500 ε)) (Collins & Mitchell) from 2ε_cr on, reached linearly from the
        # cracking point (the original law drops vertically at ε_cr, which no iteration scheme follows well);
        # plain: linear softening over the crack band; tension "none": no tension once cracked (the no-tension field
        # of plasticity theory), reached over ε_cr … 2ε_cr
        ee = np.maximum(e, 2 * ecr)
        sq = np.sqrt(500 * ee)
        cm = fct / (1 + sq)
        dcm = -fct * 250.0 / (sq * (1 + sq) ** 2)
        c2 = fct / (1 + math.sqrt(1000 * ecr))
        ramp = e < 2 * ecr
        ts = np.where(ramp, fct + (c2 - fct) * (e - ecr) / ecr, cm)
        dts = np.where(ramp, (c2 - fct) / ecr, dcm)
        eu = np.maximum(2 * mat["Gf"] / (fct * hcb), 2 * ecr)
        if mat.get("tension") == "none":
            eu = np.full_like(e, 2 * ecr)
        so = np.maximum(fct * (eu - e) / (eu - ecr), 0.0)
        dso = np.where(e < eu, -fct / (eu - ecr), 0.0)
        use_so = plain | (mat.get("tension") == "none")
        sig[cr] = np.where(use_so[cr], so[cr], ts[cr])
        dsig[cr] = np.where(use_so[cr], dso[cr], dts[cr])
    c = ~t
    if np.any(c):
        trans = np.maximum(et, 0.0)
        if mat["softening"] == "mcft":
            beta = np.minimum(1.0 / (0.8 + 170 * trans), 1.0)          # Vecchio & Collins (1986)
        else:
            beta = np.where(trans > ecr, mat["nu_ec2"], 1.0)             # EN 1992-1-1 6.5.2(2) strut factor once cracked
        eta = -e / e0
        par = -beta * fc * (2 * eta - eta * eta)
        dpar = beta * fc * (2 - 2 * eta) / e0
        sig[c] = np.where(eta[c] <= 2, par[c], 0.0)
        dsig[c] = np.where(eta[c] <= 2, dpar[c], 0.0)
    return sig, dsig


def _concrete(e, et, mat, Ec, plain, hcb, emax, emin, smax):
    """Concrete with memory: below the largest tensile strain reached after cracking (emax, where the stress was smax)
    and above the largest compressive strain reached (emin) it unloads and reloads along the secant to the origin.
    Returns σ and dσ/dε."""
    sig, dsig = _concrete_env(e, et, mat, Ec, plain, hcb)
    ecr = mat["fct"] / Ec
    ut = (e >= 0) & (e < emax) & (emax > ecr)
    if np.any(ut):
        se = smax[ut] / emax[ut]
        sig[ut], dsig[ut] = se * e[ut], se
    uc = (e < 0) & (e > emin) & (emin < 0)
    if np.any(uc):
        se = _concrete_env(emin[uc], et[uc], mat, Ec, plain[uc], hcb[uc])[0] / emin[uc]
        sig[uc], dsig[uc] = se * e[uc], se
    return sig, dsig


def _steel(e, ep, q, mat):
    """Bilinear reinforcement as 1D plasticity with linear kinematic hardening (return mapping): stress, tangent
    modulus, plastic strain and back stress. Monotonic loading follows the bilinear law with slope E_sh."""
    Es, fy, Esh = mat["Es"], mat["fy"], mat["Esh"]
    Hk = Es * Esh / (Es - Esh) if Esh < Es else 1e30
    s_tr = Es * (e - ep)
    xi = s_tr - q
    f = np.abs(xi) - fy
    pl = f > 1e-12 * fy
    dg = np.where(pl, f / (Es + Hk), 0.0)
    n = np.sign(xi)
    s = s_tr - Es * dg * n
    Et = np.where(pl, Es * Hk / (Es + Hk), Es)
    return s, Et, ep + dg * n, q + Hk * dg * n


def rc_state0(N):
    return {"e1max": np.zeros(N), "e2min": np.zeros(N), "s1max": np.zeros(N), "spx": np.zeros(N), "spy": np.zeros(N), "qx": np.zeros(N), "qy": np.zeros(N)}


def rc_update(eps, st, mat, Ec, rho, hcb, plain, tangent=True):
    """Rotating smeared crack membrane model. eps (N, 3); st the committed history; rho (N, 2) steel ratios.
    Returns σ (N, 3), the iteration matrix (tangent or secant for the concrete, tangent for the steel), the trial
    history and extras."""
    ex, ey, g = eps[:, 0], eps[:, 1], eps[:, 2]
    cen = 0.5 * (ex + ey)
    rad = np.sqrt((0.5 * (ex - ey)) ** 2 + (0.5 * g) ** 2)
    e1, e2 = cen + rad, cen - rad
    th = 0.5 * np.arctan2(g, ex - ey)
    c, s = np.cos(th), np.sin(th)
    e1max, e2min, s1max = st["e1max"], st["e2min"], st["s1max"]
    s1, d1 = _concrete(e1, e2, mat, Ec, plain, hcb, e1max, e2min, s1max)
    s2, d2 = _concrete(e2, e1, mat, Ec, plain, hcb, e1max, e2min, s1max)
    ssx, Dsx, spx, qx = _steel(ex, st["spx"], st["qx"], mat)
    ssy, Dsy, spy, qy = _steel(ey, st["spy"], st["qy"], mat)
    ecr = mat["fct"] / Ec
    # crack check (MCFT): the concrete tension across a crack must be transmitted by the reinforcement
    reinf = ~plain
    cap = rho[:, 0] * (mat["fy"] - ssx) * c * c + rho[:, 1] * (mat["fy"] - ssy) * s * s
    capped = reinf & (e1 > ecr) & (s1 > np.maximum(cap, 0.0))
    s1 = np.where(capped, np.maximum(cap, 0.0), s1)
    d1 = np.where(capped, 0.0, d1)
    sx = s1 * c * c + s2 * s * s + rho[:, 0] * ssx
    sy = s1 * s * s + s2 * c * c + rho[:, 1] * ssy
    txy = (s1 - s2) * s * c
    if tangent:
        # tangent of the coaxial rotating crack model (Newton–Raphson): dσ_i/dε_i in the principal directions, softening
        # included, which is what makes the iterations converge while cracks spread; tiny slopes are lifted for regularity
        E1 = np.where(np.abs(d1) < 1e-4 * Ec, 1e-4 * Ec, d1)
        E2 = np.where(np.abs(d2) < 1e-4 * Ec, 1e-4 * Ec, d2)
    else:
        # secant matrix (Vecchio's formulation): always positive, robust where a tangent would make the system singular
        E0c = 2 * mat["fc"] / mat["eps_c0"]
        E1 = np.where(np.abs(e1) > 1e-14, s1 / np.where(np.abs(e1) > 1e-14, e1, 1.0), np.where(e1 >= 0, Ec, E0c))
        E2 = np.where(np.abs(e2) > 1e-14, s2 / np.where(np.abs(e2) > 1e-14, e2, 1.0), np.where(e2 >= 0, Ec, E0c))
        E1, E2 = np.maximum(E1, 1e-4 * Ec), np.maximum(E2, 1e-4 * Ec)
    dd = e1 - e2
    Gs = np.where(dd > 1e-12, (s1 - s2) / (2 * np.where(dd > 1e-12, dd, 1.0)), 0.25 * (E1 + E2))
    Gs = np.maximum(Gs, 1e-3 * Ec)
    T = np.zeros((len(eps), 3, 3))
    T[:, 0, 0], T[:, 0, 1], T[:, 0, 2] = c * c, s * s, c * s
    T[:, 1, 0], T[:, 1, 1], T[:, 1, 2] = s * s, c * c, -c * s
    T[:, 2, 0], T[:, 2, 1], T[:, 2, 2] = -2 * c * s, 2 * c * s, c * c - s * s
    Dp = np.zeros((len(eps), 3, 3))
    Dp[:, 0, 0], Dp[:, 1, 1], Dp[:, 2, 2] = E1, E2, Gs
    D = np.einsum("nki,nkl,nlj->nij", T, Dp, T)
    Es = mat["Es"]
    # reinforcement: tangent modulus (it is what makes the iterations converge on a yield plateau), floored for regularity
    D[:, 0, 0] += rho[:, 0] * np.maximum(Dsx, 1e-3 * Es)
    D[:, 1, 1] += rho[:, 1] * np.maximum(Dsy, 1e-3 * Es)
    new = {"e1max": np.maximum(e1max, e1), "e2min": np.minimum(e2min, e2), "s1max": np.where(e1 >= e1max, np.maximum(s1, 0.0), s1max),
           "spx": spx, "spy": spy, "qx": qx, "qy": qy}
    extra = {"e1": e1, "e2": e2, "theta": th, "sc1": s1, "sc2": s2, "ssx": ssx, "ssy": ssy,
             "cracked": np.maximum(e1max, e1) > ecr, "open": e1 > ecr,
             "yield_x": (np.abs(spx) > 0) & (rho[:, 0] > 0), "yield_y": (np.abs(spy) > 0) & (rho[:, 1] > 0),
             "crushed": np.minimum(e2min, e2) < -mat["eps_c0"],
             "rupture": (np.maximum(np.abs(ex) * (rho[:, 0] > 0), np.abs(ey) * (rho[:, 1] > 0)) > mat["eud"])}
    return np.column_stack([sx, sy, txy]), D, new, extra


# ============================================================== options and results
@dataclass
class NLOptions:
    geometric: bool = False
    combination: str = "uls"
    control: str = "load"                # load | displacement | arc
    iteration: str = "newton"            # newton | modified | secant (RC: a failed Newton increment is retried with secants)
    steps: int = 10
    lam_max: float = 1.0
    to_failure: bool = False
    tol: float | None = None             # relative force residual; default 1e-6, reinforced concrete 1e-3 (0.1 %)
    max_iter: int = 40
    accelerate: bool = True              # Anderson acceleration of the secant iterations (reinforced concrete)
    max_steps: int = 400
    max_bisections: int = 8
    monitor: tuple | None = None         # (node index, component 0|1)
    load_pattern: str = "all"            # all: λ × (combination); variable: permanent loads first, then λ × variable loads
    line_search: bool = True             # reinforced concrete: energy line search on Newton steps
    accept_tol: float | None = None      # reinforced concrete: accept (and flag) a stalled increment below this residual, default 1e-2
    keep_states: bool = True


@dataclass
class NLStep:
    lam: float
    u: np.ndarray
    iterations: int
    residuals: list
    monitor: float
    reaction: tuple
    gp_stress: np.ndarray                # (ne, ng, 3) MPa (Cauchy for geometric analyses)
    extra: dict
    matrix: str = "tangent"              # iteration matrix of the increment: tangent | secant
    converged: bool = True               # False: accepted with a residual between tol and accept_tol


@dataclass
class NLResult:
    model: FEModel
    options: NLOptions
    steps: list = field(default_factory=list)
    status: str = ""
    message: str = ""
    lam_peak: float = 0.0
    monitor_dof: int = 0
    kernel: Kernel | None = None
    rho: np.ndarray | None = None
    elapsed_s: float = 0.0
    state: dict | None = None                # committed material history at the last converged step
    unconverged: int = 0                     # increments accepted below accept_tol but above tol


# ============================================================== element kernels
class Problem:
    def __init__(self, m: FEModel, opts: NLOptions):
        self.m, self.o = m, opts
        self.k = Kernel(m)
        mat = m.material
        self.kind = mat.get("model", "elastic")
        if opts.geometric and self.kind != "elastic":
            raise FEError("geometric nonlinearity is available with the elastic (St. Venant–Kirchhoff) material")
        self.C = elastic_C(m.E, m.nu, m.analysis)
        ne, ng = self.k.ne, self.k.ng
        N = ne * ng
        Asx, Asy, dx, dy = element_reinforcement(m)
        self.As = np.column_stack([Asx, Asy])
        self.dia = np.column_stack([dx, dy])
        rho_e = self.As * 1e-6 / m.t
        self.rho = np.repeat(rho_e, ng, axis=0)
        self.plain = np.repeat((Asx + Asy) <= 0, ng)
        self.hcb = np.repeat(np.sqrt(self.k.area), ng)
        self.secant = False                       # reinforced concrete: secant instead of tangent matrices
        self.state0 = rc_state0(N) if self.kind == "rc" else {"epsp": np.zeros((N, 4)), "alpha": np.zeros(N), "ezz": np.zeros(N)}

    def respond(self, u: np.ndarray, state: dict, want_K: bool = True):
        """Internal force vector, tangent (or secant) matrix, trial state and Gauss-point output at u."""
        k, m = self.k, self.m
        ue = u[k.dofs]                                    # (ne, 2nn)
        ne, ng, nn = k.ne, k.ng, k.nn
        extra = {}
        if self.o.geometric:
            U = ue.reshape(ne, nn, 2)
            Hm = np.einsum("eai,egja->egij", U, k.dNdX)   # ∂u_i/∂X_j
            F = Hm + np.eye(2)[None, None]
            Egl = 0.5 * (np.einsum("egki,egkj->egij", F, F) - np.eye(2)[None, None])
            Ev = np.stack([Egl[..., 0, 0], Egl[..., 1, 1], 2 * Egl[..., 0, 1]], axis=-1)
            S = np.einsum("ij,egj->egi", self.C, Ev)
            Nx, Ny = k.dNdX[:, :, 0], k.dNdX[:, :, 1]
            BL = np.zeros((ne, ng, 3, 2 * nn))
            F11, F12, F21, F22 = F[..., 0, 0, None], F[..., 0, 1, None], F[..., 1, 0, None], F[..., 1, 1, None]
            BL[:, :, 0, 0::2], BL[:, :, 0, 1::2] = F11 * Nx, F21 * Nx
            BL[:, :, 1, 0::2], BL[:, :, 1, 1::2] = F12 * Ny, F22 * Ny
            BL[:, :, 2, 0::2], BL[:, :, 2, 1::2] = F11 * Ny + F12 * Nx, F21 * Ny + F22 * Nx
            fe = np.einsum("egij,egi,eg->ej", BL, S, k.wdet)
            Ke = None
            if want_K:
                Ke = np.einsum("egki,kl,eglj,eg->eij", BL, self.C, BL, k.wdet)
                gab = (np.einsum("ega,egb->egab", Nx, S[..., 0, None] * Nx + S[..., 2, None] * Ny)
                       + np.einsum("ega,egb->egab", Ny, S[..., 2, None] * Nx + S[..., 1, None] * Ny))
                Kg = np.einsum("egab,eg->eab", gab, k.wdet)
                Ke[:, 0::2, 0::2] += Kg
                Ke[:, 1::2, 1::2] += Kg
            detF = F[..., 0, 0] * F[..., 1, 1] - F[..., 0, 1] * F[..., 1, 0]
            Sm = np.zeros((ne, ng, 2, 2))
            Sm[..., 0, 0], Sm[..., 1, 1], Sm[..., 0, 1], Sm[..., 1, 0] = S[..., 0], S[..., 1], S[..., 2], S[..., 2]
            sig = np.einsum("egik,egkl,egjl->egij", F, Sm, F) / detF[..., None, None]
            cauchy = np.stack([sig[..., 0, 0], sig[..., 1, 1], sig[..., 0, 1]], axis=-1)
            return k.assemble_f(fe), (k.assemble_K(Ke) if want_K else None), state, cauchy / 1000.0, {"detF": detF}
        eps = np.einsum("egij,ej->egi", k.B, ue).reshape(ne * ng, 3)
        new_state = state
        if self.kind == "elastic":
            sig = eps @ self.C.T
            D = None
        elif self.kind == "j2":
            sig, D, new_state, extra = j2_update(eps, state, m.material, m.E, m.nu, m.analysis)
        else:
            sig, D, new_state, extra = rc_update(eps, state, m.material, m.E, self.rho, self.hcb, self.plain, tangent=not self.secant)
        sig3 = sig.reshape(ne, ng, 3)
        fe = np.einsum("egij,egi,eg->ej", k.B, sig3, k.wdet)
        K = None
        if want_K:
            if D is None:
                Ke = np.einsum("egki,kl,eglj,eg->eij", k.B, self.C, k.B, k.wdet)
            else:
                Ke = np.einsum("egki,egkl,eglj,eg->eij", k.B, D.reshape(ne, ng, 3, 3), k.B, k.wdet)
            K = k.assemble_K(Ke)
        return k.assemble_f(fe), K, new_state, sig3 / 1000.0, {key: v.reshape(ne, ng) for key, v in extra.items()}


# ============================================================== solver
def _norm(v):
    return float(np.sqrt(np.dot(v, v)))


def _small_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Gaussian elimination with partial pivoting for the few unknowns of the Anderson mixing."""
    A, b = A.astype(float).copy(), b.astype(float).copy()
    n = len(b)
    for k in range(n):
        p = k + int(np.argmax(np.abs(A[k:, k])))
        if p != k:
            A[[k, p]], b[[k, p]] = A[[p, k]], b[[p, k]]
        if abs(A[k, k]) < 1e-300:
            return np.zeros(n)
        for i in range(k + 1, n):
            f = A[i, k] / A[k, k]
            A[i, k:] -= f * A[k, k:]
            b[i] -= f * b[k]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - A[i, i + 1:] @ x[i + 1:]) / A[i, i]
    return x


class _Anderson:
    """Anderson acceleration (Walker & Ni 2011) of the fixed-point iteration x ← x + δx of the secant method: the next
    iterate combines the last few so that the linearised residual is least. Secant iterations through spreading
    cracks converge slowly but steadily, which is the case Anderson mixing is made for."""

    def __init__(self, depth: int = 5):
        self.depth = depth
        self.G: list = []
        self.F: list = []

    def reset(self):
        self.G, self.F = [], []

    def step(self, x: np.ndarray, f: np.ndarray) -> np.ndarray:
        g = x + f
        self.G.append(g.copy())
        self.F.append(f.copy())
        if len(self.F) > self.depth + 1:
            self.G.pop(0)
            self.F.pop(0)
        if len(self.F) < 2:
            return g
        dF = np.column_stack([self.F[i + 1] - self.F[i] for i in range(len(self.F) - 1)])
        dG = np.column_stack([self.G[i + 1] - self.G[i] for i in range(len(self.G) - 1)])
        M = dF.T @ dF                              # small normal equations, regularised (same algorithm in the browser)
        M += (1e-10 * max(np.trace(M) / len(M), 1e-300)) * np.eye(len(M))
        gam = _small_solve(M, dF.T @ f)
        out = g - dG @ gam
        if not np.all(np.isfinite(out)):
            self.reset()
            return g
        return out


def run_nonlinear(m: FEModel, opts: NLOptions | None = None, _stage=None) -> NLResult:
    """Path-following analysis. λ multiplies the loads of the combination (load_pattern "all"), or – load_pattern
    "variable", a pushover – the permanent loads are applied first and λ then multiplies only the variable ones."""
    import time
    from dataclasses import replace
    t_start = time.time()
    o = opts or NLOptions()
    P = Problem(m, o)
    k = P.k
    n2 = k.n2
    fixed = np.array(sorted(m.fixed), dtype=int)
    ubar = np.array([m.fixed[d] for d in fixed], dtype=float)
    free = np.setdiff1d(np.arange(n2), fixed)
    u = np.zeros(n2)
    state = P.state0
    if _stage is not None:                         # (constant loads, loads × λ, start state, scale of prescribed displacements)
        f_const, f_ext, u, state, us = _stage
        u = u.copy()
        ubar = ubar * us
    elif o.load_pattern == "variable":
        from .fe2d import combination_factors
        aG, aQ = combination_factors(m, o.combination)
        f_var = aQ * m.forces_q
        f_all = load_vector(m, o.combination)
        if not np.any(f_var):
            raise FEError("a pushover (permanent loads first, then λ × variable loads) needs variable loads (case Q)")
        g = run_nonlinear(m, replace(o, load_pattern="all", control="load", to_failure=False, lam_max=1.0, steps=4, max_steps=200),
                          _stage=(np.zeros(n2), f_all - f_var, np.zeros(n2), P.state0, 0.0))
        if g.status != "complete":
            raise FEError(f"the permanent loads could not be applied: {g.message or g.status}")
        return run_nonlinear(m, o, _stage=(f_all - f_var, f_var, g.steps[-1].u, g.state, 1.0))
    else:
        f_const, f_ext = np.zeros(n2), load_vector(m, o.combination)
    ff = f_ext[free]
    fcf = f_const[free]
    res = NLResult(model=m, options=o, kernel=k, rho=P.rho)
    # monitor: the dof that moves most in a linear first step
    f0, K0, _, _, _ = P.respond(u, state)
    Kff, Kfc = K0[free][:, free].tocsc(), K0[free][:, fixed]
    try:
        lu = splu(Kff)
    except RuntimeError:
        raise FEError("the structure is a mechanism: add supports so that it cannot move as a rigid body") from None
    qdir = ff - Kfc @ ubar
    du = lu.solve(qdir)
    if o.monitor is None:
        # the dof that moves most (ties within 1e-6 go to the lowest dof number, so the choice is reproducible)
        a = np.abs(du)
        mon = int(free[int(np.flatnonzero(a >= (1 - 1e-6) * a.max())[0])]) if len(du) else 0
    else:
        mon = 2 * int(o.monitor[0]) + int(o.monitor[1])
    res.monitor_dof = mon
    lam = 0.0
    lam_target = 1e9 if o.to_failure else o.lam_max
    res.steps.append(_record(P, u, 0.0, 0, [], mon, fixed, state))
    dlam0 = o.lam_max / max(o.steps, 1)
    dlam = dlam0
    dl = None
    dmon = None
    mon_f = int(np.searchsorted(free, mon)) if mon in set(free.tolist()) else -1
    if o.control == "displacement" and mon_f < 0:
        raise FEError("displacement control needs a free monitor degree of freedom")
    du_prev = None
    bis = 0
    nsteps = 0
    status = "complete"

    base_secant = o.iteration == "secant"
    P.secant = base_secant
    target_it = (12 if base_secant else 6) if P.kind == "rc" else 5     # secant iterations converge linearly
    tol = o.tol if o.tol is not None else (1e-3 if P.kind == "rc" else 1e-6)
    accel = P.kind == "rc" and o.accelerate

    def retry_secant():
        """A failed Newton increment of a reinforced concrete model is tried once more with secant matrices."""
        if P.kind == "rc" and not P.secant:
            P.secant = True
            return True
        return False
    # reinforced concrete: energy line search on Newton steps, and increments whose residual stays below accept_tol
    # (default 1 %, as in the Dutch NLFEA guidelines RTD 1016) are accepted and flagged when the iterations stall
    line_search = P.kind == "rc" and o.line_search
    accept_tol = o.accept_tol if o.accept_tol is not None else (1e-2 if P.kind == "rc" else None)
    unconv = [0]
    aa = _Anderson()
    low = [0]

    def adapt(it):
        """Step-size factor from the iteration count of the last increment."""
        if it < target_it:
            return min(math.sqrt(target_it / max(it, 1)), 1.5)
        return 1.0 if it <= 2 * target_it else 0.7

    jump = [False]

    def dropped(lam_old, lam_new):
        """A converged increment that loses more than half the peak load at once is not a path that can be followed (a snap
        to a far branch, typical after strain localisation): reject it and try a smaller increment."""
        if not (o.to_failure and res.steps):
            return False
        peak = max(s_.lam for s_ in res.steps)
        jump[0] = peak > 0 and lam_old - lam_new > 0.5 * peak
        return jump[0]

    def finished(lam_now):
        """Stop criteria of path-following analyses: rupture of the reinforcement, or a load that stays below half the peak
        without recovering (a load that rises again by more than 1 % of the peak per increment restarts the count)."""
        last = res.steps[-1]
        if P.kind == "rc" and np.any(last.extra.get("rupture", False)):
            return "rupture"
        peak = max(s_.lam for s_ in res.steps)
        prev = res.steps[-2].lam if len(res.steps) > 1 else 0.0
        rising = lam_now > prev + 0.01 * abs(peak)
        low[0] = low[0] + 1 if (peak > 0 and lam_now < 0.5 * peak and not rising) else 0
        if low[0] >= 10 or (peak > 0 and lam_now <= 0):
            return "post-peak"
        return ""

    def solve_K(K, rhs):
        try:
            return splu(K[free][:, free].tocsc()).solve(rhs), K[free][:, fixed]
        except RuntimeError:
            return None, None

    while nsteps < o.max_steps:
        if lam >= lam_target - 1e-12:
            break
        nsteps += 1
        jump[0] = False
        if o.control == "load":
            lt = min(lam + dlam, lam_target)
            ut = u.copy()
            ut[fixed] = lt * ubar
            resid, conv, it = [], False, 0
            Kmod = None
            best = None
            aa.reset()
            for it in range(1, o.max_iter + 1):
                need = o.iteration != "modified" or Kmod is None
                fi, K, st, _, _ = P.respond(ut, state, want_K=need)
                if need:
                    Kmod = K
                r = f_const + lt * f_ext - fi
                rn = _norm(r[free])
                ref = max(_norm(fcf + lt * ff), _norm(fi[fixed]), 1e-9)
                resid.append(rn / ref)
                if accept_tol is not None and it > 1 and (best is None or rn / ref < best[0]):
                    best = (rn / ref, ut.copy(), st, it)
                if rn <= tol * ref and it > 1 or (rn <= 1e-14 * max(ref, 1.0)):
                    conv = True
                    break
                dx, _ = solve_K(Kmod, r[free])
                if dx is None or not np.all(np.isfinite(dx)):
                    break
                if accel and P.secant:
                    if len(resid) > 2 and resid[-1] > 10 * min(resid):
                        aa.reset()
                    ut[free] = aa.step(ut[free], dx)
                elif line_search and not P.secant:
                    g0 = float(np.dot(dx, r[free]))
                    ut1 = ut.copy()
                    ut1[free] += dx
                    fi1, _, _, _, _ = P.respond(ut1, state, want_K=False)
                    g1 = float(np.dot(dx, (f_const + lt * f_ext - fi1)[free]))
                    eta = max(0.1, g0 / (g0 - g1)) if g0 > 0 and g1 < -0.5 * g0 else 1.0
                    ut[free] += eta * dx
                else:
                    ut[free] += dx
            accepted = False
            if not conv and best is not None and best[0] <= accept_tol and P.secant:
                conv, accepted = True, True
                ut, st, it = best[1], best[2], best[3]
                unconv[0] += 1
            if conv:
                u, state, lam = ut, st, lt
                res.steps.append(_record(P, u, lam, it, resid, mon, fixed, state, converged=not accepted))
                P.secant = base_secant
                bis = 0
                if it <= target_it - 1:
                    dlam = min(dlam * 1.5, 4 * dlam0)
                if P.kind == "rc" and np.any(res.steps[-1].extra.get("rupture", False)):
                    status = "rupture"
                    break
                continue
            if retry_secant():
                nsteps -= 1
                continue
            P.secant = base_secant
            dlam /= 2
            bis += 1
            nsteps -= 1
            if bis > o.max_bisections:
                status = "limit" if lam > 0 else "failed"
                break
            continue
        if o.control == "displacement":
            # ---- displacement control of the monitor dof (Batoz & Dhatt): u_mon is prescribed, λ is unknown
            fi, K, _, _, _ = P.respond(u, state)
            dx_f, _ = solve_K(K, ff - K[free][:, fixed] @ ubar)
            if dx_f is None or abs(dx_f[mon_f]) < 1e-300:
                status = "failed"
                break
            if dmon is None:
                dmon = dlam0 * du[mon_f]
            dL = dmon / dx_f[mon_f]
            ut = u.copy()
            ut[free] += dL * dx_f
            ut[fixed] = (lam + dL) * ubar
            resid, conv, it = [], False, 0
            best = None
            aa.reset()
            for it in range(1, o.max_iter + 1):
                fi, K, st, _, _ = P.respond(ut, state)
                r = f_const + (lam + dL) * f_ext - fi
                rn = _norm(r[free])
                ref = max(_norm(fcf + (lam + dL) * ff), _norm(fi[fixed]), 1e-9)
                resid.append(rn / ref)
                if accept_tol is not None and it > 1 and (best is None or rn / ref < best[0]):
                    best = (rn / ref, ut.copy(), st, it, dL)
                if rn <= tol * ref and it > 1:
                    conv = True
                    break
                try:
                    lu_ = splu(K[free][:, free].tocsc())
                except RuntimeError:
                    break
                dr = lu_.solve(r[free])
                df = lu_.solve(ff - K[free][:, fixed] @ ubar)
                if abs(df[mon_f]) < 1e-300:
                    break
                dli = -dr[mon_f] / df[mon_f]
                if accel and P.secant:
                    if len(resid) > 2 and resid[-1] > 10 * min(resid):
                        aa.reset()
                    xn = aa.step(np.append(ut[free], dL), np.append(dr + dli * df, dli))
                    ut[free], dL = xn[:-1], float(xn[-1])
                else:
                    ut[free] += dr + dli * df
                    dL += dli
                ut[fixed] = (lam + dL) * ubar
                if not np.all(np.isfinite(ut)):
                    break
            accepted = False
            if not conv and best is not None and best[0] <= accept_tol and P.secant:
                conv, accepted = True, True
                ut, st, it, dL = best[1], best[2], best[3], best[4]
                unconv[0] += 1
            if conv and dropped(lam, lam + dL):
                conv = False
            if conv:
                u, state, lam = ut, st, lam + dL
                res.steps.append(_record(P, u, lam, it, resid, mon, fixed, state, converged=not accepted))
                P.secant = base_secant
                bis = 0
                dmon *= adapt(it)
                if not o.to_failure and lam >= o.lam_max - 1e-12:
                    break
                stop = finished(lam)
                if stop:
                    status = stop
                    break
                continue
            if not jump[0] and retry_secant():
                nsteps -= 1
                continue
            P.secant = base_secant
            dmon /= 2
            bis += 1
            nsteps -= 1
            if bis > (3 if jump[0] else o.max_bisections):
                status = "drop" if jump[0] else "limit" if lam > 0 else "failed"
                break
            continue
        # ---- arc-length (Crisfield, cylindrical)
        fi, K, _, _, _ = P.respond(u, state)
        dx_f, Kfc_ = solve_K(K, ff - K[free][:, fixed] @ ubar)
        if dx_f is None:
            status = "failed"
            break
        if dl is None:
            dl = abs(dlam0) * _norm(dx_f)
        sgn = 1.0 if du_prev is None else (1.0 if float(np.dot(du_prev, dx_f)) >= 0 else -1.0)
        dL = sgn * dl / max(_norm(dx_f), 1e-300)
        Du = dL * dx_f
        ut = u.copy()
        ut[free] += Du
        ut[fixed] = (lam + dL) * ubar
        resid, conv, it = [], False, 0
        best = None
        for it in range(1, o.max_iter + 1):
            fi, K, st, _, _ = P.respond(ut, state)
            r = f_const + (lam + dL) * f_ext - fi
            rn = _norm(r[free])
            ref = max(_norm(fcf + (lam + dL) * ff), _norm(fi[fixed]), 1e-9)
            resid.append(rn / ref)
            if accept_tol is not None and it > 1 and (best is None or rn / ref < best[0]):
                best = (rn / ref, ut.copy(), st, it, dL, Du.copy())
            if rn <= tol * ref and it > 1:
                conv = True
                break
            lu_ = None
            try:
                lu_ = splu(K[free][:, free].tocsc())
            except RuntimeError:
                break
            dr = lu_.solve(r[free])
            df = lu_.solve(ff - K[free][:, fixed] @ ubar)
            a = float(np.dot(df, df))
            b = 2 * float(np.dot(df, Du + dr))
            c = float(np.dot(Du + dr, Du + dr)) - dl * dl
            disc = b * b - 4 * a * c
            if a <= 0:
                break
            # no real root (common with secant matrices): take the closest approach to the arc instead
            roots = [(-b + math.sqrt(disc)) / (2 * a), (-b - math.sqrt(disc)) / (2 * a)] if disc >= 0 else [-b / (2 * a)]
            dlam_i = max(roots, key=lambda x: float(np.dot(Du + dr + x * df, Du)))
            Du = Du + dr + dlam_i * df
            dL += dlam_i
            ut[free] = u[free] + Du
            ut[fixed] = (lam + dL) * ubar
        accepted = False
        if not conv and best is not None and best[0] <= accept_tol and P.secant:
            conv, accepted = True, True
            ut, st, it, dL, Du = best[1], best[2], best[3], best[4], best[5]
            unconv[0] += 1
        if conv and dropped(lam, lam + dL):
            conv = False
        if conv:
            u, state, lam = ut, st, lam + dL
            du_prev = Du.copy()
            res.steps.append(_record(P, u, lam, it, resid, mon, fixed, state, converged=not accepted))
            P.secant = base_secant
            bis = 0
            dl *= adapt(it)
            if not o.to_failure and lam >= o.lam_max - 1e-12:
                break
            stop = finished(lam)
            if stop:
                status = stop
                break
            continue
        if not jump[0] and retry_secant():
            nsteps -= 1
            continue
        P.secant = base_secant
        dl /= 2
        bis += 1
        nsteps -= 1
        if bis > (3 if jump[0] else o.max_bisections):
            status = "drop" if jump[0] else "limit" if lam > 0 else "failed"
            break
    if status == "complete" and nsteps >= o.max_steps and (o.to_failure or lam < o.lam_max - 1e-12):
        status = "max-steps"
    res.status = status
    res.state = state
    res.unconverged = unconv[0]
    res.lam_peak = max(s.lam for s in res.steps)
    if status == "limit":
        res.message = (f"No equilibrium beyond λ = {lam:.4g}: a limit point (load-carrying capacity) was reached. "
                       "Displacement control or the arc-length method follow the path beyond it." if o.control == "load" else
                       f"The analysis stopped at λ = {lam:.4g}: no converged increment, even after {o.max_bisections} step reductions.")
    elif status == "drop":
        res.message = (f"Beyond the peak λ_u = {res.lam_peak:.4g} equilibrium was found only after losing more than half of it in "
                       "one increment, also with increments down to 1/8: a sudden, brittle loss of capacity. λ_u is the capacity.")
    elif status == "rupture":
        res.message = f"Reinforcement ruptured (steel strain above ε_ud) at λ = {lam:.4g}."
    elif status == "post-peak":
        res.message = f"The load fell below half the peak (λ_peak = {max(s_.lam for s_ in res.steps):.4g}): the structure has failed."
    elif status == "failed":
        res.message = "The first increment did not converge; reduce the step size or check the supports."
    res.elapsed_s = time.time() - t_start
    return res


def _record(P: Problem, u, lam, it, resid, mon, fixed, state, converged=True) -> NLStep:
    fi, _, _, sig, extra = P.respond(u, state, want_K=False)
    R = fi.copy()
    Rx = float(R[fixed][fixed % 2 == 0].sum()) if len(fixed) else 0.0
    Ry = float(R[fixed][fixed % 2 == 1].sum()) if len(fixed) else 0.0
    ex = {}
    if P.kind == "j2":
        ex = {"alpha": state["alpha"].reshape(P.k.ne, P.k.ng).copy(), "yield": extra.get("yield")}
    elif P.kind == "rc":
        ex = {key: np.asarray(v).copy() for key, v in extra.items()}
    return NLStep(lam=float(lam), u=u.copy(), iterations=int(it), residuals=list(resid), monitor=float(u[mon]), reaction=(Rx, Ry),
                  gp_stress=sig.copy(), extra=ex, matrix="secant" if P.secant else "tangent", converged=bool(converged))


# ============================================================== post-processing
def nodal_stress(res: NLResult, step: NLStep) -> np.ndarray:
    """Nodal stresses (MPa), extrapolated from the integration points and averaged."""
    m, k = res.model, res.kernel
    E, mids = k.extrap, k.mids
    ne, nn = m.elements.shape
    corners = np.einsum("cg,egs->ecs", E, step.gp_stress)          # (ne, ncorner, 3)
    en = np.zeros((ne, nn, 3))
    en[:, :corners.shape[1]] = corners
    for mi, a, b in mids:
        en[:, mi] = 0.5 * (corners[:, a] + corners[:, b])
    out = np.zeros((len(m.nodes), 3))
    cnt = np.zeros(len(m.nodes))
    np.add.at(out, m.elements.ravel(), en.reshape(-1, 3))
    np.add.at(cnt, m.elements.ravel(), 1)
    return out / np.maximum(cnt, 1)[:, None]


def crack_widths(res: NLResult, step: NLStep) -> np.ndarray:
    """Estimated crack width (mm) at every integration point: w = s_r,max,θ (ε1 − σc1/Ec) (EN 1992-1-1 7.3.4)."""
    m, k = res.model, res.kernel
    if m.material.get("model") != "rc" or "e1" not in step.extra:
        return np.zeros((k.ne, k.ng))
    P_As = element_reinforcement(m)
    t_mm = m.t * 1000.0
    c = m.reinforcement["cover_mm"]
    out = np.zeros((k.ne, k.ng))
    srs = []
    for j in range(2):
        As, dia = P_As[j], P_As[2 + j]
        hef = np.minimum(2.5 * (c + dia / 2), t_mm / 2)
        rho_eff = np.where(As > 0, (As / 2) / (hef * 1000.0), 0.0)
        sr = np.where(As > 0, 3.4 * c + 0.425 * 0.8 * 1.0 * dia / np.where(rho_eff > 0, rho_eff, 1.0), 1.3 * t_mm)
        srs.append(np.repeat(sr[:, None], k.ng, axis=1))
    th = step.extra["theta"]
    srt = 1.0 / (np.abs(np.cos(th)) / srs[0] + np.abs(np.sin(th)) / srs[1])
    open_ = np.maximum(step.extra["e1"] - step.extra["sc1"] / m.E, 0.0)
    out = np.where(step.extra["cracked"], srt * open_, 0.0)
    return out


def summary(res: NLResult) -> dict:
    last = res.steps[-1]
    out = {"status": res.status, "message": res.message, "steps": len(res.steps) - 1, "lam_final": last.lam, "lam_peak": res.lam_peak,
           "monitor_dof": res.monitor_dof, "curve": [[s.lam, s.monitor] for s in res.steps],
           "iterations": [s.iterations for s in res.steps[1:]], "elapsed_s": res.elapsed_s, "unconverged": res.unconverged}
    m = res.model
    if m.material.get("model") == "rc":
        ex = last.extra
        out["rc"] = {"cracked_points": int(np.sum(ex["cracked"])), "yielded_points": int(np.sum(ex["yield_x"] | ex["yield_y"])),
                     "crushed_points": int(np.sum(ex["crushed"])), "rupture": bool(np.any(ex["rupture"])),
                     "max_crack_width_mm": float(crack_widths(res, last).max())}
    if m.material.get("model") == "j2":
        out["j2"] = {"max_plastic_strain": float(last.extra["alpha"].max()), "yielded_points": int(np.sum(last.extra["alpha"] > 0))}
    return out


__all__ = ["NLOptions", "NLResult", "run_nonlinear", "nodal_stress", "crack_widths", "summary", "j2_return", "rc_update", "Kernel"]
