"""Two-dimensional continuum finite elements: plane stress and plane strain.

Isoparametric elements in the notation of the ETH Zurich Method of Finite
Elements I course (Prof. E. Chatzi): for an element with nodes x_i and shape
functions N_i(ξ, η)

    x(ξ, η) = Σ N_i(ξ, η) x_i                         (isoparametric map)
    J = ∂(x, y)/∂(ξ, η),   ∂N/∂x = J⁻¹ ∂N/∂ξ           (Jacobian)
    ε = [ε_xx, ε_yy, γ_xy]ᵀ = B u_e                    (strain-displacement)
    σ = C ε                                            (plane stress or plane strain)
    k_e = ∫∫ Bᵀ C B t det J dξ dη ≈ Σ_g w_g Bᵀ C B t det J
    m_e = ∫∫ ρ t Nᵀ N det J dξ dη,   f_e = ∫ Nᵀ t̄ ds + ∫∫ Nᵀ b t dA

Elements: T3 (constant strain triangle), T6 (linear strain triangle), Q4
(bilinear quadrilateral) and Q8 (serendipity quadrilateral), integrated with
Gauss quadrature (T3 1 point, T6 3 points, Q4 2×2, Q8 3×3). Stresses are
recovered at the Gauss points and extrapolated to the nodes (2×2 points for Q4
and Q8, 3 points for T6), then averaged over the elements meeting at a node.

Units: lengths m, forces kN, E in MPa on input; stresses are reported in MPa
and displacements in m. A port of the same algorithms runs in the browser
(web/fe2d.js) and is checked against this module in the test suite.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

SCHEMA_FE = "tragwerk-fe/0.1"
ELEMENT_TYPES = ("T3", "T6", "Q4", "Q8")
NNODES = {"T3": 3, "T6": 6, "Q4": 4, "Q8": 8}
EDGES = {"T3": [(0, 1), (1, 2), (2, 0)], "T6": [(0, 3, 1), (1, 4, 2), (2, 5, 0)],
         "Q4": [(0, 1), (1, 2), (2, 3), (3, 0)], "Q8": [(0, 4, 1), (1, 5, 2), (2, 6, 3), (3, 7, 0)]}
Q4_NODES = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float)
Q8_NODES = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]], dtype=float)
G2 = 1 / math.sqrt(3)
G3 = math.sqrt(0.6)


class FEError(ValueError):
    pass


# ============================================================== shape functions
def shape(etype: str, xi: float, eta: float) -> tuple[np.ndarray, np.ndarray]:
    """N (n,) and dN/d(ξ, η) (2, n) at a point of the parent element."""
    if etype == "T3":
        return np.array([1 - xi - eta, xi, eta]), np.array([[-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]])
    if etype == "T6":
        L1, L2, L3 = 1 - xi - eta, xi, eta
        N = np.array([L1 * (2 * L1 - 1), L2 * (2 * L2 - 1), L3 * (2 * L3 - 1), 4 * L1 * L2, 4 * L2 * L3, 4 * L3 * L1])
        dL = np.array([[-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]])          # dL_k/dξ, dL_k/dη
        dN = np.zeros((2, 6))
        for a in range(2):
            d1, d2, d3 = dL[a]
            dN[a] = [(4 * L1 - 1) * d1, (4 * L2 - 1) * d2, (4 * L3 - 1) * d3,
                     4 * (d1 * L2 + L1 * d2), 4 * (d2 * L3 + L2 * d3), 4 * (d3 * L1 + L3 * d1)]
        return N, dN
    if etype == "Q4":
        xs, es = Q4_NODES[:, 0], Q4_NODES[:, 1]
        N = 0.25 * (1 + xi * xs) * (1 + eta * es)
        dN = np.array([0.25 * xs * (1 + eta * es), 0.25 * es * (1 + xi * xs)])
        return N, dN
    if etype == "Q8":
        N, dN = np.zeros(8), np.zeros((2, 8))
        for i, (a, b) in enumerate(Q8_NODES):
            if i < 4:
                N[i] = 0.25 * (1 + xi * a) * (1 + eta * b) * (xi * a + eta * b - 1)
                dN[0, i] = 0.25 * a * (1 + eta * b) * (2 * xi * a + eta * b)
                dN[1, i] = 0.25 * b * (1 + xi * a) * (xi * a + 2 * eta * b)
            elif a == 0:
                N[i] = 0.5 * (1 - xi * xi) * (1 + eta * b)
                dN[0, i] = -xi * (1 + eta * b)
                dN[1, i] = 0.5 * b * (1 - xi * xi)
            else:
                N[i] = 0.5 * (1 + xi * a) * (1 - eta * eta)
                dN[0, i] = 0.5 * a * (1 - eta * eta)
                dN[1, i] = -eta * (1 + xi * a)
        return N, dN
    raise FEError(f"unknown element type {etype!r}")


def gauss(etype: str, purpose: str = "stiffness") -> list[tuple[float, float, float]]:
    """Gauss points (ξ, η, w) of the parent element."""
    if etype in ("Q4", "Q8"):
        if etype == "Q4" and purpose == "stiffness":
            pts, w = [-G2, G2], [1.0, 1.0]
        else:
            pts, w = [-G3, 0.0, G3], [5 / 9, 8 / 9, 5 / 9]
        return [(pts[i], pts[j], w[i] * w[j]) for j in range(len(pts)) for i in range(len(pts))]
    if purpose == "mass":
        a, b = 0.445948490915965, 0.091576213509771
        wa, wb = 0.223381589678011 / 2, 0.109951743655322 / 2
        return [(a, a, wa), (1 - 2 * a, a, wa), (a, 1 - 2 * a, wa), (b, b, wb), (1 - 2 * b, b, wb), (b, 1 - 2 * b, wb)]
    if etype == "T3":
        return [(1 / 3, 1 / 3, 0.5)]
    return [(1 / 6, 1 / 6, 1 / 6), (2 / 3, 1 / 6, 1 / 6), (1 / 6, 2 / 3, 1 / 6)]


def constitutive(E: float, nu: float, analysis: str) -> np.ndarray:
    """C, so that σ = C ε with ε = [ε_xx, ε_yy, γ_xy]."""
    if analysis == "plane_stress":
        return E / (1 - nu * nu) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])
    if analysis == "plane_strain":
        return E / ((1 + nu) * (1 - 2 * nu)) * np.array([[1 - nu, nu, 0], [nu, 1 - nu, 0], [0, 0, (1 - 2 * nu) / 2]])
    raise FEError(f"analysis must be plane_stress or plane_strain, not {analysis!r}")


@dataclass
class GaussPoint:
    xi: float
    eta: float
    w: float
    N: np.ndarray
    dN_dxi: np.ndarray
    J: np.ndarray
    detJ: float
    dN_dx: np.ndarray
    B: np.ndarray
    x: float
    y: float


def gauss_point(etype: str, xy: np.ndarray, xi: float, eta: float, w: float = 0.0) -> GaussPoint:
    N, dN = shape(etype, xi, eta)
    J = dN @ xy                                   # [[∂x/∂ξ, ∂y/∂ξ], [∂x/∂η, ∂y/∂η]]
    det = float(J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0])
    if det <= 0:
        raise FEError(f"element is inverted or badly distorted: det J = {det:.3g} at (ξ, η) = ({xi:.3f}, {eta:.3f})")
    dNx = np.linalg.solve(J, dN)                  # ∂N/∂x, ∂N/∂y
    n = len(N)
    B = np.zeros((3, 2 * n))
    B[0, 0::2] = dNx[0]
    B[1, 1::2] = dNx[1]
    B[2, 0::2] = dNx[1]
    B[2, 1::2] = dNx[0]
    x, y = N @ xy
    return GaussPoint(xi, eta, w, N, dN, J, det, dNx, B, float(x), float(y))


def element_stiffness(etype: str, xy: np.ndarray, C: np.ndarray, t: float) -> tuple[np.ndarray, list[GaussPoint]]:
    n = NNODES[etype]
    k = np.zeros((2 * n, 2 * n))
    gps = []
    for xi, eta, w in gauss(etype):
        g = gauss_point(etype, xy, xi, eta, w)
        k += w * t * g.detJ * (g.B.T @ C @ g.B)
        gps.append(g)
    return k, gps


def element_mass(etype: str, xy: np.ndarray, rho: float, t: float) -> np.ndarray:
    """Consistent mass m_e = ∫ ρ t Nᵀ N dA (2 dof per node, interleaved)."""
    n = NNODES[etype]
    m = np.zeros((2 * n, 2 * n))
    for xi, eta, w in gauss(etype, "mass"):
        g = gauss_point(etype, xy, xi, eta, w)
        Nm = np.zeros((2, 2 * n))
        Nm[0, 0::2] = g.N
        Nm[1, 1::2] = g.N
        m += w * rho * t * g.detJ * (Nm.T @ Nm)
    return m


def body_force(etype: str, xy: np.ndarray, b: tuple[float, float], t: float) -> np.ndarray:
    n = NNODES[etype]
    f = np.zeros(2 * n)
    for xi, eta, w in gauss(etype, "mass"):
        g = gauss_point(etype, xy, xi, eta, w)
        f[0::2] += w * t * g.detJ * g.N * b[0]
        f[1::2] += w * t * g.detJ * g.N * b[1]
    return f


# extrapolation of Gauss-point stresses to the corner nodes
def _extrapolation(etype: str) -> tuple[list[tuple[float, float]], np.ndarray]:
    """Recovery points (ξ, η) and the matrix E with corner values = E @ point values."""
    if etype == "T3":
        return [(1 / 3, 1 / 3)], np.ones((3, 1))
    if etype == "T6":
        return [(1 / 6, 1 / 6), (2 / 3, 1 / 6), (1 / 6, 2 / 3)], 2 * np.eye(3) - np.ones((3, 3)) / 3
    pts = [(-G2, -G2), (G2, -G2), (G2, G2), (-G2, G2)]
    E = np.zeros((4, 4))
    for i, (a, b) in enumerate(Q4_NODES):
        N, _ = shape("Q4", a * math.sqrt(3), b * math.sqrt(3))   # corners in the coordinates of the Gauss points
        E[i] = N
    return pts, E


def _mid_pairs(etype: str) -> list[tuple[int, int, int]]:
    """(midside node, corner a, corner b) of quadratic elements."""
    if etype == "T6":
        return [(3, 0, 1), (4, 1, 2), (5, 2, 0)]
    if etype == "Q8":
        return [(4, 0, 1), (5, 1, 2), (6, 2, 3), (7, 3, 0)]
    return []


# ============================================================== model
@dataclass
class FEModel:
    name: str
    analysis: str
    etype: str
    E: float                       # kN/m²
    nu: float
    t: float                       # m
    gamma: float                   # kN/m³
    nodes: np.ndarray              # (n, 2)
    elements: np.ndarray           # (ne, nn)
    fixed: dict = field(default_factory=dict)      # dof -> prescribed value (m)
    forces: np.ndarray = None      # (2n,)
    self_weight: bool = False
    geometry: dict = field(default_factory=dict)
    edges: dict = field(default_factory=dict)      # name -> list of element edges (tuples of node indices)
    meta: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)

    @property
    def C(self) -> np.ndarray:
        return constitutive(self.E, self.nu, self.analysis)


def _lines(breaks: list[float], h: float) -> list[float]:
    out = [breaks[0]]
    for a, b in zip(breaks[:-1], breaks[1:]):
        n = max(1, int(math.floor((b - a) / h + 0.5)))         # half up, as in the browser engine
        out += [a + (b - a) * k / n for k in range(1, n + 1)]
    return out


def mesh_rectangle(Lx: float, Ly: float, h: float, etype: str, openings=()) -> tuple[np.ndarray, np.ndarray]:
    """Structured mesh of a rectangle [0, Lx] × [0, Ly] with rectangular openings."""
    bx, by = {0.0, Lx}, {0.0, Ly}
    for o in openings:
        bx |= {min(max(o["x0"], 0.0), Lx), min(max(o["x1"], 0.0), Lx)}
        by |= {min(max(o["y0"], 0.0), Ly), min(max(o["y1"], 0.0), Ly)}
    xs, ys = _lines(sorted(bx), h), _lines(sorted(by), h)
    nx, ny = len(xs) - 1, len(ys) - 1
    keep = []
    for i in range(nx):
        for j in range(ny):
            cx, cy = (xs[i] + xs[i + 1]) / 2, (ys[j] + ys[j + 1]) / 2
            if not any(o["x0"] < cx < o["x1"] and o["y0"] < cy < o["y1"] for o in openings):
                keep.append((i, j))
    if not keep:
        raise FEError("the openings remove the whole wall")
    # points on the doubled grid (corner nodes at even indices, midside nodes at odd ones)
    X = [xs[k // 2] if k % 2 == 0 else (xs[k // 2] + xs[k // 2 + 1]) / 2 for k in range(2 * nx + 1)]
    Y = [ys[k // 2] if k % 2 == 0 else (ys[k // 2] + ys[k // 2 + 1]) / 2 for k in range(2 * ny + 1)]
    return _assemble_grid(X, Y, lambda p, q: (X[p], Y[q]), keep, nx, ny, etype)


def _assemble_grid(X, Y, point, keep, nx, ny, etype):
    """Elements on a structured (doubled) index grid, nodes numbered for a narrow band."""
    quad = etype in ("T6", "Q8")
    conn = []
    for i, j in keep:
        a, b, c, d = (2 * i, 2 * j), (2 * i + 2, 2 * j), (2 * i + 2, 2 * j + 2), (2 * i, 2 * j + 2)
        mid = lambda p, q: ((p[0] + q[0]) // 2, (p[1] + q[1]) // 2)  # noqa: E731
        if etype == "Q4":
            conn.append([a, b, c, d])
        elif etype == "Q8":
            conn.append([a, b, c, d, mid(a, b), mid(b, c), mid(c, d), mid(d, a)])
        else:
            tris = [(a, b, c), (a, c, d)] if (i + j) % 2 == 0 else [(a, b, d), (b, c, d)]
            for p, q, r in tris:
                conn.append([p, q, r] if not quad else [p, q, r, mid(p, q), mid(q, r), mid(r, p)])
    used = sorted({k for e in conn for k in e}, key=(lambda k: (k[0], k[1])) if nx >= ny else (lambda k: (k[1], k[0])))
    index = {k: n for n, k in enumerate(used)}
    nodes = np.array([point(*k) for k in used], dtype=float)
    elements = np.array([[index[k] for k in e] for e in conn], dtype=int)
    return nodes, elements


def mesh_plate_hole(W: float, a: float, n_theta: int, n_r: int, etype: str, grading: float = 1.6):
    """Quarter of a square plate 2W × 2W with a central hole of radius a (mapped mesh)."""
    if not 0 < a < W:
        raise FEError("the hole radius must be smaller than the half width")
    nt = 2 * n_theta

    def point(p, q):                                   # p along θ (doubled grid), q radial
        th = (math.pi / 2) * p / (2 * nt)
        rs = lambda j: (j / n_r) ** grading                          # noqa: E731  graded towards the hole
        s = rs(q // 2) if q % 2 == 0 else 0.5 * (rs(q // 2) + rs(q // 2 + 1))   # midside nodes halfway
        hx, hy = (a * math.cos(th), a * math.sin(th)) if p < 2 * nt else (0.0, a)
        if th <= math.pi / 4 + 1e-12:
            ox, oy = W, W * math.tan(th)
        else:
            ox, oy = W / math.tan(th), W
        if p == 2 * nt:
            ox = 0.0
        return ((1 - s) * hx + s * ox, (1 - s) * hy + s * oy)

    keep = [(i, j) for i in range(nt) for j in range(n_r)]
    # index i runs along θ (counter-clockwise), j outwards: the cells are oriented counter-clockwise when
    # the radial direction is the first parent axis, so swap the roles to keep det J > 0.
    nodes, elements = _assemble_grid(None, None, lambda p, q: point(q, p), [(j, i) for i, j in keep], n_r, nt, etype)
    return nodes, elements


def _edge_sets(nodes: np.ndarray, elements: np.ndarray, etype: str, geometry: dict) -> dict:
    lo, hi = nodes.min(axis=0), nodes.max(axis=0)
    tol = 1e-9 * max(float(np.max(hi - lo)), 1.0)
    on = {"left": lambda p: abs(p[0] - lo[0]) < tol, "right": lambda p: abs(p[0] - hi[0]) < tol,
          "bottom": lambda p: abs(p[1] - lo[1]) < tol, "top": lambda p: abs(p[1] - hi[1]) < tol}
    if geometry.get("type") == "plate_hole":
        a = geometry["a"]
        on["hole"] = lambda p: abs(math.hypot(p[0], p[1]) - a) < 1e-7 * a
    out = {k: [] for k in on}
    for e in elements:
        for ed in EDGES[etype]:
            ids = [int(e[k]) for k in ed]
            for name, f in on.items():
                if all(f(nodes[i]) for i in ids):
                    out[name].append(tuple(ids))
    return out


def _in_range(nodes: np.ndarray, edges: list, name: str, spec: dict, what: str) -> list:
    """Element edges of a boundary, optionally limited to from <= s <= to (s = x on bottom/top, y on left/right)."""
    if "from" not in spec and "to" not in spec:
        return edges
    if name not in ("left", "right", "bottom", "top"):
        raise FEError(f"{what}: from/to are available on the straight edges left, right, bottom and top")
    c = 0 if name in ("bottom", "top") else 1
    lo = _num(spec.get("from", -1e300), f"{what}.from")
    hi = _num(spec.get("to", 1e300), f"{what}.to")
    tol = 1e-9 * max(float(np.ptp(nodes[:, c])), 1.0)
    return [ed for ed in edges if all(lo - tol <= nodes[i][c] <= hi + tol for i in ed)]


def _nearest(nodes: np.ndarray, at) -> int:
    return int(np.argmin(np.hypot(nodes[:, 0] - at[0], nodes[:, 1] - at[1])))


def _edge_load(nodes, edge: tuple, qfun) -> list[tuple[int, float, float]]:
    """Consistent nodal forces of a traction q(x, y) = (qx, qy) kN/m along one element edge."""
    pts = nodes[list(edge)]
    if len(edge) == 2:
        gp, gw = [-G3, 0.0, G3], [5 / 9, 8 / 9, 5 / 9]
        out = {i: [0.0, 0.0] for i in edge}
        for s, w in zip(gp, gw):
            N = np.array([(1 - s) / 2, (1 + s) / 2])
            dx = 0.5 * (pts[1] - pts[0])
            ds = math.hypot(*dx)
            x, y = N @ pts
            qx, qy = qfun(x, y)
            for k, i in enumerate(edge):
                out[i][0] += w * N[k] * qx * ds
                out[i][1] += w * N[k] * qy * ds
        return [(i, v[0], v[1]) for i, v in out.items()]
    gp = [-0.906179845938664, -0.538469310105683, 0.0, 0.538469310105683, 0.906179845938664]
    gw = [0.236926885056189, 0.478628670499366, 0.568888888888889, 0.478628670499366, 0.236926885056189]
    out = {i: [0.0, 0.0] for i in edge}
    for s, w in zip(gp, gw):
        N = np.array([s * (s - 1) / 2, 1 - s * s, s * (s + 1) / 2])      # end, mid, end
        dN = np.array([s - 0.5, -2 * s, s + 0.5])
        dx = dN @ pts
        ds = math.hypot(*dx)
        x, y = N @ pts
        qx, qy = qfun(x, y)
        for k, i in enumerate(edge):
            out[i][0] += w * N[k] * qx * ds
            out[i][1] += w * N[k] * qy * ds
    return [(i, v[0], v[1]) for i, v in out.items()]


def _num(v, what, lo=None, hi=None):
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise FEError(f"{what} must be a number") from None
    if not math.isfinite(x) or (lo is not None and x < lo) or (hi is not None and x > hi):
        raise FEError(f"{what} = {v} is out of range")
    return x


MAX_NODES = 40000


def parse_fe(data: dict) -> FEModel:
    if data.get("schema") != SCHEMA_FE:
        raise FEError(f"expected schema {SCHEMA_FE!r}")
    analysis = data.get("analysis", "plane_stress")
    etype = data.get("element", "Q8")
    if etype not in ELEMENT_TYPES:
        raise FEError(f"element must be one of {', '.join(ELEMENT_TYPES)}")
    mat = data.get("material", {})
    E = _num(mat.get("E_MPa", 30000), "material.E_MPa", 1e-6) * 1000.0
    nu = _num(mat.get("nu", 0.2), "material.nu", 0.0, 0.499)
    t = _num(mat.get("thickness_m", 0.25), "material.thickness_m", 1e-6)
    gamma = _num(mat.get("density_kNm3", 25.0), "material.density_kNm3", 0.0)
    constitutive(E, nu, analysis)
    g = data.get("geometry", {})
    kind = g.get("type", "rectangle")
    if kind == "rectangle":
        Lx, Ly = _num(g.get("Lx"), "geometry.Lx", 1e-6), _num(g.get("Ly"), "geometry.Ly", 1e-6)
        h = _num(g.get("h", min(Lx, Ly) / 8), "geometry.h", 1e-6)
        if (Lx / h) * (Ly / h) > MAX_NODES / (4 if etype in ("T6", "Q8") else 1):
            raise FEError("mesh too fine: increase the element size h")
        openings = []
        for k, o in enumerate(g.get("openings", [])):
            q = {key: _num(o.get(key), f"geometry.openings[{k}].{key}") for key in ("x0", "y0", "x1", "y1")}
            if q["x1"] <= q["x0"] or q["y1"] <= q["y0"]:
                raise FEError(f"opening {k + 1}: x1 > x0 and y1 > y0 required")
            openings.append(q)
        nodes, elements = mesh_rectangle(Lx, Ly, h, etype, openings)
    elif kind == "plate_hole":
        W, a = _num(g.get("W"), "geometry.W", 1e-6), _num(g.get("a"), "geometry.a", 1e-6)
        nt, nr = int(_num(g.get("n_theta", 8), "geometry.n_theta", 1, 200)), int(_num(g.get("n_r", 8), "geometry.n_r", 1, 200))
        nodes, elements = mesh_plate_hole(W, a, nt, nr, etype, _num(g.get("grading", 1.6), "geometry.grading", 1.0, 4.0))
    elif kind == "mesh":
        nodes = np.array([[_num(p[0], "node x"), _num(p[1], "node y")] for p in g.get("nodes", [])], dtype=float)
        elements = np.array(g.get("elements", []), dtype=int)
        if elements.ndim != 2 or elements.shape[1] != NNODES[etype] or elements.min() < 0 or elements.max() >= len(nodes):
            raise FEError(f"geometry.elements must list {NNODES[etype]} valid node indices per {etype} element")
    else:
        raise FEError(f"geometry.type must be rectangle, plate_hole or mesh, not {kind!r}")
    if len(nodes) > MAX_NODES:
        raise FEError(f"{len(nodes)} nodes: at most {MAX_NODES}")
    edges = _edge_sets(nodes, elements, etype, g if kind != "mesh" else {})
    n = len(nodes)
    fixed: dict[int, float] = {}
    for k, s in enumerate(data.get("supports", [])):
        if "edge" in s:
            if s["edge"] not in edges:
                raise FEError(f"supports[{k}]: unknown edge {s['edge']!r} (one of {', '.join(edges)})")
            ids = sorted({i for ed in _in_range(nodes, edges[s["edge"]], s["edge"], s, f"supports[{k}]") for i in ed})
            if not ids:
                raise FEError(f"supports[{k}]: no element edges on {s['edge']!r} in the given range")
        elif "at" in s:
            ids = [_nearest(nodes, s["at"])]
        else:
            raise FEError(f"supports[{k}] needs an edge or a point 'at'")
        for i in ids:
            for c, key in enumerate(("ux", "uy")):
                v = s.get(key, False)
                if v is True:
                    fixed[2 * i + c] = 0.0
                elif v is not False and v is not None:
                    fixed[2 * i + c] = _num(v, f"supports[{k}].{key}")
    if not fixed:
        raise FEError("the model has no supports")
    f = np.zeros(2 * n)
    for k, ld in enumerate(data.get("loads", [])):
        if "edge" in ld:
            if ld["edge"] not in edges or not edges[ld["edge"]]:
                raise FEError(f"loads[{k}]: unknown or empty edge {ld.get('edge')!r}")
            eds = _in_range(nodes, edges[ld["edge"]], ld["edge"], ld, f"loads[{k}]")
            if not eds:
                raise FEError(f"loads[{k}]: no element edges on {ld['edge']!r} in the given range")
            if ld.get("profile") == "parabolic":
                P = _num(ld.get("P", 0), f"loads[{k}].P")
                along = ld.get("dir", "y")
                pts = nodes[sorted({i for ed in eds for i in ed})]
                c = 1 if along == "y" else 0                     # traction component
                s_ax = 1 if (pts[:, 0].max() - pts[:, 0].min()) < (pts[:, 1].max() - pts[:, 1].min()) else 0
                lo_, hi_ = pts[:, s_ax].min(), pts[:, s_ax].max()
                D = hi_ - lo_

                def qfun(x, y, P=P, c=c, s_ax=s_ax, lo_=lo_, hi_=hi_, D=D):
                    s_ = (x, y)[s_ax]
                    q = 6 * P * (s_ - lo_) * (hi_ - s_) / D ** 3
                    return (q, 0.0) if c == 0 else (0.0, q)
            else:
                qx, qy = _num(ld.get("qx", 0), f"loads[{k}].qx"), _num(ld.get("qy", 0), f"loads[{k}].qy")

                def qfun(x, y, qx=qx, qy=qy):
                    return qx, qy
            for ed in eds:
                for i, fx, fy in _edge_load(nodes, ed, qfun):
                    f[2 * i] += fx
                    f[2 * i + 1] += fy
        elif "at" in ld:
            i = _nearest(nodes, ld["at"])
            f[2 * i] += _num(ld.get("fx", 0), f"loads[{k}].fx")
            f[2 * i + 1] += _num(ld.get("fy", 0), f"loads[{k}].fy")
        else:
            raise FEError(f"loads[{k}] needs an edge or a point 'at'")
    return FEModel(name=str(data.get("name", "")), analysis=analysis, etype=etype, E=E, nu=nu, t=t, gamma=gamma,
                   nodes=nodes, elements=elements, fixed=fixed, forces=f, self_weight=bool(data.get("self_weight", False)),
                   geometry=dict(g), edges=edges, raw=data)


# ============================================================== analysis
@dataclass
class FEResult:
    model: FEModel
    u: np.ndarray                  # (n, 2) m
    reactions: np.ndarray          # (n, 2) kN, zero at free dofs
    f: np.ndarray                  # applied nodal forces incl. self weight (2n,)
    gp: list                       # per element: list of (x, y, σx, σy, τxy) in MPa
    elem_nodal: np.ndarray         # (ne, nn, 3) extrapolated, unaveraged, MPa
    nodal: np.ndarray              # (n, 3) averaged, MPa
    energy: float                  # ½ uᵀ K u, kNm

    def field(self, name: str, averaged: bool = True) -> np.ndarray:
        s = self.nodal if averaged else self.elem_nodal
        sx, sy, txy = s[..., 0], s[..., 1], s[..., 2]
        if name == "sxx":
            return sx
        if name == "syy":
            return sy
        if name == "txy":
            return txy
        c, r = (sx + sy) / 2, np.sqrt(((sx - sy) / 2) ** 2 + txy ** 2)
        if name == "s1":
            return c + r
        if name == "s2":
            return c - r
        if name == "mises":
            sz = self.model.nu * (sx + sy) if self.model.analysis == "plane_strain" else 0 * sx
            return np.sqrt(0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2) + 3 * txy ** 2)
        raise FEError(f"unknown field {name!r}")


def assemble(m: FEModel):
    n, et = len(m.nodes), m.etype
    C = m.C
    rows, cols, vals = [], [], []
    f = m.forces.copy()
    for e in m.elements:
        xy = m.nodes[e]
        k, _ = element_stiffness(et, xy, C, m.t)
        dofs = np.empty(2 * len(e), dtype=int)
        dofs[0::2], dofs[1::2] = 2 * e, 2 * e + 1
        rows.append(np.repeat(dofs, len(dofs)))
        cols.append(np.tile(dofs, len(dofs)))
        vals.append(k.ravel())
        if m.self_weight and m.gamma > 0:
            f[dofs] += body_force(et, xy, (0.0, -m.gamma), m.t)
    K = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(2 * n, 2 * n)).tocsr()
    return K, f


def recover(m: FEModel, u_flat: np.ndarray):
    et, C = m.etype, m.C
    pts, Ex = _extrapolation(et)
    mids = _mid_pairs(et)
    ne, nn = m.elements.shape
    elem_nodal = np.zeros((ne, nn, 3))
    gp_out = []
    for q, e in enumerate(m.elements):
        xy = m.nodes[e]
        dofs = np.empty(2 * nn, dtype=int)
        dofs[0::2], dofs[1::2] = 2 * e, 2 * e + 1
        ue = u_flat[dofs]
        sp = []
        for xi, eta in pts:
            g = gauss_point(et, xy, xi, eta)
            sp.append(C @ (g.B @ ue))
        sp = np.array(sp)                      # (npts, 3) in kN/m²
        corners = Ex @ sp
        nc = corners.shape[0]
        elem_nodal[q, :nc] = corners
        for mi, a, b in mids:
            elem_nodal[q, mi] = 0.5 * (corners[a] + corners[b])
        gps = []
        for xi, eta, w in gauss(et):
            g = gauss_point(et, xy, xi, eta, w)
            s = C @ (g.B @ ue) / 1000.0
            gps.append((g.x, g.y, float(s[0]), float(s[1]), float(s[2])))
        gp_out.append(gps)
    elem_nodal /= 1000.0
    nodal = np.zeros((len(m.nodes), 3))
    cnt = np.zeros(len(m.nodes))
    for q, e in enumerate(m.elements):
        nodal[e] += elem_nodal[q]
        cnt[e] += 1
    nodal /= np.maximum(cnt, 1)[:, None]
    return elem_nodal, nodal, gp_out


def solve_fe(m: FEModel, prescribed: dict | None = None) -> FEResult:
    K, f = assemble(m)
    n2 = K.shape[0]
    fixed = dict(m.fixed)
    if prescribed:
        fixed.update(prescribed)
    fx = np.array(sorted(fixed), dtype=int)
    free = np.setdiff1d(np.arange(n2), fx)
    u = np.zeros(n2)
    u[fx] = [fixed[d] for d in fx]
    Kff = K[free][:, free].tocsc()
    rhs = f[free] - K[free][:, fx] @ u[fx]
    if Kff.shape[0]:
        diag = Kff.diagonal()
        if np.any(diag <= 0):
            raise FEError("the structure is a mechanism: some nodes are not connected or not supported")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sol = spsolve(Kff, rhs)
        if not np.all(np.isfinite(sol)):
            raise FEError("the structure is a mechanism: add supports so that it cannot move as a rigid body")
        u[free] = sol
    R = K @ u - f
    R[free] = 0.0
    scale = max(float(np.max(np.abs(f))), float(np.max(np.abs(R))), 1e-9)
    if np.max(np.abs((K @ u - f)[free])) > 1e-6 * scale:
        raise FEError("the structure is a mechanism: add supports so that it cannot move as a rigid body")
    en, nodal, gp = recover(m, u)
    return FEResult(model=m, u=u.reshape(-1, 2), reactions=R.reshape(-1, 2), f=f, gp=gp, elem_nodal=en, nodal=nodal,
                    energy=float(0.5 * u @ (K @ u)))


def element_details(m: FEModel, eid: int, u: np.ndarray | None = None) -> dict:
    """Everything about one element for teaching: N, ∂N/∂ξ, J, det J, B, C, k_e, m_e and its stresses."""
    et = m.etype
    e = m.elements[eid]
    xy = m.nodes[e]
    k, gps = element_stiffness(et, xy, m.C, m.t)
    out = {"type": et, "nodes": [int(i) for i in e], "xy": xy.tolist(), "C": m.C.tolist(), "t": m.t,
           "k": k.tolist(), "m": element_mass(et, xy, m.gamma / 9.81, m.t).tolist(),
           "gauss": [{"xi": g.xi, "eta": g.eta, "w": g.w, "x": g.x, "y": g.y, "N": g.N.tolist(), "dN_dxi": g.dN_dxi.tolist(),
                      "J": g.J.tolist(), "detJ": g.detJ, "B": g.B.tolist()} for g in gps]}
    if u is not None:
        dofs = np.empty(2 * len(e), dtype=int)
        dofs[0::2], dofs[1::2] = 2 * e, 2 * e + 1
        ue = np.asarray(u).reshape(-1)[dofs]
        out["u_e"] = ue.tolist()
        out["f_e"] = (k @ ue).tolist()
        for g, gd in zip(gps, out["gauss"]):
            eps = g.B @ ue
            gd["strain"] = eps.tolist()
            gd["stress_MPa"] = (m.C @ eps / 1000).tolist()
    return out


def to_dict_fe(r: FEResult) -> dict:
    m = r.model
    mises = r.field("mises")
    k = int(np.argmax(mises))
    umag = np.hypot(r.u[:, 0], r.u[:, 1])
    return {"schema": SCHEMA_FE, "name": m.name, "analysis": m.analysis, "element": m.etype,
            "nodes": m.nodes.tolist(), "elements": m.elements.tolist(), "u_m": r.u.tolist(), "reactions_kN": r.reactions.tolist(),
            "stress_MPa": {"sxx": r.nodal[:, 0].tolist(), "syy": r.nodal[:, 1].tolist(), "txy": r.nodal[:, 2].tolist()},
            "summary": {"nodes": len(m.nodes), "elements": len(m.elements), "dofs": 2 * len(m.nodes),
                        "max_displacement_m": float(umag.max()), "max_mises_MPa": float(mises[k]), "at": m.nodes[k].tolist(),
                        "reaction_sum_kN": r.reactions.sum(axis=0).tolist(), "load_sum_kN": [float(r.f[0::2].sum()), float(r.f[1::2].sum())],
                        "strain_energy_kNm": r.energy}}


def run_fe(data: dict) -> FEResult:
    return solve_fe(parse_fe(data))


# ============================================================== closed-form references
def timoshenko_cantilever(x, y, P, L, D, E, nu, t=1.0):
    """Exact plane stress solution (Timoshenko & Goodier) for a cantilever x ∈ [0, L], y ∈ [−D/2, D/2],
    fixed at x = 0 and loaded by a parabolic shear traction of resultant P (+y) at x = L.
    Returns (u_x, u_y, σ_xx, σ_yy, τ_xy)."""
    I = t * D ** 3 / 12
    ux = -P * y / (6 * E * I) * ((6 * L - 3 * x) * x + (2 + nu) * (y * y - D * D / 4))
    uy = P / (6 * E * I) * (3 * nu * y * y * (L - x) + (4 + 5 * nu) * D * D * x / 4 + (3 * L - x) * x * x)
    sxx = -P * (L - x) * y / I
    txy = P / (2 * I) * (D * D / 4 - y * y)
    return ux, uy, sxx, 0.0 * sxx, txy


def kirsch(r, theta, a, s0):
    """Stresses around a circular hole of radius a in an infinite plate under uniaxial tension s0 along x."""
    ar2, ar4 = (a / r) ** 2, (a / r) ** 4
    srr = s0 / 2 * (1 - ar2) + s0 / 2 * (1 - 4 * ar2 + 3 * ar4) * math.cos(2 * theta)
    stt = s0 / 2 * (1 + ar2) - s0 / 2 * (1 + 3 * ar4) * math.cos(2 * theta)
    srt = -s0 / 2 * (1 + 2 * ar2 - 3 * ar4) * math.sin(2 * theta)
    return srr, stt, srt


def heywood_kt(d_over_w: float) -> float:
    """Gross-section stress concentration of a central hole in a finite-width strip (Heywood)."""
    return (2 + (1 - d_over_w) ** 3) / (1 - d_over_w)
