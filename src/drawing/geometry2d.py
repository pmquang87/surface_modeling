"""Pure 2D algorithms of the drawing module (numpy + scipy KD-tree only).

Both geometry sources (OCCT HLR and the trimesh depth buffer) hand their
view-plane output to these helpers before the layout sees it: hatching of
section loops, chaining of plane-slice segments into loops, nesting of loops
into outers and holes, and the clean-up of HLR edge sets (dedupe, removal of
hidden segments lying under visible ones, merge of collinear touching
segments). No shapely, no matplotlib.

Conventions and the measured facts they rely on (probe reports
``REPORT-mesh.md`` and ``REPORT-hlr.md``):

* Loops are ``(n, 2)`` float arrays WITHOUT a repeated end point; every
  function that takes loops drops a repeated end point itself (contour
  tracers and ``Path3D.discrete`` deliver first == last).
* Point containment uses the half-open crossing rule: an edge counts when
  ``min(y0, y1) <= y < max(y0, y1)`` and the crossing lies strictly to the
  right of the point. Consequence: a point exactly on a left or bottom
  boundary is INSIDE, on a right or top boundary OUTSIDE (corners: only the
  bottom-left one of an axis-aligned square is inside). Two polygons sharing
  an edge therefore claim each boundary point exactly once. ``hatch_lines``
  uses the same rule in its rotated frame, so a hatch line through a vertex
  is counted once and an edge parallel to the hatch never counts (probe: a
  line through two hole corners and two outer corners gave exactly
  [(0,0)-(3,3)] and [(7,7)-(10,10)]; hatch length / (area / spacing) was
  0.998-1.004 on squares, holes, box and torus sections).
* Hatch lines are phase-locked to a global grid: line k lies at signed
  distance ``phase + k * spacing`` from the origin along the hatch normal
  ``(-sin a, cos a)``, so hatches of neighbouring cuts line up. A hatch line
  coincident with a boundary edge parallel to it emits a segment ON that
  edge (measured); callers hatching at 0/90 degrees offset ``phase`` by
  ``spacing / 2`` to avoid that.
* Chained loop orientation is arbitrary (probe: outer loops came out with
  area +220, -120, -2400, +530), so ``nest_loops`` nests by containment and
  even-odd depth, never by sign, and then normalises: outers CCW, holes CW.
* ``chain_segments`` welds endpoints with a grid key at ``tol`` plus a
  ``cKDTree.query_pairs(tol)`` union-find (probe: correct loop counts for
  planes through vertices; a plane tangent to a hole within 1e-12 welds the
  sliver away, which is accepted).
* Exact HLR output needs three clean-ups (HLR probe): the hidden compound
  repeats edges that lie exactly on visible ones (box back edges, the far
  mouth of a through hole) -> ``subtract_covered`` for segments and
  ``drop_covered_curves`` for Circles/Arcs; the visible compound can
  hold exact duplicates (a fillet arc 270..360 twice) -> ``dedupe_primitives``;
  fillet views produce collinear chains such as 0..5 and 5..55 on one line
  -> ``merge_collinear``.

The functions that touch ``model.py`` primitives import them lazily so this
module has no import-time dependency on the rest of the package.
"""
from __future__ import annotations

import logging
import math
from typing import List, Sequence, Tuple

import numpy as np

log = logging.getLogger(__name__)

_EMPTY_SEGS_SHAPE = (0, 2, 2)


def _empty_segments() -> np.ndarray:
    return np.zeros(_EMPTY_SEGS_SHAPE, dtype=np.float64)


def _as_loop(loop) -> np.ndarray:
    """(n, 2) float copy with a repeated end point removed."""
    arr = np.asarray(loop, dtype=np.float64).reshape(-1, 2)
    if len(arr) >= 2 and np.array_equal(arr[0], arr[-1]):
        arr = arr[:-1]
    return arr


def _as_segments(segments) -> np.ndarray:
    arr = np.asarray(segments, dtype=np.float64)
    if arr.size == 0:
        return _empty_segments()
    return arr.reshape(-1, 2, 2)


def _point_segment_distance(pts: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Euclidean distance of points (n, 2) to the closed segment a-b."""
    d = b - a
    dd = float(d @ d)
    if dd <= 0.0:
        return np.linalg.norm(pts - a, axis=1)
    t = np.clip(((pts - a) @ d) / dd, 0.0, 1.0)
    proj = a + t[:, None] * d
    return np.linalg.norm(pts - proj, axis=1)


# ----------------------------------------------------------------- polylines

def simplify_polyline(points: np.ndarray, tol: float) -> np.ndarray:
    """Douglas-Peucker simplification; keeps the first and last point.

    Distances are measured to the chord SEGMENT (not the infinite line), so a
    polyline that doubles back is not collapsed. A closed input (first ==
    last) splits at the point farthest from the start. Returns a new array.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    n = len(pts)
    if n <= 2:
        return pts.copy()
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j - i < 2:
            continue
        inner = pts[i + 1:j]
        dist = _point_segment_distance(inner, pts[i], pts[j])
        k = int(np.argmax(dist))
        if dist[k] > tol:
            m = i + 1 + k
            keep[m] = True
            stack.append((i, m))
            stack.append((m, j))
    return pts[keep].copy()


def is_straight(points: np.ndarray, tol: float) -> bool:
    """True when every point lies within ``tol`` of the first-to-last chord.

    A chord shorter than ``tol`` (closed or degenerate input) counts as
    straight only when all points lie within ``tol`` of the first point.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 2:
        return True
    a, b = pts[0], pts[-1]
    if float(np.linalg.norm(b - a)) <= tol:
        return bool(np.all(np.linalg.norm(pts - a, axis=1) <= tol))
    return bool(np.all(_point_segment_distance(pts, a, b) <= tol))


# ------------------------------------------------------------------ polygons

def polygon_area(points: np.ndarray) -> float:
    """Signed shoelace area: positive for counter-clockwise loops."""
    pts = _as_loop(points)
    if len(pts) < 3:
        return 0.0
    x, y = pts[:, 0], pts[:, 1]
    return float(0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y)))


_PAIR_BLOCK = 2_000_000


def _edge_point_pairs(ys: np.ndarray, lo_idx: np.ndarray, hi_idx: np.ndarray):
    """Yield (edge index, sorted point index) arrays for every edge e and every
    sorted point position in [lo_idx[e], hi_idx[e]), in blocks of about
    ``_PAIR_BLOCK`` pairs (scanline pairing: only points in an edge's y-band)."""
    counts = np.maximum(hi_idx - lo_idx, 0)
    if counts.sum() == 0:
        return
    cum = np.cumsum(counts)
    e0 = 0
    k = len(counts)
    while e0 < k:
        base = int(cum[e0 - 1]) if e0 else 0
        e1 = int(np.searchsorted(cum, base + _PAIR_BLOCK, side="right"))
        e1 = min(max(e1, e0 + 1), k)
        c = counts[e0:e1]
        tot = int(c.sum())
        if tot:
            eidx = np.repeat(np.arange(e0, e1), c)
            start = np.cumsum(c) - c
            pidx = lo_idx[eidx] + (np.arange(tot) - np.repeat(start, c))
            yield eidx, pidx
        e0 = e1


def _points_in_polygon(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Vectorised half-open crossing test of pts (m, 2) against one loop.

    An edge crosses the ray of a point when ``min(y0, y1) <= y < max(y0, y1)``
    and the crossing lies strictly right of the point. The points are sorted
    by y so each edge is only paired with the points in its y-band.
    """
    poly = _as_loop(poly)
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if len(poly) < 3 or len(pts) == 0:
        return np.zeros(len(pts), dtype=bool)
    x0, y0 = poly[:, 0], poly[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    order = np.argsort(pts[:, 1], kind="stable")
    sp = pts[order]
    ys = sp[:, 1]
    lo_idx = np.searchsorted(ys, np.minimum(y0, y1), side="left")
    hi_idx = np.searchsorted(ys, np.maximum(y0, y1), side="left")     # half-open: y < max
    hits = np.zeros(len(pts), dtype=np.int64)
    dy_all = np.where(y1 == y0, 1.0, y1 - y0)
    for e, p in _edge_point_pairs(ys, lo_idx, hi_idx):
        py = ys[p]
        xint = x0[e] + (py - y0[e]) * (x1[e] - x0[e]) / dy_all[e]
        hit = sp[p, 0] < xint
        hits += np.bincount(p[hit], minlength=len(pts))
    inside = np.zeros(len(pts), dtype=bool)
    inside[order] = (hits % 2) == 1
    return inside


def point_in_polygon(pt, poly: np.ndarray) -> bool:
    """Half-open crossing rule (see module docstring for boundary points)."""
    p = np.asarray(pt, dtype=np.float64).reshape(1, 2)
    return bool(_points_in_polygon(p, poly)[0])


def _on_boundary(pts: np.ndarray, poly: np.ndarray, eps: float) -> np.ndarray:
    """Mask of the points (m, 2) within ``eps`` of the closed loop's edges.

    Scanline pairing: an edge is only tested against the points whose y lies
    in its y-range widened by ``eps``.
    """
    a = poly
    b = np.roll(poly, -1, axis=0)
    d = b - a
    dd = np.einsum("ij,ij->i", d, d)
    dd = np.where(dd > 0.0, dd, 1.0)
    order = np.argsort(pts[:, 1], kind="stable")
    sp = pts[order]
    ys = sp[:, 1]
    lo_idx = np.searchsorted(ys, np.minimum(a[:, 1], b[:, 1]) - eps, side="left")
    hi_idx = np.searchsorted(ys, np.maximum(a[:, 1], b[:, 1]) + eps, side="right")
    hit = np.zeros(len(pts), dtype=bool)
    for e, p in _edge_point_pairs(ys, lo_idx, hi_idx):
        rel = sp[p] - a[e]
        tt = np.clip(np.einsum("ij,ij->i", rel, d[e]) / dd[e], 0.0, 1.0)
        diff = rel - tt[:, None] * d[e]
        near = np.einsum("ij,ij->i", diff, diff) <= eps * eps
        hit[p[near]] = True
    out = np.zeros(len(pts), dtype=bool)
    out[order] = hit
    return out


def _loop_inside(inner: np.ndarray, outer: np.ndarray, eps: float) -> bool:
    """Majority vote of ``inner``'s vertices that are NOT on ``outer``'s boundary.

    The half-open rule treats boundary points asymmetrically (left inside,
    right outside), so a hole touching the right edge would lose the vote
    while the same hole touching the left edge wins it. Boundary vertices
    therefore abstain; when all vertices abstain, the edge midpoints vote.
    """
    lo_i, hi_i = inner.min(axis=0), inner.max(axis=0)
    lo_o, hi_o = outer.min(axis=0), outer.max(axis=0)
    if np.any(lo_i < lo_o - eps) or np.any(hi_i > hi_o + eps):
        return False                                    # bbox not contained: cannot be inside
    for cand in (inner, 0.5 * (inner + np.roll(inner, -1, axis=0))):
        free = ~_on_boundary(cand, outer, eps)
        if free.any():
            return float(np.mean(_points_in_polygon(cand[free], outer))) > 0.5
    log.debug("nest_loops: every vertex and edge midpoint lies on the other loop; not nested")
    return False


def nest_loops(loops: List[np.ndarray]) -> List[Tuple[np.ndarray, List[np.ndarray]]]:
    """Group closed loops into (outer, [holes]) by containment.

    Loop i lies inside loop j when |area j| > |area i| and more than half of
    i's vertices that do not lie on j's boundary (within 1e-9 x coordinate
    scale) are inside j (half-open rule); boundary vertices abstain so the
    vote does not depend on which side a touching loop touches, and when all
    vertices lie on j's boundary the edge midpoints vote. The containment depth decides
    the role: even depth -> outer, odd depth -> hole of its smallest
    containing loop of depth - 1; an island inside a hole is a new outer.
    Input orientation is ignored; outputs are normalised to outer CCW and
    holes CW. Loops with fewer than 3 points or zero area are dropped.
    Outers keep the input order.
    """
    clean: List[np.ndarray] = []
    areas: List[float] = []
    for L in loops:
        arr = _as_loop(L)
        a = polygon_area(arr) if len(arr) >= 3 else 0.0
        if len(arr) < 3 or a == 0.0:
            log.debug("nest_loops: dropped a degenerate loop with %d points", len(arr))
            continue
        clean.append(arr)
        areas.append(abs(a))
    n = len(clean)
    if n == 0:
        return []
    scale = max(1.0, max(float(np.abs(c).max()) for c in clean))
    eps = 1e-9 * scale
    inside = np.zeros((n, n), dtype=bool)          # inside[i, j]: i within j
    for j in range(n):
        for i in range(n):
            if i == j or not areas[j] > areas[i]:
                continue
            inside[i, j] = _loop_inside(clean[i], clean[j], eps)
    depth = inside.sum(axis=1)
    groups: dict = {}
    order: List[int] = []
    for i in range(n):
        if depth[i] % 2 == 0:
            groups[i] = []
            order.append(i)
    for i in range(n):
        if depth[i] % 2 == 0:
            continue
        parents = [j for j in range(n) if inside[i, j] and depth[j] == depth[i] - 1]
        if not parents:          # inconsistent containment: promote to an outer
            log.debug("nest_loops: hole without a parent, promoted to an outer")
            groups[i] = []
            order.append(i)
            continue
        parent = min(parents, key=lambda j: areas[j])
        groups.setdefault(parent, []).append(i)

    def _oriented(arr: np.ndarray, ccw: bool) -> np.ndarray:
        return arr.copy() if (polygon_area(arr) > 0) == ccw else arr[::-1].copy()

    order.sort()
    return [(_oriented(clean[o], True), [_oriented(clean[h], False) for h in groups[o]])
            for o in order]


# ------------------------------------------------------------------- hatching

_MAX_HATCH_LINES = 200_000


def hatch_lines(loops: List[np.ndarray], spacing: float, angle_deg: float = 45.0,
                phase: float = 0.0) -> np.ndarray:
    """Even-odd hatch segments (k, 2, 2) of a set of closed loops.

    All loops together form the region (even-odd), so holes need no special
    marking and their orientation does not matter. Lines run at ``angle_deg``
    from +x; line k sits at signed distance ``phase + k * spacing`` from the
    origin along the normal ``(-sin a, cos a)`` (global grid). Crossings use
    the half-open rule; zero-length pieces (corner touches) are dropped.
    """
    spacing = float(spacing)
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise ValueError(f"hatch_lines: spacing must be positive, got {spacing!r}")
    clean = [L for L in (_as_loop(x) for x in loops) if len(L) >= 3]
    if not clean:
        return _empty_segments()
    a = math.radians(float(angle_deg))
    ca, sa = math.cos(a), math.sin(a)
    rm = np.array([[ca, sa], [-sa, ca]])            # rotate by -a: hatch lines become horizontal
    e0 = np.concatenate([L @ rm.T for L in clean])
    e1 = np.concatenate([np.roll(L @ rm.T, -1, axis=0) for L in clean])
    y0, y1 = e0[:, 1], e1[:, 1]
    lo, hi = np.minimum(y0, y1), np.maximum(y0, y1)
    ymin, ymax = float(lo.min()), float(hi.max())
    k0 = math.ceil((ymin - phase) / spacing)
    k1 = math.floor((ymax - phase) / spacing)
    if k1 - k0 + 1 > _MAX_HATCH_LINES:
        raise ValueError(f"hatch_lines: {k1 - k0 + 1} lines exceed the cap {_MAX_HATCH_LINES}; "
                         f"spacing {spacing} is too small for the region")
    eps = 1e-12 * max(1.0, float(np.abs(e0).max()))
    out = []
    for k in range(k0, k1 + 1):
        c = phase + k * spacing
        m = (lo <= c) & (c < hi)                    # half-open: vertex once, parallel edges never
        if not m.any():
            continue
        t = (c - y0[m]) / (y1[m] - y0[m])
        x = np.sort(e0[m, 0] + t * (e1[m, 0] - e0[m, 0]))
        if len(x) % 2:
            log.warning("hatch_lines: odd crossing count %d at offset %.6g, line skipped", len(x), c)
            continue
        for xa, xb in zip(x[0::2], x[1::2]):
            if xb - xa > eps:
                out.append([[xa, c], [xb, c]])
    if not out:
        return _empty_segments()
    return np.asarray(out, dtype=np.float64) @ rm   # rotate back by +a


# ------------------------------------------------------------------- chaining

def _union_find_roots(n: int, pairs) -> np.ndarray:
    parent = np.arange(n)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in pairs:
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb
    return np.array([find(i) for i in range(n)], dtype=np.int64)


def chain_segments(segments: np.ndarray, tol: float) -> Tuple[List[np.ndarray], int]:
    """Chain unordered segments (k, 2, d) into closed loops.

    Endpoints closer than ``tol`` are welded (grid key at ``tol`` plus a
    KD-tree union-find); welded points become their mean. Zero-length and
    duplicate edges are ignored. Paths stop at nodes of degree != 2. Returns
    (closed loops without a repeated end point, number of open chains
    dropped). Works in 2D and 3D.
    """
    segs = np.asarray(segments, dtype=np.float64)
    if segs.size == 0:
        return [], 0
    dim = segs.shape[-1]
    segs = segs.reshape(-1, 2, dim)
    tol = float(tol)
    if not tol > 0.0:
        raise ValueError(f"chain_segments: tol must be positive, got {tol!r}")
    from scipy.spatial import cKDTree

    pts = segs.reshape(-1, dim)
    key = np.round(pts / tol).astype(np.int64)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    inv = np.asarray(inv).ravel()
    cnt = np.bincount(inv, minlength=len(uniq)).astype(np.float64)
    pts_u = np.zeros((len(uniq), dim))
    np.add.at(pts_u, inv, pts)
    pts_u /= cnt[:, None]
    pairs = cKDTree(pts_u).query_pairs(tol, output_type="ndarray")
    roots = _union_find_roots(len(uniq), pairs)
    _, node_of_u = np.unique(roots, return_inverse=True)
    node = np.asarray(node_of_u).ravel()[inv].reshape(-1, 2)
    n_nodes = int(node.max()) + 1
    node_xyz = np.zeros((n_nodes, dim))
    c2 = np.zeros(n_nodes)
    np.add.at(node_xyz, node.ravel(), pts)
    np.add.at(c2, node.ravel(), 1.0)
    node_xyz /= c2[:, None]

    edges = node[node[:, 0] != node[:, 1]]
    if len(edges) == 0:
        return [], 0
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    adj: dict = {}
    for a, b in edges:
        a, b = int(a), int(b)
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    deg = {k: len(v) for k, v in adj.items()}
    starts = sorted(k for k, v in deg.items() if v != 2) + sorted(k for k, v in deg.items() if v == 2)
    used = set()
    closed: List[np.ndarray] = []
    n_open = 0
    for s in starts:
        for nb in adj[s]:
            if (min(s, nb), max(s, nb)) in used:
                continue
            path = [s]
            prev, cur = s, nb
            used.add((min(s, nb), max(s, nb)))
            while True:
                path.append(cur)
                if cur == s:
                    break
                nxt = [x for x in adj[cur] if (min(cur, x), max(cur, x)) not in used]
                if deg[cur] != 2 or not nxt:
                    break
                prev, cur = cur, nxt[0]
                used.add((min(prev, cur), max(prev, cur)))
            if path[-1] == path[0] and len(path) > 3:
                closed.append(node_xyz[path[:-1]].copy())
            else:
                n_open += 1
    return closed, n_open


# -------------------------------------------------------- collinear handling

def _canonical_dirs(segs: np.ndarray):
    d = segs[:, 1] - segs[:, 0]
    length = np.linalg.norm(d, axis=1)
    u = d / np.where(length > 0, length, 1.0)[:, None]
    flip = (u[:, 0] < 0) | ((u[:, 0] == 0) & (u[:, 1] < 0))
    u[flip] *= -1.0
    return u, length


def merge_collinear(segments: np.ndarray, tol: float) -> np.ndarray:
    """Merge collinear segments that overlap or touch (gap <= tol).

    Candidates: two segments are on one line when both endpoints of each lie
    within ``tol`` of the other's infinite line. A merge is accepted only
    when every input endpoint of the merged run lies within ``tol`` of the
    output segment's line, so the chords of a finely faceted curve (each
    pair collinear within tol, the chain not) are never straightened: every
    input vertex stays within ``tol`` of the output. The merged segment keeps
    the original extreme endpoints. Segments not longer than ``tol`` are
    dropped. Returns (m, 2, 2).
    """
    segs = _as_segments(segments)
    if len(segs) == 0:
        return _empty_segments()
    tol = float(tol)
    u, length = _canonical_dirs(segs)
    keep = length > tol
    if not keep.all():
        log.debug("merge_collinear: dropped %d segments not longer than tol", int((~keep).sum()))
    segs, u, length = segs[keep], u[keep], length[keep]
    n = len(segs)
    if n == 0:
        return _empty_segments()
    theta = np.mod(np.arctan2(u[:, 1], u[:, 0]), math.pi)          # [0, pi)
    order = np.argsort(theta, kind="stable")
    th_sorted = theta[order]
    nrm = np.stack([-u[:, 1], u[:, 0]], axis=1)
    pairs = []
    for i in range(n):
        win = min(math.pi / 2, 2.0 * tol / length[i] + 1e-12)
        cand = []
        for shift in (-math.pi, 0.0, math.pi):           # wrap around theta = 0 / pi
            lo = np.searchsorted(th_sorted, theta[i] + shift - win, side="left")
            hi = np.searchsorted(th_sorted, theta[i] + shift + win, side="right")
            if hi > lo:
                cand.append(order[lo:hi])
        if not cand:
            continue
        # the shifted windows are disjoint unless win reaches pi/2
        j = cand[0] if len(cand) == 1 else np.unique(np.concatenate(cand))
        j = j[j > i]
        if len(j) == 0:
            continue
        # endpoints of j within tol of line i, and endpoints of i within tol of line j
        dj = np.abs((segs[j] - segs[i, 0]) @ nrm[i])                   # (m, 2)
        ok = np.all(dj <= tol, axis=1)
        if not ok.any():
            continue
        j = j[ok]
        di = np.abs(np.einsum("mk,pk->mp", nrm[j], segs[i]) - np.einsum("mk,mk->m", nrm[j], segs[j, 0])[:, None])
        ok2 = np.all(di <= tol, axis=1)
        pairs.extend((i, int(x)) for x in j[ok2])
    roots = _union_find_roots(n, pairs)
    out = []
    for r in np.unique(roots):
        members = np.nonzero(roots == r)[0]
        if len(members) == 1:
            out.append(segs[members[0]])
            continue
        out.extend(_merge_group(segs, members, length, tol))
    return np.asarray(out, dtype=np.float64).reshape(-1, 2, 2)


class _Run:
    """One merged output segment under construction (see ``_merge_group``)."""

    __slots__ = ("pts", "ps", "pe", "tmax")

    def __init__(self, a: np.ndarray, b: np.ndarray, tmax: float):
        self.pts = np.array([a, b], dtype=np.float64)
        self.ps, self.pe = a, b
        self.tmax = tmax

    def try_join(self, a: np.ndarray, b: np.ndarray, tol: float) -> bool:
        """Join a-b when it touches the run AND every endpoint of the run plus
        a and b stays within ``tol`` of the line through the new extremes."""
        ps, pe = self.ps, self.pe
        d = pe - ps
        L = math.hypot(float(d[0]), float(d[1]))
        if L <= 0.0:
            return False
        u = d / L
        ta, tb = float((a - ps) @ u), float((b - ps) @ u)
        if min(ta, tb) > L + tol or max(ta, tb) < -tol:     # gap along the run's own line
            return False
        nrm = np.array([-u[1], u[0]])
        if abs(float((a - ps) @ nrm)) > tol or abs(float((b - ps) @ nrm)) > tol:
            return False                                     # cheap reject before the full check
        cand = ((0.0, ps), (L, pe), (ta, a), (tb, b))
        nps = min(cand, key=lambda c: c[0])[1]
        npe = max(cand, key=lambda c: c[0])[1]
        nd = npe - nps
        nL = math.hypot(float(nd[0]), float(nd[1]))
        if nL <= 0.0:
            return False
        nn = np.array([-nd[1], nd[0]]) / nL
        pts = np.vstack([self.pts, a[None, :], b[None, :]])
        if float(np.max(np.abs((pts - nps) @ nn))) > tol:
            return False
        self.pts, self.ps, self.pe = pts, nps, npe
        return True


def _merge_group(segs: np.ndarray, members: np.ndarray, length: np.ndarray, tol: float) -> list:
    """Interval sweep over one union-find group of pairwise-collinear segments.

    The pairwise test is transitive under union-find: the chords of a finely
    faceted curve all pass it with their neighbours and end up in ONE group.
    The sweep therefore never trusts the group's line. A member joins an open
    run only when every endpoint of the run and of the member stays within
    ``tol`` of the line through the run's new extremes (``_Run.try_join``);
    otherwise it opens a run of its own. Runs are closed once the sweep
    (ordered along the longest member) has passed them by more than ``tol``.
    Output: one segment per run, from its original extreme endpoints.
    """
    ref = members[np.argmax(length[members])]
    p0 = segs[ref, 0]
    ur = segs[ref, 1] - p0
    ur = ur / np.linalg.norm(ur)
    t = (segs[members] - p0) @ ur                      # (m, 2)
    t_lo, t_hi = t.min(axis=1), t.max(axis=1)
    out: list = []
    runs: List[_Run] = []
    for k in np.argsort(t_lo, kind="stable"):
        m = members[k]
        a, b = (segs[m, 0], segs[m, 1]) if t[k, 0] <= t[k, 1] else (segs[m, 1], segs[m, 0])
        s_ref, e_ref = float(t_lo[k]), float(t_hi[k])
        still: List[_Run] = []
        for run in runs:
            # a projection never exceeds the true gap, so this close is safe
            if s_ref > run.tmax + tol:
                out.append(np.array([run.ps, run.pe]))
            else:
                still.append(run)
        runs = still
        for run in runs:
            if run.try_join(a, b, tol):
                run.tmax = max(run.tmax, e_ref)
                break
        else:
            runs.append(_Run(a, b, e_ref))
    out.extend(np.array([run.ps, run.pe]) for run in runs)
    return out


def subtract_covered(hidden: np.ndarray, visible: np.ndarray, tol: float) -> np.ndarray:
    """Remove the parts of hidden segments that lie on visible segments.

    A visible segment covers a hidden one where both its endpoints lie within
    ``tol`` of the hidden segment's line; its projection onto the hidden
    segment is cut out. Hidden segments (and remaining pieces) not longer
    than ``tol`` are always dropped, whatever ``visible`` holds; every other
    hidden segment with no collinear visible segment is returned unchanged.
    Only segments are handled: a hidden Circle/Arc lying on a visible one
    (the far mouth of a through hole) is removed with
    ``drop_covered_curves`` before flattening.
    """
    hid = _as_segments(hidden)
    vis = _as_segments(visible)
    tol = float(tol)
    if len(hid) == 0:
        return _empty_segments()
    hlen = np.linalg.norm(hid[:, 1] - hid[:, 0], axis=1)
    if not np.all(hlen > tol):
        log.debug("subtract_covered: dropped %d hidden segments not longer than tol",
                  int(np.count_nonzero(hlen <= tol)))
        hid = hid[hlen > tol]
        if len(hid) == 0:
            return _empty_segments()
    if len(vis) == 0:
        return hid.copy()
    vlen = np.linalg.norm(vis[:, 1] - vis[:, 0], axis=1)
    vis = vis[vlen > tol]
    out = []
    for h in hid:
        a, b = h[0], h[1]
        d = b - a
        L = float(np.linalg.norm(d))
        if L <= tol:
            continue
        u = d / L
        nrm = np.array([-u[1], u[0]])
        if len(vis):
            off = np.abs((vis - a) @ nrm)                                # (m, 2)
            on = np.all(off <= tol, axis=1)
        else:
            on = np.zeros(0, dtype=bool)
        if not on.any():
            out.append(h.copy())
            continue
        t = (vis[on] - a) @ u                                            # (m, 2)
        iv = np.sort(t, axis=1)
        iv = iv[(iv[:, 1] > 0.0) & (iv[:, 0] < L)]
        iv = iv[np.argsort(iv[:, 0])]
        cur = 0.0
        pieces = []
        for s, e in iv:
            if s > cur:
                pieces.append((cur, min(s, L)))
            cur = max(cur, e)
            if cur >= L:
                break
        if cur < L:
            pieces.append((cur, L))
        for s, e in pieces:
            if e - s > tol:
                out.append(np.array([a + s * u, a + e * u]))
    if not out:
        return _empty_segments()
    return np.asarray(out, dtype=np.float64).reshape(-1, 2, 2)


# -------------------------------------------------------------- primitives

def _norm_deg(x: float) -> float:
    return float(np.mod(x, 360.0))


def _arc_sweep(start_deg: float, end_deg: float) -> float:
    """CCW sweep in degrees; start == end (mod 360) means a full circle (DXF rule)."""
    sweep = float(np.mod(end_deg - start_deg, 360.0))
    if sweep < 1e-9 or sweep > 360.0 - 1e-9:
        return 360.0
    return sweep


def _ang_close(a: float, b: float, tol_deg: float) -> bool:
    d = abs(_norm_deg(a) - _norm_deg(b))
    return min(d, 360.0 - d) <= tol_deg


def dedupe_primitives(prims: list, tol: float) -> list:
    """Drop duplicate primitives (first occurrence kept, order preserved).

    Duplicates have the same type and style and coincide within ``tol``:
    Segments in either direction; Circles by centre and radius; Arcs by
    centre, radius, start angle (mod 360) and sweep; Polylines point by point
    forward or reversed with the same ``closed`` flag. Any other item (Text,
    Polygon, HatchLines) is passed through untouched.
    """
    from scipy.spatial import cKDTree

    from .model import Arc, Circle, Polyline, Segment

    tol = float(tol)
    prims = list(prims)
    groups: dict = {Segment: [], Circle: [], Arc: [], Polyline: []}
    for idx, p in enumerate(prims):
        t = type(p)
        if t in groups:
            groups[t].append(idx)
    drop = set()
    r = tol * (1.0 + 1e-9) + 1e-15

    def same_pt(p, q) -> bool:
        return float(np.hypot(float(p[0]) - float(q[0]), float(p[1]) - float(q[1]))) <= tol

    def equal(p, q) -> bool:
        if p.style != q.style:
            return False
        if isinstance(p, Segment):
            return (same_pt(p.a, q.a) and same_pt(p.b, q.b)) or (same_pt(p.a, q.b) and same_pt(p.b, q.a))
        if isinstance(p, Arc):
            if not (same_pt(p.center, q.center) and abs(p.radius - q.radius) <= tol):
                return False
            ang_tol = math.degrees(tol / max(abs(p.radius), tol))
            return (_ang_close(p.start_deg, q.start_deg, ang_tol)
                    and abs(_arc_sweep(p.start_deg, p.end_deg) - _arc_sweep(q.start_deg, q.end_deg)) <= ang_tol)
        if isinstance(p, Circle):
            return same_pt(p.center, q.center) and abs(p.radius - q.radius) <= tol
        if isinstance(p, Polyline):
            pa = np.asarray(p.points, float).reshape(-1, 2)
            qa = np.asarray(q.points, float).reshape(-1, 2)
            if bool(p.closed) != bool(q.closed) or pa.shape != qa.shape:
                return False
            if np.all(np.linalg.norm(pa - qa, axis=1) <= tol):
                return True
            return bool(np.all(np.linalg.norm(pa - qa[::-1], axis=1) <= tol))
        return False

    def key(p) -> Tuple[float, float]:
        if isinstance(p, Segment):
            return (0.5 * (float(p.a[0]) + float(p.b[0])), 0.5 * (float(p.a[1]) + float(p.b[1])))
        if isinstance(p, (Circle, Arc)):
            return (float(p.center[0]), float(p.center[1]))
        pts = np.asarray(p.points, float).reshape(-1, 2)
        c = pts.mean(axis=0) if len(pts) else np.zeros(2)
        return (float(c[0]), float(c[1]))

    for t, idxs in groups.items():
        if len(idxs) < 2:
            continue
        keys = np.array([key(prims[i]) for i in idxs], dtype=np.float64)
        cand = cKDTree(keys).query_pairs(r, output_type="ndarray")
        verified = [(a, b) for a, b in cand if equal(prims[idxs[a]], prims[idxs[b]])]
        if not verified:
            continue
        roots = _union_find_roots(len(idxs), verified)
        for local, root in enumerate(roots):
            if root != local:                 # root is the smallest index of its component
                drop.add(idxs[local])
    if drop:
        log.debug("dedupe_primitives: dropped %d duplicates", len(drop))
    return [p for i, p in enumerate(prims) if i not in drop]


def drop_covered_curves(hidden: list, visible: list, tol: float) -> list:
    """Drop hidden Circles/Arcs that lie on visible Circles/Arcs.

    The HLR hidden compound repeats curves that coincide with visible ones,
    e.g. the far mouth of a through hole in the axial view (a hidden Circle
    on the visible near-mouth Circle). ``dedupe_primitives`` keeps them
    because the styles differ and ``subtract_covered`` handles segments only.
    A hidden Circle or Arc is dropped when the visible Circles/Arcs with the
    same centre and radius (within ``tol``) cover its whole angular range
    (union of intervals, angular tolerance ``tol / radius``). Partly covered
    curves and every other hidden item are returned unchanged, order kept.
    """
    from .model import Arc, Circle

    tol = float(tol)
    vis = [v for v in visible if isinstance(v, (Circle, Arc))]
    if not vis:
        return list(hidden)
    vc = np.array([[float(v.center[0]), float(v.center[1])] for v in vis], dtype=np.float64)
    vr = np.array([float(v.radius) for v in vis], dtype=np.float64)
    out = []
    n_drop = 0
    for h in hidden:
        if not isinstance(h, (Circle, Arc)):
            out.append(h)
            continue
        hc = np.array([float(h.center[0]), float(h.center[1])])
        hr = float(h.radius)
        same = (np.linalg.norm(vc - hc, axis=1) <= tol) & (np.abs(vr - hr) <= tol)
        if not same.any():
            out.append(h)
            continue
        ang_tol = math.degrees(tol / max(abs(hr), tol))
        h0 = 0.0 if isinstance(h, Circle) else float(h.start_deg)
        hsw = 360.0 if isinstance(h, Circle) else _arc_sweep(h.start_deg, h.end_deg)
        iv = []
        for k in np.nonzero(same)[0]:
            v = vis[k]
            if isinstance(v, Circle):
                iv.append((-ang_tol, 360.0 + ang_tol))
                continue
            s = _norm_deg(float(v.start_deg) - h0)           # relative to the hidden start
            sw = _arc_sweep(v.start_deg, v.end_deg)
            iv.append((s - ang_tol, s + sw + ang_tol))
            iv.append((s - 360.0 - ang_tol, s - 360.0 + sw + ang_tol))
        iv.sort()
        cur = 0.0
        for s, e in iv:
            if s > cur:
                break
            cur = max(cur, e)
        if cur >= hsw:
            n_drop += 1
        else:
            out.append(h)
    if n_drop:
        log.debug("drop_covered_curves: dropped %d hidden curves lying on visible ones", n_drop)
    return out


def _circle_steps(radius: float, sweep_rad: float, deflection: float, min_full: int = 8) -> int:
    r = abs(float(radius))
    if r <= 0.0:
        return 1
    c = 1.0 - deflection / r
    step = 2.0 * math.acos(c) if c > -1.0 else 2.0 * math.pi
    step = max(step, 1e-6)
    n = int(math.ceil(sweep_rad / step - 1e-12))
    n_min = max(1, int(math.ceil(min_full * sweep_rad / (2.0 * math.pi))))
    return int(min(max(n, n_min), 4096))


def _arc_points(center, radius: float, start_rad: float, sweep_rad: float, deflection: float) -> np.ndarray:
    n = _circle_steps(radius, sweep_rad, deflection)
    t = start_rad + np.linspace(0.0, sweep_rad, n + 1)
    cx, cy = float(center[0]), float(center[1])
    return np.stack([cx + radius * np.cos(t), cy + radius * np.sin(t)], axis=1)


def _polyline_segments(pts: np.ndarray, closed: bool) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if closed and len(pts) >= 2 and np.array_equal(pts[0], pts[-1]):
        pts = pts[:-1]
    if len(pts) < 2:
        return _empty_segments()
    if closed and len(pts) >= 3:
        nxt = np.roll(pts, -1, axis=0)
        return np.stack([pts, nxt], axis=1)
    return np.stack([pts[:-1], pts[1:]], axis=1)


def primitives_to_segments(prims: list, deflection: float) -> np.ndarray:
    """Flatten primitives into line segments (k, 2, 2).

    Segment -> itself; Polyline -> its edges (closing edge when ``closed``);
    Circle / Arc -> chords whose sagitta is at most ``deflection`` (at least
    8 chords per full turn, at most 4096); an Arc with start == end (mod 360)
    is a full circle, as in DXF; Polygon -> its closed boundary; HatchLines ->
    its segments. Text carries no geometry and is skipped; any other type
    raises TypeError.
    """
    from .model import Arc, Circle, HatchLines, Polygon, Polyline, Segment, Text

    deflection = float(deflection)
    if not math.isfinite(deflection) or deflection <= 0.0:
        raise ValueError(f"primitives_to_segments: deflection must be positive, got {deflection!r}")
    chunks: List[np.ndarray] = []
    for p in prims:
        if isinstance(p, Segment):
            chunks.append(np.array([[p.a, p.b]], dtype=np.float64))
        elif isinstance(p, Polyline):
            chunks.append(_polyline_segments(p.points, bool(p.closed)))
        elif isinstance(p, Arc):
            sweep = math.radians(_arc_sweep(p.start_deg, p.end_deg))
            pts = _arc_points(p.center, float(p.radius), math.radians(float(p.start_deg)), sweep, deflection)
            chunks.append(np.stack([pts[:-1], pts[1:]], axis=1))
        elif isinstance(p, Circle):
            pts = _arc_points(p.center, float(p.radius), 0.0, 2.0 * math.pi, deflection)
            chunks.append(np.stack([pts[:-1], pts[1:]], axis=1))
        elif isinstance(p, Polygon):
            chunks.append(_polyline_segments(p.points, True))
        elif isinstance(p, HatchLines):
            chunks.append(_as_segments(p.segments))
        elif isinstance(p, Text):
            continue
        else:
            raise TypeError(f"primitives_to_segments: unsupported item {type(p).__name__}")
    chunks = [c for c in chunks if len(c)]
    if not chunks:
        return _empty_segments()
    return np.concatenate(chunks).astype(np.float64)


__all__: Sequence[str] = (
    "simplify_polyline", "is_straight", "polygon_area", "point_in_polygon", "nest_loops",
    "hatch_lines", "chain_segments", "merge_collinear", "subtract_covered",
    "dedupe_primitives", "drop_covered_curves", "primitives_to_segments",
)
