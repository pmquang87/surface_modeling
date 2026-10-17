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
import time
from typing import List, Optional, Sequence, Tuple

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


def _weld_endpoints(segs: np.ndarray, tol: float):
    """Weld segment endpoints closer than ``tol``: (node index per endpoint (k, 2), node coords).

    Grid key at ``tol`` plus a KD-tree union-find; a node sits at the mean of
    its welded endpoints. Works in 2D and 3D.
    """
    from scipy.spatial import cKDTree

    dim = segs.shape[-1]
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
    return node, node_xyz


def _check_deadline(deadline, what: str) -> None:
    if deadline is not None and time.perf_counter() > deadline:
        raise TimeoutError(f"{what}: time budget exceeded")


def chain_paths(segments: np.ndarray, tol: float, *, deadline: float = None
                ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Chain unordered 2D segments (k, 2, 2) into polylines: ``(open, closed)``.

    Endpoints closer than ``tol`` are welded as in ``chain_segments``
    (welded points move to their mean, so by at most ``tol``). Zero-length
    and duplicate edges are ignored. A path runs through nodes of degree 2
    and stops at nodes of any other degree (ends, T- and X-junctions), so
    every input edge lands in exactly one path and no path crosses a
    junction. Open paths are (n >= 2, 2) arrays from end to end; closed
    paths (cycles of degree-2 nodes, n >= 3) have no repeated end point.
    Paths are emitted in a deterministic order (by smallest start node).
    ``deadline`` (a ``time.perf_counter()`` value) raises ``TimeoutError``
    when passed during the walk.
    """
    segs = _as_segments(segments)
    if len(segs) == 0:
        return [], []
    tol = float(tol)
    if not tol > 0.0:
        raise ValueError(f"chain_paths: tol must be positive, got {tol!r}")
    node, node_xy = _weld_endpoints(segs, tol)
    edges = node[node[:, 0] != node[:, 1]]
    if len(edges) == 0:
        return [], []
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    n_nodes = len(node_xy)
    # CSR adjacency: neighbours of node v are nb[ptr[v]:ptr[v + 1]] (edge ids in eid)
    ends = np.concatenate([edges[:, 0], edges[:, 1]])
    other = np.concatenate([edges[:, 1], edges[:, 0]])
    eid = np.concatenate([np.arange(len(edges)), np.arange(len(edges))])
    order = np.argsort(ends, kind="stable")
    nb = other[order].tolist()
    eids = eid[order].tolist()
    deg = np.bincount(ends, minlength=n_nodes)
    ptr = np.concatenate([[0], np.cumsum(deg)]).tolist()
    deg = deg.tolist()
    used = [False] * len(edges)
    open_paths: List[np.ndarray] = []
    closed_paths: List[np.ndarray] = []
    steps = 0

    def walk(s: int, k: int) -> list:
        nonlocal steps
        path = [s]
        while True:
            used[eids[k]] = True
            nxt = nb[k]
            path.append(nxt)
            steps += 1
            if steps % 4096 == 0:
                _check_deadline(deadline, "chain_paths")
            if nxt == s or deg[nxt] != 2:
                return path
            a, b = ptr[nxt], ptr[nxt + 1]
            k = next((q for q in range(a, b) if not used[eids[q]]), -1)
            if k < 0:
                return path

    def emit(p: list) -> None:
        if p[-1] == p[0] and len(p) > 3:          # loop (possibly hanging on a junction)
            closed_paths.append(node_xy[p[:-1]].copy())
        else:                                      # a 2-cycle cannot occur (edges are unique)
            open_paths.append(node_xy[p].copy())

    for s in range(n_nodes):                      # paths from ends and junctions
        if deg[s] == 2 or deg[s] == 0:
            continue
        for k in range(ptr[s], ptr[s + 1]):
            if not used[eids[k]]:
                emit(walk(s, k))
    for s in range(n_nodes):                      # what is left: pure cycles
        if deg[s] != 2:
            continue
        for k in range(ptr[s], ptr[s + 1]):
            if not used[eids[k]]:
                emit(walk(s, k))
    return open_paths, closed_paths


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
    node, node_xyz = _weld_endpoints(segs, tol)

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


_SAMPLES_PER_SEG = 8.0       # cap on the mean number of KD samples per segment in _near_pairs


def _seg_samples(segs: np.ndarray, step: float):
    """Points every <= ``step`` along each segment (ends included) and their owner index."""
    d = segs[:, 1] - segs[:, 0]
    L = np.hypot(d[:, 0], d[:, 1])
    k = np.maximum(1, np.ceil(L / step)).astype(np.int64)        # intervals per segment
    cnt = k + 1
    owner = np.repeat(np.arange(len(segs), dtype=np.int64), cnt)
    first = np.cumsum(cnt) - cnt
    j = np.arange(int(cnt.sum()), dtype=np.int64) - np.repeat(first, cnt)
    t = j / np.repeat(k, cnt)
    return segs[owner, 0] + d[owner] * t[:, None], owner


_NEAR_CLASSES = 8           # sample-spacing classes in _near_pairs (spacing bound step / 2**m)


def _spacing_classes(segs: np.ndarray, step: float):
    """Samples of ``segs`` at <= ``step`` and the spacing class of each sample.

    A segment shorter than ``step`` is sampled at its two ends only, so its
    real spacing is its length; class m holds samples whose segment spacing
    is at most ``step / 2**m`` (the last class: everything finer).
    """
    pts, own = _seg_samples(segs, step)
    d = segs[:, 1] - segs[:, 0]
    L = np.hypot(d[:, 0], d[:, 1])
    k = np.maximum(1, np.ceil(L / step))
    s = np.maximum(L / k, step * 1e-12)
    m = np.clip(np.floor(np.log2(step / s)), 0, _NEAR_CLASSES - 1).astype(np.int64)
    return pts, own, m[own]


def _near_pairs(a: np.ndarray, b: Optional[np.ndarray], r: float, *, max_sin: Optional[float] = None,
                deadline: Optional[float] = None):
    """Candidate pairs of segments closer than ``r`` (a superset), O(n log n).

    Every segment is sampled at a step of at most ``s``; a sample lies
    within ``h = spacing / 2`` of every point of its segment piece, so two
    segments whose distance is at most ``r`` own samples at most
    ``r + h_i + h_j`` apart and a KD-tree radius query finds every such
    pair. ``s`` is the median segment length (at least ``2 r``), raised
    until there are at most ``_SAMPLES_PER_SEG`` samples per segment on
    average. Segments shorter than ``s`` keep their own (smaller) spacing:
    the samples are grouped into ``_NEAR_CLASSES`` spacing classes and each
    class pair is queried with its own radius, so a view of 0.2 mm chords
    with a 1.5 mm tolerance is not queried at ``r + s = 4.5`` mm (F1
    review: 03 front view, 11 M candidate pairs and 18 s before, see the
    measured numbers in the task report).

    ``max_sin`` (optional): keep only pairs whose directions differ by at
    most ``asin(max_sin)`` (the loose-mode parallelism test), applied to
    the sample pairs before the unique step. ``deadline`` raises
    ``TimeoutError`` between the class queries. ``b is None``: pairs
    (i, j), i < j, within ``a``; otherwise pairs (i into a, j into b).
    Returns two int64 arrays (unique pairs).
    """
    from scipy.spatial import cKDTree

    na = len(a)
    nb = 0 if b is None else len(b)
    empty = (np.zeros(0, np.int64), np.zeros(0, np.int64))
    if na == 0 or (b is not None and nb == 0) or (b is None and na < 2):
        return empty
    allseg = a if b is None else np.concatenate([a, b])
    L = np.linalg.norm(allseg[:, 1] - allseg[:, 0], axis=1)
    r = float(r)
    step = max(2.0 * r, float(np.median(L)), float(L.sum()) / (_SAMPLES_PER_SEG * len(allseg)), 1e-12)
    bound = step / np.power(2.0, np.arange(_NEAR_CLASSES))          # spacing bound per class
    pa, oa, ca = _spacing_classes(a, step)
    da = a[:, 1] - a[:, 0]
    if b is None:
        pb, ob, cb, db, n_key = pa, oa, ca, da, na
    else:
        pb, ob, cb = _spacing_classes(b, step)
        db, n_key = b[:, 1] - b[:, 0], nb
    groups_a = [np.flatnonzero(ca == m) for m in range(_NEAR_CLASSES)]
    groups_b = groups_a if b is None else [np.flatnonzero(cb == m) for m in range(_NEAR_CLASSES)]
    trees_a = [cKDTree(pa[g]) if len(g) else None for g in groups_a]
    trees_b = trees_a if b is None else [cKDTree(pb[g]) if len(g) else None for g in groups_b]
    keys = []
    for ma in range(_NEAR_CLASSES):
        if trees_a[ma] is None:
            continue
        for mb in range(ma if b is None else 0, _NEAR_CLASSES):
            if trees_b[mb] is None:
                continue
            _check_deadline(deadline, "near pairs")
            rad = (r + 0.5 * (bound[ma] + bound[mb])) * (1.0 + 1e-9) + 1e-12
            if b is None and ma == mb:
                pr = trees_a[ma].query_pairs(rad, output_type="ndarray")
                if len(pr) == 0:
                    continue
                i, j = oa[groups_a[ma][pr[:, 0]]], oa[groups_a[ma][pr[:, 1]]]
            else:
                sdm = trees_a[ma].sparse_distance_matrix(trees_b[mb], rad, output_type="ndarray")
                if len(sdm) == 0:
                    continue
                i, j = oa[groups_a[ma][sdm["i"]]], ob[groups_b[mb][sdm["j"]]]
            if b is None:
                m = i != j
                i, j = np.minimum(i[m], j[m]), np.maximum(i[m], j[m])
            if max_sin is not None and len(i):
                u, v = da[i], db[j]
                cross = np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0])
                keep = cross <= float(max_sin) * np.hypot(u[:, 0], u[:, 1]) * np.hypot(v[:, 0], v[:, 1])
                i, j = i[keep], j[keep]
            if len(i):
                keys.append(np.unique(i * n_key + j))
    if not keys:
        return empty
    # every segment lies in exactly one class, so a segment pair is found by exactly one
    # class pair: the per-class key sets are disjoint and need no second unique
    key = np.concatenate(keys) if len(keys) > 1 else keys[0]
    return key // n_key, key % n_key


def _components(n: int, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """Connected-component label per node (labels ordered by smallest member)."""
    if len(i) == 0:
        return np.arange(n, dtype=np.int64)
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    g = coo_matrix((np.ones(len(i), np.int8), (i, j)), shape=(n, n))
    _, lab = connected_components(g, directed=False)
    return np.asarray(lab, dtype=np.int64)


def premerge_collinear(segs: np.ndarray, tol: float) -> np.ndarray:
    """Cheap O(n log n) union of overlapping runs on (numerically) the same line.

    Edge-on rims project onto one line as hundreds of short, exactly
    collinear segments (a 2000-section cylinder in a side view: 4000 rim
    edges on two lines), and poly-HLR hidden compounds repeat edges split at
    different parameters. Segments whose direction agrees to 1e-9 rad and
    whose line offset falls in the same ``tol`` bin are unioned per line
    here (1D interval union, gap <= tol), keeping the extreme original
    endpoints. Segments in neighbouring bins are left to
    ``merge_collinear``. Zero-length segments are dropped.
    """
    s = np.asarray(segs, dtype=np.float64).reshape(-1, 2, 2)
    if len(s) < 2:
        return s
    d = s[:, 1] - s[:, 0]
    length = np.linalg.norm(d, axis=1)
    s, d, length = s[length > 0], d[length > 0], length[length > 0]
    if len(s) < 2:
        return s
    u = d / length[:, None]
    flip = (u[:, 0] < 0) | ((u[:, 0] == 0) & (u[:, 1] < 0))
    u[flip] *= -1.0
    theta = np.arctan2(u[:, 1], u[:, 0])
    off = s[:, 0, 0] * -u[:, 1] + s[:, 0, 1] * u[:, 0]
    keys = np.stack([np.round(theta / 1e-9), np.round(off / max(tol, 1e-300))], axis=1).astype(np.int64)
    _, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inv = np.asarray(inv).ravel()
    single = counts[inv] == 1
    out = [s[single]]
    multi = np.nonzero(~single)[0]
    if len(multi):
        order = multi[np.argsort(inv[multi], kind="stable")]
        g = inv[order]
        cuts = np.nonzero(np.diff(g))[0] + 1
        for members in np.split(order, cuts):
            ref_u = u[members[0]]
            p0 = s[members[0], 0]
            pts = s[members].reshape(-1, 2)              # (2m, 2)
            t = (pts - p0) @ ref_u
            t_pair = t.reshape(-1, 2)
            lo_i = np.argmin(t_pair, axis=1)
            t_lo = t_pair[np.arange(len(members)), lo_i]
            t_hi = t_pair[np.arange(len(members)), 1 - lo_i]
            p_lo = pts.reshape(-1, 2, 2)[np.arange(len(members)), lo_i]
            p_hi = pts.reshape(-1, 2, 2)[np.arange(len(members)), 1 - lo_i]
            idx = np.argsort(t_lo, kind="stable")
            run_a, run_b, run_end = p_lo[idx[0]], p_hi[idx[0]], t_hi[idx[0]]
            runs = []
            for k in idx[1:]:
                if t_lo[k] <= run_end + tol:
                    if t_hi[k] > run_end:
                        run_end, run_b = t_hi[k], p_hi[k]
                else:
                    runs.append((run_a, run_b))
                    run_a, run_b, run_end = p_lo[k], p_hi[k], t_hi[k]
            runs.append((run_a, run_b))
            out.append(np.asarray(runs, dtype=np.float64).reshape(-1, 2, 2))
    return np.concatenate(out).reshape(-1, 2, 2)


def _merge_pairs_vectorized(segs: np.ndarray, pairs: np.ndarray, tol: float):
    """``_merge_group`` for two-member groups, vectorized: (merged (m, 2, 2), ok mask)."""
    a, b = segs[pairs[:, 0]], segs[pairs[:, 1]]
    pts = np.concatenate([a, b], axis=1)                         # (m, 4, 2)
    la = np.linalg.norm(a[:, 1] - a[:, 0], axis=1)
    lb = np.linalg.norm(b[:, 1] - b[:, 0], axis=1)
    ref = np.where((la >= lb)[:, None, None], a, b)
    ur = ref[:, 1] - ref[:, 0]
    ur = ur / np.linalg.norm(ur, axis=1)[:, None]
    t = np.einsum("mpk,mk->mp", pts - ref[:, :1], ur)
    rows = np.arange(len(pts))
    ps = pts[rows, np.argmin(t, axis=1)]
    pe = pts[rows, np.argmax(t, axis=1)]
    nd = pe - ps
    nL = np.linalg.norm(nd, axis=1)
    nn = np.stack([-nd[:, 1], nd[:, 0]], axis=1) / np.where(nL > 0, nL, 1.0)[:, None]
    dev = np.abs(np.einsum("mpk,mk->mp", pts - ps[:, None, :], nn)).max(axis=1)
    ok = (nL > 0) & (dev <= tol)
    return np.stack([ps, pe], axis=1), ok


def merge_collinear(segments: np.ndarray, tol: float, *, deadline: Optional[float] = None) -> np.ndarray:
    """Merge collinear segments that overlap or touch (gap <= tol).

    Candidates: two segments are on one line when both endpoints of each lie
    within ``tol`` of the other's infinite line, and they touch: one's
    extent along the other's line overlaps it or leaves a gap <= ``tol``.
    A merge is accepted only when every input endpoint of the merged run
    lies within ``tol`` of the output segment's line, so the chords of a
    finely faceted curve (each pair collinear within tol, the chain not)
    are never straightened: every input vertex stays within ``tol`` of the
    output. The merged segment keeps the original extreme endpoints.
    Segments not longer than ``tol`` are dropped. Returns (m, 2, 2).

    Cost O(n log n) (QA issue 3: the old angular-window scan was quadratic
    on short chords, 338 s for 175k poly-HLR chords): exactly collinear
    runs are unioned first (``premerge_collinear``), the touching
    candidates come from a KD-tree over points sampled along the segments
    (``_near_pairs``), groups are connected components, two-member groups
    are merged vectorized and larger ones by the ``_merge_group`` sweep.
    ``deadline`` (a ``time.perf_counter()`` value) raises ``TimeoutError``
    once passed.
    """
    segs = _as_segments(segments)
    if len(segs) == 0:
        return _empty_segments()
    tol = float(tol)
    # exact runs first, so tiny exactly collinear pieces (edge-on rim chords near a
    # silhouette) still extend their run before the length filter drops them
    segs = premerge_collinear(segs, tol)
    u, length = _canonical_dirs(segs)
    keep = length > tol
    if not keep.all():
        log.debug("merge_collinear: dropped %d segments not longer than tol", int((~keep).sum()))
    segs = segs[keep]
    if len(segs) == 0:
        return _empty_segments()
    n = len(segs)
    if n < 2:
        return segs.copy().reshape(-1, 2, 2)
    _check_deadline(deadline, "merge_collinear")
    d = segs[:, 1] - segs[:, 0]
    length = np.linalg.norm(d, axis=1)
    u = d / length[:, None]
    nrm = np.stack([-u[:, 1], u[:, 0]], axis=1)
    I, J = _near_pairs(segs, None, tol)
    _check_deadline(deadline, "merge_collinear")
    if len(I):
        dj = np.abs(np.einsum("mpk,mk->mp", segs[J] - segs[I, 0][:, None, :], nrm[I]))
        ok = np.all(dj <= tol, axis=1)
        I, J = I[ok], J[ok]
    if len(I):
        di = np.abs(np.einsum("mpk,mk->mp", segs[I] - segs[J, 0][:, None, :], nrm[J]))
        ok = np.all(di <= tol, axis=1)
        I, J = I[ok], J[ok]
    if len(I):
        tj = np.einsum("mpk,mk->mp", segs[J] - segs[I, 0][:, None, :], u[I])
        ok = (tj.max(axis=1) >= -tol) & (tj.min(axis=1) <= length[I] + tol)
        I, J = I[ok], J[ok]
    lab = _components(n, I, J)
    order = np.argsort(lab, kind="stable")
    cuts = np.flatnonzero(np.diff(lab[order])) + 1
    groups = np.split(order, cuts)
    sizes = np.fromiter((len(g) for g in groups), dtype=np.int64, count=len(groups))
    out = [segs[np.concatenate([g for g, k in zip(groups, sizes) if k == 1])]] if (sizes == 1).any() else []
    two = [g for g, k in zip(groups, sizes) if k == 2]
    if two:
        pairs = np.asarray(two, dtype=np.int64).reshape(-1, 2)
        merged, ok = _merge_pairs_vectorized(segs, pairs, tol)
        out.append(merged[ok])
        out.append(segs[pairs[~ok].ravel()])
    big = [g for g, k in zip(groups, sizes) if k > 2]
    rest = []
    for c, members in enumerate(big):
        if c % 256 == 0:
            _check_deadline(deadline, "merge_collinear")
        rest.extend(_merge_group(segs, members, length, tol))
    if rest:
        out.append(np.asarray(rest, dtype=np.float64).reshape(-1, 2, 2))
    out = [o.reshape(-1, 2, 2) for o in out if len(o)]
    if not out:
        return _empty_segments()
    return np.concatenate(out).astype(np.float64)


_MERGE_MAX_TRIES = 64        # open runs tried per member in _merge_group (newest first)


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
        # cheap rejects in plain floats (this is called O(m * runs) times per group)
        px, py = float(ps[0]), float(ps[1])
        dx, dy = float(pe[0]) - px, float(pe[1]) - py
        L = math.hypot(dx, dy)
        if L <= 0.0:
            return False
        ux, uy = dx / L, dy / L
        ax, ay = float(a[0]) - px, float(a[1]) - py
        bx, by = float(b[0]) - px, float(b[1]) - py
        ta, tb = ax * ux + ay * uy, bx * ux + by * uy
        if min(ta, tb) > L + tol or max(ta, tb) < -tol:     # gap along the run's own line
            return False
        if abs(ay * ux - ax * uy) > tol or abs(by * ux - bx * uy) > tol:
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
        # newest runs first, at most _MERGE_MAX_TRIES of them: a finely faceted closed
        # curve is ONE group with hundreds of open runs, and trying them all made the
        # sweep quadratic (wheel top view: 597k try_join calls, 8.8 s). Skipping a join
        # only leaves a segment unmerged; it never changes geometry.
        for run in runs[:-_MERGE_MAX_TRIES - 1:-1]:
            if run.try_join(a, b, tol):
                run.tmax = max(run.tmax, e_ref)
                break
        else:
            runs.append(_Run(a, b, e_ref))
    out.extend(np.array([run.ps, run.pe]) for run in runs)
    return out


_LOOSE_SIN = math.sin(math.radians(15.0))   # max angle between a hidden piece and the visible edge it lies on


def _cut_intervals(a: np.ndarray, u: np.ndarray, L: float, tv: np.ndarray, tol: float) -> list:
    """Pieces of [0, L] along a + t u left after removing the intervals ``tv`` (m, 2)."""
    iv = np.sort(tv, axis=1)
    iv = iv[(iv[:, 1] > 0.0) & (iv[:, 0] < L)]
    iv = iv[np.argsort(iv[:, 0], kind="stable")]
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
    return [np.array([a + s * u, a + e * u]) for s, e in pieces if e - s > tol]


def _point_seg_dist_pairs(p: np.ndarray, sa: np.ndarray, sb: np.ndarray) -> np.ndarray:
    """Row-wise distance of points p (m, 2) to segments sa-sb (m, 2)."""
    d = sb - sa
    dd = np.einsum("mk,mk->m", d, d)
    t = np.clip(np.einsum("mk,mk->m", p - sa, d) / np.where(dd > 0, dd, 1.0), 0.0, 1.0)
    return np.linalg.norm(p - (sa + t[:, None] * d), axis=1)


def _drop_on_visible(pieces: np.ndarray, vis: np.ndarray, tol: float,
                     deadline: Optional[float] = None) -> np.ndarray:
    """Drop hidden pieces whose ends and midpoint each lie within ``tol`` of a
    near-parallel (<= 15 deg) visible segment (possibly different ones)."""
    if len(pieces) == 0 or len(vis) == 0:
        return pieces
    I, J = _near_pairs(pieces, vis, tol, max_sin=_LOOSE_SIN, deadline=deadline)
    if len(I) == 0:
        return pieces
    dh = pieces[I, 1] - pieces[I, 0]
    dv = vis[J, 1] - vis[J, 0]
    cross = np.abs(dh[:, 0] * dv[:, 1] - dh[:, 1] * dv[:, 0])
    par = cross <= _LOOSE_SIN * np.linalg.norm(dh, axis=1) * np.linalg.norm(dv, axis=1)
    I, J = I[par], J[par]
    if len(I) == 0:
        return pieces
    hit = np.zeros((len(pieces), 3), dtype=bool)
    probes = (pieces[I, 0], 0.5 * (pieces[I, 0] + pieces[I, 1]), pieces[I, 1])
    for k, p in enumerate(probes):
        near = _point_seg_dist_pairs(p, vis[J, 0], vis[J, 1]) <= tol
        hit[I[near], k] = True
    drop = hit.all(axis=1)
    if drop.any():
        log.debug("subtract_covered: dropped %d hidden pieces lying on visible edges", int(drop.sum()))
    return pieces[~drop]


def subtract_covered(hidden: np.ndarray, visible: np.ndarray, tol: float, *, loose: bool = False,
                     deadline: Optional[float] = None, min_len: Optional[float] = None) -> np.ndarray:
    """Remove the parts of hidden segments that lie on visible segments.

    A visible segment covers a hidden one where both its endpoints lie within
    ``tol`` of the hidden segment's line; its projection onto the hidden
    segment is cut out. Hidden segments (and remaining pieces) not longer
    than ``min_len`` are always dropped, whatever ``visible`` holds, and
    visible segments not longer than ``min_len`` cover nothing; every other
    hidden segment with no collinear visible segment is returned unchanged.
    ``min_len=None`` means ``min_len = tol`` (the old behaviour). Callers
    that enlarge ``tol`` beyond the geometric tolerance (the paper-scaled
    cover tolerance of the poly-HLR route, the 3 px outline cover of the
    mesh route) must pass the geometric tolerance as ``min_len``, otherwise
    every short chord of a finely faceted curve is deleted anywhere in the
    view (F1 review: 256-section r10 rim at scale 1 -> 0 hidden lines).
    Only segments are handled: a hidden Circle/Arc lying on a visible one
    (the far mouth of a through hole) is removed with
    ``drop_covered_curves`` before flattening.

    ``loose=True`` (poly-HLR and mesh routes, where the hidden and visible
    copies of one edge come from different tessellations, QA issue 27): a
    visible segment also covers the hidden one when both hidden endpoints
    lie within ``tol`` of the visible segment's line; in loose mode every
    covering pair must be near-parallel (at most 15 degrees apart), and
    afterwards every remaining piece whose two ends and midpoint each lie
    within ``tol`` of a visible segment at most 15 degrees off its direction
    is dropped. The angle guard keeps hidden lines that merely cross visible
    ones, also short chords whose two ends both lie within ``tol`` of the
    visible line.

    Candidates come from a KD-tree over points sampled along the segments
    (``_near_pairs``), O((H + V) log(H + V)) instead of the old O(H V) scan
    (QA issue 3: 128 s for 184k x 10k). ``deadline`` (a
    ``time.perf_counter()`` value) raises ``TimeoutError`` once passed.
    """
    hid = _as_segments(hidden)
    vis = _as_segments(visible)
    tol = float(tol)
    floor = tol if min_len is None else max(float(min_len), 0.0)
    if len(hid) == 0:
        return _empty_segments()
    hlen = np.linalg.norm(hid[:, 1] - hid[:, 0], axis=1)
    if not np.all(hlen > floor):
        log.debug("subtract_covered: dropped %d hidden segments not longer than %.4g",
                  int(np.count_nonzero(hlen <= floor)), floor)
        hid = hid[hlen > floor]
        if len(hid) == 0:
            return _empty_segments()
    if len(vis) == 0:
        return hid.copy()
    vlen = np.linalg.norm(vis[:, 1] - vis[:, 0], axis=1)
    vis = vis[vlen > floor]
    if len(vis) == 0:
        return hid.copy()
    _check_deadline(deadline, "subtract_covered")
    d = hid[:, 1] - hid[:, 0]
    L = np.linalg.norm(d, axis=1)
    uh = d / L[:, None]
    nh = np.stack([-uh[:, 1], uh[:, 0]], axis=1)
    Hc, Vc = _near_pairs(hid, vis, tol, max_sin=_LOOSE_SIN if loose else None, deadline=deadline)
    _check_deadline(deadline, "subtract_covered")
    if len(Hc):
        off = np.abs(np.einsum("mpk,mk->mp", vis[Vc] - hid[Hc, 0][:, None, :], nh[Hc]))
        on = np.all(off <= tol, axis=1)
        if loose:
            dv = vis[Vc, 1] - vis[Vc, 0]
            nvl = np.linalg.norm(dv, axis=1)
            nv = np.stack([-dv[:, 1], dv[:, 0]], axis=1) / nvl[:, None]
            offh = np.abs(np.einsum("mpk,mk->mp", hid[Hc] - vis[Vc, 0][:, None, :], nv))
            on |= np.all(offh <= tol, axis=1)
            # angle guard (as in _drop_on_visible): a short chord whose two ends lie
            # within tol of the visible line, or a short visible chord within tol of
            # the hidden line, crossing at a steep angle covers nothing
            dh = d[Hc]
            cross = np.abs(dh[:, 0] * dv[:, 1] - dh[:, 1] * dv[:, 0])
            on &= cross <= _LOOSE_SIN * L[Hc] * nvl
        Hc, Vc = Hc[on], Vc[on]
    covered = np.zeros(len(hid), dtype=bool)
    covered[Hc] = True
    out = [hid[~covered]]
    if len(Hc):
        o = np.argsort(Hc, kind="stable")
        Hc, Vc = Hc[o], Vc[o]
        tv_all = np.einsum("mpk,mk->mp", vis[Vc] - hid[Hc, 0][:, None, :], uh[Hc])     # (m, 2)
        cuts = np.flatnonzero(np.diff(Hc)) + 1
        starts = np.r_[0, cuts]
        stops = np.r_[cuts, len(Hc)]
        # hidden segments with ONE covering visible segment (most of them): vectorized,
        # same result as _cut_intervals (pieces [0, s) and (e, L] longer than the floor)
        single = (stops - starts) == 1
        if single.any():
            s_idx = starts[single]
            h1 = Hc[s_idx]
            iv = np.sort(tv_all[s_idx], axis=1)
            L1 = L[h1]
            hit = (iv[:, 1] > 0.0) & (iv[:, 0] < L1)
            a1 = np.where(hit, np.minimum(iv[:, 0], L1), L1)
            b0 = np.maximum(iv[:, 1], 0.0)
            k_a = np.where(hit, iv[:, 0] > 0.0, True) & (a1 > floor)
            k_b = hit & (b0 < L1) & (L1 - b0 > floor)
            p0, p1 = hid[h1, 0], uh[h1]
            first = np.stack([p0, p0 + a1[:, None] * p1], axis=1)
            second = np.stack([p0 + b0[:, None] * p1, p0 + L1[:, None] * p1], axis=1)
            out.append(first[k_a])
            out.append(second[k_b])
        pieces = []
        multi = np.flatnonzero(~single)
        for c, g in enumerate(multi):
            if c % 2048 == 0:
                _check_deadline(deadline, "subtract_covered")
            s0, s1 = int(starts[g]), int(stops[g])
            h = int(Hc[s0])
            pieces.extend(_cut_intervals(hid[h, 0], uh[h], float(L[h]), tv_all[s0:s1], floor))
        if pieces:
            out.append(np.asarray(pieces, dtype=np.float64).reshape(-1, 2, 2))
    res = np.concatenate([o.reshape(-1, 2, 2) for o in out]).astype(np.float64)
    if loose and len(res):
        _check_deadline(deadline, "subtract_covered")
        res = _drop_on_visible(res, vis, tol, deadline)
    if len(res) == 0:
        return _empty_segments()
    return res


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


def chain_to_primitives(segments: np.ndarray, style, tol: float, *, short_len: float = 0.0,
                        min_chain_len: float = 0.0, deadline: Optional[float] = None) -> list:
    """Chain segments with ``chain_paths`` and emit drawing primitives (QA issue 6).

    Hidden edges written one element per short chord never show the dash
    pattern (the dash restarts on every element), so a chain that contains
    at least two edges shorter than ``short_len`` (callers pass one dash
    length on paper, in model units) becomes ONE ``Polyline`` (``closed``
    for cycles). Other chains are emitted as separate ``Segment`` objects
    (box edges and other long lines keep their exact one-element-per-edge
    form; a single short edge between long ones cannot break the pattern of
    its neighbours); a two-point chain is always a ``Segment``. Chains (open or closed) whose total length is below
    ``min_chain_len`` are dropped (mesh noise, QA issues 13 and 24).
    ``style`` is the ``LineStyle`` of every output primitive.
    """
    from .model import Polyline, Segment

    segs = _as_segments(segments)
    if len(segs) == 0:
        return []
    opened, closed = chain_paths(segs, tol, deadline=deadline)
    out: list = []
    n_drop = 0
    for pts, is_closed in [(p, False) for p in opened] + [(p, True) for p in closed]:
        ring = np.vstack([pts, pts[:1]]) if is_closed else pts
        edge_len = np.linalg.norm(np.diff(ring, axis=0), axis=1)
        if float(edge_len.sum()) < min_chain_len:
            n_drop += 1
            continue
        if len(pts) == 2 and not is_closed:
            out.append(Segment((float(pts[0, 0]), float(pts[0, 1])),
                               (float(pts[1, 0]), float(pts[1, 1])), style))
        elif int(np.count_nonzero(edge_len < short_len)) < 2:
            out.extend(Segment((float(a[0]), float(a[1])), (float(b[0]), float(b[1])), style)
                       for a, b in zip(ring[:-1], ring[1:]))
        else:
            out.append(Polyline(np.ascontiguousarray(pts, dtype=np.float64), closed=is_closed,
                                style=style))
    if n_drop:
        log.debug("chain_to_primitives: dropped %d chains shorter than %.4g", n_drop, min_chain_len)
    return out


def _prim_ends_and_length(p):
    """(end points (2, 2), length) of an open primitive, or None for closed ones."""
    from .model import Arc, Polyline, Segment
    if isinstance(p, Segment):
        a = np.array([[float(p.a[0]), float(p.a[1])], [float(p.b[0]), float(p.b[1])]])
        return a, float(np.linalg.norm(a[1] - a[0]))
    if isinstance(p, Polyline):
        pts = np.asarray(p.points, dtype=np.float64).reshape(-1, 2)
        if p.closed or len(pts) < 2:
            return None
        return pts[[0, -1]].copy(), float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
    if isinstance(p, Arc):
        sweep = _arc_sweep(p.start_deg, p.end_deg)
        if sweep >= 360.0:
            return None
        c = np.array([float(p.center[0]), float(p.center[1])])
        r = float(p.radius)
        s, e = math.radians(float(p.start_deg)), math.radians(float(p.start_deg) + sweep)
        ends = np.array([c + r * np.array([math.cos(s), math.sin(s)]),
                         c + r * np.array([math.cos(e), math.sin(e)])])
        return ends, abs(r) * math.radians(sweep)
    return None


def drop_isolated_fragments(prims: list, min_len: float, touch_tol: float, deflection: float,
                            context: Optional[list] = None) -> Tuple[list, int]:
    """Drop short open primitives that touch nothing (QA issue 35).

    A ``Segment``, open ``Polyline`` or ``Arc`` shorter than ``min_len``
    whose two end points both lie farther than ``touch_tol`` from every
    OTHER primitive of ``prims`` and ``context`` (flattened at
    ``deflection``; interiors count, so a stub ending on a line is kept) is
    an isolated fragment and removed. Closed items, Circles and anything
    longer are always kept. Returns (kept primitives, number dropped).
    """
    prims = list(prims)
    if min_len <= 0.0 or not prims:
        return prims, 0
    context = list(context or [])
    cand = []
    for i, p in enumerate(prims):
        el = _prim_ends_and_length(p)
        if el is not None and el[1] < min_len:
            cand.append((i, el[0]))
    if not cand:
        return prims, 0
    from .model import Segment

    allp = prims + context
    chunks, owner = [], []
    seg_k = [k for k, p in enumerate(allp) if type(p) is Segment]
    if seg_k:                                       # plain Segments in bulk (most of a view)
        chunks.append(np.array([[allp[k].a, allp[k].b] for k in seg_k], dtype=np.float64).reshape(-1, 2, 2))
        owner.append(np.asarray(seg_k, dtype=np.int64))
    for k, p in enumerate(allp):
        if type(p) is Segment:
            continue
        try:
            s = primitives_to_segments([p], deflection)
        except (TypeError, ValueError):
            continue
        if len(s):
            chunks.append(s)
            owner.append(np.full(len(s), k, dtype=np.int64))
    if not chunks:
        return prims, 0
    flat = np.concatenate(chunks)
    own = np.concatenate(owner)
    ends = np.array([e for _, e in cand], dtype=np.float64).reshape(-1, 2)    # (2c, 2)
    pseg = np.stack([ends, ends], axis=1)                                    # zero-length probes
    I, J = _near_pairs(pseg, flat, touch_tol)
    probe_owner = np.repeat(np.array([i for i, _ in cand], dtype=np.int64), 2)
    other = own[J] != probe_owner[I]
    I, J = I[other], J[other]
    touched = np.zeros(len(pseg), dtype=bool)
    if len(I):
        dist = _point_seg_dist_pairs(ends[I], flat[J, 0], flat[J, 1])
        touched[I[dist <= touch_tol]] = True
    touched = touched.reshape(-1, 2)
    drop = {cand[k][0] for k in range(len(cand)) if not touched[k].any()}
    if drop:
        log.debug("drop_isolated_fragments: dropped %d fragments shorter than %.4g", len(drop), min_len)
    return [p for i, p in enumerate(prims) if i not in drop], len(drop)


__all__: Sequence[str] = (
    "simplify_polyline", "is_straight", "polygon_area", "point_in_polygon", "nest_loops",
    "hatch_lines", "chain_segments", "chain_paths", "merge_collinear", "premerge_collinear",
    "subtract_covered", "dedupe_primitives", "drop_covered_curves", "primitives_to_segments",
    "chain_to_primitives", "drop_isolated_fragments",
)
