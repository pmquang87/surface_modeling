"""Orthographic views of a triangle mesh with a numpy depth buffer.

The mesh route's answer to OCCT HLR (spec 3.5): per view, the vertices go to
view coordinates with the shared ``ViewSpec`` (x = Vx, y = N x Vx, depth =
N.p, larger = nearer the viewer), a scanline depth buffer is filled, and
every candidate edge (sharp + boundary + silhouette) is sampled about one
pixel apart and split into visible and hidden runs. Organic meshes take
their outline from the coverage mask instead of silhouette edges.

Measured facts this module relies on (mesh probe, REPORT-mesh.md):

* The fully vectorised scanline rasterizer ("span") gives exactly the same
  coverage as a per-triangle loop (dCov = 0, dZ <= 7e-13) and costs
  0.05-0.65 s at N = 1024 for 12 to 328k triangles. Pixel centres sit at
  integer + 0.5; a pixel is covered when its centre lies inside or on the
  triangle (closed rule).
* Visibility test: a sample is visible when its depth is within ``tol_px``
  (1.5 px) of the MINIMUM of the depth buffer over a 3x3 window. A
  single-pixel test calls outline-coincident edges hidden (a box's back
  edges behind its front edges, hole rims on the outline): 2-54 % wrong.
  With the 3x3 window 0.00-0.14 % of samples disagree with a jittered
  ray-cast truth on CAD meshes; tolerances 0.5-3 px give identical results.
* Two-box occlusion: the back box edge x = 1 split at z = -2.992 / 2.976
  (expected -3 / 3, 1 px = 0.012).
* Silhouette edges (adjacent faces on opposite sides of N) are exact on
  convex meshes but fragment on noisy blobs (254 silhouette edges -> 252
  visible + 172 hidden runs). Organic meshes therefore use
  ``skimage.measure.find_contours`` on the padded coverage mask +
  ``approximate_polygon(0.75 px)``: closed outer and inner loops (torus
  hole), error <= 1 px. The first ``approximate_polygon`` call in a process
  costs 0.4-2.9 s; it is done lazily and its time is recorded in
  ``ProjectedView.stats["outline_s"]``.
* Axis views produce exactly coincident projected segments (4 duplicates for
  a box) and end-on edges of zero projected length; both are removed (merge
  of collinear runs, ``subtract_covered`` for hidden runs under visible
  ones). Edge-on rims of fine tessellations put thousands of exactly
  collinear pieces on one line; ``premerge_collinear`` unions them in
  O(n log n) first (measured: 2000-section cylinder side view 5.1 s -> 2.4 s
  per view, the rest is the raster; under the profiler ``merge_collinear``
  alone spent 5.7 s of 7.9 s on it).
* A face is front-facing when ``dot(normal, spec.n) > 1e-9`` (the probe's
  ``n.d < -1e-9`` with d = -N); edge-on faces count as not front-facing, so
  the edge between a front face and an edge-on face is a silhouette.

No OCP, shapely or matplotlib import. No output to stdout.
"""
from __future__ import annotations

import logging
import math
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .geometry2d import (chain_to_primitives, dedupe_primitives, merge_collinear, weld_chain_ends,
                         premerge_collinear, subtract_covered)
from .model import BBox2D, LineStyle, Polyline, ProjectedView, Segment

log = logging.getLogger(__name__)

FRONT_EPS = 1e-9
CHUNK_PX = 8_000_000              # bound on per-chunk pixel arrays in raster_span
CLEANUP_TOL_PX = 0.5              # 2D clean-up tolerance (merge, subtract, dedupe), in pixels
_OUTLINE_COVER_PX = 3.0           # organic: runs within this of the mask outline are the outline
_MIN_CHAIN_PX = 2.0               # chains shorter than this are noise (QA issue 24)
_ORGANIC_MIN_CHAIN_PAPER_MM = 3.0  # organic meshes: shorter chains are dropped (QA issues 13, 24)
_DASH_PERIOD_PAPER_MM = 3.0       # one hidden dash: chains with >= 2 shorter chords -> Polyline
_ORGANIC_MIN_LOOP_PAPER_MM = 1.0  # organic meshes: closed loops shorter than this are dropped (QA2 #25)
_WELD_PAPER_MM = 0.3              # dangling chain ends this close are joined before chaining (QA2 #19)
_SIMPLIFY_PAPER_MM = 0.05         # Douglas-Peucker tolerance of the chains (QA2 #19)
_HAIRPIN_PAPER_MM = 0.5           # hairpin vertices with an adjacent edge shorter than this go (QA2 #19)
_SLIVER_REL = 1e-5                # faces with a min altitude below this x bbox diagonal: no normal (QA2 #19)
_DEFAULT_VIEW_PAPER_MM = 120.0    # paper-scale guess: the view's long side spans this on paper
_MAX_ORGANIC_SILHOUETTE_FACES = 2_000_000   # above: organic silhouettes skipped and disclosed
_SKIMAGE_WARM = {"done": False}


# ------------------------------------------------------------------ raster

def _make_grid(xy: np.ndarray, n_px: int, margin_px: int) -> dict:
    lo = xy.min(axis=0)
    hi = xy.max(axis=0)
    span = float(max(hi - lo))
    if not span > 0.0:
        span = 1.0
    usable = int(n_px) - 2 * int(margin_px)
    if usable < 2:
        raise ValueError(f"raster_span: n_px {n_px} too small for margin {margin_px}")
    scale = usable / span                      # pixels per model mm
    px = 1.0 / scale
    W = int(math.ceil((hi[0] - lo[0]) * scale)) + 2 * margin_px
    H = int(math.ceil((hi[1] - lo[1]) * scale)) + 2 * margin_px
    return {"x0": float(lo[0] - margin_px * px), "y0": float(lo[1] - margin_px * px),
            "px": float(px), "W": int(max(W, 1)), "H": int(max(H, 1)), "margin_px": int(margin_px)}


def _to_px(pts: np.ndarray, grid: dict) -> np.ndarray:
    """View (x, y[, depth]) -> pixel units (pixel c spans [c, c + 1), centre c + 0.5)."""
    q = np.array(pts, dtype=np.float64, copy=True)
    q[..., 0] = (q[..., 0] - grid["x0"]) / grid["px"]
    q[..., 1] = (q[..., 1] - grid["y0"]) / grid["px"]
    return q


def _px_to_view(cr: np.ndarray, grid: dict) -> np.ndarray:
    return np.stack([grid["x0"] + cr[..., 0] * grid["px"], grid["y0"] + cr[..., 1] * grid["px"]], axis=-1)


def _raster_triangles(buf: np.ndarray, T: np.ndarray, chunk_px: int = CHUNK_PX) -> None:
    """Scanline fill of triangles T (k, 3, 3) in pixel units into ``buf`` (max depth).

    The probe's 'span' variant: rows whose centre lies in the triangle's
    y-range, the x-interval at the row centre from the three edges, every
    covered pixel centre gets the plane depth (clipped to the triangle's
    depth range) through ``np.maximum.at``.
    """
    H, W = buf.shape
    flat = buf.ravel()
    a, b, c = T[:, 0], T[:, 1], T[:, 2]
    A = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    ok = np.abs(A) > 1e-12
    T = np.ascontiguousarray(T[ok])
    A = A[ok]
    if len(T) == 0:
        return
    x0, y0, z0 = T[:, 0, 0], T[:, 0, 1], T[:, 0, 2]
    x1, y1, z1 = T[:, 1, 0], T[:, 1, 1], T[:, 1, 2]
    x2, y2, z2 = T[:, 2, 0], T[:, 2, 1], T[:, 2, 2]
    dzdx = ((z1 - z0) * (y2 - y0) - (z2 - z0) * (y1 - y0)) / A
    dzdy = ((x1 - x0) * (z2 - z0) - (x2 - x0) * (z1 - z0)) / A
    zlo = T[:, :, 2].min(axis=1)
    zhi = T[:, :, 2].max(axis=1)
    r0 = np.clip(np.ceil(T[:, :, 1].min(axis=1) - 0.5), 0, H).astype(np.int64)
    r1 = np.clip(np.floor(T[:, :, 1].max(axis=1) - 0.5), -1, H - 1).astype(np.int64)
    nrow = np.maximum(r1 - r0 + 1, 0)
    est = nrow * (np.ceil(T[:, :, 0].max(axis=1) - T[:, :, 0].min(axis=1)) + 1)
    csum = np.cumsum(est)
    if csum[-1] <= 0:
        return
    bounds = np.searchsorted(csum, np.arange(chunk_px, csum[-1] + chunk_px, chunk_px))
    starts = np.r_[0, np.unique(np.clip(bounds, 1, len(T)))]
    if starts[-1] != len(T):
        starts = np.r_[starts, len(T)]
    for sa, sb in zip(starts[:-1], starts[1:]):
        nr = nrow[sa:sb]
        ti = np.repeat(np.arange(sa, sb), nr)
        if len(ti) == 0:
            continue
        rs = np.arange(len(ti)) - np.repeat(np.cumsum(nr) - nr, nr) + np.repeat(r0[sa:sb], nr)
        yc = rs + 0.5
        t = T[ti]
        xs = np.full((len(ti), 3), np.nan)
        for k, (p, q) in enumerate(((0, 1), (1, 2), (2, 0))):
            py, qy = t[:, p, 1], t[:, q, 1]
            pxx, qx = t[:, p, 0], t[:, q, 0]
            cross = (np.minimum(py, qy) <= yc) & (yc <= np.maximum(py, qy)) & (py != qy)
            with np.errstate(divide="ignore", invalid="ignore"):
                xk = pxx + (yc - py) * (qx - pxx) / (qy - py)
            xs[cross, k] = xk[cross]
        good = np.isfinite(xs).sum(axis=1) >= 1
        # rows without any crossing become all-zero rows (no all-NaN slice); ncol masks them
        xl = np.nanmin(np.where(good[:, None], xs, 0.0), axis=1)
        xr = np.nanmax(np.where(good[:, None], xs, 0.0), axis=1)
        c0 =np.clip(np.ceil(xl - 0.5), 0, W).astype(np.int64)
        c1 = np.clip(np.floor(xr - 0.5), -1, W - 1).astype(np.int64)
        ncol = np.where(good, np.maximum(c1 - c0 + 1, 0), 0)
        pr = np.repeat(np.arange(len(ti)), ncol)
        if len(pr) == 0:
            continue
        cs = np.arange(len(pr)) - np.repeat(np.cumsum(ncol) - ncol, ncol) + np.repeat(c0, ncol)
        tt = ti[pr]
        rr = rs[pr]
        z = z0[tt] + dzdx[tt] * (cs + 0.5 - x0[tt]) + dzdy[tt] * (rr + 0.5 - y0[tt])
        z = np.clip(z, zlo[tt], zhi[tt])
        np.maximum.at(flat, rr * W + cs, z)


def raster_span(xy: np.ndarray, depth: np.ndarray, faces: np.ndarray, n_px: int,
                margin_px: int = 4) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Scanline depth buffer of a triangle mesh in view coordinates.

    ``xy`` (n, 2) view x/y of the vertices, ``depth`` (n,) their depth
    (larger = nearer), ``faces`` (k, 3). The long side of the vertex bbox
    spans ``n_px - 2 * margin_px`` pixels. Returns ``(depth_buffer, mask,
    grid)``: buffer (H, W) with -inf where empty, row index growing with
    view y; mask = covered pixels; grid ``{x0, y0, px, W, H, margin_px}``
    where pixel (r, c) spans x in [x0 + c px, x0 + (c+1) px] and y likewise.
    """
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    depth = np.asarray(depth, dtype=np.float64).reshape(-1)
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if len(xy) == 0:
        raise ValueError("raster_span: no vertices")
    grid = _make_grid(xy, int(n_px), int(margin_px))
    buf = np.full((grid["H"], grid["W"]), -np.inf)
    if len(faces):
        pts = _to_px(np.column_stack([xy, depth]), grid)
        _raster_triangles(buf, pts[faces])
    return buf, np.isfinite(buf), grid


def fill_loops_depth(buf: np.ndarray, grid: dict, loops: Sequence[np.ndarray], depth: float) -> int:
    """Raise ``buf`` to ``depth`` at every pixel centre inside the loops (even-odd).

    Used for the cut face of a section body: the cap lies in the section
    plane, which is perpendicular to the view direction, so its depth is
    constant. Half-open crossing rule per row. Returns the number of pixels
    inside the loops.
    """
    H, W = buf.shape
    e0, e1 = [], []
    for L in loops:
        q = _to_px(np.asarray(L, dtype=np.float64).reshape(-1, 2), grid)
        if len(q) >= 2 and np.array_equal(q[0], q[-1]):
            q = q[:-1]
        if len(q) < 3:
            continue
        e0.append(q)
        e1.append(np.roll(q, -1, axis=0))
    if not e0:
        return 0
    a = np.concatenate(e0)
    b = np.concatenate(e1)
    ylo, yhi = np.minimum(a[:, 1], b[:, 1]), np.maximum(a[:, 1], b[:, 1])
    r_start = max(int(math.ceil(float(ylo.min()) - 0.5)), 0)
    r_end = min(int(math.floor(float(yhi.max()) - 0.5)), H - 1)
    n_in = 0
    for r in range(r_start, r_end + 1):
        yc = r + 0.5
        m = (ylo <= yc) & (yc < yhi)
        if not m.any():
            continue
        t = (yc - a[m, 1]) / (b[m, 1] - a[m, 1])
        xs = np.sort(a[m, 0] + t * (b[m, 0] - a[m, 0]))
        if len(xs) % 2:
            log.debug("fill_loops_depth: odd crossing count at row %d, row skipped", r)
            continue
        for xa, xb in zip(xs[0::2], xs[1::2]):
            c0 = max(int(math.ceil(xa - 0.5)), 0)
            c1 = min(int(math.ceil(xb - 0.5)) - 1, W - 1)     # centre in [xa, xb)
            if c1 >= c0:
                np.maximum(buf[r, c0:c1 + 1], depth, out=buf[r, c0:c1 + 1])
                n_in += c1 - c0 + 1
    return n_in


# -------------------------------------------------------------- visibility

def _min_filter(buf: np.ndarray, nbhd: int) -> np.ndarray:
    if nbhd <= 0:
        return buf
    H, W = buf.shape
    pad = np.pad(buf, nbhd, constant_values=-np.inf)
    out = pad[0:H, 0:W].copy()
    for dy in range(2 * nbhd + 1):
        for dx in range(2 * nbhd + 1):
            np.minimum(out, pad[dy:dy + H, dx:dx + W], out=out)
    return out


def _hysteresis(vis: np.ndarray, rep: np.ndarray) -> np.ndarray:
    """Absorb single-sample state flips inside one segment (QA issue 24).

    A sample whose two neighbours (same segment) agree with each other and
    not with it takes their state; an end sample takes the state of the next
    two when those agree with each other and not with it. A run therefore
    keeps its state for at least two samples before switching.
    """
    v = vis.copy()
    n = len(v)
    if n < 3:
        return v
    same_prev = np.zeros(n, dtype=bool)
    same_prev[1:] = rep[1:] == rep[:-1]
    same_next = np.zeros(n, dtype=bool)
    same_next[:-1] = rep[:-1] == rep[1:]
    prev = np.r_[v[:1], v[:-1]]
    nxt = np.r_[v[1:], v[-1:]]
    inner = same_prev & same_next & (prev == nxt) & (v != prev)
    nxt2 = np.r_[v[2:], v[-1:], v[-1:]]
    same_next2 = np.zeros(n, dtype=bool)
    same_next2[:-2] = rep[:-2] == rep[2:]
    first = ~same_prev & same_next & same_next2 & (nxt == nxt2) & (v != nxt)
    prev2 = np.r_[v[:1], v[:1], v[:-2]]
    same_prev2 = np.zeros(n, dtype=bool)
    same_prev2[2:] = rep[2:] == rep[:-2]
    last = ~same_next & same_prev & same_prev2 & (prev == prev2) & (v != prev)
    flip = inner | first | last
    v[flip] = ~v[flip]
    return v


def edge_visibility(segments_view: np.ndarray, depth_buffer: np.ndarray, grid: dict,
                    tol_px: float = 1.5, nbhd: int = 1,
                    hysteresis: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Split 3D view-space segments into visible and hidden 2D runs.

    ``segments_view`` (k, 2, 3) holds view columns [x, y, depth]. Each
    segment is sampled about one pixel apart; a sample is visible when its
    depth >= (minimum of the buffer over a (2 nbhd + 1)^2 window) - tol_px *
    px. With ``hysteresis`` (default) a single sample of the other state
    inside a run is absorbed, so a crease near the threshold does not
    flicker into one-pixel runs (QA issue 24). Runs break halfway between
    two samples of different state. Segments seen end-on (projected length
    below 1e-6 px) are dropped. Returns ``(visible (m, 2, 2), hidden (h, 2,
    2))`` in view coordinates.
    """
    segs = np.asarray(segments_view, dtype=np.float64)
    empty = np.zeros((0, 2, 2))
    if segs.size == 0:
        return empty, empty.copy()
    segs = segs.reshape(-1, 2, 3)
    H, W = depth_buffer.shape
    px = float(grid["px"])
    tol = float(tol_px) * px
    bmin = _min_filter(depth_buffer, int(nbhd))
    A, B = segs[:, 0], segs[:, 1]
    lpx = np.linalg.norm((B[:, :2] - A[:, :2]) / px, axis=1)
    keep = lpx > 1e-6
    A, B, lpx = A[keep], B[keep], lpx[keep]
    if len(A) == 0:
        return empty, empty.copy()
    ns = np.maximum(2, np.ceil(lpx).astype(np.int64) + 1)
    rep = np.repeat(np.arange(len(A)), ns)
    starts = np.r_[0, np.cumsum(ns)[:-1]]
    k = np.arange(len(rep)) - np.repeat(starts, ns)
    t = k / np.repeat(ns - 1, ns)
    P = A[rep] + (B[rep] - A[rep]) * t[:, None]
    q = _to_px(P, grid)
    c = np.clip(np.floor(q[:, 0]).astype(np.int64), 0, W - 1)
    r = np.clip(np.floor(q[:, 1]).astype(np.int64), 0, H - 1)
    vis = P[:, 2] >= bmin[r, c] - tol
    if hysteresis:
        vis = _hysteresis(vis, rep)
    # run boundaries: segment starts and state changes inside a segment
    same_seg = rep[1:] == rep[:-1]
    change = same_seg & (vis[1:] != vis[:-1])
    run_start = np.zeros(len(rep), dtype=bool)
    run_start[starts] = True
    run_start[1:][change] = True
    rs_idx = np.nonzero(run_start)[0]
    re_idx = np.r_[rs_idx[1:] - 1, len(rep) - 1]
    seg_of = rep[rs_idx]
    t_lo = np.where(np.isin(rs_idx, starts), 0.0, 0.5 * (t[np.maximum(rs_idx - 1, 0)] + t[rs_idx]))
    ends = starts + ns - 1
    is_end = np.isin(re_idx, ends)
    t_hi = np.where(is_end, 1.0, 0.5 * (t[re_idx] + t[np.minimum(re_idx + 1, len(t) - 1)]))
    a2, b2 = A[seg_of, :2], B[seg_of, :2]
    p0 = a2 + (b2 - a2) * t_lo[:, None]
    p1 = a2 + (b2 - a2) * t_hi[:, None]
    runs = np.stack([p0, p1], axis=1)
    state = vis[rs_idx]
    return runs[state], runs[~state]


# ----------------------------------------------------------------- outline

def mask_outline(mask: np.ndarray, grid: dict, tol_px: float = 0.75,
                 smooth_px: float = 0.7) -> List[np.ndarray]:
    """Closed outline loops of a coverage mask, in view coordinates.

    ``skimage.measure.find_contours`` on the mask padded by one empty pixel
    (so every contour closes) at level 0.5, simplified with
    ``approximate_polygon(tol_px)``. With ``smooth_px > 0`` the mask is
    first blurred with a Gaussian of that sigma (pixels), so the level-0.5
    contour runs at sub-pixel positions instead of along the pixel
    staircase (QA issue 24; corners round off by about one pixel). Contour
    points sit on the pixel-centre lattice (half a pixel inside the covered
    pixels' outer edges). Outer boundaries and holes both come back; loops
    have no repeated end point and at least 3 points.
    """
    from skimage import measure

    m = np.asarray(mask, dtype=float)
    if m.size == 0 or not m.any():
        return []
    padded = np.pad(m, 2 if smooth_px and smooth_px > 0 else 1)
    if smooth_px and smooth_px > 0:
        from scipy.ndimage import gaussian_filter
        padded = gaussian_filter(padded, float(smooth_px), mode="constant", cval=0.0)
    off = 2.0 if smooth_px and smooth_px > 0 else 1.0
    contours = measure.find_contours(padded, 0.5)
    loops: List[np.ndarray] = []
    for cont in contours:
        if tol_px and tol_px > 0:
            cont = measure.approximate_polygon(cont, tolerance=float(tol_px))
        rc = cont - off                                   # undo the pad
        pxc = np.stack([rc[:, 1] + 0.5, rc[:, 0] + 0.5], axis=-1)
        loop = _px_to_view(pxc, grid)
        if len(loop) >= 2 and np.allclose(loop[0], loop[-1]):
            loop = loop[:-1]
        if len(loop) >= 3:
            loops.append(np.ascontiguousarray(loop))
    _SKIMAGE_WARM["done"] = True
    return loops


# ----------------------------------------------------------- edge classes

def candidate_edges(mesh, n: np.ndarray, feature_angle_deg: float,
                    silhouettes: bool = True) -> Tuple[np.ndarray, dict]:
    """Vertex index pairs (k, 2) of the edges worth drawing for view normal ``n``.

    sharp (dihedral angle > feature angle) + boundary (+ silhouette when
    ``silhouettes``: adjacent faces on opposite sides, front-facing meaning
    ``dot(normal, n) > 1e-9``). Returns (unique sorted pairs, counts).

    QA round 2 issue 19: an adjacency pair is not classified (neither sharp
    nor silhouette) when one of its faces is degenerate - smallest altitude
    (2 x area / longest edge) below ``_SLIVER_REL`` (1e-5) x the bbox
    diagonal - because such a face has no reliable normal and gives
    spurious sharp/silhouette wedges (``counts["sliver_pairs"]``). The
    triage's shape test (inradius / longest edge < 0.02) is NOT used: it
    flags every face of a 256-section cylinder (ratio 0.004, their normals
    are exact) and 360 of the 1534 sharp edges of NIST FTC 08, median
    dihedral 90 degrees (real rims).
    """
    import trimesh

    adj = np.asarray(mesh.face_adjacency)
    adj_e = np.asarray(mesh.face_adjacency_edges)
    counts = {"sharp": 0, "silhouette": 0, "boundary": 0, "sliver_pairs": 0}
    parts = []
    if len(adj):
        ang = np.asarray(mesh.face_adjacency_angles, dtype=float)
        sharp = ang > math.radians(float(feature_angle_deg))
        bad = _degenerate_faces(mesh)
        if bad is not None and bad.any():
            unreliable = bad[adj[:, 0]] | bad[adj[:, 1]]
            counts["sliver_pairs"] = int(unreliable.sum())
        else:
            unreliable = None
        sel = sharp.copy()
        counts["sharp"] = int(sharp.sum())
        if silhouettes:
            s = np.asarray(mesh.face_normals, dtype=float) @ np.asarray(n, dtype=float)
            front = s > FRONT_EPS
            sil = front[adj[:, 0]] != front[adj[:, 1]]
            if unreliable is not None:
                sil &= ~unreliable
            counts["silhouette"] = int(sil.sum())
            sel |= sil
        if unreliable is not None:
            sel &= ~unreliable
            counts["sharp"] = int((sharp & ~unreliable).sum())
        parts.append(adj_e[sel])
    es = np.asarray(mesh.edges_sorted)
    if len(es):
        bnd = es[trimesh.grouping.group_rows(es, require_count=1)]
        counts["boundary"] = int(len(bnd))
        if len(bnd):
            parts.append(bnd)
    parts = [p.reshape(-1, 2) for p in parts if len(p)]
    if not parts:
        return np.zeros((0, 2), dtype=np.int64), counts
    cand = np.unique(np.sort(np.vstack(parts), axis=1), axis=0)
    return cand.astype(np.int64), counts


def _degenerate_faces(mesh) -> Optional[np.ndarray]:
    """Bool per face: smallest altitude below ``_SLIVER_REL`` x the bbox diagonal."""
    try:
        v = np.asarray(mesh.vertices, dtype=np.float64)[np.asarray(mesh.faces, dtype=np.int64)]
        if len(v) == 0:
            return None
        e = np.linalg.norm(v[:, [1, 2, 0]] - v, axis=2).max(axis=1)
        area = np.asarray(mesh.area_faces, dtype=np.float64)
        diag = float(np.linalg.norm(np.ptp(np.asarray(mesh.vertices, dtype=np.float64), axis=0)))
        if not diag > 0.0:
            return None
        hmin = 2.0 * area / np.where(e > 0.0, e, 1.0)
        return hmin < _SLIVER_REL * diag
    except Exception as exc:  # the classification just runs unfiltered
        log.debug("degenerate-face test failed: %s", exc)
        return None


# --------------------------------------------------------------- projection

def _clean_segments(segs: np.ndarray, tol: float, deadline: Optional[float] = None) -> np.ndarray:
    return merge_collinear(segs, tol, deadline=deadline)          # premerges exact runs itself


def _segments_to_prims(segs: np.ndarray, style: LineStyle) -> list:
    return [Segment((float(s[0, 0]), float(s[0, 1])), (float(s[1, 0]), float(s[1, 1])), style)
            for s in segs]


def _loop_edges(loops) -> np.ndarray:
    chunks = [np.stack([L, np.roll(L, -1, axis=0)], axis=1) for L in loops if len(L) >= 2]
    return np.concatenate(chunks) if chunks else np.zeros((0, 2, 2))


def _project(mesh, spec, *, raster_px: int, feature_angle_deg: float, organic: bool,
             hidden: bool = True, cap: Optional[Tuple[Sequence[np.ndarray], float]] = None,
             exclude_plane: Optional[Tuple[np.ndarray, np.ndarray, float]] = None,
             grid_points: Optional[np.ndarray] = None, paper_scale: Optional[float] = None,
             time_budget_s: Optional[float] = None) -> ProjectedView:
    """Shared body of ``project_mesh`` and the section body projection.

    ``cap`` = (loops in view coords, depth) is filled into the depth buffer
    before the visibility test (the cut face of a section body).
    ``exclude_plane`` = (point, unit normal, tol) drops candidate edges whose
    two endpoints lie on that plane (the cut outline, drawn from the section
    loops instead). ``grid_points`` (m, 2) extra view points that the raster
    must cover. ``paper_scale`` / ``time_budget_s``: see ``project_mesh``.
    """
    t_start = time.perf_counter()
    deadline = None if time_budget_s is None else t_start + float(time_budget_s)
    stats: Dict[str, object] = {"raster_px": int(raster_px), "organic": bool(organic),
                                "hidden_dropped": False, "interior_silhouettes_omitted": False}
    verts = np.array(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if len(verts) == 0 or len(faces) == 0:
        stats.update(seconds=time.perf_counter() - t_start, empty=True)
        return ProjectedView(spec=spec, visible=[], hidden=[], bbox=None, algorithm="depth_buffer",
                             stats=stats)
    V = spec.to_view(verts)
    xy = V[:, :2]
    if grid_points is not None and len(grid_points):
        grid_src = np.vstack([xy, np.asarray(grid_points, dtype=np.float64).reshape(-1, 2)])
    else:
        grid_src = xy
    t0 = time.perf_counter()
    grid = _make_grid(grid_src, int(raster_px), 4)
    buf = np.full((grid["H"], grid["W"]), -np.inf)
    _raster_triangles(buf, _to_px(V, grid)[faces])
    if cap is not None:
        loops, cap_depth = cap
        stats["cap_px"] = fill_loops_depth(buf, grid, loops, float(cap_depth))
    mask = np.isfinite(buf)
    stats["raster_s"] = time.perf_counter() - t0
    stats.update(px=grid["px"], W=grid["W"], H=grid["H"])
    px = float(grid["px"])
    span = float(np.max(xy.max(axis=0) - xy.min(axis=0))) or 1.0
    scale = (float(paper_scale) if paper_scale is not None and float(paper_scale) > 0.0
             else _DEFAULT_VIEW_PAPER_MM / span)
    stats["paper_scale"] = scale

    t0 = time.perf_counter()
    sil = True
    if organic and len(faces) > _MAX_ORGANIC_SILHOUETTE_FACES:
        sil = False                       # silhouettes of huge scans: too costly, disclosed below
        stats["interior_silhouettes_omitted"] = True
    cand, counts = candidate_edges(mesh, spec.n, feature_angle_deg, silhouettes=sil)
    if exclude_plane is not None and len(cand):
        p0, nrm, ptol = exclude_plane
        dist = np.abs((verts - np.asarray(p0, float)) @ np.asarray(nrm, float))
        on = dist <= ptol
        drop = on[cand[:, 0]] & on[cand[:, 1]]
        stats["cut_edges_dropped"] = int(drop.sum())
        cand = cand[~drop]
    stats["candidates"] = int(len(cand))
    stats.update({f"n_{k}": v for k, v in counts.items()})
    vis, hid = edge_visibility(V[cand], buf, grid, tol_px=1.5, nbhd=1)
    tol = CLEANUP_TOL_PX * px

    loops: list = []
    if organic:
        t1 = time.perf_counter()
        cold = not _SKIMAGE_WARM["done"]
        loops = mask_outline(mask, grid)
        stats["outline_s"] = time.perf_counter() - t1
        stats["outline_warmup"] = bool(cold)
        stats["outline_loops"] = len(loops)
        stats["outline_vertices"] = int(sum(len(L) for L in loops))
    outline_edges = _loop_edges(loops)
    if len(outline_edges) and len(vis):
        # silhouette/sharp runs on the coverage outline are drawn by the outline itself
        # (length floor = the 0.5 px clean-up tol, not the 3 px cover tol: runs of a fine
        # tessellation are shorter than 3 px and were all deleted, F1 review)
        vis = subtract_covered(vis, outline_edges, _OUTLINE_COVER_PX * px, loose=True, min_len=tol)
    vis = _clean_segments(vis, tol)
    short_len = _DASH_PERIOD_PAPER_MM / scale
    min_chain = max(_MIN_CHAIN_PX * px, (_ORGANIC_MIN_CHAIN_PAPER_MM / scale) if organic else 0.0)
    min_loop = max(_MIN_CHAIN_PX * px, (_ORGANIC_MIN_LOOP_PAPER_MM / scale) if organic else 0.0)
    min_loop = min(min_loop, min_chain)
    weld_r = max(tol, _WELD_PAPER_MM / scale)
    chain_kw = dict(short_len=short_len, min_chain_len=min_chain, min_closed_len=min_loop,
                    simplify_tol=max(tol, _SIMPLIFY_PAPER_MM / scale),
                    hairpin_len=_HAIRPIN_PAPER_MM / scale)
    stats["min_chain_len"] = min_chain
    stats["min_loop_len"] = min_loop
    cst: dict = {}
    vis, n_weld = weld_chain_ends(vis, tol, weld_r)
    visible = chain_to_primitives(vis, LineStyle.VISIBLE, tol, stats=cst, **chain_kw)
    hidden_prims: list = []
    if hidden and len(hid):
        try:
            _check(deadline)
            hid = premerge_collinear(hid, tol)
            cover = np.concatenate([vis, outline_edges]) if len(outline_edges) else vis
            hid = subtract_covered(hid, cover, tol, loose=True, deadline=deadline)
            if len(outline_edges) and len(hid):
                hid = subtract_covered(hid, outline_edges, _OUTLINE_COVER_PX * px, loose=True,
                                       deadline=deadline, min_len=tol)
            hid = _clean_segments(hid, tol, deadline)
            hid, nw_h = weld_chain_ends(hid, tol, weld_r)
            n_weld += nw_h
            hidden_prims = chain_to_primitives(hid, LineStyle.HIDDEN, tol, deadline=deadline,
                                               stats=cst, **chain_kw)
        except TimeoutError:
            hidden_prims = []
            stats["hidden_dropped"] = True
            stats["hidden_dropped_reason"] = "time_budget"
    stats["welded_ends"] = int(n_weld)
    stats["closed_loops_dropped"] = int(cst.get("closed_dropped", 0))
    stats["dropped_chains"] = int(cst.get("open_dropped", 0) + cst.get("closed_dropped", 0))
    stats["hairpins_removed"] = int(cst.get("hairpins_removed", 0))
    visible = dedupe_primitives(visible, tol)
    hidden_prims = dedupe_primitives(hidden_prims, tol) if hidden else []
    stats["visibility_s"] = time.perf_counter() - t0
    if loops:
        visible = [Polyline(L, closed=True, style=LineStyle.VISIBLE) for L in loops] + visible

    bbox = BBox2D.from_points(xy)
    stats["n_visible"] = len(visible)
    stats["n_hidden"] = len(hidden_prims)
    stats["seconds"] = time.perf_counter() - t_start
    log.debug("project_mesh %s: %d visible, %d hidden, %.3f s", spec.name, len(visible),
              len(hidden_prims), stats["seconds"])
    return ProjectedView(spec=spec, visible=visible, hidden=hidden_prims, bbox=bbox,
                         algorithm="depth_buffer", stats=stats)


def _check(deadline: Optional[float]) -> None:
    if deadline is not None and time.perf_counter() > deadline:
        raise TimeoutError("mesh view: time budget exceeded")


def project_mesh(mesh, spec, *, raster_px: int, feature_angle_deg: float, organic: bool,
                 hidden: bool = True, paper_scale: Optional[float] = None,
                 time_budget_s: Optional[float] = None) -> ProjectedView:
    """Visible/hidden line view of a triangle mesh (``algorithm="depth_buffer"``).

    Sharp + boundary + silhouette edges are split by the depth buffer into
    visible and hidden runs (hidden runs lying on visible ones removed,
    collinear runs merged, duplicates dropped) and chained: chains holding
    two or more chords shorter than one hidden dash (3 mm on paper) become Polylines,
    chains of long edges stay Segments (QA issue 6), and chains shorter than
    2 px are dropped (noise, QA issue 24). Organic meshes add the outline
    loops of the coverage mask as closed Polylines; their silhouette runs on
    that outline are removed, the interior and hidden silhouettes (through
    holes, overlapping lobes) are kept when their chain is at least
    ``_ORGANIC_MIN_CHAIN_PAPER_MM`` (3 mm) long on paper (QA issue 13). The
    outline subtraction uses a 3 px cover tolerance but only the 0.5 px
    clean-up tolerance as its length floor, so the short runs of a fine
    tessellation survive until they are chained. For organic meshes above
    ``_MAX_ORGANIC_SILHOUETTE_FACES`` faces the silhouettes are skipped and
    ``stats["interior_silhouettes_omitted"]`` is True.

    QA round 2 (issues 19, 25): before chaining, dangling chain ends within
    0.3 mm on paper are joined (``weld_chain_ends``; ``stats["welded_ends"]``);
    every chain is simplified (Douglas-Peucker 0.05 mm on paper, hairpin
    vertices under 20 degrees with an edge shorter than 0.5 mm on paper
    removed; ``stats["hairpins_removed"]``); closed loops of organic meshes
    are kept down to 1 mm on paper instead of 3 mm (small through-holes on
    the iso view; ``stats["closed_loops_dropped"]``, ``stats["dropped_chains"]``
    counts open and closed drops, ``stats["min_loop_len"]``); faces whose
    smallest altitude is below 1e-5 x the bbox diagonal do not classify
    their edges (``stats["n_sliver_pairs"]``, see ``candidate_edges``).

    ``paper_scale`` (paper mm per model mm) sets those paper lengths; the
    default guesses that the view's long side spans 120 mm on paper.
    ``time_budget_s``: when passed before or during the hidden clean-up,
    the view is returned without hidden lines and
    ``stats["hidden_dropped"]`` is True. ``bbox`` is the exact extent of the
    projected vertices. ``stats`` carries ``px`` (pixel size in model mm)
    and the timings.
    """
    return _project(mesh, spec, raster_px=raster_px, feature_angle_deg=feature_angle_deg,
                    organic=organic, hidden=hidden, paper_scale=paper_scale,
                    time_budget_s=time_budget_s)


__all__ = ["project_mesh", "raster_span", "edge_visibility", "mask_outline", "candidate_edges",
           "fill_loops_depth", "premerge_collinear"]
