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

from .geometry2d import dedupe_primitives, merge_collinear, subtract_covered
from .model import BBox2D, LineStyle, Polyline, ProjectedView, Segment

log = logging.getLogger(__name__)

FRONT_EPS = 1e-9
CHUNK_PX = 8_000_000              # bound on per-chunk pixel arrays in raster_span
CLEANUP_TOL_PX = 0.5              # 2D clean-up tolerance (merge, subtract, dedupe), in pixels
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


def edge_visibility(segments_view: np.ndarray, depth_buffer: np.ndarray, grid: dict,
                    tol_px: float = 1.5, nbhd: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Split 3D view-space segments into visible and hidden 2D runs.

    ``segments_view`` (k, 2, 3) holds view columns [x, y, depth]. Each
    segment is sampled about one pixel apart; a sample is visible when its
    depth >= (minimum of the buffer over a (2 nbhd + 1)^2 window) - tol_px *
    px. Runs break halfway between two samples of different state. Segments
    seen end-on (projected length below 1e-6 px) are dropped. Returns
    ``(visible (m, 2, 2), hidden (h, 2, 2))`` in view coordinates.
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

def mask_outline(mask: np.ndarray, grid: dict, tol_px: float = 0.75) -> List[np.ndarray]:
    """Closed outline loops of a coverage mask, in view coordinates.

    ``skimage.measure.find_contours`` on the mask padded by one empty pixel
    (so every contour closes) at level 0.5, simplified with
    ``approximate_polygon(tol_px)``. Contour points sit on the pixel-centre
    lattice (half a pixel inside the covered pixels' outer edges). Outer
    boundaries and holes both come back; loops have no repeated end point
    and at least 3 points.
    """
    from skimage import measure

    m = np.asarray(mask, dtype=float)
    if m.size == 0 or not m.any():
        return []
    contours = measure.find_contours(np.pad(m, 1), 0.5)
    loops: List[np.ndarray] = []
    for cont in contours:
        if tol_px and tol_px > 0:
            cont = measure.approximate_polygon(cont, tolerance=float(tol_px))
        rc = cont - 1.0                                   # undo the pad
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
    """
    import trimesh

    adj = np.asarray(mesh.face_adjacency)
    adj_e = np.asarray(mesh.face_adjacency_edges)
    counts = {"sharp": 0, "silhouette": 0, "boundary": 0}
    parts = []
    if len(adj):
        ang = np.asarray(mesh.face_adjacency_angles, dtype=float)
        sharp = ang > math.radians(float(feature_angle_deg))
        sel = sharp.copy()
        counts["sharp"] = int(sharp.sum())
        if silhouettes:
            s = np.asarray(mesh.face_normals, dtype=float) @ np.asarray(n, dtype=float)
            front = s > FRONT_EPS
            sil = front[adj[:, 0]] != front[adj[:, 1]]
            counts["silhouette"] = int(sil.sum())
            sel |= sil
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


# --------------------------------------------------------------- projection

def premerge_collinear(segs: np.ndarray, tol: float) -> np.ndarray:
    """Cheap O(n log n) union of overlapping runs on (numerically) the same line.

    Edge-on rims project onto one line as hundreds of short, exactly
    collinear segments (a 2000-section cylinder in a side view: 4000 rim
    edges on two lines). ``geometry2d.merge_collinear`` pairs them all
    (union-find over ~n^2/2 pairs: 5.7 s measured for that cylinder), so
    segments whose direction agrees to 1e-9 rad and whose line offset falls
    in the same ``tol`` bin are first unioned per line here (1D interval
    union, gap <= tol), keeping the extreme original endpoints. Segments in
    neighbouring bins are left to ``merge_collinear``. Zero-length segments
    are dropped.
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


def _clean_segments(segs: np.ndarray, tol: float) -> np.ndarray:
    return merge_collinear(premerge_collinear(segs, tol), tol)


def _segments_to_prims(segs: np.ndarray, style: LineStyle) -> list:
    return [Segment((float(s[0, 0]), float(s[0, 1])), (float(s[1, 0]), float(s[1, 1])), style)
            for s in segs]


def _project(mesh, spec, *, raster_px: int, feature_angle_deg: float, organic: bool,
             hidden: bool = True, cap: Optional[Tuple[Sequence[np.ndarray], float]] = None,
             exclude_plane: Optional[Tuple[np.ndarray, np.ndarray, float]] = None,
             grid_points: Optional[np.ndarray] = None) -> ProjectedView:
    """Shared body of ``project_mesh`` and the section body projection.

    ``cap`` = (loops in view coords, depth) is filled into the depth buffer
    before the visibility test (the cut face of a section body).
    ``exclude_plane`` = (point, unit normal, tol) drops candidate edges whose
    two endpoints lie on that plane (the cut outline, drawn from the section
    loops instead). ``grid_points`` (m, 2) extra view points that the raster
    must cover.
    """
    t_start = time.perf_counter()
    stats: Dict[str, object] = {"raster_px": int(raster_px), "organic": bool(organic)}
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

    t0 = time.perf_counter()
    cand, counts = candidate_edges(mesh, spec.n, feature_angle_deg, silhouettes=not organic)
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
    tol = CLEANUP_TOL_PX * grid["px"]
    vis = _clean_segments(vis, tol)
    if hidden and len(hid):
        hid = premerge_collinear(hid, tol)
        hid = subtract_covered(hid, vis, tol)
        hid = _clean_segments(hid, tol)
    else:
        hid = np.zeros((0, 2, 2))
    visible = dedupe_primitives(_segments_to_prims(vis, LineStyle.VISIBLE), tol)
    hidden_prims = dedupe_primitives(_segments_to_prims(hid, LineStyle.HIDDEN), tol) if hidden else []
    stats["visibility_s"] = time.perf_counter() - t0

    if organic:
        t0 = time.perf_counter()
        cold = not _SKIMAGE_WARM["done"]
        loops = mask_outline(mask, grid)
        stats["outline_s"] = time.perf_counter() - t0
        stats["outline_warmup"] = bool(cold)
        stats["outline_loops"] = len(loops)
        visible = [Polyline(L, closed=True, style=LineStyle.VISIBLE) for L in loops] + visible

    bbox = BBox2D.from_points(xy)
    stats["n_visible"] = len(visible)
    stats["n_hidden"] = len(hidden_prims)
    stats["seconds"] = time.perf_counter() - t_start
    log.debug("project_mesh %s: %d visible, %d hidden, %.3f s", spec.name, len(visible),
              len(hidden_prims), stats["seconds"])
    return ProjectedView(spec=spec, visible=visible, hidden=hidden_prims, bbox=bbox,
                         algorithm="depth_buffer", stats=stats)


def project_mesh(mesh, spec, *, raster_px: int, feature_angle_deg: float, organic: bool,
                 hidden: bool = True) -> ProjectedView:
    """Visible/hidden line view of a triangle mesh (``algorithm="depth_buffer"``).

    Prismatic meshes: sharp + boundary + silhouette edges, split by the depth
    buffer into visible Segments and hidden Segments (hidden runs lying on
    visible ones removed, collinear runs merged, duplicates dropped).
    Organic meshes: the outline loops of the coverage mask as closed
    Polylines plus the sharp/boundary edges (no silhouette edges). ``bbox``
    is the exact extent of the projected vertices. ``stats`` carries ``px``
    (pixel size in model mm) and the timings.
    """
    return _project(mesh, spec, raster_px=raster_px, feature_angle_deg=feature_angle_deg,
                    organic=organic, hidden=hidden)


__all__ = ["project_mesh", "raster_span", "edge_visibility", "mask_outline", "candidate_edges",
           "fill_loops_depth", "premerge_collinear"]
