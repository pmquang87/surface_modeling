"""Section views of a triangle mesh: plane choice, closed loops, cut body.

Spec 3.5 (section) with the nudge rule of 3.4. Measured facts this module
relies on (mesh probe, trimesh 5.0.0, REPORT-mesh.md):

* ``trimesh.intersections.mesh_plane(mesh, plane_normal=, plane_origin=)``
  must be called with KEYWORD arguments (``Trimesh.section`` silently
  returns None when origin and normal are swapped positionally). It returns
  an (k, 2, 3) array; a plane that misses the part AND a plane lying exactly
  on a planar face both give 0 segments, so "0 closed loops" triggers one
  nudge of 1 % of the depth extent (toward the part's bbox centre, so a
  plane on a face moves into the material), then the section is reported
  empty.
* Chaining with a KD-tree weld at 1e-6 x the bbox diagonal gave the expected
  loop counts in every probe case (planes through vertices, 42 planes on an
  icosphere, torus through the axis -> two r = 3.000 circles at +-10,
  perpendicular -> rings r 13.000 / 7.000). Loop orientation is arbitrary;
  layout nests by containment.
* An open mesh leaves the contour open where the plane crosses a hole. The
  chain ends closer than ``BRIDGE_K`` (3) x the median mesh edge length are
  joined nearest first before open chains are dropped (QA round 1, bunny99
  top/right planes: kept perimeter share 0.14 / 0.105 -> 1.0; wheel top
  plane: 21 open chains -> 1). A triangle soup would also be bridged into
  junk loops (QA corpus 22: 75 bridges -> 3 loops, kept length ratio 1.0),
  so bridging is rejected when the open chains average fewer than
  ``BRIDGE_MIN_CHAIN`` segments or the bridges exceed ``BRIDGE_MAX_SHARE``
  of the segments of the loops they create (``stats["bridge_rejected"]``).
  The caller should still skip the section for a ``not_solid`` mesh.
* Loops are expressed in the section view's own x/y (``spec.to_view``), so
  they line up with the body projection and with the B-rep route.
* ``trimesh.intersections.slice_faces_plane`` (the worker of
  ``slice_mesh_plane``, which imports shapely even with ``cap=False``
  (measured)) keeps the part on the POSITIVE side of the plane normal; the
  section body is the half BEHIND the plane (away from the viewer), so it is
  called with ``-v`` where ``v`` is the plane normal oriented toward the
  viewer. The slice is not capped (trimesh capping needs shapely and a
  polygon triangulator); instead the cut face is filled into
  the depth buffer at the plane depth (``_CAP_SECTION``), otherwise the open
  cut would expose material behind it as visible lines. Body edges lying in
  the cut plane are dropped: the layout draws the section loops instead.

No OCP, shapely or matplotlib import. No output to stdout.
"""
from __future__ import annotations

import logging
import time
from typing import Optional, Tuple

import numpy as np

from .geometry2d import chain_segments, nest_loops, polygon_area
from .mesh_views import _project
from .model import SectionView
from .projection import VIEWS

log = logging.getLogger(__name__)

_CAP_SECTION = True               # fill the cut face into the body's depth buffer
NUDGE_FRACTION = 0.01             # nudge = 1 % of the depth extent
WELD_REL = 1e-6                   # chain weld tolerance, fraction of the bbox diagonal
PARALLEL_DOT = 0.999              # plane normal counts as the view direction above this
BRIDGE_K = 3.0                    # bridge open chain ends closer than k x median edge length
BRIDGE_MIN_CHAIN = 4.0            # no bridging when open chains average fewer segments (soup)
BRIDGE_MAX_SHARE = 0.25           # no bridging when bridges exceed this share of the new loop segments

# mode -> (view whose plane the cut is parallel to, principal view showing the cutting line)
_MODES = {
    "auto": ("front", "top"),
    "front": ("front", "top"),
    "top": ("top", "front"),
    "right": ("right", "front"),
}


def _marker_for_normal(normal: np.ndarray) -> str:
    n = np.asarray(normal, dtype=float)
    n = n / max(float(np.linalg.norm(n)), 1e-300)
    for view, marker in (("front", "top"), ("top", "front"), ("right", "front")):
        if abs(float(n @ VIEWS[view].n)) > PARALLEL_DOT:
            return marker
    return "top"


def choose_section_plane_mesh(mesh, mode: str) -> Tuple[np.ndarray, np.ndarray, str]:
    """(point, normal, marker_view) of the section plane for the mesh route.

    The mesh route has no hole features, so every mode cuts through the bbox
    centre. ``auto`` / ``front``: plane parallel to the front view, normal
    (0, -1, 0) (= ``VIEWS["front"].n``), cutting line shown in the top view;
    ``top``: normal (0, 0, 1), marker in the front view; ``right``: normal
    (1, 0, 0), marker in the front view. Other modes raise ``ValueError``.
    """
    if mode not in _MODES:
        raise ValueError(f"section mode must be one of {sorted(_MODES)}, got {mode!r}")
    view, marker = _MODES[mode]
    bounds = np.asarray(mesh.bounds, dtype=float)
    centre = 0.5 * (bounds[0] + bounds[1])
    return centre.copy(), np.array(VIEWS[view].n, dtype=float), marker


def _dangling_ends(segs: np.ndarray, weld: float) -> np.ndarray:
    """Coordinates of the chain end points (welded node degree 1) of 3D segments."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    pts = segs.reshape(-1, 3)
    pairs = cKDTree(pts).query_pairs(weld, output_type="ndarray")
    n = len(pts)
    g = coo_matrix((np.ones(len(pairs), np.int8), (pairs[:, 0], pairs[:, 1])), shape=(n, n)) \
        if len(pairs) else coo_matrix((n, n), dtype=np.int8)
    _, node = connected_components(g, directed=False)
    node = np.asarray(node).reshape(-1, 2)
    edges = node[node[:, 0] != node[:, 1]]
    if len(edges) == 0:
        return np.zeros((0, 3))
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    n_nodes = int(node.max()) + 1
    deg = np.bincount(edges.ravel(), minlength=n_nodes)
    xyz = np.zeros((n_nodes, 3))
    cnt = np.zeros(n_nodes)
    np.add.at(xyz, node.ravel(), pts)
    np.add.at(cnt, node.ravel(), 1.0)
    xyz /= np.maximum(cnt, 1.0)[:, None]
    return xyz[deg == 1]


def _bridges(ends: np.ndarray, tol: float) -> np.ndarray:
    """Greedy nearest pairing of chain ends closer than ``tol``: (m, 2, 3) segments."""
    from scipy.spatial import cKDTree

    if len(ends) < 2 or not tol > 0.0:
        return np.zeros((0, 2, 3))
    pairs = cKDTree(ends).query_pairs(tol, output_type="ndarray")
    if len(pairs) == 0:
        return np.zeros((0, 2, 3))
    d = np.linalg.norm(ends[pairs[:, 0]] - ends[pairs[:, 1]], axis=1)
    used = np.zeros(len(ends), dtype=bool)
    out = []
    for k in np.argsort(d, kind="stable"):
        i, j = int(pairs[k, 0]), int(pairs[k, 1])
        if used[i] or used[j]:
            continue
        used[i] = used[j] = True
        out.append((ends[i], ends[j]))
    return np.asarray(out, dtype=float).reshape(-1, 2, 3)


def face_components(mesh) -> np.ndarray:
    """Body (component) label per face: components of the vertex graph, so
    faces joined through any shared vertex or edge (non-manifold ones
    included) are one body, as ``mesh_load.mesh_topology``'s ``body_label``
    (QA round 2 issue 3: ``SectionView.loop_body``). Labels are numbered in
    the order of the first face of each body."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    faces = np.asarray(mesh.faces, dtype=np.int64)
    if len(faces) == 0:
        return np.zeros(0, dtype=np.int64)
    nv = max(int(len(mesh.vertices)), int(faces.max()) + 1)
    a = np.concatenate([faces[:, 0], faces[:, 1]])
    b = np.concatenate([faces[:, 1], faces[:, 2]])
    g = coo_matrix((np.ones(len(a), dtype=np.int8), (a, b)), shape=(nv, nv))
    _, vlab = connected_components(g, directed=False)
    lab = np.asarray(vlab, dtype=np.int64)[faces[:, 0]]
    _, first, inv = np.unique(lab, return_index=True, return_inverse=True)
    rank = np.empty(len(first), dtype=np.int64)
    rank[np.argsort(first, kind="stable")] = np.arange(len(first))
    return rank[np.asarray(inv).ravel()]


def component_volumes(mesh, face_body: np.ndarray) -> np.ndarray:
    """Signed volume per component (``face_components`` labels): the face
    sum of each component; an inward-wound cavity shell is negative."""
    tri = np.asarray(mesh.triangles, dtype=float)
    fv = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])) / 6.0
    n = int(face_body.max()) + 1 if len(face_body) else 0
    return np.bincount(face_body, weights=fv, minlength=n)


def _merge_void_bodies(loops: list, ids: list, comp_volume=None) -> list:
    """The inner shell of a hollow mesh (an internal void) is a separate
    component of the vertex graph, but its cut loop is a hole of the outer
    body; as its own body the layout would nest it alone and fill it.
    ``brep_sections.merge_void_bodies`` joins it to the body around it when
    all its loops are holes and its signed volume (``comp_volume``, from
    ``component_volumes``) is negative; without volumes nothing is joined."""
    from .brep_sections import merge_void_bodies

    if comp_volume is None:
        return list(ids)
    vols = np.asarray(comp_volume, dtype=float)
    return merge_void_bodies(loops, ids,
                             lambda b: 0 <= int(b) < len(vols) and float(vols[int(b)]) < 0.0)


def _loop_bodies(closed3: list, segs: np.ndarray, seg_faces: np.ndarray,
                 face_body: np.ndarray) -> list:
    """Body label of each closed 3D chain: the body of the face of the cut
    segment whose end point is nearest the chain's first point."""
    from scipy.spatial import cKDTree

    if not closed3 or len(segs) == 0:
        return [0] * len(closed3)
    tree = cKDTree(segs.reshape(-1, 3))
    out = []
    for L in closed3:
        _, k = tree.query(np.asarray(L, float)[0])
        f = int(seg_faces[int(k) // 2])
        out.append(int(face_body[f]) if 0 <= f < len(face_body) else 0)
    return out


def _loops_at(mesh, point: np.ndarray, normal: np.ndarray, spec, weld: float,
              bridge_tol: float = 0.0, face_body: Optional[np.ndarray] = None,
              comp_volume: Optional[np.ndarray] = None):
    """(loops, open chains dropped, segment count, info) of one plane.

    When chains stay open and ``bridge_tol > 0``, chain end points closer
    than ``bridge_tol`` are joined pairwise (nearest first) and the segments
    are chained again (QA issue 33: an open mesh lost its main contour).
    ``info``: ``bridged_gaps``, ``bridge_max_gap``, ``segments_bbox_area``
    (view-plane bbox of every cut segment, the reference of the kept-area
    ratio), ``loop_body`` (body label per loop from ``face_body``, the
    ``face_components`` labels, void shells joined to their container with
    ``comp_volume``; None without ``face_body``).
    """
    from trimesh.intersections import mesh_plane

    info = {"bridged_gaps": 0, "bridge_max_gap": 0.0, "segments_bbox_area": 0.0,
            "segments_length": 0.0, "bridge_rejected": False,
            "open_chain_mean_segments": None, "bridge_share": None, "loop_body": None}
    seg_faces = None
    if face_body is not None:
        raw, seg_faces = mesh_plane(mesh, plane_normal=normal, plane_origin=point,
                                    return_faces=True)
        segs = np.asarray(raw, dtype=float)
        seg_faces = np.asarray(seg_faces, dtype=np.int64).ravel()
    else:
        segs = np.asarray(mesh_plane(mesh, plane_normal=normal, plane_origin=point), dtype=float)
    n_segs = int(len(segs))
    if n_segs == 0:
        if face_body is not None:
            info["loop_body"] = []
        return [], 0, 0, info
    segs = segs.reshape(-1, 2, 3)
    info["segments_length"] = float(np.linalg.norm(segs[:, 1] - segs[:, 0], axis=1).sum())
    vxy = spec.to_view(segs.reshape(-1, 3))[:, :2]
    ext = vxy.max(axis=0) - vxy.min(axis=0)
    info["segments_bbox_area"] = float(ext[0] * ext[1])
    closed, n_open = chain_segments(segs, weld)
    if n_open > 0 and bridge_tol > 0.0:
        try:
            ends = _dangling_ends(segs, weld)
            br = _bridges(ends, float(bridge_tol))
            if len(br):
                closed2, n_open2 = chain_segments(np.concatenate([segs, br]), weld)
                if n_open2 < n_open:
                    # soup guard: an open mesh's cut has long open chains and
                    # few bridges per loop; a triangle soup's cut has chains of
                    # one segment each, and its bridged "loops" alternate
                    # segment / bridge (QA corpus 22: 75 bridges -> 3 loops)
                    seg_closed = sum(len(L) for L in closed)
                    seg_new = sum(len(L) for L in closed2) - seg_closed
                    mean_chain = (n_segs - seg_closed) / max(0.5 * len(ends), 1.0)
                    share = len(br) / max(seg_new, 1)
                    info["open_chain_mean_segments"] = float(mean_chain)
                    info["bridge_share"] = float(share)
                    if mean_chain < BRIDGE_MIN_CHAIN or share > BRIDGE_MAX_SHARE:
                        info["bridge_rejected"] = True
                        log.info("section: bridging rejected (mean open chain %.3g segments, "
                                 "%d bridges for %d new loop segments): triangle soup?",
                                 mean_chain, len(br), seg_new)
                    else:
                        closed, n_open = closed2, n_open2
                        info["bridged_gaps"] = int(len(br))
                        info["bridge_max_gap"] = float(np.linalg.norm(br[:, 1] - br[:, 0], axis=1).max())
        except Exception as exc:  # bridging is an improvement, never a failure
            log.debug("section bridging failed: %s: %s", type(exc).__name__, exc)
    loops = []
    kept3 = []
    for L in closed:
        xy = np.ascontiguousarray(spec.to_view(L)[:, :2])
        if len(xy) >= 3 and abs(polygon_area(xy)) > 0.0:
            loops.append(xy)
            kept3.append(L)
    if face_body is not None and seg_faces is not None and len(seg_faces) == n_segs:
        try:
            info["loop_body"] = _merge_void_bodies(
                loops, _loop_bodies(kept3, segs, seg_faces, face_body), comp_volume)
        except Exception as exc:  # body ids are an extra, never a failure
            log.debug("section loop bodies failed: %s: %s", type(exc).__name__, exc)
    return loops, int(n_open), n_segs, info


_NEST_MAX_LOOPS = 200             # nest_loops is O(n^2) in the loop count


def _material_area(loops) -> float:
    """Cut material area: outer loops minus their holes (nesting by containment)."""
    if not loops:
        return 0.0
    if len(loops) > _NEST_MAX_LOOPS:
        return float(sum(abs(polygon_area(L)) for L in loops))
    try:
        return float(sum(abs(polygon_area(o)) - sum(abs(polygon_area(h)) for h in holes)
                         for o, holes in nest_loops(loops)))
    except Exception as exc:  # pragma: no cover - defensive
        log.debug("section_mesh: nest_loops failed: %s", exc)
        return float(sum(abs(polygon_area(L)) for L in loops))


def _half_behind(mesh, point: np.ndarray, v: np.ndarray):
    """The part of ``mesh`` on the side of the plane opposite to ``v`` (None if empty).

    ``slice_faces_plane`` is the worker inside ``slice_mesh_plane``; the
    wrapper imports ``trimesh.path.polygons`` (shapely) even with
    ``cap=False`` (measured, trimesh 5.0.0), so it is not used. The kept side
    is the positive side of the passed normal, hence ``-v``.
    """
    import trimesh
    from trimesh.intersections import slice_faces_plane

    verts, faces, _uv = slice_faces_plane(vertices=np.asarray(mesh.vertices, dtype=float),
                                          faces=np.asarray(mesh.faces, dtype=np.int64),
                                          plane_normal=-v, plane_origin=point)
    if faces is None or len(faces) == 0:
        return None
    return trimesh.Trimesh(vertices=verts, faces=faces, process=True)


def section_mesh(mesh, point, normal, spec, *, raster_px: int, feature_angle_deg: float,
                 organic: bool, depth_extent: float,
                 bridge_k: Optional[float] = None) -> SectionView:
    """Section of ``mesh`` by the plane (point, normal), seen in ``spec``.

    Loops: closed polygons in ``spec`` view coordinates without a repeated
    end point. Open chains (open mesh) are first bridged: chain ends closer
    than ``bridge_k`` (default ``BRIDGE_K``) x the median mesh edge length
    are joined nearest first and the segments re-chained; ``bridge_k=0``
    turns this off. Chains still open are counted in
    ``stats["open_wires_dropped"]``; ``stats["bridged_gaps"]``,
    ``stats["bridge_max_gap"]``, ``stats["kept_area_ratio"]`` (material
    area of the kept loops, holes subtracted, / area of the view-plane bbox
    of all cut segments; None without segments) and
    ``stats["kept_length_ratio"]`` (perimeter of the kept loops / length
    of all cut segments, capped at 1; None without segments) describe the
    result. The length ratio is the reliable "contour lost" signal: a
    sliver left after dropping the main contour gives 0.10-0.14 (QA
    bunny99), a legit thin rim section 1.0 while its area ratio is 0.034.
    Zero closed loops -> one nudge of ``1 % * depth_extent`` toward the bbox
    centre (``stats["nudged"] True``; ``plane_point`` is the nudged point);
    still zero -> ``loops == []``. ``body``: depth-buffer view of the half
    behind the plane with the cut face as an occluder (None when nothing
    lies behind the plane). ``marker_view`` follows the normal (front-
    parallel -> "top", top/right-parallel -> "front").
    """
    t_start = time.perf_counter()
    p = np.asarray(point, dtype=float).reshape(3).copy()
    n = np.asarray(normal, dtype=float).reshape(3)
    nn = float(np.linalg.norm(n))
    if not nn > 0.0:
        raise ValueError("section_mesh: normal must be non-zero")
    n = n / nn
    view_dot = float(n @ spec.n)
    v = n if view_dot >= 0.0 else -n                       # plane normal toward the viewer
    parallel = abs(view_dot) > PARALLEL_DOT
    if not parallel:
        log.warning("section_mesh: plane normal %s is not parallel to view %s; no cap", n, spec.name)

    bounds = np.asarray(mesh.bounds, dtype=float)
    diag = float(np.linalg.norm(bounds[1] - bounds[0]))
    weld = WELD_REL * max(diag, 1e-9)
    stats = {"open_wires_dropped": 0, "nudged": False, "plane_shift": 0.0}
    k = BRIDGE_K if bridge_k is None else float(bridge_k)
    bridge_tol = 0.0
    if k > 0.0 and len(mesh.faces):
        try:
            bridge_tol = k * float(np.median(np.asarray(mesh.edges_unique_length, dtype=float)))
        except Exception as exc:  # pragma: no cover - defensive
            log.debug("section_mesh: median edge length failed: %s", exc)
    stats["bridge_tol"] = float(bridge_tol)
    face_body, comp_volume = None, None
    try:
        face_body = face_components(mesh)
        if len(face_body) and int(face_body.max()) > 0:
            comp_volume = component_volumes(mesh, face_body)
    except Exception as exc:  # pragma: no cover - defensive
        log.debug("section_mesh: face components failed: %s", exc)

    loops, n_open, n_segs, info = _loops_at(mesh, p, n, spec, weld, bridge_tol, face_body,
                                            comp_volume)
    if not loops:
        extent = float(depth_extent)
        if not extent > 0.0:
            extent = float(np.ptp(np.asarray(mesh.vertices, float) @ n)) if len(mesh.vertices) else 1.0
        shift = NUDGE_FRACTION * extent
        centre = 0.5 * (bounds[0] + bounds[1])
        side = float((centre - p) @ v)
        sign = 1.0 if side > 0.0 else -1.0                   # toward the centre; behind on a tie
        p = p + sign * shift * v
        stats["nudged"] = True
        stats["plane_shift"] = float(sign * shift)
        loops, n_open, n_segs, info = _loops_at(mesh, p, n, spec, weld, bridge_tol, face_body,
                                                comp_volume)
        log.info("section_mesh: no closed loop, plane nudged by %.6g -> %d loops", sign * shift, len(loops))
    stats["open_wires_dropped"] = int(n_open)
    stats["n_segments"] = int(n_segs)
    stats["bridged_gaps"] = int(info["bridged_gaps"])
    stats["bridge_max_gap"] = float(info["bridge_max_gap"])
    stats["bridge_rejected"] = bool(info["bridge_rejected"])
    stats["open_chain_mean_segments"] = info["open_chain_mean_segments"]
    stats["bridge_share"] = info["bridge_share"]
    ref =float(info["segments_bbox_area"])
    stats["kept_area_ratio"] = (_material_area(loops) / ref) if ref > 0.0 else None
    seg_len = float(info["segments_length"])
    kept_len = float(sum(np.linalg.norm(np.roll(L, -1, axis=0) - L, axis=1).sum() for L in loops))
    stats["kept_length_ratio"] = min(1.0, kept_len / seg_len) if seg_len > 0.0 else None

    body = None
    try:
        half = _half_behind(mesh, p, v)
        if half is not None:
            cap = None
            if _CAP_SECTION and parallel and loops:
                cap = (loops, float(p @ spec.n))
            loop_pts = np.vstack(loops) if loops else None
            body = _project(half, spec, raster_px=raster_px, feature_angle_deg=feature_angle_deg,
                            organic=organic, hidden=True, cap=cap,
                            exclude_plane=(p, v, max(weld, 1e-9)), grid_points=loop_pts)
            body.stats["capped"] = cap is not None
    except Exception as exc:  # the section loops stay useful without a body
        log.warning("section_mesh: body projection failed: %s: %s", type(exc).__name__, exc)
        stats["body_error"] = f"{type(exc).__name__}: {exc}"
        body = None

    stats["seconds"] = time.perf_counter() - t_start
    lb = info.get("loop_body")
    lb = list(lb) if lb is not None and len(lb) == len(loops) else None
    return SectionView(spec=spec, plane_point=p, plane_normal=v.copy(), loops=loops, body=body,
                       label="A", marker_view=_marker_for_normal(n), stats=stats, loop_body=lb)


__all__ = ["choose_section_plane_mesh", "section_mesh", "face_components"]
