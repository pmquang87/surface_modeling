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
from typing import Tuple

import numpy as np

from .geometry2d import chain_segments, polygon_area
from .mesh_views import _project
from .model import SectionView
from .projection import VIEWS

log = logging.getLogger(__name__)

_CAP_SECTION = True               # fill the cut face into the body's depth buffer
NUDGE_FRACTION = 0.01             # nudge = 1 % of the depth extent
WELD_REL = 1e-6                   # chain weld tolerance, fraction of the bbox diagonal
PARALLEL_DOT = 0.999              # plane normal counts as the view direction above this

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


def _loops_at(mesh, point: np.ndarray, normal: np.ndarray, spec, weld: float):
    from trimesh.intersections import mesh_plane

    segs = np.asarray(mesh_plane(mesh, plane_normal=normal, plane_origin=point), dtype=float)
    n_segs = int(len(segs))
    if n_segs == 0:
        return [], 0, 0
    closed, n_open = chain_segments(segs.reshape(-1, 2, 3), weld)
    loops = []
    for L in closed:
        xy = np.ascontiguousarray(spec.to_view(L)[:, :2])
        if len(xy) >= 3 and abs(polygon_area(xy)) > 0.0:
            loops.append(xy)
    return loops, int(n_open), n_segs


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
                 organic: bool, depth_extent: float) -> SectionView:
    """Section of ``mesh`` by the plane (point, normal), seen in ``spec``.

    Loops: closed polygons in ``spec`` view coordinates without a repeated
    end point; open chains are counted in ``stats["open_wires_dropped"]``.
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

    loops, n_open, n_segs = _loops_at(mesh, p, n, spec, weld)
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
        loops, n_open, n_segs = _loops_at(mesh, p, n, spec, weld)
        log.info("section_mesh: no closed loop, plane nudged by %.6g -> %d loops", sign * shift, len(loops))
    stats["open_wires_dropped"] = int(n_open)
    stats["n_segments"] = int(n_segs)

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
    return SectionView(spec=spec, plane_point=p, plane_normal=v.copy(), loops=loops, body=body,
                       label="A", marker_view=_marker_for_normal(n), stats=stats)


__all__ = ["choose_section_plane_mesh", "section_mesh"]
