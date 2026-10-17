"""Section view of an OCCT shape (B-rep route): plane choice, loops, body.

``choose_section_plane_brep`` picks the cutting plane; ``section_brep``
cuts the shape and returns a ``SectionView``: the closed section loops in
the view's 2D coordinates (the hatch boundary) plus the HLR view of the
body that remains behind the plane.

Conventions: the section is seen with ``spec`` (a ``ViewSpec``); the plane
normal equals ``spec.n`` and points toward the viewer, so the material on
the +n side of the plane is removed and the viewer looks at the cut face.
Loops are ``(n, 2)`` arrays without a repeated end point in the same
coordinates as ``project_brep`` (x = spec.x . p, y = spec.y . p, origin not
shifted); outers come out counter-clockwise, holes clockwise
(``geometry2d.nest_loops``).

Measured facts this module relies on (brep-analysis probe, OCP 7.9.3,
REPORT-brep-analysis.md):

* ``BRepAlgoAPI_Section(shape, gp_Pln, False)`` + ``ComputePCurveOn1(True)``
  + ``Approximation(True)`` + ``Build()``, edges joined with
  ``ShapeAnalysis_FreeBounds.ConnectEdgesToWires_s(edges, 1e-4, False,
  wires)``: plate section y = 30 -> 12 edges, 2 closed wires of area +600
  and -900 (orientation is arbitrary, so nesting is by containment).
* The default ``gp_Ax2(P, N)`` in-plane basis is arbitrary (N = -Y gave
  u = (0,0,-1)); the loops are therefore mapped with the view's own x / y.
* Empty and degenerate sections raise nothing: a plane missing the shape
  gives 0 edges; a plane coplanar with a face gives that face's outline.
* A plane tangent to a cylinder (plate x = 93, tangent to the d30 boss)
  makes ConnectEdgesToWires return an OPEN wire that holds a real closed
  loop (the middle slab) plus the dangling tangent line (measured here: 3
  wires, the open one with 6 edges, one of them (93,30,15)-(93,30,25)).
  Open wires are counted in ``stats["open_wires_dropped"]``; their dangling
  edges are pruned (degree-1 endpoints, repeatedly) and the remaining
  closed chains are recovered with ``geometry2d.chain_segments``
  (``stats["loops_recovered"]``), so the tangent line does not take the
  slab with it.
* Body: ``BRepPrimAPI_MakeHalfSpace(face, point_in_removed_half).Solid()``
  + ``BRepAlgoAPI_Cut`` (6.2 ms on the plate, valid solid of exactly half
  the volume); its cut faces coincide with the section loops to 4e-14 mm.
  A cut that misses returns the shape unchanged; one that removes
  everything returns an empty shape whose HLR gives 0 edges.
* A plane coplanar with a planar face is NOT reported as empty by OCCT: the
  plate cut at z = 15 (its top face) gives 7 loops with the boss footprint
  (area 705.6) as a HOLE in the outer loop, and z = 25 gives only the boss
  top disc. Such a plane (checked on the planar faces before cutting,
  ``stats["coplanar"]`` = number of faces in it) and a plane with zero
  closed loops are nudged once by 1 % of ``depth_extent`` along the normal
  toward the shape's bbox centre (``stats["nudged"]``); still empty ->
  ``loops == []`` and no body (the pipeline flags ``section_empty``).
"""
from __future__ import annotations

import logging
import math
import time
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .geometry2d import chain_segments, nest_loops
from .model import CylFeature, SectionView
from .projection import VIEWS

log = logging.getLogger(__name__)

# Behaviour switch, module-level so the negative-control test can turn it off.
_RECOVER_OPEN = True       # recover closed loops inside open wires (tangent planes)
_NUDGE_COPLANAR = True     # nudge a plane that lies in a planar face before cutting

_WIRE_TOL = 1e-4           # ConnectEdgesToWires tolerance (probe value)
_NUDGE_FRACTION = 0.01
_COPLANAR_COS_TOL = 1e-9   # |n_face . n| > 1 - this -> parallel
_COPLANAR_DIST_TOL = 1e-6  # mm; the face tolerance is used when larger

# mode -> (section view name, marker view name)
_MODES = {"auto": ("front", "top"), "front": ("front", "top"),
          "top": ("top", "front"), "right": ("right", "front")}


# ----------------------------------------------------------------- plane choice

def _exact_bbox(shape) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    try:
        from OCP.Bnd import Bnd_Box
        from OCP.BRepBndLib import BRepBndLib
        if shape is None or shape.IsNull():
            return None
        box = Bnd_Box()
        BRepBndLib.AddOptimal_s(shape, box, False, False)
        if box.IsVoid():
            return None
        x0, y0, z0, x1, y1, z1 = box.Get()
        return np.array([x0, y0, z0], float), np.array([x1, y1, z1], float)
    except Exception as exc:
        log.debug("exact bbox failed: %s", exc)
        return None


def _model_bbox(model) -> Tuple[np.ndarray, np.ndarray]:
    census = getattr(model, "census", None) or {}
    lo, hi = census.get("bbox_min"), census.get("bbox_max")
    if lo is not None and hi is not None:
        lo = np.asarray(lo, float).reshape(3)
        hi = np.asarray(hi, float).reshape(3)
        if np.all(np.isfinite(lo)) and np.all(np.isfinite(hi)):
            return lo, hi
    bb = _exact_bbox(getattr(model, "shape", None))
    if bb is None:
        raise ValueError("choose_section_plane_brep: no bounding box (census and shape both empty)")
    return bb


def choose_section_plane_brep(model, features: Sequence[CylFeature], mode: str
                              ) -> Tuple[np.ndarray, np.ndarray, str]:
    """(point, normal, marker_view) of the cutting plane.

    ``mode``: "auto" / "front" (normal = front view n = (0,-1,0), cutting
    line shown in the top view), "top" (normal (0,0,1), shown in the front
    view), "right" (normal (1,0,0), shown in the front view). The plane goes
    through the axis of the largest hole (``kind == "hole"``) whose axis is
    perpendicular to the normal (the point is the hole centre); without one
    it goes through the bbox centre. ``model`` needs ``census["bbox_min"/
    "bbox_max"]`` or a ``shape`` (exact ``AddOptimal_s`` bbox then).
    """
    if mode not in _MODES:
        raise ValueError(f"section mode must be one of {sorted(_MODES)}, got {mode!r}")
    view_name, marker = _MODES[mode]
    normal = np.array(VIEWS[view_name].n, dtype=float)
    lo, hi = _model_bbox(model)
    centre = 0.5 * (lo + hi)
    best = None
    for f in features or ():
        if getattr(f, "kind", "") != "hole":
            continue
        axis = np.asarray(f.axis, float).reshape(3)
        na = float(np.linalg.norm(axis))
        if na < 1e-12 or abs(float(np.dot(axis / na, normal))) > 1e-6:
            continue
        c = np.asarray(f.center, float).reshape(3)
        key = (float(f.radius), -float(np.linalg.norm(c - centre)))
        if best is None or key > best[0]:
            best = (key, c)
    point = best[1].copy() if best is not None else centre
    log.debug("section plane %s: point %s normal %s (%s)", mode, point.tolist(), normal.tolist(),
              "largest hole" if best is not None else "bbox centre")
    return point, normal, marker


# ----------------------------------------------------------------- loops

def _edge_points(edge, deflection: float) -> np.ndarray:
    """3D sample points of an edge in its own parametric direction (no wire orientation)."""
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.GCPnts import GCPnts_QuasiUniformDeflection, GCPnts_TangentialDeflection
    from OCP.GeomAbs import GeomAbs_Line
    a = BRepAdaptor_Curve(edge)
    u0, u1 = a.FirstParameter(), a.LastParameter()
    if a.GetType() == GeomAbs_Line:
        pts = [a.Value(u0), a.Value(u1)]
    else:
        pts = None
        try:
            q = GCPnts_QuasiUniformDeflection(a, deflection, u0, u1)
            if q.IsDone() and q.NbPoints() >= 2:
                pts = [q.Value(i) for i in range(1, q.NbPoints() + 1)]
        except Exception as exc:
            log.debug("QuasiUniformDeflection failed on a section edge: %s", exc)
        if pts is None:
            t = GCPnts_TangentialDeflection(a, u0, u1, 0.1, deflection)
            pts = [t.Value(i) for i in range(1, t.NbPoints() + 1)]
    return np.array([[p.X(), p.Y(), p.Z()] for p in pts], dtype=np.float64)


def _wire_loop(wire, deflection: float) -> np.ndarray:
    """Closed wire -> (n, 3) points in wire order, no repeated end point."""
    from OCP.BRepTools import BRepTools_WireExplorer
    from OCP.TopAbs import TopAbs_REVERSED
    pts: List[np.ndarray] = []
    we = BRepTools_WireExplorer(wire)
    while we.More():
        seg = _edge_points(we.Current(), deflection)
        if we.Orientation() == TopAbs_REVERSED:
            seg = seg[::-1]
        if pts and np.linalg.norm(pts[-1] - seg[0]) < 1e-6:
            seg = seg[1:]
        pts.extend(seg)
        we.Next()
    P = np.array(pts, dtype=np.float64).reshape(-1, 3)
    if len(P) > 1 and np.linalg.norm(P[0] - P[-1]) < 1e-6:
        P = P[:-1]
    return P


def _wire_segments(wire, deflection: float) -> np.ndarray:
    """All edges of a (possibly open) wire as 3D segments (k, 2, 3)."""
    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS
    chunks = []
    ex = TopExp_Explorer(wire, TopAbs_EDGE)
    while ex.More():
        try:
            pts = _edge_points(TopoDS.Edge_s(ex.Current()), deflection)
            if len(pts) >= 2:
                chunks.append(np.stack([pts[:-1], pts[1:]], axis=1))
        except Exception as exc:
            log.debug("section edge skipped: %s", exc)
        ex.Next()
    if not chunks:
        return np.zeros((0, 2, 3))
    return np.concatenate(chunks)


def _prune_dangling(segs: np.ndarray, tol: float) -> Tuple[np.ndarray, int]:
    """Remove segments with a free (degree-1) endpoint, repeatedly. Returns (kept, n_pruned)."""
    from scipy.spatial import cKDTree
    k = len(segs)
    if k == 0:
        return segs, 0
    pts = segs.reshape(-1, segs.shape[-1])
    parent = np.arange(len(pts))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in cKDTree(pts).query_pairs(tol):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    node = np.array([find(i) for i in range(len(pts))]).reshape(k, 2)
    alive = node[:, 0] != node[:, 1]              # zero-length segments carry no topology
    n_pruned = int(np.count_nonzero(~alive))
    while True:
        deg = {}
        for (u, v) in node[alive]:
            deg[u] = deg.get(u, 0) + 1
            deg[v] = deg.get(v, 0) + 1
        leaf = alive & np.array([deg.get(u, 0) == 1 or deg.get(v, 0) == 1 for u, v in node])
        if not leaf.any():
            break
        n_pruned += int(np.count_nonzero(leaf))
        alive &= ~leaf
    return segs[alive], n_pruned


def _section_loops(shape, point, normal, spec, deflection: float):
    """Closed section loops in view 2D + stats for one plane."""
    from OCP.BRep import BRep_Tool
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Section
    from OCP.ShapeAnalysis import ShapeAnalysis_FreeBounds
    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopTools import TopTools_HSequenceOfShape
    from OCP.TopoDS import TopoDS
    from OCP.gp import gp_Dir, gp_Pln, gp_Pnt

    stats = {"edges": 0, "wires": 0, "open_wires_dropped": 0, "dangling_segments_pruned": 0,
             "loops_recovered": 0}
    sec = BRepAlgoAPI_Section(shape, gp_Pln(gp_Pnt(*map(float, point)), gp_Dir(*map(float, normal))),
                              False)
    sec.ComputePCurveOn1(True)
    sec.Approximation(True)
    sec.Build()
    if not sec.IsDone():
        raise RuntimeError("BRepAlgoAPI_Section did not finish")
    edges = TopTools_HSequenceOfShape()
    ex = TopExp_Explorer(sec.Shape(), TopAbs_EDGE)
    while ex.More():
        edges.Append(ex.Current())
        ex.Next()
    stats["edges"] = edges.Length()
    loops3: List[np.ndarray] = []
    if edges.Length() == 0:
        return [], stats
    wires = TopTools_HSequenceOfShape()
    ShapeAnalysis_FreeBounds.ConnectEdgesToWires_s(edges, _WIRE_TOL, False, wires)
    stats["wires"] = wires.Length()
    open_segs = []
    for i in range(1, wires.Length() + 1):
        w = TopoDS.Wire_s(wires.Value(i))
        if BRep_Tool.IsClosed_s(w):
            try:
                P = _wire_loop(w, deflection)
            except Exception as exc:
                log.debug("closed section wire skipped: %s", exc)
                continue
            if len(P) >= 3:
                loops3.append(P)
        else:
            stats["open_wires_dropped"] += 1
            if _RECOVER_OPEN:
                open_segs.append(_wire_segments(w, deflection))
    basis = np.stack([np.asarray(spec.x, float), np.asarray(spec.y, float)], axis=1)   # (3, 2)
    loops2 = [P @ basis for P in loops3]
    if open_segs:
        segs3 = np.concatenate(open_segs)
        kept, pruned = _prune_dangling(segs3, _WIRE_TOL)
        stats["dangling_segments_pruned"] = pruned
        if len(kept):
            rec, n_open = chain_segments(kept @ basis, _WIRE_TOL)
            rec = [L for L in rec if len(L) >= 3]
            stats["loops_recovered"] = len(rec)
            loops2 += rec
    if stats["open_wires_dropped"]:
        log.info("section: %d open wire(s) dropped, %d loop(s) recovered from them",
                 stats["open_wires_dropped"], stats["loops_recovered"])
    return loops2, stats


def _coplanar_face_count(shape, point, normal) -> int:
    """Number of planar faces lying in the plane (point, normal).

    A face counts when its surface is a ``GeomAbs_Plane`` whose normal is
    parallel to ``normal`` (|cos| > 1 - _COPLANAR_COS_TOL) and whose plane
    passes within max(_COPLANAR_DIST_TOL, face tolerance) of ``point``.
    Errors on single faces are logged and skipped.
    """
    from OCP.BRep import BRep_Tool
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_Plane
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS
    p = np.asarray(point, float)
    n = np.asarray(normal, float)
    count = 0
    ex = TopExp_Explorer(shape, TopAbs_FACE)
    while ex.More():
        try:
            face = TopoDS.Face_s(ex.Current())
            s = BRepAdaptor_Surface(face, False)
            if s.GetType() == GeomAbs_Plane:
                pl = s.Plane()
                d = pl.Axis().Direction()
                fn = np.array([d.X(), d.Y(), d.Z()])
                if abs(float(fn @ n)) > 1.0 - _COPLANAR_COS_TOL:
                    o = pl.Location()
                    dist = abs(float((p - np.array([o.X(), o.Y(), o.Z()])) @ fn))
                    if dist <= max(_COPLANAR_DIST_TOL, float(BRep_Tool.Tolerance_s(face))):
                        count += 1
        except Exception as exc:
            log.debug("coplanar check skipped a face: %s", exc)
        ex.Next()
    return count


def _nudge_sign(shape, p, n) -> float:
    """+1 / -1: direction along ``n`` toward the shape's bbox centre (-1 on a tie)."""
    bb = _exact_bbox(shape)
    if bb is not None:
        d = float(np.dot(0.5 * (bb[0] + bb[1]) - p, n))
        if abs(d) > 1e-12:
            return math.copysign(1.0, d)
    return -1.0


def _normalise_loops(loops: List[np.ndarray]) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for outer, holes in nest_loops(loops):
        out.append(np.asarray(outer, float))
        out.extend(np.asarray(h, float) for h in holes)
    return out


# ----------------------------------------------------------------- body

def _half_space_cut(shape, point, normal):
    """Keep the material on the -normal side of the plane (the viewer is on +normal)."""
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeHalfSpace
    from OCP.gp import gp_Dir, gp_Pln, gp_Pnt
    p = np.asarray(point, float)
    n = np.asarray(normal, float)
    face = BRepBuilderAPI_MakeFace(gp_Pln(gp_Pnt(*p), gp_Dir(*n))).Face()
    ref = gp_Pnt(*(p + n))                        # a point inside the removed half
    cut = BRepAlgoAPI_Cut(shape, BRepPrimAPI_MakeHalfSpace(face, ref).Solid())
    if not cut.IsDone():
        raise RuntimeError("half-space cut failed")
    return cut.Shape()


# ----------------------------------------------------------------- API

def section_brep(shape, point, normal, spec, *, exact: bool, deflection: float,
                 depth_extent: float, hidden: bool = True) -> SectionView:
    """Cut ``shape`` with the plane (``point``, ``normal``) and view it with ``spec``.

    ``normal`` must equal ``spec.n`` (ValueError otherwise). A plane that
    lies in a planar face of the shape (it would return that face's outline,
    e.g. a boss footprint as a hole) or that gives zero closed loops is
    nudged ONCE by 1 % of ``depth_extent`` toward the shape's bbox centre.
    Returns a ``SectionView`` with ``loops`` (view 2D, outers CCW then their
    holes CW), ``body`` (``project_brep`` of the half behind the plane with
    ``exact`` / ``deflection`` / ``hidden``; None when there are no loops;
    ``hidden=False`` leaves out the hidden lines behind the cut, as ISO
    section views usually do) and ``stats``: nudged, nudge_mm, coplanar
    (faces in the original plane), open_wires_dropped, loops_recovered,
    dangling_segments_pruned, edges, wires, seconds, seconds_body.
    """
    from .brep_views import project_brep

    t0 = time.perf_counter()
    p = np.asarray(point, dtype=float).reshape(3)
    n = np.asarray(normal, dtype=float).reshape(3)
    nn = float(np.linalg.norm(n))
    if nn < 1e-12 or not np.all(np.isfinite(p)):
        raise ValueError(f"section_brep: bad plane point {point!r} / normal {normal!r}")
    n = n / nn
    if float(np.dot(n, np.asarray(spec.n, float))) < 1.0 - 1e-9:
        raise ValueError(f"section_brep: normal {n.tolist()} must equal the view normal "
                         f"{np.asarray(spec.n).tolist()} of {spec.name!r}")
    deflection = float(deflection)
    if not (deflection > 0.0 and math.isfinite(deflection)):
        raise ValueError(f"section_brep: deflection must be positive, got {deflection!r}")

    step = _NUDGE_FRACTION * abs(float(depth_extent))
    nudged = False
    nudge_mm = 0.0
    coplanar = _coplanar_face_count(shape, p, n) if _NUDGE_COPLANAR else 0
    if coplanar:
        nudge_mm = _nudge_sign(shape, p, n) * step
        p = p + nudge_mm * n
        nudged = True
        log.info("section plane lies in %d planar face(s); nudged by %.4g mm", coplanar, nudge_mm)
    try:
        loops, lstats = _section_loops(shape, p, n, spec, deflection)
    except Exception as exc:
        raise RuntimeError(f"section failed: {exc}") from exc
    loops = _normalise_loops(loops)
    if not loops and not nudged:
        nudge_mm = _nudge_sign(shape, p, n) * step
        p = p + nudge_mm * n
        nudged = True
        try:
            loops, lstats = _section_loops(shape, p, n, spec, deflection)
        except Exception as exc:
            raise RuntimeError(f"section failed after nudge: {exc}") from exc
        loops = _normalise_loops(loops)
        log.info("section plane had no closed loop; nudged by %.4g mm -> %d loop(s)",
                 nudge_mm, len(loops))

    body = None
    t_body = 0.0
    if loops:
        t1 = time.perf_counter()
        try:
            kept = _half_space_cut(shape, p, n)
        except Exception as exc:
            raise RuntimeError(f"section body cut failed: {exc}") from exc
        body = project_brep(kept, spec, exact=exact, deflection=deflection, hidden=bool(hidden))
        t_body = time.perf_counter() - t1

    stats = dict(lstats)
    stats.update(nudged=nudged, nudge_mm=nudge_mm, coplanar=int(coplanar), loops=len(loops),
                 seconds=round(time.perf_counter() - t0, 6), seconds_body=round(t_body, 6))
    marker = {"front": "top", "back": "top", "top": "front", "bottom": "front",
              "right": "front", "left": "front"}.get(spec.name, "top")
    return SectionView(spec=spec, plane_point=p, plane_normal=n.copy(), loops=loops, body=body,
                       marker_view=marker, stats=stats)


__all__ = ["choose_section_plane_brep", "section_brep"]
