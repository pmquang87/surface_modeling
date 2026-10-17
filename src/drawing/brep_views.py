"""Orthographic views of an OCCT shape by hidden-line removal (B-rep route).

``project_brep`` runs OCCT HLR for one ``ViewSpec`` and returns a
``ProjectedView`` of 2D primitives in view coordinates (model millimetres,
x = spec.x . p, y = spec.y . p, not shifted). Two algorithms:

* ``exact=True``: ``HLRBRep_Algo`` (``Add``, ``Projector``, ``Update``,
  ``Hide``, ``HLRBRep_HLRToShape``) for true B-rep input (STEP).
* ``exact=False``: ``HLRBRep_PolyAlgo`` (``Load``, ``Projector``,
  ``Update``, ``HLRBRep_PolyHLRToShape().Update(algo)``) for faceted shells
  from the STL route; the faces need a triangulation at least as fine as
  ``deflection`` (``prepare_faceted_shell``, done here automatically when
  missing or coarser; the shape is meshed in place).

Measured facts this module relies on (HLR probe, OCP 7.9.3, REPORT-hlr.md):

* ``HLRAlgo_Projector(gp_Ax2(P, N, Vx))`` shows the faces whose normals point
  toward +N; output 2D x = Vx.(p - P), y = (N x Vx).(p - P), Z = 0. The
  ``ViewSpec`` table (``projection.py``) uses exactly this convention.
* Drawing layers: visible = ``VCompound`` + ``OutLineVCompound``, hidden =
  ``HCompound`` + ``OutLineHCompound``. ``Rg1*`` / ``RgN*`` / ``IsoLine*``
  are dropped: fillet tangent edges are already coded G1 by the fillet API,
  sewing and the STEP reader, and the outline stays complete in
  ``VCompound`` (filleted box front view: dropping ``Rg1LineVCompound``
  removes only the tangent lines x = 3 / 57).
* HLR result edges carry only 2D pcurves: ``BRepLib.BuildCurve3d_s`` is
  called PER EDGE inside try/except (``BuildCurves3d_s`` on the compound
  raises ``Standard_NullObject`` when one edge has no geometry). An edge
  that still has no curve falls back to the chord of its vertices.
* Circle arcs: the parametric direction is clockwise in the view plane when
  the circle axis has Z < 0 (2 of the 4 fillet arcs); start and end are
  swapped then, otherwise the 270-degree complement would be drawn.
* Straight silhouettes and edge-on circles arrive as ``BSplineCurve``
  edges; a discretized curve whose samples all lie within
  ``0.1 * deflection`` of its chord is collapsed to a ``Segment``. Other
  curves (ellipses: a hole seen 30 degrees off its axis) become
  ``GCPnts_QuasiUniformDeflection`` polylines (r5 half ellipse at
  deflection 0.01: 33 points, max deviation 0.0059).
* ``HCompound`` repeats edges that lie exactly on visible ones (box back
  edges, the far mouth of a through hole, edge-on mouths in side views) and
  ``VCompound`` can hold exact duplicates (a fillet arc twice). Clean-up:
  ``dedupe_primitives``, ``drop_covered_curves`` (hidden circles/arcs on
  visible ones), ``subtract_covered`` (hidden segments under visible
  segments), ``merge_collinear`` (chains such as 0..5 and 5..55).
* Exact HLR must not be used on regularity-encoded faceted shells (it loses
  a third of the silhouette into ``Rg1``: ico2 16 of 24 edges); PolyAlgo
  with the DEFAULT ``TolAngular`` (0.001) gives exactly the 24 mesh
  silhouette edges (raising it to 0.35 shrinks the outline to +-19.02).
* Cost: exact HLR 0.4-200 ms per view on B-rep parts, 9.4 s on a 20k-face
  unencoded faceted compound; PolyAlgo 1.2 s + 2.6 s meshing at 20k
  triangles. ``time_budget_s`` is checked after ``Update``/``Hide`` (the
  OCCT call itself cannot be interrupted in-process; the sandbox wall clock
  is the hard limit).
"""
from __future__ import annotations

import logging
import math
import time
from typing import List, Optional

import numpy as np

from .geometry2d import (dedupe_primitives, drop_covered_curves, is_straight, merge_collinear,
                         subtract_covered)
from .model import Arc, BBox2D, Circle, LineStyle, Polyline, Primitive2D, ProjectedView, Segment

log = logging.getLogger(__name__)

_TWO_PI = 2.0 * math.pi

# Behaviour switches, module-level so negative-control tests can turn one off.
_SUBTRACT_COVERED = True   # drop hidden edges lying on visible ones
_SWAP_CW_ARCS = True       # swap arc ends when the circle axis Z < 0

_VISIBLE_COMPOUNDS = ("VCompound", "OutLineVCompound")
_HIDDEN_COMPOUNDS = ("HCompound", "OutLineHCompound")
_ALL_COMPOUNDS = ("VCompound", "OutLineVCompound", "Rg1LineVCompound", "RgNLineVCompound",
                  "HCompound", "OutLineHCompound", "Rg1LineHCompound", "RgNLineHCompound")


# ----------------------------------------------------------------- helpers

def _geo_tol(deflection: float) -> float:
    """Coincidence tolerance for the clean-up (HLR output is exact to ~1e-12)."""
    return max(1e-6, 0.01 * float(deflection))


def _straight_tol(deflection: float) -> float:
    return max(1e-9, 0.1 * float(deflection))


def _projector(spec):
    from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt
    from OCP.HLRAlgo import HLRAlgo_Projector
    origin, n, x = spec.gp_ax2_args()
    return HLRAlgo_Projector(gp_Ax2(gp_Pnt(*origin), gp_Dir(*n), gp_Dir(*x)))


def _is_null(shape) -> bool:
    try:
        return shape is None or shape.IsNull()
    except Exception:
        return True


def _edge_vertices_xy(edge) -> list:
    from OCP.BRep import BRep_Tool
    from OCP.TopAbs import TopAbs_VERTEX
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS
    pts = []
    ex = TopExp_Explorer(edge, TopAbs_VERTEX)
    while ex.More():
        p = BRep_Tool.Pnt_s(TopoDS.Vertex_s(ex.Current()))
        pts.append((p.X(), p.Y()))
        ex.Next()
    return pts


def _discretize(adaptor, u0: float, u1: float, deflection: float, angular: float = 0.1) -> list:
    """3D sample points of a curve adaptor; QuasiUniformDeflection, Tangential as fallback."""
    from OCP.GCPnts import GCPnts_QuasiUniformDeflection, GCPnts_TangentialDeflection
    try:
        q = GCPnts_QuasiUniformDeflection(adaptor, deflection, u0, u1)
        if q.IsDone() and q.NbPoints() >= 2:
            return [q.Value(i) for i in range(1, q.NbPoints() + 1)]
    except Exception as exc:
        log.debug("QuasiUniformDeflection failed (%s); using TangentialDeflection", exc)
    t = GCPnts_TangentialDeflection(adaptor, u0, u1, angular, deflection)
    return [t.Value(i) for i in range(1, t.NbPoints() + 1)]


def edges_to_primitives(compound, deflection: float, style: LineStyle) -> List[Primitive2D]:
    """HLR result compound (edges in the z = 0 view plane) -> 2D primitives.

    Line -> ``Segment``; Circle -> ``Circle`` (full turn) or ``Arc``
    (counter-clockwise start -> end in degrees, ends swapped when the circle
    axis Z < 0); anything else -> polyline sampled at ``deflection``,
    collapsed to a ``Segment`` when straight within ``0.1 * deflection``.
    Degenerated edges are skipped; an edge without usable geometry becomes
    the chord of its vertices. ``None`` or a null compound -> ``[]``.
    """
    from OCP.BRep import BRep_Tool
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.BRepLib import BRepLib
    from OCP.GeomAbs import GeomAbs_Circle, GeomAbs_Line
    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    prims: List[Primitive2D] = []
    if _is_null(compound):
        return prims
    deflection = float(deflection)
    if not (deflection > 0.0 and math.isfinite(deflection)):
        raise ValueError(f"edges_to_primitives: deflection must be positive, got {deflection!r}")
    min_len = 1e-9
    straight_tol = _straight_tol(deflection)
    ex = TopExp_Explorer(compound, TopAbs_EDGE)
    while ex.More():
        edge = TopoDS.Edge_s(ex.Current())
        ex.Next()
        try:
            if BRep_Tool.Degenerated_s(edge):
                continue
        except Exception:
            continue
        try:
            BRepLib.BuildCurve3d_s(edge)   # per edge: the compound-level call raises on one bad edge
            a = BRepAdaptor_Curve(edge)
            kind = a.GetType()
            u0, u1 = a.FirstParameter(), a.LastParameter()
        except Exception as exc:
            try:
                pts = _edge_vertices_xy(edge)
            except Exception:
                pts = []
            if len(pts) >= 2 and math.dist(pts[0], pts[-1]) > min_len:
                prims.append(Segment(pts[0], pts[-1], style))
            log.debug("edge without geometry (%s): chord used", exc)
            continue
        try:
            if kind == GeomAbs_Line:
                p0, p1 = a.Value(u0), a.Value(u1)
                if math.hypot(p1.X() - p0.X(), p1.Y() - p0.Y()) > min_len:
                    prims.append(Segment((p0.X(), p0.Y()), (p1.X(), p1.Y()), style))
                continue
            if kind == GeomAbs_Circle:
                c = a.Circle()
                ctr = c.Location()
                r = float(c.Radius())
                centre = (ctr.X(), ctr.Y())
                if abs((u1 - u0) - _TWO_PI) < 1e-9:
                    prims.append(Circle(centre, r, style))
                    continue
                ps, pe = a.Value(u0), a.Value(u1)
                ang_s = math.degrees(math.atan2(ps.Y() - ctr.Y(), ps.X() - ctr.X())) % 360.0
                ang_e = math.degrees(math.atan2(pe.Y() - ctr.Y(), pe.X() - ctr.X())) % 360.0
                if _SWAP_CW_ARCS and c.Axis().Direction().Z() < 0:   # parametric direction is CW
                    ang_s, ang_e = ang_e, ang_s
                prims.append(Arc(centre, r, ang_s, ang_e, style))
                continue
            pts3 = _discretize(a, u0, u1, deflection)
        except Exception as exc:
            log.debug("edge conversion failed (%s): skipped", exc)
            continue
        pts = np.array([[p.X(), p.Y()] for p in pts3], dtype=np.float64)
        if len(pts) < 2:
            continue
        chord = float(np.linalg.norm(pts[-1] - pts[0]))
        if chord > min_len and is_straight(pts, straight_tol):
            prims.append(Segment((float(pts[0, 0]), float(pts[0, 1])),
                                 (float(pts[-1, 0]), float(pts[-1, 1])), style))
            continue
        closed = chord <= min_len
        if closed:
            pts = pts[:-1]
        if len(pts) < 2 or (closed and len(pts) < 3):
            continue
        prims.append(Polyline(pts, closed=closed, style=style))
    return prims


def prepare_faceted_shell(shape, deflection: float):
    """Give every face a triangulation for PolyAlgo; returns ``shape`` itself.

    ``BRepMesh_IncrementalMesh(shape, deflection, False, 0.35, True)``
    (relative = False, angular 0.35 rad, parallel). MakeShapeOnMesh faces
    have no triangulation, and PolyAlgo returns EMPTY compounds without one;
    a hand-attached ``Poly_Triangulation`` does not work either (measured).
    """
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    BRepMesh_IncrementalMesh(shape, float(deflection), False, 0.35, True)
    return shape


def _needs_mesh(shape, deflection: float) -> bool:
    """True when some face has no triangulation or one coarser than ``deflection``.

    ``BRepTools.Triangulation_s(shape, deflection, False)`` is the check
    BRepMesh itself uses: it is False for a face without a triangulation and
    for one whose stored deflection exceeds the request (measured: a
    cylinder meshed at 0.5 -> ok at 0.5, not ok at 0.01; planar facets of a
    meshed faceted shell store 0 and stay ok at 1e-6). A mesh left over from
    an earlier step (importers / step_audit mesh at 0.1-0.2) therefore no
    longer sets the outline accuracy instead of the requested deflection.
    """
    from OCP.BRepTools import BRepTools
    try:
        return not BRepTools.Triangulation_s(shape, float(deflection), False)
    except Exception as exc:
        log.debug("triangulation check failed (%s): meshing", exc)
        return True


def _compounds_exact(shape, spec):
    from OCP.HLRBRep import HLRBRep_Algo, HLRBRep_HLRToShape
    algo = HLRBRep_Algo()
    algo.Add(shape)
    algo.Projector(_projector(spec))
    algo.Update()
    algo.Hide()
    return algo, HLRBRep_HLRToShape


def _compounds_poly(shape, spec):
    from OCP.HLRBRep import HLRBRep_PolyAlgo, HLRBRep_PolyHLRToShape
    algo = HLRBRep_PolyAlgo()
    algo.Load(shape)
    algo.Projector(_projector(spec))
    # keep the default TolAngular() == 0.001: 0.35 shrinks the outline (measured)
    algo.Update()
    return algo, HLRBRep_PolyHLRToShape


def _count_edges(compound) -> int:
    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopExp import TopExp_Explorer
    if _is_null(compound):
        return 0
    n = 0
    ex = TopExp_Explorer(compound, TopAbs_EDGE)
    while ex.More():
        n += 1
        ex.Next()
    return n


def _split(prims):
    segs = [p for p in prims if isinstance(p, Segment)]
    other = [p for p in prims if not isinstance(p, Segment)]
    return segs, other


def _seg_array(segs) -> np.ndarray:
    if not segs:
        return np.zeros((0, 2, 2))
    return np.array([[s.a, s.b] for s in segs], dtype=np.float64).reshape(-1, 2, 2)


def _to_segments(arr: np.ndarray, style: LineStyle) -> List[Segment]:
    return [Segment((float(s[0, 0]), float(s[0, 1])), (float(s[1, 0]), float(s[1, 1])), style)
            for s in np.asarray(arr).reshape(-1, 2, 2)]


def _polyline_edges(prims) -> np.ndarray:
    chunks = []
    for p in prims:
        if isinstance(p, Polyline):
            pts = np.asarray(p.points, float).reshape(-1, 2)
            if p.closed and len(pts) >= 3:
                chunks.append(np.stack([pts, np.roll(pts, -1, axis=0)], axis=1))
            elif len(pts) >= 2:
                chunks.append(np.stack([pts[:-1], pts[1:]], axis=1))
    if not chunks:
        return np.zeros((0, 2, 2))
    return np.concatenate(chunks)


def _clean(visible: list, hidden: list, tol: float):
    """Dedupe, drop/subtract hidden edges on visible ones, merge collinear segments."""
    visible = dedupe_primitives(visible, tol)
    hidden = dedupe_primitives(hidden, tol)
    vis_segs, vis_other = _split(visible)
    hid_segs, hid_other = _split(hidden)
    vis_arr = merge_collinear(_seg_array(vis_segs), tol) if vis_segs else np.zeros((0, 2, 2))
    hid_arr = _seg_array(hid_segs)
    if _SUBTRACT_COVERED:
        hid_other = drop_covered_curves(hid_other, vis_other, tol)
        if len(hid_arr):
            cover = np.concatenate([vis_arr, _polyline_edges(vis_other)]) if len(vis_other) else vis_arr
            hid_arr = subtract_covered(hid_arr, cover, tol)
    if len(hid_arr):
        hid_arr = merge_collinear(hid_arr, tol)
    vis_style = vis_segs[0].style if vis_segs else LineStyle.VISIBLE
    hid_style = hid_segs[0].style if hid_segs else LineStyle.HIDDEN
    return (_to_segments(vis_arr, vis_style) + vis_other,
            _to_segments(hid_arr, hid_style) + hid_other)


# ----------------------------------------------------------------- API

def project_brep(shape, spec, *, exact: bool, deflection: float, hidden: bool = True,
                 time_budget_s: Optional[float] = None) -> ProjectedView:
    """Hidden-line projection of ``shape`` for ``spec`` (a ``ViewSpec``).

    Returns a ``ProjectedView`` with ``visible`` (style VISIBLE) and
    ``hidden`` (style HIDDEN; ``[]`` when ``hidden=False``) primitives in view
    coordinates, ``bbox`` over both (None when nothing was drawn) and
    ``algorithm`` "hlr_exact" or "hlr_poly". ``stats``: seconds (hlr,
    extract, total), raw edge counts per compound, primitive counts.

    Raises ``TimeoutError`` when ``time_budget_s`` is exceeded (checked after
    ``Update``/``Hide``), ``ValueError`` for a non-positive deflection and
    ``RuntimeError`` when OCCT fails.
    """
    deflection = float(deflection)
    if not (deflection > 0.0 and math.isfinite(deflection)):
        raise ValueError(f"project_brep: deflection must be positive, got {deflection!r}")
    algorithm = "hlr_exact" if exact else "hlr_poly"
    t0 = time.perf_counter()
    stats: dict = {"meshed": False}
    if _is_null(shape):
        stats.update(seconds=0.0, empty_shape=True)
        return ProjectedView(spec, [], [], None, algorithm, stats)

    try:
        if exact:
            algo, to_shape = _compounds_exact(shape, spec)
        else:
            if _needs_mesh(shape, deflection):
                prepare_faceted_shell(shape, deflection)
                stats["meshed"] = True
            algo, to_shape = _compounds_poly(shape, spec)
    except Exception as exc:
        raise RuntimeError(f"{algorithm} failed for view {spec.name!r}: {exc}") from exc
    t_hlr = time.perf_counter() - t0
    if time_budget_s is not None and t_hlr > float(time_budget_s):
        raise TimeoutError(f"{algorithm} for view {spec.name!r} took {t_hlr:.3f} s "
                           f"> budget {float(time_budget_s):.3f} s")

    t1 = time.perf_counter()
    try:
        if exact:
            h = to_shape(algo)
        else:
            h = to_shape()
            h.Update(algo)
        comps = {}
        for name in _ALL_COMPOUNDS:
            fn = getattr(h, name, None)
            c = fn() if fn is not None else None
            comps[name] = None if _is_null(c) else c
    except Exception as exc:
        raise RuntimeError(f"{algorithm} result extraction failed for view {spec.name!r}: {exc}") from exc
    stats["edge_counts"] = {k: _count_edges(v) for k, v in comps.items()}

    visible: list = []
    for name in _VISIBLE_COMPOUNDS:
        visible += edges_to_primitives(comps[name], deflection, LineStyle.VISIBLE)
    hidden_prims: list = []
    if hidden:
        for name in _HIDDEN_COMPOUNDS:
            hidden_prims += edges_to_primitives(comps[name], deflection, LineStyle.HIDDEN)
    stats["raw_visible"] = len(visible)
    stats["raw_hidden"] = len(hidden_prims)
    visible, hidden_prims = _clean(visible, hidden_prims, _geo_tol(deflection))

    bbox = BBox2D.of_primitives(visible + hidden_prims) if (visible or hidden_prims) else None
    t_end = time.perf_counter()
    stats.update(seconds=round(t_end - t0, 6), seconds_hlr=round(t_hlr, 6),
                 seconds_extract=round(t_end - t1, 6), n_visible=len(visible),
                 n_hidden=len(hidden_prims))
    log.debug("%s view %s: %d visible, %d hidden, %.3f s", algorithm, spec.name,
              len(visible), len(hidden_prims), t_end - t0)
    return ProjectedView(spec, visible, hidden_prims, bbox, algorithm, stats)


__all__ = ["project_brep", "prepare_faceted_shell", "edges_to_primitives"]
