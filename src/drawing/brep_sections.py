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
  toward the shape's bbox centre (``stats["nudged"]``); a plane inside the
  bbox that is still empty (a gap between bodies) is moved through the
  centroid of the largest body (``stats["retried_at_body"]``); still empty
  -> ``loops == []`` and no body (the pipeline flags ``section_empty``).
* QA round 1 (assembly as1_pe_203, plate + bolts + nuts): ONE
  ``BRepAlgoAPI_Cut`` of the whole compound against the half-space kept
  whole bolts and nuts from the removed half (IsDone True, no error), and
  one ``BRepAlgoAPI_Section`` of the compound merged the coincident wires
  of a bolt and its hole into open wires. Synthetic plate + bolt + nut cut
  at y = 10: kept solids reached ymin 5 and 2, 1 merged loop with 2 open
  wires; per body: every kept solid at ymin 10, 4 closed loops. Both the
  loops and the body are therefore built body by body, and bodies entirely
  on one side of the plane are kept or dropped by their bbox without a
  boolean.
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
_SECTION_PER_BODY = True   # section and half-space-cut each solid / shell on its own
_RETRY_AT_BODY = True      # an empty plane is retried through the centroid of a body

_SIDE_TOL_MIN = 1e-6       # mm; a body this close to the plane counts as on it
_SIDE_TOL_REL = 1e-9       # x bbox diagonal
_RETRY_BODIES = 3          # largest bodies tried by the empty-plane retry
_EXCESS_TOL_REL = 1e-4     # x bbox diagonal: a cut body may overshoot the plane by this

_WIRE_TOL = 1e-4           # ConnectEdgesToWires tolerance (probe value)
_NUDGE_FRACTION = 0.01
_COPLANAR_COS_TOL = 1e-9   # |n_face . n| > 1 - this -> parallel
_COPLANAR_DIST_TOL = 1e-6  # mm; the face tolerance is used when larger

# mode -> (section view name, marker view name)
_MODES = {"auto": ("front", "top"), "front": ("front", "top"),
          "top": ("top", "front"), "right": ("right", "front")}
_PERP_TOL = 1e-6           # |axis . normal| below this -> the hole axis lies in the plane
_THIN_RATIO = 0.1          # auto: depth < this x min(width, height) -> thin part, top plane


# ----------------------------------------------------------------- plane choice

def _exact_bbox(shape) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """The ``AddOptimal_s`` box, by ``brep_load.exact_bbox_bounded`` (the same box,
    integrating only the faces near the extremes): AddOptimal on the bodies near
    the plane cost 18.6 of the 39.7 s of the RC buggy section (F7-perf)."""
    try:
        from .brep_load import exact_bbox_bounded
        if shape is None or shape.IsNull():
            return None
        bb = exact_bbox_bounded(shape)
        if bb is None:
            return None
        return np.array(bb[0], float), np.array(bb[1], float)
    except Exception as exc:
        log.debug("exact bbox failed: %s", exc)
        return None


def _fast_bbox(shape) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Conservative bbox by ``BRepBndLib.Add_s`` (control hull / existing
    triangulation, enlarged by the shape tolerance): never smaller than the
    shape, much cheaper than ``AddOptimal_s`` on B-spline bodies. Measured on
    the 211-solid, 10665-face QA assembly (RC buggy suspension): the exact
    box of every body took 32 s per pass, which made the per-body section
    run out of time; classification only needs a box that is never too
    small (a body wrongly seen as straddling is just cut, correctly)."""
    try:
        from OCP.Bnd import Bnd_Box
        from OCP.BRepBndLib import BRepBndLib
        if shape is None or shape.IsNull():
            return None
        box = Bnd_Box()
        BRepBndLib.Add_s(shape, box, True)
        if box.IsVoid():
            return None
        x0, y0, z0, x1, y1, z1 = box.Get()
        return np.array([x0, y0, z0], float), np.array([x1, y1, z1], float)
    except Exception as exc:
        log.debug("fast bbox failed: %s", exc)
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


_SECTION_HOLE_KINDS = ("hole", "filled_hole")   # filled_hole: a fastener in an assembly hole


def _holes_by_size(features: Sequence[CylFeature], centre: np.ndarray, with_kind: bool = False
                   ) -> List[Tuple]:
    """(unit axis, centre, radius) of the hole features (``kind`` "hole" and
    "filled_hole", see ``_SECTION_HOLE_KINDS``), largest first; a filled hole
    comes after a real hole of the same radius; then nearest the bbox centre
    first. ``with_kind=True`` appends the kind to each tuple.

    A hole filled by a bolt or pin of another solid (``brep_features``
    marks it "filled_hole" in an assembly) is a valid section axis: the cut
    through the fastener axis is the standard section of a bolted joint, and
    the per-body cut sections it cleanly (QA as1_pe_203: 13 closed loops over
    7 bodies, 0 open wires, against 2 loops on the bbox-centre fallback)."""
    out = []
    for f in features or ():
        kind = getattr(f, "kind", "")
        if kind not in _SECTION_HOLE_KINDS:
            continue
        axis = np.asarray(f.axis, float).reshape(3)
        na = float(np.linalg.norm(axis))
        if na < 1e-12:
            continue
        c = np.asarray(f.center, float).reshape(3)
        out.append((axis / na, c, float(f.radius), kind))
    out.sort(key=lambda h: (-round(h[2], 6), h[3] != "hole", float(np.linalg.norm(h[1] - centre))))
    return out if with_kind else [h[:3] for h in out]


def _perpendicular(axis: np.ndarray, normal: np.ndarray) -> bool:
    return abs(float(np.dot(axis, normal))) <= _PERP_TOL


def choose_section_plane_brep_info(model, features: Sequence[CylFeature], mode: str
                                   ) -> Tuple[np.ndarray, np.ndarray, str, dict]:
    """``(point, normal, marker_view, info)`` of the cutting plane.

    Explicit modes: "front" (normal = front view n = (0,-1,0), cutting line
    shown in the top view), "top" (normal (0,0,1), shown in the front view),
    "right" (normal (1,0,0), shown in the front view). The plane goes through
    the axis of the largest hole (``kind == "hole"``) whose axis is
    perpendicular to the normal (the point is the hole centre); without one
    it goes through the bbox centre.

    "auto" ranks the normals front, top, right. The holes are tried largest
    first; the first normal perpendicular to the hole's axis wins, so a plate
    whose big hole runs along the front normal (a washer, a flange seen face
    on) is cut by the TOP plane through that hole and the cutting line shows
    in the front view (a front plane would repeat the front outline). With no
    usable hole a thin part (depth along the front normal < ``_THIN_RATIO``
    x the smaller of the other two extents) is cut by the top plane through
    the bbox centre; anything else by the front plane through it.

    A "filled_hole" (a hole an assembly fastener fills) is a section axis
    too, ranked after a real hole of the same radius (``_holes_by_size``).

    ``info``: ``reason`` ("hole_axis" | "thin_part" | "bbox_centre_fallback"),
    ``view`` (the section view name), ``mode``, ``hole_radius`` (None without
    a hole), ``hole_kind`` ("hole" | "filled_hole" | None) and ``skip_hint``
    ("no_solid" when ``census["solids"] == 0``: an open surface model gives
    no closed section loop on any plane, so an auto section is pointless;
    None otherwise; the plane is still returned). ``model`` needs
    ``census["bbox_min"/"bbox_max"]`` or a ``shape`` (exact ``AddOptimal_s``
    bbox then).
    """
    if mode not in _MODES:
        raise ValueError(f"section mode must be one of {sorted(_MODES)}, got {mode!r}")
    lo, hi = _model_bbox(model)
    centre = 0.5 * (lo + hi)
    holes = _holes_by_size(features, centre, with_kind=True)
    names = ("front", "top", "right") if mode == "auto" else (_MODES[mode][0],)
    chosen = None
    for axis, c, r, kind in holes:
        for name in names:
            if _perpendicular(axis, np.asarray(VIEWS[name].n, float)):
                chosen = (name, c.copy(), r, kind)
                break
        if chosen is not None:
            break
    hole_kind = None
    if chosen is not None:
        view_name, point, radius, hole_kind = chosen
        reason = "hole_axis"
    else:
        radius = None
        point = centre.copy()
        view_name = names[0]
        reason = "bbox_centre_fallback"
        if mode == "auto":
            ext = np.abs(hi - lo)
            depth = float(abs(np.dot(ext, VIEWS["front"].n)))
            others = [float(abs(np.dot(ext, VIEWS[k].n))) for k in ("top", "right")]
            if min(others) > 0.0 and depth < _THIN_RATIO * min(others):
                view_name, reason = "top", "thin_part"
    normal = np.array(VIEWS[view_name].n, dtype=float)
    marker = _MODES[view_name][1]
    census = getattr(model, "census", None) or {}
    solids = census.get("solids")
    no_solid = isinstance(solids, (int, np.integer)) and not isinstance(solids, bool) and solids == 0
    info = {"reason": reason, "view": view_name, "mode": mode, "hole_radius": radius,
            "hole_kind": hole_kind, "skip_hint": "no_solid" if no_solid else None}
    log.debug("section plane %s: point %s normal %s (%s)", mode, point.tolist(), normal.tolist(),
              reason)
    return point, normal, marker, info


def choose_section_plane_brep(model, features: Sequence[CylFeature], mode: str
                              ) -> Tuple[np.ndarray, np.ndarray, str]:
    """(point, normal, marker_view) of the cutting plane; see
    ``choose_section_plane_brep_info`` for the rules and the reason."""
    point, normal, marker, _ = choose_section_plane_brep_info(model, features, mode)
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


def _parts(shape) -> List:
    """The bodies of ``shape`` that are sectioned and cut one by one: every
    solid, every free shell (not in a solid), and ONE compound of all free
    faces (not in a shell; a faceted or surface model can have thousands).
    Loose edges and vertices carry no material and are left out. A null
    shape gives []."""
    from OCP.BRep import BRep_Builder
    from OCP.TopAbs import TopAbs_FACE, TopAbs_SHELL, TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopTools import TopTools_MapOfShape
    from OCP.TopoDS import TopoDS_Compound
    if shape is None or shape.IsNull():
        return []

    def explore(kind, avoid=None):
        seen = TopTools_MapOfShape()
        out = []
        ex = TopExp_Explorer(shape, kind) if avoid is None else TopExp_Explorer(shape, kind, avoid)
        while ex.More():
            if seen.Add(ex.Current()):
                out.append(ex.Current())
            ex.Next()
        return out

    parts = explore(TopAbs_SOLID) + explore(TopAbs_SHELL, TopAbs_SOLID)
    faces = explore(TopAbs_FACE, TopAbs_SHELL)
    if faces:
        b = BRep_Builder()
        comp = TopoDS_Compound()
        b.MakeCompound(comp)
        for f in faces:
            b.Add(comp, f)
        parts.append(comp)
    return parts


def _box_range(bb, point, normal) -> Optional[Tuple[float, float]]:
    if bb is None:
        return None
    lo, hi = bb
    n = np.asarray(normal, float)
    base = float(np.dot(np.asarray(point, float), n))
    d_lo = float(np.dot(np.where(n >= 0, lo, hi), n)) - base
    d_hi = float(np.dot(np.where(n >= 0, hi, lo), n)) - base
    return d_lo, d_hi


class _Bodies:
    """The bodies of a shape (``_parts``) with their boxes cached, so the
    loops, the nudge / retry planes and the body cut share them.

    ``range(i, point, normal)`` is the (min, max) signed distance of body i
    to the plane along ``normal`` (None for a void box). The cheap
    conservative box (``_fast_bbox``) answers first; only when it straddles
    the plane (beyond ``tol`` on both sides) is the exact ``AddOptimal_s``
    box computed (once) and used, so only bodies near the plane pay for it.
    ``tol`` is the on-the-plane tolerance and ``diag`` the bbox diagonal of
    the whole shape (from the fast box).
    """

    def __init__(self, shape) -> None:
        self.parts = _parts(shape)
        self._fast = [_fast_bbox(p) for p in self.parts]
        self._exact: List = [None] * len(self.parts)
        self._has_exact = [False] * len(self.parts)
        fb = _fast_bbox(shape)
        self.diag = float(np.linalg.norm(fb[1] - fb[0])) if fb is not None else 0.0
        self.tol = max(_SIDE_TOL_MIN, _SIDE_TOL_REL * self.diag)

    def range(self, i: int, point, normal) -> Optional[Tuple[float, float]]:
        rng = _box_range(self._fast[i], point, normal)
        if rng is None or (rng[0] < -self.tol and rng[1] > self.tol):
            if not self._has_exact[i]:
                self._exact[i] = _exact_bbox(self.parts[i])
                self._has_exact[i] = True
            exact_rng = _box_range(self._exact[i], point, normal)
            if exact_rng is not None:
                return exact_rng
        return rng


def _section_part_edges(part, point, normal) -> List:
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Section
    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopExp import TopExp_Explorer
    from OCP.gp import gp_Dir, gp_Pln, gp_Pnt
    sec = BRepAlgoAPI_Section(part, gp_Pln(gp_Pnt(*map(float, point)), gp_Dir(*map(float, normal))),
                              False)
    sec.ComputePCurveOn1(True)
    sec.Approximation(True)
    sec.Build()
    if not sec.IsDone():
        raise RuntimeError("BRepAlgoAPI_Section did not finish")
    out = []
    ex = TopExp_Explorer(sec.Shape(), TopAbs_EDGE)
    while ex.More():
        out.append(ex.Current())
        ex.Next()
    return out


def _section_loops(shape, point, normal, spec, deflection: float, bodies=None,
                   skip=None, ids_out: Optional[list] = None):
    """Closed section loops in view 2D + stats for one plane.

    Each body (``_parts``) whose bbox reaches the plane is sectioned on its
    own and its edges are joined into wires on their own, so coincident
    faces of touching bodies (a bolt in its hole, a nut on a bolt) do not
    merge their wires into open ones. ``stats["bodies_sectioned"]`` counts
    them. ``bodies`` (a ``_Bodies`` of ``shape``) shares the box cache.
    Loops are recovered from open wires body by body too.

    ``skip`` (indices into ``_parts``) are not sectioned (fasteners, QA
    round 2 issue 28); ``stats["bodies_skipped"]`` counts those the plane
    reaches. ``ids_out`` (a list) receives the ``_parts`` index of each
    returned loop (QA round 2 issue 3: ``SectionView.loop_body``)."""
    from OCP.BRep import BRep_Tool
    from OCP.ShapeAnalysis import ShapeAnalysis_FreeBounds
    from OCP.TopTools import TopTools_HSequenceOfShape
    from OCP.TopoDS import TopoDS

    stats = {"edges": 0, "wires": 0, "open_wires_dropped": 0, "dangling_segments_pruned": 0,
             "loops_recovered": 0, "bodies_sectioned": 0, "bodies_skipped": 0}
    skip = set(skip or ())
    groups: List[Tuple[int, List]] = []
    if _SECTION_PER_BODY:
        if bodies is None:
            bodies = _Bodies(shape)
        parts = bodies.parts
    else:
        parts = [shape]
    for i, part in enumerate(parts):
        if _SECTION_PER_BODY and len(parts) > 1:
            rng = bodies.range(i, point, normal)
            if rng is None or rng[0] > bodies.tol or rng[1] < -bodies.tol:
                continue                      # the plane misses this body
            if i in skip:
                stats["bodies_skipped"] += 1
                continue
        edges = _section_part_edges(part, point, normal)
        stats["bodies_sectioned"] += 1
        if edges:
            groups.append((i, edges))
    if not groups:
        return [], stats
    basis = np.stack([np.asarray(spec.x, float), np.asarray(spec.y, float)], axis=1)   # (3, 2)
    loops2: List[np.ndarray] = []
    ids: List[int] = []
    for body_id, edges in groups:
        seq = TopTools_HSequenceOfShape()
        for e in edges:
            seq.Append(e)
        stats["edges"] += seq.Length()
        wires = TopTools_HSequenceOfShape()
        ShapeAnalysis_FreeBounds.ConnectEdgesToWires_s(seq, _WIRE_TOL, False, wires)
        stats["wires"] += wires.Length()
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
                    loops2.append(P @ basis)
                    ids.append(body_id)
            else:
                stats["open_wires_dropped"] += 1
                if _RECOVER_OPEN:
                    open_segs.append(_wire_segments(w, deflection))
        if open_segs:
            segs3 = np.concatenate(open_segs)
            kept, pruned = _prune_dangling(segs3, _WIRE_TOL)
            stats["dangling_segments_pruned"] += pruned
            if len(kept):
                rec, n_open = chain_segments(kept @ basis, _WIRE_TOL)
                rec = [L for L in rec if len(L) >= 3]
                stats["loops_recovered"] += len(rec)
                loops2 += rec
                ids += [body_id] * len(rec)
    if stats["open_wires_dropped"]:
        log.info("section: %d open wire(s) dropped, %d loop(s) recovered from them",
                 stats["open_wires_dropped"], stats["loops_recovered"])
    if ids_out is not None:
        ids_out[:] = ids
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
    bb = _fast_bbox(shape)
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


_MERGE_MAX_LOOPS = 200     # the containment test is O(n^2) in the loop count
_SMOOTH_TURN = 0.5         # rad: a sharper vertex is a corner, not a discretised curve


def _loop_sagitta(a: np.ndarray) -> float:
    """Largest chord-to-curve deviation of a closed polygon that discretises
    curves: per edge ``L x theta / 8`` (the sagitta of a chord L on a circle
    of radius L / theta), theta the mean turning angle of its smooth end
    vertices (turn <= ``_SMOOTH_TURN``; corners are not curves). 0 for a
    polygon of corners only. Straight-through vertices (turn < 1e-6 rad: a
    mesh cut puts one on every chord where it crosses a facet diagonal) are
    dropped first, else they halve the chords and the estimate."""
    a = np.asarray(a, float).reshape(-1, 2)
    if len(a) < 3:
        return 0.0
    for _ in range(2):
        e = np.roll(a, -1, axis=0) - a
        prev = np.roll(e, 1, axis=0)
        cross = prev[:, 0] * e[:, 1] - prev[:, 1] * e[:, 0]
        dot = np.einsum("ij,ij->i", prev, e)
        keep = (np.abs(np.arctan2(cross, dot)) >= 1e-6) & (np.einsum("ij,ij->i", e, e) > 0.0)
        if keep.all() or keep.sum() < 3:
            break
        a = a[keep]
    e = np.roll(a, -1, axis=0) - a                       # edge i: vertex i -> i + 1
    L = np.linalg.norm(e, axis=1)
    prev = np.roll(e, 1, axis=0)                         # edge into vertex i
    cross = prev[:, 0] * e[:, 1] - prev[:, 1] * e[:, 0]
    dot = np.einsum("ij,ij->i", prev, e)
    turn = np.abs(np.arctan2(cross, dot))                # at vertex i
    smooth = turn <= _SMOOTH_TURN
    t0, t1 = np.where(smooth, turn, 0.0), np.roll(np.where(smooth, turn, 0.0), -1)
    k = smooth.astype(float) + np.roll(smooth, -1).astype(float)
    theta = np.where(k > 0, (t0 + t1) / np.maximum(k, 1.0), 0.0)
    return float(np.max(L * theta) / 8.0)


def _max_dist_to_polygon(pts: np.ndarray, poly: np.ndarray, chunk: int = 2048) -> float:
    """Largest distance of ``pts`` to the closed polygon ``poly`` (its edges)."""
    a = poly
    b = np.roll(poly, -1, axis=0)
    ab = b - a
    ab2 = np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-300)
    worst = 0.0
    for s in range(0, len(pts), chunk):
        p = pts[s:s + chunk, None, :]                    # (m, 1, 2)
        t = np.clip(np.einsum("mkj,kj->mk", p - a[None], ab) / ab2, 0.0, 1.0)
        q = a[None] + t[..., None] * ab[None]
        d = np.sqrt(np.min(np.sum((p - q) ** 2, axis=2), axis=1))
        worst = max(worst, float(d.max()))
    return worst


def merge_void_bodies(loops: Sequence[np.ndarray], ids: Sequence[int], is_void,
                      coincide_tol: Optional[float] = None) -> List[int]:
    """Body ids with void bodies joined to the body around them (QA round 2 issue 3).

    An internal cavity can arrive as a body of its own: the faceted STL
    route makes the cavity shell a separate solid (healed to a positive
    volume: 48000 + 2000 mm3 measured for a 60 x 40 x 20 box with a 20 x 10
    x 10 cavity), a mesh has it as a separate (negative-volume) component.
    Its section loop is a hole of the outer body; as a body of its own the
    layout would nest it alone and fill the void. A body whose loops ALL
    lie at odd containment depth (holes only) is a void when
    ``is_void(body_id)`` is True (negative signed volume) or when none of
    its loops coincides with a loop of another body (it sits in the
    material of the body around it: a healed cavity or an interference); it
    takes the body of the smallest loop containing it one level up. Bodies
    with an outer loop, and bodies whose loop coincides with a hole of
    another body (an exact-fit bolt seen across its axis: equal areas, so
    the bolt loop is not inside the hole loop), keep their id. Runs only
    with more than one body and at most ``_MERGE_MAX_LOOPS`` loops; ``ids``
    returned unchanged otherwise.

    Two loops coincide when every vertex of each lies within a distance
    tolerance of the other polygon (symmetric Hausdorff on the vertices;
    after a cheap area / bbox filter). The tolerance is the larger of
    ``coincide_tol`` (``section_brep`` passes 2 x deflection + the wire
    tolerance), 2 x the larger ``_loop_sagitta`` of the pair (an
    estimate from the polygons themselves, which covers mesh cuts) and
    1e-6 x the coordinate scale. A bolt modelled with its seam 37 deg away
    from the hole's has an equal-area loop whose bbox differs by about the
    sagitta (review of issue 3: the former 1e-6 x scale bbox test joined it
    to the plate and drew it white)."""
    from .geometry2d import _loop_inside, polygon_area

    n = len(loops)
    ids = list(ids)
    if n < 2 or len(ids) != n or len(set(ids)) < 2 or n > _MERGE_MAX_LOOPS:
        return ids
    arrs = [np.asarray(L, float).reshape(-1, 2) for L in loops]
    areas = [abs(polygon_area(a)) if len(a) >= 3 else 0.0 for a in arrs]
    scale = max(1.0, max(float(np.abs(a).max()) if a.size else 0.0 for a in arrs))
    eps = 1e-9 * scale
    inside = np.zeros((n, n), dtype=bool)
    for j in range(n):
        for i in range(n):
            if i != j and areas[j] > areas[i] > 0.0:
                try:
                    inside[i, j] = _loop_inside(arrs[i], arrs[j], eps)
                except Exception:
                    inside[i, j] = False
    depth = inside.sum(axis=1)
    has_outer = {ids[i] for i in range(n) if depth[i] % 2 == 0}
    cand = {ids[i] for i in range(n)} - has_outer
    boxes = [np.concatenate([a.min(axis=0), a.max(axis=0)]) if a.size else np.zeros(4)
             for a in arrs]
    base = max(1e-6 * scale, float(coincide_tol) if coincide_tol else 0.0)
    sag: dict = {}
    perim: dict = {}

    def sagitta(i: int) -> float:
        if i not in sag:
            try:
                sag[i] = _loop_sagitta(arrs[i])
            except Exception:
                sag[i] = 0.0
        return sag[i]

    def perimeter(i: int) -> float:
        if i not in perim:
            a = arrs[i]
            perim[i] = float(np.linalg.norm(np.roll(a, -1, axis=0) - a, axis=1).sum()) if len(a) else 0.0
        return perim[i]

    def pair(i: int, k: int) -> bool:
        tol = max(base, 2.0 * max(sagitta(i), sagitta(k)) + base)
        if float(np.max(np.abs(boxes[k] - boxes[i]))) > tol:
            return False
        if abs(areas[k] - areas[i]) > (2.0 * max(perimeter(i), perimeter(k)) * tol
                                       + 1e-6 * max(areas[i], 1.0)):
            return False
        if len(arrs[i]) < 3 or len(arrs[k]) < 3:
            return False
        try:
            return (_max_dist_to_polygon(arrs[i], arrs[k]) <= tol
                    and _max_dist_to_polygon(arrs[k], arrs[i]) <= tol)
        except Exception as exc:
            log.debug("loop coincidence check failed: %s", exc)
            return False

    def coincides(i: int) -> bool:
        return any(ids[k] != ids[i] and pair(i, k) for k in range(n))

    voids = set()
    for b in cand:
        mine = [i for i in range(n) if ids[i] == b]
        if not any(coincides(i) for i in mine):
            voids.add(b)
            continue
        try:
            if is_void(b):
                voids.add(b)
        except Exception as exc:
            log.debug("void check of body %s failed: %s", b, exc)
    out = list(ids)
    for i in np.argsort(depth, kind="stable"):
        if ids[i] not in voids:
            continue
        parents = [j for j in range(n) if inside[i, j] and depth[j] == depth[i] - 1]
        if parents:
            out[i] = out[min(parents, key=lambda j: areas[j])]
    return out


def _part_is_void(part) -> bool:
    """True for a solid with a negative signed volume (an inward-wound cavity shell)."""
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    from OCP.TopAbs import TopAbs_SOLID
    if part.ShapeType() != TopAbs_SOLID:
        return False
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(part, props)
    return float(props.Mass()) < 0.0


def _normalise_loops_by_body(loops: List[np.ndarray], ids: Sequence[int], is_void=None,
                             coincide_tol: Optional[float] = None
                             ) -> Tuple[List[np.ndarray], List[int]]:
    """``_normalise_loops`` body by body: the loops of each body id (in the
    order of first appearance) are nested on their own, so a hole belongs to
    an outer of its own body. Returns the loops and the body id per loop.
    ``ids`` not aligned with ``loops`` -> all loops nested as one body 0.
    ``is_void`` (``body id -> bool``) first joins void bodies to the body
    around them (``merge_void_bodies`` with ``coincide_tol``)."""
    if len(ids) != len(loops):
        out = _normalise_loops(loops)
        return out, [0] * len(out)
    if is_void is not None:
        ids = merge_void_bodies(loops, ids, is_void, coincide_tol=coincide_tol)
    order: List[int] = []
    by: dict = {}
    for lp, b in zip(loops, ids):
        if b not in by:
            by[b] = []
            order.append(b)
        by[b].append(lp)
    out: List[np.ndarray] = []
    out_ids: List[int] = []
    for b in order:
        got = _normalise_loops(by[b])
        out += got
        out_ids += [int(b)] * len(got)
    return out, out_ids


# ----------------------------------------------------------------- body

def _half_space_solid(p: np.ndarray, n: np.ndarray, removed: bool = True):
    """Half-space solid bounded by the plane (p, n): the removed half (+n side)
    by default, the kept half (-n side) with ``removed=False``."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeHalfSpace
    from OCP.gp import gp_Dir, gp_Pln, gp_Pnt
    face = BRepBuilderAPI_MakeFace(gp_Pln(gp_Pnt(*p), gp_Dir(*n))).Face()
    ref = gp_Pnt(*(p + n)) if removed else gp_Pnt(*(p - n))
    return BRepPrimAPI_MakeHalfSpace(face, ref).Solid()


def _cut_one(part, removed_half):
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    cut = BRepAlgoAPI_Cut(part, removed_half)
    if not cut.IsDone():
        raise RuntimeError("half-space cut failed")
    return cut.Shape()


def _common_one(part, kept_half):
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
    com = BRepAlgoAPI_Common(part, kept_half)
    if not com.IsDone():
        raise RuntimeError("half-space common failed")
    return com.Shape()


def _half_space_cut(shape, point, normal, stats: Optional[dict] = None, bodies=None,
                    keep_whole=None):
    """Keep the material on the -normal side of the plane (the viewer is on +normal).

    Each body (``_parts``: solid, free shell, the free faces) is classified
    by its bbox against the plane (``_Bodies``; ``bodies`` shares the box
    cache of the loops): entirely on the kept side -> kept
    whole, entirely on the removed side -> dropped, straddling -> cut on its
    own. One boolean of a whole assembly compound (bolts in holes, nuts on
    bolts: coincident faces between arguments) kept whole solids from the
    removed half. Every cut result is checked against the half-space; a
    result that reaches past the plane by more than the tolerance is redone
    with ``BRepAlgoAPI_Common`` against the kept half-space, and the better
    of the two is kept. ``stats`` (optional dict) receives kept_whole,
    dropped_whole, cut, common_retries, half_space_excess_mm (largest
    distance a kept body reaches past the plane, checked on a triangulation
    when the AddOptimal box says it overshoots, see ``_excess``; 0 when
    clean), half_space_excess_tol_mm (the verification tolerance: tolerance
    or 1e-4 x bbox diagonal) and excess_bodies (kept bodies still past the
    plane by more than that after the Common retry; they are kept, so the
    caller can flag an inexact cut when this is > 0).

    ``keep_whole`` (indices into ``_parts``; fasteners, QA round 2 issue
    28): such a body that straddles the plane is kept whole instead of cut
    (ISO: bolts, pins and shafts are not sectioned lengthwise); one entirely
    on the removed side is still dropped. ``stats["fasteners_unsectioned"]``
    lists the straddling ones kept whole.
    """
    from OCP.BRep import BRep_Builder
    from OCP.TopoDS import TopoDS_Compound
    p = np.asarray(point, float)
    n = np.asarray(normal, float)
    st = {"kept_whole": 0, "dropped_whole": 0, "cut": 0, "common_retries": 0,
          "half_space_excess_mm": 0.0, "half_space_excess_tol_mm": 0.0, "excess_bodies": 0,
          "fasteners_unsectioned": []}
    keep_whole = set(keep_whole or ())
    if not _SECTION_PER_BODY:
        out = _cut_one(shape, _half_space_solid(p, n))
        if stats is not None:
            st["cut"] = 1
            stats.update(st)
        return out
    if bodies is None:
        bodies = _Bodies(shape)
    tol = bodies.tol
    vtol = max(tol, _EXCESS_TOL_REL * bodies.diag)   # verification: tolerance-sized overshoot is ok
    st["half_space_excess_tol_mm"] = float(vtol)
    removed = None
    kept_half = None
    kept = []
    for i, part in enumerate(bodies.parts):
        rng = bodies.range(i, p, n)
        if rng is None:
            continue
        d_lo, d_hi = rng
        if d_hi <= tol:
            kept.append(part)
            st["kept_whole"] += 1
            continue
        if d_lo >= -tol:
            st["dropped_whole"] += 1
            continue
        if i in keep_whole and len(bodies.parts) > 1:
            kept.append(part)                  # a fastener on the plane: not sectioned
            st["kept_whole"] += 1
            st["fasteners_unsectioned"].append(int(i))
            continue
        if removed is None:
            removed = _half_space_solid(p, n)
        res = _cut_one(part, removed)
        excess = _excess(res, p, n, vtol)
        if excess > vtol:
            if kept_half is None:
                kept_half = _half_space_solid(p, n, removed=False)
            st["common_retries"] += 1
            try:
                alt = _common_one(part, kept_half)
                alt_excess = _excess(alt, p, n, vtol)
                if alt_excess < excess:
                    res, excess = alt, alt_excess
            except Exception as exc:
                log.debug("half-space common retry failed: %s", exc)
            if excess > vtol:
                st["excess_bodies"] += 1
                log.info("section body: a cut body still reaches %.4g mm past the plane",
                         excess)
        st["half_space_excess_mm"] = max(st["half_space_excess_mm"], float(excess))
        if res is not None and not res.IsNull():
            kept.append(res)
        st["cut"] += 1
    b = BRep_Builder()
    comp = TopoDS_Compound()
    b.MakeCompound(comp)
    for s in kept:
        b.Add(comp, s)
    if stats is not None:
        stats.update(st)
    return comp


def _mesh_excess(shape, p, n, deflection: float, tol: Optional[float] = None
                 ) -> Optional[float]:
    """Reach past the plane of a triangulated copy of ``shape`` meshed with
    ``deflection`` (the copy keeps the triangulation off the shape that is
    projected later). First the
    ``BRepBndLib.AddOptimal_s(useTriangulation=True, useShapeTolerance=False)``
    box (C++; nodes enlarged by the stored triangulation deflection: about
    2 x deflection above the true reach on a cut sphere, 0.0094 at 0.005).
    When that is still above ``tol``, the nodes themselves are scanned and
    their largest distance + ``deflection`` is used (the nodes lie on the
    surfaces): on the RC buggy run a tolerance-enlarged box kept one cleanly
    cut body 0.095 mm 'past' the plane and the deflection-enlarged one
    0.049 mm (vtol 0.035), while its nodes at deflection 0.01 were all
    behind the plane. None when meshing fails or a face gets no
    triangulation."""
    from OCP.BRep import BRep_Tool
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Copy
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS
    try:
        cp = BRepBuilderAPI_Copy(shape, False, False).Shape()
        BRepMesh_IncrementalMesh(cp, float(deflection), False, 0.5, True)
        tris = []
        ex = TopExp_Explorer(cp, TopAbs_FACE)
        while ex.More():
            loc = TopLoc_Location()
            tri = BRep_Tool.Triangulation_s(TopoDS.Face_s(ex.Current()), loc)
            if tri is None or tri.NbNodes() == 0:
                return None                    # the box would fall back to the surface
            tris.append((tri, loc))
            ex.Next()
        box = Bnd_Box()
        BRepBndLib.AddOptimal_s(cp, box, True, False)   # nodes + deflection, no tolerance
        rng = _box_range(None if box.IsVoid() else
                         (np.array(box.Get()[:3], float), np.array(box.Get()[3:], float)), p, n)
        if rng is None:
            return None
        reach = float(rng[1])
        if tol is None or reach <= tol:
            return reach
        px, py, pz = (float(v) for v in p)
        nx, ny, nz = (float(v) for v in n)
        best = -math.inf
        for tri, loc in tris:
            tr = loc.Transformation()
            for i in range(1, tri.NbNodes() + 1):
                q = tri.Node(i).Transformed(tr)
                d = (q.X() - px) * nx + (q.Y() - py) * ny + (q.Z() - pz) * nz
                if d > best:
                    best = d
        if math.isfinite(best):
            reach = min(reach, best + float(deflection))
        return reach
    except Exception as exc:
        log.debug("triangulated excess check failed: %s", exc)
        return None


def _excess(shape, p, n, tol, refine: bool = True) -> float:
    """How far ``shape`` reaches past the plane on the removed (+n) side; 0
    when it stays behind it (or is empty).

    The fast box passing is proof. ``AddOptimal_s`` without triangulation
    is NOT exact on trimmed faces: an analytic sphere r10 cut at y = -2
    keeps its full-sphere box (8 mm 'past' the plane while every vertex and
    triangulation node is behind it), and on the RC buggy assembly 6 of the
    7 bodies reported 0.07 to 1.96 mm past the plane with their
    triangulation (deflection 0.01) at 0.0. So an AddOptimal excess above
    ``tol`` is checked (``refine``) on a triangulation of a copy with
    deflection ``tol / 4`` (``_mesh_excess``: the triangulation box, then
    the largest node distance + the deflection; a clean cut comes out at
    or below tol / 2, a real overshoot such as the invalid RC body that
    reached 1.97 mm at its vertices stays) and the smaller value is
    returned."""
    if shape is None or shape.IsNull():
        return 0.0
    rng = _box_range(_fast_bbox(shape), p, n)        # conservative: passing it is proof
    if rng is not None and rng[1] <= tol:
        return 0.0
    rng = _box_range(_exact_bbox(shape), p, n)
    if rng is None:
        return 0.0
    reach = max(0.0, float(rng[1]))
    if refine and reach > tol and tol > 0.0:
        m = _mesh_excess(shape, p, n, 0.25 * tol, tol=tol)
        if m is not None:
            reach = min(reach, max(0.0, m))
    return reach


def _body_centroids(shape, parts=None) -> List[np.ndarray]:
    """Centroids of the bodies of ``shape`` (or of ``parts``), largest
    (volume, or area for shells / faces) first."""
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    from OCP.TopAbs import TopAbs_SOLID
    out = []
    for part in (parts if parts is not None else _parts(shape)):
        try:
            props = GProp_GProps()
            if part.ShapeType() == TopAbs_SOLID:
                BRepGProp.VolumeProperties_s(part, props)
            else:
                BRepGProp.SurfaceProperties_s(part, props)
            c = props.CentreOfMass()
            out.append((abs(float(props.Mass())), np.array([c.X(), c.Y(), c.Z()], float)))
        except Exception as exc:
            log.debug("body centroid failed: %s", exc)
    out.sort(key=lambda t: -t[0])
    return [c for _, c in out]


# ----------------------------------------------------------------- API

_FASTENER_AXIS_COS = 0.5   # |axis . n| below this: the plane cuts the fastener lengthwise


def fastener_parts(features: Optional[Sequence[CylFeature]], normal=None,
                   max_cos: float = _FASTENER_AXIS_COS) -> List[int]:
    """Solid indices (``_parts`` / ``brep_features._solid_list`` order) of the
    fasteners named by ``features``: the ``filler_bodies`` of every
    ``"filled_hole"`` feature (``brep_features.cylinder_features``).

    With a section plane ``normal``, only the fillers of holes whose axis
    lies in the plane (``|axis . n| < max_cos``: the bolt is cut lengthwise)
    are returned: ISO 128-50 leaves bolts, pins and shafts unhatched only in
    a lengthwise cut, a cross-cut bolt is sectioned. A feature without a
    usable axis counts as lengthwise."""
    nv = None
    if normal is not None:
        nv = np.asarray(normal, float).reshape(3)
        nn = float(np.linalg.norm(nv))
        nv = nv / nn if nn > 1e-12 and np.all(np.isfinite(nv)) else None
    out = set()
    for f in features or ():
        if getattr(f, "kind", "") != "filled_hole":
            continue
        if nv is not None:
            try:
                ax = np.asarray(getattr(f, "axis"), float).reshape(3)
                an = float(np.linalg.norm(ax))
                if an > 1e-12 and abs(float(ax @ nv)) / an >= float(max_cos):
                    continue                  # cut across its axis: sectioned
            except Exception:
                pass
        for b in getattr(f, "filler_bodies", ()) or ():
            try:
                if int(b) >= 0:
                    out.add(int(b))
            except (TypeError, ValueError):
                continue
    return sorted(out)


def section_brep(shape, point, normal, spec, *, exact: bool, deflection: float,
                 depth_extent: float, hidden: bool = True,
                 time_budget_s: Optional[float] = None,
                 features: Optional[Sequence[CylFeature]] = None,
                 keep_whole: Optional[Sequence[int]] = None) -> SectionView:
    """Cut ``shape`` with the plane (``point``, ``normal``) and view it with ``spec``.

    ``normal`` must equal ``spec.n`` (ValueError otherwise). A plane that
    lies in a planar face of the shape (it would return that face's outline,
    e.g. a boss footprint as a hole) or that gives zero closed loops is
    nudged ONCE by 1 % of ``depth_extent`` toward the shape's bbox centre.
    A plane that is still empty but lies inside the shape's bbox (a gap
    between bodies) is moved through the centroid of the largest body, then
    the next ones (up to ``_RETRY_BODIES``; ``stats["retried_at_body"]``).
    Each body is sectioned and cut on its own (see ``_section_loops`` and
    ``_half_space_cut``), so an assembly with touching or overlapping
    solids keeps nothing of the removed half.
    Returns a ``SectionView`` with ``loops`` (view 2D, outers CCW then their
    holes CW), ``body`` (``project_brep`` of the half behind the plane with
    ``exact`` / ``deflection`` / ``hidden``; None when there are no loops;
    ``hidden=False`` leaves out the hidden lines behind the cut, as ISO
    section views usually do) and ``stats``: nudged, nudge_mm, coplanar
    (faces in the original plane), retried_at_body, retry_mm,
    open_wires_dropped, loops_recovered, dangling_segments_pruned, edges,
    wires, bodies_sectioned, kept_whole, dropped_whole, cut, common_retries,
    half_space_excess_mm, half_space_excess_tol_mm, excess_bodies, seconds,
    seconds_body.

    ``time_budget_s`` (None = none) is handed to the body's ``project_brep``
    as its budget with ``hlr_raise_after_s = inf``: an overrun never raises,
    it only cuts the Python clean-up of the body (hidden lines dropped,
    visible segments left unmerged; the pipeline's run deadline).

    QA round 2: ``loop_body`` of the result gives the body (``_parts``
    index: solids first, in explorer order) of every loop; the loops are
    nested body by body (issue 3). In an assembly (more than one body) the
    fasteners named by ``features`` (``fastener_parts(features, normal)``:
    the ``filler_bodies`` of the ``"filled_hole"`` features whose axis lies
    in the plane; a bolt cut across its axis is sectioned) and the indices in
    ``keep_whole`` are not sectioned: no loops, kept whole in the body when
    the plane runs through them (issue 28); ``stats["fasteners_unsectioned"]``
    lists those, ``stats["bodies_skipped"]`` counts them on the final plane.
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

    skip = set(fastener_parts(features, normal=n)) | {int(i) for i in (keep_whole or ())}
    step = _NUDGE_FRACTION * abs(float(depth_extent))
    nudged = False
    nudge_mm = 0.0
    coplanar = _coplanar_face_count(shape, p, n) if _NUDGE_COPLANAR else 0
    if coplanar:
        nudge_mm = _nudge_sign(shape, p, n) * step
        p = p + nudge_mm * n
        nudged = True
        log.info("section plane lies in %d planar face(s); nudged by %.4g mm", coplanar, nudge_mm)
    ids: List[int] = []
    try:
        bodies = _Bodies(shape) if _SECTION_PER_BODY else None
        if bodies is None or len(bodies.parts) < 2:
            skip = set()                      # one body: nothing to leave unsectioned
        loops, lstats = _section_loops(shape, p, n, spec, deflection, bodies=bodies,
                                       skip=skip, ids_out=ids)
    except Exception as exc:
        raise RuntimeError(f"section failed: {exc}") from exc
    is_void = (lambda b: _part_is_void(bodies.parts[b])) if bodies is not None else None
    ctol = 2.0 * deflection + _WIRE_TOL               # two discretisations of one curve
    loops, ids = _normalise_loops_by_body(loops, ids, is_void, ctol)
    if not loops and not nudged:
        nudge_mm = _nudge_sign(shape, p, n) * step
        p = p + nudge_mm * n
        nudged = True
        try:
            loops, lstats = _section_loops(shape, p, n, spec, deflection, bodies=bodies,
                                           skip=skip, ids_out=ids)
        except Exception as exc:
            raise RuntimeError(f"section failed after nudge: {exc}") from exc
        loops, ids = _normalise_loops_by_body(loops, ids, is_void, ctol)
        log.info("section plane had no closed loop; nudged by %.4g mm -> %d loop(s)",
                 nudge_mm, len(loops))

    retried = False
    retry_mm = 0.0
    whole = _box_range(_fast_bbox(shape), p, n) if (not loops and _RETRY_AT_BODY) else None
    if whole is not None and whole[0] < 0.0 < whole[1]:
        # the plane runs through a gap inside the bbox (between bodies): move it
        # through the centroid of the largest bodies, one after the other. A
        # plane outside the bbox is the caller's choice and stays empty.
        for c in _body_centroids(shape, bodies.parts if bodies else None)[:_RETRY_BODIES]:
            d = float(np.dot(c - p, n))
            if abs(d) <= max(_SIDE_TOL_MIN, 1e-9 * step):
                continue
            q = p + d * n
            if _NUDGE_COPLANAR and _coplanar_face_count(shape, q, n):
                q = q + _nudge_sign(shape, q, n) * step * n
            cids: List[int] = []
            try:
                cand, cstats = _section_loops(shape, q, n, spec, deflection, bodies=bodies,
                                              skip=skip, ids_out=cids)
            except Exception as exc:
                log.debug("section retry at a body failed: %s", exc)
                continue
            cand, cids = _normalise_loops_by_body(cand, cids, is_void, ctol)
            if cand:
                retry_mm = float(np.dot(q - p, n))
                p, loops, lstats, retried, ids = q, cand, cstats, True, cids
                log.info("section plane moved %.4g mm to a body centroid -> %d loop(s)",
                         retry_mm, len(loops))
                break

    body = None
    t_body = 0.0
    hs: dict = {}
    if loops:
        t1 = time.perf_counter()
        try:
            kept = _half_space_cut(shape, p, n, stats=hs, bodies=bodies, keep_whole=skip)
        except Exception as exc:
            raise RuntimeError(f"section body cut failed: {exc}") from exc
        if time_budget_s is None:
            body = project_brep(kept, spec, exact=exact, deflection=deflection,
                                hidden=bool(hidden))
        else:
            budget = max(float(time_budget_s) - (time.perf_counter() - t0), 1e-3)
            body = project_brep(kept, spec, exact=exact, deflection=deflection,
                                hidden=bool(hidden), time_budget_s=budget,
                                hlr_raise_after_s=math.inf)
        t_body = time.perf_counter() - t1

    stats = dict(lstats)
    stats.update(hs)
    stats.setdefault("fasteners_unsectioned", [])
    stats.update(nudged=nudged, nudge_mm=nudge_mm, coplanar=int(coplanar), loops=len(loops),
                 retried_at_body=retried, retry_mm=retry_mm,
                 seconds=round(time.perf_counter() - t0, 6), seconds_body=round(t_body, 6))
    marker = {"front": "top", "back": "top", "top": "front", "bottom": "front",
              "right": "front", "left": "front"}.get(spec.name, "top")
    return SectionView(spec=spec, plane_point=p, plane_normal=n.copy(), loops=loops, body=body,
                       marker_view=marker, stats=stats,
                       loop_body=list(ids) if len(ids) == len(loops) else None)


__all__ = ["choose_section_plane_brep", "choose_section_plane_brep_info", "section_brep",
           "fastener_parts"]
