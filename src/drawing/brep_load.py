"""STEP loading, shape census, rotation and faceted shells for the B-rep route.

Purpose: turn a STEP file (or an in-memory OCCT shape, or a triangle mesh) into
a ``BrepModel`` - the shape plus a JSON-ready census that the quality report
and the drawing pipeline read. OCP is imported lazily inside the functions so
importing this module stays cheap; every OCP name is a flat
``from OCP.<Module> import <Name>`` that exists on OCP 7.9.3.

Measured facts this module relies on (REPORT-brep-analysis.md and
REPORT-hlr.md, OCP 7.9.3.1 on this machine):

* After ``STEPControl_Reader.ReadFile`` returns anything but
  ``IFSelect_RetDone`` (garbage, truncated or missing file), any call to
  ``NbRootsForTransfer`` / ``TransferRoots`` / ``OneShape`` is a native access
  violation (rc 0xC0000005) that Python cannot catch. ``load_step`` gates on
  ``ReadFile`` first and never touches the transfer after a failed read.
* An empty ``DATA; ENDSEC;`` file reads as RetDone with 0 roots and a null
  ``OneShape``; ``BRepCheck_Analyzer`` on a null shape raises
  ``Standard_NullObject``. OCP exception classes derive from ``Exception``
  directly (not from ``Standard_Failure``), so ``except Exception`` is used.
* The parse banner ``**** ERR StepFile`` goes to stdout unless
  ``src.io.occt_utils.quiet_occt()`` was called first.
* ``FileUnits`` reports ``['millimetre']`` / ``['INCH']``; OCCT converts inch
  files to mm on read (a 25.4 mm cube written in INCH reads back as 25.4).
  The OCP-free Part 21 text sniff keeps only records containing
  ``LENGTH_UNIT(`` and prefers ``CONVERSION_BASED_UNIT`` over ``SI_UNIT``
  (a first-match regex would return the DEGREE angle unit). OCCT writes the
  unit records near the end of DATA, so the sniff reads the head and the
  tail of a large file and streams the middle only when both miss.
* Unique topology counts need ``TopExp.MapShapes_s`` (an explorer counts the
  30 plate edges as 60). Plate census: Plane 7 / Cylinder 6, 13 faces,
  30 edges, 20 vertices, volume 90365.210146, area 19016.393617.
* ``BRepBndLib.AddOptimal_s(shape, box, False, False)`` is exact on analytic
  and B-spline shapes; ``Add_s`` over-reports a NURBS r=10 cylinder by 10 mm
  (the control hull) and grows by the deflection once a mesh exists.
* ``BRepCheck_Analyzer.IsValid`` is True for an open shell and for an
  inverted solid (volume -1000 on ``box.Reversed()``), so closure, free edges,
  solid count and the signed volume are reported as separate flags.
* ``src.io.step_audit.measure_shape`` covers the same validity numbers but
  meshes the shape (deflection 0.2) to get a tessellated bbox the drawing does
  not use; the census here runs the same calls (free bounds, ``IsClosed_s``,
  BRepCheck, ``step_audit.brep_volume``) without meshing.
* ``BRepLib.EncodeRegularity_s(shape, angle)`` codes the facet edges of a
  SHELL or SOLID built from ``BRepBuilderAPI_MakeShapeOnMesh`` faces as C1
  (icosphere(2): 480 edges C0 -> C1 at 0.35 rad, unchanged at 0.1 rad since
  its dihedral angles are 0.12..0.20 rad); on a bare compound of faces it
  silently does nothing.
* QA round 1: ``OneShape`` holds every root, so a ``GEOMETRIC_CURVE_SET``
  (sketch / PMI curves) arrives as loose edges and a
  ``SHELL_BASED_SURFACE_MODEL`` as a free shell next to the solid (a
  STEPControl_Writer round trip of box + line + face gives exactly 1 solid,
  1 free shell, 1 loose edge). They were drawn, bounded and counted as part
  geometry; ``select_drawn_shape`` keeps the solids (NIST CTC 01: 78 loose
  edges and 22 loose vertices dropped, CTC 02: 1 free shell).
* A tessellated-only AP242 file (``TESSELLATED_SOLID`` of
  ``COMPLEX_TRIANGULATED_FACE``) transfers into faces with a
  ``Poly_Triangulation`` and a null surface, 0 edges (NIST FTC 08: 273
  faces, 3377 triangles); the stored node normals disagree with the
  triangle winding on all of them, so the winding follows the normals.
* ``FileUnits`` spells 'metre', 'centimetre', 'kilometre', 'micrometre',
  'FOOT', 'MILE', 'MIL' for files written with those ``write.step.unit``
  values; OCCT converts all of them to mm on read.
"""
from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_SURF_NAMES = {
    "GeomAbs_Plane": "Plane", "GeomAbs_Cylinder": "Cylinder", "GeomAbs_Cone": "Cone",
    "GeomAbs_Sphere": "Sphere", "GeomAbs_Torus": "Torus",
    "GeomAbs_BezierSurface": "BezierSurface", "GeomAbs_BSplineSurface": "BSplineSurface",
    "GeomAbs_SurfaceOfRevolution": "SurfaceOfRevolution",
    "GeomAbs_SurfaceOfExtrusion": "SurfaceOfExtrusion",
    "GeomAbs_OffsetSurface": "OffsetSurface", "GeomAbs_OtherSurface": "OtherSurface",
}

# a rotation matrix must satisfy R R^T = I and det R = +1 to this tolerance
_ROTATION_TOL = 1e-6


@dataclass
class BrepModel:
    """A shape plus its census.

    ``census`` keys: solids shells faces edges vertices face_types
    brepcheck_valid free_edges closed volume_mm3 area_mm2 bbox_min bbox_max
    inverted solids_unique faces_unique tessellated_faces, plus
    ``volume_signed_open`` when the volume was withheld. ``solids`` / ``faces``
    ... count INSTANCES (an assembly part placed twice counts twice);
    ``solids_unique`` / ``faces_unique`` count distinct ``TShape`` s.
    ``volume_mm3`` is None for an open model (no solid, or a shell that is not
    closed): the signed number is kept under ``volume_signed_open``.
    ``bbox_min``/``bbox_max`` are lists of three floats (None for an
    empty shape). ``brepcheck_valid`` is a bool, or None when the census ran
    with ``full_check=False``. ``warnings`` are human-readable flags for the
    report.

    ``shape`` is the DRAWN shape: for a STEP file (``load_step``) the solids
    only when there is at least one, else the free shells and faces, and the
    loose edges only for a pure wireframe (see ``select_drawn_shape``). The
    census, the bbox, the features and the sections all use it.
    ``original_shape`` is the full transferred compound (None when the model
    was built by ``analyze_shape``). ``bodies`` is the split report of
    ``select_drawn_shape`` (empty for ``analyze_shape``): counts of solids,
    free_shells, free_faces, loose_edges, loose_vertices, tessellated_bodies,
    ``drawn`` ("solids" | "shells_faces" | "wireframe" | "empty") and
    ``dropped`` (dict of the counts left out of the drawing).
    ``source_unit_factor`` is the length factor source unit -> mm (1.0 for mm,
    25.4 for inch, 1000 for m; None when unknown).
    """

    shape: object
    source_unit: str
    roots: int
    census: dict
    warnings: List[str] = field(default_factory=list)
    source_unit_factor: Optional[float] = None
    bodies: dict = field(default_factory=dict)
    original_shape: object = None


class TessellatedOnlyError(ValueError):
    """The STEP file holds only tessellated faces (``TESSELLATED_SOLID`` /
    ``TRIANGULATED_FACE``): OCCT gives faces with a triangulation and no
    surface, so the B-rep route (HLR, census, bbox) cannot run.

    Subclass of ``ValueError`` so an unaware caller still fails with a clear
    message. ``mesh`` is a ``trimesh.Trimesh`` built from the triangulations
    (nodes in mm, moved by the face location, winding flipped on reversed
    faces, duplicate nodes merged) for the mesh route; ``faces`` and
    ``triangles`` count what it came from; ``source_unit`` /
    ``source_unit_factor`` / ``roots`` / ``bodies`` are as on ``BrepModel``.
    ``dropped`` (QA round 2 issue 24, ``tessellated_dropped``): what the mesh
    leaves out, e.g. ``{"free_shells": 1, "tessellated_shells_outside": 1,
    "annotation_occurrences": 60}`` on NIST FTC 08; only non-zero counts.
    """

    def __init__(self, message: str, mesh: Any = None, *, faces: int = 0, triangles: int = 0,
                 source_unit: str = "unknown", source_unit_factor: Optional[float] = None,
                 roots: int = 1, bodies: Optional[dict] = None,
                 dropped: Optional[dict] = None) -> None:
        super().__init__(message)
        self.mesh = mesh
        self.faces = int(faces)
        self.triangles = int(triangles)
        self.source_unit = source_unit
        self.source_unit_factor = source_unit_factor
        self.roots = int(roots)
        self.bodies = dict(bodies or {})
        self.dropped = dict(dropped or {})


# ---------------------------------------------------------------------------
# census
# ---------------------------------------------------------------------------

def _count_unique(shape: Any, kind: Any) -> int:
    """Number of sub-shape INSTANCES of ``kind`` (TShape + location; an
    explorer would count a shared edge twice, this does not)."""
    from OCP.TopExp import TopExp
    from OCP.TopTools import TopTools_IndexedMapOfShape
    m = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape, kind, m)
    return m.Extent()


def _count_tshapes(shape: Any, kind: Any) -> int:
    """Number of distinct ``TShape`` s of ``kind``: every instance is moved to
    the identity location before it goes into the map, so one part placed
    twice in an assembly counts once."""
    from OCP.TopExp import TopExp
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopTools import TopTools_IndexedMapOfShape, TopTools_MapOfShape
    m = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape, kind, m)
    seen = TopTools_MapOfShape()
    ident = TopLoc_Location()
    for i in range(1, m.Extent() + 1):
        seen.Add(m.FindKey(i).Located(ident))
    return seen.Extent()


def _empty_census() -> Dict[str, Any]:
    return {"solids": 0, "shells": 0, "faces": 0, "edges": 0, "vertices": 0,
            "face_types": {}, "brepcheck_valid": False, "free_edges": 0,
            "closed": False, "volume_mm3": 0.0, "area_mm2": 0.0,
            "bbox_min": None, "bbox_max": None, "inverted": False,
            "solids_unique": 0, "faces_unique": 0, "tessellated_faces": 0,
            "loose_faces": 0, "volume_scope": "shape"}


def _face_types(shape: Any) -> Dict[str, int]:
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp
    from OCP.TopoDS import TopoDS
    from OCP.TopTools import TopTools_IndexedMapOfShape

    m = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape, TopAbs_FACE, m)
    out: Dict[str, int] = {}
    for i in range(1, m.Extent() + 1):
        try:
            t = BRepAdaptor_Surface(TopoDS.Face_s(m.FindKey(i))).GetType()
            name = _SURF_NAMES.get(str(t).split(".")[-1], "Unknown")
        except Exception as exc:
            logger.debug("surface type failed: %s", exc)
            name = "Unknown"
        out[name] = out.get(name, 0) + 1
    return out


def exact_bbox(shape: Any) -> Optional[np.ndarray]:
    """``(2, 3)`` array [min, max] by ``AddOptimal_s`` (exact on B-splines), or None."""
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    try:
        box = Bnd_Box()
        BRepBndLib.AddOptimal_s(shape, box, False, False)
        if box.IsVoid():
            return None
        x0, y0, z0, x1, y1, z1 = box.Get()
        return np.array([[x0, y0, z0], [x1, y1, z1]], dtype=float)
    except Exception as exc:
        logger.warning("exact bounding box failed: %s", exc)
        return None


def loose_bbox(shape: Any) -> Optional[np.ndarray]:
    """``(2, 3)`` box by ``BRepBndLib.Add_s`` without triangulation: never smaller
    than the shape (control hull + tolerance), cheap, but not exact (0.43 mm
    too large on the RC buggy assembly). None when void or on failure."""
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    try:
        box = Bnd_Box()
        BRepBndLib.Add_s(shape, box, False)
        if box.IsVoid():
            return None
        x0, y0, z0, x1, y1, z1 = box.Get()
        return np.array([[x0, y0, z0], [x1, y1, z1]], dtype=float)
    except Exception as exc:
        logger.warning("loose bounding box failed: %s", exc)
        return None


def exact_bbox_bounded(shape: Any) -> Optional[np.ndarray]:
    """The ``exact_bbox`` box, computed by branch and bound over the faces.

    ``AddOptimal_s`` on a compound is the union of the optimal boxes of its
    faces (plus edges outside faces and vertices outside edges). The loose
    ``Add_s`` box of a face is never smaller than its optimal box, so for each
    of the six box sides the faces are visited in the order of their loose
    bound and the walk stops as soon as the next loose bound cannot beat the
    best optimal value found: only the few faces near the extremes are
    integrated. Measured on the 10665-face RC buggy assembly: 19 faces
    integrated, 0.31 s instead of 33 s, the box bit-identical to
    ``exact_bbox``. Shapes with edges outside faces or vertices outside edges
    (wireframes) use ``exact_bbox``. None when void; raises nothing (falls
    back to ``exact_bbox`` on any OCCT failure).
    """
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_VERTEX
    from OCP.TopExp import TopExp, TopExp_Explorer
    from OCP.TopTools import TopTools_IndexedMapOfShape
    try:
        if shape is None or shape.IsNull():
            return None
        if TopExp_Explorer(shape, TopAbs_EDGE, TopAbs_FACE).More() or \
                TopExp_Explorer(shape, TopAbs_VERTEX, TopAbs_EDGE).More():
            return exact_bbox(shape)
        m = TopTools_IndexedMapOfShape()
        TopExp.MapShapes_s(shape, TopAbs_FACE, m)
        faces = [m.FindKey(i) for i in range(1, m.Extent() + 1)]
        if not faces:
            return exact_bbox(shape)
        loose = np.empty((len(faces), 6))
        for i, f in enumerate(faces):
            b = Bnd_Box()
            BRepBndLib.Add_s(f, b, False)
            loose[i] = (b.Get() if not b.IsVoid()
                        else (np.inf, np.inf, np.inf, -np.inf, -np.inf, -np.inf))
        optimal: Dict[int, Optional[np.ndarray]] = {}

        def opt(i: int) -> Optional[np.ndarray]:
            if i not in optimal:
                b = Bnd_Box()
                BRepBndLib.AddOptimal_s(faces[i], b, False, False)
                optimal[i] = None if b.IsVoid() else np.asarray(b.Get(), float)
            return optimal[i]

        res = np.empty(6)
        for k in range(6):
            sign = -1.0 if k < 3 else 1.0          # the minima as maxima of -x
            key = sign * loose[:, k]
            best = -np.inf
            for i in np.argsort(-key, kind="stable"):
                if not key[i] > best:
                    break
                o = opt(int(i))
                if o is not None:
                    best = max(best, sign * float(o[k]))
            if not math.isfinite(best):
                return exact_bbox(shape)
            res[k] = sign * best
        logger.debug("bounded exact bbox: %d of %d faces integrated", len(optimal), len(faces))
        return res.reshape(2, 3)
    except Exception as exc:
        logger.warning("bounded exact bbox failed (%s); full AddOptimal", exc)
        return exact_bbox(shape)


def _unique_solids(shape: Any) -> Optional[List[tuple]]:
    """``[(solid at the identity location, FORWARD), [factor per instance]], ...]``
    grouped by ``TShape``, or None when the shape holds faces outside solids.

    ``factor`` turns the volume of the unique solid into the signed volume of
    the instance: ``ScaleFactor ** 3`` of its location (negative for a mirror)
    times -1 for a REVERSED instance. Verified on the RC buggy assembly: the
    sum over the 211 instances of the per-instance adaptive volume equals
    the sum over the 91 unique solids times their factors to 1e-12.
    """
    from OCP.TopAbs import TopAbs_FACE, TopAbs_FORWARD, TopAbs_REVERSED, TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopTools import TopTools_IndexedMapOfShape
    if TopExp_Explorer(shape, TopAbs_FACE, TopAbs_SOLID).More():
        return None
    m = TopTools_IndexedMapOfShape()
    factors: Dict[int, List[float]] = {}
    ident = TopLoc_Location()
    ex = TopExp_Explorer(shape, TopAbs_SOLID)
    while ex.More():
        s = ex.Current()
        ex.Next()
        idx = m.Add(s.Located(ident).Oriented(TopAbs_FORWARD))
        f = float(s.Location().Transformation().ScaleFactor()) ** 3
        if s.Orientation() == TopAbs_REVERSED:
            f = -f
        factors.setdefault(idx, []).append(f)
    if m.Extent() == 0:
        return None
    return [(m.FindKey(i), factors.get(i, [])) for i in range(1, m.Extent() + 1)]


def _census(shape: Any, full_check: bool = True,
            fast_above_faces: Optional[int] = None) -> Dict[str, Any]:
    """The census dict (see ``BrepModel``).

    ``fast_above_faces`` (None = always the exact path): above that many
    face instances, ``BRepCheck_Analyzer`` and the volume run once per
    UNIQUE solid (``_unique_solids``: instances grouped by ``TShape``, the
    volume multiplied by each instance's factor) and the bbox comes from
    ``exact_bbox_bounded`` (same box as ``AddOptimal_s``, a few faces
    integrated). Extra keys: ``census_mode`` ("exact" | "fast"),
    ``bbox_source`` ("exact" | "exact_bounded"), and on the fast path with
    solids ``solid_instances`` and ``volume_method`` "per_unique_solid".
    Measured on the RC buggy assembly (211 instances of 91 solids, 10665
    faces): BRepCheck 4.4 -> 3.4 s, volume 7.9 -> 4.1 s, bbox 21.7 -> 0.3 s.
    The per-unique volume (291371.87 mm3) agrees with the fixed Gauss volume
    of the compound (291326.62) to 0.016 %; the adaptive integration of the
    whole compound gives 292280.79 (+0.31 %), the sum of its own per-instance
    integrations is again 291371.87.
    """
    c = _empty_census()
    if shape is None or shape.IsNull():
        return c
    from OCP.BRep import BRep_Tool
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    from OCP.ShapeAnalysis import ShapeAnalysis_FreeBounds
    from OCP.TopAbs import (TopAbs_EDGE, TopAbs_FACE, TopAbs_SHELL, TopAbs_SOLID,
                            TopAbs_VERTEX)
    from OCP.TopExp import TopExp
    from OCP.TopoDS import TopoDS
    from OCP.TopTools import TopTools_IndexedMapOfShape
    from src.io.step_audit import brep_volume

    c["solids"] = _count_unique(shape, TopAbs_SOLID)
    c["faces"] = _count_unique(shape, TopAbs_FACE)
    c["edges"] = _count_unique(shape, TopAbs_EDGE)
    c["vertices"] = _count_unique(shape, TopAbs_VERTEX)
    try:
        c["solids_unique"] = _count_tshapes(shape, TopAbs_SOLID)
        c["faces_unique"] = _count_tshapes(shape, TopAbs_FACE)
    except Exception as exc:
        logger.debug("unique TShape count failed: %s", exc)
        c["solids_unique"], c["faces_unique"] = c["solids"], c["faces"]
    c["tessellated_faces"] = count_tessellated_faces(shape)
    c["face_types"] = _face_types(shape)

    shells = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape, TopAbs_SHELL, shells)
    c["shells"] = shells.Extent()
    closed = []
    for i in range(1, shells.Extent() + 1):
        try:
            closed.append(bool(BRep_Tool.IsClosed_s(TopoDS.Shell_s(shells.FindKey(i)))))
        except Exception as exc:
            logger.debug("IsClosed failed: %s", exc)
            closed.append(False)
    shells_closed = bool(closed) and all(closed)
    # faces outside any shell (a stray surface next to a solid): they make the
    # model open and must stay out of the volume integral (a box 10^3 with a
    # loose face at z = 200 integrated to 6571.4 mm3)
    try:
        from OCP.TopExp import TopExp_Explorer
        loose = TopTools_IndexedMapOfShape()
        ex = TopExp_Explorer(shape, TopAbs_FACE, TopAbs_SHELL)
        while ex.More():
            loose.Add(ex.Current())
            ex.Next()
        c["loose_faces"] = loose.Extent()
    except Exception as exc:
        logger.debug("loose face count failed: %s", exc)
    c["closed"] = shells_closed and c["loose_faces"] == 0
    vol_target = shape
    if c["loose_faces"] and c["solids"] > 0 and shells_closed:
        try:
            from OCP.BRep import BRep_Builder
            from OCP.TopExp import TopExp_Explorer
            from OCP.TopoDS import TopoDS_Compound
            bld = BRep_Builder()
            only = TopoDS_Compound()
            bld.MakeCompound(only)
            ex = TopExp_Explorer(shape, TopAbs_SOLID)
            while ex.More():
                bld.Add(only, ex.Current())
                ex.Next()
            vol_target = only
            c["volume_scope"] = "solids"
        except Exception as exc:
            logger.debug("solid-only compound failed: %s", exc)

    try:
        fb = ShapeAnalysis_FreeBounds(shape)
        c["free_edges"] = (_count_unique(fb.GetClosedWires(), TopAbs_EDGE)
                           + _count_unique(fb.GetOpenWires(), TopAbs_EDGE))
    except Exception as exc:
        logger.warning("free-boundary analysis failed: %s", exc)

    fast = fast_above_faces is not None and c["faces"] > int(fast_above_faces)
    c["census_mode"] = "fast" if fast else "exact"
    unique = None
    if fast and c["solids"] > 0 and vol_target is shape:
        try:
            unique = _unique_solids(shape)
        except Exception as exc:
            logger.warning("unique solid grouping failed (%s); exact census", exc)
            unique = None
    if unique is not None:
        c["solid_instances"] = int(sum(len(f) for _s, f in unique))
        c["volume_method"] = "per_unique_solid"

    def volume_of(target) -> float:
        if full_check:
            return brep_volume(target)       # adaptive Gauss, Precision.Confusion
        props = GProp_GProps()               # fixed Gauss: exact on planar facets
        BRepGProp.VolumeProperties_s(target, props)
        return float(props.Mass())

    if full_check:
        try:
            if unique is not None:
                c["brepcheck_valid"] = all(bool(BRepCheck_Analyzer(s).IsValid())
                                           for s, _f in unique)
            else:
                c["brepcheck_valid"] = bool(BRepCheck_Analyzer(shape).IsValid())
        except Exception as exc:
            logger.warning("BRepCheck failed: %s", exc)
            c["brepcheck_valid"] = False
    else:
        c["brepcheck_valid"] = None          # not checked (fast census)

    if c["faces"]:
        try:
            if unique is not None:
                vol = sum(volume_of(s) * sum(f) for s, f in unique if f)
            else:
                vol = volume_of(vol_target)
            c["volume_mm3"] = float(vol) if math.isfinite(vol) else None
        except Exception as exc:
            logger.warning("volume integration failed: %s", exc)
            c["volume_mm3"] = None
        try:
            props = GProp_GProps()
            BRepGProp.SurfaceProperties_s(shape, props)
            c["area_mm2"] = float(props.Mass())
        except Exception as exc:
            logger.warning("area integration failed: %s", exc)
            c["area_mm2"] = None
    vol = c["volume_mm3"]
    vol_ok = shells_closed and (c["loose_faces"] == 0 or c["volume_scope"] == "solids")
    c["inverted"] = bool(c["solids"] > 0 and vol_ok and vol is not None and vol < 0)
    if c["faces"] and (c["solids"] == 0 or not vol_ok):
        # an open model encloses no volume: the signed integral is meaningless
        # (-6.57 on a spline cage) and is kept for diagnostics only
        c["volume_signed_open"] = vol
        c["volume_mm3"] = None

    bb = exact_bbox_bounded(shape) if fast else exact_bbox(shape)
    c["bbox_source"] = "exact_bounded" if fast else "exact"
    if bb is not None:
        c["bbox_min"] = [float(v) for v in bb[0]]
        c["bbox_max"] = [float(v) for v in bb[1]]
    return c


def _census_warnings(c: Dict[str, Any], source_unit: str, roots: int) -> List[str]:
    w: List[str] = []
    if c["faces"] == 0:
        w.append("shape has no faces")
        return w
    if c["solids"] == 0:
        w.append("no solid body (surface model or open shell): volume and "
                 "sections are not meaningful")
    elif c["solids"] > 1:
        w.append(f"{c['solids']} solids in one shape (multiple bodies); drawn as one part")
    if roots > 1:
        w.append(f"{roots} STEP roots transferred into one compound")
    if c["free_edges"] > 0:
        w.append(f"{c['free_edges']} free edge(s): the shape is not closed")
    if c["inverted"]:
        w.append(f"inverted solid (signed volume {c['volume_mm3']:.6g} mm3); "
                 f"ShapeFix_Solid would heal it")
    if c["brepcheck_valid"] is False:        # None = not checked (fast census)
        w.append("BRepCheck_Analyzer reports the shape invalid")
    if c.get("loose_faces"):
        w.append(f"{c['loose_faces']} face(s) outside any shell (stray surfaces)"
                 + ("; the volume counts the solids only" if c.get("volume_scope") == "solids"
                    else ""))
    if "volume_signed_open" in c:
        w.append("volume not reported: the model is open (no closed solid)")
    elif c["volume_mm3"] is None:
        w.append("volume integration failed")
    if c.get("tessellated_faces"):
        w.append(f"{c['tessellated_faces']} tessellated face(s) without a surface "
                 f"(triangulation only)")
    if c["bbox_min"] is None:
        w.append("bounding box is void")
    if source_unit != "mm":
        w.append(f"source length unit is {source_unit}; OCCT converted the "
                 f"geometry to mm on read")
    return w


def analyze_shape(shape, source_unit: str = "mm", roots: int = 1, *,
                  full_check: bool = True,
                  fast_above_faces: Optional[int] = None) -> BrepModel:
    """Census + warnings of an in-memory shape. Never raises on a null shape
    (the census is all zeros, ``bbox_min`` None, ``brepcheck_valid`` False).

    Cost (OCP 7.9.3, this machine, faceted icosphere bodies): the full census
    takes 1.7 s at 5120 faces and 8.2 s at 20480 faces; ``BRepCheck_Analyzer``
    and the adaptive volume (``step_audit.brep_volume``) are almost all of it,
    free bounds and the bbox are < 0.1 s. ``full_check=False`` skips
    BRepCheck (``brepcheck_valid`` is then None, meaning "not checked", and no
    invalid-shape warning is raised) and uses OCCT's fixed Gauss volume
    instead of the adaptive one. The fixed scheme is exact on planar facets
    (icosphere(5): equal to the mesh volume to 1e-12 relative, 0.6 s instead
    of 3.1 s) but can be off by up to ~2 % on curved B-rep faces, so use it for
    faceted bodies from ``faceted_shape_from_mesh`` (the STL route, whose mesh
    census already has the validity numbers) and keep the default for STEP.
    ``fast_above_faces``: see ``_census`` (per-unique-solid BRepCheck and
    volume, bounded exact bbox above that many face instances; None = never).
    """
    census = _census(shape, full_check=bool(full_check), fast_above_faces=fast_above_faces)
    return BrepModel(shape=shape, source_unit=source_unit, roots=int(roots),
                     census=census, warnings=_census_warnings(census, source_unit, roots),
                     source_unit_factor=UNIT_FACTORS.get(source_unit))


# ---------------------------------------------------------------------------
# drawn bodies: solids / free shells / free faces / loose edges
# ---------------------------------------------------------------------------

def _is_tessellated_face(face: Any) -> bool:
    """True for a face with a triangulation and no surface (a STEP
    ``TRIANGULATED_FACE`` / ``COMPLEX_TRIANGULATED_FACE`` after transfer)."""
    from OCP.BRep import BRep_Tool
    from OCP.TopLoc import TopLoc_Location
    try:
        if BRep_Tool.Surface_s(face, TopLoc_Location()) is not None:
            return False
        return BRep_Tool.Triangulation_s(face, TopLoc_Location()) is not None
    except Exception as exc:
        logger.debug("tessellation check failed: %s", exc)
        return False


def _faces_of(shape: Any) -> List[Any]:
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp
    from OCP.TopoDS import TopoDS
    from OCP.TopTools import TopTools_IndexedMapOfShape
    m = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape, TopAbs_FACE, m)
    return [TopoDS.Face_s(m.FindKey(i)) for i in range(1, m.Extent() + 1)]


def count_tessellated_faces(shape: Any) -> int:
    """Faces of ``shape`` that carry a triangulation but no surface."""
    if shape is None:
        return 0
    try:
        if shape.IsNull():
            return 0
        return sum(1 for f in _faces_of(shape) if _is_tessellated_face(f))
    except Exception as exc:
        logger.debug("tessellated face count failed: %s", exc)
        return 0


def _explore(shape: Any, kind: Any, avoid: Any = None) -> List[Any]:
    """Sub-shapes of ``kind`` not reached through ``avoid``, each instance once."""
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopTools import TopTools_MapOfShape
    seen = TopTools_MapOfShape()
    out: List[Any] = []
    ex = TopExp_Explorer(shape, kind) if avoid is None else TopExp_Explorer(shape, kind, avoid)
    while ex.More():
        s = ex.Current()
        if seen.Add(s):
            out.append(s)
        ex.Next()
    return out


def _compound_of(shapes: List[Any]) -> Any:
    from OCP.BRep import BRep_Builder
    from OCP.TopoDS import TopoDS_Compound
    b = BRep_Builder()
    comp = TopoDS_Compound()
    b.MakeCompound(comp)
    for s in shapes:
        b.Add(comp, s)
    return comp


def _all_tessellated(shape: Any) -> bool:
    faces = _faces_of(shape)
    return bool(faces) and all(_is_tessellated_face(f) for f in faces)


def split_bodies(shape: Any) -> Dict[str, List[Any]]:
    """Split a transferred shape into ``solids``, free ``shells`` (not in a
    solid), free ``faces`` (not in a shell), loose ``edges`` (not in a face,
    e.g. a STEP ``GEOMETRIC_CURVE_SET`` or PMI sketch) and loose ``vertices``
    (not in an edge). Every instance appears once (TShape + location)."""
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_SHELL, TopAbs_SOLID, TopAbs_VERTEX
    out: Dict[str, List[Any]] = {"solids": [], "shells": [], "faces": [], "edges": [],
                                 "vertices": []}
    if shape is None or shape.IsNull():
        return out
    out["solids"] = _explore(shape, TopAbs_SOLID)
    out["shells"] = _explore(shape, TopAbs_SHELL, TopAbs_SOLID)
    out["faces"] = _explore(shape, TopAbs_FACE, TopAbs_SHELL)
    out["edges"] = _explore(shape, TopAbs_EDGE, TopAbs_FACE)
    out["vertices"] = _explore(shape, TopAbs_VERTEX, TopAbs_EDGE)
    return out


def select_drawn_shape(shape: Any) -> "tuple[Any, Dict[str, Any]]":
    """``(drawn_shape, bodies)``: the part of a transferred STEP shape that is
    drawn, bounded, dimensioned and counted.

    Rule: the solids when there is at least one; else the free shells and
    free faces (a surface model); the loose edges and vertices only when
    there is no face at all (a pure wireframe). When exact bodies exist,
    tessellated-only bodies (every face a triangulation without surface,
    e.g. an AP242 file that carries both representations) are left out too.
    The input is returned unchanged when nothing is left out.

    ``bodies``: solids, free_shells, free_faces, loose_edges,
    loose_vertices, tessellated_bodies (counts in the input), ``drawn``
    ("solids" | "shells_faces" | "wireframe" | "empty") and ``dropped``
    (counts left out, only the non-zero ones).
    """
    parts = split_bodies(shape)
    info: Dict[str, Any] = {"solids": len(parts["solids"]), "free_shells": len(parts["shells"]),
                            "free_faces": len(parts["faces"]),
                            "loose_edges": len(parts["edges"]),
                            "loose_vertices": len(parts["vertices"]),
                            "tessellated_bodies": 0, "drawn": "empty", "dropped": {}}
    if shape is None or shape.IsNull():
        return shape, info

    def exact_only(items: List[Any]) -> "tuple[List[Any], int]":
        try:
            flags = [_all_tessellated(s) for s in items]
        except Exception as exc:
            logger.debug("tessellated body check failed: %s", exc)
            return items, 0
        if all(flags):
            return items, 0              # nothing exact: keep (load_step routes it)
        return [s for s, t in zip(items, flags) if not t], int(sum(flags))

    dropped: Dict[str, int] = {}
    if parts["solids"]:
        kept, n_tess = exact_only(parts["solids"])
        info["drawn"] = "solids"
        for key, name in (("shells", "free_shells"), ("faces", "free_faces"),
                          ("edges", "loose_edges"), ("vertices", "loose_vertices")):
            if parts[key]:
                dropped[name] = len(parts[key])
    elif parts["shells"] or parts["faces"]:
        kept, n_tess = exact_only(parts["shells"] + parts["faces"])
        info["drawn"] = "shells_faces"
        for key, name in (("edges", "loose_edges"), ("vertices", "loose_vertices")):
            if parts[key]:
                dropped[name] = len(parts[key])
    else:
        info["drawn"] = "wireframe" if (parts["edges"] or parts["vertices"]) else "empty"
        info["dropped"] = {}
        return shape, info
    if n_tess:
        dropped["tessellated_bodies"] = n_tess
    info["tessellated_bodies"] = n_tess
    info["dropped"] = dropped
    if not dropped:
        return shape, info
    return _compound_of(kept), info


def _dropped_warning(info: Dict[str, Any]) -> Optional[str]:
    d = info.get("dropped") or {}
    if not d:
        return None
    words = {"free_shells": "free shell(s)", "free_faces": "free face(s)",
             "loose_edges": "loose edge(s)", "loose_vertices": "loose vertex(es)",
             "tessellated_bodies": "tessellated body(ies)"}
    what = ", ".join(f"{n} {words.get(k, k)}" for k, n in d.items())
    kept = ("the solid(s)" if info.get("drawn") == "solids" else "the shells and faces")
    return f"supplemental geometry not drawn: {what}; the drawing uses {kept} only"


def triangulation_mesh(shape: Any) -> "tuple[Any, int, int]":
    """``(trimesh.Trimesh, faces_used, triangles)`` from the stored face
    triangulations of ``shape``: nodes moved by the face location, the
    winding flipped on ``TopAbs_REVERSED`` faces, duplicate nodes merged.
    Faces without a triangulation are skipped. Raises ``ValueError`` when no
    triangle is found."""
    import trimesh
    from OCP.BRep import BRep_Tool
    from OCP.TopAbs import TopAbs_REVERSED
    from OCP.TopLoc import TopLoc_Location

    verts: List[np.ndarray] = []
    tris: List[np.ndarray] = []
    offset = 0
    used = 0
    flipped_by_normals = 0
    for face in _faces_of(shape):
        try:
            loc = TopLoc_Location()
            T = BRep_Tool.Triangulation_s(face, loc)
            if T is None or T.NbTriangles() == 0:
                continue
            trsf = None if loc.IsIdentity() else loc.Transformation()
            nn = T.NbNodes()
            pts = np.empty((nn, 3), dtype=np.float64)
            for i in range(1, nn + 1):
                p = T.Node(i)
                if trsf is not None:
                    p = p.Transformed(trsf)
                pts[i - 1] = (p.X(), p.Y(), p.Z())
            nt = T.NbTriangles()
            idx = np.empty((nt, 3), dtype=np.int64)
            for i in range(1, nt + 1):
                idx[i - 1] = T.Triangle(i).Get()
            idx -= 1
            if _winding_against_normals(T, idx):
                idx = idx[:, ::-1]
                flipped_by_normals += 1
            if face.Orientation() == TopAbs_REVERSED:
                idx = idx[:, ::-1]
        except Exception as exc:
            logger.debug("triangulation of a face skipped: %s", exc)
            continue
        verts.append(pts)
        tris.append(idx + offset)
        offset += nn
        used += 1
    if not tris:
        raise ValueError("no face carries a triangulation")
    if flipped_by_normals:
        logger.info("tessellated faces: %d of %d wound against their stored normals; flipped",
                    flipped_by_normals, used)
    mesh = trimesh.Trimesh(vertices=np.vstack(verts), faces=np.vstack(tris), process=False)
    mesh.merge_vertices()
    return mesh, used, int(len(mesh.faces))


def _winding_against_normals(T: Any, idx: np.ndarray) -> bool:
    """True when the triangles of ``T`` (0-based ``idx``) are mostly wound
    against the node normals stored with the triangulation (a STEP
    ``COMPLEX_TRIANGULATED_FACE`` carries them). Measured on the NIST FTC 08
    tessellated file: all 3370 non-degenerate triangles disagree with their
    stored normals, and the unflipped mesh has a negative volume
    (-503281 mm3). False when no normals are stored."""
    try:
        if not T.HasNormals():
            return False
        nn = T.NbNodes()
        P = np.empty((nn, 3))
        N = np.empty((nn, 3))
        for i in range(1, nn + 1):
            p = T.Node(i)
            P[i - 1] = (p.X(), p.Y(), p.Z())
            d = T.Normal(i)
            N[i - 1] = (d.X(), d.Y(), d.Z())
    except Exception as exc:
        logger.debug("stored triangulation normals unreadable: %s", exc)
        return False
    g = np.cross(P[idx[:, 1]] - P[idx[:, 0]], P[idx[:, 2]] - P[idx[:, 0]])
    s = np.einsum("ij,ij->i", g, N[idx].sum(axis=1))
    return int(np.count_nonzero(s < 0)) > int(np.count_nonzero(s > 0))


# ---------------------------------------------------------------------------
# STEP reading
# ---------------------------------------------------------------------------

_SI_PREFIX = {"MILLI": "MM", "CENTI": "CM", "DECI": "DM", "": "M", "KILO": "KM",
              "MICRO": "UM", "NANO": "NM"}


# Head and tail window of the unit sniff. OCCT writes the unit records near the
# END of DATA, other exporters near the start; the middle of a large file is
# only streamed when neither window holds a length unit.
_SNIFF_WINDOW = 8 * 1024 * 1024
_SNIFF_CHUNK = 8 * 1024 * 1024
_SNIFF_RECORD_MAX = 4096            # longest unit record kept around a hit
_LENGTH_UNIT_RE = re.compile(rb"LENGTH_UNIT\s*\(", re.IGNORECASE)
_CONV_RE = re.compile(r"CONVERSION_BASED_UNIT\s*\(\s*'([^']*)'")
_SI_RE = re.compile(r"SI_UNIT\s*\(\s*(?:\.([A-Z]+)\.|\$)\s*,\s*\.METRE\.\s*\)")


def _unit_records(buf: bytes) -> List[str]:
    """The ';'-delimited records around every ``LENGTH_UNIT(`` hit in ``buf``,
    whitespace-collapsed and upper-case. ``LENGTH_MEASURE_WITH_UNIT`` lacks the
    token, so it never matches."""
    out: List[str] = []
    last_end = -1
    for m in _LENGTH_UNIT_RE.finditer(buf):
        if m.start() < last_end:
            continue                         # second hit inside the same record
        lo = buf.rfind(b";", max(0, m.start() - _SNIFF_RECORD_MAX), m.start()) + 1
        hi = buf.find(b";", m.end(), m.end() + _SNIFF_RECORD_MAX)
        if hi < 0:
            hi = min(len(buf), m.end() + _SNIFF_RECORD_MAX)
        last_end = hi
        rec = buf[lo:hi].decode("latin-1", errors="replace")
        out.append(" ".join(rec.split()).upper())
    return out


def _classify_records(records: List[str]) -> Optional[str]:
    conv: List[str] = []
    si: List[str] = []
    for r in records:
        m = _CONV_RE.search(r)
        if m:
            conv.append(m.group(1).strip())
            continue
        m = _SI_RE.search(r)
        if m:
            si.append(_SI_PREFIX.get(m.group(1) or "", (m.group(1) or "") + "METRE"))
    if conv:
        return conv[0]
    return si[0] if si else None


def _sniff_raw(path: str) -> Optional[str]:
    """First length unit of a Part 21 file as an upper-case token
    ('MM', 'INCH', 'M', ...) or None. OCP-free.

    Reads the head and the tail window (``_SNIFF_WINDOW`` each) and searches
    them for ``LENGTH_UNIT(`` with a regex, keeping only the record around each
    hit (no upper-case copy of the file, no split of every record). Only when
    neither window has a length unit is the middle streamed in chunks.
    ``CONVERSION_BASED_UNIT`` wins over ``SI_UNIT`` (a first-match rule would
    return the DEGREE angle unit).
    """
    import os
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as fh:
            if size <= 2 * _SNIFF_WINDOW:
                return _classify_records(_unit_records(fh.read()))
            head = fh.read(_SNIFF_WINDOW)
            fh.seek(size - _SNIFF_WINDOW)
            tail = fh.read(_SNIFF_WINDOW)
            found = _classify_records(_unit_records(head) + _unit_records(tail))
            if found is not None:
                return found
            # stream the middle with an overlap so no record is cut in two
            overlap = 2 * _SNIFF_RECORD_MAX
            pos = max(0, _SNIFF_WINDOW - overlap)
            stop = size - _SNIFF_WINDOW + overlap
            records: List[str] = []
            while pos < stop:
                fh.seek(pos)
                buf = fh.read(min(_SNIFF_CHUNK, stop - pos))
                if not buf:
                    break
                records.extend(_unit_records(buf))
                if len(buf) <= overlap:
                    break
                pos += len(buf) - overlap
            return _classify_records(records)
    except OSError:
        return None


# Normalised length-unit names and their factor to mm. The spellings are the
# ones OCCT's FileUnits returns ('millimetre', 'centimetre', 'metre',
# 'kilometre', 'micrometre', 'INCH', 'FOOT', 'MILE', 'MIL' - measured with
# files written by STEPControl_Writer in each write.step.unit) and the tokens
# of the text sniff ('MM', 'CM', 'M', 'KM', 'UM', 'DM', 'NM' from SI prefixes,
# CONVERSION_BASED_UNIT names such as 'INCH' / 'FOOT').
UNIT_FACTORS: Dict[str, float] = {
    "mm": 1.0, "cm": 10.0, "dm": 100.0, "m": 1000.0, "km": 1.0e6, "um": 1.0e-3,
    "nm": 1.0e-6, "inch": 25.4, "ft": 304.8, "yd": 914.4, "mile": 1609344.0,
    "mil": 0.0254, "microinch": 2.54e-5,
}
_UNIT_ALIASES = {
    "mm": ("MM", "MILLIMETRE", "MILLIMETER", "MILLIMETRES", "MILLIMETERS"),
    "cm": ("CM", "CENTIMETRE", "CENTIMETER", "CENTIMETRES", "CENTIMETERS"),
    "dm": ("DM", "DECIMETRE", "DECIMETER"),
    "m": ("M", "METRE", "METER", "METRES", "METERS"),
    "km": ("KM", "KILOMETRE", "KILOMETER"),
    "um": ("UM", "MICROMETRE", "MICROMETER", "MICRON", "MICRONS"),
    "nm": ("NM", "NANOMETRE", "NANOMETER"),
    "inch": ("INCH", "IN", "INCHES"),
    "ft": ("FOOT", "FT", "FEET"),
    "yd": ("YARD", "YD", "YARDS"),
    "mile": ("MILE", "MI", "MILES"),
    "mil": ("MIL", "MILS", "THOU"),
    "microinch": ("MICROINCH", "UIN", "MICROINCHES"),
}
_UNIT_LOOKUP = {alias: name for name, aliases in _UNIT_ALIASES.items() for alias in aliases}


def normalize_unit(token: Optional[str]) -> "tuple[str, Optional[float]]":
    """``(name, factor_to_mm)`` of a length-unit token, e.g. ('m', 1000.0),
    ('inch', 25.4); ('unknown', None) for an empty or unrecognised token."""
    if not token:
        return "unknown", None
    name = _UNIT_LOOKUP.get(" ".join(str(token).split()).upper())
    if name is None:
        return "unknown", None
    return name, UNIT_FACTORS[name]


def _normalize_unit(token: Optional[str]) -> str:
    return normalize_unit(token)[0]


_TESS_BODY_RE = re.compile(rb"=\s*(TESSELLATED_SOLID|TESSELLATED_SHELL)\s*\(")
# simple instances and complex ones whose first (alphabetical) part is the annotation
_ANNOTATION_RE = re.compile(rb"=\s*\(?\s*[A-Z0-9_]*ANNOTATION[A-Z0-9_]*OCCURRENCE\s*\(")
_SCAN_OVERLAP = 512


def step_entity_census(path: str) -> Dict[str, int]:
    """OCP-free count of the Part 21 records the OCCT transfer does not turn
    into drawn geometry (QA round 2 issue 24): ``tessellated_solids``,
    ``tessellated_shells`` (``TESSELLATED_SOLID`` / ``TESSELLATED_SHELL``
    records) and ``annotation_occurrences`` (``*ANNOTATION*OCCURRENCE``
    records: ``TESSELLATED_ANNOTATION_OCCURRENCE``,
    ``ANNOTATION_CURVE_OCCURRENCE``, ... ; a complex instance counts when an
    annotation occurrence is its first part). Streams the file in
    ``_SNIFF_CHUNK`` pieces with an overlap; NIST FTC 08 (2.1 MB): 1 solid,
    1 shell, 60 annotations. Zeros for an unreadable file."""
    out = {"tessellated_solids": 0, "tessellated_shells": 0, "annotation_occurrences": 0}
    try:
        with open(path, "rb") as fh:
            carry = b""
            while True:
                chunk = fh.read(_SNIFF_CHUNK)
                buf = carry + chunk
                last = not chunk
                # count matches starting before the carried tail (the next buffer
                # starts with it); the last buffer counts everything
                limit = len(buf) if last else max(0, len(buf) - _SCAN_OVERLAP)
                for m in _TESS_BODY_RE.finditer(buf):
                    if m.start() < limit:
                        key = ("tessellated_solids" if m.group(1) == b"TESSELLATED_SOLID"
                               else "tessellated_shells")
                        out[key] += 1
                out["annotation_occurrences"] += sum(
                    1 for m in _ANNOTATION_RE.finditer(buf) if m.start() < limit)
                if last:
                    break
                carry = buf[limit:]
    except OSError as exc:
        logger.debug("STEP entity census failed: %s", exc)
    return out


def tessellated_dropped(path: str, bodies: Dict[str, Any]) -> Dict[str, int]:
    """What a tessellated-only STEP leaves out of the drawn mesh (issue 24).

    The transfer's own left-outs (``bodies["dropped"]``: free shells next to
    the solid, loose edges, ...) plus, from the text
    (``step_entity_census``): ``tessellated_shells_outside`` = tessellated
    solids and shells beyond the drawn bodies (the larger of the transfer's
    dropped free shells / faces / tessellated bodies and the text count
    minus the drawn bodies; NIST FTC 08: 1, a 'BRepFtrWithContract' shell
    of the supplemental geometry) and ``annotation_occurrences`` (PMI
    presentation, never transferred as shapes; FTC 08: 60). Only non-zero
    counts are kept."""
    out: Dict[str, int] = {k: int(v) for k, v in (bodies.get("dropped") or {}).items()
                           if int(v or 0) > 0}
    text = step_entity_census(path)
    if bodies.get("drawn") == "solids":
        drawn = int(bodies.get("solids") or 0)
    else:
        drawn = int(bodies.get("free_shells") or 0) + (1 if bodies.get("free_faces") else 0)
    from_transfer = sum(int(out.get(k, 0)) for k in ("free_shells", "free_faces",
                                                      "tessellated_bodies"))
    outside = max(from_transfer,
                  text["tessellated_solids"] + text["tessellated_shells"] - drawn)
    if outside > 0:
        out["tessellated_shells_outside"] = int(outside)
    if text["annotation_occurrences"] > 0:
        out["annotation_occurrences"] = int(text["annotation_occurrences"])
    return out


def step_unit_sniff(path: str) -> str:
    """Length unit from the Part 21 text, normalised: "mm", "inch", "m",
    "cm", "ft", ... (keys of ``UNIT_FACTORS``) or "unknown" for an
    unrecognised unit and unreadable files.
    """
    return _normalize_unit(_sniff_raw(path))


def _file_units(reader: Any) -> List[str]:
    """OCCT's length units of a read (not yet transferred) STEP file, e.g.
    ``['millimetre']`` or ``['INCH']``; [] when OCCT reports none."""
    from OCP.TColStd import TColStd_SequenceOfAsciiString
    try:
        lu, au, su = (TColStd_SequenceOfAsciiString(), TColStd_SequenceOfAsciiString(),
                      TColStd_SequenceOfAsciiString())
        reader.FileUnits(lu, au, su)
        return [lu.Value(i).ToCString() for i in range(1, lu.Length() + 1)]
    except Exception as exc:
        logger.debug("FileUnits failed: %s", exc)
        return []


def load_step(path: str, fast_census_above_faces: Optional[int] = None) -> BrepModel:
    """Read a STEP file safely and return its ``BrepModel``.

    Raises ``ValueError`` when the file is missing, cannot be parsed or
    transfers no geometry. The transfer is never
    attempted after a failed ``ReadFile`` (that is a native crash).

    ``source_unit`` comes from OCCT ``FileUnits`` first; the OCP-free text
    sniff (``step_unit_sniff``) is the fallback when FileUnits is empty or
    names an unsupported unit, and an independent cross-check otherwise: a
    disagreement adds a "length unit disagreement" warning.
    ``source_unit_factor`` is its factor to mm (OCCT already scaled the
    geometry by it).

    The model's ``shape`` is the drawn part of the transfer
    (``select_drawn_shape``: the solids when there are any, loose curve sets
    and supplemental surface shells left out); ``bodies`` says what was left
    out and a "supplemental geometry not drawn" warning names it.
    ``original_shape`` is the full transfer.

    Raises ``TessellatedOnlyError`` (a ``ValueError``) when every drawn face
    is a triangulation without a surface; its ``mesh`` holds the triangles
    for the mesh route.

    ``fast_census_above_faces`` is passed to the census as
    ``fast_above_faces`` (the pipeline passes ``exact_hlr_max_faces``).
    """
    import os
    if not os.path.isfile(path):
        raise ValueError(f"STEP file not found: {path}")
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_Reader
    from OCP.TopAbs import TopAbs_FACE
    from src.io.occt_utils import quiet_occt

    quiet_occt()
    reader = STEPControl_Reader()
    try:
        status = reader.ReadFile(str(path))
    except Exception as exc:
        raise ValueError(f"STEP read failed: {exc}") from exc
    if status != IFSelect_RetDone:
        # NEVER call NbRootsForTransfer / TransferRoots / OneShape here
        raise ValueError(f"not a readable STEP file (ReadFile status {status})")

    file_units = _file_units(reader)
    try:
        roots = int(reader.NbRootsForTransfer())
        reader.TransferRoots()
        shape = reader.OneShape()
    except Exception as exc:
        raise ValueError(f"STEP transfer failed: {exc}") from exc
    if shape is None or shape.IsNull():
        raise ValueError("STEP file contains no transferable geometry")

    occt_unit = _normalize_unit(file_units[0]) if file_units else "unknown"
    text_unit = step_unit_sniff(path)
    source_unit = occt_unit if occt_unit != "unknown" else text_unit
    factor = UNIT_FACTORS.get(source_unit)

    drawn, bodies = select_drawn_shape(shape)
    n_faces = _count_unique(drawn, TopAbs_FACE)
    if n_faces and count_tessellated_faces(drawn) == n_faces:
        try:
            mesh, used, n_tri = triangulation_mesh(drawn)
        except Exception as exc:
            mesh, used, n_tri = None, 0, 0
            logger.warning("tessellated STEP: no mesh could be built: %s", exc)
        try:
            dropped = tessellated_dropped(path, bodies)
        except Exception as exc:
            logger.debug("tessellated left-out census failed: %s", exc)
            dropped = dict(bodies.get("dropped") or {})
        raise TessellatedOnlyError(
            f"tessellated-only STEP: all {n_faces} face(s) are triangulations without a "
            f"surface (AP242 TESSELLATED_SOLID / TRIANGULATED_FACE); use the mesh route",
            mesh, faces=used, triangles=n_tri, source_unit=source_unit,
            source_unit_factor=factor, roots=roots, bodies=bodies, dropped=dropped)

    model = analyze_shape(drawn, source_unit=source_unit, roots=roots,
                          fast_above_faces=fast_census_above_faces)
    model.source_unit_factor = factor
    model.bodies = bodies
    model.original_shape = shape
    msg = _dropped_warning(bodies)
    if msg:
        model.warnings.append(msg)
    if file_units and text_unit != "unknown" and occt_unit != text_unit:
        model.warnings.append(f"length unit disagreement: OCCT FileUnits says "
                              f"{file_units[0]!r}, the text sniff says {text_unit}")
    if model.census["faces"] == 0:
        raise ValueError("STEP file contains no faces")
    return model



# ---------------------------------------------------------------------------
# transform
# ---------------------------------------------------------------------------

def transform_shape(shape, matrix):
    """Rotated copy of ``shape``: world' = R @ world.

    ``matrix`` is a 3x3 proper rotation (det +1), a 3x4 / 4x4 rigid transform
    (rotation + translation), or an object with a ``.matrix`` attribute such as
    ``projection.Orientation``. Raises ``ValueError`` for a mirror, a scale or
    a non-orthogonal matrix. The input shape is not modified.
    """
    M = np.asarray(getattr(matrix, "matrix", matrix), dtype=float)
    if M.shape == (3, 3):
        R, t = M, np.zeros(3)
    elif M.shape in ((3, 4), (4, 4)):
        R, t = M[:3, :3], M[:3, 3]
        if M.shape == (4, 4) and not np.allclose(M[3], [0, 0, 0, 1], atol=_ROTATION_TOL):
            raise ValueError("4x4 matrix is not affine (last row != 0 0 0 1)")
    else:
        raise ValueError(f"transform matrix must be 3x3, 3x4 or 4x4, got {M.shape}")
    if not np.all(np.isfinite(M)):
        raise ValueError("transform matrix is not finite")
    if not np.allclose(R @ R.T, np.eye(3), atol=_ROTATION_TOL):
        raise ValueError("transform matrix is not orthonormal (scale or shear)")
    det = float(np.linalg.det(R))
    if abs(det - 1.0) > _ROTATION_TOL:
        raise ValueError(f"transform matrix is not a proper rotation (det {det:+.6f})")

    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCP.gp import gp_Trsf
    trsf = gp_Trsf()
    trsf.SetValues(float(R[0, 0]), float(R[0, 1]), float(R[0, 2]), float(t[0]),
                   float(R[1, 0]), float(R[1, 1]), float(R[1, 2]), float(t[1]),
                   float(R[2, 0]), float(R[2, 1]), float(R[2, 2]), float(t[2]))
    try:
        return BRepBuilderAPI_Transform(shape, trsf, True).Shape()
    except Exception as exc:
        raise ValueError(f"shape transform failed: {exc}") from exc


# ---------------------------------------------------------------------------
# faceted shell from a mesh (STL route)
# ---------------------------------------------------------------------------

def _encode(shape: Any, angle: float, warn: Callable[[str], None]) -> None:
    from OCP.BRepLib import BRepLib
    try:
        BRepLib.EncodeRegularity_s(shape, float(angle))
    except Exception as exc:
        warn(f"regularity coding failed ({exc}); facet edges will be drawn")


def faceted_shape_from_mesh(mesh, smooth_angle_rad: float, warn) -> object:
    """One planar face per triangle, edges shared, smooth edges coded C1.

    Returns a SOLID for a closed component (oriented outward), a SHELL for an
    open one, and a COMPOUND of those for several components. The edges whose
    dihedral angle is below ``smooth_angle_rad`` are coded by
    ``BRepLib.EncodeRegularity_s`` on each shell / solid (never on a bare
    compound, where it does nothing), so ``HLRBRep_PolyAlgo`` draws the
    silhouette instead of every facet. ``warn`` receives human-readable
    problems (degenerate triangles, open or non-manifold components).
    Raises ``ValueError`` when nothing can be built.
    """
    import trimesh
    from OCP.BRep import BRep_Builder
    from OCP.TopoDS import TopoDS_Compound
    from src.io.faceted_step import (_build_direct, _edge_census, _nondegenerate_mask,
                                     _oriented_solid, _shell_from_faces)
    from src.io.occt_utils import quiet_occt
    from OCP.BRep import BRep_Tool

    quiet_occt()
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if faces.ndim != 2 or len(faces) == 0 or faces.shape[1] != 3:
        raise ValueError("mesh has no triangles")
    work = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    work.merge_vertices()
    vertices = np.asarray(work.vertices, dtype=np.float64)
    faces = np.asarray(work.faces, dtype=np.int64)
    keep = _nondegenerate_mask(vertices, faces)
    if int((~keep).sum()):
        warn(f"{int((~keep).sum())} degenerate triangle(s) skipped")
    faces = faces[keep]
    if len(faces) == 0:
        raise ValueError("every triangle is degenerate")

    work = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    # repair=False: trimesh's split() would otherwise fill holes
    parts = work.split(only_watertight=False, repair=False) or [work]
    bodies: List[Any] = []
    for index, part in enumerate(parts):
        p_vertices = np.asarray(part.vertices, dtype=np.float64)
        p_faces = np.asarray(part.faces, dtype=np.int64)
        if len(p_faces) == 0:
            continue
        census = _edge_census(p_faces)
        if census["winding_conflicts"]:
            try:
                trimesh.repair.fix_normals(part)
                p_faces = np.asarray(part.faces, dtype=np.int64)
                census = _edge_census(p_faces)
            except Exception as exc:
                warn(f"component {index}: winding repair failed ({exc})")
        if census["nonmanifold_edges"]:
            warn(f"component {index}: {census['nonmanifold_edges']} non-manifold edge(s)")
        try:
            shell = _shell_from_faces(_build_direct(p_vertices, p_faces))
        except Exception as exc:
            warn(f"component {index}: faceted build failed ({exc}); skipped")
            continue
        closed = False
        try:
            closed = bool(BRep_Tool.IsClosed_s(shell))
            shell.Closed(closed)
        except Exception as exc:
            logger.debug("IsClosed failed: %s", exc)
        body = shell
        if closed:
            try:
                body = _oriented_solid(shell, warn)
            except Exception as exc:
                warn(f"component {index}: solid build failed ({exc}); kept as a shell")
                body = shell
        else:
            warn(f"component {index}: open shell ({census['open_edges']} open mesh "
                 f"edge(s)); drawn as a surface body")
        _encode(body, smooth_angle_rad, warn)
        bodies.append(body)

    if not bodies:
        raise ValueError("no faceted body could be built from the mesh")
    if len(bodies) == 1:
        return bodies[0]
    builder = BRep_Builder()
    comp = TopoDS_Compound()
    builder.MakeCompound(comp)
    for b in bodies:
        builder.Add(comp, b)
    return comp
