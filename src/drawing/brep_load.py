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
    inverted. ``bbox_min``/``bbox_max`` are lists of three floats (None for an
    empty shape). ``brepcheck_valid`` is a bool, or None when the census ran
    with ``full_check=False``. ``warnings`` are human-readable flags for the
    report.
    """

    shape: object
    source_unit: str
    roots: int
    census: dict
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# census
# ---------------------------------------------------------------------------

def _count_unique(shape: Any, kind: Any) -> int:
    from OCP.TopExp import TopExp
    from OCP.TopTools import TopTools_IndexedMapOfShape
    m = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape, kind, m)
    return m.Extent()


def _empty_census() -> Dict[str, Any]:
    return {"solids": 0, "shells": 0, "faces": 0, "edges": 0, "vertices": 0,
            "face_types": {}, "brepcheck_valid": False, "free_edges": 0,
            "closed": False, "volume_mm3": 0.0, "area_mm2": 0.0,
            "bbox_min": None, "bbox_max": None, "inverted": False}


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


def _census(shape: Any, full_check: bool = True) -> Dict[str, Any]:
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
    c["closed"] = bool(closed) and all(closed)

    try:
        fb = ShapeAnalysis_FreeBounds(shape)
        c["free_edges"] = (_count_unique(fb.GetClosedWires(), TopAbs_EDGE)
                           + _count_unique(fb.GetOpenWires(), TopAbs_EDGE))
    except Exception as exc:
        logger.warning("free-boundary analysis failed: %s", exc)
    if full_check:
        try:
            c["brepcheck_valid"] = bool(BRepCheck_Analyzer(shape).IsValid())
        except Exception as exc:
            logger.warning("BRepCheck failed: %s", exc)
            c["brepcheck_valid"] = False
    else:
        c["brepcheck_valid"] = None          # not checked (fast census)

    if c["faces"]:
        try:
            if full_check:
                vol = brep_volume(shape)     # adaptive Gauss, Precision.Confusion
            else:
                props = GProp_GProps()       # fixed Gauss: exact on planar facets
                BRepGProp.VolumeProperties_s(shape, props)
                vol = float(props.Mass())
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
    c["inverted"] = bool(c["solids"] > 0 and vol is not None and vol < 0)

    bb = exact_bbox(shape)
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
    if c["volume_mm3"] is None:
        w.append("volume integration failed")
    if c["bbox_min"] is None:
        w.append("bounding box is void")
    if source_unit != "mm":
        w.append(f"source length unit is {source_unit}; OCCT converted the "
                 f"geometry to mm on read")
    return w


def analyze_shape(shape, source_unit: str = "mm", roots: int = 1, *,
                  full_check: bool = True) -> BrepModel:
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
    """
    census = _census(shape, full_check=bool(full_check))
    return BrepModel(shape=shape, source_unit=source_unit, roots=int(roots),
                     census=census, warnings=_census_warnings(census, source_unit, roots))


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


def _normalize_unit(token: Optional[str]) -> str:
    if not token:
        return "unknown"
    t = token.strip().upper()
    if t in ("MM", "MILLIMETRE", "MILLIMETER"):
        return "mm"
    if t in ("INCH", "IN", "INCHES"):
        return "inch"
    return "unknown"


def step_unit_sniff(path: str) -> str:
    """Length unit from the Part 21 text: "mm" | "inch" | "unknown".

    Any other unit (metre, foot, ...) and unreadable files give "unknown".
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


def load_step(path: str) -> BrepModel:
    """Read a STEP file safely and return its ``BrepModel``.

    Raises ``ValueError`` when the file is missing, cannot be parsed or
    transfers no geometry. The transfer is never
    attempted after a failed ``ReadFile`` (that is a native crash).

    ``source_unit`` comes from OCCT ``FileUnits`` first; the OCP-free text
    sniff (``step_unit_sniff``) is the fallback when FileUnits is empty or
    names an unsupported unit, and an independent cross-check otherwise: a
    disagreement adds a "length unit disagreement" warning.
    """
    import os
    if not os.path.isfile(path):
        raise ValueError(f"STEP file not found: {path}")
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_Reader
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
    model = analyze_shape(shape, source_unit=source_unit, roots=roots)
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
