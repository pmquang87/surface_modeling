"""Independent read-back audit of a written STEP file.

Two functions with a deliberately thin seam between them:

``measure_step``  re-reads the file with OpenCascade and returns a plain dict of
                  measurements (JSON-serialisable). No opinions.
``verdict``       a pure function of that dict: exit code + list of reasons.
                  Testable without OCP, and every gate can be driven by an
                  injected defect (see tests/test_step_audit.py).

What is measured, and why each item is there:

* faces / shells / solids           - a STEP with zero solids imports into
                                      SolidWorks as a surface body or nothing.
* closure, MEASURED with            - the stored ``Closed()`` flag is what the
  ``BRep_Tool.IsClosed``              builder believed; both are reported.
* free edges (ShapeAnalysis_FreeBounds) and BRepCheck_Analyzer validity.
* signed B-Rep volume with OCCT's adaptive integrator (``Eps = Precision::
  Confusion``): the default overload under-integrates curved faces with many-
  span wires (stl2step measured 0.06 %, this project measured +1.9 % on a
  1200 mm part). Closure + BRepCheck cannot see an inside-out or doubled
  shell; the signed volume can. Compared against the input mesh volume only
  when the mesh is watertight.
* tessellated bounding box vs the STL bounding box: caught runaway control
  points that random deviation sampling missed (never use Bnd_Box on
  B-splines - it bounds the control-point hull).
* sampled deviation STEP -> STL (one-sided surface proximity).
* an OCP-free Part 21 text census of the file when ``src.io.step_census`` is
  available: a witness that shares no code with the writer/reader.

Structure and the "volume is the cheap global falsifier" argument follow
BlinkingSun/stl2step (MIT).
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Relative tolerance under which the OCP face count and the text census are
# considered to agree (they must be equal; kept as a named constant so the
# verdict reads as policy, not magic).
_CENSUS_MUST_MATCH = True


def brep_volume(shape: Any) -> float:
    """Signed volume of a shape with OCCT's adaptive 2D Gauss integration.

    ``BRepGProp.VolumeProperties_s(shape, props)`` without an ``Eps`` uses a
    fixed Gauss scheme. Passing ``Precision.Confusion()`` (<= 0.001 switches
    OCCT to the adaptive integrator) removed a +1.9 % phantom error on a valid
    1200 mm solid in this project and a 0.06 % artefact in stl2step.
    """
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    from OCP.Precision import Precision
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, props, Precision.Confusion_s())
    return float(props.Mass())


def tessellated_nodes(shape: Any, deflection: float = 0.2) -> np.ndarray:
    """All triangulation nodes of ``shape`` after meshing it (N x 3)."""
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopoDS import TopoDS
    from OCP.BRep import BRep_Tool
    from OCP.TopLoc import TopLoc_Location

    BRepMesh_IncrementalMesh(shape, deflection)
    points: List[List[float]] = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = TopoDS.Face_s(exp.Current())
        loc = TopLoc_Location()
        poly = BRep_Tool.Triangulation_s(face, loc)
        if poly:
            for i in range(1, poly.NbNodes() + 1):
                node = poly.Node(i)
                if not loc.IsIdentity():
                    node.Transform(loc.Transformation())
                points.append([node.X(), node.Y(), node.Z()])
        exp.Next()
    return np.asarray(points, dtype=np.float64).reshape(-1, 3)


def _count(shape: Any, kind: Any) -> int:
    from OCP.TopExp import TopExp_Explorer
    n = 0
    exp = TopExp_Explorer(shape, kind)
    while exp.More():
        n += 1
        exp.Next()
    return n


def _read_step(step_path: str) -> Optional[Any]:
    """Return the shape in ``step_path`` or None when OCCT cannot produce one."""
    from OCP.STEPControl import STEPControl_Reader
    from src.io.occt_utils import quiet_occt

    quiet_occt()
    reader = STEPControl_Reader()
    try:
        if reader.ReadFile(step_path) != 1:
            return None
        if reader.TransferRoots() == 0:
            return None
        shape = reader.OneShape()
    except Exception as exc:  # OCP raises Standard_Failure subclasses
        logger.error("STEP re-read failed: %s", exc)
        return None
    if shape is None or shape.IsNull():
        return None
    return shape


def _census(step_path: str) -> Optional[Dict[str, Any]]:
    """OCP-free text census of the written file, if the module is present."""
    try:
        from src.io.step_census import census_path
    except ImportError:
        return None
    try:
        return census_path(step_path)
    except Exception as exc:  # a census that crashes must not hide the audit
        logger.warning("STEP text census failed: %s", exc)
        return {"error": str(exc)}


def _load_reference(reference: Any):
    """Accept a path to an STL/OBJ or an in-memory ``trimesh.Trimesh``."""
    if reference is None:
        return None
    import trimesh
    if isinstance(reference, trimesh.Trimesh):
        return reference
    if isinstance(reference, (str, os.PathLike)) and os.path.exists(reference):
        ref = trimesh.load_mesh(reference)
        if isinstance(ref, trimesh.Scene):
            ref = ref.dump(concatenate=True)
        return ref
    return None


def _empty_measurement(expected_faces: Optional[int]) -> Dict[str, Any]:
    return {
        "readable": False,
        "faces": 0, "solids": 0, "shells": 0,
        "closed_stored": [], "closed_measured": [],
        "free_edges": None, "brepcheck_valid": None,
        "brep_volume_mm3": None,
        "mesh_volume_mm3": None, "mesh_watertight": None,
        "volume_delta_pct": None,
        "bbox_step": None, "bbox_stl": None, "bbox_max_diff_mm": None,
        "dev_mean_mm": None, "dev_p95_mm": None, "dev_max_mm": None,
        "expected_faces": expected_faces,
        "census_faces": None, "census": None,
    }


def measure_step(step_path: str, reference_stl: Optional[str] = None,
                 expected_faces: Optional[int] = None,
                 deviation_samples: int = 5000, seed: int = 0,
                 deflection: float = 0.2) -> Dict[str, Any]:
    """Re-read ``step_path`` and measure it. Never raises on a bad file."""
    if not os.path.exists(step_path):
        m = _empty_measurement(expected_faces)
        m["step_path"] = step_path
        return m
    shape = _read_step(step_path)
    if shape is None:
        m = _empty_measurement(expected_faces)
        m["step_path"] = step_path
        m["census"] = _census(step_path)
        return m
    m = measure_shape(shape, reference_stl=reference_stl,
                      expected_faces=expected_faces,
                      deviation_samples=deviation_samples, seed=seed,
                      deflection=deflection)
    m["step_path"] = step_path
    census = _census(step_path)
    m["census"] = census
    if census and "faces" in census:
        m["census_faces"] = int(census["faces"])
    return m


def measure_shape(shape: Any, reference_stl: Optional[str] = None,
                  expected_faces: Optional[int] = None,
                  deviation_samples: int = 5000, seed: int = 0,
                  deflection: float = 0.2) -> Dict[str, Any]:
    """Measure an in-memory shape (no census - there is no file yet)."""
    m = _empty_measurement(expected_faces)
    if shape is None or shape.IsNull():
        return m
    m["readable"] = True

    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_FACE, TopAbs_SHELL, TopAbs_EDGE, TopAbs_SOLID
    from OCP.TopoDS import TopoDS
    from OCP.ShapeAnalysis import ShapeAnalysis_FreeBounds
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.BRep import BRep_Tool

    m["faces"] = _count(shape, TopAbs_FACE)
    m["solids"] = _count(shape, TopAbs_SOLID)
    shells = []
    exp = TopExp_Explorer(shape, TopAbs_SHELL)
    while exp.More():
        shells.append(TopoDS.Shell_s(exp.Current()))
        exp.Next()
    m["shells"] = len(shells)
    m["closed_stored"] = [bool(sh.Closed()) for sh in shells]
    m["closed_measured"] = [bool(BRep_Tool.IsClosed_s(sh)) for sh in shells]

    fb = ShapeAnalysis_FreeBounds(shape)
    m["free_edges"] = _count(fb.GetClosedWires(), TopAbs_EDGE) + \
        _count(fb.GetOpenWires(), TopAbs_EDGE)
    try:
        m["brepcheck_valid"] = bool(BRepCheck_Analyzer(shape).IsValid())
    except Exception as exc:
        logger.error("BRepCheck failed: %s", exc)
        m["brepcheck_valid"] = False
    try:
        m["brep_volume_mm3"] = brep_volume(shape) if m["faces"] else 0.0
    except Exception as exc:
        logger.error("volume integration failed: %s", exc)

    points = tessellated_nodes(shape, deflection) if m["faces"] else np.zeros((0, 3))
    if len(points):
        lo, hi = points.min(axis=0), points.max(axis=0)
        m["bbox_step"] = [lo.tolist(), hi.tolist()]

    ref = _load_reference(reference_stl)
    if ref is not None:
        import trimesh
        m["mesh_watertight"] = bool(ref.is_watertight)
        if ref.is_watertight:
            m["mesh_volume_mm3"] = float(ref.volume)
            if m["brep_volume_mm3"] is not None and ref.volume != 0:
                m["volume_delta_pct"] = float(
                    100.0 * (m["brep_volume_mm3"] - ref.volume) / abs(ref.volume))
        m["bbox_stl"] = [ref.bounds[0].tolist(), ref.bounds[1].tolist()]
        if m["bbox_step"] is not None:
            m["bbox_max_diff_mm"] = float(np.abs(
                np.asarray(m["bbox_step"]) - ref.bounds).max())
        if len(points):
            rng = np.random.default_rng(seed)
            k = min(deviation_samples, len(points))
            sample = points[rng.choice(len(points), k, replace=False)]
            _, dist, _ = trimesh.proximity.closest_point(ref, sample)
            m["dev_mean_mm"] = float(dist.mean())
            m["dev_p95_mm"] = float(np.percentile(dist, 95))
            m["dev_max_mm"] = float(dist.max())
    return m


def verdict(m: Dict[str, Any], volume_tol_pct: Optional[float] = None
            ) -> Tuple[int, List[str]]:
    """Exit code (0 clean, 2 written-but-suspect, 1 unusable) + reasons.

    ``volume_tol_pct``: when given, |brep - mesh| / mesh * 100 above it fails.
    None = report only. A smooth NURBS fit differs from its chord mesh by
    ~0.1 % on good parts, so a tight budget is only right for the faceted
    exporter, which must reproduce the mesh volume to machine precision.
    """
    if not m.get("readable"):
        return 1, ["written STEP cannot be re-read as a shape"]
    reasons: List[str] = []
    if m.get("shells", 0) == 0:
        reasons.append("no shell in the file")
    if m.get("solids", 0) < 1:
        reasons.append("no solid body (SolidWorks would import a surface body)")
    if m.get("free_edges") is None or m["free_edges"] != 0:
        reasons.append(f"free edges: {m.get('free_edges')}")
    if not m.get("brepcheck_valid"):
        reasons.append("BRepCheck reports the shape invalid")
    closed = m.get("closed_measured") or []
    if closed and not all(closed):
        reasons.append(f"measured shell closure: {closed}")
    vol = m.get("brep_volume_mm3")
    if vol is None or vol <= 0:
        reasons.append(f"B-Rep volume not positive: {vol}")
    delta = m.get("volume_delta_pct")
    if (volume_tol_pct is not None and m.get("mesh_watertight")
            and delta is not None and abs(delta) > volume_tol_pct):
        reasons.append(f"volume differs from the mesh by {delta:.4g} % "
                       f"(budget {volume_tol_pct:g} %)")
    exp_faces = m.get("expected_faces")
    if exp_faces is not None and m.get("faces") != exp_faces:
        reasons.append(f"{exp_faces} faces expected but {m.get('faces')} in the file")
    cf = m.get("census_faces")
    if _CENSUS_MUST_MATCH and cf is not None and cf != m.get("faces"):
        reasons.append(f"text census counts {cf} faces, OpenCascade read {m.get('faces')}")
    return (2 if reasons else 0), reasons


def format_measurement(m: Dict[str, Any]) -> str:
    """One human-readable line for the progress log."""
    parts = [f"faces={m.get('faces')}", f"solids={m.get('solids')}",
             f"shells={m.get('shells')}",
             f"closed={m.get('closed_measured')}",
             f"free-edges={m.get('free_edges')}",
             f"BRepCheck-valid={m.get('brepcheck_valid')}"]
    if m.get("brep_volume_mm3") is not None:
        parts.append(f"volume={m['brep_volume_mm3']:.3f}mm3")
    if m.get("volume_delta_pct") is not None:
        parts.append(f"dV={m['volume_delta_pct']:+.4f}%")
    if m.get("bbox_max_diff_mm") is not None:
        parts.append(f"bbox-diff={m['bbox_max_diff_mm']:.3f}mm")
    if m.get("census_faces") is not None:
        parts.append(f"census-faces={m['census_faces']}")
    return " ".join(parts)
