"""Faceted STEP export: one planar B-Rep face per triangle, coplanar merged.

Why this exists. A smooth NURBS body reproduces a scanned or topology-
optimised part beautifully and then loses every boolean: OCC and Parasolid
both fail to subtract two dense, tangentially-touching spline shells. A
*faceted* solid - every triangle carried into the B-Rep as its own planar
face, sharing its edges with its neighbours - has no such trouble: the
geometry is exact (the mesh *is* the truth), the surfaces are planes, and
subtraction is a purely combinatorial problem. This module is the in-repo
replacement for the FreeCAD "polyhedral STEP" detour the foxcore project had
to take by hand.

Pattern and pipeline semantics from BlinkingSun/stl2step (MIT); the cited
line numbers refer to that repository:

  * ``stl2step.cpp:400-406``, ``:710``  degenerate-triangle gates: the index
        gate (a==b) and a *relative* area gate ``mag^2 < l2^2 * 1e-20``, so a
        sliver is judged against the size of its own edges, not an absolute mm.
  * ``stl2step.cpp:497-527``  mesh-side edge census in plain arithmetic,
        before any OCCT object exists: an undirected edge used once is an
        open boundary, used more than twice is non-manifold, used twice in the
        SAME direction is a winding conflict.
  * ``stl2step.cpp:618-772``  the clean ("direct") path: one shell built from
        faces that already share their edges, closure MEASURED (not assumed),
        the stored ``Closed()`` flag re-stamped with the measured value, and
        the solid re-built from the reversed shell when the classifier says an
        infinite point is INSIDE it.
  * ``stl2step.cpp:774-810``  the dirty ("sewn") path: an independent planar
        face per triangle, ``BRepBuilderAPI_Sewing`` at a size-relative
        tolerance, ``ShapeFix_Shell``, then the same closure test.
  * ``stl2step.cpp:895-905`` and ``refit_prism_build.cpp:750-804``
        (``unifySameOnce``)  the coplanar merge, with a volume post-condition
        that discards a merge which moved the solid.
  * ``stl2step.cpp:212-241`` (``fitPlanarTolerances``)  after merging, raise
        vertex/edge tolerances to the measured distance from the merged plane.
        Float32 STL coordinates put the corners of a merged face microns off
        its plane; the geometry is honest, only the stored tolerance is too
        optimistic. Measured on a 1200 mm box, subdivided, rotated obliquely
        and quantised to float32: without this pass BRepCheck_Analyzer calls
        the merged solid invalid; with it (31 vertices, 40 edges raised, max
        deviation 9.6e-5 mm) it is valid.

Contract: never raise for bad *geometry* - a mesh with holes still produces a
file, with the damage reported. Raise only for programmer errors (wrong type).
"""
from __future__ import annotations

import logging
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Policy constants (named so the code reads as policy, not as magic numbers)
# ---------------------------------------------------------------------------

#: Absolute floor on |cross product| below which a triangle has no usable
#: normal at all (stl2step.cpp:710).
_DEGENERATE_ABS = 1e-12

#: Relative area gate: ``mag^2 < l2^2 * 1e-20`` with ``l2`` the longer of the
#: two edge vectors squared (stl2step.cpp:710). Scale free.
_DEGENERATE_REL = 1e-20

#: Post-condition on the coplanar merge: the merged body may differ from the
#: faceted one by at most this *relative* volume, otherwise the merge is
#: discarded and the facets are kept. stl2step uses the same 1e-6 floor
#: (``refit_prism_build.cpp:782``, ``vFloor = 1e-6 * |V0|``).
#:
#: This is NOT the accuracy of the geometry - the merged face spans exactly
#: the same boundary as the facets it replaces. It is the resolution of the
#: comparison: OCCT's adaptive volume integrator returns a slightly different
#: number for one planar face with a 32-segment wire than for the 32 triangles
#: covering it. Measured on the rotated float32 1200 mm test part:
#: 576000004.27 mm3 faceted vs 576000024.15 mm3 merged, i.e. 3.5e-8 relative.
#: A 1e-9 budget would therefore reject every merge on real float32 STL data.
_UNIFY_VOLUME_TOL_REL = 1e-6

#: Rough STEP cost per triangle before merging, for the size warning.
_BYTES_PER_TRIANGLE = 2500


# ---------------------------------------------------------------------------
# Mesh-side helpers - plain numpy, no OCCT
# ---------------------------------------------------------------------------

def _nondegenerate_mask(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Boolean mask of triangles worth building (stl2step.cpp:400-406, :710).

    Two gates, both from stl2step: a repeated vertex index, and a cross
    product too small *relative to the triangle's own edge lengths*.
    """
    if len(faces) == 0:
        return np.zeros(0, dtype=bool)
    a = vertices[faces[:, 0]]
    b = vertices[faces[:, 1]]
    c = vertices[faces[:, 2]]
    n = np.cross(b - a, c - a)
    mag = np.linalg.norm(n, axis=1)
    l2 = np.maximum(((b - a) ** 2).sum(axis=1), ((c - a) ** 2).sum(axis=1))
    distinct = ((faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2])
                & (faces[:, 0] != faces[:, 2]))
    return distinct & (mag >= _DEGENERATE_ABS) & (mag * mag >= l2 * l2 * _DEGENERATE_REL)


def _edge_census(faces: np.ndarray) -> Dict[str, Any]:
    """Undirected edge census of a triangle soup (stl2step.cpp:497-527).

    * used once            -> open boundary edge
    * used more than twice -> non-manifold edge
    * used exactly twice, both times in the same direction -> winding conflict
    """
    if len(faces) == 0:
        return {"open_edges": 0, "nonmanifold_edges": 0, "winding_conflicts": 0,
                "clean": False}
    directed = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    undirected = np.sort(directed, axis=1)
    _, inverse, counts = np.unique(undirected, axis=0,
                                   return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    forward = np.zeros(len(counts), dtype=np.int64)
    np.add.at(forward, inverse, (directed[:, 0] < directed[:, 1]).astype(np.int64))
    open_edges = int((counts == 1).sum())
    nonmanifold = int((counts > 2).sum())
    # a manifold edge is traversed once in each direction; anything else means
    # its two triangles are wound the same way round.
    conflicts = int(((counts == 2) & (forward != 1)).sum())
    return {"open_edges": open_edges, "nonmanifold_edges": nonmanifold,
            "winding_conflicts": conflicts,
            "clean": open_edges == 0 and nonmanifold == 0 and conflicts == 0}


def _signed_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    """Divergence-theorem volume of a closed triangle soup (stl2step.cpp:490-496)."""
    if len(faces) == 0:
        return 0.0
    a = vertices[faces[:, 0]]
    b = vertices[faces[:, 1]]
    c = vertices[faces[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


# ---------------------------------------------------------------------------
# OCCT helpers
# ---------------------------------------------------------------------------

def _count(shape: Any, kind: Any) -> int:
    from OCP.TopExp import TopExp
    from OCP.TopTools import TopTools_IndexedMapOfShape
    mp = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape, kind, mp)
    return mp.Extent()


def _triangulation(vertices: np.ndarray, faces: np.ndarray) -> Any:
    """Fill a 1-based ``Poly_Triangulation`` from numpy arrays."""
    from OCP.Poly import Poly_Triangulation, Poly_Triangle
    from OCP.gp import gp_Pnt

    tri = Poly_Triangulation(len(vertices), len(faces), False)
    for i, v in enumerate(vertices):
        tri.SetNode(i + 1, gp_Pnt(float(v[0]), float(v[1]), float(v[2])))
    for i, f in enumerate(faces):
        tri.SetTriangle(i + 1, Poly_Triangle(int(f[0]) + 1, int(f[1]) + 1,
                                             int(f[2]) + 1))
    return tri


def _shell_from_faces(faces: List[Any]) -> Any:
    from OCP.BRep import BRep_Builder
    from OCP.TopoDS import TopoDS_Shell
    builder = BRep_Builder()
    shell = TopoDS_Shell()
    builder.MakeShell(shell)
    for face in faces:
        builder.Add(shell, face)
    return shell


def _oriented_solid(shell: Any, warn) -> Any:
    """Solid from ``shell``, rebuilt from the reversed shell when it is
    inside out (``orientedSolid``, stl2step.cpp:734-751)."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeSolid
    from OCP.BRepClass3d import BRepClass3d_SolidClassifier
    from OCP.Precision import Precision
    from OCP.TopAbs import TopAbs_IN
    from OCP.TopoDS import TopoDS

    solid = BRepBuilderAPI_MakeSolid(shell).Solid()
    try:
        classifier = BRepClass3d_SolidClassifier(solid)
        classifier.PerformInfinitePoint(Precision.Confusion_s())
        if classifier.State() == TopAbs_IN:
            solid = BRepBuilderAPI_MakeSolid(TopoDS.Shell_s(shell.Reversed())).Solid()
    except Exception as exc:  # OCP raises Standard_Failure subclasses
        warn(f"solid orientation check failed ({exc}); kept as built")
    return solid


def _planar_face(points: np.ndarray) -> Optional[Any]:
    """One independent planar face through three points (sewn path)."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakePolygon, BRepBuilderAPI_MakeFace
    from OCP.gp import gp_Pnt

    polygon = BRepBuilderAPI_MakePolygon(
        gp_Pnt(float(points[0][0]), float(points[0][1]), float(points[0][2])),
        gp_Pnt(float(points[1][0]), float(points[1][1]), float(points[1][2])),
        gp_Pnt(float(points[2][0]), float(points[2][1]), float(points[2][2])),
        True)
    if not polygon.IsDone():
        return None
    face = BRepBuilderAPI_MakeFace(polygon.Wire(), True)
    if not face.IsDone():
        return None
    return face.Face()


def unify_verdict(faces_before: int, faces_after: int, volume_before: float,
                  volume_after: float, volume_tol_rel: float
                  ) -> Tuple[bool, Optional[str], float]:
    """Post-condition on a coplanar merge. Pure, so both branches are testable.

    Returns ``(accepted, reason_when_rejected, relative_volume_delta)``.

    A merge is a topology edit that must not move the geometry: it replaces a
    fan of coplanar facets with one face spanning the same boundary. Two ways
    that goes wrong, both seen in stl2step's history: the merged face is built
    over the wrong wire and the volume changes, or the merge splits a face
    instead of joining it and the count goes UP. Either one means the faceted
    body was the better answer.
    """
    if volume_before:
        rel = abs(volume_after - volume_before) / abs(volume_before)
    else:
        rel = abs(volume_after)
    if rel > volume_tol_rel:
        return (False, f"volume moved by {rel:.3g} relative "
                       f"(budget {volume_tol_rel:g})", rel)
    if faces_after > faces_before:
        return False, f"face count grew {faces_before} -> {faces_after}", rel
    return True, None, rel


def _unify_same_domain(shape: Any, angle_deg: float, volume_tol_rel: float,
                       warn) -> Tuple[Any, Dict[str, Any]]:
    """Merge coplanar faces, keeping the result only if it did not move.

    ``ShapeUpgrade_UnifySameDomain(shape, UnifyEdges, UnifyFaces,
    ConcatBSplines=False)`` with ``SetSafeInputMode(True)``
    (stl2step.cpp:895-905, refit_prism_build.cpp:750-804). The post-condition
    is the point of the exercise: a merge that changes the volume, or that
    *increases* the face count, is a merge that broke something - discard it
    and keep the facets.
    """
    from OCP.Precision import Precision
    from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
    from OCP.TopAbs import TopAbs_FACE
    from src.io.step_audit import brep_volume

    info: Dict[str, Any] = {"applied": False, "faces_before": _count(shape, TopAbs_FACE),
                            "faces_after": None, "volume_rel_delta": None,
                            "rejected": None}
    try:
        volume_before = brep_volume(shape)
        usd = ShapeUpgrade_UnifySameDomain(shape, True, True, False)
        usd.SetLinearTolerance(Precision.Confusion_s())
        usd.SetAngularTolerance(math.radians(angle_deg))
        usd.SetSafeInputMode(True)
        usd.Build()
        merged = usd.Shape()
        if merged is None or merged.IsNull():
            info["rejected"] = "unify produced a null shape"
            warn("coplanar merge produced nothing; kept the faceted body")
            return shape, info
        volume_after = brep_volume(merged)
        faces_after = _count(merged, TopAbs_FACE)
        accepted, reason, rel = unify_verdict(
            info["faces_before"], faces_after, volume_before, volume_after,
            volume_tol_rel)
        info["volume_rel_delta"] = float(rel)
        info["faces_after"] = faces_after
        if not accepted:
            info["rejected"] = reason
            warn(f"coplanar merge discarded ({reason}); kept the faceted body")
            return shape, info
        info["applied"] = True
        return merged, info
    except Exception as exc:
        info["rejected"] = f"{type(exc).__name__}: {exc}"
        warn(f"coplanar merge failed ({exc}); kept the faceted body")
        return shape, info


def _fit_planar_tolerances(shape: Any) -> Dict[str, Any]:
    """Raise vertex/edge tolerances to the measured deviation from each plane.

    Line-for-line port of ``fitPlanarTolerances`` (stl2step.cpp:212-241) from
    BlinkingSun/stl2step, MIT License, Copyright (c) 2026 stl2step
    contributors.

    Straight edges reach their maximum distance from a plane at an endpoint,
    so measuring the two vertices is exact. Linear in the number of faces and
    a no-op on exact geometry; it replaces a full ShapeFix pass. The counts
    returned are per face occurrence, not per unique vertex/edge: the same
    vertex is measured once against every planar face that carries it.
    """
    from OCP.BRep import BRep_Builder, BRep_Tool
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_SurfaceType
    from OCP.Precision import Precision
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_VERTEX
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    builder = BRep_Builder()
    updated_vertices = 0
    updated_edges = 0
    max_deviation = 0.0

    faces = TopExp_Explorer(shape, TopAbs_FACE)
    while faces.More():
        face = TopoDS.Face_s(faces.Current())
        faces.Next()
        surface = BRepAdaptor_Surface(face, False)
        if surface.GetType() != GeomAbs_SurfaceType.GeomAbs_Plane:
            continue
        plane = surface.Plane()

        vertices = TopExp_Explorer(face, TopAbs_VERTEX)
        while vertices.More():
            vertex = TopoDS.Vertex_s(vertices.Current())
            vertices.Next()
            d = plane.Distance(BRep_Tool.Pnt_s(vertex))
            max_deviation = max(max_deviation, d)
            if d > BRep_Tool.Tolerance_s(vertex):
                builder.UpdateVertex(vertex, d * 1.001 + Precision.Confusion_s())
                updated_vertices += 1

        edges = TopExp_Explorer(face, TopAbs_EDGE)
        while edges.More():
            edge = TopoDS.Edge_s(edges.Current())
            edges.Next()
            d = 0.0
            ends = TopExp_Explorer(edge, TopAbs_VERTEX)
            while ends.More():
                d = max(d, plane.Distance(
                    BRep_Tool.Pnt_s(TopoDS.Vertex_s(ends.Current()))))
                ends.Next()
            if d > BRep_Tool.Tolerance_s(edge):
                builder.UpdateEdge(edge, d * 1.001 + Precision.Confusion_s())
                updated_edges += 1

    return {"updated_vertices": updated_vertices, "updated_edges": updated_edges,
            "max_deviation_mm": float(max_deviation)}


def _premeasure(shape: Any) -> Dict[str, Any]:
    """Counts, closure, volume and validity of the shape *before* writing.

    Deliberately not ``step_audit.measure_shape``: that one also tessellates
    the shape to compare bounding boxes, which is wasted work here (the full
    audit runs on the written file when ``verify=True``). The one number that
    the read-back audit CANNOT give back is BRepCheck validity: STEP stores
    tolerances as a file-wide uncertainty, so the reader hands out its own
    tolerances and a shape that was invalid in memory reads back valid.
    """
    from OCP.BRep import BRep_Tool
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.TopAbs import TopAbs_FACE, TopAbs_SHELL, TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS
    from src.io.step_audit import brep_volume

    out: Dict[str, Any] = {
        "faces": _count(shape, TopAbs_FACE),
        "solids": _count(shape, TopAbs_SOLID),
        "shells": _count(shape, TopAbs_SHELL),
    }
    closed: List[bool] = []
    exp = TopExp_Explorer(shape, TopAbs_SHELL)
    while exp.More():
        closed.append(bool(BRep_Tool.IsClosed_s(TopoDS.Shell_s(exp.Current()))))
        exp.Next()
    out["closed_measured"] = closed
    try:
        out["brep_volume_mm3"] = brep_volume(shape)
    except Exception as exc:
        logger.error("volume integration failed: %s", exc)
        out["brep_volume_mm3"] = None
    try:
        out["brepcheck_valid"] = bool(BRepCheck_Analyzer(shape).IsValid())
    except Exception as exc:
        logger.error("BRepCheck failed: %s", exc)
        out["brepcheck_valid"] = False
    return out


# ---------------------------------------------------------------------------
# per-component construction
# ---------------------------------------------------------------------------

def _finish_shell(shell: Any, make_solids: bool, index: int, how: str,
                  open_edges: int, warn) -> Tuple[Any, bool, bool]:
    """Measure closure, re-stamp the flag, solidify (stl2step.cpp:752-771).

    Returns ``(shape, closed, is_solid)``. The stored ``Closed()`` flag is
    never trusted - it is overwritten with what ``BRep_Tool::IsClosed``
    actually measured, because a shell that merely *claims* to be closed
    imports as a solid and then fails every boolean downstream.
    """
    from OCP.BRep import BRep_Tool

    closed = bool(BRep_Tool.IsClosed_s(shell))
    shell.Closed(closed)
    if closed and make_solids:
        return _oriented_solid(shell, warn), True, True
    if make_solids:
        warn(f"component {index}: open shell ({open_edges} open mesh edges, "
             f"{how}); exported as a surface body, CAM and booleans will "
             f"not treat it as a volume")
    return shell, closed, False


def _build_direct(vertices: np.ndarray, faces: np.ndarray) -> List[Any]:
    """Clean component -> one face per triangle with SHARED edges.

    ``BRepBuilderAPI_MakeShapeOnMesh`` returns a COMPOUND of faces (no shell)
    whose neighbouring faces already share their edge and vertex TShapes:
    measured on a subdivision-3 icosphere, 1280 faces and exactly 1920 unique
    edges = 3F/2, the Euler count for a closed triangulated surface. That is
    what makes the shell close without any sewing at all.
    """
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeShapeOnMesh
    from OCP.TopoDS import TopoDS, TopoDS_Iterator

    maker = BRepBuilderAPI_MakeShapeOnMesh(_triangulation(vertices, faces))
    maker.Build()
    if not maker.IsDone():
        raise RuntimeError("BRepBuilderAPI_MakeShapeOnMesh did not finish")
    compound = maker.Shape()
    if compound is None or compound.IsNull():
        raise RuntimeError("BRepBuilderAPI_MakeShapeOnMesh produced nothing")
    out: List[Any] = []
    it = TopoDS_Iterator(compound)
    while it.More():
        out.append(TopoDS.Face_s(it.Value()))
        it.Next()
    if not out:
        raise RuntimeError("BRepBuilderAPI_MakeShapeOnMesh produced no faces")
    return out


def _build_sewn(vertices: np.ndarray, faces: np.ndarray, warn
                ) -> Tuple[List[Any], int]:
    """Dirty component -> independent planar faces, sewn (stl2step.cpp:774-810).

    Returns ``(shells, skipped_faces)``. The sewing tolerance is relative to
    the component's own bounding-box diagonal so a 1 mm feature and a 1200 mm
    plate get the same treatment.
    """
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Sewing
    from OCP.ShapeFix import ShapeFix_Shell
    from OCP.TopAbs import TopAbs_FACE, TopAbs_SHELL
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    lo = vertices.min(axis=0)
    hi = vertices.max(axis=0)
    diagonal = float(np.linalg.norm(hi - lo))
    tolerance = max(1e-4 * diagonal, 1e-3)

    sewing = BRepBuilderAPI_Sewing(tolerance)
    added = 0
    skipped = 0
    for f in faces:
        face = _planar_face(vertices[f])
        if face is None:
            skipped += 1
            continue
        sewing.Add(face)
        added += 1
    if added == 0:
        return [], skipped
    sewing.Perform()
    sewed = sewing.SewedShape()
    if sewed is None or sewed.IsNull():
        warn("sewing produced nothing for one component; component dropped")
        return [], skipped

    shells: List[Any] = []
    exp = TopExp_Explorer(sewed, TopAbs_SHELL)
    while exp.More():
        shell = TopoDS.Shell_s(exp.Current())
        exp.Next()
        try:
            fix = ShapeFix_Shell(shell)
            fix.Perform()
            inner = TopExp_Explorer(fix.Shape(), TopAbs_SHELL)
            found = False
            while inner.More():
                shells.append(TopoDS.Shell_s(inner.Current()))
                inner.Next()
                found = True
            if not found:
                shells.append(shell)
        except Exception as exc:
            warn(f"ShapeFix_Shell failed ({exc}); kept the sewn shell as is")
            shells.append(shell)

    # faces the sewing could not attach to any shell: keep them, do not lose
    # material silently (stl2step.cpp:797-805 "unsewn leftovers").
    leftovers: List[Any] = []
    free = TopExp_Explorer(sewed, TopAbs_FACE, TopAbs_SHELL)
    while free.More():
        leftovers.append(TopoDS.Face_s(free.Current()))
        free.Next()
    if leftovers:
        shells.append(_shell_from_faces(leftovers))
    if not shells:
        warn("sewing returned no shells or faces for one component; dropped")
    return shells, skipped


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def export_step_faceted(mesh, filepath, *, schema: str = "AP214IS",
                        product_name: Optional[str] = None, unify: bool = True,
                        unify_angle_deg: float = 0.001, make_solids: bool = True,
                        fit_tolerances: bool = True, verify: bool = True,
                        volume_tol_pct: float = 1e-4,
                        size_warn_triangles: int = 200_000,
                        unify_volume_tol_rel: float = _UNIFY_VOLUME_TOL_REL,
                        ) -> Dict[str, Any]:
    """Write ``mesh`` as a faceted STEP body and report what was built.

    Args:
        mesh: a ``trimesh.Trimesh`` of triangles.
        filepath: output ``.step`` path.
        schema: STEP application protocol ("AP203", "AP214IS", "AP242DIS").
        product_name: PRODUCT name in the file (defaults to the file stem).
        unify: merge coplanar neighbouring faces after building.
        unify_angle_deg: angular tolerance of that merge, in degrees.
        make_solids: turn closed shells into solids (off = surface bodies).
        fit_tolerances: run the planar tolerance fit after the merge.
        verify: re-read the written file with ``step_audit.measure_step``.
        volume_tol_pct: budget handed to ``step_audit.verdict``, in PERCENT.
            The default 1e-4 % (1e-6 relative) matches the coplanar-merge
            post-condition: merging N triangles into one planar face with an
            N-segment wire shifts OCCT's integrated volume by up to ~1e-6
            relative although the geometry is identical (measured 5.4e-7 on
            the 63,676-triangle foxcore void body, 4,408 faces merged, while
            the sampled deviation and bbox error stayed below 1e-9 mm).
        size_warn_triangles: warn above this triangle count (never refuses).
        unify_volume_tol_rel: relative volume budget of the merge
            post-condition; see ``_UNIFY_VOLUME_TOL_REL`` for why it is 1e-6.

    Returns:
        A JSON-serialisable report. ``ok`` is True whenever a file was
        written, even if the input had holes - the damage is in
        ``mesh_census``, ``open_shells``, ``inverted_components`` and
        ``warnings``.

    Raises:
        TypeError: for a wrong argument type. Bad *geometry* never raises.
    """
    import trimesh

    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"mesh must be a trimesh.Trimesh, got {type(mesh).__name__}")
    if not isinstance(filepath, (str, os.PathLike)):
        raise TypeError(f"filepath must be a path, got {type(filepath).__name__}")
    filepath = os.fspath(filepath)

    t0 = time.time()
    warnings: List[str] = []

    def warn(message: str) -> None:
        warnings.append(message)
        logger.warning(message)

    report: Dict[str, Any] = {
        "ok": False,
        "output": os.path.abspath(filepath),
        "schema": schema,
        "triangles": int(len(mesh.faces)),
        "components": 0,
        "skipped_degenerate": 0,
        "mesh_census": {"open_edges": 0, "nonmanifold_edges": 0,
                        "winding_conflicts": 0, "clean": False},
        "bodies": [],
        "faces": 0,
        "solids": 0,
        "open_shells": 0,
        "inverted_components": 0,
        "volume_mesh_mm3": None,
        "volume_brep_mm3": None,
        "volume_delta_pct": None,
        "brepcheck_valid": None,
        "warnings": warnings,
        "seconds": 0.0,
    }

    def finish(ok: bool, error: Optional[str] = None) -> Dict[str, Any]:
        report["ok"] = ok
        if error is not None:
            report["error"] = error
        report["seconds"] = round(time.time() - t0, 3)
        return report

    try:
        from src.io.occt_utils import quiet_occt
        quiet_occt()

        if len(mesh.faces) == 0:
            return finish(False, "mesh has no triangles")
        if int(np.asarray(mesh.faces).shape[1]) != 3:
            return finish(False, "mesh is not triangulated")

        if len(mesh.faces) > size_warn_triangles:
            warn(f"{len(mesh.faces)} triangles: the STEP will be roughly "
                 f"{len(mesh.faces) * _BYTES_PER_TRIANGLE / 1e6:.0f} MB before the "
                 f"coplanar merge; consider decimating the mesh first")

        # -- 1. weld, drop degenerates, then split -------------------------
        # Welding BEFORE the split is deliberate: an STL arrives as loose
        # triangles, and splitting an unwelded soup reports every seam as an
        # open edge and every triangle as its own component.
        work = mesh.copy()
        work.merge_vertices()
        vertices = np.asarray(work.vertices, dtype=np.float64)
        faces = np.asarray(work.faces, dtype=np.int64)
        keep = _nondegenerate_mask(vertices, faces)
        report["skipped_degenerate"] = int((~keep).sum())
        faces = faces[keep]
        if len(faces) == 0:
            return finish(False, "every triangle is degenerate")
        if report["skipped_degenerate"]:
            warn(f"{report['skipped_degenerate']} degenerate triangle(s) skipped")

        # -- 2. mesh-side census of the whole input ------------------------
        report["mesh_census"] = _edge_census(faces)
        if not report["mesh_census"]["clean"]:
            c = report["mesh_census"]
            warn(f"input mesh is not a clean closed surface: "
                 f"{c['open_edges']} open edge(s), {c['nonmanifold_edges']} "
                 f"non-manifold edge(s), {c['winding_conflicts']} winding conflict(s)")

        work = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        # repair=False is load bearing: trimesh's split() forwards repair=True
        # to submesh(), which FILLS HOLES. Measured on a box with one triangle
        # deleted: split returns a component with 12 triangles, the shell
        # closes, and the exporter reports a solid for an input that has a
        # hole in it. An exporter that quietly repairs its input is an
        # exporter that lies about what it wrote.
        parts = work.split(only_watertight=False, repair=False)
        if not parts:
            parts = [work]
        report["components"] = len(parts)
        split_triangles = sum(len(p.faces) for p in parts)
        if split_triangles != len(faces):
            # effect check, not a trust check: if a future trimesh changes the
            # split again, this says so instead of shipping the difference.
            warn(f"component split changed the triangle count "
                 f"{len(faces)} -> {split_triangles}; the STEP no longer "
                 f"matches the input mesh exactly")
        report["split_triangles"] = int(split_triangles)

        # -- 3/4. per component: build, close, solidify --------------------
        bodies: List[Any] = []
        volume_mesh = 0.0
        signed_volumes: List[float] = []
        for index, part in enumerate(parts):
            part.merge_vertices()
            p_vertices = np.asarray(part.vertices, dtype=np.float64)
            p_faces = np.asarray(part.faces, dtype=np.int64)
            if len(p_faces) == 0:
                continue
            census = _edge_census(p_faces)
            if census["winding_conflicts"] > 0:
                # one repair attempt, then re-measure (stl2step never trusts a
                # repair it did not measure afterwards)
                try:
                    trimesh.repair.fix_normals(part)
                    p_faces = np.asarray(part.faces, dtype=np.int64)
                    fixed = _edge_census(p_faces)
                    if fixed["winding_conflicts"] < census["winding_conflicts"]:
                        warn(f"component {index}: winding conflicts repaired "
                             f"({census['winding_conflicts']} -> "
                             f"{fixed['winding_conflicts']})")
                    census = fixed
                except Exception as exc:
                    warn(f"component {index}: winding repair failed ({exc})")
            signed = _signed_volume(p_vertices, p_faces)
            signed_volumes.append(signed)
            volume_mesh += abs(signed)

            path = "direct"
            shells: List[Any] = []
            if census["clean"]:
                try:
                    shells = [_shell_from_faces(_build_direct(p_vertices, p_faces))]
                except Exception as exc:
                    warn(f"component {index}: direct build failed ({exc}); "
                         f"falling back to sewing")
                    path = "sewn"
            else:
                path = "sewn"
            if path == "sewn":
                warn(f"component {index}: needed repair "
                     f"({census['open_edges']} open, "
                     f"{census['nonmanifold_edges']} non-manifold, "
                     f"{census['winding_conflicts']} conflicting edges); sewn")
                shells, skipped_faces = _build_sewn(p_vertices, p_faces, warn)
                if skipped_faces:
                    warn(f"component {index}: {skipped_faces} triangle(s) could "
                         f"not be turned into a face and were dropped")
            if not shells:
                warn(f"component {index}: produced no shell; component dropped")
                continue

            for shell in shells:
                shape, closed, is_solid = _finish_shell(
                    shell, make_solids, index, path, census["open_edges"], warn)
                bodies.append({"component": index, "shape": shape,
                               "triangles": int(len(p_faces)), "closed": closed,
                               "solid": is_solid, "path": path})

        # -- 4b. cavities: an inverted shell is a VOID, not a body ---------
        # A hollow part arrives as two components, the inner one wound
        # inwards. ``_oriented_solid`` re-orients every shell it builds, so
        # that cavity leaves here as a second SOLID inside the first: material
        # where the mesh says void. Nothing else in the pipeline can see it -
        # BRepCheck says valid, the shells are closed, and the volume check
        # compares |sum| against |sum| - so it is reported here or not at all.
        # Building a real solid-with-voids is a separate feature; this is the
        # honest report that the exporter did not build one.
        negatives = sum(1 for v in signed_volumes if v < 0.0)
        positives = sum(1 for v in signed_volumes if v > 0.0)
        report["inverted_components"] = int(negatives)
        if negatives and positives:
            overstated = 2.0 * sum(-v for v in signed_volumes if v < 0.0)
            warn(f"{negatives} component(s) enclose negative volume beside "
                 f"{positives} positive one(s): these are internal cavities in "
                 f"the mesh. Each was re-oriented and written as its own solid, "
                 f"not as a void, so the exported material volume is roughly "
                 f"{overstated:.4g} mm3 too high and a boolean will see filled "
                 f"pockets. Split the shells or subtract them in CAD.")

        if not bodies:
            return finish(False, "no body could be built from this mesh")

        # -- 5/6. coplanar merge + planar tolerance fit, per body ----------
        from OCP.TopAbs import TopAbs_FACE
        for body in bodies:
            body["faces_before_unify"] = _count(body["shape"], TopAbs_FACE)
            if unify:
                merged, info = _unify_same_domain(
                    body["shape"], unify_angle_deg, unify_volume_tol_rel, warn)
                body["shape"] = merged
                body["unify"] = info
            else:
                body["unify"] = {"applied": False, "faces_before": body["faces_before_unify"],
                                 "faces_after": None, "volume_rel_delta": None,
                                 "rejected": "unify=False"}
            if fit_tolerances:
                # same contract as the merge above: a throw in here must cost
                # the tolerance fit, not the finished body.
                try:
                    body["tolerance_fit"] = _fit_planar_tolerances(body["shape"])
                except Exception as exc:
                    body["tolerance_fit"] = {
                        "updated_vertices": 0, "updated_edges": 0,
                        "max_deviation_mm": 0.0,
                        "error": f"{type(exc).__name__}: {exc}"}
                    warn(f"component {body['component']}: planar tolerance fit "
                         f"failed ({exc}); tolerances left as built")
            else:
                body["tolerance_fit"] = {"updated_vertices": 0, "updated_edges": 0,
                                         "max_deviation_mm": 0.0}
            body["faces"] = _count(body["shape"], TopAbs_FACE)

        # -- 7. assemble, measure, write ----------------------------------
        shapes = [b["shape"] for b in bodies]
        if len(shapes) == 1:
            shape = shapes[0]
        else:
            from OCP.BRep import BRep_Builder
            from OCP.TopoDS import TopoDS_Compound
            builder = BRep_Builder()
            compound = TopoDS_Compound()
            builder.MakeCompound(compound)
            for s in shapes:
                builder.Add(compound, s)
            shape = compound

        report["bodies"] = [{k: v for k, v in b.items() if k != "shape"}
                            for b in bodies]
        report["solids"] = sum(1 for b in bodies if b["solid"])
        report["open_shells"] = sum(1 for b in bodies if not b["solid"])

        pre = _premeasure(shape)
        report["faces"] = pre["faces"]
        report["shells"] = pre["shells"]
        report["closed_measured"] = pre["closed_measured"]
        report["brepcheck_valid"] = pre["brepcheck_valid"]
        report["volume_brep_mm3"] = pre["brep_volume_mm3"]
        report["volume_mesh_mm3"] = float(volume_mesh)
        if volume_mesh and pre["brep_volume_mm3"] is not None:
            report["volume_delta_pct"] = float(
                100.0 * (pre["brep_volume_mm3"] - volume_mesh) / volume_mesh)
        if pre["solids"] != report["solids"]:
            warn(f"{report['solids']} solids were built but the assembled shape "
                 f"holds {pre['solids']}")
        if not pre["brepcheck_valid"]:
            warn("BRepCheck_Analyzer reports the assembled shape invalid")

        from src.io.exporters import export_step
        export_step(shape, filepath, schema=schema, product_name=product_name)
        if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
            return finish(False, f"STEP writer left no usable file at {filepath}")
        report["file_bytes"] = int(os.path.getsize(filepath))

        # -- 8. read-back audit -------------------------------------------
        if verify:
            from src.io.step_audit import measure_step, verdict
            # the welded, degenerate-free mesh is the reference: without it the
            # audit has no mesh volume / bbox / deviation and the volume budget
            # could never fire (caught by the heavy-merge regression test)
            measurement = measure_step(filepath, reference_stl=work,
                                       expected_faces=report["faces"])
            measurement.pop("census", None)
            report["audit"] = measurement
            _, reasons = verdict(measurement, volume_tol_pct=volume_tol_pct)
            report["audit_reasons"] = reasons
            for reason in reasons:
                warn("audit: " + reason)

        return finish(True)
    except Exception as exc:  # noqa: BLE001 - bad geometry must not raise
        logger.exception("faceted STEP export failed")
        return finish(False, f"{type(exc).__name__}: {exc}")
