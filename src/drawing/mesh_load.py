"""Checked STL loading, mesh census and the prismatic/organic classifier.

The mesh route of the drawing module starts here (spec 3.2 item 3, 3.5).
Measured facts this module relies on (mesh probe, trimesh 5.0.0,
REPORT-mesh.md):

* ``trimesh.load`` NEVER raises on bad STL input: a 0-byte file, random
  bytes, STEP text saved as ``.stl`` and a truncated binary STL (header count
  larger than the data) all come back as an empty Trimesh with
  ``force="mesh"``. ``load_stl_checked`` therefore refuses 0 bytes and STEP
  text itself and refuses a result with 0 faces.
* ``process=True`` (the default) merges vertices so ``face_adjacency`` works
  (box: 36 -> 8 vertices, 18 adjacency pairs); it does NOT drop degenerate or
  duplicate faces, those are counted with ``nondegenerate_faces`` and
  ``unique_faces``.
* Components via ``graph.connected_components`` on the face adjacency (same
  count as ``split(only_watertight=False)``, 10x faster).
* An inside-out closed mesh has negative volume (the repo's genus-28 lattice
  fixture: -15356.87), so the census reports ``abs(volume)`` when the mesh is
  watertight and flags ``inverted``.
* Classifier (14 meshes): ``planar_big`` = area of the coplanar facets that
  are each larger than 1 % of the total area, divided by the total area. CAD
  meshes gave >= 0.20 (worst: a 128-segment cylinder), organic meshes
  (spheres, torus, smoothed blob, genus-28 lattice) 0.000; threshold 0.10.
  Raw ``facets_area.sum() / area`` is useless (1.0 for any quad strip).
  ``sharp_turn`` = share of sum(edge length x dihedral angle) carried by edges
  above 30 deg; reported, not gating (the MC lattice has 0.625).

No OCP import: trimesh + numpy only. No output to stdout (logging only).
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import List

import numpy as np

log = logging.getLogger(__name__)

PLANAR_BIG_FACET_SHARE = 0.01     # a facet counts when larger than 1 % of the total area
SHARP_TURN_DEG = 30.0


class MeshLoadError(ValueError):
    """The file cannot be read as a non-empty triangle mesh."""


@dataclass
class MeshModel:
    mesh: "object"                   # trimesh.Trimesh
    census: dict
    classifier: dict
    warnings: List[str] = field(default_factory=list)


def load_stl_checked(path: str):
    """Load an STL as a processed ``trimesh.Trimesh`` or raise ``ValueError``.

    Refuses a 0-byte file, a file holding STEP text and any file from which
    trimesh reads no triangle (garbage, truncated binary). ``process=True``
    merges vertices so the face adjacency used by the views works.
    """
    import trimesh

    try:
        size = os.path.getsize(path)
    except OSError as exc:
        raise MeshLoadError(f"{path}: cannot stat file ({exc})") from exc
    if size == 0:
        raise MeshLoadError(f"{path}: empty file")
    with open(path, "rb") as fh:
        head = fh.read(512)
    if head.lstrip().upper().startswith(b"ISO-10303-21"):
        raise MeshLoadError(f"{path}: file holds STEP text, not STL")
    try:
        mesh = trimesh.load(path, force="mesh", file_type="stl", process=True)
    except Exception as exc:  # trimesh raises assorted types on odd input
        raise MeshLoadError(f"{path}: STL could not be parsed ({type(exc).__name__}: {exc})") from exc
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise MeshLoadError(f"{path}: no triangles could be read (corrupt or truncated STL)")
    return mesh


def mesh_census(mesh) -> dict:
    """Quality-report census of a triangle mesh (JSON-ready values).

    Keys: triangles vertices watertight winding_consistent components
    volume_mm3 (abs, None when not watertight) inverted area_mm2 bbox_min
    bbox_max extents euler degenerate_faces duplicate_faces boundary_edges
    nonmanifold_edges.
    """
    import trimesh

    n_faces = int(len(mesh.faces))
    out: dict = {"triangles": n_faces, "vertices": int(len(mesh.vertices))}
    if n_faces == 0:
        out.update(watertight=False, winding_consistent=False, components=0, volume_mm3=None,
                   inverted=False, area_mm2=0.0, bbox_min=None, bbox_max=None, extents=None,
                   euler=0, degenerate_faces=0, duplicate_faces=0, boundary_edges=0,
                   nonmanifold_edges=0)
        return out
    watertight = bool(mesh.is_watertight)
    out["watertight"] = watertight
    out["winding_consistent"] = bool(mesh.is_winding_consistent)
    try:
        comps = trimesh.graph.connected_components(mesh.face_adjacency, nodes=np.arange(n_faces),
                                                   min_len=1)
        out["components"] = int(len(comps))
    except Exception as exc:  # pragma: no cover - defensive
        log.debug("connected_components failed: %s", exc)
        out["components"] = int(len(mesh.split(only_watertight=False)))
    vol = float(mesh.volume) if watertight else None
    out["volume_mm3"] = abs(vol) if vol is not None else None
    out["inverted"] = bool(vol is not None and vol < 0.0)
    out["area_mm2"] = float(mesh.area)
    bounds = np.asarray(mesh.bounds, dtype=float)
    out["bbox_min"] = [float(v) for v in bounds[0]]
    out["bbox_max"] = [float(v) for v in bounds[1]]
    out["extents"] = [float(v) for v in bounds[1] - bounds[0]]
    out["euler"] = int(mesh.euler_number)
    out["degenerate_faces"] = int((~mesh.nondegenerate_faces(height=1e-8)).sum())
    out["duplicate_faces"] = int(n_faces - int(np.asarray(mesh.unique_faces()).sum()))
    es = mesh.edges_sorted
    out["boundary_edges"] = int(len(trimesh.grouping.group_rows(es, require_count=1)))
    _, counts = np.unique(es, axis=0, return_counts=True)
    out["nonmanifold_edges"] = int((counts > 2).sum())
    return out


def classify(mesh, threshold: float = 0.10) -> dict:
    """Prismatic/organic verdict from ``planar_big`` (see module docstring).

    Returns ``{"verdict", "planar_big", "sharp_turn", "threshold"}``;
    ``verdict`` is ``"prismatic"`` when ``planar_big >= threshold``.
    """
    threshold = float(threshold)
    area = float(mesh.area) if len(mesh.faces) else 0.0
    if not area > 0.0:
        return {"verdict": "organic", "planar_big": 0.0, "sharp_turn": 0.0, "threshold": threshold}
    fa = np.asarray(mesh.facets_area, dtype=float)
    planar_big = float(fa[fa > PLANAR_BIG_FACET_SHARE * area].sum() / area) if len(fa) else 0.0
    ang = np.degrees(np.asarray(mesh.face_adjacency_angles, dtype=float))
    sharp_turn = 0.0
    if len(ang):
        v = np.asarray(mesh.vertices, dtype=float)
        e = np.asarray(mesh.face_adjacency_edges)
        el = np.linalg.norm(v[e[:, 0]] - v[e[:, 1]], axis=1)
        turn = float((el * ang).sum())
        if turn > 0.0:
            sharp_turn = float((el * ang)[ang > SHARP_TURN_DEG].sum() / turn)
    verdict = "prismatic" if planar_big >= threshold else "organic"
    return {"verdict": verdict, "planar_big": planar_big, "sharp_turn": sharp_turn,
            "threshold": threshold}


def transform_mesh(mesh, matrix):
    """Return a rotated copy: ``v' = matrix @ v`` (3x3, det +1 expected).

    The input mesh is not modified. A matrix with det -1 is applied too
    (trimesh then flips the face winding to keep normals outward) but logged.
    """
    m = np.asarray(matrix, dtype=float)
    if m.shape != (3, 3):
        raise ValueError(f"transform_mesh expects a 3x3 matrix, got shape {m.shape}")
    det = float(np.linalg.det(m))
    if abs(det - 1.0) > 1e-6:
        log.warning("transform_mesh: matrix is not a proper rotation (det %.6g)", det)
    t = np.eye(4)
    t[:3, :3] = m
    out = mesh.copy()
    out.apply_transform(t)
    return out


def load_mesh_model(path: str, threshold: float = 0.10) -> MeshModel:
    """``load_stl_checked`` + census + classifier + human-readable warnings."""
    mesh = load_stl_checked(path)
    census = mesh_census(mesh)
    classifier = classify(mesh, threshold)
    warnings: List[str] = []
    if not census["watertight"]:
        warnings.append(f"mesh is not watertight ({census['boundary_edges']} boundary edges)")
    if census["inverted"]:
        warnings.append("mesh is inside out (negative volume); the absolute volume is reported")
    if census["degenerate_faces"]:
        warnings.append(f"{census['degenerate_faces']} degenerate (zero-area) faces")
    if census["duplicate_faces"]:
        warnings.append(f"{census['duplicate_faces']} duplicate faces")
    if census["nonmanifold_edges"]:
        warnings.append(f"{census['nonmanifold_edges']} non-manifold edges")
    for w in warnings:
        log.info("mesh %s: %s", os.path.basename(path), w)
    return MeshModel(mesh=mesh, census=census, classifier=classifier, warnings=warnings)


__all__ = ["MeshLoadError", "MeshModel", "load_stl_checked", "mesh_census", "classify",
           "transform_mesh", "load_mesh_model"]
