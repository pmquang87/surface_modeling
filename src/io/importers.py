import os
import numpy as np
from typing import Dict, Any, Optional
import trimesh

import logging
from src.core.halfedge_mesh import HalfEdgeMesh

logger = logging.getLogger(__name__)

def import_step(filepath: str) -> Dict[str, Any]:
    """Import a STEP file and return shape data.
    
    Returns dict with:
        'shape': the OCC TopoDS_Shape (if OCP available)
        'mesh': HalfEdgeMesh (tessellated version for display)
        'vertices': np.ndarray (Nx3)
        'faces': list of lists of vertex indices
    
    Uses OCP (OpenCascade Python bindings) if available, else cadquery.

    Raises:
        FileNotFoundError: the path does not exist.
        ValueError: the file exists but OpenCascade cannot read it or it
            carries no shape (e.g. an empty DATA section). An unreadable STEP
            used to come back as an all-empty result dict, which every caller
            had to recognise by hand.
        RuntimeError: neither OCP nor cadquery is importable, so there is no
            STEP support at all.
    """
    result = {
        'shape': None,
        'mesh': None,
        'vertices': np.array([]),
        'faces': []
    }
    
    if not os.path.exists(filepath):
        logger.error(f"File not found: {filepath}")
        raise FileNotFoundError(f"File not found: {filepath}")

    # Try OCP (support both 'OCP.Module' and cadquery-style 'OCP.OCP.Module')
    try:
        try:
            from OCP.STEPControl import STEPControl_Reader
            from OCP.BRepMesh import BRepMesh_IncrementalMesh
            from OCP.TopExp import TopExp_Explorer
            from OCP.TopAbs import TopAbs_FACE
            from OCP.BRep import BRep_Tool
            from OCP.TopLoc import TopLoc_Location
            from OCP.TopoDS import TopoDS
        except ImportError:
            # cadquery-ocp nests under OCP.OCP.*
            from OCP.OCP.STEPControl import STEPControl_Reader
            from OCP.OCP.BRepMesh import BRepMesh_IncrementalMesh
            from OCP.OCP.TopExp import TopExp_Explorer
            from OCP.OCP.TopAbs import TopAbs_FACE
            from OCP.OCP.BRep import BRep_Tool
            from OCP.OCP.TopLoc import TopLoc_Location
            from OCP.OCP.TopoDS import TopoDS
        
        reader = STEPControl_Reader()
        status = reader.ReadFile(filepath)
        if status != 1:
            logger.error(f"Error reading STEP file with OCP (status {status}).")
            raise ValueError(
                f"Cannot read STEP file '{filepath}': OpenCascade "
                f"STEPControl_Reader returned status {status} (1 = OK).")

        transfer_status = reader.TransferRoots()
        if not transfer_status:
            logger.error("Failed to transfer roots in STEP file.")

        shape = reader.OneShape()
        if shape.IsNull():
            logger.error("STEP file contains a null shape.")
            raise ValueError(
                f"STEP file '{filepath}' contains no geometry: OpenCascade "
                f"read it (status 1) but transferred a null shape "
                f"(TransferRoots -> {transfer_status}).")


        result['shape'] = shape
        
        # Tessellate
        mesh_algo = BRepMesh_IncrementalMesh(shape, 0.1)
        if not mesh_algo.IsDone():
            logger.warning("BRepMesh_IncrementalMesh failed to generate a mesh.")
        
        from OCP.TopAbs import TopAbs_REVERSED

        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        vertices = []
        faces = []
        vertex_offset = 0

        while explorer.More():
            # Downcast from TopoDS_Shape to TopoDS_Face
            face = TopoDS.Face_s(explorer.Current())
            loc = TopLoc_Location()
            poly = BRep_Tool.Triangulation_s(face, loc)

            if poly:
                for i in range(1, poly.NbNodes() + 1):
                    node = poly.Node(i)
                    if not loc.IsIdentity():
                        node.Transform(loc.Transformation())
                    vertices.append([node.X(), node.Y(), node.Z()])

                reversed_face = face.Orientation() == TopAbs_REVERSED
                for i in range(1, poly.NbTriangles() + 1):
                    t = poly.Triangle(i)
                    n1, n2, n3 = t.Get()
                    if reversed_face:
                        # keep outward-facing winding for reversed faces
                        n2, n3 = n3, n2
                    faces.append([n1 - 1 + vertex_offset, n2 - 1 + vertex_offset, n3 - 1 + vertex_offset])

                vertex_offset += poly.NbNodes()

            explorer.Next()

        result['vertices'] = np.array(vertices, dtype=np.float64) if vertices else np.array([])
        result['faces'] = faces
        if len(vertices) > 0:
            # Per-face triangulations duplicate every boundary node; weld them
            # so the half-edge mesh is connected instead of a face soup.
            import trimesh as _trimesh
            welded = _trimesh.Trimesh(vertices=result['vertices'], faces=result['faces'], process=False)
            welded.merge_vertices()
            result['vertices'] = np.asarray(welded.vertices, dtype=np.float64)
            result['faces'] = welded.faces.tolist()
            result['mesh'] = HalfEdgeMesh.from_arrays(result['vertices'], result['faces'])
        logger.info(f"STEP loaded via OCP: {len(result['vertices'])} vertices, {len(result['faces'])} faces")
        return result
        
    except ImportError:
        logger.info("OCP not available. Trying cadquery...")
        
    # Try cadquery
    try:
        import cadquery as cq
    except ImportError:
        logger.warning("CadQuery not available. Falling back to simple warning.")
        logger.error("No OCP or CadQuery installed. Cannot parse STEP B-Rep properly.")
        raise RuntimeError(
            "STEP support is unavailable: neither OCP nor cadquery could be "
            "imported. Install one of them (pip install cadquery-ocp) or "
            "convert the file to STL/OBJ first.")

    try:
        shape = cq.importers.importStep(filepath)
        result['shape'] = shape

        # Simple tessellation using CQ
        # CQ 2.x uses OCP under the hood, so if OCP failed, this might also fail.
        # But just in case:
        tess = shape.val().tessellate(0.1)
        vertices = [[v.x, v.y, v.z] for v in tess[0]]
        # tess[1] contains triangle indices
        faces = [[i, j, k] for i, j, k in tess[1]]
        
        result['vertices'] = np.array(vertices, dtype=np.float64)
        result['faces'] = faces
        result['mesh'] = HalfEdgeMesh.from_arrays(result['vertices'], result['faces'])
        return result

    except Exception as e:
        logger.error(f"CadQuery could not read the STEP file: {e}")
        raise ValueError(
            f"Cannot read STEP file '{filepath}' with cadquery: {e}") from e


def import_stl(filepath: str) -> 'HalfEdgeMesh':
    """Import an STL file using trimesh and convert to HalfEdgeMesh."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    try:
        mesh = trimesh.load_mesh(filepath)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        return HalfEdgeMesh.from_trimesh(mesh)
    except Exception as e:
        logger.error(f"Error loading STL: {e}")
        raise ValueError(f"Error loading STL '{filepath}': {e}") from e


def import_obj(filepath: str) -> 'HalfEdgeMesh':
    """Import an OBJ file using trimesh and convert to HalfEdgeMesh."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    try:
        mesh = trimesh.load_mesh(filepath)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        return HalfEdgeMesh.from_trimesh(mesh)
    except Exception as e:
        logger.error(f"Error loading OBJ: {e}")
        raise ValueError(f"Error loading OBJ '{filepath}': {e}") from e
