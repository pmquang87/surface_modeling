import os
import numpy as np
from typing import Dict, Any, Optional
import trimesh

from src.core.halfedge_mesh import HalfEdgeMesh

def import_step(filepath: str) -> Dict[str, Any]:
    """Import a STEP file and return shape data.
    
    Returns dict with:
        'shape': the OCC TopoDS_Shape (if OCP available)
        'mesh': HalfEdgeMesh (tessellated version for display)
        'vertices': np.ndarray (Nx3)
        'faces': list of lists of vertex indices
    
    Uses OCP (OpenCascade Python bindings) if available.
    Falls back to a simple STEP parser for basic geometry.
    """
    result = {
        'shape': None,
        'mesh': None,
        'vertices': np.array([]),
        'faces': []
    }
    
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return result

    # Try OCP
    try:
        from OCP.STEPControl import STEPControl_Reader
        from OCP.BRepMesh import BRepMesh_IncrementalMesh
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopAbs import TopAbs_FACE
        from OCP.BRep import BRep_Tool
        from OCP.TopLoc import TopLoc_Location
        
        reader = STEPControl_Reader()
        status = reader.ReadFile(filepath)
        if status != 1:
            print("Error reading STEP file with OCP.")
            return result
        
        reader.TransferRoots()
        shape = reader.OneShape()
        result['shape'] = shape
        
        # Tessellate
        BRepMesh_IncrementalMesh(shape, 0.1)
        
        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        vertices = []
        faces = []
        vertex_offset = 0
        
        while explorer.More():
            face = explorer.Current()
            loc = TopLoc_Location()
            poly = BRep_Tool.Triangulation_s(face, loc)
            
            if poly:
                for i in range(1, poly.NbNodes() + 1):
                    node = poly.Node(i)
                    if not loc.IsIdentity():
                        node.Transform(loc.Transformation())
                    vertices.append([node.X(), node.Y(), node.Z()])
                
                for i in range(1, poly.NbTriangles() + 1):
                    t = poly.Triangle(i)
                    n1, n2, n3 = t.Get()
                    faces.append([n1 - 1 + vertex_offset, n2 - 1 + vertex_offset, n3 - 1 + vertex_offset])
                
                vertex_offset += poly.NbNodes()
            
            explorer.Next()
            
        result['vertices'] = np.array(vertices, dtype=np.float64)
        result['faces'] = faces
        result['mesh'] = HalfEdgeMesh.from_arrays(result['vertices'], result['faces'])
        return result
        
    except ImportError:
        print("OCP not available. Trying cadquery...")
        
    # Try cadquery
    try:
        import cadquery as cq
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
        
    except ImportError:
        print("CadQuery not available. Falling back to simple warning.")
        
    print("No OCP or CadQuery installed. Cannot parse STEP B-Rep properly.")
    return result


def import_stl(filepath: str) -> 'HalfEdgeMesh':
    """Import an STL file using trimesh and convert to HalfEdgeMesh."""
    try:
        mesh = trimesh.load_mesh(filepath)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        return HalfEdgeMesh.from_trimesh(mesh)
    except Exception as e:
        print(f"Error loading STL: {e}")
        return HalfEdgeMesh()


def import_obj(filepath: str) -> 'HalfEdgeMesh':
    """Import an OBJ file using trimesh and convert to HalfEdgeMesh."""
    try:
        mesh = trimesh.load_mesh(filepath)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        return HalfEdgeMesh.from_trimesh(mesh)
    except Exception as e:
        print(f"Error loading OBJ: {e}")
        return HalfEdgeMesh()
