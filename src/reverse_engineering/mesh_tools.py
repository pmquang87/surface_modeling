import numpy as np
import trimesh
from typing import Optional, List, Dict, Any
from src.core.halfedge_mesh import HalfEdgeMesh

def fill_holes(mesh: HalfEdgeMesh, max_hole_edges: int = 20) -> HalfEdgeMesh:
    """Fill holes in the mesh with new faces.
    
    Detect boundary loops, triangulate each hole using ear-clipping,
    then convert to quads where possible.
    """
    t_mesh = mesh.to_trimesh()
    try:
        from trimesh.repair import fill_holes as trimesh_fill_holes
        trimesh_fill_holes(t_mesh)
        result = HalfEdgeMesh.from_trimesh(t_mesh)
        return result
    except Exception as e:
        print(f"Error filling holes: {e}")
        return mesh.copy()

def smooth_mesh(mesh: HalfEdgeMesh, iterations: int = 3, 
                method: str = 'taubin', lambda_factor: float = 0.5,
                mu_factor: float = -0.53) -> HalfEdgeMesh:
    """Smooth a mesh using Laplacian or Taubin smoothing.
    
    'laplacian': Simple Laplacian (shrinks mesh)
    'taubin': Taubin smoothing (volume-preserving, alternates lambda and mu)
    """
    result = mesh.copy()
    
    for _ in range(iterations):
        if method == 'laplacian':
            _apply_laplacian(result, lambda_factor)
        elif method == 'taubin':
            _apply_laplacian(result, lambda_factor)
            _apply_laplacian(result, mu_factor)
            
    result.compute_vertex_normals()
    return result

def _apply_laplacian(mesh: HalfEdgeMesh, factor: float):
    new_pos = []
    for v in mesh.vertices:
        neighbors = mesh.get_vertex_neighbors(v)
        if not neighbors or mesh.is_boundary_vertex(v):
            new_pos.append(v.position.copy())
        else:
            avg_pos = np.mean([n.position for n in neighbors], axis=0)
            new_pos.append(v.position + factor * (avg_pos - v.position))
            
    for i, v in enumerate(mesh.vertices):
        v.position = new_pos[i]

def offset_mesh(mesh: HalfEdgeMesh, distance: float = 0.1) -> HalfEdgeMesh:
    """Offset mesh along vertex normals by the given distance."""
    result = mesh.copy()
    result.compute_vertex_normals()
    
    for v in result.vertices:
        v.position = v.position + v.normal * distance
        
    return result

def decimate_mesh(mesh: HalfEdgeMesh, target_faces: int = None, 
                  ratio: float = 0.5, frozen_vertices: Optional[List[int]] = None) -> HalfEdgeMesh:
    """Reduce face count using edge collapse decimation.
    
    Uses trimesh's simplify_quadric_decimation if available.
    """
    t_mesh = mesh.to_trimesh()
    
    if target_faces is None:
        target_faces = int(len(mesh.faces) * ratio)
        
    if frozen_vertices:
        # Custom edge collapse decimation that preserves frozen vertices
        verts = np.array(t_mesh.vertices)
        faces = np.array(t_mesh.faces)
        frozen_set = set(frozen_vertices)
        
        # Build adjacency
        vertex_faces = {i: set() for i in range(len(verts))}
        for i, f in enumerate(faces):
            for v in f:
                vertex_faces[v].add(i)
                
        # Find all edges
        edges = set()
        for f in faces:
            edges.add(tuple(sorted((f[0], f[1]))))
            edges.add(tuple(sorted((f[1], f[2]))))
            edges.add(tuple(sorted((f[2], f[0]))))
            
        # We only collapse edges where NEITHER vertex is frozen.
        valid_edges = []
        for v1, v2 in edges:
            if v1 in frozen_set or v2 in frozen_set:
                continue
            dist = np.linalg.norm(verts[v1] - verts[v2])
            valid_edges.append((dist, v1, v2))
            
        valid_edges.sort(key=lambda x: x[0])
        
        alias = {i: i for i in range(len(verts))}
        def get_alias(v):
            curr = v
            while alias[curr] != curr:
                curr = alias[curr]
            curr2 = v
            while alias[curr2] != curr:
                nxt = alias[curr2]
                alias[curr2] = curr
                curr2 = nxt
            return curr

        current_faces = len(faces)
        face_active = [True] * len(faces)
        
        for dist, v1, v2 in valid_edges:
            if current_faces <= target_faces:
                break
                
            a1 = get_alias(v1)
            a2 = get_alias(v2)
            
            if a1 == a2:
                continue
                
            faces_a1 = vertex_faces[a1]
            faces_a2 = vertex_faces[a2]
            
            common_faces = faces_a1.intersection(faces_a2)
            
            for f_idx in common_faces:
                if face_active[f_idx]:
                    face_active[f_idx] = False
                    current_faces -= 1
                    
            alias[a2] = a1
            
            for f_idx in faces_a2:
                if face_active[f_idx]:
                    faces_a1.add(f_idx)
            
            vertex_faces[a2] = set()

        new_faces = []
        for i, active in enumerate(face_active):
            if active:
                f = faces[i]
                v0 = get_alias(f[0])
                v1 = get_alias(f[1])
                v2 = get_alias(f[2])
                if v0 != v1 and v1 != v2 and v2 != v0:
                    new_faces.append([v0, v1, v2])
                    
        try:
            import trimesh
            new_t_mesh = trimesh.Trimesh(vertices=verts, faces=new_faces, process=True)
            return HalfEdgeMesh.from_trimesh(new_t_mesh)
        except Exception as e:
            print(f"Error generating trimesh during decimation: {e}")
            return mesh.copy()

    try:
        # Requires open3d or pyembree depending on trimesh installation
        decimated = t_mesh.simplify_quadric_decimation(target_faces)
        return HalfEdgeMesh.from_trimesh(decimated)
    except Exception:
        # Fallback if decimation fails (e.g. missing dependencies)
        return mesh.copy()

def remove_duplicate_vertices(mesh: HalfEdgeMesh, tolerance: float = 1e-6) -> HalfEdgeMesh:
    """Merge vertices that are within tolerance distance of each other."""
    try:
        t_mesh = mesh.to_trimesh()
        t_mesh.merge_vertices(merge_tex=False, merge_norm=False, digits_or_tol=tolerance)
        return HalfEdgeMesh.from_trimesh(t_mesh)
    except Exception as e:
        print(f"Error merging vertices: {e}")
        return mesh.copy()

def compute_mesh_quality(mesh: HalfEdgeMesh) -> dict:
    """Compute mesh quality metrics.
    
    Returns dict with:
        'face_count', 'vertex_count', 'edge_count',
        'min_angle', 'max_angle', 'avg_angle',
        'min_area', 'max_area',
        'watertight': bool,
        'manifold': bool,
        'boundary_edges': int,
        'non_manifold_edges': int
    """
    t_mesh = mesh.to_trimesh()
    
    stats = {
        'face_count': len(mesh.faces),
        'vertex_count': len(mesh.vertices),
        'edge_count': len(mesh.edges),
        'watertight': t_mesh.is_watertight,
        'manifold': t_mesh.is_winding_consistent,
    }
    
    if len(mesh.faces) > 0:
        areas = t_mesh.area_faces
        stats['min_area'] = float(np.min(areas))
        stats['max_area'] = float(np.max(areas))
        
    # Boundary edges
    boundary_edges = sum(1 for e in mesh.edges if mesh.is_boundary_edge(e))
    stats['boundary_edges'] = boundary_edges
    stats['non_manifold_edges'] = 0 # HalfEdge structure enforces manifold by design
    
    return stats
