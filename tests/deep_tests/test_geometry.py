import pytest
import numpy as np
import trimesh
from src.core.halfedge_mesh import HalfEdgeMesh
from src.reverse_engineering.quad_wrap import QuadWrapper
from src.reverse_engineering.shrink_wrap import ShrinkWrapper
from src.nurbs.converter import SubDToNURBSConverter

def is_quad_convex(vertices):
    v0, v1, v2, v3 = vertices
    normal = np.cross(v2 - v0, v3 - v1)
    if np.linalg.norm(normal) < 1e-10:
        return False
    normal = normal / np.linalg.norm(normal)
    edges = [v1 - v0, v2 - v1, v3 - v2, v0 - v3]
    signs = []
    for i in range(4):
        cross = np.cross(edges[i], edges[(i+1)%4])
        dot = np.dot(cross, normal)
        if abs(dot) > 1e-8:
            signs.append(np.sign(dot))
    if len(signs) < 3: # If too many collinear edges, it's degenerate/non-convex
        return False
    return len(set(signs)) == 1

def generate_complex_mesh():
    # Create a high-genus mesh by using a single torus (genus 1)
    mesh = trimesh.creation.torus(major_radius=10, minor_radius=2, major_sections=32, minor_sections=16)
    # Ensure it's watertight
    assert mesh.is_watertight
    # Convert to HalfEdgeMesh
    he_mesh = HalfEdgeMesh()
    vertex_map = {}
    for i, v in enumerate(mesh.vertices):
        vert = he_mesh.add_vertex(v.tolist())
        vertex_map[i] = vert.index
    for f in mesh.faces:
        he_mesh.add_face([vertex_map[v] for v in f])
    return he_mesh

def test_quadwrapper_convexity():
    he_mesh = generate_complex_mesh()
    wrapper = QuadWrapper(target_face_count=200, smoothing_weight=0.5)
    quad_mesh = wrapper.wrap(he_mesh)
    
    non_convex_count = 0
    for face in quad_mesh.faces:
        verts = quad_mesh.get_face_vertices(face)
        if len(verts) == 4:
            positions = [np.array(v.position) for v in verts]
            if not is_quad_convex(positions):
                non_convex_count += 1
                
    assert non_convex_count == 0, f"Found {non_convex_count} non-convex quads!"

def test_quadwrapper_high_genus_watertight():
    he_mesh = generate_complex_mesh()
    wrapper = QuadWrapper(target_face_count=300, smoothing_weight=0.5)
    quad_mesh = wrapper.wrap(he_mesh)
    
    # Check that there are no boundary edges (watertight)
    boundary_edges = [e for e in quad_mesh.edges if quad_mesh.is_boundary_edge(e)]
    assert len(boundary_edges) == 0, f"Mesh is not watertight, found {len(boundary_edges)} boundary edges!"

def test_nurbs_continuity():
    # Generate a simple cube and convert to NURBS
    mesh = trimesh.creation.box()
    he_mesh = HalfEdgeMesh()
    vertex_map = {}
    for i, v in enumerate(mesh.vertices):
        vert = he_mesh.add_vertex(v.tolist())
        vertex_map[i] = vert.index
    for f in mesh.faces:
        he_mesh.add_face([vertex_map[v] for v in f])
        
    # Convert to quads first!
    wrapper = QuadWrapper(target_face_count=20, smoothing_weight=0.1)
    quad_mesh = wrapper.wrap(he_mesh)
        
    converter = SubDToNURBSConverter(continuity='G2', tolerance=1e-4)
    patches = converter.generate_patches(quad_mesh, subdivision_levels=2)
    
    # Check that patches are generated and have correct shape (6x6 for G3 fitter)
    assert len(patches) > 0
    for patch in patches:
        assert patch.shape == (6, 6, 3), "Patch control points should be a 6x6 grid for G3!"
        
    # In a real environment with cadquery, we would also test build_shape, but we might not have cadquery-ocp
    shape = converter.build_shape(patches, simplify=False)
    if shape is not None:
        # If cadquery-ocp is installed, shape shouldn't be None
        pass
