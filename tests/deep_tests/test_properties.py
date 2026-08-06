import numpy as np
import pytest
import trimesh
import trimesh.creation as creation
from hypothesis import given, strategies as st
from hypothesis.extra.numpy import arrays

from src.core.halfedge_mesh import HalfEdgeMesh
from src.reverse_engineering.quad_wrap import QuadWrapper

# Strategy for generating random degenerate or valid meshes (vertices and triangle faces)
@st.composite
def random_trimesh_data(draw, max_vertices=50, max_faces=100):
    num_vertices = draw(st.integers(min_value=3, max_value=max_vertices))
    # Allow finite floats
    vertices = draw(arrays(
        dtype=float,
        shape=(num_vertices, 3),
        elements=st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False)
    ))
    
    num_faces = draw(st.integers(min_value=1, max_value=max_faces))
    faces = draw(arrays(
        dtype=int,
        shape=(num_faces, 3),
        elements=st.integers(min_value=0, max_value=num_vertices - 1)
    ))
    
    return vertices, faces

# Strategy for valid closed manifold meshes
@st.composite
def valid_manifold_meshes(draw):
    mesh_type = draw(st.sampled_from(['box', 'icosahedron', 'icosphere']))
    if mesh_type == 'box':
        mesh = creation.box()
    elif mesh_type == 'icosahedron':
        mesh = creation.icosahedron()
    elif mesh_type == 'icosphere':
        subdivisions = draw(st.integers(min_value=1, max_value=3))
        mesh = creation.icosphere(subdivisions=subdivisions)
    return mesh

@given(random_trimesh_data())
def test_halfedge_mesh_random_topology_no_crash(mesh_data):
    vertices, faces = mesh_data
    
    # Check that from_arrays does not crash with degenerate meshes (e.g. out of bounds, non-manifold)
    mesh = HalfEdgeMesh.from_arrays(vertices, faces.tolist())
    assert isinstance(mesh, HalfEdgeMesh)
    
    # Check that basic functions don't crash
    if len(mesh.faces) > 0:
        mesh.compute_face_normals()
        mesh.compute_vertex_normals()
        
    mesh_copy = mesh.copy()
    assert len(mesh_copy.vertices) == len(mesh.vertices)
    assert len(mesh_copy.faces) == len(mesh.faces)

@given(random_trimesh_data())
def test_quad_wrapper_random_topology_no_crash(mesh_data):
    vertices, faces = mesh_data
    mesh = HalfEdgeMesh.from_arrays(vertices, faces.tolist())
    
    wrapper = QuadWrapper(target_face_count=10)
    # The wrap function shouldn't crash (it has internal try/except, but we test the whole pipeline)
    result = wrapper.wrap(mesh)
    assert isinstance(result, HalfEdgeMesh)

@given(valid_manifold_meshes())
def test_topological_invariants_manifold(trimesh_obj):
    he_mesh = HalfEdgeMesh.from_trimesh(trimesh_obj)
    
    V = len(he_mesh.vertices)
    E = len(he_mesh.edges)
    F = len(he_mesh.faces)
    
    # For a closed sphere-like manifold (genus 0), Euler characteristic V - E + F == 2
    assert V - E + F == 2

    # Check that every edge has exactly two adjacent faces
    for edge in he_mesh.edges:
        f1, f2 = he_mesh.get_edge_faces(edge)
        assert f1 is not None
        assert f2 is not None

    # Check that boundary detection returns False for all edges and vertices
    for vertex in he_mesh.vertices:
        assert not he_mesh.is_boundary_vertex(vertex)
    for edge in he_mesh.edges:
        assert not he_mesh.is_boundary_edge(edge)

@given(valid_manifold_meshes())
def test_halfedge_mesh_pointers_manifold(trimesh_obj):
    he_mesh = HalfEdgeMesh.from_trimesh(trimesh_obj)
    
    # Check halfedge next/prev circularity
    for he in he_mesh.half_edges:
        assert he.next is not None
        assert he.prev is not None
        assert he.next.prev == he
        assert he.prev.next == he
        
        # Check twin consistency
        if he.twin is not None:
            assert he.twin.twin == he
            assert he.edge == he.twin.edge
        else:
            assert he.edge is not None
            assert he.edge.half_edge == he

        # Check vertex, face, edge back-pointers
        assert he.vertex is not None
        if he.face is not None:
            # he belongs to face, its next's vertex is the starting vertex of the next edge
            pass
        
    for v in he_mesh.vertices:
        if v.half_edge is not None:
            # he.vertex stores the target vertex. 
            # The source vertex is he.prev.vertex
            assert v.half_edge.prev.vertex == v
            
    for f in he_mesh.faces:
        if f.half_edge is not None:
            assert f.half_edge.face == f
            
    for e in he_mesh.edges:
        assert e.half_edge is not None
        assert e.half_edge.edge == e
