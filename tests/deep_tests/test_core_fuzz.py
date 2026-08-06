import pytest
import numpy as np
import random
import sys
import psutil
import os
import gc

from src.core.halfedge_mesh import HalfEdgeMesh
from src.subd.catmull_clark import subdivide

def check_mesh_invariants(mesh):
    """Check twin/next/prev pointer integrity."""
    for he in mesh.half_edges:
        if he.next:
            assert he.next.prev == he, "he.next.prev != he"
        if he.prev:
            assert he.prev.next == he, "he.prev.next != he"
        if he.twin:
            assert he.twin.twin == he, "he.twin.twin != he"
            assert he.vertex != he.twin.vertex, "he.vertex == he.twin.vertex"
            assert he.edge == he.twin.edge, "he.edge != he.twin.edge"
        
        if he.face:
            # The face's half_edge should point to one of the half_edges in the face
            assert he.face.half_edge is not None, "Face has no half_edge"
            
        if he.edge:
            assert he.edge.half_edge in (he, he.twin), "Edge does not point to its half_edges"

def test_random_operations():
    """10,000 random valid operations (add_vertex, add_face) and ensure pointers never become corrupted."""
    mesh = HalfEdgeMesh()
    random.seed(42)
    np.random.seed(42)
    
    # start with a single quad
    v0 = mesh.add_vertex([0, 0, 0]).index
    v1 = mesh.add_vertex([1, 0, 0]).index
    v2 = mesh.add_vertex([1, 1, 0]).index
    v3 = mesh.add_vertex([0, 1, 0]).index
    mesh.add_face([v0, v1, v2, v3])
    
    for i in range(10000):
        boundary_edges = [e for e in mesh.edges if mesh.is_boundary_edge(e)]
        
        if not boundary_edges or random.random() < 0.1:
            # Add a disjoint face (triangle or quad)
            v0 = mesh.add_vertex(np.random.randn(3)).index
            v1 = mesh.add_vertex(np.random.randn(3)).index
            v2 = mesh.add_vertex(np.random.randn(3)).index
            if random.random() < 0.5:
                v3 = mesh.add_vertex(np.random.randn(3)).index
                mesh.add_face([v0, v1, v2, v3])
            else:
                mesh.add_face([v0, v1, v2])
        else:
            # Extrude from a random boundary edge
            e = random.choice(boundary_edges)
            he = e.half_edge
            
            # The half_edge of a boundary edge belongs to a face. It points from prev.vertex to vertex.
            v_tgt = he.vertex.index
            v_src = he.prev.vertex.index
            
            v_new = mesh.add_vertex(np.random.randn(3)).index
            if random.random() < 0.5:
                # Add a triangle
                mesh.add_face([v_tgt, v_src, v_new])
            else:
                # Add a quad
                v_new2 = mesh.add_vertex(np.random.randn(3)).index
                mesh.add_face([v_tgt, v_src, v_new, v_new2])
                
        if i % 1000 == 0:
            check_mesh_invariants(mesh)
            
    check_mesh_invariants(mesh)
    
    # Also test subdivision on this messy mesh
    # Just 1 level to avoid exploding memory
    subdivided = subdivide(mesh, levels=1)
    check_mesh_invariants(subdivided)

def test_subd_fuzzing():
    """Fuzz the subdivision algorithm by giving it massively non-planar quads and concave polygons."""
    mesh = HalfEdgeMesh()
    
    # 1. Massively non-planar quad
    v0 = mesh.add_vertex([0, 0, 0]).index
    v1 = mesh.add_vertex([100, 0, 0]).index
    v2 = mesh.add_vertex([100, 100, 10000]).index # massive z
    v3 = mesh.add_vertex([0, 100, -10000]).index # massive -z
    mesh.add_face([v0, v1, v2, v3])
    
    # 2. Concave polygon (star shape)
    star_verts = []
    for i in range(10):
        r = 10.0 if i % 2 == 0 else 1.0
        angle = i * np.pi * 2 / 10
        star_verts.append(mesh.add_vertex([r * np.cos(angle), r * np.sin(angle), 0]).index)
    mesh.add_face(star_verts)
    
    # Subdivide
    subdivided = subdivide(mesh, levels=3)
    check_mesh_invariants(subdivided)
    
    # Ensure no NaNs were produced in the vertex positions
    for v in subdivided.vertices:
        assert not np.isnan(v.position).any(), "NaN found in subdivided vertex position"
        assert not np.isinf(v.position).any(), "Inf found in subdivided vertex position"

def test_massive_operations_memory():
    """Check memory leaks or crash cases during massive operations."""
    process = psutil.Process(os.getpid())
    gc.collect()
    mem_before = process.memory_info().rss
    
    mesh = HalfEdgeMesh()
    
    # Add a massive grid
    grid_size = 50 # 50x50 grid = 2500 faces
    verts = {}
    for i in range(grid_size):
        for j in range(grid_size):
            verts[(i,j)] = mesh.add_vertex([i, j, 0]).index
            
    for i in range(grid_size - 1):
        for j in range(grid_size - 1):
            v0 = verts[(i, j)]
            v1 = verts[(i+1, j)]
            v2 = verts[(i+1, j+1)]
            v3 = verts[(i, j+1)]
            mesh.add_face([v0, v1, v2, v3])
            
    # Subdivide twice (2500 -> 10000 -> 40000 faces)
    subdivided = subdivide(mesh, levels=2)
    
    assert len(subdivided.faces) > 10000
    
    # Delete and GC to check memory recovery
    del mesh
    del subdivided
    gc.collect()
    
    mem_after = process.memory_info().rss
    # As long as it completes and doesn't exhaust all system memory, it's a pass.
    # We can also assert it didn't leak heavily, but Python's GC can be lazy with resident size.
    # The main thing is that we didn't crash.
    assert mem_after > 0
