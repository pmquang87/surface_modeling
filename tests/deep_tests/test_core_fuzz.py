import pytest
import numpy as np
import random
import sys
import psutil
import os
import gc
import weakref
from collections import defaultdict

from src.core.halfedge_mesh import HalfEdgeMesh, Vertex
from src.subd.catmull_clark import subdivide

def check_mesh_invariants(mesh, expect_faces=True, expect_twins=False):
    """Check twin/next/prev pointer integrity.

    The pointer assertions are UNCONDITIONAL on purpose: guarding them behind
    `if he.next:` turns every defect that nulls a pointer into a skipped check
    instead of a failure, and an add_face that builds nothing then satisfies the
    whole function by iterating over empty lists.
    """
    if expect_faces:
        assert mesh.faces, "mesh has no faces"
        assert mesh.half_edges, "mesh has no half-edges"
        assert mesh.edges, "mesh has no edges"

    for he in mesh.half_edges:
        assert he.next is not None, "he.next is None"
        assert he.prev is not None, "he.prev is None"
        assert he.vertex is not None, "he.vertex is None"
        assert he.face is not None, "he.face is None"
        assert he.edge is not None, "he.edge is None"

        assert he.next.prev == he, "he.next.prev != he"
        assert he.prev.next == he, "he.prev.next != he"

        if he.twin:
            assert he.twin.twin == he, "he.twin.twin != he"
            assert he.vertex != he.twin.vertex, "he.vertex == he.twin.vertex"
            assert he.edge == he.twin.edge, "he.edge != he.twin.edge"
            # the twin must run the opposite way along the same edge
            assert he.twin.vertex == he.prev.vertex, "twin does not reverse the edge"
            assert he.twin.prev.vertex == he.vertex, "twin does not reverse the edge"

        # The face's half_edge should point to one of the half_edges in the face
        assert he.face.half_edge is not None, "Face has no half_edge"
        assert he.edge.half_edge in (he, he.twin), "Edge does not point to its half_edges"

    # Twin pairing accounting: two twinned half-edges share ONE Edge record, an
    # unpaired one owns its own. A build that never pairs twins keeps this
    # identity true but drives the pair count to zero.
    n_twinned = sum(1 for he in mesh.half_edges if he.twin is not None)
    assert n_twinned % 2 == 0, "odd number of twinned half-edges"
    n_pairs = n_twinned // 2
    assert len(mesh.edges) + n_pairs == len(mesh.half_edges), \
        f"edge bookkeeping broken: {len(mesh.edges)} edges + {n_pairs} pairs " \
        f"!= {len(mesh.half_edges)} half-edges"
    if expect_twins:
        assert n_pairs > 0, "no half-edge was ever twinned -- the mesh is dust"

    # Every face cycle closes after exactly len(face) steps
    for f in mesh.faces:
        n = len(mesh.get_face_vertices(f))
        assert n >= 3, f"face with {n} vertices"
        assert f.half_edge is not None
        curr = f.half_edge
        for _ in range(n):
            assert curr.face == f, "face cycle leaves the face"
            curr = curr.next
        assert curr is f.half_edge, "face cycle does not close"

    return n_pairs


def expected_twin_pairs(recorded_faces):
    """Twin pairs implied by the face list alone.

    add_face pairs a directed edge (a, b) with the FIRST registered (b, a) and
    only while that one is still free, so an undirected pair {a, b} yields at
    most one twin pairing -- exactly when both directions occur.
    """
    directed = defaultdict(int)
    for face in recorded_faces:
        n = len(face)
        for i in range(n):
            directed[(face[i], face[(i + 1) % n])] += 1
    pairs = set()
    for (a, b) in directed:
        if directed.get((b, a), 0) > 0 and a != b:
            pairs.add((min(a, b), max(a, b)))
    return len(pairs)


def test_random_operations():
    """10,000 random valid operations (add_vertex, add_face) and ensure pointers never become corrupted."""
    mesh = HalfEdgeMesh()
    random.seed(42)
    np.random.seed(42)

    recorded_faces = []

    # start with a single quad
    v0 = mesh.add_vertex([0, 0, 0]).index
    v1 = mesh.add_vertex([1, 0, 0]).index
    v2 = mesh.add_vertex([1, 1, 0]).index
    v3 = mesh.add_vertex([0, 1, 0]).index
    mesh.add_face([v0, v1, v2, v3])
    recorded_faces.append([v0, v1, v2, v3])

    for i in range(10000):
        boundary_edges = [e for e in mesh.edges if mesh.is_boundary_edge(e)]

        if not boundary_edges or random.random() < 0.1:
            # Add a disjoint face (triangle or quad)
            v0 = mesh.add_vertex(np.random.randn(3)).index
            v1 = mesh.add_vertex(np.random.randn(3)).index
            v2 = mesh.add_vertex(np.random.randn(3)).index
            if random.random() < 0.5:
                v3 = mesh.add_vertex(np.random.randn(3)).index
                new_face = [v0, v1, v2, v3]
            else:
                new_face = [v0, v1, v2]
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
                new_face = [v_tgt, v_src, v_new]
            else:
                # Add a quad
                v_new2 = mesh.add_vertex(np.random.randn(3)).index
                new_face = [v_tgt, v_src, v_new, v_new2]

        assert mesh.add_face(new_face) is not None, f"add_face rejected {new_face}"
        recorded_faces.append(new_face)

        if i % 1000 == 0:
            check_mesh_invariants(mesh)

    n_pairs = check_mesh_invariants(mesh, expect_twins=True)

    # --- the 10,000 operations really happened -------------------------------
    assert len(mesh.faces) == len(recorded_faces) == 10001
    assert len(mesh.half_edges) == sum(len(f) for f in recorded_faces)
    assert len(mesh.vertices) > 10000

    # Twin pairing cross-checked against the recorded face list, so a build that
    # silently stops pairing twins (or pairs too eagerly) fails here.
    assert n_pairs == expected_twin_pairs(recorded_faces), \
        f"twin pairs {n_pairs} != {expected_twin_pairs(recorded_faces)} implied by the faces"
    assert n_pairs > 1000, "the extrusion branch never shared an edge"

    # Also test subdivision on this messy mesh
    # Just 1 level to avoid exploding memory
    expected_faces = sum(len(mesh.get_face_vertices(f)) for f in mesh.faces)
    expected_vertices = len(mesh.vertices) + len(mesh.edges) + len(mesh.faces)

    subdivided = subdivide(mesh, levels=1)
    check_mesh_invariants(subdivided, expect_twins=True)

    # Catmull-Clark splits every n-gon into n quads and adds one point per face
    # and per edge; an identity "subdivision" returns the input untouched.
    assert subdivided is not mesh
    assert len(subdivided.faces) == expected_faces, \
        f"subdivision produced {len(subdivided.faces)} faces, expected {expected_faces}"
    assert len(subdivided.vertices) == expected_vertices
    assert all(len(subdivided.get_face_vertices(f)) == 4 for f in subdivided.faces), \
        "Catmull-Clark output must be pure quad"

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

    assert len(mesh.faces) == 2
    before = np.array([v.position for v in mesh.vertices])
    lo, hi = before.min(axis=0), before.max(axis=0)

    # Subdivide
    subdivided = subdivide(mesh, levels=3)
    check_mesh_invariants(subdivided, expect_twins=True)

    # --- subdivision actually ran --------------------------------------------
    # 4-gon + 10-gon -> 14 quads -> 56 -> 224. An identity "subdivision" that
    # hands the input back would leave 2 faces here.
    assert subdivided is not mesh
    assert len(subdivided.faces) == 224, \
        f"3 Catmull-Clark levels must give 224 faces, got {len(subdivided.faces)}"
    assert all(len(subdivided.get_face_vertices(f)) == 4 for f in subdivided.faces)
    assert len(subdivided.vertices) == 282

    P = np.array([v.position for v in subdivided.vertices])
    # The smoothing mask really ran: the +-10000 z spike is averaged down.
    assert np.abs(P[:, 2]).max() == pytest.approx(5468.75), \
        "vertex positions were not smoothed by the Catmull-Clark mask"
    # ... and Catmull-Clark is bounded by the convex hull of the input cage.
    assert (P >= lo - 1e-9).all() and (P <= hi + 1e-9).all(), \
        "subdivided vertices left the input bounding box"

    # Ensure no NaNs were produced in the vertex positions
    for v in subdivided.vertices:
        assert not np.isnan(v.position).any(), "NaN found in subdivided vertex position"
        assert not np.isinf(v.position).any(), "Inf found in subdivided vertex position"

    # --- the NaN check needs geometry that can actually produce one ----------
    # Zero-area faces make every averaging mask divide by a vanishing quantity;
    # the check above can only fail if such input is fed in.
    degenerate = HalfEdgeMesh()
    coincident = [0.0, 0.0, 0.0]
    d0 = degenerate.add_vertex(coincident).index
    d1 = degenerate.add_vertex(coincident).index
    d2 = degenerate.add_vertex(coincident).index
    d3 = degenerate.add_vertex([1.0, 0.0, 0.0]).index
    degenerate.add_face([d0, d1, d2, d3])          # three coincident corners
    t0 = degenerate.add_vertex([5.0, 0.0, 0.0]).index
    t1 = degenerate.add_vertex([6.0, 0.0, 0.0]).index
    t2 = degenerate.add_vertex([7.0, 0.0, 0.0]).index
    degenerate.add_face([t0, t1, t2])              # collinear, zero area
    assert len(degenerate.faces) == 2

    deg_sub = subdivide(degenerate, levels=2)
    check_mesh_invariants(deg_sub, expect_twins=True)
    assert len(deg_sub.faces) == 28
    D = np.array([v.position for v in deg_sub.vertices])
    assert np.isfinite(D).all(), "degenerate input produced NaN/Inf positions"

def test_massive_operations_memory():
    """Check memory leaks or crash cases during massive operations."""
    process = psutil.Process(os.getpid())
    gc.collect()
    mem_before = process.memory_info().rss
    # Vertex/HalfEdge use __slots__ without __weakref__, so they cannot be
    # weak-referenced; count the live instances instead. The half-edge graph is
    # densely cyclic (vertex -> half_edge -> vertex), so it survives refcounting
    # and only the cycle collector can reclaim it -- exactly where a leak hides.
    baseline_vertices = sum(1 for o in gc.get_objects() if type(o) is Vertex)

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

    assert len(mesh.faces) == (grid_size - 1) ** 2 == 2401

    # Subdivide twice (2401 -> 9604 -> 38416 faces)
    subdivided = subdivide(mesh, levels=2)

    # Exact, not "> 10000": every quad splits into 4 per level.
    assert len(subdivided.faces) == 2401 * 16 == 38416
    assert all(len(subdivided.get_face_vertices(f)) == 4 for f in subdivided.faces)
    assert len(subdivided.vertices) == 38809

    live_vertices = sum(1 for o in gc.get_objects() if type(o) is Vertex)
    assert live_vertices >= baseline_vertices + len(mesh.vertices) + len(subdivided.vertices)

    mesh_ref = weakref.ref(mesh)
    subdivided_ref = weakref.ref(subdivided)

    # Delete and GC to check memory recovery
    del mesh
    del subdivided
    gc.collect()

    # Nothing may outlive the meshes: no dangling registry, no cycle the
    # collector cannot break.
    assert mesh_ref() is None, "the cage mesh outlived its last reference"
    assert subdivided_ref() is None, "the subdivided mesh outlived its last reference"
    leaked = sum(1 for o in gc.get_objects() if type(o) is Vertex) - baseline_vertices
    assert leaked <= 10, f"{leaked} Vertex objects leaked after gc.collect()"

    mem_after = process.memory_info().rss
    # RSS is allocator-dependent (Python does not always return arenas to the
    # OS), so this is only a coarse backstop -- the live-object count above is
    # the real check. Peak growth for this mesh is ~100 MB; retaining it fails.
    assert mem_after > 0
    assert mem_after - mem_before < 64 * 1024 * 1024, \
        f"resident set grew by {(mem_after - mem_before) / 1e6:.1f} MB after cleanup"
