"""Regression tests for the core-mesh entries of the bug report written by the
automated bug hunt on commit 4e2241347de43175401df20c6722f38fa70820b4.

Covers:
  H4  (cand 6)          boundary vertices lose the neighbour reached only
                        through their incoming boundary half-edge
  M12 (cand 13)         identify_regular_regions calls a boundary vertex with
                        3 quads (valence 4) regular - a consequence of H4
  M15 (cand 7, cand 8)  small but valid triangles get non-unit / zero normals
  M16 (cand 9)          MeshSOA twin links are not symmetric on non-manifold edges
  L2a (cand 10)         add_face with a bad index leaves a partial face behind
  L2b (cand 31)         MeshSOA built from an empty HalfEdgeMesh cannot compute normals
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.halfedge_mesh import HalfEdgeMesh
from src.core.mesh_soa import MeshSOA
from src.subd.catmull_clark import identify_regular_regions
from src.subd.primitives import create_box


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def quad_grid(nx, ny):
    """(nx x ny) CCW quads in the XY plane."""
    mesh = HalfEdgeMesh()
    for j in range(ny + 1):
        for i in range(nx + 1):
            mesh.add_vertex([float(i), float(j), 0.0])

    def idx(i, j):
        return j * (nx + 1) + i

    for j in range(ny):
        for i in range(nx):
            mesh.add_face([idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)])
    return mesh


def undirected_edges(mesh):
    """Every undirected edge, straight from the face loops (independent of the
    half-edge walk under test)."""
    edges = set()
    for f in mesh.faces:
        fv = [v.index for v in mesh.get_face_vertices(f)]
        for a, b in zip(fv, fv[1:] + fv[:1]):
            edges.add((min(a, b), max(a, b)))
    return edges


def true_neighbours(mesh, v_idx):
    return {b if a == v_idx else a for a, b in undirected_edges(mesh) if v_idx in (a, b)}


# ---------------------------------------------------------------------------
# H4 - boundary neighbours
# ---------------------------------------------------------------------------

def test_h4_single_triangle_every_vertex_sees_both_others():
    """H4 (cand 6): in one triangle every vertex is a boundary vertex with
    exactly two neighbours; vertex 0 must give {1, 2}, valence 2."""
    mesh = HalfEdgeMesh.from_arrays(np.eye(3), [[0, 1, 2]])
    for v in mesh.vertices:
        assert mesh.is_boundary_vertex(v)
        assert len(mesh.get_vertex_fan(v)) == 1     # precondition: one spoke only
        got = [n.index for n in mesh.get_vertex_neighbors(v)]
        assert sorted(got) == sorted({0, 1, 2} - {v.index}), f"vertex {v.index}: {got}"
        assert mesh.vertex_valence(v) == 2
    assert sorted(mesh.get_adjacent_vertices([0])) == [1, 2]


def test_h4_open_fan_keeps_ring_order_and_prepends_missing_neighbour():
    """H4 (cand 6): the boundary fan 0->3, 0->2, 0->1 must report the neighbour
    4 (reached only through the boundary half-edge 4->0) at the head of the
    list, so that consecutive neighbours always share a face with the centre."""
    mesh = HalfEdgeMesh()
    for p in ([0, 0, 0], [1, 0, 0], [0.5, 1, 0], [-0.5, 1, 0], [-1, 0, 0]):
        mesh.add_vertex(p)
    mesh.add_face([0, 2, 3])
    mesh.add_face([0, 1, 2])
    mesh.add_face([0, 3, 4])
    v0 = mesh.vertices[0]
    assert [he.vertex.index for he in mesh.get_vertex_fan(v0)] == [3, 2, 1]

    got = [n.index for n in mesh.get_vertex_neighbors(v0)]
    assert got == [4, 3, 2, 1]
    assert mesh.vertex_valence(v0) == 4
    face_sets = [{v.index for v in mesh.get_face_vertices(f)} for f in mesh.faces]
    for a, b in zip(got, got[1:]):
        assert {0, a, b} in face_sets, f"neighbours {a},{b} are not ring-adjacent"
    # the two ends are the boundary neighbours: their edges have one face only
    for end in (got[0], got[-1]):
        assert sum(1 for s in face_sets if {0, end} <= s) == 1
    assert sum(1 for s in face_sets if {0, got[1]} <= s) == 2   # the checker can say "no"


def test_h4_valence_matches_edge_count_everywhere_on_open_and_closed_meshes():
    """H4 (cand 6): valence == number of incident edges for every vertex, i.e.
    corners 2, boundary sides 3, interior 4 on a grid; 3 on a closed box. The
    handshake lemma (sum of valences == 2E) brackets the total."""
    grid = quad_grid(3, 3)
    valences = {v.index: grid.vertex_valence(v) for v in grid.vertices}
    for v in grid.vertices:
        assert {n.index for n in grid.get_vertex_neighbors(v)} == true_neighbours(grid, v.index)
    corners = [0, 3, 12, 15]
    interior = [5, 6, 9, 10]
    sides = [i for i in range(16) if i not in corners + interior]
    assert [valences[i] for i in corners] == [2] * 4
    assert [valences[i] for i in sides] == [3] * 8
    assert [valences[i] for i in interior] == [4] * 4
    assert sum(valences.values()) == 2 * len(undirected_edges(grid)) == 48

    box = create_box()
    assert not any(box.is_boundary_vertex(v) for v in box.vertices)
    assert [box.vertex_valence(v) for v in box.vertices] == [3] * 8


def test_h4_connected_and_isolated_vertices():
    """H4 (cand 6): get_adjacent_vertices of a grid corner is its two edge
    neighbours; an isolated vertex still has no neighbours."""
    grid = quad_grid(1, 1)                    # vertices 0 1 / 2 3
    for v in range(4):
        assert sorted(grid.get_adjacent_vertices([v])) == sorted(true_neighbours(grid, v))
        assert len(grid.get_adjacent_vertices([v])) == 2
    lonely = grid.add_vertex([5.0, 5.0, 0.0])
    assert grid.get_vertex_neighbors(lonely) == []
    assert grid.vertex_valence(lonely) == 0


# ---------------------------------------------------------------------------
# M12 - identify_regular_regions at the boundary
# ---------------------------------------------------------------------------

def test_m12_boundary_vertex_with_three_quads_is_irregular():
    """M12 (cand 13): on an L-shaped patch the reflex boundary vertex has 3
    quads and valence 4, so it is irregular; boundary vertices with 1 or 2
    quads stay regular."""
    # 6 7        the 3x3 grid without its top-right cell (and without the
    # 3 4 5      now unused vertex 8); vertex 4 is the reflex corner
    # 0 1 2
    pts = [[i, j, 0.0] for j in range(3) for i in range(3)][:8]
    mesh = HalfEdgeMesh.from_arrays(np.array(pts, dtype=float),
                                    [[0, 1, 4, 3], [1, 2, 5, 4], [3, 4, 7, 6]])
    v4 = mesh.vertices[4]
    assert mesh.is_boundary_vertex(v4)
    assert len(mesh.get_vertex_faces(v4)) == 3
    assert all(len(mesh.get_face_vertices(f)) == 4 for f in mesh.get_vertex_faces(v4))
    assert len(true_neighbours(mesh, 4)) == 4

    regular, irregular = identify_regular_regions(mesh)
    reg = {v.index for v in regular}
    irr = {v.index for v in irregular}
    assert len(reg) + len(irr) == len(mesh.vertices) == 8
    assert 4 in irr, "boundary vertex with 3 quads must be irregular"
    assert reg == {0, 1, 2, 3, 5, 6, 7}, f"regular: {sorted(reg)}"
    assert irr == {4}


# ---------------------------------------------------------------------------
# M15 - normals of small triangles
# ---------------------------------------------------------------------------

def test_m15_halfedge_face_normal_of_small_triangle_is_unit():
    """M15 (cand 8): a 1e-5 triangle gets the unit normal (0, 0, 1), not 0."""
    for offset in ([0.0, 0.0, 0.0], [1150.0, -20.0, 37.0]):
        pts = np.array([[0, 0, 0], [1e-5, 0, 0], [0, 1e-5, 0]]) + offset
        mesh = HalfEdgeMesh.from_arrays(pts, [[0, 1, 2]])
        mesh.compute_face_normals()
        assert np.allclose(mesh.faces[0].normal, [0.0, 0.0, 1.0], atol=1e-6), \
            f"offset {offset}: {mesh.faces[0].normal}"


def test_m15_halfedge_normals_are_scale_invariant_and_degenerate_is_zero():
    """M15 (cand 8): the same quad scaled by 1e-7..1e7 gives the same unit
    normal; a collinear (zero-area) face returns the zero vector at any scale."""
    quad = np.array([[0, 0, 0], [2, 0, 0], [2, 1, 1], [0, 1, 1]], dtype=float)
    expected = np.array([0.0, -1.0, 1.0]) / np.sqrt(2.0)
    for s in (1e-7, 1e-5, 1.0, 1e7):
        mesh = HalfEdgeMesh.from_arrays(quad * s, [[0, 1, 2, 3]])
        mesh.compute_face_normals()
        assert np.allclose(mesh.faces[0].normal, expected, atol=1e-9), f"scale {s}"

        line = HalfEdgeMesh.from_arrays(np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2.0]]) * s,
                                        [[0, 1, 2]])
        line.compute_face_normals()
        assert np.array_equal(line.faces[0].normal, np.zeros(3)), f"scale {s}"


def test_m15_halfedge_zero_area_face_far_from_origin_is_zero():
    """M15 (cand 8): with a scale-relative threshold, the rounding of Newell's
    P + Q sums at |x| ~ 1e3 must not turn an exactly collinear small face into
    a noise "normal": it returns the zero vector. A valid face of the same size
    at the same place still gets its unit normal."""
    from fractions import Fraction
    p0 = np.array([1150.3, -20.7, 37.1])
    d = np.array([3.0, -5.0, 7.0]) * 2.0 ** -17          # dyadic: P0 + t*d is exact
    line = np.array([p0 + t * d for t in range(4)])
    exact = [[Fraction(x) for x in p] for p in line]
    for p, q in zip(exact[1:], exact[2:]):                # precondition: exactly collinear
        a = [p[i] - exact[0][i] for i in range(3)]
        b = [q[i] - exact[0][i] for i in range(3)]
        assert [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0]] == [0, 0, 0]
    mesh = HalfEdgeMesh.from_arrays(line, [[0, 1, 2, 3]])
    mesh.compute_face_normals()
    assert np.array_equal(mesh.faces[0].normal, np.zeros(3)), mesh.faces[0].normal

    e = 2.0 ** -17
    square = p0 + np.array([[0, 0, 0], [e, 0, 0], [e, e, 0], [0, e, 0]])
    mesh = HalfEdgeMesh.from_arrays(square, [[0, 1, 2, 3]])
    mesh.compute_face_normals()
    assert np.allclose(mesh.faces[0].normal, [0.0, 0.0, 1.0], atol=1e-6)


def test_m15_expand_selection_by_angle_on_small_flat_strip():
    """M15 (cand 8): a flat strip of two 1e-5 triangles is one 0-degree region,
    so a 10-degree flood fill from face 0 must reach face 1; a 90-degree fold
    of the same size must stop it."""
    s = 1e-5
    flat = HalfEdgeMesh.from_arrays(np.array([[0, 0, 0], [s, 0, 0], [s, s, 0], [0, s, 0]]),
                                    [[0, 1, 2], [0, 2, 3]])
    assert sorted(flat.expand_selection_by_angle([0], 10.0)) == [0, 1]

    fold = HalfEdgeMesh.from_arrays(np.array([[0, 0, 0], [s, 0, 0], [s / 2, s, 0], [s / 2, 0, s]]),
                                    [[0, 1, 2], [1, 0, 3]])
    assert sorted(fold.expand_selection_by_angle([0], 10.0)) == [0]


def test_m15_mesh_soa_small_triangle_normals_are_unit():
    """M15 (cand 7): MeshSOA face and vertex normals of 1e-5 triangles are unit
    length (0, 0, 1), not the raw cross product (0, 0, 1e-10)."""
    s = 1e-5
    soa = MeshSOA(np.array([[0, 0, 0], [s, 0, 0], [s, s, 0], [0, s, 0]]),
                  np.array([[0, 1, 2], [0, 2, 3]]))
    fn = soa.compute_face_normals()
    vn = soa.compute_vertex_normals()
    assert np.allclose(fn, [[0, 0, 1]] * 2, atol=1e-9), fn
    assert np.allclose(vn, [[0, 0, 1]] * 4, atol=1e-9), vn


def test_m15_mesh_soa_degenerate_faces_and_isolated_vertices_give_zero():
    """M15 (cand 7): a collinear face gets the zero normal (finite, no NaN) and
    leaves the normals of the valid neighbours untouched; a vertex that only
    touches degenerate faces, or none at all, gets the zero vector."""
    for s in (1e-6, 1.0, 1e6):
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0],      # valid, +Z
                          [2, 0, 0], [3, 0, 0],                 # collinear with 1
                          [9, 9, 9]], dtype=float) * s          # isolated
        soa = MeshSOA(verts, np.array([[0, 1, 2], [1, 3, 4]]))
        fn = soa.compute_face_normals()
        vn = soa.compute_vertex_normals()
        assert np.all(np.isfinite(fn)) and np.all(np.isfinite(vn))
        assert np.allclose(fn[0], [0, 0, 1], atol=1e-9)
        assert np.array_equal(fn[1], np.zeros(3))
        assert np.allclose(vn[[0, 1, 2]], [[0, 0, 1]] * 3, atol=1e-9)
        assert np.array_equal(vn[[3, 4, 5]], np.zeros((3, 3)))


# ---------------------------------------------------------------------------
# M16 - MeshSOA twins on non-manifold edges
# ---------------------------------------------------------------------------

def test_m16_mesh_soa_twins_are_symmetric_on_non_manifold_edge():
    """M16 (cand 9): three faces on edge 0-1 (plus a manifold edge 1-2 shared
    by faces 0 and 3): twin(twin(h)) == h everywhere, the three half-edges of
    the non-manifold edge stay unpaired, and the manifold edge is paired."""
    verts = np.random.default_rng(0).normal(size=(6, 3))
    faces = np.array([[0, 1, 2], [1, 0, 3], [1, 0, 4], [2, 1, 5]])
    soa = MeshSOA(verts, faces)
    v_from = soa.faces.ravel()
    on_01 = [h for h in range(soa.H) if {v_from[h], soa.he_vertex[h]} == {0, 1}]
    on_12 = [h for h in range(soa.H) if {v_from[h], soa.he_vertex[h]} == {1, 2}]
    assert len(on_01) == 3 and len(on_12) == 2      # precondition: the fin exists

    paired = np.where(soa.he_twin != -1)[0]
    for h in paired:
        assert soa.he_twin[soa.he_twin[h]] == h, \
            f"he{h}.twin = {soa.he_twin[h]} but he{soa.he_twin[h]}.twin = {soa.he_twin[soa.he_twin[h]]}"
    assert all(soa.he_twin[h] == -1 for h in on_01)
    assert soa.he_twin[on_12[0]] == on_12[1] and soa.he_twin[on_12[1]] == on_12[0]
    assert sorted(paired.tolist()) == sorted(on_12)
    # adjacency built from the twins is now symmetric too
    ff = soa.get_face_face_adjacency()
    assert ff[0].tolist().count(3) == 1 and ff[3].tolist().count(0) == 1
    assert ff[1].tolist().count(-1) == 3 and ff[2].tolist().count(-1) == 3
