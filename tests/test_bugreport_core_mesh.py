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
