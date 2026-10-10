"""Regression tests for the RE mesh-tool bugs of the automated bug hunt on
commit 4e2241347de43175401df20c6722f38fa70820b4 (bug-report.md).

Covered:
  M4  (cand 5)  _decimate_with_frozen folded triangles over (link condition
                is purely topological, nothing checked the face normals).
  M5  (cand 20) _decimate_with_frozen pinched an open boundary into a bowtie
                vertex by collapsing an interior edge joining two boundary
                vertices.
  M17 (cand 21) compute_mesh_quality overcounted edge_count and
                boundary_edges on non-manifold meshes.
  Suspicions from the report's "Not checked" list:
  S:collapse_short_edges_pinch  collapse_short_edges had the same boundary
                pinch as M5.
  S:decimate_ratio  decimate_mesh took ``ratio`` of the quad count but
                applied it to the triangulated mesh.

Every geometry is built in code; nothing reads a fixture file.
"""
import warnings

import numpy as np
import pytest
import trimesh

from src.core.halfedge_mesh import HalfEdgeMesh
from src.reverse_engineering.mesh_tools import (
    _decimate_with_frozen,
    collapse_short_edges,
    compute_mesh_quality,
    decimate_mesh,
)


# --------------------------------------------------------------------------
# helpers / checkers
# --------------------------------------------------------------------------

def _face_lists(mesh: HalfEdgeMesh):
    return [[v.index for v in mesh.get_face_vertices(f)] for f in mesh.faces]


def _signed_areas_z(verts, faces):
    """Signed area (z of the cross product / 2) of each planar triangle."""
    P = np.asarray(verts, dtype=np.float64)
    out = []
    for a, b, c in faces:
        out.append(0.5 * float(np.cross(P[b] - P[a], P[c] - P[a])[2]))
    return np.array(out)


def _bowtie_vertices(faces):
    """Vertices whose incident faces split into more than one edge-connected
    fan (a pinched / bowtie vertex), plus any vertex on an edge with more than
    two faces."""
    bad = set()
    edge_faces = {}
    for fi, f in enumerate(faces):
        for i in range(3):
            a, b = int(f[i]), int(f[(i + 1) % 3])
            edge_faces.setdefault((min(a, b), max(a, b)), []).append(fi)
    for (a, b), fs in edge_faces.items():
        if len(fs) > 2:
            bad.update((a, b))
    vert_faces = {}
    for fi, f in enumerate(faces):
        for v in f:
            vert_faces.setdefault(int(v), set()).add(fi)
    for v, fs in vert_faces.items():
        # flood-fill the faces around v across the edges that contain v
        fs = set(fs)
        start = next(iter(fs))
        seen = {start}
        stack = [start]
        while stack:
            fi = stack.pop()
            for w in faces[fi]:
                w = int(w)
                if w == v:
                    continue
                for g in edge_faces[(min(v, w), max(v, w))]:
                    if g in fs and g not in seen:
                        seen.add(g)
                        stack.append(g)
        if seen != fs:
            bad.add(v)
    return bad


def test_checkers_say_no_to_known_bad_input():
    """The checkers must flag the outputs the report gives for M4 and M5."""
    # M5's reported output: two triangles meeting only at vertex 1
    assert _bowtie_vertices([[0, 3, 1], [1, 5, 2]]) == {1}
    assert _bowtie_vertices([[0, 1, 2], [0, 2, 3]]) == set()
    # three faces on edge 0-1 is non-manifold too
    assert {0, 1} <= _bowtie_vertices([[0, 1, 2], [1, 0, 3], [0, 1, 4]])
    # a clockwise triangle has negative signed area
    V = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    assert _signed_areas_z(V, [[0, 1, 2]])[0] == pytest.approx(0.5)
    assert _signed_areas_z(V, [[0, 2, 1]])[0] == pytest.approx(-0.5)


# --------------------------------------------------------------------------
# M4 (cand 5): frozen-vertex decimation folds triangles
# --------------------------------------------------------------------------

def _fold_fan():
    """Planar fan around a frozen centre (vertex 4). Ring vertices, CCW:
    0 at 2 deg, 2 at 175 deg, 1 at 185 deg, 3 at 270 deg. The shortest
    collapsible edge is the ring edge 1-2; moving 2 onto 1 turns face
    (4,0,2) (173 deg span) into (4,0,1) (183 deg span) -- a fold."""
    ang = np.radians([2.0, 185.0, 175.0, 270.0])
    V = np.zeros((5, 3))
    V[:4, 0] = np.cos(ang)
    V[:4, 1] = np.sin(ang)
    F = [[4, 0, 2], [4, 2, 1], [4, 1, 3], [4, 3, 0]]
    return V, F


def test_m4_frozen_decimation_does_not_fold_triangles():
    """M4 (cand 5): decimate_mesh(frozen_vertices=...) must not flip a face.

    The fan with frozen={4}, target_faces=3 must lose exactly one face and
    every surviving face must keep a positive signed area. On the buggy code
    the 1-2 collapse produced face (4,0,1) with signed area -0.026.
    """
    V, F = _fold_fan()
    # precondition: the input is a valid CCW fan and 1-2 is the shortest
    # unfrozen edge, so it is the collapse tried first
    assert np.all(_signed_areas_z(V, F) > 0)
    unfrozen = [(0, 2), (1, 2), (1, 3), (0, 3)]
    lens = {e: np.linalg.norm(V[e[0]] - V[e[1]]) for e in unfrozen}
    assert min(lens, key=lens.get) == (1, 2)

    out = decimate_mesh(HalfEdgeMesh.from_arrays(V, F), target_faces=3,
                        frozen_vertices=[4])
    out_V = np.array([v.position for v in out.vertices])
    out_F = _face_lists(out)
    # the decimation did happen, and only down to the target
    assert len(out_F) == 3
    assert len(out_V) == 4
    areas = _signed_areas_z(out_V, out_F)
    assert np.all(areas > 0), f"folded face(s): {out_F} areas {areas}"
    # the frozen centre is still there, unmoved
    assert any(np.allclose(p, [0, 0, 0], atol=1e-12) for p in out_V)


def test_m4_direct_collapse_rejects_fold_and_takes_the_safe_edge():
    """M4 (cand 5): the private pass must skip the folding 1-2 collapse and
    collapse 1-3 instead (vertex 3 onto 1), leaving (4,0,2),(4,2,1),(4,1,0),
    all with positive area: 0.0609, 0.0868, 0.0262 (+-1e-4)."""
    V, F = _fold_fan()
    new_faces = _decimate_with_frozen(V.copy(), [list(f) for f in F], {4}, 3)
    assert sorted(map(tuple, new_faces)) == sorted([(4, 0, 2), (4, 2, 1), (4, 1, 0)])
    areas = _signed_areas_z(V, new_faces)
    expected = {(4, 0, 2): 0.5 * np.sin(np.radians(173.0)),
                (4, 2, 1): 0.5 * np.sin(np.radians(10.0)),
                (4, 1, 0): 0.5 * np.sin(np.radians(177.0))}
    for f, a in zip(new_faces, areas):
        assert a == pytest.approx(expected[tuple(f)], abs=1e-4)


# --------------------------------------------------------------------------
# M5 (cand 20): frozen-vertex decimation pinches boundaries
# --------------------------------------------------------------------------

def _pinch_strip(short_interior=True):
    """Open 4-triangle strip. Top row 0,1,2 and bottom row 3,4,5 -- every
    vertex is on the boundary. With short_interior the shortest edge is the
    INTERIOR edge 1-4 (0.2 long), joining two boundary vertices."""
    if short_interior:
        V = np.array([[0, 1, 0], [1, .6, 0], [2, 1, 0],
                      [0, 0, 0], [1, .4, 0], [2, 0, 0]], dtype=float)
    else:
        # the shortest edge is the BOUNDARY edge 0-1 (0.1 long)
        V = np.array([[0, 1, 0], [0.1, 1, 0], [2, 1, 0],
                      [0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
    F = [[3, 4, 0], [0, 4, 1], [4, 5, 1], [1, 5, 2]]
    return V, F


def test_m5_strip_precondition():
    """M5 (cand 20) precondition: the strip is manifold, CCW, and its
    shortest edge is the interior edge 1-4 with both endpoints on the
    boundary (so the link condition alone accepts it)."""
    V, F = _pinch_strip()
    assert _bowtie_vertices(F) == set()
    assert np.all(_signed_areas_z(V, F) > 0)
    tm = trimesh.Trimesh(V, F, process=False)
    edges = tm.edges_unique
    lens = tm.edges_unique_length
    assert tuple(sorted(edges[np.argmin(lens)])) == (1, 4)
    assert lens.min() == pytest.approx(0.2, abs=1e-12)
    # every vertex is a boundary vertex
    q = compute_mesh_quality(HalfEdgeMesh.from_arrays(V, F))
    assert q['boundary_edges'] == 6


def test_m5_direct_decimation_keeps_boundary_manifold():
    """M5 (cand 20): with no frozen vertex and target 2 the pass must still
    reach 2 faces but never via the pinching 1-4 collapse; the result must
    have no bowtie vertex. The buggy code returned [[3,1,0],[1,5,2]]."""
    V, F = _pinch_strip()
    new_faces = _decimate_with_frozen(V.copy(), [list(f) for f in F], set(), 2)
    assert len(new_faces) == 2
    assert _bowtie_vertices(new_faces) == set(), new_faces
    assert np.all(_signed_areas_z(V, new_faces) > 0)
    assert sorted(map(tuple, new_faces)) == sorted([(0, 4, 1), (4, 2, 1)])


def test_m5_boundary_status_is_inherited_by_the_survivor():
    """M5 (cand 20): after the interior vertex 4 absorbs a boundary vertex it
    is itself on the boundary, so a later interior edge from it to another
    boundary vertex must not be collapsed either. On this jittered 2x2 sheet
    (target 3) forgetting that gives the bowtie [[0,2,3],[3,8,6]]; the right
    result is [[0,4,3],[0,2,4],[4,8,3]]."""
    V = np.array([[0.297, -0.11], [0.81, 0.228], [2.187, 0.101],
                  [0.275, 1.255], [1.149, 1.216], [1.848, 0.785],
                  [0.102, 2.129], [0.8, 1.937], [2.246, 2.037]])
    V = np.column_stack([V, np.zeros(len(V))])
    F = [[0, 1, 3], [1, 4, 3], [1, 2, 5], [1, 5, 4],
         [3, 4, 6], [4, 7, 6], [4, 5, 7], [5, 8, 7]]
    assert _bowtie_vertices(F) == set()
    assert np.all(_signed_areas_z(V, F) > 0)
    new_faces = _decimate_with_frozen(V.copy(), [list(f) for f in F], set(), 3)
    assert sorted(map(tuple, new_faces)) == sorted([(0, 4, 3), (0, 2, 4), (4, 8, 3)])
    assert _bowtie_vertices(new_faces) == set()
    assert np.all(_signed_areas_z(V, new_faces) > 0)


def test_m4_m5_random_open_sheets_stay_manifold_and_fold_free():
    """M4 (cand 5) + M5 (cand 20): 300 seeded random jittered open sheets,
    decimated to a random target: no output may contain a bowtie vertex or a
    face with non-positive signed area, and the pass must actually remove
    faces on most of them (it is not allowed to get there by doing nothing)."""
    rng = np.random.default_rng(12345)
    reduced = 0
    for _ in range(300):
        n, m = int(rng.integers(1, 4)), int(rng.integers(1, 4))
        xs, ys = np.meshgrid(np.arange(n + 1.0), np.arange(m + 1.0))
        V = np.column_stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)])
        V[:, :2] += rng.uniform(-0.3, 0.3, (len(V), 2))
        F = []
        for j in range(m):
            for i in range(n):
                a = j * (n + 1) + i
                if rng.random() < 0.5:
                    F += [[a, a + 1, a + n + 2], [a, a + n + 2, a + n + 1]]
                else:
                    F += [[a, a + 1, a + n + 1], [a + 1, a + n + 2, a + n + 1]]
        assert np.all(_signed_areas_z(V, F) > 0)
        tgt = int(rng.integers(1, len(F))) if len(F) > 1 else 1
        out = _decimate_with_frozen(V.copy(), [list(f) for f in F], set(), tgt)
        assert out, "decimation emptied the mesh"
        assert _bowtie_vertices(out) == set(), (V.tolist(), F, tgt, out)
        assert np.all(_signed_areas_z(V, out) > 0), (V.tolist(), F, tgt, out)
        reduced += len(out) < len(F)
    assert reduced >= 150


def test_m5_public_decimation_keeps_boundary_manifold():
    """M5 (cand 20): same strip through decimate_mesh(frozen_vertices=[2]),
    target 2: two faces, manifold, frozen vertex 2 kept at (2,1,0)."""
    V, F = _pinch_strip()
    out = decimate_mesh(HalfEdgeMesh.from_arrays(V, F), target_faces=2,
                        frozen_vertices=[2])
    out_F = _face_lists(out)
    out_V = np.array([v.position for v in out.vertices])
    assert len(out_F) == 2
    assert _bowtie_vertices(out_F) == set(), out_F
    assert np.all(_signed_areas_z(out_V, out_F) > 0)
    assert any(np.allclose(p, [2, 1, 0], atol=1e-12) for p in out_V)


# --------------------------------------------------------------------------
# S:collapse_short_edges_pinch
# --------------------------------------------------------------------------

def test_collapse_short_edges_does_not_pinch_open_boundary():
    """Suspicion (same pinch as M5 in collapse_short_edges): the only short
    edge (rel_threshold 0.3 -> threshold 0.323) is the interior edge 1-4
    joining two boundary vertices. It must be left alone: 4 faces, 6
    vertices, no bowtie. The buggy code merged it into [[3,1,0],[1,4,2]]."""
    V, F = _pinch_strip()
    tm = trimesh.Trimesh(V, F, process=False)
    lens = np.sort(tm.edges_unique_length)
    thr = 0.3 * float(np.median(lens))
    assert np.sum(lens < thr) == 1        # exactly one short edge (1-4)

    out = collapse_short_edges(tm, rel_threshold=0.3)
    assert len(out.faces) == 4
    assert len(out.vertices) == 6
    assert _bowtie_vertices(out.faces.tolist()) == set()


def test_collapse_short_edges_still_collapses_short_boundary_edge():
    """Companion to the pinch test: a short BOUNDARY edge (0-1, 0.1 long)
    is still collapsed -- 4 faces -> 3, 6 vertices -> 5, manifold."""
    V, F = _pinch_strip(short_interior=False)
    tm = trimesh.Trimesh(V, F, process=False)
    lens = np.sort(tm.edges_unique_length)
    assert np.sum(lens < 0.3 * float(np.median(lens))) == 1

    out = collapse_short_edges(tm, rel_threshold=0.3)
    assert len(out.faces) == 3
    assert len(out.vertices) == 5
    assert _bowtie_vertices(out.faces.tolist()) == set()
