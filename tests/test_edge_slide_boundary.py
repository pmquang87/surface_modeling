"""edge_slide at the mesh boundary (follow-up to bug H4 of the bug report
``bug-report.md``, automated bug hunt on commit
4e2241347de43175401df20c6722f38fa70820b4).

H4 made ``get_vertex_neighbors`` return both boundary neighbours of a
boundary vertex. ``edge_slide`` moves a vertex along the mean of its unit
edge vectors, so after H4 a boundary vertex slid INTO the mesh: a grid corner
went along the bisector, a vertex on a straight boundary straight inward, and
the outline got dented. (Before H4 the missing neighbour made it slide along
one boundary edge by accident.) The rule is now explicit:

* an interior vertex keeps the mean-direction rule;
* a boundary vertex stays on the outline: it slides along whichever of its
  boundary edges the move favours, by at most that edge's length, and stays
  put when the move favours neither (a straight boundary, a symmetric corner);
* indices are normalised like the other editing ops, and only an exactly
  zero-length edge is skipped (an absolute 1e-6 cut-off froze every vertex of
  a sub-micron cage - the M15 pattern).

Test discipline: preconditions are asserted (the vertex really is on the
boundary, the defect input really moves it off the outline under the old
rule), and the outline checker has inputs it must accept and reject.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.halfedge_mesh import HalfEdgeMesh  # noqa: E402
from src.subd.editing import edge_slide  # noqa: E402


def _grid():
    """3 x 3 vertices (index = 3*y + x), four CCW quads, normal +z."""
    pts = [[x, y, 0.0] for y in range(3) for x in range(3)]
    faces = [[0, 1, 4, 3], [1, 2, 5, 4], [3, 4, 7, 6], [4, 5, 8, 7]]
    return HalfEdgeMesh.from_arrays(pts, faces)


def _fan():
    """Boundary vertex 0 at the origin with boundary neighbours 1 (+x) and
    2 (-x) and one interior edge 0-3 leaning toward +x."""
    pts = [[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0.5, 1, 0], [1.5, 1, 0]]
    faces = [[0, 1, 4, 3], [2, 0, 3]]
    return HalfEdgeMesh.from_arrays(pts, faces)


def _boundary_neighbours(mesh, v_idx):
    """Neighbours joined to v by an edge that only one face uses."""
    use = {}
    for f in mesh.faces:
        poly = [v.index for v in mesh.get_face_vertices(f)]
        for a, b in zip(poly, poly[1:] + poly[:1]):
            key = (min(a, b), max(a, b))
            use[key] = use.get(key, 0) + 1
    return sorted(n for (a, b), c in use.items() if c == 1
                  for n in ((b,) if a == v_idx else (a,) if b == v_idx else ()))


def _dist_to_segment(p, a, b):
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0.0, 1.0)
    return float(np.linalg.norm(p - (a + t * ab)))


def _on_outline(mesh, v_idx, p, tol=1e-12):
    """True when p lies on one of v's two ORIGINAL boundary edges."""
    v = mesh.vertices[v_idx].position
    return any(_dist_to_segment(p, v, mesh.vertices[n].position) <= tol
               for n in _boundary_neighbours(mesh, v_idx))


def _pos(mesh, i):
    return np.asarray(mesh.vertices[i].position, dtype=float)


def test_outline_checker_accepts_and_rejects():
    """Control for the checker the boundary tests rely on."""
    g = _grid()
    assert _boundary_neighbours(g, 0) == [1, 3]
    assert _boundary_neighbours(g, 4) == []          # interior vertex
    assert _on_outline(g, 0, np.array([0.3, 0.0, 0.0]))
    assert _on_outline(g, 0, np.array([0.0, 0.7, 0.0]))
    assert not _on_outline(g, 0, np.array([0.3536, 0.3536, 0.0]))
    assert not _on_outline(g, 1, np.array([1.0, 0.5, 0.0]))


@pytest.mark.parametrize("v_idx", [0, 1, 2, 3, 5, 6, 7, 8])
@pytest.mark.parametrize("amount", [0.3, -0.3, 5.0])
def test_grid_boundary_vertices_stay_on_the_outline(v_idx, amount):
    """Every boundary vertex of the 2 x 2 grid stays on its own boundary
    edges. On the old rule the corners moved along the bisector and the
    edge midpoints straight inward or outward."""
    g = _grid()
    assert len(_boundary_neighbours(g, v_idx)) == 2   # precondition
    out = edge_slide(g, [v_idx], amount)
    assert _on_outline(g, v_idx, _pos(out, v_idx)), _pos(out, v_idx)
    # nothing else moved
    for i in range(len(g.vertices)):
        if i != v_idx:
            assert np.array_equal(_pos(out, i), _pos(g, i))


@pytest.mark.parametrize("v_idx", [0, 1, 8])
def test_symmetric_boundary_vertices_do_not_move(v_idx):
    """A corner (only boundary edges: the mean direction is their bisector,
    which favours neither) and a vertex on a straight boundary (the move is
    perpendicular to it) stay exactly where they are."""
    g = _grid()
    out = edge_slide(g, [v_idx], 0.4)
    assert np.array_equal(_pos(out, v_idx), _pos(g, v_idx))


def test_boundary_vertex_slides_along_the_favoured_edge():
    """The fan's interior edge leans toward +x, so vertex 0 slides along its
    boundary edge to vertex 1; a negative amount slides it toward vertex 2;
    a slide longer than the edge stops at the neighbour."""
    m = _fan()
    assert _boundary_neighbours(m, 0) == [1, 2]       # precondition
    np.testing.assert_allclose(_pos(edge_slide(m, [0], 0.3), 0), [0.3, 0, 0], atol=1e-15)
    np.testing.assert_allclose(_pos(edge_slide(m, [0], -0.3), 0), [-0.3, 0, 0], atol=1e-15)
    np.testing.assert_allclose(_pos(edge_slide(m, [0], 5.0), 0), [1.0, 0, 0], atol=1e-15)


def test_interior_vertices_keep_the_mean_direction_rule():
    """Control: an interior vertex still moves by `amount` along the unit
    mean of its unit edge vectors (unchanged behaviour)."""
    g = _grid()
    pts = np.array([v.position for v in g.vertices], dtype=float)
    pts[5] += [0.3, 0.2, 0.1]                         # make vertex 4 irregular
    g = HalfEdgeMesh.from_arrays(pts, [[0, 1, 4, 3], [1, 2, 5, 4], [3, 4, 7, 6], [4, 5, 8, 7]])
    v = g.vertices[4]
    units = [(n.position - v.position) / np.linalg.norm(n.position - v.position)
             for n in g.get_vertex_neighbors(v)]
    mean = np.sum(units, axis=0) / len(units)
    expected = v.position + mean / np.linalg.norm(mean) * 0.25
    np.testing.assert_allclose(_pos(edge_slide(g, [4], 0.25), 4), expected, atol=1e-15)


@pytest.mark.parametrize("scale", [1e-7, 1e3])
def test_slide_is_scale_invariant(scale):
    """The same cage at another scale slides by the scaled amount. The old
    absolute 1e-6 edge-length cut-off skipped every edge of a sub-micron
    cage, so nothing moved there."""
    pts = np.array([v.position for v in _grid().vertices], dtype=float)
    pts[5] += [0.3, 0.2, 0.1]
    faces = [[0, 1, 4, 3], [1, 2, 5, 4], [3, 4, 7, 6], [4, 5, 8, 7]]
    ref = edge_slide(HalfEdgeMesh.from_arrays(pts, faces), [4, 0], 0.25)
    out = edge_slide(HalfEdgeMesh.from_arrays(pts * scale, faces), [4, 0], 0.25 * scale)
    assert not np.allclose(_pos(ref, 4), pts[4])      # precondition: it moves
    np.testing.assert_allclose(_pos(out, 4) / scale, _pos(ref, 4), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(_pos(out, 0) / scale, _pos(ref, 0), rtol=1e-12, atol=1e-12)


def test_indices_are_normalised():
    """A repeated index slides once, -1 is the last vertex, and indices out
    of range on either side are skipped instead of raising."""
    pts = np.array([v.position for v in _grid().vertices], dtype=float)
    pts[5] += [0.3, 0.2, 0.1]
    faces = [[0, 1, 4, 3], [1, 2, 5, 4], [3, 4, 7, 6], [4, 5, 8, 7]]
    g = HalfEdgeMesh.from_arrays(pts, faces)
    once = edge_slide(g, [4], 0.25)
    twice = edge_slide(g, [4, 4, 4 - 9], 0.25)        # 4, 4 and its alias -5
    assert np.array_equal(_pos(twice, 4), _pos(once, 4))
    assert np.array_equal(_pos(edge_slide(g, [-1], 0.25), 8),
                          _pos(edge_slide(g, [8], 0.25), 8))
    untouched = edge_slide(g, [-10, 9, 100], 0.25)
    assert all(np.array_equal(_pos(untouched, i), pts[i]) for i in range(9))
