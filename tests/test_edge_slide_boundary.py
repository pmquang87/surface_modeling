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
boundary; on the jittered grid the boundary vertices really move, and an
inline copy of the old mean-direction rule really takes them off the
outline), and the outline checker has inputs it must accept and reject.
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


def _jittered_grid(nx=5, ny=4, seed=7, scale=1.0):
    """An open quad grid with seeded in-plane jitter, so boundary vertices
    have asymmetric edges and really slide."""
    rng = np.random.default_rng(seed)
    pts = np.array([[x, y, 0.0] for y in range(ny) for x in range(nx)], dtype=float)
    pts[:, :2] += rng.uniform(-0.25, 0.25, size=(len(pts), 2))
    faces = [[y * nx + x, y * nx + x + 1, (y + 1) * nx + x + 1, (y + 1) * nx + x]
             for y in range(ny - 1) for x in range(nx - 1)]
    return HalfEdgeMesh.from_arrays(pts * scale, faces), pts * scale, faces


def _old_rule(mesh, v_idx, amount):
    """The pre-H4-follow-up rule, inline: move along the unit mean of the
    unit edge vectors, boundary or not (zero-length edges skipped, as the
    old code's length cut-off did)."""
    v = mesh.vertices[v_idx]
    units = [(n.position - v.position) / np.linalg.norm(n.position - v.position)
             for n in mesh.get_vertex_neighbors(v)
             if np.linalg.norm(n.position - v.position) > 0.0]
    mean = np.sum(units, axis=0) / len(units)
    return v.position + mean / np.linalg.norm(mean) * amount


def _oracle(mesh, v_idx, amount):
    """Independent statement of the documented rule for a boundary vertex:
    (position, chosen neighbour or None)."""
    v = mesh.vertices[v_idx]
    units = [(n.position - v.position) / np.linalg.norm(n.position - v.position)
             for n in mesh.get_vertex_neighbors(v)]
    mean = np.sum(units, axis=0) / len(units)
    move = mean / np.linalg.norm(mean) * amount
    cands = []
    for n in _boundary_neighbours(mesh, v_idx):
        e = mesh.vertices[n].position - v.position
        cands.append((float(np.dot(move, e / np.linalg.norm(e))), n, np.linalg.norm(e)))
    cands.sort(key=lambda c: -c[0])
    tol = 1e-9 * abs(amount)
    if cands[0][0] <= tol or (len(cands) > 1 and cands[1][0] >= cands[0][0] - tol):
        return v.position.copy(), None
    _, n, length = cands[0]
    if abs(amount) >= length:
        return mesh.vertices[n].position.copy(), n
    e = mesh.vertices[n].position - v.position
    return v.position + e / length * abs(amount), n


def _boundary_ring(nx=5, ny=4):
    return [i for i in range(nx * ny)
            if i % nx in (0, nx - 1) or i // nx in (0, ny - 1)]


@pytest.mark.parametrize("amount", [0.15, -0.15, 0.4, 50.0])
def test_jittered_boundary_vertices_slide_along_the_favoured_edge(amount):
    """On a jittered grid most boundary vertices really move; each lands on
    one of its own boundary edges - the one the move favours - by `amount`,
    or exactly on that neighbour when the edge is shorter."""
    m, _, _ = _jittered_grid()
    ring = _boundary_ring()
    assert all(len(_boundary_neighbours(m, i)) == 2 for i in ring)   # precondition
    old_off = sum(not _on_outline(m, i, _old_rule(m, i, amount)) for i in ring)
    assert old_off >= len(ring) - 2, old_off        # the old rule really leaves it
    moved = 0
    for i in ring:
        expected, n = _oracle(m, i, amount)
        got = _pos(edge_slide(m, [i], amount), i)
        assert _on_outline(m, i, got), (i, got)
        np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)
        if n is not None:
            moved += 1
            if abs(amount) >= np.linalg.norm(m.vertices[n].position - m.vertices[i].position):
                assert np.array_equal(got, _pos(m, n))   # exact end stop
    # bracket: at least half the ring slid (a negative amount points the
    # move away from both boundary edges at the corners, so fewer move then)
    assert moved >= len(ring) // 2, moved


def test_all_boundary_vertices_at_once_match_one_at_a_time():
    """Moves use the original positions, so sliding the whole ring at once
    gives each vertex the same place as sliding it alone."""
    m, _, _ = _jittered_grid()
    ring = _boundary_ring()
    together = edge_slide(m, ring, 0.2)
    for i in ring:
        assert np.array_equal(_pos(together, i), _pos(edge_slide(m, [i], 0.2), i))


@pytest.mark.parametrize("scale", [1e-7, 1e3])
def test_jittered_boundary_slide_is_scale_invariant(scale):
    """Boundary slides scale with the cage (the old 1e-6 edge cut-off froze
    the whole sub-micron ring)."""
    ref, _, _ = _jittered_grid()
    big, _, _ = _jittered_grid(scale=scale)
    ring = _boundary_ring()
    for i in ring:
        a = _pos(edge_slide(ref, [i], 0.15), i)
        b = _pos(edge_slide(big, [i], 0.15 * scale), i) / scale
        np.testing.assert_allclose(b, a, rtol=1e-9, atol=1e-12)
    assert sum(not np.array_equal(_pos(edge_slide(ref, [i], 0.15), i), _pos(ref, i))
               for i in ring) >= len(ring) - 4          # precondition: they move


def test_clamped_slide_lands_exactly_on_the_neighbour():
    """A slide longer than the edge stops exactly on the neighbour, not an
    ulp off it (an ulp-short edge then steered the neighbour's next slide)."""
    pts = [[0, 0, 0], [2.3943513636620226, -0.027479281000940815, 0],
           [-1, 0, 0], [0.5, 1, 0], [1.5, 1, 0]]
    m = HalfEdgeMesh.from_arrays(pts, [[0, 1, 4, 3], [2, 0, 3]])
    out = edge_slide(m, [0], 100.0)
    assert np.array_equal(_pos(out, 0), _pos(m, 1)), _pos(out, 0) - _pos(m, 1)


def test_pinched_vertex_keeps_to_its_boundary_edges():
    """A closed cone fan around vertex 0 plus a flap triangle that touches it
    only at vertex 0: get_vertex_neighbors walks one fan, but the vertex's
    boundary edges are the flap's 0-6 and 0-7. It must stay on them (here:
    stay put, the move points away from both), whichever fan is walked."""
    ring = [[np.cos(a), np.sin(a), -1.0] for a in np.linspace(0, 2 * np.pi, 6)[:-1]]
    pts = [[0, 0, 0]] + ring + [[0.3, 0.1, 1.0], [-0.2, 0.3, 1.0]]
    fan = [[0, k, k % 5 + 1] for k in range(1, 6)]
    for faces in (fan + [[0, 6, 7]], [[0, 6, 7]] + fan):
        m = HalfEdgeMesh.from_arrays(pts, faces)
        assert _boundary_neighbours(m, 0) == [6, 7]          # precondition
        out = edge_slide(m, [0], 0.3)
        assert _on_outline(m, 0, _pos(out, 0)), _pos(out, 0)


def test_boundary_vertex_with_only_zero_length_boundary_edges_stays():
    """Vertices 0, 1 and 2 of the grid coincide, so vertex 1's boundary edges
    have zero length: its outline is a single point. It must stay on it, not
    fall through to the interior rule and move into the mesh."""
    pts = np.array([v.position for v in _grid().vertices], dtype=float)
    pts[0] = pts[2] = pts[1]
    m = HalfEdgeMesh.from_arrays(pts, [[0, 1, 4, 3], [1, 2, 5, 4], [3, 4, 7, 6], [4, 5, 8, 7]])
    assert _boundary_neighbours(m, 1) == [0, 2]                  # precondition
    assert all(np.array_equal(_pos(m, n), _pos(m, 1)) for n in (0, 2))
    assert not np.allclose(_old_rule(m, 1, 0.3), _pos(m, 1))     # old rule moves it
    assert np.array_equal(_pos(edge_slide(m, [1], 0.3), 1), _pos(m, 1))
