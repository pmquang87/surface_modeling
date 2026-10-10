"""Same-pattern follow-ups to the bug report ``bug-report.md`` (automated bug
hunt on commit 4e2241347de43175401df20c6722f38fa70820b4), found by the
integration review of the merged fix branch.

Each test names its item and the original bug it mirrors:

* R1 (mirrors M15, cand 7/8 - "small but valid faces get bad normals") -
  ``SubDToNURBSConverter._evaluate_dense_grid`` took the bump normal from
  ``cross(c1 - c0, c3 - c0)`` with the absolute test ``n_len > 1e-6`` and fell
  back to (0, 0, 1). A valid quad with edges below ~1e-3 (a mm-size cage
  modelled in metres) was bumped in-plane along z instead of along its normal,
  and a truly degenerate quad got an arbitrary +z bump.
* R2 (mirrors M14, cand 17 - "bridge_faces punches holes") - ``bridge_faces``
  still punched holes when the two groups shared a face (also through index
  aliases such as [4] and [-2] on a 6-face box), and for opposite faces of a
  box, whose new side quads coincide with the existing walls.

Test discipline (the repo's vacuous-test rules): each test first asserts that
the defect / precondition really is in its input, every checker gets an input
it must accept and one it must reject, and counts are bracketed from both
sides (what must be refused AND what must still be done).
"""
import os
import sys
from collections import Counter

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.halfedge_mesh import HalfEdgeMesh  # noqa: E402
from src.nurbs.converter import SubDToNURBSConverter  # noqa: E402
from src.subd.editing import bridge_faces  # noqa: E402
from src.subd.primitives import create_box, create_plane  # noqa: E402


# --------------------------------------------------------------------------
# R1 (mirrors M15): the dense-grid bump normal of a small quad
# --------------------------------------------------------------------------

def _xz_quad(side, origin=(0.0, 0.0, 0.0)):
    """A square in the x-z plane; its true normal is -y (c1-c0 = +x, c3-c0 = +z)."""
    p = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]], dtype=np.float64)
    p = p * side + np.asarray(origin, dtype=np.float64)
    m = HalfEdgeMesh.from_arrays(p, [[0, 1, 2, 3]])
    return m, p


def _grid(mesh, positions, grid_size=5):
    return SubDToNURBSConverter()._evaluate_dense_grid(
        mesh, mesh.faces[0], positions, grid_size=grid_size)


def _bilinear(corners, grid_size):
    c0, c1, c2, c3 = corners
    u = np.linspace(0.0, 1.0, grid_size)
    g = np.zeros((grid_size, grid_size, 3))
    for i, a in enumerate(u):
        for j, b in enumerate(u):
            g[i, j] = (1 - a) * (1 - b) * c0 + a * (1 - b) * c1 + a * b * c2 + (1 - a) * b * c3
    return g


def _old_dense_grid(mesh, face, limit_positions, grid_size):
    """Verbatim copy of the aec2331 algorithm, the 'before' of the R1 fix."""
    corners = [limit_positions[v.index] for v in mesh.get_face_vertices(face)]
    c0, c1, c2, c3 = corners
    normal = np.cross(c1 - c0, c3 - c0)
    n_len = np.linalg.norm(normal)
    if n_len > 1e-6:
        normal = normal / n_len
    else:
        normal = np.array([0, 0, 1])
    grid = np.zeros((grid_size, grid_size, 3))
    vals = np.linspace(0, 1, grid_size)
    for i, u in enumerate(vals):
        for j, v in enumerate(vals):
            p = (1-u)*(1-v)*c0 + u*(1-v)*c1 + u*v*c2 + (1-u)*v*c3
            bump = u * (1 - u) * v * (1 - v) * 4.0
            side_len = np.linalg.norm(c1 - c0)
            grid[i, j] = p + normal * (side_len * 0.1) * bump
    return grid


def test_r1_old_reference_reproduces_the_defect():
    """R1 (mirrors M15): the checker below must be able to say 'no'. The
    aec2331 algorithm bumps the 1e-4 quad along +z, in its own plane."""
    m, p = _xz_quad(1e-4)
    centre = _old_dense_grid(m, m.faces[0], p, 5)[2, 2] - p.mean(axis=0)
    np.testing.assert_allclose(centre / 1e-4, [0.0, 0.0, 0.025], atol=1e-12)


@pytest.mark.parametrize("side", [1e-6, 1e-4, 1e-2, 1e3])
def test_r1_dense_grid_is_scale_invariant(side):
    """R1 (mirrors M15): the bump of a valid quad goes along its normal at
    every scale. Expected: grid / side is the same for side 1 and `side`, and
    the centre is displaced by 0.025 * side along the face normal (-y)."""
    m1, p1 = _xz_quad(1.0)
    g1 = _grid(m1, p1)
    # precondition: at side 1 the bump exists and is along the true normal
    np.testing.assert_allclose(g1[2, 2] - p1.mean(axis=0), [0.0, -0.025, 0.0], atol=1e-15)

    m, p = _xz_quad(side)
    g = _grid(m, p)
    centre = (g[2, 2] - p.mean(axis=0)) / side
    np.testing.assert_allclose(centre, [0.0, -0.025, 0.0], atol=1e-12,
                               err_msg=f"side {side}: bump not along the face normal")
    np.testing.assert_allclose(g / side, g1, atol=1e-12)


def test_r1_degenerate_quad_gets_no_bump():
    """R1 (mirrors M15): a zero-area quad has no normal. Expected: zero
    displacement (the plain bilinear grid), not an arbitrary +z bump, at
    unit size and for a small exactly collinear quad far from the origin
    (dyadic steps, so P0 + t*d is exact, as in the M15 core test)."""
    lines = [
        np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=np.float64) * 0.5,
        np.array([1150.3, -20.7, 37.1])
        + np.array([t * np.array([3.0, -5.0, 7.0]) * 2.0 ** -17 for t in range(4)]),
    ]
    for p in lines:
        side = np.linalg.norm(p[1] - p[0])
        # precondition: exactly zero area
        assert np.array_equal(np.cross(p[1] - p[0], p[3] - p[0]), np.zeros(3))
        m = HalfEdgeMesh.from_arrays(p, [[0, 1, 2, 3]])
        # precondition: the aec2331 code does bump it, along +z
        old = _old_dense_grid(m, m.faces[0], p, 5)
        # (the bump peaks at 0.25 * 0.1 * side in the centre)
        assert old[2, 2, 2] - _bilinear(p, 5)[2, 2, 2] > 0.02 * side
        g = _grid(m, p)
        assert np.all(np.isfinite(g))
        np.testing.assert_array_equal(g, _bilinear(p, 5))

    # positive control: a valid quad is not mistaken for a degenerate one
    m, p = _xz_quad(1e-6, origin=(0.0, 0.0, 0.0))
    assert np.abs(_grid(m, p) - _bilinear(p, 5)).max() > 0.02 * 1e-6


def test_r1_ordinary_scale_grids_are_unchanged():
    """R1 (mirrors M15): at ordinary scale the GUI 'Convert to NURBS' output
    must not change. Expected: bit-identical to the aec2331 algorithm for
    every face of a limit-evaluated box and for random non-degenerate quads."""
    from src.subd.catmull_clark import evaluate_limit_surface
    box = create_box()
    lim, _ = evaluate_limit_surface(box)
    for f in box.faces:
        for gs in (5, 6, 17):
            new = SubDToNURBSConverter()._evaluate_dense_grid(box, f, lim, grid_size=gs)
            np.testing.assert_array_equal(new, _old_dense_grid(box, f, lim, gs))

    rng = np.random.default_rng(1234)
    for _ in range(50):
        p = rng.uniform(-2.0, 2.0, size=(4, 3))
        m = HalfEdgeMesh.from_arrays(p, [[0, 1, 2, 3]])
        if np.linalg.norm(np.cross(p[1] - p[0], p[3] - p[0])) < 1e-3:
            continue
        np.testing.assert_array_equal(_grid(m, p, 6), _old_dense_grid(m, m.faces[0], p, 6))


# --------------------------------------------------------------------------
# R2 (mirrors M14): bridge_faces on overlapping groups / coinciding walls
# --------------------------------------------------------------------------

def _faces(mesh):
    return [[v.index for v in mesh.get_face_vertices(f)] for f in mesh.faces]


def _directed(mesh):
    d = Counter()
    for fv in _faces(mesh):
        for i in range(len(fv)):
            d[(fv[i], fv[(i + 1) % len(fv)])] += 1
    return d


def _boundary(mesh):
    """Directed edges with no opposite partner: the open boundary."""
    d = _directed(mesh)
    return {k for k in d if (k[1], k[0]) not in d}


def _same(a, b):
    if len(a.vertices) != len(b.vertices) or _faces(a) != _faces(b):
        return False
    pa = np.array([v.position for v in a.vertices])
    pb = np.array([v.position for v in b.vertices])
    return np.array_equal(pa, pb)


def _sound_bridge(before, after) -> bool:
    """Every directed edge used at most once, the open boundary unchanged
    (none for a closed input), and every face has 3+ distinct vertices."""
    if max(_directed(after).values(), default=0) > 1:
        return False
    if any(len(set(fv)) < 3 for fv in _faces(after)):
        return False
    return _boundary(after) == _boundary(before)


def test_r2_checkers_reject_broken_meshes():
    """The checkers must be able to say 'no'."""
    box = create_box()
    assert _sound_bridge(box, box) and _same(box, box.copy())
    holed = HalfEdgeMesh.from_arrays([v.position for v in box.vertices], _faces(box)[1:])
    assert not _sound_bridge(box, holed) and not _same(box, holed)
    doubled = HalfEdgeMesh.from_arrays([v.position for v in box.vertices],
                                       _faces(box) + [_faces(box)[0]])
    assert not _sound_bridge(box, doubled)
    moved = box.copy()
    moved.vertices[0].position = moved.vertices[0].position + 1e-9
    assert not _same(box, moved)


@pytest.mark.parametrize("a,b", [([f], [f]) for f in range(6)] + [([4], [-2]), ([-6], [0]),
                                                                  ([0, 2], [2, 0])])
def test_r2_overlapping_groups_return_the_untouched_mesh(a, b):
    """R2 (mirrors M14): groups that share a face (after negative-index
    normalisation) cannot be bridged. The old code deleted the face and built
    side quads from it twice: 6 -> 9 faces, 6 open half-edges. Expected: the
    untouched mesh, as the docstring promises."""
    box = create_box()
    assert {i % 6 for i in a} & {i % 6 for i in b}, "precondition: groups overlap"
    out = bridge_faces(box, a, b)
    assert _same(out, box), (f"{len(box.faces)} faces in, {len(out.faces)} out, "
                             f"{sum(1 for he in out.half_edges if he.twin is None)} open half-edges")
    assert out is not box


def test_r2_every_ordered_pair_of_box_faces():
    """R2 (mirrors M14): for opposite faces of a box the four new side quads
    are the four existing walls, so the old code gave 6 -> 8 faces with 8 open
    half-edges. Expected for every ordered pair: a sound closed mesh or exactly
    the untouched box; and bracketed both ways, the 6 same-face and 6
    opposite-face pairs are refused while the 24 adjacent pairs still bridge."""
    box = create_box()
    fl = _faces(box)
    refused, bridged = set(), set()
    for a in range(6):
        for b in range(6):
            out = bridge_faces(box, [a], [b])
            if _same(out, box):
                refused.add((a, b))
            else:
                assert _sound_bridge(box, out), (a, b, _faces(out))
                bridged.add((a, b))
    opposite = {(a, b) for a in range(6) for b in range(6)
                if a != b and not set(fl[a]) & set(fl[b])}
    # precondition: opposite faces really span all 8 corners, so every wall's
    # vertex set is an edge of a joined to an edge of b
    assert len(opposite) == 6
    assert all(len(set(fl[a]) | set(fl[b])) == 8 for a, b in opposite)
    same = {(a, a) for a in range(6)}
    assert refused == same | opposite
    assert len(bridged) == 24


@pytest.mark.parametrize("nx,ny", [(3, 3), (4, 2), (3, 1), (1, 3)])
def test_r2_every_ordered_pair_of_grid_faces(nx, ny):
    """R2 (mirrors M14) on open input: every ordered pair of cells of a flat
    grid gives a sound bridge (boundary unchanged, every directed edge used at
    most once) or exactly the untouched grid."""
    grid = create_plane(subdivisions_x=nx, subdivisions_y=ny)
    n = len(grid.faces)
    assert n == nx * ny and len(_boundary(grid)) == 2 * (nx + ny)
    n_bridged = 0
    for a in range(n):
        for b in range(n):
            out = bridge_faces(grid, [a], [b])
            if _same(out, grid):
                continue
            assert _sound_bridge(grid, out), (a, b, _faces(out))
            n_bridged += 1
    # both sides of the bracket: same-cell pairs are always refused, and the
    # operation is not refusing everything
    assert n_bridged <= n * n - n
    assert n_bridged > 0


def test_r2_facing_quads_on_parallel_grids_still_bridge():
    """R2 (mirrors M14) positive control: the centre cells of two parallel
    3x3 grids facing each other are genuinely bridgeable. Expected: 18 - 2 + 4
    faces, no new vertices, a sound bridge whose boundary is the two outer
    grid rims."""
    verts, faces = [], []
    for z, flip in ((0.0, False), (1.0, True)):
        off = len(verts)
        verts.extend([x, y, z] for y in range(4) for x in range(4))
        for j in range(3):
            for i in range(3):
                q = [off + 4 * j + i, off + 4 * j + i + 1,
                     off + 4 * (j + 1) + i + 1, off + 4 * (j + 1) + i]
                faces.append(q[::-1] if flip else q)
    m = HalfEdgeMesh.from_arrays(verts, faces)
    assert len(m.faces) == 18 and len(_boundary(m)) == 24

    out = bridge_faces(m, [4], [13])
    assert not _same(out, m)
    assert len(out.faces) == 20 and len(out.vertices) == 32
    assert _sound_bridge(m, out)


def test_r2_side_quad_on_an_existing_reversed_wall_is_refused():
    """R2 (mirrors M14): two facing quads plus a loose wall between them. The
    side quad over edge 0-1 is [0, 1, 5, 4]; the wall [4, 5, 1, 0] has the
    same corners wound the other way, so that quad reuses none of the wall's
    directed edges (its neighbour [1, 2, 6, 5] reuses 5 -> 1), and the old code
    glued a second face onto the wall. Expected: the untouched mesh; without
    the wall, the same request still bridges."""
    pts = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
           [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]
    two = HalfEdgeMesh.from_arrays(pts, [[0, 1, 2, 3], [7, 6, 5, 4]])
    bridged = bridge_faces(two, [0], [1])
    # precondition: the bridge builds exactly the side quad the wall reverses
    assert [0, 1, 5, 4] in _faces(bridged) and [1, 2, 6, 5] in _faces(bridged)
    assert _sound_bridge(two, bridged)

    walled = HalfEdgeMesh.from_arrays(pts, [[0, 1, 2, 3], [7, 6, 5, 4], [4, 5, 1, 0]])
    out = bridge_faces(walled, [0], [1])
    assert _same(out, walled), _faces(out)
