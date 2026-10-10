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

Test discipline (the repo's vacuous-test rules): each test first asserts that
the defect / precondition really is in its input, every checker gets an input
it must accept and one it must reject, and counts are bracketed from both
sides (what must be refused AND what must still be done).
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.halfedge_mesh import HalfEdgeMesh  # noqa: E402
from src.nurbs.converter import SubDToNURBSConverter  # noqa: E402
from src.subd.primitives import create_box  # noqa: E402


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

