"""Tests for src/drawing/projection.py: view table, orientation, arrangement cells.

The expected values come from the HLR probe (REPORT-hlr.md, view table
verified with labelled features on OCP 7.9.3) and ISO 5456-2 (arrangement,
REPORT-standards.md). No OCP import here: projection.py is pure numpy.
"""
import itertools

import numpy as np
import pytest

from src.drawing.projection import (
    PRINCIPAL,
    VIEWS,
    Orientation,
    ViewSpec,
    auto_orientation,
    grid_cell,
)

# Probe table: name -> (N toward the viewer, Vx), unnormalised as printed.
PROBE_TABLE = {
    "front": ((0, -1, 0), (1, 0, 0)),
    "back": ((0, 1, 0), (-1, 0, 0)),
    "top": ((0, 0, 1), (1, 0, 0)),
    "bottom": ((0, 0, -1), (1, 0, 0)),
    "right": ((1, 0, 0), (0, 1, 0)),
    "left": ((-1, 0, 0), (0, -1, 0)),
    "iso": ((1, -1, 1), (1, 1, 0)),
}

P = np.array([[1.0, 2.0, 3.0]])

# 2D (x, y) of the point (X, Y, Z) = (1, 2, 3) per the probe's "2D =" column,
# depth = dot(p, n).
EXPECTED_VIEW_OF_P = {
    "front": (1.0, 3.0, -2.0),   # (X, Z)
    "back": (-1.0, 3.0, 2.0),    # (-X, Z)
    "top": (1.0, 2.0, 3.0),      # (X, Y)
    "bottom": (1.0, -2.0, -3.0), # (X, -Y)
    "right": (2.0, 3.0, 1.0),    # (Y, Z)
    "left": (-2.0, 3.0, -1.0),   # (-Y, Z)
}


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


# ------------------------------------------------------------------ view table

def test_views_has_exactly_the_seven_names():
    assert set(VIEWS) == set(PROBE_TABLE)
    assert PRINCIPAL == ("front", "back", "top", "bottom", "right", "left")
    for name, spec in VIEWS.items():
        assert isinstance(spec, ViewSpec)
        assert spec.name == name


@pytest.mark.parametrize("name", sorted(PROBE_TABLE))
def test_view_axes_orthonormal_right_handed(name):
    s = VIEWS[name]
    for v in (s.n, s.x, s.y):
        assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-12)
    assert abs(np.dot(s.n, s.x)) < 1e-12
    assert abs(np.dot(s.n, s.y)) < 1e-12
    assert abs(np.dot(s.x, s.y)) < 1e-12
    np.testing.assert_allclose(s.y, np.cross(s.n, s.x), atol=1e-12)
    # right-handed frame (x, y, n): det [x y n] = +1
    assert np.linalg.det(np.stack([s.x, s.y, s.n], axis=1)) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("name", sorted(PROBE_TABLE))
def test_table_equals_probe(name):
    n, x = PROBE_TABLE[name]
    np.testing.assert_allclose(VIEWS[name].n, _unit(n), atol=1e-12)
    np.testing.assert_allclose(VIEWS[name].x, _unit(x), atol=1e-12)


@pytest.mark.parametrize("name", sorted(EXPECTED_VIEW_OF_P))
def test_to_view_of_point_matches_table(name):
    got = VIEWS[name].to_view(P)
    assert got.shape == (1, 3)
    np.testing.assert_allclose(got[0], EXPECTED_VIEW_OF_P[name], atol=1e-12)


def test_to_view_vectorised_and_depth_nearer_is_larger():
    pts = np.array([[0.0, -10.0, 0.0], [0.0, 10.0, 0.0], [5.0, 0.0, 7.0]])
    v = VIEWS["front"].to_view(pts)
    assert v.shape == (3, 3)
    # front viewer sits at -Y: the y=-10 point is nearer -> larger depth
    assert v[0, 2] > v[1, 2]
    np.testing.assert_allclose(v[2], (5.0, 7.0, 0.0), atol=1e-12)
    # single (3,) point is accepted and returns (1, 3)
    assert VIEWS["front"].to_view(np.array([1.0, 2.0, 3.0])).shape == (1, 3)


def test_iso_z_projects_upward():
    s = VIEWS["iso"]
    assert s.y[2] > 0
    np.testing.assert_allclose(s.y, (-0.408, 0.408, 0.816), atol=1e-3)
    # the world +Z axis maps to a positive 2D y and zero 2D x
    v = s.to_view(np.array([[0.0, 0.0, 1.0]]))[0]
    assert v[0] == pytest.approx(0.0, abs=1e-12)
    assert v[1] > 0.8


def test_gp_ax2_args_plain_float_tuples():
    for name, spec in VIEWS.items():
        origin, n, x = spec.gp_ax2_args()
        assert origin == (0.0, 0.0, 0.0)
        for t in (origin, n, x):
            assert isinstance(t, tuple) and len(t) == 3
            assert all(type(c) is float for c in t)
        np.testing.assert_allclose(n, spec.n, atol=0)
        np.testing.assert_allclose(x, spec.x, atol=0)


def test_viewspec_arrays_are_read_only():
    s = VIEWS["front"]
    with pytest.raises(ValueError):
        s.n[0] = 5.0
    # the shared table cannot be corrupted through the returned axis
    assert s.n[1] == pytest.approx(-1.0)


def test_viewspec_rejects_non_orthogonal_axes():
    with pytest.raises(ValueError):
        ViewSpec("bad", np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 1.0]))
    with pytest.raises(ValueError):
        ViewSpec("zero", np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))


# ---------------------------------------------------------------- orientation

def _bbox_corners(extents):
    ex = np.asarray(extents, float)
    return np.array(list(itertools.product(*[(0.0, e) for e in ex])))


def _extents_after(o, extents):
    pts = o.apply(_bbox_corners(extents))
    return pts.max(axis=0) - pts.min(axis=0)


def test_auto_orientation_identity_case_from_plan():
    # NOTE: the plan's extents (60, 20, 40) are already in target order, so
    # this test alone passes against a constant identity implementation. The
    # permutation itself is checked by test_auto_orientation_every_permutation,
    # test_auto_orientation_odd_permutation_needs_flip and
    # test_fixed_mode_is_identity_even_for_permuted_extents. The guard below
    # makes this test fail against identity as well.
    assert not np.allclose(auto_orientation(np.array([20.0, 60.0, 40.0])).matrix, np.eye(3))
    o = auto_orientation(np.array([60.0, 20.0, 40.0]))
    assert isinstance(o, Orientation)
    assert o.mode == "auto"
    assert np.linalg.det(o.matrix) == pytest.approx(1.0)
    # 60 axis (world X) -> X, 20 axis (world Y) -> Y
    np.testing.assert_allclose(o.matrix @ [1, 0, 0], [1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(np.abs(o.matrix @ [0, 1, 0]), [0, 1, 0], atol=1e-12)
    np.testing.assert_allclose(_extents_after(o, (60, 20, 40)), (60, 20, 40), atol=1e-9)


@pytest.mark.parametrize("extents", list(itertools.permutations((60.0, 20.0, 40.0))))
def test_auto_orientation_every_permutation(extents):
    o = auto_orientation(np.array(extents))
    M = o.matrix
    assert M.shape == (3, 3)
    assert np.linalg.det(M) == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_allclose(M @ M.T, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(_extents_after(o, extents), (60, 20, 40), atol=1e-9)
    # permutation / flips describe the matrix: row i = flips[i] * e_{perm[i]}
    for i in range(3):
        row = np.zeros(3)
        row[o.permutation[i]] = o.flips[i]
        np.testing.assert_allclose(M[i], row, atol=0)
    assert extents[o.permutation[0]] == 60.0
    assert extents[o.permutation[1]] == 20.0
    assert extents[o.permutation[2]] == 40.0
    assert all(f in (1, -1) for f in o.flips)


def test_auto_orientation_odd_permutation_needs_flip():
    # (40, 20, 60): the needed axis order (2, 1, 0) is an odd permutation, so a
    # plain permutation matrix would have det -1 (a mirror); one axis must flip.
    ext = (40.0, 20.0, 60.0)
    o = auto_orientation(np.array(ext))
    assert o.permutation == (2, 1, 0)
    pm = np.zeros((3, 3))
    pm[[0, 1, 2], list(o.permutation)] = 1.0
    assert np.linalg.det(pm) == pytest.approx(-1.0)   # precondition: odd
    assert -1 in o.flips
    assert np.linalg.det(o.matrix) == pytest.approx(1.0)
    np.testing.assert_allclose(_extents_after(o, ext), (60, 20, 40), atol=1e-9)


def test_auto_orientation_preserves_distances_and_maps_points():
    ext = (10.0, 50.0, 30.0)
    o = auto_orientation(np.array(ext))
    pts = np.array([[1.0, 2.0, 3.0], [-4.0, 5.0, 0.5]])
    q = o.apply(pts)
    np.testing.assert_allclose(q, pts @ o.matrix.T, atol=1e-12)
    assert np.linalg.norm(q[0] - q[1]) == pytest.approx(np.linalg.norm(pts[0] - pts[1]))
    # a single point (3,) keeps its shape
    assert o.apply(np.array([1.0, 2.0, 3.0])).shape == (3,)
    # the 50 axis (world Y) becomes X
    np.testing.assert_allclose(np.abs(o.apply(np.array([0.0, 1.0, 0.0]))), [1, 0, 0], atol=1e-12)


def test_auto_orientation_ties_keep_identity():
    # Positive tie-break test. Ties alone would pass against a constant
    # identity, so first check that a tie with the wrong order still permutes
    # (10, 30, 30): 30 must move to X, so this is not identity.
    o_tie = auto_orientation(np.array([10.0, 30.0, 30.0]))
    assert not np.allclose(o_tie.matrix, np.eye(3))
    np.testing.assert_allclose(_extents_after(o_tie, (10, 30, 30)), (30, 10, 30), atol=1e-9)
    for ext in ((10.0, 10.0, 10.0), (30.0, 10.0, 30.0), (30.0, 10.0, 10.0)):
        o = auto_orientation(np.array(ext))
        np.testing.assert_allclose(o.matrix, np.eye(3), atol=0)
        assert o.permutation == (0, 1, 2) and o.flips == (1, 1, 1)


def test_fixed_mode_is_identity_even_for_permuted_extents():
    ext = np.array([20.0, 60.0, 40.0])
    # precondition: auto would NOT be identity for these extents
    assert not np.allclose(auto_orientation(ext).matrix, np.eye(3))
    o = auto_orientation(ext, mode="fixed")
    assert o.mode == "fixed"
    np.testing.assert_allclose(o.matrix, np.eye(3), atol=0)
    assert o.permutation == (0, 1, 2) and o.flips == (1, 1, 1)


def test_up_y_rotation():
    o = auto_orientation(np.array([60.0, 40.0, 20.0]), mode="fixed", up="y")
    np.testing.assert_allclose(o.apply(np.array([0.0, 1.0, 0.0])), [0, 0, 1], atol=1e-12)
    np.testing.assert_allclose(o.apply(np.array([0.0, 0.0, 1.0])), [0, -1, 0], atol=1e-12)
    np.testing.assert_allclose(o.apply(np.array([1.0, 0.0, 0.0])), [1, 0, 0], atol=1e-12)
    assert np.linalg.det(o.matrix) == pytest.approx(1.0)
    # mode "y_up" is the same as fixed + up="y"
    o2 = auto_orientation(np.array([60.0, 40.0, 20.0]), mode="y_up")
    assert o2.mode == "y_up"
    np.testing.assert_allclose(o2.matrix, o.matrix, atol=0)


def test_up_y_then_auto_uses_rotated_extents():
    # File extents (60, 20, 40). The y-up rotation (x, y, z) -> (x, -z, y)
    # gives (60, 40, 20); auto then sends 60 to X, 20 to Y and 40 to Z.
    ext = (60.0, 20.0, 40.0)
    o = auto_orientation(np.array(ext), mode="auto", up="y")
    assert np.linalg.det(o.matrix) == pytest.approx(1.0)
    np.testing.assert_allclose(_extents_after(o, ext), (60, 20, 40), atol=1e-9)
    # file Y (the up axis, extent 20) ends up as depth here because it is the
    # shortest; auto overrides the up hint by design.
    assert o.permutation[1] == 1


def test_orientation_to_dict_for_report():
    o = auto_orientation(np.array([20.0, 60.0, 40.0]))
    d = o.to_dict()
    assert d["mode"] == "auto"
    assert d["permutation"] == list(o.permutation)
    assert d["flips"] == list(o.flips)
    assert all(type(v) is int for v in d["permutation"] + d["flips"])


@pytest.mark.parametrize("kw", [dict(mode="bogus"), dict(up="x")])
def test_auto_orientation_rejects_bad_args(kw):
    with pytest.raises(ValueError):
        auto_orientation(np.array([1.0, 2.0, 3.0]), **kw)


def test_auto_orientation_rejects_bad_extents():
    with pytest.raises(ValueError):
        auto_orientation(np.array([1.0, 2.0]))
    with pytest.raises(ValueError):
        auto_orientation(np.array([1.0, np.nan, 3.0]))


def test_auto_orientation_rejects_negative_extents():
    # precondition: the positive version is accepted
    auto_orientation(np.array([60.0, 20.0, 40.0]))
    with pytest.raises(ValueError, match="non-negative"):
        auto_orientation(np.array([60.0, -20.0, 40.0]))
    # zero extent (a flat part) stays legal
    auto_orientation(np.array([60.0, 0.0, 40.0]))


def test_viewspec_equality_by_value():
    import pickle
    front = VIEWS["front"]
    rebuilt = ViewSpec("front", np.array([0.0, -2.0, 0.0]), np.array([3.0, 0.0, 0.0]))
    assert rebuilt is not front                       # precondition: distinct objects
    assert rebuilt == front and hash(rebuilt) == hash(front)
    assert pickle.loads(pickle.dumps(front)) == front
    assert rebuilt in {front}
    # different axes or name are not equal
    assert VIEWS["back"] != front
    assert ViewSpec("other", front.n, front.x) != front
    assert ViewSpec("front", front.n, np.array([-1.0, 0.0, 0.0])) != front
    assert len(set(VIEWS.values())) == 7


def test_orientation_equality_by_value():
    import pickle
    a = auto_orientation(np.array([20.0, 60.0, 40.0]))
    b = auto_orientation(np.array([20.0, 60.0, 40.0]))
    assert a is not b                                 # precondition
    assert a == b and hash(a) == hash(b)
    assert pickle.loads(pickle.dumps(a)) == a
    assert a != auto_orientation(np.array([60.0, 20.0, 40.0]))
    assert a != auto_orientation(np.array([20.0, 60.0, 40.0]), mode="fixed")


# ------------------------------------------------------------------ grid cells

FIRST = {"front": (0, 0), "top": (0, -1), "bottom": (0, 1), "right": (-1, 0),
         "left": (1, 0), "back": (2, 0), "iso": (1, 1)}
THIRD = {"front": (0, 0), "top": (0, 1), "bottom": (0, -1), "right": (1, 0),
         "left": (-1, 0), "back": (-2, 0), "iso": (1, 1)}


def test_grid_cell_plan_cases():
    assert grid_cell("top", "first") == (0, -1)
    assert grid_cell("top", "third") == (0, 1)
    assert grid_cell("right", "first") == (-1, 0)
    assert grid_cell("right", "third") == (1, 0)


@pytest.mark.parametrize("view", sorted(FIRST))
def test_grid_cell_full_table(view):
    assert grid_cell(view, "first") == FIRST[view]
    assert grid_cell(view, "third") == THIRD[view]
    # long names accepted (the report spells them first_angle / third_angle)
    assert grid_cell(view, "first_angle") == FIRST[view]
    assert grid_cell(view, "third_angle") == THIRD[view]


def test_grid_cells_distinct_and_mirror():
    for table in (FIRST, THIRD):
        assert len(set(table.values())) == len(table)
    # principal cells of third angle are the first-angle cells mirrored through F
    for v in PRINCIPAL:
        c1, c3 = grid_cell(v, "first"), grid_cell(v, "third")
        assert c3 == (-c1[0], -c1[1])
    # iso sits in a cell no default principal view takes in either projection
    for proj in ("first", "third"):
        taken = {grid_cell(v, proj) for v in ("front", "top", "right")}
        assert grid_cell("iso", proj) not in taken


@pytest.mark.parametrize("args", [("side", "first"), ("top", "second"), ("top", "")])
def test_grid_cell_rejects_unknown(args):
    with pytest.raises(ValueError):
        grid_cell(*args)
