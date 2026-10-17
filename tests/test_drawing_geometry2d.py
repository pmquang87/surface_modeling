"""Tests of ``src.drawing.geometry2d`` (task A2): pure 2D algorithms.

Every negative / defect test asserts its precondition first, and each
behaviour is checked so that a no-op or constant implementation fails
(repo rule, see PROJECT_MISTAKES.md and the vacuous-test discipline note).
"""
import math

import numpy as np
import pytest

from src.drawing import geometry2d as g2
from src.drawing.model import Arc, Circle, LineStyle, Polyline, Segment, Text


def _square(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)


def _seg_lengths(segs):
    segs = np.asarray(segs, float).reshape(-1, 2, 2)
    return np.linalg.norm(segs[:, 1] - segs[:, 0], axis=1)


def _same_segment(s, a, b, tol=1e-9):
    s = np.asarray(s, float)
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    fwd = np.allclose(s[0], a, atol=tol) and np.allclose(s[1], b, atol=tol)
    rev = np.allclose(s[0], b, atol=tol) and np.allclose(s[1], a, atol=tol)
    return fwd or rev


# --------------------------------------------------------------- polygon_area

def test_polygon_area_is_signed():
    sq = _square(0, 0, 10, 5)
    assert g2.polygon_area(sq) == pytest.approx(50.0)
    assert g2.polygon_area(sq[::-1]) == pytest.approx(-50.0)
    # a repeated end point does not change the area
    assert g2.polygon_area(np.vstack([sq, sq[:1]])) == pytest.approx(50.0)


# ----------------------------------------------------------- point_in_polygon

def test_point_in_polygon_interior_exterior():
    sq = _square(0, 0, 10, 10)
    assert g2.point_in_polygon((5, 5), sq) is True
    assert g2.point_in_polygon((15, 5), sq) is False
    assert g2.point_in_polygon((-1, 5), sq) is False
    assert g2.point_in_polygon((5, 11), sq) is False


@pytest.mark.parametrize("reverse", [False, True])
def test_point_in_polygon_half_open_boundary(reverse):
    """Half-open rule: left/bottom boundary inside, right/top boundary outside.

    Vertices follow from the same rule: (0,0) is inside (left AND bottom),
    the other three corners touch a right or top edge and are outside.
    """
    sq = _square(0, 0, 10, 10)
    if reverse:
        sq = sq[::-1].copy()
    # edge midpoints
    assert g2.point_in_polygon((0, 5), sq) is True      # left
    assert g2.point_in_polygon((5, 0), sq) is True      # bottom
    assert g2.point_in_polygon((10, 5), sq) is False    # right
    assert g2.point_in_polygon((5, 10), sq) is False    # top
    # vertices
    assert g2.point_in_polygon((0, 0), sq) is True
    assert g2.point_in_polygon((10, 0), sq) is False
    assert g2.point_in_polygon((10, 10), sq) is False
    assert g2.point_in_polygon((0, 10), sq) is False


def test_point_in_polygon_shared_edge_belongs_to_exactly_one():
    a = _square(0, 0, 10, 10)
    b = _square(10, 0, 20, 10)
    for y in (0.0, 2.5, 5.0, 7.5):
        p = (10.0, y)
        ina = g2.point_in_polygon(p, a)
        inb = g2.point_in_polygon(p, b)
        assert ina != inb, (p, ina, inb)
        assert inb is True                  # the shared edge is the left edge of b


def _dense_half_open(pts, poly):
    """Reference: the dense (m x k) half-open crossing test."""
    x0, y0 = poly[:, 0], poly[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    px, py = pts[:, :1], pts[:, 1:]
    crosses = (y0[None, :] > py) != (y1[None, :] > py)
    dy = np.where(y1 == y0, 1.0, y1 - y0)[None, :]
    xint = x0[None, :] + (py - y0[None, :]) * (x1 - x0)[None, :] / dy
    return (np.count_nonzero(crosses & (px < xint), axis=1) % 2) == 1


def test_scanline_containment_matches_dense_reference():
    rng = np.random.default_rng(11)
    t = np.sort(rng.uniform(0, 2 * math.pi, 300))
    rad = 10 + 3 * np.sin(5 * t)                                  # star-like loop
    poly = np.stack([rad * np.cos(t), rad * np.sin(t)], 1)
    poly = np.vstack([poly, _square(-2, -2, 2, 2)[[0, 1]]])       # plus a horizontal edge
    pts = np.vstack([rng.uniform(-14, 14, (3000, 2)), poly,       # random, vertices,
                     0.5 * (poly + np.roll(poly, -1, axis=0))])   # edge midpoints
    ref = _dense_half_open(pts, poly)
    assert 0 < ref.sum() < len(ref)                               # precondition: mixed answers
    old_block = g2._PAIR_BLOCK
    try:
        for block in (old_block, 97):                             # also many small blocks
            g2._PAIR_BLOCK = block
            assert np.array_equal(g2._points_in_polygon(pts, poly), ref)
            # boundary mask: vertices and midpoints yes; midpoints pushed off the
            # edge by 0.5 eps yes, by 10 eps no
            assert g2._on_boundary(pts, poly, 1e-9)[3000:].all()
            e = np.roll(poly, -1, axis=0) - poly
            nrm = np.stack([-e[:, 1], e[:, 0]], 1) / np.linalg.norm(e, axis=1)[:, None]
            mid = 0.5 * (poly + np.roll(poly, -1, axis=0))
            assert g2._on_boundary(mid + 0.5e-9 * nrm, poly, 1e-9).all()
            assert not g2._on_boundary(mid + 1e-8 * nrm, poly, 1e-9).any()
    finally:
        g2._PAIR_BLOCK = old_block


# ----------------------------------------------------------------- nest_loops

def test_nest_loops_by_containment_regardless_of_orientation():
    outer = _square(0, 0, 10, 10)
    inner = _square(3, 3, 7, 7)[::-1].copy()     # reversed
    other = _square(20, 0, 25, 5)
    # precondition: the input orientations differ, so sign-based nesting would be wrong
    assert np.sign(g2.polygon_area(outer)) != np.sign(g2.polygon_area(inner))
    res = g2.nest_loops([outer, inner, other])
    assert len(res) == 2
    (o1, h1), (o2, h2) = res
    assert abs(g2.polygon_area(o1)) == pytest.approx(100.0)
    assert len(h1) == 1 and abs(g2.polygon_area(h1[0])) == pytest.approx(16.0)
    assert abs(g2.polygon_area(o2)) == pytest.approx(25.0)
    assert h2 == []
    # outputs are normalised: outer CCW, holes CW
    assert g2.polygon_area(o1) > 0 and g2.polygon_area(o2) > 0
    assert g2.polygon_area(h1[0]) < 0


def test_nest_loops_island_in_hole_is_a_new_outer():
    outer = _square(0, 0, 20, 20)
    hole = _square(4, 4, 16, 16)
    island = _square(8, 8, 12, 12)
    res = g2.nest_loops([island, hole, outer])   # deliberately shuffled
    areas = sorted((abs(g2.polygon_area(o)), len(h)) for o, h in res)
    assert areas == [(16.0, 0), (400.0, 1)]


@pytest.mark.parametrize("hole", [_square(6, 2, 10, 6), _square(0, 2, 4, 6),
                                  _square(3, 6, 7, 10), _square(3, 0, 7, 4)],
                         ids=["right", "left", "top", "bottom"])
def test_nest_loops_hole_touching_the_boundary_on_any_side(hole):
    """Regression: the old half-open majority vote nested a hole touching the
    LEFT edge but returned the same hole touching the RIGHT edge as an island."""
    outer = _square(0, 0, 10, 10)
    votes = [g2.point_in_polygon(p, outer) for p in hole]
    if np.isclose(hole[:, 0].max(), 10.0) or np.isclose(hole[:, 1].max(), 10.0):
        # precondition: on the right/top side the half-open rule calls the two
        # contact vertices outside, so a plain vertex vote is a 0.5 tie
        assert sum(votes) == 2
    res = g2.nest_loops([outer, hole])
    assert len(res) == 1, [(g2.polygon_area(o), len(h)) for o, h in res]
    assert abs(g2.polygon_area(res[0][0])) == pytest.approx(100.0)
    assert len(res[0][1]) == 1 and abs(g2.polygon_area(res[0][1][0])) == pytest.approx(16.0)


def test_nest_loops_touching_outside_is_not_nested():
    """Control: a square touching the outer's right edge from OUTSIDE stays separate."""
    outer = _square(0, 0, 10, 10)
    other = _square(10, 2, 14, 6)
    assert sum(g2.point_in_polygon(p, outer) for p in other) == 0
    res = g2.nest_loops([outer, other])
    assert sorted((abs(g2.polygon_area(o)), len(h)) for o, h in res) == [(16.0, 0), (100.0, 0)]


def test_nest_loops_accepts_repeated_end_point():
    outer = _square(0, 0, 10, 10)
    closed = np.vstack([outer, outer[:1]])
    res = g2.nest_loops([closed])
    assert len(res) == 1 and len(res[0][0]) == 4


# ---------------------------------------------------------------- hatch_lines

def test_hatch_skips_hole_and_totals_area_over_spacing():
    outer = _square(0, 0, 10, 10)
    hole = _square(3, 3, 7, 7)
    segs = g2.hatch_lines([outer, hole], spacing=1.0, angle_deg=45.0)
    assert segs.ndim == 3 and segs.shape[1:] == (2, 2)
    assert len(segs) > 10
    for f in (0.25, 0.5, 0.75):
        pts = segs[:, 0] + f * (segs[:, 1] - segs[:, 0])
        for p in pts:
            assert not g2.point_in_polygon(p, hole), p
            assert g2.point_in_polygon(p, outer), p
    total = float(_seg_lengths(segs).sum())
    assert total == pytest.approx(84.0, rel=0.03)
    # every segment runs at 45 degrees
    d = segs[:, 1] - segs[:, 0]
    assert np.allclose(np.abs(d[:, 0]), np.abs(d[:, 1]), atol=1e-9)


def test_hatch_without_hole_differs():
    """Control: the hole really changes the result (not a constant output)."""
    outer = _square(0, 0, 10, 10)
    full = g2.hatch_lines([outer], spacing=1.0)
    assert float(_seg_lengths(full).sum()) == pytest.approx(100.0, rel=0.03)


def test_hatch_line_through_corners_gives_two_segments():
    outer = _square(0, 0, 10, 10)
    hole = _square(3, 3, 7, 7)
    segs = g2.hatch_lines([outer, hole], spacing=1.0, angle_deg=45.0, phase=0.0)
    # the line through the origin at 45 degrees is y = x: pick the segments on it
    on = [s for s in segs if np.allclose(s[:, 0], s[:, 1], atol=1e-9)]
    assert len(on) == 2, on
    assert any(_same_segment(s, (0, 0), (3, 3)) for s in on)
    assert any(_same_segment(s, (7, 7), (10, 10)) for s in on)


def test_hatch_phase_shifts_the_grid():
    outer = _square(0, 0, 10, 10)
    a = math.radians(45.0)
    nrm = np.array([-math.sin(a), math.cos(a)])
    for phase in (0.0, 0.3):
        segs = g2.hatch_lines([outer], spacing=1.0, angle_deg=45.0, phase=phase)
        assert len(segs) > 0
        offs = segs[:, 0] @ nrm
        frac = np.mod(offs - phase, 1.0)
        frac = np.minimum(frac, 1.0 - frac)
        assert np.all(frac < 1e-9), (phase, offs)


def test_hatch_rejects_bad_spacing_and_empty():
    with pytest.raises(ValueError):
        g2.hatch_lines([_square(0, 0, 1, 1)], spacing=0.0)
    assert g2.hatch_lines([], spacing=1.0).shape == (0, 2, 2)


# ------------------------------------------------------------- chain_segments

def _square_edges(sq):
    return [np.array([sq[i], sq[(i + 1) % 4]]) for i in range(4)]


def _two_square_segments():
    a = _square(0, 0, 10, 10)
    b = _square(20, 0, 30, 10)
    edges = _square_edges(a) + [e[::-1] for e in _square_edges(b)][::-1]
    rng = np.random.default_rng(1)
    order = rng.permutation(len(edges))
    return np.array([edges[i] for i in order])


def test_chain_segments_two_squares():
    segs = _two_square_segments()
    loops, n_open = g2.chain_segments(segs, tol=1e-6)
    assert n_open == 0
    assert len(loops) == 2
    assert sorted(len(L) for L in loops) == [4, 4]
    assert sorted(round(abs(g2.polygon_area(L)), 6) for L in loops) == [100.0, 100.0]


def test_chain_segments_missing_edge_drops_one_open_chain():
    segs = _two_square_segments()
    # precondition: the full set chains to 2 loops, and exactly one edge is removed
    assert len(g2.chain_segments(segs, tol=1e-6)[0]) == 2
    assert len(segs) == 8 and len(segs[1:]) == 7
    loops, n_open = g2.chain_segments(segs[1:], tol=1e-6)
    assert len(loops) == 1
    assert n_open == 1


def test_chain_segments_welds_within_tol_only():
    segs = _two_square_segments()
    rng = np.random.default_rng(7)
    small = segs + rng.uniform(-1e-9, 1e-9, segs.shape)
    loops, n_open = g2.chain_segments(small, tol=1e-6)
    assert len(loops) == 2 and n_open == 0
    # control: a gap far above tol must NOT be welded
    big = segs.copy()
    big[0, 0] += 1e-3
    assert np.linalg.norm(big[0, 0] - segs[0, 0]) > 1e-6     # precondition
    loops2, n_open2 = g2.chain_segments(big, tol=1e-6)
    assert len(loops2) == 1 and n_open2 == 1


def test_chain_segments_empty():
    loops, n_open = g2.chain_segments(np.zeros((0, 2, 2)), tol=1e-6)
    assert loops == [] and n_open == 0


# ---------------------------------------------------- simplify / is_straight

def test_simplify_noisy_line_to_two_points():
    rng = np.random.default_rng(3)
    x = np.linspace(0, 50, 100)
    pts = np.stack([x, 0.5 * x], 1) + rng.uniform(-1e-6, 1e-6, (100, 2))
    out = g2.simplify_polyline(pts, 1e-3)
    assert out.shape == (2, 2)
    assert np.allclose(out[0], pts[0]) and np.allclose(out[-1], pts[-1])
    assert g2.is_straight(pts, 1e-3) is True


def test_simplify_keeps_corner_and_arc_not_straight():
    pts = np.array([[0, 0], [5, 0], [10, 0], [10, 5], [10, 10]], float)
    out = g2.simplify_polyline(pts, 1e-3)
    assert out.tolist() == [[0, 0], [10, 0], [10, 10]]
    t = np.linspace(0, math.pi / 2, 50)
    arc = np.stack([10 * np.cos(t), 10 * np.sin(t)], 1)
    assert g2.is_straight(arc, 1e-3) is False
    assert len(g2.simplify_polyline(arc, 1e-3)) > 3


# ------------------------------------------------------------ merge_collinear

def test_merge_collinear_chain_and_keeps_gaps():
    segs = np.array([[[0, 0], [5, 0]], [[5, 0], [55, 0]], [[0, 0], [0, 10]]], float)
    out = g2.merge_collinear(segs, 1e-6)
    assert len(out) == 2
    assert any(_same_segment(s, (0, 0), (55, 0)) for s in out)
    assert any(_same_segment(s, (0, 0), (0, 10)) for s in out)
    # control: a gap above tol is kept apart
    gapped = np.array([[[0, 0], [20, 0]], [[40, 0], [60, 0]]], float)
    gap = gapped[1, 0, 0] - gapped[0, 1, 0]
    assert gap > 1e-6                                       # precondition: gap above tol
    assert np.allclose(gapped[:, :, 1], 0.0)                # precondition: collinear
    assert len(g2.merge_collinear(gapped, 1e-6)) == 2


def _arc_chords(r, n, sweep=math.pi / 2):
    t = np.linspace(0.0, sweep, n + 1)
    pts = np.stack([r * np.cos(t), r * np.sin(t)], 1)
    return pts, np.stack([pts[:-1], pts[1:]], 1)


def _pairwise_collinear(s, q, tol):
    """The candidate test of merge_collinear: each endpoint within tol of the other's line."""
    def off(seg, pts):
        d = seg[1] - seg[0]
        nrm = np.array([-d[1], d[0]]) / np.linalg.norm(d)
        return np.abs((pts - seg[0]) @ nrm)
    return bool(np.all(off(s, q) <= tol) and np.all(off(q, s) <= tol))


def _max_vertex_deviation(pts, segs):
    """Largest distance of the points to the nearest output segment."""
    segs = np.asarray(segs, float).reshape(-1, 2, 2)
    best = np.full(len(pts), np.inf)
    for a, b in segs:
        d = b - a
        tt = np.clip(((pts - a) @ d) / float(d @ d), 0.0, 1.0)
        best = np.minimum(best, np.linalg.norm(pts - (a + tt[:, None] * d), axis=1))
    return float(best.max())


@pytest.mark.parametrize("n, tol", [(400, 1e-3), (1000, 1e-4)])
def test_merge_collinear_does_not_straighten_a_faceted_arc(n, tol):
    """Regression: transitive grouping chained a fine quarter arc r=10 into one
    straight chord (max deviation 2.93 mm). Every input vertex must stay
    within tol of the output."""
    pts, chords = _arc_chords(10.0, n)
    # precondition: neighbouring chords pass the pairwise test, so union-find
    # really chains the whole arc into one candidate group (test not vacuous)
    assert all(_pairwise_collinear(chords[i], chords[i + 1], tol) for i in range(n - 1))
    assert _max_vertex_deviation(pts, [[pts[0], pts[-1]]]) > 2.0     # one chord would be far off
    out = g2.merge_collinear(chords, tol)
    assert _max_vertex_deviation(pts, out) <= tol * (1 + 1e-9)
    assert len(out) > 1
    # neighbouring merges are still made where they are legitimate
    assert len(out) < n


def test_merge_collinear_full_circle_and_fine_straight_run():
    tol = 1e-4
    pts, chords = _arc_chords(10.0, 4000, sweep=2 * math.pi)
    out = g2.merge_collinear(chords, tol)
    assert _max_vertex_deviation(pts, out) <= tol * (1 + 1e-9)
    # control: 200 touching pieces of one straight line (noise far below tol) still merge to one
    rng = np.random.default_rng(5)
    x = np.linspace(0.0, 100.0, 201)
    line = np.stack([x, 0.3 * x], 1) + rng.uniform(-1e-7, 1e-7, (201, 2))
    pieces = np.stack([line[:-1], line[1:]], 1)[rng.permutation(200)]
    merged = g2.merge_collinear(pieces, tol)
    assert len(merged) == 1
    assert _same_segment(merged[0], line[0], line[-1], tol=1e-12)


def test_merge_collinear_overlap_reversed_and_oblique():
    segs = np.array([[[0, 0], [10, 10]], [[15, 15], [5, 5]], [[0, 1], [10, 11]]], float)
    out = g2.merge_collinear(segs, 1e-6)
    assert len(out) == 2
    assert any(_same_segment(s, (0, 0), (15, 15)) for s in out)


# ----------------------------------------------------------- subtract_covered

def test_subtract_covered_middle_piece():
    hidden = np.array([[[0, 0], [60, 0]]], float)
    visible = np.array([[[0, 0], [20, 0]], [[60, 0], [40, 0]]], float)
    out = g2.subtract_covered(hidden, visible, 1e-6)
    assert len(out) == 1
    assert _same_segment(out[0], (20, 0), (40, 0), tol=1e-6)


def test_subtract_covered_unrelated_unchanged_and_full_cover_empty():
    hidden = np.array([[[0, 5], [60, 5]]], float)
    visible = np.array([[[0, 0], [60, 0]]], float)
    # precondition: the hidden segment is parallel but off the visible line
    assert abs(hidden[0, 0, 1] - visible[0, 0, 1]) > 1e-6
    out = g2.subtract_covered(hidden, visible, 1e-6)
    assert out.shape == (1, 2, 2) and np.allclose(out, hidden)
    full = g2.subtract_covered(np.array([[[0, 0], [60, 0]]], float),
                               np.array([[[-1, 0], [61, 0]]], float), 1e-6)
    assert full.shape == (0, 2, 2)


def test_subtract_covered_mixed_batch():
    hidden = np.array([[[0, 0], [60, 0]], [[0, 0], [0, 10]]], float)
    visible = np.array([[[10, 0], [20, 0]]], float)
    out = g2.subtract_covered(hidden, visible, 1e-6)
    assert len(out) == 3
    assert any(_same_segment(s, (0, 0), (10, 0)) for s in out)
    assert any(_same_segment(s, (20, 0), (60, 0)) for s in out)
    assert any(_same_segment(s, (0, 0), (0, 10)) for s in out)


def test_subtract_covered_tiny_hidden_dropped_whatever_visible_holds():
    tiny = np.array([[[0, 0], [5e-7, 0]], [[0, 5], [60, 5]]], float)
    assert _seg_lengths(tiny)[0] <= 1e-6 < _seg_lengths(tiny)[1]      # precondition
    unrelated = np.array([[[0, -50], [10, -50]]], float)
    for vis in (np.zeros((0, 2, 2)), unrelated):
        out = g2.subtract_covered(tiny, vis, 1e-6)
        assert out.shape == (1, 2, 2) and np.allclose(out[0], tiny[1]), vis


# ------------------------------------------------------- drop_covered_curves

def test_drop_covered_curves_far_mouth_circle():
    near = Circle((30.0, 20.0), 5.0)
    far = Circle((30.0 + 1e-9, 20.0), 5.0, LineStyle.HIDDEN)
    other = Circle((30.0, 20.0), 6.0, LineStyle.HIDDEN)
    seg = Segment((0, 0), (1, 0), LineStyle.HIDDEN)
    # precondition: dedupe keeps the far mouth (styles differ), so the helper is needed
    assert len(g2.dedupe_primitives([near, far], 1e-6)) == 2
    out = g2.drop_covered_curves([far, other, seg], [near], 1e-6)
    assert out == [other, seg]
    # control: no visible curve -> unchanged
    assert g2.drop_covered_curves([far, other, seg], [], 1e-6) == [far, other, seg]


def test_drop_covered_curves_arc_union_and_partial():
    c = (0.0, 0.0)
    hid = Arc(c, 3.0, 0.0, 180.0, LineStyle.HIDDEN)
    full = [Arc(c, 3.0, 350.0, 90.0), Arc(c, 3.0, 90.0, 200.0)]     # wraps through 0
    part = [Arc(c, 3.0, 0.0, 90.0)]
    assert g2.drop_covered_curves([hid], full, 1e-6) == []
    assert g2.drop_covered_curves([hid], part, 1e-6) == [hid]
    # a hidden full circle needs full coverage: two half arcs do, one does not
    hc = Circle(c, 3.0, LineStyle.HIDDEN)
    halves = [Arc(c, 3.0, 0.0, 180.0), Arc(c, 3.0, 180.0, 360.0)]
    assert g2.drop_covered_curves([hc], halves, 1e-6) == []
    assert g2.drop_covered_curves([hc], halves[:1], 1e-6) == [hc]


# ---------------------------------------------------------- dedupe_primitives

def test_dedupe_primitives_arcs_and_reversed_segments():
    c = (57.0, 3.0)
    a, b = (0.0, 0.0), (60.0, 0.0)
    prims = [Arc(c, 3.0, 270.0, 360.0), Arc(c, 3.0, 270.0, 360.0), Segment(a, b), Segment(b, a)]
    out = g2.dedupe_primitives(prims, 1e-6)
    assert len(out) == 2
    assert isinstance(out[0], Arc) and isinstance(out[1], Segment)


def test_dedupe_primitives_keeps_distinct():
    c = (0.0, 0.0)
    prims = [Arc(c, 3.0, 0.0, 90.0), Arc(c, 3.0, 90.0, 180.0), Arc((0.0, 0.0), 3.0, 360.0, 450.0),
             Circle(c, 3.0), Circle(c, 3.0), Circle(c, 4.0),
             Segment((0, 0), (1, 0)), Segment((0, 0), (1, 0), LineStyle.HIDDEN),
             Polyline(np.array([[0, 0], [1, 1], [2, 0]], float)),
             Polyline(np.array([[2, 0], [1, 1], [0, 0]], float)),
             Text((0, 0), "A", 3.5), Text((0, 0), "A", 3.5)]
    out = g2.dedupe_primitives(prims, 1e-6)
    # arcs: 0..90 and 360..450 are the same arc; circles: one duplicate;
    # segments differ by style; polylines reversed duplicate; Text untouched
    assert len(out) == 2 + 2 + 2 + 1 + 2


# ----------------------------------------------------- primitives_to_segments

def test_primitives_to_segments_flattens_with_deflection():
    r, defl = 10.0, 0.01
    segs = g2.primitives_to_segments([Circle((1.0, 2.0), r)], defl)
    assert segs.shape[1:] == (2, 2) and len(segs) >= 8
    mids = 0.5 * (segs[:, 0] + segs[:, 1])
    sag = r - np.linalg.norm(mids - np.array([1.0, 2.0]), axis=1)
    assert sag.max() <= defl + 1e-12
    ends = np.linalg.norm(segs.reshape(-1, 2) - np.array([1.0, 2.0]), axis=1)
    assert np.allclose(ends, r)
    assert _seg_lengths(segs).sum() == pytest.approx(2 * math.pi * r, rel=1e-3)


def test_primitives_to_segments_arc_polyline_segment():
    arc = Arc((0.0, 0.0), 5.0, 270.0, 360.0)
    segs = g2.primitives_to_segments([arc], 0.01)
    pts = segs.reshape(-1, 2)
    ang = np.degrees(np.arctan2(pts[:, 1], pts[:, 0])) % 360.0
    ang[ang < 1e-9] = 360.0
    assert ang.min() == pytest.approx(270.0) and ang.max() == pytest.approx(360.0)
    assert _seg_lengths(segs).sum() == pytest.approx(0.5 * math.pi * 5.0, rel=1e-3)
    pl = Polyline(np.array([[0, 0], [1, 0], [1, 1]], float), closed=True)
    assert len(g2.primitives_to_segments([pl], 0.01)) == 3
    pl.closed = False
    assert len(g2.primitives_to_segments([pl], 0.01)) == 2
    s = g2.primitives_to_segments([Segment((0, 0), (3, 4))], 0.01)
    assert s.shape == (1, 2, 2) and _seg_lengths(s)[0] == pytest.approx(5.0)
    assert g2.primitives_to_segments([], 0.01).shape == (0, 2, 2)
