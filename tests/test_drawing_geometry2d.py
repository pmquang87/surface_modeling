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


# ============================================ QA round 1 fixes (task F1-views-perf)
# Issue 3: merge_collinear / subtract_covered were quadratic (175k poly-HLR chords:
# merge 338 s, subtract 128 s); issue 6: chain_paths; issue 27: loose subtraction.

def _synthetic_hlr_like(n, seed):
    """Poly-HLR-like hidden set: faceted arcs, straight edges split twice at different
    parameters (overlapping collinear copies) and random short fragments."""
    rng = np.random.default_rng(seed)
    out, total = [], 0
    while total < n:
        kind = rng.random()
        if kind < 0.5:
            c = rng.uniform(0, 200, 2)
            r = rng.uniform(0.5, 20.0)
            t = rng.uniform(0, 2 * math.pi) + np.linspace(0, rng.uniform(0.3, 2 * math.pi),
                                                         int(rng.integers(16, 400)) + 1)
            pts = np.stack([c[0] + r * np.cos(t), c[1] + r * np.sin(t)], 1)
            segs = np.stack([pts[:-1], pts[1:]], 1)
        elif kind < 0.8:
            a = rng.uniform(0, 200, 2)
            ang = rng.uniform(0, math.pi) if rng.random() < 0.6 else float(rng.choice([0.0, math.pi / 2]))
            L = rng.uniform(1, 60)
            u = np.array([math.cos(ang), math.sin(ang)])
            m = int(rng.integers(2, 60))
            segs = []
            for _ in range(2):
                cuts = np.r_[0.0, np.sort(rng.uniform(0, L, m)), L]
                p = a + cuts[:, None] * u
                segs.append(np.stack([p[:-1], p[1:]], 1))
            segs = np.concatenate(segs)
        else:
            a = rng.uniform(0, 200, (200, 2))
            ang = rng.uniform(0, 2 * math.pi, 200)
            L = rng.uniform(0.01, 1.0, 200)
            segs = np.stack([a, a + L[:, None] * np.stack([np.cos(ang), np.sin(ang)], 1)], 1)
        out.append(segs)
        total += len(segs)
    s = np.concatenate(out)[:n]
    return s[rng.permutation(len(s))]


def _brute_merge_reference(segs, tol):
    """merge_collinear with the candidate pairs found by an O(n^2) scan (no KD index)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    segs = g2.premerge_collinear(segs, tol)
    segs = segs[_seg_lengths(segs) > tol]
    n = len(segs)
    d = segs[:, 1] - segs[:, 0]
    L = np.linalg.norm(d, axis=1)
    u = d / L[:, None]
    nrm = np.stack([-u[:, 1], u[:, 0]], 1)
    I, J = np.triu_indices(n, 1)
    dj = np.abs(np.einsum("mpk,mk->mp", segs[J] - segs[I, 0][:, None], nrm[I]))
    di = np.abs(np.einsum("mpk,mk->mp", segs[I] - segs[J, 0][:, None], nrm[J]))
    tj = np.einsum("mpk,mk->mp", segs[J] - segs[I, 0][:, None], u[I])
    ok = (dj.max(1) <= tol) & (di.max(1) <= tol) & (tj.max(1) >= -tol) & (tj.min(1) <= L[I] + tol)
    I, J = I[ok], J[ok]
    _, lab = connected_components(coo_matrix((np.ones(len(I)), (I, J)), shape=(n, n)), directed=False)
    out = []
    for c in np.unique(lab):
        members = np.nonzero(lab == c)[0]
        if len(members) == 1:
            out.append(segs[members[0]])
        else:
            out.extend(g2._merge_group(segs, members, L, tol))
    return np.asarray(out).reshape(-1, 2, 2)


def _brute_subtract_reference(hid, vis, tol):
    """The pre-fix O(H V) subtract_covered, kept verbatim as the reference."""
    hid = hid[_seg_lengths(hid) > tol]
    vis = vis[_seg_lengths(vis) > tol]
    out = []
    for h in hid:
        a, b = h
        L = float(np.linalg.norm(b - a))
        uu = (b - a) / L
        nn = np.array([-uu[1], uu[0]])
        on = np.all(np.abs((vis - a) @ nn) <= tol, axis=1)
        if not on.any():
            out.append(h)
            continue
        iv = np.sort((vis[on] - a) @ uu, axis=1)
        iv = iv[(iv[:, 1] > 0.0) & (iv[:, 0] < L)]
        iv = iv[np.argsort(iv[:, 0])]
        cur, pieces = 0.0, []
        for s, e in iv:
            if s > cur:
                pieces.append((cur, min(s, L)))
            cur = max(cur, e)
            if cur >= L:
                break
        if cur < L:
            pieces.append((cur, L))
        out.extend(np.array([a + s * uu, a + e * uu]) for s, e in pieces if e - s > tol)
    return np.asarray(out).reshape(-1, 2, 2)


def _as_set(segs, nd=9):
    """Order- and direction-independent set of rounded segments."""
    out = set()
    for s in np.asarray(segs).reshape(-1, 2, 2):
        a, b = tuple(np.round(s[0], nd)), tuple(np.round(s[1], nd))
        out.add((a, b) if a <= b else (b, a))
    return out


def test_merge_collinear_matches_brute_force_candidates_on_dense_subset():
    """The KD candidate index must find every touching collinear pair the O(n^2) scan finds."""
    tol = 0.0025
    segs = _synthetic_hlr_like(2000, seed=11) * 0.1     # squeezed to 20 x 20 mm: dense
    ref = _brute_merge_reference(segs, tol)
    assert len(ref) < 0.95 * len(segs)                  # precondition: many merges happen
    out = g2.merge_collinear(segs, tol)
    assert _as_set(out) == _as_set(ref)


def test_subtract_covered_matches_old_algorithm_on_subset():
    tol = 0.0025
    hid = _synthetic_hlr_like(2000, seed=3)
    rng = np.random.default_rng(4)
    vis = np.concatenate([hid[rng.choice(len(hid), 600, replace=False)],
                          _synthetic_hlr_like(400, seed=5)])
    ref = _brute_subtract_reference(hid, vis, tol)
    assert len(ref) < len(hid)                          # precondition: something is cut
    out = g2.subtract_covered(hid, vis, tol)
    assert _as_set(out) == _as_set(ref)


def test_cleanup_50k_segments_runs_fast():
    """Issue 3 regression: before the fix this set took merge 12.0 s + subtract 10.8 s."""
    import time
    from scipy.spatial import cKDTree
    tol = 0.0025
    hid = _synthetic_hlr_like(50_000, seed=0)
    rng = np.random.default_rng(1)
    vis = np.concatenate([hid[rng.choice(len(hid), 2500, replace=False)],
                          _synthetic_hlr_like(2500, seed=2)])
    g2.merge_collinear(hid[:50], tol)                   # warm the scipy imports
    t = time.perf_counter()
    mv = g2.merge_collinear(vis, tol)
    sub = g2.subtract_covered(hid, mv, tol)
    out = g2.merge_collinear(sub, tol)
    dt = time.perf_counter() - t
    assert len(sub) < len(hid) and len(out) < len(sub)  # real work was done
    assert dt < 5.0, dt
    # merged segments keep original extreme endpoints: every output vertex is an input vertex
    d, _ = cKDTree(sub.reshape(-1, 2)).query(out.reshape(-1, 2))
    assert d.max() <= 1e-9


def test_merge_and_subtract_honour_deadline():
    import time
    segs = _synthetic_hlr_like(5000, seed=7)
    past = time.perf_counter() - 1.0
    with pytest.raises(TimeoutError):
        g2.merge_collinear(segs, 0.0025, deadline=past)
    with pytest.raises(TimeoutError):
        g2.subtract_covered(segs, segs[:100], 0.0025, deadline=past)
    with pytest.raises(TimeoutError):
        g2.chain_paths(_arc_chords(10.0, 20000)[1], 1e-6, deadline=past)
    fut = time.perf_counter() + 600.0                   # a future deadline changes nothing
    assert _as_set(g2.merge_collinear(segs, 0.0025, deadline=fut)) == _as_set(g2.merge_collinear(segs, 0.0025))


# ---------------------------------------------------------------- chain_paths

def test_chain_paths_shuffled_reversed_chords_give_one_open_polyline():
    pts, chords = _arc_chords(10.0, 48)
    rng = np.random.default_rng(0)
    shuffled = chords[rng.permutation(48)].copy()
    flip = rng.random(48) < 0.5
    shuffled[flip] = shuffled[flip][:, ::-1]
    opened, closed = g2.chain_paths(shuffled, 1e-6)
    assert closed == [] and len(opened) == 1
    path = opened[0]
    assert len(path) == 49
    if not np.allclose(path[0], pts[0]):
        path = path[::-1]
    assert np.allclose(path, pts, atol=1e-9)


def test_chain_paths_closed_loop_junction_and_weld():
    sq = _square(0, 0, 10, 10)
    loop = np.stack([sq, np.roll(sq, -1, axis=0)], 1)
    opened, closed = g2.chain_paths(loop, 1e-6)
    assert opened == [] and len(closed) == 1 and len(closed[0]) == 4
    # T junction at (5, 0): three open paths meeting there, none crossing it
    tee = np.array([[[0, 0], [5, 0]], [[5, 0], [10, 0]], [[5, 0], [5, 5]]], float)
    opened, closed = g2.chain_paths(tee, 1e-6)
    assert closed == [] and len(opened) == 3
    assert all(np.any(np.all(np.isclose(p[[0, -1]], [5, 0]), axis=1)) for p in opened)
    # gap of 1e-4: welded at tol 1e-3, two paths at tol 1e-5
    gap = np.array([[[0, 0], [5, 0]], [[5.0001, 0], [9, 3]]], float)
    assert len(g2.chain_paths(gap, 1e-3)[0]) == 1
    assert len(g2.chain_paths(gap, 1e-5)[0]) == 2
    # duplicate and zero-length edges ignored
    dup = np.array([[[0, 0], [1, 0]], [[1, 0], [0, 0]], [[2, 2], [2, 2]]], float)
    opened, closed = g2.chain_paths(dup, 1e-6)
    assert len(opened) == 1 and len(opened[0]) == 2 and closed == []
    assert g2.chain_paths(np.zeros((0, 2, 2)), 1e-6) == ([], [])
    with pytest.raises(ValueError):
        g2.chain_paths(loop, 0.0)


def test_chain_paths_loop_hanging_on_a_junction_is_closed():
    sq = _square(0, 0, 10, 10)
    loop = np.stack([sq, np.roll(sq, -1, axis=0)], 1)
    tail = np.array([[[0, 0], [-5, -5]]], float)
    opened, closed = g2.chain_paths(np.concatenate([loop, tail]), 1e-6)
    assert len(opened) == 1 and len(opened[0]) == 2
    assert len(closed) == 1 and len(closed[0]) == 4


# --------------------------------------------------- subtract_covered loose mode

def test_chain_to_primitives_polyline_for_short_chords_segments_for_long_edges():
    pts, chords = _arc_chords(10.0, 48)                  # chords 0.33 long
    sq = _square(20, 0, 40, 20)
    box = np.stack([sq, np.roll(sq, -1, axis=0)], 1)     # edges 20 long
    stub = np.array([[[50.0, 0.0], [50.0, 0.1]]])        # 0.1 long, isolated
    out = g2.chain_to_primitives(np.concatenate([chords, box, stub]), LineStyle.HIDDEN, 1e-6,
                                 short_len=3.75)
    polys = [p for p in out if isinstance(p, Polyline)]
    segs = [p for p in out if isinstance(p, Segment)]
    assert len(polys) == 1 and len(polys[0].points) == 49 and not polys[0].closed
    assert len(segs) == 5                                 # 4 box edges + the stub
    assert all(p.style == LineStyle.HIDDEN for p in out)
    # min_chain_len drops the stub only
    out2 = g2.chain_to_primitives(np.concatenate([chords, box, stub]), LineStyle.HIDDEN, 1e-6,
                                  short_len=3.75, min_chain_len=0.5)
    assert len(out2) == len(out) - 1
    # a closed ring of short chords is one closed Polyline
    ring_pts = np.stack([np.cos(np.linspace(0, 2 * np.pi, 41)[:-1]),
                         np.sin(np.linspace(0, 2 * np.pi, 41)[:-1])], 1)
    ring = np.stack([ring_pts, np.roll(ring_pts, -1, axis=0)], 1)
    out3 = g2.chain_to_primitives(ring, LineStyle.HIDDEN, 1e-6, short_len=3.75)
    assert len(out3) == 1 and isinstance(out3[0], Polyline) and out3[0].closed


def test_drop_isolated_fragments():
    long_line = Segment((0.0, 0.0), (10.0, 0.0))
    isolated = Segment((20.0, 5.0), (20.1, 5.0))
    on_line = Segment((5.0, 0.0), (5.0, 0.1))            # T onto the long line's interior
    tiny_loop = Polyline(_square(30, 0, 30.1, 0.1), closed=True)
    arc_frag = Arc((40.0, 40.0), 1.0, 0.0, 5.0)          # 0.087 long, isolated
    prims = [long_line, isolated, on_line, tiny_loop, arc_frag]
    assert math.dist(isolated.a, isolated.b) < 0.3        # precondition: short
    kept, n = g2.drop_isolated_fragments(prims, 0.3, 0.01, 0.01)
    assert n == 2 and isolated not in kept and arc_frag not in kept
    assert long_line in kept and on_line in kept and tiny_loop in kept
    # a context primitive touching the fragment keeps it
    ctx = [Segment((20.1, 5.0), (25.0, 9.0), LineStyle.HIDDEN)]
    kept2, n2 = g2.drop_isolated_fragments(prims, 0.3, 0.01, 0.01, context=ctx)
    assert n2 == 1 and isolated in kept2
    assert g2.drop_isolated_fragments(prims, 0.0, 0.01, 0.01) == (prims, 0)


def test_subtract_covered_loose_removes_tilted_copy_only():
    """Issue 27: a short hidden chord lying within tol on a long visible edge but tilted
    by 0.0016 rad survives the strict test (the visible endpoints are 0.06 off the
    chord's line) and not the loose one; a hidden line crossing the visible edge
    (its midpoint ON the edge) survives both."""
    tol = 0.02
    vis = np.array([[[0.0, 0.0], [100.0, 0.0]]])
    tilted = np.array([[[40.0, 0.004], [45.0, 0.012]]])
    cross = np.array([[[50.0, -5.0], [50.0, 5.0]]])
    hid = np.concatenate([tilted, cross])
    # precondition: the far visible endpoint is off the tilted chord's line by more than tol
    d = tilted[0, 1] - tilted[0, 0]
    nrm = np.array([-d[1], d[0]]) / np.linalg.norm(d)
    assert abs((vis[0, 0] - tilted[0, 0]) @ nrm) > tol
    strict = g2.subtract_covered(hid, vis, tol)
    assert len(strict) == 2
    loose = g2.subtract_covered(hid, vis, tol, loose=True)
    assert len(loose) == 1 and np.allclose(loose[0], cross[0])
    # a hidden copy split across two visible chords meeting at a slight bend is dropped
    # by the ends+midpoint probe (no single visible chord covers it)
    bent = np.array([[[0.0, 0.0], [10.0, 0.0]], [[10.0, 0.0], [20.0, 0.01]]])
    span = np.array([[[6.0, 0.003], [14.0, 0.006]]])
    assert len(g2.subtract_covered(span, bent, tol, loose=True)) == 0


def test_subtract_covered_loose_keeps_short_chords_crossing_a_visible_edge():
    """F1 review: loose mode accepted a hidden chord whose two ends lie within tol of the
    visible line at ANY angle, so short chords crossing a visible edge were removed whole
    (0.5 mm at 90 deg, 1.0 mm at 30 deg with tol 0.3). Only near-parallel (<= 15 deg)
    pairs may cover."""
    tol = 0.3
    vis = np.array([[[0.0, 0.0], [100.0, 0.0]]])
    a30 = math.radians(30.0)
    u30 = np.array([math.cos(a30), math.sin(a30)])
    c30 = np.array([[[50.0, 0.0] - 0.5 * u30, [50.0, 0.0] + 0.5 * u30]])     # 1.0 long, 30 deg
    c90 = np.array([[[20.0, -0.25], [20.0, 0.25]]])                          # 0.5 long, 90 deg
    hid = np.concatenate([c30, c90])
    # precondition: both ends of both chords lie within tol of the visible line, and the
    # chords are longer than tol (so the length floor does not drop them)
    assert np.all(np.abs(hid[:, :, 1]) <= tol) and np.all(_seg_lengths(hid) > tol)
    out = g2.subtract_covered(hid, vis, tol, loose=True)
    assert len(out) == 2
    assert _seg_lengths(out).sum() == pytest.approx(1.5, abs=1e-9)
    # a near-parallel copy (5 deg) at the same place is still covered
    a5 = math.radians(5.0)
    u5 = np.array([math.cos(a5), math.sin(a5)])
    c5 = np.array([[[50.0, 0.0] - 1.5 * u5, [50.0, 0.0] + 1.5 * u5]])
    assert np.all(np.abs(c5[:, :, 1]) <= tol)                                # precondition
    assert len(g2.subtract_covered(c5, vis, tol, loose=True)) == 0


def _seg_dist_matrix(a, b):
    """Exact segment-segment distance (2D, no intersections assumed away) by brute force."""
    def pt_seg(p, s0, s1):
        d = s1 - s0
        dd = np.einsum("...k,...k->...", d, d)
        t = np.clip(np.einsum("...k,...k->...", p - s0, d) / np.where(dd > 0, dd, 1.0), 0, 1)
        return np.linalg.norm(p - (s0 + t[..., None] * d), axis=-1)
    A0, A1 = a[:, None, 0], a[:, None, 1]
    B0, B1 = b[None, :, 0], b[None, :, 1]
    m = np.minimum.reduce([pt_seg(A0, B0, B1), pt_seg(A1, B0, B1), pt_seg(B0, A0, A1), pt_seg(B1, A0, A1)])
    # proper crossings have distance 0
    def orient(p, q, r):
        return np.sign((q[..., 0] - p[..., 0]) * (r[..., 1] - p[..., 1])
                       - (q[..., 1] - p[..., 1]) * (r[..., 0] - p[..., 0]))
    x = (orient(A0, A1, B0) * orient(A0, A1, B1) < 0) & (orient(B0, B1, A0) * orient(B0, B1, A1) < 0)
    return np.where(x, 0.0, m)


def test_near_pairs_is_a_superset_with_mixed_segment_lengths():
    """The spacing-class radii of _near_pairs (F1 review perf follow-up) must still find
    every pair within r: short chords (0.02-0.3) next to long edges (5-40), r = 1.5."""
    rng = np.random.default_rng(11)
    def rand_segs(n, lo, hi):
        p = rng.uniform(0, 30, (n, 2))
        ang = rng.uniform(0, np.pi, n)
        ln = rng.uniform(lo, hi, n)
        return np.stack([p, p + ln[:, None] * np.stack([np.cos(ang), np.sin(ang)], 1)], 1)
    a = np.concatenate([rand_segs(300, 0.02, 0.3), rand_segs(20, 5.0, 40.0)])
    b = np.concatenate([rand_segs(200, 0.02, 0.3), rand_segs(15, 5.0, 40.0)])
    r = 1.5
    dist = _seg_dist_matrix(a, b)
    want = set(zip(*np.nonzero(dist <= r)))
    assert 50 < len(want) < dist.size                                       # precondition
    I, J = g2._near_pairs(a, b, r)
    assert want <= set(zip(I.tolist(), J.tolist()))
    # self pairs (b is None), i < j
    dself = _seg_dist_matrix(a, a)
    want_self = {(i, j) for i, j in zip(*np.nonzero(dself <= r)) if i < j}
    I, J = g2._near_pairs(a, None, r)
    assert np.all(I < J) and want_self <= set(zip(I.tolist(), J.tolist()))
    # max_sin keeps exactly the near-parallel subset
    I, J = g2._near_pairs(a, b, r, max_sin=g2._LOOSE_SIN)
    da, db = a[:, 1] - a[:, 0], b[:, 1] - b[:, 0]
    cross = np.abs(np.outer(da[:, 0], db[:, 1]) - np.outer(da[:, 1], db[:, 0]))
    par = cross <= g2._LOOSE_SIN * np.outer(np.linalg.norm(da, axis=1), np.linalg.norm(db, axis=1))
    want_par = {p for p in want if par[p]}
    got = set(zip(I.tolist(), J.tolist()))
    assert want_par <= got and all(par[p] for p in got)
    assert len(want_par) < len(want)                                        # precondition


def test_subtract_covered_min_len_is_separate_from_the_cover_tolerance():
    """F1 review: the enlarged cover tolerance was also the minimum length, so on the
    poly route every hidden chord shorter than 0.3 mm on paper was deleted anywhere in
    the view (a fine faceted circle vanished). ``min_len`` sets the floor separately;
    the default (None = tol) keeps the old behaviour."""
    tol = 0.3
    t = np.linspace(0.0, 2.0 * np.pi, 257)
    ring_pts = 10.0 * np.stack([np.cos(t), np.sin(t)], 1)
    ring = np.stack([ring_pts[:-1], ring_pts[1:]], 1)                        # chords 0.245
    vis = np.array([[[-20.0, -20.0], [20.0, -20.0]]])                       # far away
    assert _seg_lengths(ring).max() < tol                                    # precondition
    assert len(g2.subtract_covered(ring, vis, tol)) == 0                     # old default
    kept = g2.subtract_covered(ring, vis, tol, min_len=1e-3, loose=True)
    assert len(kept) == 256
    assert _seg_lengths(kept).sum() == pytest.approx(_seg_lengths(ring).sum(), rel=1e-12)
    # short visible chords are part of the cover set when min_len is small: a hidden line
    # under a run of 0.2 long visible chords is removed (with the old floor they were
    # filtered out of the cover set and the hidden line stayed whole)
    hid = np.array([[[0.0, 0.0], [10.0, 0.0]]])
    xs = np.arange(0.0, 10.0 + 1e-9, 0.2)
    run = np.stack([np.stack([xs[:-1], np.full(len(xs) - 1, 0.1)], 1),
                    np.stack([xs[1:], np.full(len(xs) - 1, 0.1)], 1)], 1)
    assert _seg_lengths(run).max() < tol                                     # precondition
    assert len(g2.subtract_covered(hid, run, tol)) == 1                      # old default
    assert len(g2.subtract_covered(hid, run, tol, min_len=1e-3, loose=True)) == 0
    assert len(g2.subtract_covered(hid, run, tol, min_len=1e-3)) == 0
    # pieces left by the cut are filtered by min_len, not tol: a 0.2 gap piece survives
    vis2 = np.array([[[0.0, 0.0], [4.9, 0.0]], [[5.1, 0.0], [10.0, 0.0]]])
    left = g2.subtract_covered(hid, vis2, tol, min_len=1e-3)
    assert len(left) == 1 and _seg_lengths(left)[0] == pytest.approx(0.2, abs=1e-9)
    assert len(g2.subtract_covered(hid, vis2, tol)) == 0                     # old default
