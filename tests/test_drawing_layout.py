"""Dimension plan, dimension rendering and sheet layout of ``src.drawing``.

Synthetic views stand in for the geometry sources: a front view of a
100 x 15 rectangle, a top view of a 100 x 60 rectangle with five circles
(four Ø6.5 at the corners, one Ø20 in the middle), a right view 60 x 15, an
iso view of the box, and a section with two loops. The views are
asymmetric (different widths and heights) so a swapped placement fails.

Test discipline: every negative test asserts its precondition first, and
the behaviours are pinned by pairs of inputs (first vs third angle, holes
on vs off, cap 6 vs 1, axis parallel vs oblique) so a no-op or constant
implementation fails.
"""
import math
import os
import sys
from collections import Counter

import numpy as np
import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)

from src.drawing.dimensions import (  # noqa: E402
    DimensionPlan, format_value, plan_dimensions, render_dimension)
from src.drawing.layout import (  # noqa: E402
    LayoutResult, band_counts, choose_scale, compose, grid_cells)
from src.drawing.model import (  # noqa: E402
    BBox2D, Circle, CylFeature, Dimension, HatchLines, LineStyle, Polygon, Polyline,
    ProjectedView, SectionView, Segment, Text)
from src.drawing.options import DrawingOptions  # noqa: E402
from src.drawing.projection import VIEWS  # noqa: E402
from src.drawing.style import DEFAULT_STYLE, SHEETS, STANDARD_SCALES, hatch_spacing  # noqa: E402

ST = DEFAULT_STYLE
HOLES = [(8.0, 8.0), (92.0, 8.0), (8.0, 52.0), (92.0, 52.0)]
TOL = 1e-6


# ------------------------------------------------------------ fixtures --

def _rect(x0, y0, x1, y1, style=LineStyle.VISIBLE):
    return [Segment((x0, y0), (x1, y0), style), Segment((x1, y0), (x1, y1), style),
            Segment((x1, y1), (x0, y1), style), Segment((x0, y1), (x0, y0), style)]


def _box_iso(ex, ey, ez):
    spec = VIEWS["iso"]
    corners = np.array([[x, y, z] for x in (0, ex) for y in (0, ey) for z in (0, ez)], float)
    v = spec.to_view(corners)[:, :2]
    segs = []
    for i in range(8):
        for j in range(i + 1, 8):
            if np.count_nonzero(corners[i] != corners[j]) == 1:
                segs.append(Segment(tuple(v[i]), tuple(v[j])))
    return ProjectedView(spec, segs, [], BBox2D.from_points(v), "hlr_exact")


def _box_views(ex, ey, ez, iso=True):
    views = {
        "front": ProjectedView(VIEWS["front"], _rect(0, 0, ex, ez), [], BBox2D(0, 0, ex, ez), "hlr_exact"),
        "top": ProjectedView(VIEWS["top"], _rect(0, 0, ex, ey), [], BBox2D(0, 0, ex, ey), "hlr_exact"),
        "right": ProjectedView(VIEWS["right"], _rect(0, 0, ey, ez), [], BBox2D(0, 0, ey, ez), "hlr_exact"),
    }
    if iso:
        views["iso"] = _box_iso(ex, ey, ez)
    return views


def _plate_views():
    hidden = [Segment((x, 0.0), (x, 15.0), LineStyle.HIDDEN)
              for cx in (8.0, 92.0) for x in (cx - 3.25, cx + 3.25)]
    hidden += [Segment((40.0, 0.0), (40.0, 15.0), LineStyle.HIDDEN),
               Segment((60.0, 0.0), (60.0, 15.0), LineStyle.HIDDEN)]
    front = ProjectedView(VIEWS["front"], _rect(0, 0, 100, 15), hidden, BBox2D(0, 0, 100, 15), "hlr_exact")
    top_vis = _rect(0, 0, 100, 60) + [Circle(h, 3.25) for h in HOLES] + [Circle((50.0, 30.0), 10.0)]
    top = ProjectedView(VIEWS["top"], top_vis, [], BBox2D(0, 0, 100, 60), "hlr_exact")
    right = ProjectedView(VIEWS["right"], _rect(0, 0, 60, 15),
                          [Segment((20.0, 0.0), (20.0, 15.0), LineStyle.HIDDEN)],
                          BBox2D(0, 0, 60, 15), "hlr_exact")
    return {"front": front, "top": top, "right": right, "iso": _box_iso(100, 60, 15)}


def _features(extra=()):
    f = [CylFeature(np.array([x, y, 7.5]), np.array([0.0, 0.0, 1.0]), 3.25, 15.0, "hole", True, 1)
         for x, y in HOLES]
    # the big hole's axis points down: parallelism must use |dot|
    f.append(CylFeature(np.array([50.0, 30.0, 7.5]), np.array([0.0, 0.0, -1.0]), 10.0, 15.0, "hole", True, 1))
    f.append(CylFeature(np.array([78.0, 30.0, 20.0]), np.array([0.0, 0.0, 1.0]), 15.0, 10.0, "boss", True, 1))
    return f + list(extra)


def _section():
    loops = [np.array([[0, 0], [40, 0], [40, 15], [0, 15]], float),
             np.array([[60, 0], [100, 0], [100, 15], [93, 15], [93, 25], [63, 25], [63, 15], [60, 15]], float)]
    body = ProjectedView(VIEWS["front"], _rect(0, 0, 100, 15), [], BBox2D(0, 0, 100, 25), "hlr_exact")
    return SectionView(VIEWS["front"], np.array([50.0, 30.0, 7.5]), np.array([0.0, -1.0, 0.0]),
                       loops, body, label="A", marker_view="top")


EXTENTS = np.array([100.0, 60.0, 15.0])


def _plan(views=None, features=None, **opt):
    views = _plate_views() if views is None else views
    features = _features() if features is None else features
    return plan_dimensions(views, features, DrawingOptions(**opt), ST, EXTENTS)


def _compose(projection="first", sheet="A3", views=None, section="default", plan=None, **opt):
    plate = views is None
    views = _plate_views() if plate else views
    sec = _section() if section == "default" else section
    options = DrawingOptions(projection=projection, sheet=sheet, **opt)
    if plan is None:
        ext = EXTENTS if plate else np.array([views["front"].bbox.width, views["top"].bbox.height,
                                              views["front"].bbox.height])
        plan = plan_dimensions(views, _features() if plate else [], options, ST, ext)
    meta = {"title": "Plate", "ident": "plate", "owner": "ACME", "source_note": "STEP"}
    return compose(views, sec, plan, options, ST, meta), views, plan


def _geom_box(res: LayoutResult, views, name):
    """Sheet bbox of a view's geometry (without bands), from its placement offset."""
    (ox, oy), _ = res.placements[name]
    bb = views[name].bbox
    s = res.scale
    return BBox2D(ox + s * bb.xmin, oy + s * bb.ymin, ox + s * bb.xmax, oy + s * bb.ymax)


def _same_segment(seg, a, b, tol=TOL):
    p, q = np.asarray(seg.a), np.asarray(seg.b)
    a, b = np.asarray(a, float), np.asarray(b, float)
    return ((np.allclose(p, a, atol=tol) and np.allclose(q, b, atol=tol))
            or (np.allclose(p, b, atol=tol) and np.allclose(q, a, atol=tol)))


def _arrow_tip_and_tail(poly):
    """A 3-point arrowhead: (tip, midpoint of the base) — the tip is the vertex opposite the shortest side."""
    pts = np.asarray(poly.points, float)
    best = None
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        base = np.linalg.norm(pts[j] - pts[k])
        if best is None or base < best[0]:
            best = (base, pts[i], (pts[j] + pts[k]) / 2)
    return best[1], best[2]


# ------------------------------------------------------- format_value --

@pytest.mark.parametrize("v, text", [(60.0, "60"), (6.5, "6.5"), (12.345, "12.35"), (100, "100"),
                                     (0.004, "0"), (-0.001, "0"), (2.675, "2.68"), (1e-12, "0")])
def test_format_value(v, text):
    assert format_value(v) == text


def test_format_value_rejects_nan():
    with pytest.raises(ValueError):
        format_value(float("nan"))


# ----------------------------------------------------- plan_dimensions --

def _linear(plan):
    return [d for d in plan.dims if d.kind == "linear"]


def _diam(plan):
    return [d for d in plan.dims if d.kind == "diameter"]


def test_plan_overall_extents_once_each_in_the_right_view():
    plan = _plan(hole_positions=False)
    lin = _linear(plan)
    values = Counter(round(d.value, 6) for d in lin)
    assert values[100.0] == 1 and values[15.0] == 1 and values[60.0] == 1
    by_value = {round(d.value, 6): d for d in lin}
    assert by_value[100.0].view == "front" and by_value[100.0].angle_deg == 0.0
    assert by_value[15.0].view == "front" and by_value[15.0].angle_deg == 90.0
    assert by_value[60.0].view == "top" and by_value[60.0].angle_deg == 90.0
    assert not [d for d in plan.dims if d.view in ("right", "iso")]   # nothing repeated
    assert [d.text for d in lin] == [format_value(d.value) for d in lin]


def test_plan_overall_extents_fall_back_when_front_missing():
    views = _plate_views()
    del views["front"]
    plan = _plan(views=views, hole_positions=False)
    by_value = {round(d.value, 6): d for d in _linear(plan)}
    assert by_value[100.0].view == "top"            # width moves under the top view
    assert by_value[15.0].view == "right"           # height moves beside the right view
    assert by_value[60.0].view == "top"


def test_plan_hole_callouts_grouped_by_radius():
    plan = _plan()
    d = _diam(plan)
    assert sorted(x.text for x in d) == ["4 x Ø6.5", "Ø20"]
    assert all(x.view == "top" for x in d)
    four = next(x for x in d if x.count == 4)
    assert four.radius == pytest.approx(3.25) and four.value == pytest.approx(6.5)
    assert four.center == pytest.approx((92.0, 52.0))       # circle nearest the top-right corner
    assert four.leader_angle_deg == pytest.approx(45.0)
    one = next(x for x in d if x.count == 1)
    assert one.center == pytest.approx((50.0, 30.0)) and one.value == pytest.approx(20.0)
    assert plan.undimensioned == []


def test_plan_hole_positions_on_and_off():
    on = _plan(hole_positions=True)
    off = _plan(hole_positions=False)
    overall = {100.0, 60.0, 15.0}
    pos_on = [d for d in _linear(on) if round(d.value, 6) not in overall]
    pos_off = [d for d in _linear(off) if round(d.value, 6) not in overall]
    assert len(pos_on) == 4                       # precondition for the "off" assertion
    assert pos_off == []
    assert sorted(round(d.value, 6) for d in pos_on) == [30.0, 50.0, 52.0, 92.0]
    assert all(d.view == "top" for d in pos_on)
    horizontal = sorted(round(d.value, 6) for d in pos_on if d.angle_deg == 0.0)
    assert horizontal == [50.0, 92.0]
    # bands: shorter dims nearer the outline, the overall depth outermost on the left of the top view
    bands = {round(d.value, 6): b for d, b in zip(on.dims, on.bands) if d.view == "top" and d.kind == "linear"}
    assert bands[30.0] == ("left", 0) and bands[52.0] == ("left", 1) and bands[60.0] == ("left", 2)
    assert bands[50.0] == ("bottom", 0) and bands[92.0] == ("bottom", 1)


def test_plan_cap_moves_groups_to_undimensioned():
    full = _plan(max_hole_callouts=6)
    assert len(_diam(full)) == 2 and full.undimensioned == []     # precondition
    capped = _plan(max_hole_callouts=1)
    d = _diam(capped)
    assert len(d) == 1 and d[0].count == 4                        # bigger group first
    assert len(capped.undimensioned) == 1
    u = capped.undimensioned[0]
    assert u["reason"] == "cap" and u["radius"] == pytest.approx(10.0) and u["count"] == 1
    assert u["kind"] == "hole_group"
    # the dropped group gets no position dims either
    pos = [x for x in _linear(capped) if round(x.value, 6) in (50.0, 30.0)]
    assert pos == []


def test_plan_hole_without_axial_view_is_undimensioned():
    axis_ok = CylFeature(np.array([30.0, 20.0, 7.5]), np.array([0.0, 0.0, 1.0]), 2.0, 15.0, "hole", True, 1)
    ok = _plan(features=_features([axis_ok]))
    assert any(x.text == "Ø4" for x in _diam(ok))            # precondition: a Z hole is called out
    oblique_axis = np.array([1.0, 1.0, 0.0]) / math.sqrt(2.0)
    oblique = CylFeature(np.array([30.0, 20.0, 7.5]), oblique_axis, 2.0, 15.0, "hole", True, 1)
    plan = _plan(features=_features([oblique]))
    assert not any(x.text == "Ø4" for x in _diam(plan))
    u = [e for e in plan.undimensioned if e["reason"] == "no_axial_view"]
    assert len(u) == 1 and u[0]["radius"] == pytest.approx(2.0) and u[0]["count"] == 1


def test_plan_x_axis_hole_goes_to_the_right_view_only_when_drawn():
    xhole = CylFeature(np.array([50.0, 30.0, 7.5]), np.array([1.0, 0.0, 0.0]), 2.5, 100.0, "hole", True, 1)
    plan = _plan(features=_features([xhole]))
    d = [x for x in _diam(plan) if x.text == "Ø5"]
    assert len(d) == 1 and d[0].view == "right"
    assert d[0].center == pytest.approx((30.0, 7.5))              # right view 2D = (Y, Z)
    views = _plate_views()
    del views["right"]
    plan2 = _plan(views=views, features=_features([xhole]))
    assert not [x for x in _diam(plan2) if x.text == "Ø5"]
    assert [e["reason"] for e in plan2.undimensioned] == ["no_axial_view"]


def test_plan_ignores_bosses_and_mesh_route_has_no_callouts():
    plan = _plan(features=[f for f in _features() if f.kind == "boss"])
    assert _diam(plan) == [] and plan.undimensioned == []
    assert len(_linear(plan)) == 3


# ---------------------------------------------------- render_dimension --

def _split(items):
    segs = [i for i in items if isinstance(i, Segment)]
    polys = [i for i in items if isinstance(i, Polygon)]
    texts = [i for i in items if isinstance(i, Text)]
    return segs, polys, texts


def test_render_horizontal_linear():
    d = Dimension("linear", "100", 100.0, p1=(10.0, 20.0), p2=(110.0, 20.0), base=(60.0, 5.0), angle_deg=0.0)
    items = render_dimension(d, ST)
    segs, polys, texts = _split(items)
    assert len(segs) == 3 and len(polys) == 2 and len(texts) == 1
    assert all(s.style == LineStyle.DIMENSION for s in segs)
    # extension lines: start ext_gap from the point, end ext_overshoot past the dimension line
    assert any(_same_segment(s, (10, 20 - ST.ext_gap), (10, 5 - ST.ext_overshoot)) for s in segs)
    assert any(_same_segment(s, (110, 20 - ST.ext_gap), (110, 5 - ST.ext_overshoot)) for s in segs)
    assert any(_same_segment(s, (10, 5), (110, 5)) for s in segs)
    tips = []
    for p in polys:
        assert np.asarray(p.points).shape == (3, 2)
        tip, tail = _arrow_tip_and_tail(p)
        assert np.linalg.norm(tip - tail) == pytest.approx(ST.arrow_length, abs=TOL)
        tips.append((tuple(np.round(tip, 9)), tail))
    assert sorted(t for t, _ in tips) == [(10.0, 5.0), (110.0, 5.0)]
    for tip, tail in tips:                       # arrows inside the span point outward
        assert 10.0 < tail[0] < 110.0
    t = texts[0]
    assert t.text == "100" and t.rotation_deg == 0.0 and t.anchor == "middle"
    assert t.position == pytest.approx((60.0, 5.0 + ST.dim_text_gap))
    assert t.height == pytest.approx(ST.text_height)


def test_render_vertical_linear_reads_from_the_right():
    d = Dimension("linear", "60", 60.0, p1=(10.0, 20.0), p2=(10.0, 80.0), base=(0.0, 50.0), angle_deg=90.0)
    segs, polys, texts = _split(render_dimension(d, ST))
    assert len(segs) == 3 and len(polys) == 2 and len(texts) == 1
    assert any(_same_segment(s, (0, 20), (0, 80)) for s in segs)
    assert any(_same_segment(s, (10 - ST.ext_gap, 20), (0 - ST.ext_overshoot, 20)) for s in segs)
    t = texts[0]
    assert t.rotation_deg == pytest.approx(90.0)
    assert t.position[0] < 0.0                    # text to the left of the line at x = 0
    assert t.position == pytest.approx((-ST.dim_text_gap, 50.0))


def test_render_short_linear_puts_arrows_outside():
    d = Dimension("linear", "4", 4.0, p1=(0.0, 0.0), p2=(4.0, 0.0), base=(2.0, -10.0), angle_deg=0.0)
    assert 4.0 < 2 * ST.arrow_length                                   # precondition: no room inside
    segs, polys, _ = _split(render_dimension(d, ST))
    tails = sorted(_arrow_tip_and_tail(p)[1][0] for p in polys)
    tips = sorted(_arrow_tip_and_tail(p)[0][0] for p in polys)
    assert tips == pytest.approx([0.0, 4.0])
    assert tails[0] < 0.0 and tails[1] > 4.0
    dim_line = [s for s in segs if abs(s.a[1] + 10.0) < TOL and abs(s.b[1] + 10.0) < TOL]
    assert len(dim_line) == 1
    xs = sorted([dim_line[0].a[0], dim_line[0].b[0]])
    assert xs[0] < -ST.arrow_length + TOL and xs[1] > 4.0 + ST.arrow_length - TOL


def test_render_diameter_leader():
    d = Dimension("diameter", "Ø20", 20.0, center=(50.0, 50.0), radius=10.0, leader_angle_deg=45.0,
                  base=(50.0, 50.0))
    segs, polys, texts = _split(render_dimension(d, ST))
    assert len(polys) == 1 and len(texts) == 1
    u = np.array([math.cos(math.pi / 4), math.sin(math.pi / 4)])
    on_circle = np.array([50.0, 50.0]) + 10.0 * u
    leader = [s for s in segs if np.allclose(s.a, on_circle, atol=TOL) or np.allclose(s.b, on_circle, atol=TOL)]
    assert len(leader) == 1
    far = np.asarray(leader[0].b if np.allclose(leader[0].a, on_circle, atol=TOL) else leader[0].a)
    direction = (far - on_circle) / np.linalg.norm(far - on_circle)
    assert direction == pytest.approx(u, abs=1e-9)                  # leader at leader_angle_deg
    tip, tail = _arrow_tip_and_tail(polys[0])
    assert np.linalg.norm(tip - np.array([50.0, 50.0])) == pytest.approx(10.0, abs=TOL)
    assert np.linalg.norm(tail - np.array([50.0, 50.0])) > 10.0      # arrow sits outside, points in
    t = texts[0]
    assert t.anchor == "start" and t.text == "Ø20"
    assert np.linalg.norm(np.asarray(t.position) - far) < 2.0         # text at the leader end


def test_render_diameter_base_extends_the_leader():
    short = Dimension("diameter", "Ø6.5", 6.5, center=(0.0, 0.0), radius=3.25, base=(0.0, 0.0))
    far_base = (40.0, 40.0)
    long_ = Dimension("diameter", "Ø6.5", 6.5, center=(0.0, 0.0), radius=3.25, base=far_base)

    def leader_end(dim):
        segs, _, _ = _split(render_dimension(dim, ST))
        ends = [np.asarray(p) for s in segs for p in (s.a, s.b)]
        return max(ends, key=lambda p: p[0] + p[1] if abs(p[0] - p[1]) < 1e-9 else -1e9)

    e_short = leader_end(short)
    e_long = leader_end(long_)
    assert np.linalg.norm(e_short) < 20.0                       # default leader is short
    assert e_long == pytest.approx(far_base, abs=1e-9)          # base sets the leader end


# ---------------------------------------------------------- choose_scale --

def _boxes(views):
    return {k: v.bbox for k, v in views.items()}


def test_choose_scale_plate_on_a3_is_full_size():
    views = _plate_views()
    plan = _plan(views=views)
    s, info = choose_scale(_boxes(views), band_counts(plan), SHEETS["A3"], ST, "first", True, True)
    assert s == 1.0
    assert info["cells"]["section"] == (1, 0)
    # 2:1 was tried first and did not fit
    assert info["tried"] == [(50, False), (20, False), (10, False), (5, False), (2, False), (1, True)]
    # at 1:1 the iso row above F would push the top view into the title block: the iso view floats
    assert info["iso_mode"] == "free" and "iso" not in info["cells"]
    _assert_iso_clear(info["rects"], SHEETS["A3"])


def _assert_iso_clear(rects, size):
    iso = rects["iso"]
    for name, r in rects.items():
        if name == "iso":
            continue
        gap_x = max(r.xmin - iso.xmax, iso.xmin - r.xmax)
        gap_y = max(r.ymin - iso.ymax, iso.ymin - r.ymax)
        assert max(gap_x, gap_y) >= ST.view_gap - 1e-6, name
    fx1 = size.width - ST.margin
    tz = BBox2D(fx1 - ST.title_block_w, ST.margin, fx1, ST.margin + ST.title_block_h + ST.strip_h)
    assert iso.xmax <= tz.xmin or iso.ymin >= tz.ymax


def test_choose_scale_iso_keeps_its_cell_when_there_is_room():
    views = _box_views(60.0, 60.0, 60.0)
    s, info = choose_scale(_boxes(views), {}, SHEETS["A3"], ST, "first", False, True)
    assert s == 1.0                       # 2:1 fails (front + right 240 wide plus the iso)
    assert info["iso_mode"] == "cell" and info["cells"]["iso"] == (1, 1)
    assert info["iso_scale"] == 1.0
    iso, front = info["rects"]["iso"], info["rects"]["front"]
    assert iso.ymin > front.ymax and iso.xmin > front.xmax         # top-right of F
    _assert_iso_clear(info["rects"], SHEETS["A3"])


def test_choose_scale_iso_drops_its_own_scale_before_the_sheet(monkeypatch):
    import src.drawing.layout as layout_mod
    views = _box_views(50.0, 50.0, 50.0)
    monkeypatch.setattr(layout_mod, "ISO_SCALE_STEPS", 0)
    s0, info0 = choose_scale(_boxes(views), {}, SHEETS["A3"], ST, "first", False, True)
    assert s0 == 1.0 and info0["iso_scale"] == 1.0   # control: iso tied to the sheet scale -> 1:1
    monkeypatch.setattr(layout_mod, "ISO_SCALE_STEPS", 2)
    s, info = choose_scale(_boxes(views), {}, SHEETS["A3"], ST, "first", False, True)
    assert s == 2.0                                  # principal views at 2:1 ...
    assert info["iso_mode"] == "free" and info["iso_scale"] == 1.0    # ... iso one step smaller
    iso_bb = views["iso"].bbox
    r = info["rects"]["iso"]
    assert (r.width, r.height) == pytest.approx((iso_bb.width, iso_bb.height))   # drawn at 1:1
    _assert_iso_clear(info["rects"], SHEETS["A3"])


def test_choose_scale_big_part_and_overflow():
    big = _box_views(900.0, 600.0, 135.0, iso=False)
    plan = plan_dimensions(big, [], DrawingOptions(), ST, np.array([900.0, 600.0, 135.0]))
    s, _ = choose_scale(_boxes(big), band_counts(plan), SHEETS["A3"], ST, "first", False, False)
    assert s == pytest.approx(1 / 5)
    assert 0.5 * (900 + 600) > 390                     # precondition: 1:2 cannot fit the width
    huge = _box_views(60000.0, 40000.0, 20000.0, iso=False)
    s2, info2 = choose_scale(_boxes(huge), {}, SHEETS["A3"], ST, "first", False, False)
    assert s2 is None and info2["overflow"] is True
    assert info2["scale"] == pytest.approx(STANDARD_SCALES[-1])


def test_grid_cells_first_and_third():
    first = grid_cells(["front", "top", "right", "iso"], "first", has_section=True)
    third = grid_cells(["front", "top", "right", "iso"], "third", has_section=True)
    assert first == {"front": (0, 0), "top": (0, -1), "right": (-1, 0), "section": (1, 0), "iso": (1, 1)}
    assert third == {"front": (0, 0), "top": (0, 1), "right": (1, 0), "section": (-1, 0), "iso": (1, 1)}
    both = grid_cells(["front", "top", "right", "left"], "first", has_section=True)
    assert both["section"] == (1, -1)                 # middle row full -> row below


# ------------------------------------------------------------- compose --

def test_compose_auto_sheet_picks_first_fitting_at_full_size():
    views = _plate_views()
    plan = _plan(views=views)
    a4, _ = choose_scale(_boxes(views), band_counts(plan), SHEETS["A4"], ST, "first", True, True)
    assert a4 is not None and a4 < 1.0                # precondition: A4 needs a reduction
    res, _, _ = _compose(sheet="auto")
    assert (res.sheet_size.name, res.scale) == ("A3", 1.0)
    small = _box_views(20.0, 12.0, 3.0, iso=False)
    res2, _, plan2 = _compose(sheet="auto", views=small, section=None)
    assert len(plan2.dims) == 3
    s4, _ = choose_scale(_boxes(small), band_counts(plan2), SHEETS["A4"], ST, "first", False, False)
    assert s4 is not None and s4 >= 1.0               # precondition: fits A4 at 1:1 or larger
    assert (res2.sheet_size.name, res2.scale) == ("A4", s4)
    # the strip prints the chosen scale, not a constant 1:1
    assert res2.scale != 1.0 and res2.scale_text != "1:1"          # precondition
    assert f"SCALE {res2.scale_text}" in _strip_texts(res2.sheet)
    assert "SCALE 1:1" not in _strip_texts(res2.sheet)


def _strip_texts(sheet):
    x0 = sheet.width - ST.margin - ST.title_block_w
    y0 = ST.margin + ST.title_block_h
    return {t.text for t in sheet.items if isinstance(t, Text)
            and y0 <= t.position[1] <= y0 + ST.strip_h and t.position[0] >= x0}


def test_compose_strip_scale_on_a_reduced_sheet():
    res, _, _ = _compose("first", sheet="A4")
    assert res.scale == pytest.approx(1 / 5) and res.scale_text == "1:5"     # precondition: reduced
    texts = _strip_texts(res.sheet)
    assert "SCALE 1:5" in texts and "SCALE 1:1" not in texts
    assert res.sheet.meta["scale_text"] == "1:5"


def test_compose_first_angle_placement():
    res, views, plan = _compose("first")
    assert res.scale == 1.0 and res.scale_text == "1:1" and not res.overflow
    f, t, r = (_geom_box(res, views, n) for n in ("front", "top", "right"))
    assert (f.width, f.height) != (t.width, t.height) != (r.width, r.height)   # precondition: asymmetric
    assert t.ymax < f.ymin                                       # top BELOW front
    assert t.xmin == pytest.approx(f.xmin, abs=TOL) and t.xmax == pytest.approx(f.xmax, abs=TOL)
    assert r.xmax < f.xmin                                       # right view LEFT of front
    assert r.ymin == pytest.approx(f.ymin, abs=TOL) and r.ymax == pytest.approx(f.ymax, abs=TOL)
    sec = res.placements["section"][0]
    assert sec[1] == pytest.approx(res.placements["front"][0][1], abs=TOL)   # section shares Z
    sb = res.placements["section"][1]
    assert sb.xmin > res.placements["front"][1].xmax
    _assert_iso_clear({n: p[1] for n, p in res.placements.items()}, SHEETS["A3"])


def test_compose_third_angle_mirrors():
    res, views, _ = _compose("third")
    f, t, r = (_geom_box(res, views, n) for n in ("front", "top", "right"))
    assert t.ymin > f.ymax                                       # top ABOVE front
    assert t.xmin == pytest.approx(f.xmin, abs=TOL)
    assert r.xmin > f.xmax                                       # right view RIGHT of front
    assert r.ymin == pytest.approx(f.ymin, abs=TOL)
    assert res.placements["section"][1].xmax < res.placements["front"][1].xmin
    assert res.sheet.meta["projection"] == "third_angle"


def _strip_symbol(sheet):
    w = sheet.width
    x0 = w - ST.margin - ST.title_block_w
    y0 = ST.margin + ST.title_block_h
    circles = [i for i in sheet.items if isinstance(i, Circle)
               and x0 <= i.center[0] <= w - ST.margin and y0 <= i.center[1] <= y0 + ST.strip_h]
    polys = [i for i in sheet.items if isinstance(i, Polyline) and len(i.points) == 4
             and np.all(np.asarray(i.points)[:, 1] >= y0) and np.all(np.asarray(i.points)[:, 1] <= y0 + ST.strip_h)]
    return circles, polys


def test_projection_symbol_side_flips_with_projection():
    res1, _, _ = _compose("first")
    res3, _, _ = _compose("third")
    for res, circles_right in ((res1, True), (res3, False)):
        circles, polys = _strip_symbol(res.sheet)
        assert len(circles) == 2 and len(polys) == 1
        assert sorted(c.radius for c in circles) == pytest.approx([ST.symbol_h / 2, ST.symbol_h])
        assert circles[0].center == pytest.approx(circles[1].center)
        cone = np.asarray(polys[0].points)
        assert polys[0].closed
        cx = circles[0].center[0]
        if circles_right:
            assert cx > cone[:, 0].max()
        else:
            assert cx < cone[:, 0].min()
        # cone small end (h) on the left in both projections
        left = cone[np.isclose(cone[:, 0], cone[:, 0].min())]
        right = cone[np.isclose(cone[:, 0], cone[:, 0].max())]
        assert np.ptp(left[:, 1]) == pytest.approx(ST.symbol_h)
        assert np.ptp(right[:, 1]) == pytest.approx(2 * ST.symbol_h)


def _frame_lines(W, H):
    x0, y0, x1, y1 = ST.margin_left, ST.margin, W - ST.margin, H - ST.margin
    return [((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))]


def test_compose_frame_title_block_and_strip():
    res, _, _ = _compose("first")
    sh = res.sheet
    assert (sh.size_name, sh.width, sh.height) == ("A3", 420, 297)
    frame_segs = [i for i in sh.items if isinstance(i, Segment) and i.style == LineStyle.FRAME]
    for a, b in _frame_lines(420, 297):
        assert sum(_same_segment(s, a, b) for s in frame_segs) == 1
    segs = [i for i in sh.items if isinstance(i, Segment) and i.style in (LineStyle.TITLE, LineStyle.FRAME)]
    x0, x1, y0 = 410.0 - 180.0, 410.0, 10.0
    assert any(_same_segment(s, (x0, y0 + 36), (x1, y0 + 36)) for s in segs)    # block top
    assert any(_same_segment(s, (x0, y0), (x0, y0 + 36)) for s in segs)         # block left
    for k in (1, 2, 3):                                                          # 4 rows of 9
        assert any(_same_segment(s, (x0, y0 + 9 * k), (x1, y0 + 9 * k)) for s in segs)
    for c in (32.0, 74.2, 129.0):
        vert = [s for s in segs if abs(s.a[0] - (x0 + c)) < TOL and abs(s.b[0] - (x0 + c)) < TOL
                and max(s.a[1], s.b[1]) <= y0 + 36 + TOL and min(s.a[1], s.b[1]) >= y0 - TOL]
        assert sum(abs(s.a[1] - s.b[1]) for s in vert) == pytest.approx(36.0)   # full height
    texts = [i for i in sh.items if isinstance(i, Text)]
    strip_texts = {t.text for t in texts if y0 + 36 <= t.position[1] <= y0 + 45 and t.position[0] >= x0}
    assert "SCALE 1:1" in strip_texts and "Dimensions in mm" in strip_texts and "STEP" in strip_texts
    assert any(t.text == "Plate" for t in texts) and any(t.text == "plate" for t in texts)
    assert sh.meta["scale_text"] == "1:1" and sh.meta["projection"] == "first_angle"
    assert res.flags == []


def test_compose_everything_inside_the_frame():
    res, _, _ = _compose("first")
    sh = res.sheet
    frame = BBox2D(ST.margin_left, ST.margin, sh.width - ST.margin, sh.height - ST.margin)
    frame_lines = _frame_lines(sh.width, sh.height)
    n_checked = 0
    for item in sh.items:
        if isinstance(item, Segment) and item.style == LineStyle.FRAME and \
                not any(_same_segment(item, a, b) for a, b in frame_lines):
            continue      # centring marks cross the frame on purpose (ISO 5457 4.3)
        bb = BBox2D.of_primitives([item])
        assert bb is not None
        assert frame.expanded(TOL).contains((bb.xmin, bb.ymin)) and \
            frame.expanded(TOL).contains((bb.xmax, bb.ymax)), (item, bb)
        n_checked += 1
    assert n_checked > 100


@pytest.mark.parametrize("projection", ["first", "third"])
def test_views_keep_clear_of_the_centring_marks(projection):
    res, _, _ = _compose(projection)
    sh = res.sheet
    frame = BBox2D(ST.margin_left, ST.margin, sh.width - ST.margin, sh.height - ST.margin)
    marks = [i for i in sh.items if isinstance(i, Segment) and i.style == LineStyle.FRAME
             and (not frame.expanded(TOL).contains(i.a) or not frame.expanded(TOL).contains(i.b))]
    assert len(marks) == 4                                     # precondition: the marks exist
    assert max(abs(m.a[0] - m.b[0]) + abs(m.a[1] - m.b[1]) for m in marks) == pytest.approx(
        ST.centring_mark_outside + ST.centring_mark_inside)
    for name, (_, r) in res.placements.items():
        for m in marks:
            mb = BBox2D.of_primitives([m])
            hit = mb.xmin < r.xmax and mb.xmax > r.xmin and mb.ymin < r.ymax and mb.ymax > r.ymin
            assert not hit, (name, m)


def test_compose_hidden_lines_follow_the_option():
    on, _, _ = _compose("first", hidden_lines=True)
    off, _, _ = _compose("first", hidden_lines=False)
    n_on = sum(1 for i in on.sheet.items if getattr(i, "style", None) == LineStyle.HIDDEN)
    n_off = sum(1 for i in off.sheet.items if getattr(i, "style", None) == LineStyle.HIDDEN)
    assert n_on == 7 and n_off == 0               # 6 front + 1 right hidden segments


def test_compose_section_label_marker_and_hatch():
    res, views, _ = _compose("first")
    sh = res.sheet
    s = res.scale
    (sox, soy), _ = res.placements["section"]
    sec_top = soy + s * 25.0
    labels = [t for t in sh.items if isinstance(t, Text) and t.text == "A-A"]
    assert len(labels) == 1 and labels[0].height == pytest.approx(ST.label_height)
    assert labels[0].position[1] > sec_top
    sec_mid_x = sox + s * 50.0
    assert labels[0].position[0] == pytest.approx(sec_mid_x, abs=1.0)
    # cutting line marker in the top view at world Y = 30
    tb = _geom_box(res, views, "top")
    y_cut = res.placements["top"][0][1] + s * 30.0
    cutting = [i for i in sh.items if isinstance(i, Segment) and i.style == LineStyle.CUTTING]
    cut = [c for c in cutting if abs(c.a[1] - c.b[1]) < TOL]          # wide dashes along the line
    shafts = [c for c in cutting if abs(c.a[0] - c.b[0]) < TOL]       # arrow shafts across it
    assert len(cut) == 2 and len(shafts) == 2 and len(cutting) == 4
    for c in cut:
        assert c.a[1] == pytest.approx(y_cut) and c.b[1] == pytest.approx(y_cut)
        assert math.hypot(c.b[0] - c.a[0], c.b[1] - c.a[1]) == pytest.approx(ST.section_dash_len)
    assert min(min(c.a[0], c.b[0]) for c in cut) < tb.xmin and max(max(c.a[0], c.b[0]) for c in cut) > tb.xmax
    # shafts are on the cutting-line layer, not among the VISIBLE model edges
    assert not [i for i in sh.items if isinstance(i, Segment) and i.style == LineStyle.VISIBLE
                and abs(i.a[0] - i.b[0]) < TOL and min(i.a[1], i.b[1]) == pytest.approx(y_cut)
                and (i.a[0] < tb.xmin - TOL or i.a[0] > tb.xmax + TOL)]
    for sh_ in shafts:
        assert min(sh_.a[1], sh_.b[1]) == pytest.approx(y_cut)
        assert abs(sh_.a[1] - sh_.b[1]) == pytest.approx(ST.section_arrow_len - ST.section_head_len)
    arrows = [i for i in sh.items if isinstance(i, Polygon) and i.style == LineStyle.CUTTING]
    assert len(arrows) == 2
    for a in arrows:
        tip, tail = _arrow_tip_and_tail(a)
        assert np.linalg.norm(tip - tail) == pytest.approx(ST.section_head_len, abs=TOL)
        # viewing direction of the section = -normal = +Y world = +y in the top view
        assert tip[1] > tail[1]
        assert abs(tip[0] - tail[0]) < TOL
    letters = [t for t in sh.items if isinstance(t, Text) and t.text == "A"]
    assert len(letters) == 2 and all(t.height == pytest.approx(ST.label_height) for t in letters)
    # hatch: spacing from the section's shorter extent on the sheet (25 mm -> 3 mm)
    hatches = [i for i in sh.items if isinstance(i, HatchLines)]
    assert len(hatches) == 1
    h = hatches[0]
    assert h.segments.shape[0] > 10 and len(h.loops) == 2
    spacing = hatch_spacing(ST, 25.0 * s)
    assert spacing == 3.0
    d = h.segments[:, 1, :] - h.segments[:, 0, :]
    ang = np.degrees(np.arctan2(d[:, 1], d[:, 0])) % 180.0
    assert np.allclose(ang, 45.0, atol=1e-6)
    nrm = np.array([-math.sin(math.pi / 4), math.cos(math.pi / 4)])
    offs = np.unique(np.round(h.segments[:, 0, :] @ nrm, 6))
    assert np.allclose(np.diff(offs), spacing, atol=1e-5)
    outlines = [i for i in sh.items if isinstance(i, Polyline) and i.style == LineStyle.SECTION]
    assert len(outlines) == 2 and all(o.closed for o in outlines)


def test_compose_dimensions_in_sheet_coordinates():
    res, views, plan = _compose("first")
    sh = res.sheet
    assert len(sh.dimensions) == len(plan.dims) == 9
    (fox, foy), _ = res.placements["front"]
    s = res.scale
    width = [d for d in sh.dimensions if d.view == "front" and d.angle_deg == 0.0 and d.kind == "linear"]
    assert len(width) == 1
    w = width[0]
    assert w.p1 == pytest.approx((fox, foy)) and w.p2 == pytest.approx((fox + 100 * s, foy))
    assert w.base[1] == pytest.approx(foy - ST.dim_first_offset)        # band 0 below the front view
    four = next(d for d in sh.dimensions if d.kind == "diameter" and d.count == 4)
    (tox, toy), _ = res.placements["top"]
    assert four.center == pytest.approx((tox + 92 * s, toy + 52 * s))
    assert four.radius == pytest.approx(3.25 * s)
    assert four.value == pytest.approx(6.5)                              # value stays model mm
    dim_texts = [t.text for t in sh.items if isinstance(t, Text)]
    for txt in ("100", "15", "60", "4 x Ø6.5", "Ø20"):
        assert txt in dim_texts
    arrows = [i for i in sh.items if isinstance(i, Polygon) and i.style == LineStyle.DIMENSION]
    assert len(arrows) == 7 * 2 + 2


def test_compose_band_offsets_10_plus_7k():
    res, views, plan = _compose("first")
    sh = res.sheet
    assert len(sh.dimensions) == len(plan.dims)            # every view drawn: same order
    tb, fb = _geom_box(res, views, "top"), _geom_box(res, views, "front")
    seen = {}
    for d, b in zip(sh.dimensions, plan.bands):
        if d.kind != "linear":
            continue
        side, k = b
        off = ST.dim_first_offset + k * ST.dim_pitch
        g = tb if d.view == "top" else fb
        if side == "left":
            assert d.base[0] == pytest.approx(g.xmin - off), (d.text, b)
        elif side == "bottom":
            assert d.base[1] == pytest.approx(g.ymin - off), (d.text, b)
        else:
            raise AssertionError(f"unexpected side {side}")
        seen[(d.view, d.text)] = (side, k)
    # precondition: several bands per side, so a layout ignoring k fails
    assert seen[("top", "30")] == ("left", 0) and seen[("top", "52")] == ("left", 1)
    assert seen[("top", "60")] == ("left", 2)
    assert seen[("top", "50")] == ("bottom", 0) and seen[("top", "92")] == ("bottom", 1)
    lefts = sorted(d.base[0] for d in sh.dimensions if d.view == "top" and d.kind == "linear"
                   and d.angle_deg == 90.0)
    assert np.diff(lefts) == pytest.approx([ST.dim_pitch, ST.dim_pitch])   # distinct lines, 7 mm apart


def test_compose_callout_leader_ends_outside_the_outline():
    from src.drawing.layout import LEADER_EXIT_MARGIN
    res, views, _ = _compose("first")
    sh = res.sheet
    tb = _geom_box(res, views, "top")
    u = np.array([math.cos(math.radians(45.0)), math.sin(math.radians(45.0))])
    callouts = [d for d in sh.dimensions if d.kind == "diameter"]
    assert len(callouts) == 2
    for d in callouts:
        base = np.asarray(d.base, float)
        tip = np.asarray(d.center, float) + d.radius * u
        assert not tb.contains(tuple(base))                       # the leader leaves the view
        q = base - LEADER_EXIT_MARGIN * u                          # ... where it crosses the outline
        assert tb.expanded(1e-6).contains(tuple(q))
        assert min(abs(q[0] - tb.xmax), abs(q[1] - tb.ymax)) < 1e-6, d.text
        # the rendered leader runs from the circle to that base
        g = sh.meta["dimension_items"][sh.dimensions.index(d)]
        leaders = [sh.items[i] for i in g if isinstance(sh.items[i], Segment)
                   and _same_segment(sh.items[i], tip, base)]
        assert len(leaders) == 1, d.text


def _assert_inside(bb, rect, what):
    r = rect.expanded(1e-6)
    assert r.xmin <= bb.xmin and r.ymin <= bb.ymin and bb.xmax <= r.xmax and bb.ymax <= r.ymax, \
        (what, bb, rect)


def _short_spans(sh):
    out = []
    for d in sh.dimensions:
        if d.kind == "linear":
            span = math.hypot(d.p2[0] - d.p1[0], d.p2[1] - d.p1[1])
            if span < 2 * ST.arrow_length + 0.5:
                out.append(d.text)
    return out


_CONTAIN_CASES = {
    "plate_a3": dict(sheet="A3"),
    "plate_a3_third": dict(projection="third", sheet="A3"),
    "plate_a4": dict(sheet="A4"),
    "bar_1000x10x10": dict(sheet="A3", views=_box_views(1000.0, 10.0, 10.0, iso=False), section=None),
    "small_auto": dict(sheet="auto", views=_box_views(20.0, 12.0, 3.0, iso=False), section=None),
}


@pytest.mark.parametrize("case", sorted(_CONTAIN_CASES))
def test_compose_reserves_room_for_everything_a_view_draws(case):
    kw = dict(_CONTAIN_CASES[case])
    projection = kw.pop("projection", "first")
    res, views, _ = _compose(projection, **kw)
    sh = res.sheet
    if case in ("plate_a4", "bar_1000x10x10", "small_auto"):
        assert _short_spans(sh), case        # precondition: arrows-outside dims are present
    groups = sh.meta["dimension_items"]
    assert len(groups) == len(sh.dimensions) > 0
    for d, g in zip(sh.dimensions, groups):
        bb = BBox2D.of_primitives([sh.items[i] for i in g])
        _assert_inside(bb, res.placements[d.view][1], (d.view, d.text))
    texts = [t for t in sh.items if isinstance(t, Text)]
    if "section" in res.placements:
        mv = sh.meta["marker_view"]
        marker = [i for i in sh.items if getattr(i, "style", None) in (LineStyle.CUTTING, LineStyle.CENTER)
                  and isinstance(i, (Segment, Polygon))] + [t for t in texts if t.text == "A"]
        assert len(marker) >= 8                       # 2 dashes, 2 shafts, 2 heads, 2 letters
        for it in marker:
            _assert_inside(BBox2D.of_primitives([it]), res.placements[mv][1], ("marker", it))
        label = [t for t in texts if t.text == "A-A"]
        assert len(label) == 1
        _assert_inside(BBox2D.of_primitives(label), res.placements["section"][1], "A-A")
    if "iso" in res.placements:
        iso_label = [t for t in texts if t.text.startswith("ISO (")]
        assert len(iso_label) == 1
        _assert_inside(BBox2D.of_primitives(iso_label), res.placements["iso"][1], "iso label")


def test_compose_iso_view_is_labelled_with_its_own_scale():
    from src.drawing.style import scale_text
    res, views, _ = _compose("first")
    sh = res.sheet
    iso_s = sh.meta["iso_scale"]
    assert res.scale == 1.0 and iso_s == pytest.approx(0.5)     # measured: iso drops to 1:2 on A3
    labels = [t for t in sh.items if isinstance(t, Text) and t.text.startswith("ISO (")]
    assert [t.text for t in labels] == [f"ISO ({scale_text(iso_s)})"] == ["ISO (1:2)"]
    (ox, oy), _ = res.placements["iso"]
    bb = views["iso"].bbox
    top = oy + iso_s * bb.ymax
    assert labels[0].position[1] > top and labels[0].height == pytest.approx(ST.label_height)
    # the iso geometry is drawn at the iso scale: the 100 mm X edge is 50 mm long on the sheet
    xdir = VIEWS["iso"].to_view(np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]]))[:, :2]
    want = iso_s * np.linalg.norm(xdir[1] - xdir[0])
    iso_box = BBox2D(ox + iso_s * bb.xmin, oy + iso_s * bb.ymin, ox + iso_s * bb.xmax, top)
    lens = [math.hypot(i.b[0] - i.a[0], i.b[1] - i.a[1]) for i in sh.items
            if isinstance(i, Segment) and i.style == LineStyle.VISIBLE
            and iso_box.expanded(1e-6).contains(i.a) and iso_box.expanded(1e-6).contains(i.b)]
    assert len(lens) >= 9 and any(abs(x - want) < 1e-6 for x in lens)


def test_compose_section_marker_clears_the_dimension_bands():
    from src.drawing.layout import MARKER_ARROW_AT, MARKER_BAND_CLEAR, band_extent
    res, views, plan = _compose("first")
    sh = res.sheet
    assert sh.meta["marker_view"] == "top"
    k_left = band_counts(plan)["top"][0]
    k_right = band_counts(plan)["top"][2]
    assert k_left == 3 and k_right == 0                       # precondition: bands on the left only
    tb = _geom_box(res, views, "top")
    y_cut = res.placements["top"][0][1] + res.scale * 30.0
    shafts = sorted((i for i in sh.items if isinstance(i, Segment) and i.style == LineStyle.CUTTING
                     and abs(i.a[0] - i.b[0]) < TOL), key=lambda i: i.a[0])
    assert len(shafts) == 2
    assert shafts[0].a[0] == pytest.approx(tb.xmin - (band_extent(ST, k_left) + MARKER_BAND_CLEAR))
    assert shafts[1].a[0] == pytest.approx(tb.xmax + MARKER_ARROW_AT)
    # a narrow chain joins the wide dash to the far arrow, on the banded side only
    chain = [i for i in sh.items if isinstance(i, Segment) and i.style == LineStyle.CENTER]
    assert len(chain) == 1
    xs = sorted([chain[0].a[0], chain[0].b[0]])
    assert chain[0].a[1] == pytest.approx(y_cut) and chain[0].b[1] == pytest.approx(y_cut)
    assert xs == pytest.approx([shafts[0].a[0], tb.xmin - (ST.section_dash_len - 3.0)])
    # arrows and letters keep clear of every dimension item
    letters = [t for t in sh.items if isinstance(t, Text) and t.text == "A"]
    dim_boxes = [BBox2D.of_primitives([sh.items[i]]) for g in sh.meta["dimension_items"] for i in g]
    for it in shafts + letters:
        b = BBox2D.of_primitives([it]).expanded(0.5)
        for db in dim_boxes:
            assert not (b.xmin < db.xmax and b.xmax > db.xmin and b.ymin < db.ymax and b.ymax > db.ymin), \
                (it, db)


def test_compose_names_the_rendered_items_of_each_dimension():
    res, _, plan = _compose("first")
    sh = res.sheet
    groups = sh.meta["dimension_items"]
    assert len(groups) == len(sh.dimensions) == len(plan.dims)
    flat = [i for g in groups for i in g]
    assert len(flat) == len(set(flat))
    dim_styled = {i for i, it in enumerate(sh.items)
                  if isinstance(it, (Segment, Polygon)) and it.style == LineStyle.DIMENSION}
    assert dim_styled and dim_styled <= set(flat)          # every dimension line/arrow is named
    for d, g in zip(sh.dimensions, groups):
        texts = [sh.items[i] for i in g if isinstance(sh.items[i], Text)]
        assert [t.text for t in texts] == [d.text]


def test_compose_flags_a4l_and_overflow():
    res, _, _ = _compose("first", sheet="A4L")
    assert res.sheet_size.name == "A4L" and (res.sheet.width, res.sheet.height) == (297, 210)
    assert [f[:2] for f in res.flags] == [("sheet_not_iso5457", "warning")]
    a3, _, _ = _compose("first", sheet="A3")
    assert a3.flags == []
    huge = _box_views(60000.0, 40000.0, 20000.0, iso=False)
    res2, _, _ = _compose("first", views=huge, section=None)
    assert res2.overflow is True and res2.scale == pytest.approx(STANDARD_SCALES[-1])
    assert ("views_overflow", "warning") in [f[:2] for f in res2.flags]


def test_compose_without_section_has_no_marker():
    with_sec, _, _ = _compose("first")                        # precondition: the default draws them
    w = with_sec.sheet.items
    assert "section" in with_sec.placements
    assert len([i for i in w if isinstance(i, Segment) and i.style == LineStyle.CUTTING]) == 4
    assert len([i for i in w if isinstance(i, HatchLines)]) == 1
    assert len([i for i in w if isinstance(i, Text) and i.text == "A-A"]) == 1
    res, _, _ = _compose("first", section=None)
    assert {"front", "top", "right"} <= set(res.placements)   # the rest of the drawing is still there
    assert res.sheet.dimensions
    items = res.sheet.items
    assert "section" not in res.placements
    assert not [i for i in items if isinstance(i, Segment) and i.style == LineStyle.CUTTING]
    assert not [i for i in items if isinstance(i, HatchLines)]
    assert not [i for i in items if isinstance(i, Text) and i.text == "A-A"]


def test_dimension_plan_type():
    plan = _plan()
    assert isinstance(plan, DimensionPlan)
    assert len(plan.bands) == len(plan.dims)
    for d, b in zip(plan.dims, plan.bands):
        assert (b is None) == (d.kind == "diameter")
