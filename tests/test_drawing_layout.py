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


def _holes_undim(plan):
    return [u for u in plan.undimensioned if u["kind"] == "hole_group"]


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
    # the top-left circle with a 45 degree leader: the shortest leader whose text runs over
    # the view instead of past its right edge (the top-right circle would widen the sheet)
    assert four.center == pytest.approx((8.0, 52.0))
    assert four.leader_angle_deg == pytest.approx(45.0)
    one = next(x for x in d if x.count == 1)
    assert one.center == pytest.approx((50.0, 30.0)) and one.value == pytest.approx(20.0)
    # a different direction: its leader would otherwise cross the first one above the view
    assert one.leader_angle_deg != pytest.approx(45.0)
    # the boss is recorded, not silently dropped (QA issue 26); no hole is undimensioned
    assert [u["kind"] for u in plan.undimensioned] == ["boss_group"]


def test_plan_hole_positions_on_and_off():
    on = _plan(hole_positions=True)
    off = _plan(hole_positions=False)
    overall = {100.0, 60.0, 15.0}
    pos_on = [d for d in _linear(on) if round(d.value, 6) not in overall]
    pos_off = [d for d in _linear(off) if round(d.value, 6) not in overall]
    assert len(pos_on) == 4                       # precondition for the "off" assertion
    assert pos_off == []
    # QA round 2 issue 15: the position reference of a group is the hole nearest the datum
    # corner (bottom-left), so the 4 x Ø6.5 group is measured at (8, 8) and the Ø20 hole at
    # (50, 30); extension lines stay short instead of crossing the view
    assert sorted(round(d.value, 6) for d in pos_on) == [8.0, 8.0, 30.0, 50.0]
    assert all(d.view == "top" for d in pos_on)
    horizontal = sorted(round(d.value, 6) for d in pos_on if d.angle_deg == 0.0)
    assert horizontal == [8.0, 50.0]
    bands = {(round(d.value, 6), d.angle_deg): b for d, b in zip(on.dims, on.bands)
             if d.view == "top" and d.kind == "linear"}
    # shorter dims nearer the outline, the overall depth outermost on the left of the top view
    assert bands[(8.0, 90.0)] == ("left", 0) and bands[(30.0, 90.0)] == ("left", 1)
    assert bands[(60.0, 90.0)] == ("left", 2)
    assert bands[(8.0, 0.0)] == ("bottom", 0) and bands[(50.0, 0.0)] == ("bottom", 1)


def test_plan_cap_moves_groups_to_undimensioned():
    full = _plan(max_hole_callouts=6)
    assert len(_diam(full)) == 2 and _holes_undim(full) == []     # precondition
    capped = _plan(max_hole_callouts=1)
    d = _diam(capped)
    assert len(d) == 1 and d[0].count == 4                        # bigger group first
    assert len(_holes_undim(capped)) == 1
    u = _holes_undim(capped)[0]
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
    assert [e["reason"] for e in _holes_undim(plan2)] == ["no_axial_view"]


def test_plan_bosses_are_undimensioned_not_supported_and_mesh_route_has_no_callouts():
    plan = _plan(features=[f for f in _features() if f.kind == "boss"])
    assert _diam(plan) == []
    assert plan.undimensioned == [{"kind": "boss_group", "radius": 15.0, "count": 1,
                                   "reason": "not_supported", "view": None}]
    assert len(_linear(plan)) == 3
    mesh = _plan(features=[])
    assert _diam(mesh) == [] and mesh.undimensioned == [] and len(_linear(mesh)) == 3


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
    # QA round 2 issue 6: the section leaves the side cell for the front column, so the plate on
    # A4 is drawn at 1:2 (it was 1:5 with the section beside the front view)
    assert res.scale == pytest.approx(1 / 2) and res.scale_text == "1:2"     # precondition: reduced
    texts = _strip_texts(res.sheet)
    assert "SCALE 1:2" in texts and "SCALE 1:1" not in texts
    assert res.sheet.meta["scale_text"] == "1:2"


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
    # hatch: one HatchLines per region (outer loop + holes, QA issue 29), all at the
    # section spacing from the section's shorter extent on the sheet (25 mm -> 3 mm):
    # both regions are wide enough, so ISO 128-50 "one part, one hatching" holds
    hatches = [i for i in sh.items if isinstance(i, HatchLines)]
    assert len(hatches) == 2 and all(len(h.loops) == 1 for h in hatches)
    segs = np.vstack([h.segments for h in hatches])
    assert all(h.segments.shape[0] > 5 for h in hatches)
    spacing = hatch_spacing(ST, 25.0 * s)
    assert spacing == 3.0
    d = segs[:, 1, :] - segs[:, 0, :]
    ang = np.degrees(np.arctan2(d[:, 1], d[:, 0])) % 180.0
    assert np.allclose(ang, 45.0, atol=1e-6)
    nrm = np.array([-math.sin(math.pi / 4), math.cos(math.pi / 4)])
    offs = np.unique(np.round(segs[:, 0, :] @ nrm, 6))
    assert np.allclose(np.diff(offs), spacing, atol=1e-5)
    assert res.sheet.meta["section_layout"]["hatch_spacings"] == [3.0, 3.0]
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
    assert four.center == pytest.approx((tox + 8 * s, toy + 52 * s))
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
        elif side == "right":
            assert d.base[0] == pytest.approx(g.xmax + off), (d.text, b)
        else:
            assert d.base[1] == pytest.approx(g.ymax + off), (d.text, b)
        seen[(d.view, d.text, d.angle_deg)] = (side, k)
    # precondition: several bands on the left, so a layout ignoring k fails
    assert seen[("top", "8", 90.0)] == ("left", 0) and seen[("top", "30", 90.0)] == ("left", 1)
    assert seen[("top", "60", 90.0)] == ("left", 2)
    assert seen[("top", "8", 0.0)] == ("bottom", 0) and seen[("top", "50", 0.0)] == ("bottom", 1)
    lefts = sorted(d.base[0] for d in sh.dimensions if d.view == "top" and d.kind == "linear"
                   and d.angle_deg == 90.0 and d.base[0] < tb.xmin)
    assert np.diff(lefts) == pytest.approx([ST.dim_pitch, ST.dim_pitch])   # distinct lines, 7 mm apart


def test_compose_callout_leader_ends_outside_the_outline():
    from src.drawing.layout import LEADER_EXIT_MARGIN
    res, views, _ = _compose("first")
    sh = res.sheet
    tb = _geom_box(res, views, "top")
    callouts = [d for d in sh.dimensions if d.kind == "diameter"]
    assert len(callouts) == 2
    assert len({d.leader_angle_deg for d in callouts}) == 2               # one direction each
    for d in callouts:
        u = np.array([math.cos(math.radians(d.leader_angle_deg)), math.sin(math.radians(d.leader_angle_deg))])
        base = np.asarray(d.base, float)
        tip = np.asarray(d.center, float) + d.radius * u
        assert not tb.contains(tuple(base))                       # the leader leaves the view
        # where it crosses the outline: LEADER_EXIT_MARGIN before its end for an upward leader;
        # a downward one runs on until the text above its shelf clears the outline
        ts = [((tb.xmax if u[0] > 0 else tb.xmin) - tip[0]) / u[0], ((tb.ymax if u[1] > 0 else tb.ymin) - tip[1]) / u[1]]
        t_exit = min(ts)
        length = float(np.linalg.norm(base - tip))
        if u[1] > 0:
            assert length == pytest.approx(t_exit + LEADER_EXIT_MARGIN, abs=1e-6), d.text
        else:
            assert length >= t_exit + LEADER_EXIT_MARGIN - 1e-6, d.text
            g = sh.meta["dimension_items"][sh.dimensions.index(d)]
            t = next(sh.items[i] for i in g if isinstance(sh.items[i], Text))
            assert not _boxes_overlap(BBox2D.of_primitives([t]), tb), d.text
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
    # measured: the iso view used to drop to 1:2 on A3 (only eight fixed float positions and
    # the callout reach added to the bands); with the float sweep and the exact callout room
    # it is drawn at the sheet scale (QA issue 7)
    assert res.scale == 1.0 and iso_s == pytest.approx(1.0)
    labels = [t for t in sh.items if isinstance(t, Text) and t.text.startswith("ISO (")]
    assert [t.text for t in labels] == [f"ISO ({scale_text(iso_s)})"] == ["ISO (1:1)"]
    a4, _, _ = _compose("first", sheet="A4")
    assert a4.sheet.meta["iso_scale"] == pytest.approx(0.5)          # round 2 issue 6: A4 at 1:2
    assert [t.text for t in a4.sheet.items if isinstance(t, Text) and t.text.startswith("ISO (")] == ["ISO (1:2)"]
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
    from src.drawing.layout import MARKER_ARROW_AT, MARKER_BAND_CLEAR, MARKER_GAP, band_extent
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
    # QA round 2 issue 14: the plane y = 30 runs through the Ø20 hole centre, whose Y position dim
    # has its extension line on the cutting line on the banded left side: that whole end moves
    # past the outermost band (line, overshoot and a far-side text) instead of lying on the
    # extension line; no narrow chain is drawn any more
    outer_line = ST.dim_first_offset + (k_left - 1) * ST.dim_pitch
    start = outer_line + max(ST.ext_overshoot, ST.dim_text_gap + ST.text_height) + MARKER_GAP
    assert start > band_extent(ST, k_left) - ST.dim_pitch + ST.ext_overshoot
    assert shafts[0].a[0] == pytest.approx(tb.xmin - (start + MARKER_ARROW_AT - MARKER_GAP))
    assert shafts[1].a[0] == pytest.approx(tb.xmax + MARKER_ARROW_AT)
    assert [i for i in sh.items if isinstance(i, Segment) and i.style == LineStyle.CENTER] == []
    wide = sorted((i for i in sh.items if isinstance(i, Segment) and i.style == LineStyle.CUTTING
                   and abs(i.a[1] - i.b[1]) < TOL), key=lambda i: min(i.a[0], i.b[0]))
    assert len(wide) == 2 and all(w.a[1] == pytest.approx(y_cut) for w in wide)
    assert max(wide[0].a[0], wide[0].b[0]) == pytest.approx(tb.xmin - start)
    assert min(wide[1].a[0], wide[1].b[0]) == pytest.approx(tb.xmax + MARKER_GAP)
    assert MARKER_BAND_CLEAR > 0
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
    assert len([i for i in w if isinstance(i, HatchLines)]) == 2
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
    assert plan.flags == []


# =========================================================================
# QA round 1 (task F3): issues 2, 4, 7, 8, 9, 23, 26, 29
# =========================================================================

def _hole(x, y, r, z=7.5, length=15.0, axis=(0.0, 0.0, 1.0)):
    return CylFeature(np.array([x, y, z], float), np.array(axis, float), r, length, "hole", True, 1)


def _text_boxes(sheet, skip_title=True):
    x0 = sheet.width - ST.margin - ST.title_block_w
    y1 = ST.margin + ST.title_block_h + ST.strip_h
    out = []
    for t in sheet.items:
        if not isinstance(t, Text):
            continue
        if skip_title and t.position[0] >= x0 - 1e-6 and t.position[1] <= y1 + 1e-6:
            continue                                   # title block and strip
        out.append((t.text, BBox2D.of_primitives([t])))
    return out


def _boxes_overlap(a, b):
    return a.xmin < b.xmax and a.xmax > b.xmin and a.ymin < b.ymax and a.ymax > b.ymin


def _overlapping_texts(sheet):
    boxes = _text_boxes(sheet)
    return [(boxes[i][0], boxes[j][0]) for i in range(len(boxes)) for j in range(i + 1, len(boxes))
            if _boxes_overlap(boxes[i][1], boxes[j][1])]


def _seg_hits_box(seg, box):
    """numpy segment-vs-rectangle clip (shapely-free)."""
    a, b = np.asarray(seg.a, float), np.asarray(seg.b, float)
    d = b - a
    t0, t1 = 0.0, 1.0
    for p, q in ((-d[0], a[0] - box.xmin), (d[0], box.xmax - a[0]), (-d[1], a[1] - box.ymin), (d[1], box.ymax - a[1])):
        if abs(p) < 1e-12:
            if q <= 0:
                return False
            continue
        t = q / p
        if p < 0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
        if t0 >= t1:
            return False
    return True


# ---- issue 2: values from the exact model extents, mismatch is a returned flag --

def test_plan_overall_values_come_from_model_extents_not_the_view_bbox():
    views = _plate_views()
    control = _plan(views=views, hole_positions=False)
    assert control.flags == []                                   # control: boxes agree
    v = views["top"]
    # the top view's box 5 % too big (an arc boxed as a full circle, QA files 12 / 16)
    views["top"] = ProjectedView(v.spec, v.visible, v.hidden, BBox2D(0, 0, 105, 63), "hlr_exact")
    plan = _plan(views=views, hole_positions=False)
    depth = [d for d in _linear(plan) if d.view == "top" and d.angle_deg == 90.0]
    assert len(depth) == 1 and depth[0].value == pytest.approx(60.0) and depth[0].text == "60"
    flags = [f for f in plan.flags if f[0] == "dimension_extent_mismatch"]
    assert len(flags) == 1 and flags[0][1] == "warning" and "depth in view top" in flags[0][2]
    # 0.3 mm (a discretised spline falling short) stays below the 0.5 mm flag threshold
    views["top"] = ProjectedView(v.spec, v.visible, v.hidden, BBox2D(0, 0, 100, 59.7), "hlr_exact")
    short = _plan(views=views, hole_positions=False)
    assert short.flags == [] and [d.text for d in _linear(short) if d.view == "top"] == ["60"]


def test_compose_forwards_the_planner_flags_to_the_layout_flags():
    # review of issue 2: the mismatch must reach the report through LayoutResult.flags, which the
    # pipeline already reports; a flag that stays on the plan is only logged
    views = _plate_views()
    v = views["top"]
    views["top"] = ProjectedView(v.spec, v.visible, v.hidden, BBox2D(0, 0, 105, 63), "hlr_exact")
    plan = _plan(views=views, hole_positions=False)
    assert [f[0] for f in plan.flags] == ["dimension_extent_mismatch"]      # precondition
    res, _, _ = _compose(plan=plan)
    codes = [f[0] for f in res.flags]
    assert codes.count("dimension_extent_mismatch") == 1
    assert tuple(plan.flags[0]) in [tuple(f) for f in res.flags]
    control, _, _ = _compose(plan=_plan(hole_positions=False))
    assert "dimension_extent_mismatch" not in [f[0] for f in control.flags]


def test_plan_arc_silhouette_gives_the_exact_overall_width():
    # screw-head analogue: dome arc R 55 centred 45 below the base chord (QA file 12: 109.75 vs 42.3)
    half = math.sqrt(55.0 ** 2 - 45.0 ** 2)
    a0 = math.degrees(math.asin(45.0 / 55.0))
    from src.drawing.model import Arc
    prims = [Segment((-half, 0.0), (half, 0.0)), Arc((0.0, -45.0), 55.0, a0, 180.0 - a0)]
    bb = BBox2D.of_primitives(prims)
    assert bb.width == pytest.approx(2 * half) and bb.height == pytest.approx(10.0)
    views = {"front": ProjectedView(VIEWS["front"], prims, [], bb, "hlr_exact")}
    plan = plan_dimensions(views, [], DrawingOptions(), ST, np.array([2 * half, 2 * half, 10.0]))
    by = {d.angle_deg: d for d in _linear(plan)}
    assert by[0.0].text == format_value(2 * half) and by[90.0].text == "10"
    assert plan.flags == []
    # the dome top is anchored on the arc's 90 degree quadrant point, not a bbox corner
    assert by[90.0].p2 == pytest.approx((0.0, 10.0))


def test_brep_dome_on_a_cylinder_boxes_exactly_end_to_end():
    """Issue 2 through the real B-rep route: a sphere cap (R 55, base circle R 31.62) fused on a
    R 20 cylinder, projected from all four sides by brep_views. The cap silhouette arcs set the
    width and the height; boxed as full circles (the QA file 12 defect) or with swapped ends
    (a CW arc, the ``_SWAP_CW_ARCS`` case) the box would run to R 55."""
    pytest.importorskip("OCP")
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Common, BRepAlgoAPI_Fuse
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakeSphere
    from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt
    from src.drawing.brep_views import project_brep
    from src.drawing.model import Arc
    sph = BRepPrimAPI_MakeSphere(gp_Pnt(0, 0, -35.0), 55.0).Shape()
    cap = BRepAlgoAPI_Common(sph, BRepPrimAPI_MakeBox(gp_Pnt(-60, -60, 10.0), 120.0, 120.0, 60.0).Shape()).Shape()
    cyl = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), 20.0, 10.0).Shape()
    shape = BRepAlgoAPI_Fuse(cyl, cap).Shape()
    half = math.sqrt(55.0 ** 2 - 45.0 ** 2)
    ext = np.array([2 * half, 2 * half, 20.0])
    for side in ("front", "back", "right", "left"):
        v = project_brep(shape, VIEWS[side], exact=True, deflection=0.05)
        arcs = [p for p in v.visible if isinstance(p, Arc)]
        assert arcs and all(abs(a.radius - 55.0) < 1e-6 for a in arcs), side     # precondition: exact arcs
        full = BBox2D.of_primitives([Circle(a.center, a.radius) for a in arcs])
        assert full.width > 100.0, side                                          # what a full-circle box gives
        bb = BBox2D.of_primitives(list(v.visible) + list(v.hidden))
        assert (bb.width, bb.height) == pytest.approx((2 * half, 20.0), abs=1e-6), side
        assert (v.bbox.width, v.bbox.height) == pytest.approx((2 * half, 20.0), abs=1e-6), side
        for a in arcs:                                                           # each arc boxes inside
            ab = BBox2D.of_primitives([a])
            assert ab.ymin >= 10.0 - 1e-6 and ab.ymax <= 20.0 + 1e-6, (side, a)
        plan = plan_dimensions({side: v}, [], DrawingOptions(), ST, ext)
        assert plan.flags == [], (side, plan.flags)
        assert sorted(d.text for d in _linear(plan)) == sorted([format_value(2 * half), "20"]), side
    views = _plate_views()
    v = views["top"]
    # the right edge drawn 0.3 mm short (spline discretisation): the view bbox ends at 99.7
    vis = [Segment((0, 0), (99.7, 0)), Segment((99.7, 0), (99.7, 60)), Segment((99.7, 60), (0, 60)),
           Segment((0, 60), (0, 0))] + [p for p in v.visible if isinstance(p, Circle)]
    views["top"] = ProjectedView(v.spec, vis, [], BBox2D(0, 0, 99.7, 60), "hlr_exact")
    feats = [_hole(92.0, 52.0, 3.25)]
    no_bounds = plan_dimensions(views, feats, DrawingOptions(), ST, EXTENTS)
    x_no = [d.text for d in _linear(no_bounds) if d.view == "top" and d.angle_deg == 0.0]
    assert x_no == ["7.7"]                                       # precondition: the short bbox shows
    bounds = np.array([[0.0, 0.0, 0.0], [100.0, 60.0, 15.0]])
    exact = plan_dimensions(views, feats, DrawingOptions(), ST, EXTENTS, model_bounds=bounds)
    x_ex = [d for d in _linear(exact) if d.view == "top" and d.angle_deg == 0.0]
    assert [d.text for d in x_ex] == ["8"] and x_ex[0].p1[0] == pytest.approx(100.0)


# ---- issue 23: extension lines start at real geometry, positions from the nearer edge --

def test_plan_overall_dims_anchor_at_real_geometry_points():
    front_p = _rect(0, 0, 40, 10) + _rect(60, 20, 100, 50)      # two bodies, L arrangement
    top_p = _rect(0, 0, 40, 30) + _rect(60, 10, 100, 60)
    views = {"front": ProjectedView(VIEWS["front"], front_p, [], BBox2D.of_primitives(front_p), "hlr_exact"),
             "top": ProjectedView(VIEWS["top"], top_p, [], BBox2D.of_primitives(top_p), "hlr_exact")}
    plan = plan_dimensions(views, [], DrawingOptions(), ST, np.array([100.0, 60.0, 50.0]))
    lin = _linear(plan)
    assert len(lin) == 3
    for d in lin:
        ends = np.array([p for s in views[d.view].visible for p in (s.a, s.b)], float)
        for p in (d.p1, d.p2):
            assert np.min(np.linalg.norm(ends - np.asarray(p), axis=1)) < 1e-6, (d.view, d.text, p)
    width = next(d for d in lin if d.view == "front" and d.angle_deg == 0.0)
    assert width.p2 == pytest.approx((100.0, 20.0))             # not the empty corner (100, 0)
    height = next(d for d in lin if d.view == "front" and d.angle_deg == 90.0)
    assert height.p2 == pytest.approx((60.0, 50.0))             # not the empty corner (0, 50)


def test_plan_hole_positions_from_the_nearer_edge_keep_extension_lines_short():
    views = _box_views(100.0, 60.0, 15.0, iso=False)
    top = views["top"]
    views["top"] = ProjectedView(top.spec, list(top.visible) + [Circle((90.0, 50.0), 3.0)], [], top.bbox, "hlr_exact")
    plan = plan_dimensions(views, [_hole(90.0, 50.0, 3.0)], DrawingOptions(), ST, EXTENTS)
    pos = [(d, b) for d, b in zip(plan.dims, plan.bands)
           if d.kind == "linear" and d.view == "top" and round(d.value, 6) == 10.0]
    assert len(pos) == 2
    sides = sorted(b[0] for _, b in pos)
    assert sides == ["right", "top"]
    for d, (side, _) in pos:
        nrm = 1 if d.angle_deg == 0.0 else 0          # the extension runs across the dimension line
        line = 60.0 if side == "top" else 100.0
        ext = max(abs(line - d.p1[nrm]), abs(line - d.p2[nrm]))
        assert ext < 0.5 * min(100.0, 60.0), (d.text, side, ext)


def _positions_by_axis(plan):
    """{axis: {(datum edge coordinate, value text)}} of the top view's hole positions."""
    out = {0: set(), 1: set()}
    for d in _linear(plan):
        if d.view != "top" or round(d.value, 6) in (100.0, 60.0):
            continue
        axis = 0 if d.angle_deg == 0.0 else 1
        out[axis].add((round(float(d.p1[axis]), 6), d.text))
    return out


def test_plan_hole_positions_use_one_datum_per_view_and_axis():
    # review of issue 23: one group near the bottom-left, one near the top-right corner. The
    # nearer-edge-per-hole rule measured X '10' from the left and '15' from the right, Y '10'
    # from the bottom and '10' from the top: two datums per axis in one view.
    views = _box_views(100.0, 60.0, 15.0, iso=False)
    holes = [_hole(10.0, 10.0, 3.0), _hole(85.0, 50.0, 2.0)]
    near = {0: {h.center[0] < 50.0 for h in holes}, 1: {h.center[1] < 30.0 for h in holes}}
    assert near == {0: {True, False}, 1: {True, False}}          # precondition: nearer edges differ
    pos = _positions_by_axis(plan_dimensions(views, holes, DrawingOptions(), ST, EXTENTS))
    assert {e for e, _ in pos[0]} == {0.0} and {e for e, _ in pos[1]} == {0.0}   # tie: left, bottom
    assert pos[0] == {(0.0, "10"), (0.0, "85")} and pos[1] == {(0.0, "10"), (0.0, "50")}
    # a third hole near the top-right gives that corner the majority: every position from there
    holes3 = holes + [_hole(90.0, 45.0, 1.5)]
    pos3 = _positions_by_axis(plan_dimensions(views, holes3, DrawingOptions(), ST, EXTENTS))
    assert {e for e, _ in pos3[0]} == {100.0} and {e for e, _ in pos3[1]} == {60.0}
    assert {t for _, t in pos3[0]} == {"90", "15", "10"} and {t for _, t in pos3[1]} == {"50", "10", "15"}


# ---- issue 9: coaxial clusters (counterbores, stepped bores) -----------------

_CB = [(15.0, 15.0), (85.0, 15.0), (15.0, 45.0), (85.0, 45.0)]


def _cb_views():
    views = _box_views(100.0, 60.0, 20.0, iso=False)
    top = views["top"]
    circles = [Circle(c, r) for c in _CB for r in (5.0, 8.0)]
    views["top"] = ProjectedView(top.spec, list(top.visible) + circles, [], top.bbox, "hlr_exact")
    return views


def _cb_features():
    out = []
    for x, y in _CB:
        out.append(_hole(x, y, 5.0, z=7.5, length=15.0))          # through part, z 0..15
        out.append(_hole(x, y, 8.0, z=17.5, length=5.0))          # counterbore, z 15..20
    return out


def test_plan_counterbore_is_one_feature_with_one_callout():
    plan = plan_dimensions(_cb_views(), _cb_features(), DrawingOptions(), ST, np.array([100.0, 60.0, 20.0]))
    d = _diam(plan)
    assert len(d) == 1
    assert d[0].text == "4 x Ø10 / Ø16 counterbore" and d[0].count == 4
    assert d[0].value == pytest.approx(10.0) and d[0].radius == pytest.approx(8.0)   # arrow on the Ø16
    keys = [(x.view, x.angle_deg, x.text) for x in _linear(plan)]
    assert len(keys) == len(set(keys)), keys                     # no duplicated position dims
    assert len([x for x in _linear(plan) if x.view == "top"]) == 3     # depth + one X + one Y
    u = [e for e in plan.undimensioned if e["kind"] == "counterbore_depth"]
    assert len(u) == 1 and u[0]["reason"] == "not_supported" and u[0]["count"] == 4


def test_plan_stepped_bore_and_same_edge_position_dedupe():
    views = _cb_views()
    stepped = [_hole(50.0, 30.0, 3.0, z=5.0, length=10.0), _hole(50.0, 30.0, 5.0, z=12.5, length=5.0),
               _hole(50.0, 30.0, 8.0, z=2.5, length=5.0)]
    plan = plan_dimensions(views, stepped, DrawingOptions(), ST, np.array([100.0, 60.0, 20.0]))
    assert [x.text for x in _diam(plan)] == ["Ø6 / Ø10 / Ø16 stepped"]
    assert [e["kind"] for e in plan.undimensioned] == ["step_depth"]
    # two groups whose callout circles share x = 90 (10 from the right edge): one X dimension
    feats = [_hole(90.0, 50.0, 3.0), _hole(90.0, 10.0, 3.0), _hole(90.0, 40.0, 2.0), _hole(90.0, 25.0, 2.0)]
    p2 = plan_dimensions(_box_views(100.0, 60.0, 15.0, iso=False), feats, DrawingOptions(), ST, EXTENTS)
    callouts = _diam(p2)
    assert len(callouts) == 2 and {c.center[0] for c in callouts} == {90.0}   # precondition: same x
    xs = [x for x in _linear(p2) if x.view == "top" and x.angle_deg == 0.0]
    assert [x.text for x in xs] == ["10"]
    keys = [(x.view, x.angle_deg, x.text) for x in _linear(p2)]
    assert len(keys) == len(set(keys))


def test_plan_counterbore_from_a_real_occt_shape():
    pytest.importorskip("OCP")
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
    from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt
    from src.drawing.brep_features import cylinder_features
    shape = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 100.0, 60.0, 20.0).Shape()
    for x, y in _CB:
        thru = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(x, y, -1.0), gp_Dir(0, 0, 1)), 5.0, 22.0).Shape()
        cb = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(x, y, 15.0), gp_Dir(0, 0, 1)), 8.0, 6.0).Shape()
        shape = BRepAlgoAPI_Cut(shape, thru).Shape()
        shape = BRepAlgoAPI_Cut(shape, cb).Shape()
    feats = cylinder_features(shape)
    holes = [f for f in feats if f.kind == "hole"]
    assert sorted(round(f.radius, 6) for f in holes) == [5.0] * 4 + [8.0] * 4      # precondition
    plan = plan_dimensions(_cb_views(), feats, DrawingOptions(), ST, np.array([100.0, 60.0, 20.0]))
    d = _diam(plan)
    assert len(d) == 1 and "Ø10" in d[0].text and "Ø16" in d[0].text and "counterbore" in d[0].text
    keys = [(x.view, x.angle_deg, round(x.value, 6)) for x in _linear(plan)]
    assert len(keys) == len(set(keys))


# ---- issue 8: distinct circles and leader angles, no overlapping texts ----

def _crowded():
    views = _box_views(120.0, 80.0, 20.0, iso=False)
    holes = [_hole(100.0, 60.0, 5.0), _hole(110.0, 70.0, 5.0), _hole(90.0, 70.0, 3.0), _hole(110.0, 50.0, 3.0),
             _hole(95.0, 50.0, 2.0), _hole(80.0, 70.0, 2.0), _hole(105.0, 40.0, 1.5)]
    top = views["top"]
    views["top"] = ProjectedView(top.spec, list(top.visible) + [Circle((h.center[0], h.center[1]), h.radius)
                                                                for h in holes], [], top.bbox, "hlr_exact")
    return views, holes


def test_plan_callout_text_runs_over_the_view_at_the_hinted_scale():
    feats = [h for x in (8.0, 60.0) for h in (_hole(x, 52.0, 3.0, z=5.0, length=10.0),
                                               _hole(x, 52.0, 5.0, z=12.5, length=5.0))]
    at_1 = plan_dimensions(_plate_views(), feats, DrawingOptions(), ST, EXTENTS, scale_hint=1.0)
    at_50 = plan_dimensions(_plate_views(), feats, DrawingOptions(), ST, EXTENTS, scale_hint=50.0)
    assert [d.text for d in _diam(at_1)] == ["2 x Ø6 / Ø10 counterbore"]
    # at 1:1 the 50 mm text would stick out past the right edge from the x = 60 circle
    assert _diam(at_1)[0].center == pytest.approx((8.0, 52.0))
    # at 50:1 it is 1 view mm wide: the circle nearest the top-right corner wins the tie
    assert _diam(at_50)[0].center == pytest.approx((60.0, 52.0))
    assert _diam(at_1)[0].leader_angle_deg == _diam(at_50)[0].leader_angle_deg == 45.0


def test_plan_callouts_get_distinct_circles_and_leader_angles():
    views, holes = _crowded()
    plan = plan_dimensions(views, holes, DrawingOptions(), ST, np.array([120.0, 80.0, 20.0]))
    d = _diam(plan)
    assert len(d) == 4                                            # precondition: at least 3 callouts
    assert len({tuple(x.center) for x in d}) == 4                 # one circle per group
    assert {x.leader_angle_deg for x in d} <= {45.0, 135.0, -45.0, -135.0}
    assert len({x.leader_angle_deg for x in d}) >= 2              # not all on one fixed 45 degrees
    # short leaders: the lone Ø3 near the right edge points right, not across the plate
    lone = next(x for x in d if x.count == 1)
    assert math.cos(math.radians(lone.leader_angle_deg)) > 0
    # no two leaders cross (out to 30 mm beyond the circle)
    from src.drawing.dimensions import _segments_cross
    segs = []
    for x in d:
        u = np.array([math.cos(math.radians(x.leader_angle_deg)), math.sin(math.radians(x.leader_angle_deg))])
        c = np.asarray(x.center, float)
        segs.append((c + x.radius * u, c + (x.radius + 30.0) * u))
    assert not any(_segments_cross(segs[i], segs[j]) for i in range(4) for j in range(i + 1, 4))


def _cluster():
    """Four two-hole groups whose circles nearest the top-right corner sit 8 mm apart."""
    views = _box_views(120.0, 80.0, 20.0, iso=False)
    holes = [_hole(112.0, 72.0, 2.0), _hole(20.0, 20.0, 2.0), _hole(104.0, 72.0, 1.5), _hole(20.0, 40.0, 1.5),
             _hole(112.0, 64.0, 1.0), _hole(40.0, 20.0, 1.0), _hole(104.0, 64.0, 1.25), _hole(40.0, 40.0, 1.25)]
    top = views["top"]
    views["top"] = ProjectedView(top.spec, list(top.visible) + [Circle((h.center[0], h.center[1]), h.radius)
                                                                for h in holes], [], top.bbox, "hlr_exact")
    return views, holes


def _old_style_callouts(plan, holes, corner):
    """The QA round 1 planner: every callout on its group's circle nearest ``corner``, leader at 45."""
    import dataclasses
    dims = []
    for d in plan.dims:
        if d.kind == "diameter":
            cs = [(float(h.center[0]), float(h.center[1])) for h in holes if abs(2.0 * h.radius - d.value) < 1e-9]
            c = min(cs, key=lambda p: math.hypot(corner[0] - p[0], corner[1] - p[1]))
            d = dataclasses.replace(d, center=c, base=c, leader_angle_deg=45.0)
        dims.append(d)
    return DimensionPlan(dims=dims, undimensioned=plan.undimensioned, bands=plan.bands)


def test_compose_crowded_callouts_never_overlap(monkeypatch):
    import src.drawing.layout as layout_mod
    views, holes = _cluster()
    plan = plan_dimensions(views, holes, DrawingOptions(), ST, np.array([120.0, 80.0, 20.0]))
    assert len(_diam(plan)) == 4
    old = _old_style_callouts(plan, holes, (120.0, 80.0))
    # precondition: placed the QA round 1 way and composed without the collision pass, the
    # callout texts of this fixture overlap on the sheet (measured: 4 overlapping pairs)
    monkeypatch.setattr(layout_mod, "_resolve_text_collisions", lambda *a, **k: {}, raising=False)
    bare, _, _ = _compose(views=views, section=None, plan=old)
    monkeypatch.undo()
    assert len(_overlapping_texts(bare.sheet)) >= 2
    # the real planner and the real collision pass: no overlap, no flag
    for p in (plan, old):                                          # the pass alone also clears the old plan
        res, _, _ = _compose(views=views, section=None, plan=p)
        assert len([x for x in res.sheet.dimensions if x.kind == "diameter"]) == 4
        assert _overlapping_texts(res.sheet) == []
        assert "layout_text_collision" not in [f[0] for f in res.flags]


def test_compose_moves_coincident_callouts_apart():
    # the QA round 1 failure: two callouts on the same circle with the same 45 degree leader
    views = _plate_views()
    plan = _plan(views=views, hole_positions=False)
    dims = [d for d in plan.dims if d.kind == "linear"] + [
        Dimension("diameter", "4 x Ø6.5", 6.5, center=(92.0, 52.0), radius=3.25, base=(92.0, 52.0),
                  leader_angle_deg=45.0, view="top", count=4),
        Dimension("diameter", "4 x Ø12.7", 12.7, center=(92.0, 52.0), radius=3.25, base=(92.0, 52.0),
                  leader_angle_deg=45.0, view="top", count=4)]
    bands = [b for d, b in zip(plan.dims, plan.bands) if d.kind == "linear"] + [None, None]
    crowded = DimensionPlan(dims=dims, undimensioned=[], bands=bands)
    # precondition: rendered as planned, the two texts coincide
    t = [next(i for i in render_dimension(d, ST) if isinstance(i, Text)) for d in dims[-2:]]
    assert _boxes_overlap(BBox2D.of_primitives([t[0]]), BBox2D.of_primitives([t[1]]))
    res, _, _ = _compose(plan=crowded)
    assert _overlapping_texts(res.sheet) == []
    assert res.sheet.meta["text_collisions"]["callouts_moved"] >= 1
    assert "layout_text_collision" not in [f[0] for f in res.flags]


def test_compose_without_the_collision_pass_would_overlap(monkeypatch):
    """No-op control for the two tests above: the pass is what separates the texts."""
    import src.drawing.layout as layout_mod
    monkeypatch.setattr(layout_mod, "_resolve_text_collisions", lambda *a, **k: {})
    views = _plate_views()
    plan = _plan(views=views, hole_positions=False)
    dims = [d for d in plan.dims if d.kind == "linear"] + [
        Dimension("diameter", "4 x Ø6.5", 6.5, center=(92.0, 52.0), radius=3.25, base=(92.0, 52.0),
                  leader_angle_deg=45.0, view="top", count=4),
        Dimension("diameter", "4 x Ø12.7", 12.7, center=(92.0, 52.0), radius=3.25, base=(92.0, 52.0),
                  leader_angle_deg=45.0, view="top", count=4)]
    bands = [b for d, b in zip(plan.dims, plan.bands) if d.kind == "linear"] + [None, None]
    res, _, _ = _compose(plan=DimensionPlan(dims=dims, undimensioned=[], bands=bands))
    assert _overlapping_texts(res.sheet)
    # QA round 2 issue 10: leftover overlaps are a warning (was info)
    assert ("layout_text_collision", "warning") in [f[:2] for f in res.flags]


# ---- issue 4: the cutting-line connector does not run through a dimension text --

def test_compose_cutting_line_avoids_the_depth_dimension_text():
    views = _box_views(60.0, 20.0, 30.0, iso=False)
    loops = [np.array([[0, 0], [60, 0], [60, 30], [0, 30]], float)]
    sec = SectionView(VIEWS["front"], np.array([30.0, 10.0, 15.0]), np.array([0.0, -1.0, 0.0]),
                      loops, None, label="A", marker_view="top")
    res, _, plan = _compose(views=views, section=sec)
    depth = [d for d in res.sheet.dimensions if d.view == "top" and d.angle_deg == 90.0]
    assert len(depth) == 1 and depth[0].text == "20"
    # precondition: the plane is at mid-depth = the depth dimension's midpoint, so the centred
    # text (text_offset 0) sits on the cutting line's extension
    y_cut = res.placements["top"][0][1] + res.scale * 10.0
    assert 0.5 * (depth[0].p1[1] + depth[0].p2[1]) == pytest.approx(y_cut)
    marker = [i for i in res.sheet.items if isinstance(i, Segment) and i.style in (LineStyle.CUTTING, LineStyle.CENTER)]
    centred = next(i for i in render_dimension(Dimension(**{**depth[0].__dict__, "text_offset": 0.0}), ST)
                   if isinstance(i, Text))
    # QA round 2 issue 14: with bands on that side the wide dash ends before the first dimension
    # line and no connector runs past the bands (round 1 crossed this text and shifted it), so the
    # centred text is clear and stays centred
    assert not any(_seg_hits_box(m, BBox2D.of_primitives([centred])) for m in marker)
    assert depth[0].text_offset == 0.0
    for name, box in _text_boxes(res.sheet):
        for m in marker:
            assert not _seg_hits_box(m, box), (name, m)
    # the shifted text stays on its (extended) dimension line
    g = res.sheet.meta["dimension_items"][res.sheet.dimensions.index(depth[0])]
    t = next(res.sheet.items[i] for i in g if isinstance(res.sheet.items[i], Text))
    from src.drawing.dimensions import linear_text_position
    assert t.position == pytest.approx(linear_text_position(depth[0], ST))
    lines = [res.sheet.items[i] for i in g if isinstance(res.sheet.items[i], Segment)
             and abs(res.sheet.items[i].a[0] - res.sheet.items[i].b[0]) < 1e-9
             and abs(res.sheet.items[i].a[0] - depth[0].base[0]) < 1e-9]
    assert len(lines) == 1
    lo, hi = sorted([lines[0].a[1], lines[0].b[1]])
    assert lo - 1e-9 <= t.position[1] <= hi + 1e-9


# ---- issue 7: scale --------------------------------------------------------

def _ftc_boxes():
    return {"front": BBox2D(0, 0, 127, 35), "top": BBox2D(0, 0, 127, 63.5), "right": BBox2D(0, 0, 63.5, 35)}


def test_choose_scale_iso_finds_free_space_at_the_sheet_scale(monkeypatch):
    import src.drawing.layout as layout_mod
    s, info = choose_scale(_ftc_boxes(), {}, SHEETS["A3"], ST, "first", False, False)
    assert s == 1.0                                               # the empty-band case fits 1:1
    s2, info2 = choose_scale(_ftc_boxes(), {}, SHEETS["A3"], ST, "first", True, True)
    assert s2 == 1.0 and info2["iso_scale"] == 1.0
    _assert_iso_clear(info2["rects"], SHEETS["A3"])
    # a 40 x 80 x 35 part on A4: the sweep finds room for the iso view at 1:1 ...
    tall = {"front": BBox2D(0, 0, 40, 35), "top": BBox2D(0, 0, 40, 80), "right": BBox2D(0, 0, 80, 35)}
    s4, info4 = choose_scale(tall, {}, SHEETS["A4"], ST, "first", False, True)
    assert s4 == 1.0 and info4["iso_scale"] == 1.0 and info4["iso_mode"] == "free"
    _assert_iso_clear(info4["rects"], SHEETS["A4"])
    # ... control: with the eight fixed spots only it dropped to 1:2 (measured)
    monkeypatch.setattr(layout_mod, "_float_spots", lambda *a, **k: None)
    s5, info5 = choose_scale(tall, {}, SHEETS["A4"], ST, "first", False, True)
    assert s5 == 1.0 and info5["iso_scale"] == 0.5


def test_callout_room_is_the_exact_rendered_reach():
    # the callout room used to be a fixed reach measured from the circle and ADDED to the
    # bands (measured: the plate on A4 fell to 1:10 with four-sided bands); it is now the
    # rendered callout's reach beyond the outline at each scale, taken as max with the bands
    from src.drawing.layout import _callout_room
    for sheet in ("A3", "A4"):
        res, views, plan = _compose("first", sheet=sheet)
        boxes = {n: v.bbox for n, v in views.items()}
        room = _callout_room(plan, boxes, ST)(res.scale)
        g_top = _geom_box(res, views, "top")
        reach = [0.0, 0.0, 0.0, 0.0]
        for d, g in zip(res.sheet.dimensions, res.sheet.meta["dimension_items"]):
            if d.kind != "diameter":
                continue
            bb = BBox2D.of_primitives([res.sheet.items[i] for i in g])
            _assert_inside(bb, res.placements[d.view][1], d.text)
            for i, v in enumerate((g_top.xmin - bb.xmin, g_top.ymin - bb.ymin, bb.xmax - g_top.xmax,
                                   bb.ymax - g_top.ymax)):
                reach[i] = max(reach[i], v)
        assert max(reach) > 0.0                                   # precondition: callouts reach out
        assert room["top"] == pytest.approx(tuple(reach), abs=1e-6)
    assert res.scale == pytest.approx(1 / 2)                      # A4 (round 2 issue 6: was 1:5)


# ---- issue 26: bosses / fillets recorded, concentric bore without positions --

def test_plan_bosses_and_fillets_go_to_undimensioned():
    fil = [CylFeature(np.array([x, 0.0, 7.5]), np.array([0.0, 0.0, 1.0]), 2.0, 15.0, "fillet", False, 1)
           for x in (0.0, 100.0)]
    plan = _plan(features=_features(fil))
    u = {e["kind"]: e for e in plan.undimensioned}
    assert u["boss_group"]["reason"] == "not_supported" and u["boss_group"]["radius"] == pytest.approx(15.0)
    assert u["fillet_group"]["reason"] == "not_supported" and u["fillet_group"]["count"] == 2


def test_plan_concentric_bore_gets_no_position_dims():
    def views_for(outline):
        front = ProjectedView(VIEWS["front"], _rect(0, 0, 40, 10), [], BBox2D(0, 0, 40, 10), "hlr_exact")
        top = ProjectedView(VIEWS["top"], outline + [Circle((20.0, 20.0), 5.0)], [], BBox2D(0, 0, 40, 40), "hlr_exact")
        return {"front": front, "top": top}
    bore = [_hole(20.0, 20.0, 5.0, z=5.0, length=10.0)]
    washer = plan_dimensions(views_for([Circle((20.0, 20.0), 20.0)]), bore, DrawingOptions(), ST,
                             np.array([40.0, 40.0, 10.0]))
    assert [d.text for d in _diam(washer)] == ["Ø10"]
    assert [d.text for d in _linear(washer) if d.view == "top"] == ["40"]        # the depth only
    # control: the same bore in a square plate keeps its two position dims
    square = plan_dimensions(views_for(_rect(0, 0, 40, 40)), bore, DrawingOptions(), ST,
                             np.array([40.0, 40.0, 10.0]))
    assert sorted(d.text for d in _linear(square) if d.view == "top") == ["20", "20", "40"]


# ---- issue 29: small section regions ------------------------------------------

def test_section_hatch_small_region_gets_lines_or_fill():
    from src.drawing.geometry2d import hatch_lines
    from src.drawing.layout import _section_items
    big = np.array([[0, 0], [80, 0], [80, 40], [0, 40]], float)
    x0 = 97.5
    small = np.array([[x0, 0], [x0 + 4, 0], [x0 + 4, 4], [x0, 4]], float)
    tiny = np.array([[120, 0], [121.5, 0], [121.5, 1.5], [120, 1.5]], float)
    spacing = hatch_spacing(ST, 40.0)
    assert spacing == 3.0
    assert len(hatch_lines([small], spacing, 45.0)) <= 1          # precondition: the old single spacing
    sec = SectionView(VIEWS["front"], np.zeros(3), np.array([0.0, -1.0, 0.0]), [big, small, tiny], None)
    # QA round 2 issue 3: the fill is decided per body, so the tiny piece is a body of its own here
    # (in one body with the big region it is hatched, see test_section_fill_is_decided_per_body)
    sec.loop_body = [0, 0, 1]
    items, stats = _section_items(sec, (0.0, 0.0), 1.0, ST, BBox2D(0, 0, 121.5, 40))
    hatches = [i for i in items if isinstance(i, HatchLines)]
    assert len(hatches) == 2
    by_loop = {round(float(h.loops[0][:, 0].min()), 6): h for h in hatches}
    assert len(by_loop[x0].segments) >= 2
    assert stats["hatch_spacings"][0] == 3.0                     # the big region keeps the section spacing
    fills = [i for i in items if isinstance(i, Polygon)]
    assert len(fills) == 1 and np.allclose(fills[0].points, tiny) and stats["filled_regions"] == 1
    assert stats["hatch_spacing"] == 3.0 and stats["loops"] == 3


def test_section_thin_strip_is_filled_at_any_angle():
    # review of issue 29: the fill test used the shorter bbox side, so a 1.5 mm strip at 45 degrees
    # (bbox about 43.5 x 43.5) got 0.7 mm hatching while the same strip lying flat was filled
    from src.drawing.layout import _min_width, _section_items
    big = np.array([[0, 0], [80, 0], [80, 40], [0, 40]], float)
    u, n = np.array([1.0, 1.0]) / math.sqrt(2.0), np.array([-1.0, 1.0]) / math.sqrt(2.0)
    o = np.array([100.0, 0.0])
    diag = np.array([o, o + 60.0 * u, o + 60.0 * u + 1.5 * n, o + 1.5 * n])
    flat = np.array([[0, 50], [60, 50], [60, 51.5], [0, 51.5]], float)
    db = BBox2D.from_points(diag)
    assert min(db.width, db.height) > 40.0                       # precondition: the bbox rule hatches it
    assert _min_width(diag) == pytest.approx(1.5) and _min_width(flat) == pytest.approx(1.5)
    assert _min_width(big) == pytest.approx(40.0)
    for strip in (diag, flat):
        sec = SectionView(VIEWS["front"], np.zeros(3), np.array([0.0, -1.0, 0.0]), [big, strip], None)
        sec.loop_body = [0, 1]                    # round 2 issue 3: the strip is a body of its own
        items, stats = _section_items(sec, (0.0, 0.0), 1.0, ST, BBox2D.from_points(np.vstack([big, strip])))
        fills = [i for i in items if isinstance(i, Polygon)]
        assert len(fills) == 1 and np.allclose(fills[0].points, strip) and stats["filled_regions"] == 1
        assert len([i for i in items if isinstance(i, HatchLines)]) == 1          # only the big region


# =========================================================================
# QA round 2 (task R4-layout): issues 3, 6, 7, 10, 13, 14, 17 (room), 28, 30
# =========================================================================

def _sec_of(loops, bodies=None, spec="front"):
    sec = SectionView(VIEWS[spec], np.zeros(3), -VIEWS[spec].n, [np.asarray(lp, float) for lp in loops], None)
    if bodies is not None:
        sec.loop_body = list(bodies)
    return sec


def _ring(r_out, r_in, n=256, c=(0.0, 0.0)):
    a = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    u = np.column_stack([np.cos(a), np.sin(a)])
    return [np.asarray(c, float) + r_out * u, np.asarray(c, float) + r_in * u[::-1]]


def _hatch_angles(h):
    d = h.segments[:, 1, :] - h.segments[:, 0, :]
    return set(np.round(np.degrees(np.arctan2(d[:, 1], d[:, 0])) % 180.0, 6))


# ---- issue 3: section fill per body from the mean thickness 2A/P --------------------

STRIP = [[0, 0], [20, 0], [20, 1.5], [0, 1.5]]
U_WALL = [[30, 0], [50, 0], [50, 20], [48.5, 20], [48.5, 1.5], [31.5, 1.5], [31.5, 20], [30, 20]]


def test_section_fill_is_decided_per_body_from_the_mean_thickness():
    from src.drawing.layout import _min_width, _section_items
    # precondition: the round 1 per-region rule (bbox / hull caliper < 2 mm) fills the straight
    # piece and hatches the U piece of the same 1.5 mm wall (mixed, the 08 / 14 defect)
    assert _min_width(np.array(STRIP, float)) < 2.0 < _min_width(np.array(U_WALL, float))
    items, stats = _section_items(_sec_of([STRIP, U_WALL]), (0.0, 0.0), 1.0, ST, BBox2D(0, 0, 50, 20))
    fills = [i for i in items if isinstance(i, Polygon)]
    hatched = [i for i in items if isinstance(i, HatchLines) and len(i.segments)]
    assert len(fills) == 2 and not hatched
    assert stats["fill"] == "solid" and stats["filled_regions"] == 2
    assert stats["filled_area_fraction"] == pytest.approx(1.0)
    assert stats["body_thickness"][0] == pytest.approx(2 * (30 + 85.5) / (43 + 117), rel=1e-6)


def test_section_thin_region_in_a_thick_body_is_hatched_not_filled():
    from src.drawing.layout import _section_items
    big = [[0, 0], [80, 0], [80, 40], [0, 40]]
    strip = [[0, 50], [60, 50], [60, 51.5], [0, 51.5]]
    one = _sec_of([big, strip])                                  # no loop_body: one body
    items, stats = _section_items(one, (0.0, 0.0), 1.0, ST, BBox2D(0, 0, 80, 51.5))
    assert not [i for i in items if isinstance(i, Polygon)]      # one part, one treatment
    hatches = [i for i in items if isinstance(i, HatchLines)]
    assert len(hatches) == 2 and all(len(h.segments) >= 2 for h in hatches)
    assert stats["fill"] == "hatch" and stats["hatch_spacings"] == [3.0, pytest.approx(0.7)]
    two = _sec_of([big, strip], bodies=[0, 1])                   # control: the strip as its own body
    items2, stats2 = _section_items(two, (0.0, 0.0), 1.0, ST, BBox2D(0, 0, 80, 51.5))
    assert len([i for i in items2 if isinstance(i, Polygon)]) == 1 and stats2["fill"] == "mixed"


def test_section_thin_ring_is_filled_or_finely_hatched():
    from src.drawing.layout import _section_items
    thin = _sec_of(_ring(40.0, 39.2))
    items, stats = _section_items(thin, (0.0, 0.0), 1.0, ST, BBox2D(-40, -40, 40, 40))
    fills = [i for i in items if isinstance(i, Polygon)]
    hatches = [i for i in items if isinstance(i, HatchLines) and len(i.segments)]
    assert fills or (hatches and max(stats["hatch_spacings"]) <= 0.7 + 1e-9)
    assert stats["body_thickness"][0] == pytest.approx(0.8, rel=0.01)
    # a 4 mm ring is hatched at about t / 3, not at the section spacing (3 mm, round 1: a hull of 80 mm)
    ring4 = _sec_of(_ring(40.0, 36.0))
    items4, stats4 = _section_items(ring4, (0.0, 0.0), 1.0, ST, BBox2D(-40, -40, 40, 40))
    assert hatch_spacing(ST, 80.0) == 3.0
    assert not [i for i in items4 if isinstance(i, Polygon)]
    assert stats4["hatch_spacings"] == [pytest.approx(4.0 / 3.0, rel=0.01)]


# ---- issue 28: one hatch angle / phase per body ---------------------------------------

def test_section_hatch_alternates_per_body():
    from src.drawing.layout import _section_items
    a = [[0, 0], [30, 0], [30, 30], [0, 30]]
    b = [[30, 0], [60, 0], [60, 30], [30, 30]]
    c = [[60, 0], [90, 0], [90, 30], [60, 30]]
    one, _ = _section_items(_sec_of([a, b, c]), (0.0, 0.0), 1.0, ST, BBox2D(0, 0, 90, 30))
    h1 = [i for i in one if isinstance(i, HatchLines)]
    assert len(h1) == 3 and all(_hatch_angles(h) == {45.0} for h in h1)     # control: one body
    three, stats = _section_items(_sec_of([a, b, c], bodies=[7, 3, 9]), (0.0, 0.0), 1.0, ST, BBox2D(0, 0, 90, 30))
    h3 = {round(float(h.loops[0][:, 0].min())): h for h in three if isinstance(h, HatchLines)}
    # bodies ranked by id: 3 -> 45, 7 -> 135, 9 -> 45 with a half-spacing phase shift
    assert _hatch_angles(h3[30]) == {45.0} and _hatch_angles(h3[0]) == {135.0} and _hatch_angles(h3[60]) == {45.0}
    nrm = np.array([-math.sin(math.pi / 4), math.cos(math.pi / 4)])
    sp = stats["hatch_spacing"]
    ph_b = (h3[30].segments[0, 0] @ nrm) % sp
    ph_c = (h3[60].segments[0, 0] @ nrm) % sp
    assert abs(((ph_c - ph_b) % sp) - 0.5 * sp) < 1e-6
    assert stats["bodies"] == 3


# ---- issue 6: alternative section placements before a smaller scale -----------------

def _to220_views():
    return _box_views(28.525, 4.455, 10.25)


def _to220_section():
    loops = [np.array([[0, 0], [28.525, 0], [28.525, 4.455], [0, 4.455]], float)]
    return SectionView(VIEWS["top"], np.array([14.0, 2.2, 5.1]), np.array([0.0, 0.0, 1.0]), loops, None,
                       label="A", marker_view="front")


def _to220_plan(views):
    hole = _hole(24.4, 2.2, 1.87, z=5.125, length=4.455, axis=(0.0, 1.0, 0.0))
    return plan_dimensions(views, [hole], DrawingOptions(), ST, np.array([28.525, 4.455, 10.25]))


def test_choose_scale_tries_other_section_cells(monkeypatch):
    import src.drawing.layout as layout_mod
    # measured: the triage's 12.7 mm deep variant already fits 1:2 with the side cell; at 50.8 mm
    # the side cell costs 1:5 and the front column keeps 1:2
    boxes = {"front": BBox2D(0, 0, 304.8, 203.2), "top": BBox2D(0, 0, 304.8, 50.8),
             "right": BBox2D(0, 0, 50.8, 203.2), "section": BBox2D(0, 0, 304.8, 50.8)}
    bands = {"front": (1, 1, 0, 0), "top": (1, 0, 0, 0)}
    monkeypatch.setattr(layout_mod, "_SECTION_ALTERNATIVES", False)
    s_old, _ = choose_scale(boxes, bands, SHEETS["A3"], ST, "first", True, True)
    monkeypatch.setattr(layout_mod, "_SECTION_ALTERNATIVES", True)
    s, info = choose_scale(boxes, bands, SHEETS["A3"], ST, "first", True, True)
    assert s_old is not None and s_old < 0.5                   # precondition: the side cell costs a step
    assert s >= 0.5
    assert info["section_mode"] in ("cell", "free")
    assert info["scale_candidates"][-1][:2] == (s, True)
    assert all(not ok for sc, ok, _ in info["scale_candidates"][:-1])


def test_compose_to220_keeps_5_to_1_with_a_section(monkeypatch):
    import src.drawing.layout as layout_mod
    views = _to220_views()
    plan = _to220_plan(views)
    monkeypatch.setattr(layout_mod, "_SECTION_ALTERNATIVES", False)
    old, _, _ = _compose(views=views, section=_to220_section(), plan=plan)
    monkeypatch.setattr(layout_mod, "_SECTION_ALTERNATIVES", True)
    res, _, _ = _compose(views=views, section=_to220_section(), plan=plan)
    assert old.scale < 5.0                                      # precondition: the side cell drops a step
    assert res.scale == 5.0 and not res.overflow
    rects = {n: r for n, (_, r) in res.placements.items()}
    names = sorted(rects)
    for i, a in enumerate(names):                               # no two view rectangles overlap
        for b in names[i + 1:]:
            assert not _boxes_overlap(rects[a], rects[b]), (a, b)
    assert res.sheet.meta["scale_candidates"][-1][:2] == (5.0, True)


# ---- issue 7: cap position bands at 3 per side when that wins a scale --------------

def _deep_plan(W=2400.0, H=680.0, D=300.0, n_pos=12):
    # measured (probe over W): at W = 2400 the 13 left bands (10 + 7 x 13 = 101 mm) cost 1:10 -> 1:20
    # on A3; capped at 3 position bands (38 mm) the drawing keeps 1:10
    views = _box_views(W, D, H, iso=False)
    fb, tb = views["front"].bbox, views["top"].bbox
    dims = [Dimension("linear", format_value(W), W, p1=(0.0, 0.0), p2=(W, 0.0), base=(W / 2, 0.0),
                      angle_deg=0.0, view="front"),
            Dimension("linear", format_value(H), H, p1=(0.0, 0.0), p2=(0.0, H), base=(0.0, H / 2),
                      angle_deg=90.0, view="front"),
            Dimension("linear", format_value(D), D, p1=(0.0, 0.0), p2=(0.0, D), base=(0.0, D / 2),
                      angle_deg=90.0, view="top")]
    bands = [("bottom", 0), ("left", n_pos), ("left", 0)]
    for i in range(n_pos):
        y = (i + 1) * H / (n_pos + 2)
        dims.append(Dimension("linear", format_value(y), y, p1=(0.0, 0.0), p2=(W / 2, y), base=(0.0, y / 2),
                              angle_deg=90.0, view="front"))
        bands.append(("left", i))
    assert fb.width == W and tb.height == D
    return views, DimensionPlan(dims=dims, undimensioned=[], bands=bands)


def test_compose_caps_deep_position_bands_to_keep_the_scale(monkeypatch):
    import src.drawing.layout as layout_mod
    views, plan = _deep_plan()
    monkeypatch.setattr(layout_mod, "POSITION_BAND_CAP", None)        # no cap: the old behaviour
    old, _, _ = _compose(views=views, section=None, plan=plan)
    monkeypatch.setattr(layout_mod, "POSITION_BAND_CAP", 3)
    res, _, _ = _compose(views=views, section=None, plan=plan)
    assert old.scale == pytest.approx(1 / 20) and len(old.sheet.dimensions) == 15     # precondition
    assert res.scale == pytest.approx(1 / 10)
    pos = [d for d in res.sheet.dimensions if d.view == "front" and d.angle_deg == 90.0 and d.value < 680.0]
    assert len(pos) == 3 and sorted(d.value for d in pos) == pytest.approx([680 / 14 * k for k in (1, 2, 3)])
    overall = {d.text for d in res.sheet.dimensions if d.value in (2400.0, 680.0, 300.0)}
    assert overall == {"2400", "680", "300"}                         # never drop overall dims
    dropped = [u for u in res.plan.undimensioned if u.get("reason") == "layout_space"]
    assert len(dropped) == 9 and all(u["view"] == "front" for u in dropped)
    assert plan.undimensioned == [] and len(plan.dims) == 15         # the caller's plan is untouched
    assert ("layout_dims_dropped", "info") in [f[:2] for f in res.flags]
    # the outermost left band now holds the overall height
    k = band_counts(res.plan)["front"]
    assert k[0] == 4


def test_compose_does_not_cap_when_the_scale_does_not_improve():
    views, plan = _deep_plan(W=86.0, H=68.0, D=30.0)
    res, _, _ = _compose(views=views, section=None, plan=plan)
    assert len(res.sheet.dimensions) == 15 and not [u for u in res.plan.undimensioned
                                                     if u.get("reason") == "layout_space"]


# ---- issue 10: leftover text collisions are a warning ----------------------------

def test_compose_leftover_collision_is_a_warning_and_moves_are_info(monkeypatch):
    import src.drawing.layout as layout_mod
    views = _plate_views()
    plan = _plan(views=views, hole_positions=False)
    dims = [d for d in plan.dims if d.kind == "linear"] + [
        Dimension("diameter", "4 x Ø6.5", 6.5, center=(92.0, 52.0), radius=3.25, base=(92.0, 52.0),
                  leader_angle_deg=45.0, view="top", count=4),
        Dimension("diameter", "4 x Ø12.7", 12.7, center=(92.0, 52.0), radius=3.25, base=(92.0, 52.0),
                  leader_angle_deg=45.0, view="top", count=4)]
    bands = [b for d, b in zip(plan.dims, plan.bands) if d.kind == "linear"] + [None, None]
    crowded = DimensionPlan(dims=dims, undimensioned=[], bands=bands)
    res, _, _ = _compose(plan=crowded)
    codes = {f[0]: f[1] for f in res.flags}
    assert codes.get("layout_text_moved") == "info" and "layout_text_collision" not in codes
    monkeypatch.setattr(layout_mod, "_resolve_text_collisions", lambda *a, **k: {})
    bad, _, _ = _compose(plan=crowded)
    assert _overlapping_texts(bad.sheet)                                 # precondition
    assert ("layout_text_collision", "warning") in [f[:2] for f in bad.flags]
    assert "layout_text_moved" not in [f[0] for f in bad.flags]


def test_callout_leader_never_runs_over_the_marker():
    from src.drawing.layout import _segments_intersect
    assert _segments_intersect((0, 0), (10, 0), (5, -1), (5, 1))
    assert not _segments_intersect((0, 0), (10, 0), (11, -1), (11, 1))
    assert not _segments_intersect((0, 0), (10, 0), (0, 1), (10, 1))
    assert _segments_intersect((0, 0), (10, 0), (5, 0), (12, 0))            # collinear overlap
    views = _box_views(100.0, 60.0, 15.0, iso=False)
    top = views["top"]
    views["top"] = ProjectedView(top.spec, list(top.visible) + [Circle((47.0, 58.0), 1.0)], [], top.bbox,
                                 "hlr_exact")
    call = Dimension("diameter", "Ø2", 2.0, center=(47.0, 58.0), radius=1.0, base=(47.0, 58.0),
                     leader_angle_deg=45.0, view="top")
    plan = DimensionPlan(dims=[call], undimensioned=[], bands=[None])
    sec = SectionView(VIEWS["right"], np.array([50.0, 30.0, 7.5]), np.array([1.0, 0.0, 0.0]),
                      [np.array([[0, 0], [60, 0], [60, 15], [0, 15]], float)], None, marker_view="top")
    res, _, _ = _compose(views=views, section=sec, plan=plan)
    sh = res.sheet
    marker = [i for i in sh.items if isinstance(i, Segment) and i.style == LineStyle.CUTTING]
    (ox, oy), _ = res.placements["top"]
    s = res.scale
    planned = [i for i in render_dimension(_compose_sheet_callout(call, res, views), ST) if isinstance(i, Segment)]
    assert any(_segments_intersect(p.a, p.b, m.a, m.b) for p in planned for m in marker)   # precondition
    g = sh.meta["dimension_items"][0]
    leader = [sh.items[i] for i in g if isinstance(sh.items[i], Segment)]
    for p in leader:
        for m in marker:
            assert not _segments_intersect(p.a, p.b, m.a, m.b), (p, m)
    assert sh.dimensions[0].leader_angle_deg != 45.0 and (ox, oy, s)


def _compose_sheet_callout(d, res, views):
    """The callout as compose would draw it unmoved (the planned 45 degree leader)."""
    from src.drawing.layout import _dim_to_sheet
    (ox, oy), _ = res.placements[d.view]
    s = res.scale
    bb = views[d.view].bbox
    geom = BBox2D(ox + s * bb.xmin, oy + s * bb.ymin, ox + s * bb.xmax, oy + s * bb.ymax)
    return _dim_to_sheet(d, None, (ox, oy), s, geom, ST)


# ---- issue 13: the wide cutting-line ends start outside the outline ----------------

def test_marker_wide_ends_start_outside_the_view():
    from src.drawing.layout import _marker_geometry, _marker_items, _segment_hits_box
    views = _box_views(10.0, 10.0, 2.0, iso=False)
    sec = SectionView(VIEWS["top"], np.array([5.0, 5.0, 1.0]), np.array([0.0, 0.0, 1.0]),
                      [np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)], None, marker_view="top")
    boxes = {k: v.bbox for k, v in views.items()}
    geo = _marker_geometry(sec, views, boxes)
    assert geo is not None and geo[0] == "front"                # the top view sees the plane face-on
    s = 5.0
    geom = BBox2D(0.0, 0.0, s * 10.0, s * 2.0)
    items = _marker_items(geo, (0.0, 0.0), s, ST, "A")
    cutting = [i for i in items if isinstance(i, Segment) and i.style == LineStyle.CUTTING]
    assert len(cutting) == 4
    g = geom.expanded(0.5)
    for c in cutting:
        assert not g.contains(c.a) and not g.contains(c.b), c
        assert not _segment_hits_box(c.a, c.b, g), c
    wide = [c for c in cutting if abs(c.a[1] - c.b[1]) < 1e-9]
    assert sorted(round(min(abs(c.a[0] - x), abs(c.b[0] - x)), 6) for c, x in zip(
        sorted(wide, key=lambda c: c.a[0]), (0.0, 50.0))) == [1.5, 1.5]


# ---- issue 14: the marker keeps off dimension and extension lines ------------------

def _segs_collinear_overlap(p, q, tol=0.1):
    a, b = np.asarray(p.a, float), np.asarray(p.b, float)
    c, d = np.asarray(q.a, float), np.asarray(q.b, float)
    u = b - a
    L = np.linalg.norm(u)
    if L < 1e-9:
        return False
    u /= L
    n = np.array([-u[1], u[0]])
    if abs((c - a) @ n) > tol or abs((d - a) @ n) > tol:
        return False
    lo, hi = sorted([(c - a) @ u, (d - a) @ u])
    return min(hi, L) - max(lo, 0.0) > 1e-6


def _plate_hole_case():
    views = _box_views(100.0, 50.0, 10.0, iso=False)
    top = views["top"]
    views["top"] = ProjectedView(top.spec, list(top.visible) + [Circle((30.0, 25.0), 5.0)], [], top.bbox,
                                 "hlr_exact")
    hole = _hole(30.0, 25.0, 5.0, z=5.0, length=10.0)
    plan = plan_dimensions(views, [hole], DrawingOptions(), ST, np.array([100.0, 50.0, 10.0]))
    sec = SectionView(VIEWS["right"], np.array([30.0, 25.0, 5.0]), np.array([1.0, 0.0, 0.0]),
                      [np.array([[0, 0], [50, 0], [50, 10], [0, 10]], float)], None, label="A", marker_view="top")
    return views, plan, sec


def test_marker_does_not_lie_on_extension_lines_or_cross_dimension_lines():
    from src.drawing.layout import _segments_intersect
    views, plan, sec = _plate_hole_case()
    pos30 = [d for d in plan.dims if d.view == "top" and d.angle_deg == 0.0 and d.text == "30"]
    assert len(pos30) == 1                                       # precondition: the hole position dim
    res, _, _ = _compose(views=views, section=sec, plan=plan)
    sh = res.sheet
    assert sh.meta["marker_view"] == "top"
    dim_idx = {i for g in sh.meta["dimension_items"] for i in g}
    dim_segs = [sh.items[i] for i in dim_idx if isinstance(sh.items[i], Segment)]
    cutting = [i for i in sh.items if isinstance(i, Segment) and i.style == LineStyle.CUTTING]
    center = [i for i in sh.items if isinstance(i, Segment) and i.style == LineStyle.CENTER]
    x_cut = res.placements["top"][0][0] + res.scale * 30.0
    ext30 = [s for s in dim_segs if abs(s.a[0] - x_cut) < 1e-6 and abs(s.b[0] - x_cut) < 1e-6]
    assert ext30                                                 # precondition: an extension line on the trace
    for c in cutting:
        for s in dim_segs:
            assert not _segs_collinear_overlap(c, s), (c, s)
    for c in center:
        for s in dim_segs:
            assert not _segments_intersect(c.a, c.b, s.a, s.b), (c, s)
    # both wide ends are still drawn, on the trace, outside the view
    wide = [c for c in cutting if abs(c.a[0] - c.b[0]) < 1e-9 and abs(c.a[0] - x_cut) < 1e-6]
    assert len(wide) == 2
    # and everything the marker draws stays in the reserved rect of the top view
    rect = res.placements["top"][1]
    for it in cutting + center:
        _assert_inside(BBox2D.of_primitives([it]), rect, ("marker", it))


def test_marker_with_bands_on_its_side_has_no_connector_across_them():
    sec = _section()
    sec.plane_point = np.array([50.0, 22.0, 7.5])                 # off every hole centre and quadrant
    res, views, plan = _compose("first", section=sec)
    sh = res.sheet
    k_left = band_counts(plan)["top"][0]
    assert sh.meta["marker_view"] == "top" and k_left == 3       # precondition: bands on the cut side
    center = [i for i in sh.items if isinstance(i, Segment) and i.style == LineStyle.CENTER]
    assert center == []
    tb = _geom_box(res, views, "top")
    first_line = tb.xmin - ST.dim_first_offset
    wide_left = [i for i in sh.items if isinstance(i, Segment) and i.style == LineStyle.CUTTING
                 and abs(i.a[1] - i.b[1]) < 1e-9 and max(i.a[0], i.b[0]) < tb.xmin]
    assert len(wide_left) == 1
    assert min(wide_left[0].a[0], wide_left[0].b[0]) > first_line   # stops before the first dimension line
    assert max(wide_left[0].a[0], wide_left[0].b[0]) == pytest.approx(tb.xmin - 1.5)
    # the letter sits beyond the arrow head, not on the first dimension line, and no dimension
    # line or text is touched by any marker item (extension lines run parallel to this cutting
    # line; an arrow shaft crossing one at a right angle is accepted)
    letters = [t for t in sh.items if isinstance(t, Text) and t.text == "A"]
    dims = [sh.items[i] for g in sh.meta["dimension_items"] for i in g]
    marker = [i for i in sh.items if getattr(i, "style", None) == LineStyle.CUTTING] + letters
    for m in marker:
        mb = BBox2D.of_primitives([m])
        for d in dims:
            if isinstance(d, Segment) and abs(d.a[1] - d.b[1]) > 1e-9:
                assert not _seg_hits_box(d, mb.expanded(0.1)), (m, d)
            elif isinstance(d, Text):
                assert not _boxes_overlap(BBox2D.of_primitives([d]), mb), (m, d)


# ---- issue 17 (room part): a text wider than its span goes outside, inside the rect --

def test_compose_places_a_wide_text_outside_its_span_and_reserves_it():
    views = _box_views(100.0, 40.0, 15.0, iso=False)
    assert views["front"].bbox.width == 100.0
    dims = [Dimension("linear", "100", 100.0, p1=(0.0, 0.0), p2=(100.0, 0.0), base=(50.0, 0.0),
                      angle_deg=0.0, view="front"),
            Dimension("linear", "2.25", 2.25, p1=(0.0, 0.0), p2=(2.25, 0.0), base=(1.125, 0.0),
                      angle_deg=0.0, view="front")]
    plan = DimensionPlan(dims=dims, undimensioned=[], bands=[("bottom", 1), ("bottom", 0)])
    res, _, _ = _compose(views=views, section=None, plan=plan, sheet="A4")
    sh = res.sheet
    k = [i for i, d in enumerate(sh.dimensions) if d.text == "2.25"][0]
    d = sh.dimensions[k]
    g = sh.meta["dimension_items"][k]
    t = next(sh.items[i] for i in g if isinstance(sh.items[i], Text))
    tbox = BBox2D.of_primitives([t])
    span = abs(d.p2[0] - d.p1[0])
    assert tbox.width + 2 * ST.dim_text_gap > span               # precondition: the text does not fit
    ext = [sh.items[i] for i in g if isinstance(sh.items[i], Segment)
           and abs(sh.items[i].a[0] - sh.items[i].b[0]) < 1e-9]
    assert len(ext) == 2
    for e in ext:
        assert not _seg_hits_box(e, tbox), e
    # the text sits beyond the hole end (inside the view's extent), not past the datum edge; the
    # layout sets the offset itself, so the room it reserved is what is drawn
    assert tbox.xmin > max(d.p1[0], d.p2[0]) and d.text_offset > 0.0
    _assert_inside(BBox2D.of_primitives([sh.items[i] for i in g]), res.placements["front"][1], "2.25")


def test_outside_text_avoids_the_extension_lines_of_outer_bands():
    views, plan = _deep_plan()
    res, _, _ = _compose(views=views, section=None, plan=plan)
    sh = res.sheet
    k = [i for i, d in enumerate(sh.dimensions) if d.text == "48.57"][0]
    d = sh.dimensions[k]
    g = sh.meta["dimension_items"][k]
    tbox = BBox2D.of_primitives([next(sh.items[i] for i in g if isinstance(sh.items[i], Text))])
    assert tbox.height + 2 * ST.dim_text_gap > abs(d.p2[1] - d.p1[1])          # precondition: outside
    # past the hole end the 145.71 extension line (an outer band) would cross it: it goes past the
    # datum end instead, where no outer extension line runs
    foreign = [sh.items[i] for kk, gg in enumerate(sh.meta["dimension_items"]) if kk != k for i in gg
               if isinstance(sh.items[i], Segment)]
    assert not [s for s in foreign if _seg_hits_box(s, tbox)]
    assert tbox.ymax < min(d.p1[1], d.p2[1])
    _assert_inside(tbox, res.placements["front"][1], "48.57")


# ---- issue 30: the unit note goes on the sheet --------------------------------------

def test_strip_prints_the_unit_note_when_given():
    note = "Dimensions in mm (unit assumed, see report)"
    views = _box_views(100.0, 40.0, 15.0, iso=False)
    plan = plan_dimensions(views, [], DrawingOptions(), ST, np.array([100.0, 40.0, 15.0]))
    res = compose(views, None, plan, DrawingOptions(), ST, {"unit_note": note})
    texts = [t.text for t in res.sheet.items if isinstance(t, Text)]
    # review R4: one line would need 1.9 mm text (below the 2.5 mm lettering minimum), so the
    # note is set on two lines of the 9 mm strip, each at least 2.5 mm high
    assert 0.6 * 2.5 * len(note) > 48.0                         # precondition: one line cannot keep 2.5 mm
    assert note not in texts
    lines = [t for t in res.sheet.items if isinstance(t, Text) and t.text in
             ("Dimensions in mm (unit", "Dimensions in mm", "(unit assumed, see report)", "assumed, see report)")]
    assert " ".join(t.text for t in sorted(lines, key=lambda t: -t.position[1])) == note
    assert len(lines) == 2 and all(t.height >= 2.5 - 1e-9 for t in lines)
    plain = compose(views, None, plan, DrawingOptions(), ST, {})
    assert "Dimensions in mm" in [t.text for t in plain.sheet.items if isinstance(t, Text)]


def test_strip_two_line_unit_note_stays_inside_its_cell():
    from src.drawing.layout import STRIP_COLS, strip_items
    note = "Dimensions in mm (unit assumed, see report)"
    size = SHEETS["A3"]
    items = strip_items(size, ST, "1:1", "first", "STEP", unit_note=note)
    texts = [t for t in items if isinstance(t, Text) and t.text not in ("SCALE 1:1", "STEP")]
    assert len(texts) == 2 and all(t.height >= 2.5 - 1e-9 for t in texts)
    assert " ".join(t.text for t in sorted(texts, key=lambda t: -t.position[1])) == note
    segs = [s for s in items if isinstance(s, Segment)]
    y0 = min(min(s.a[1], s.b[1]) for s in segs)
    y1 = max(max(s.a[1], s.b[1]) for s in segs)
    for t in texts:                                             # inside the 9 mm strip, no overlap
        assert y0 < t.position[1] - 0.5 * t.height and t.position[1] + 0.5 * t.height < y1
        assert 0.6 * t.height * len(t.text) <= (STRIP_COLS[3] - STRIP_COLS[2]) - 2.0 + 1e-9
    a, b = sorted(texts, key=lambda t: t.position[1])
    assert a.position[1] + 0.5 * a.height <= b.position[1] - 0.5 * b.height
    short = strip_items(size, ST, "1:1", "first", "STEP", unit_note="Dimensions in mm (assumed)")
    assert "Dimensions in mm (assumed)" in [t.text for t in short if isinstance(t, Text)]


# ---- review R4: auto sheet, overall-dim recognition, tiny / finned section regions ----

def test_compose_auto_takes_a3_before_dropping_dims():
    import src.drawing.layout as layout_mod
    views, plan = _deep_plan(W=60.0, H=40.0, D=20.0, n_pos=8)
    a4, _, _ = _compose(views=views, section=None, plan=plan, sheet="A4")
    # precondition (the reviewed defect): on A4 the cap drops position dims to keep a scale
    assert [u for u in a4.plan.undimensioned if u.get("reason") == "layout_space"]
    a3, _, _ = _compose(views=views, section=None, plan=plan, sheet="A3")
    assert not [u for u in a3.plan.undimensioned if u.get("reason") == "layout_space"]
    res, _, _ = _compose(views=views, section=None, plan=plan, sheet="auto")
    assert res.sheet_size.name == "A3"
    assert not [u for u in res.plan.undimensioned if u.get("reason") == "layout_space"]
    assert len(res.sheet.dimensions) == len(plan.dims) and res.scale >= a4.scale
    assert res.scale == a3.scale
    assert layout_mod.POSITION_BAND_CAP == 3


def test_compose_auto_still_caps_on_the_final_sheet():
    views, plan = _deep_plan()
    res, _, _ = _compose(views=views, section=None, plan=plan, sheet="auto")
    assert res.sheet_size.name == "A3" and res.scale == pytest.approx(1 / 10)
    assert len([u for u in res.plan.undimensioned if u.get("reason") == "layout_space"]) == 9


def test_compose_auto_keeps_a4_when_it_holds_everything_at_1_to_1():
    views, plan = _deep_plan(W=40.0, H=40.0, D=20.0, n_pos=8)
    a4, _, _ = _compose(views=views, section=None, plan=plan, sheet="A4")
    assert a4.scale >= 1.0 and not a4.sheet.meta["dims_dropped"]          # precondition
    res, _, _ = _compose(views=views, section=None, plan=plan, sheet="auto")
    assert res.sheet_size.name == "A4" and res.scale == a4.scale and not res.sheet.meta["dims_dropped"]


def test_compose_auto_does_not_cap_a4_when_a3_shows_every_dim_at_that_scale():
    views, plan = _deep_plan(W=30.0, H=20.0, D=10.0, n_pos=8)
    a4, _, _ = _compose(views=views, section=None, plan=plan, sheet="A4")
    a3, _, _ = _compose(views=views, section=None, plan=plan, sheet="A3")
    # precondition: on A4 the cap wins a step (1:1 -> 2:1) by dropping dims; A3 holds all of them
    assert a4.sheet.meta["dims_dropped"] and a4.scale >= 2.0
    assert not a3.sheet.meta["dims_dropped"] and a3.scale >= a4.scale
    res, _, _ = _compose(views=views, section=None, plan=plan, sheet="auto")
    assert res.sheet_size.name == "A3" and res.scale == a3.scale and not res.sheet.meta["dims_dropped"]


def test_cap_bands_keeps_an_overall_dim_whose_view_bbox_is_slightly_larger():
    from src.drawing.layout import _cap_bands
    views, plan = _deep_plan(W=100.0, H=100.0, D=50.0, n_pos=8)
    boxes = {k: v.bbox for k, v in views.items()}
    fb = boxes["front"]
    boxes["front"] = BBox2D(fb.xmin, fb.ymin - 0.1, fb.xmax, fb.ymax + 0.1)   # 0.2 mm larger
    ext = boxes["front"].height
    h = plan.dims[1]
    assert h.value == 100.0
    # precondition: the numeric (round 2) test misses it, so it would be dropped first
    assert abs(abs(h.p2[1] - h.p1[1]) - ext) > max(0.01, 1e-3 * ext) and h.value < ext - max(0.01, 1e-3 * ext)
    new, dropped = _cap_bands(plan, boxes, 3)
    assert h in new.dims and len(dropped) == 5
    assert all(u["value"] < 100.0 for u in dropped)


def test_cap_bands_honours_a_role_marker_and_labels_dropped_kinds():
    from src.drawing.layout import _cap_bands, _is_overall
    views, plan = _deep_plan(W=100.0, H=100.0, D=50.0, n_pos=8)
    boxes = {k: v.bbox for k, v in views.items()}
    d0 = plan.dims[3]                                           # the innermost position dim (band 0)
    assert not _is_overall(d0, boxes)                           # precondition
    d0.role = "overall"
    plan.dims[10].role = "step"
    assert _is_overall(d0, boxes)
    new, dropped = _cap_bands(plan, boxes, 3)
    assert d0 in new.dims
    kinds = {u["text"]: u["kind"] for u in dropped}
    assert kinds[plan.dims[10].text] == "step"
    assert set(kinds.values()) == {"step", "linear"}           # no role: generic, not "hole_position"


def _sq(x, y, w, h):
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


def test_section_tiny_region_of_a_hatched_body_is_filled():
    from src.drawing.geometry2d import hatch_lines as hl
    from src.drawing.layout import _section_items
    big = _sq(0, 0, 80, 40)
    tiny8 = _sq(100, 10, 0.8, 0.8)
    tiny6 = _sq(100, 20, 0.6, 0.6)
    assert len(hl([np.array(tiny8, float)], 0.7, 45.0)) < 2     # precondition: one segment or none
    items, stats = _section_items(_sec_of([big, tiny8, tiny6]), (0.0, 0.0), 1.0, ST, BBox2D(0, 0, 101, 40))
    fills = [i for i in items if isinstance(i, Polygon)]
    assert len(fills) == 2 and stats["filled_regions"] == 2 and stats["fill"] == "mixed"
    hatches = [i for i in items if isinstance(i, HatchLines)]
    assert len(hatches) == 1 and len(hatches[0].segments) > 10   # the big region keeps its hatch
    # control: a 1.5 mm strip in a thick body is still hatched (t = 2A/P = 1.46 >= 1.4)
    strip = _sq(0, 50, 60, 1.5)
    items2, stats2 = _section_items(_sec_of([big, strip]), (0.0, 0.0), 1.0, ST, BBox2D(0, 0, 80, 51.5))
    assert not [i for i in items2 if isinstance(i, Polygon)] and stats2["fill"] == "hatch"


def _comb(base_h, n=30, fin_w=1.0, fin_h=30.0, width=100.0):
    pts = [[0.0, 0.0], [width, 0.0]]
    for k in range(n):
        xr = width - k * (width / n)
        xl = xr - fin_w
        pts += [[xr, base_h], [xr, base_h + fin_h], [xl, base_h + fin_h], [xl, base_h]]
    pts += [[0.0, base_h]]
    a = np.array(pts, float)
    return a[np.r_[True, np.any(np.diff(a, axis=0) != 0, axis=1)]]


def test_section_finned_body_with_a_thick_base_is_hatched_not_filled():
    from src.drawing.layout import _section_items
    comb = _comb(10.0)
    items, stats = _section_items(_sec_of([comb]), (0.0, 0.0), 1.0, ST, BBox2D(0, 0, 100, 40))
    assert stats["body_thickness"][0] < 2.1                     # precondition: 2A/P says "thin"
    assert not [i for i in items if isinstance(i, Polygon)]
    assert stats["fill"] == "hatch" and stats["thick_core_bodies"] == 1
    hatches = [i for i in items if isinstance(i, HatchLines)]
    assert hatches and len(hatches[0].segments) > 30
    # control: the same comb on a 1.5 mm base is thin everywhere and stays filled
    thin = _comb(1.5)
    items2, stats2 = _section_items(_sec_of([thin]), (0.0, 0.0), 1.0, ST, BBox2D(0, 0, 100, 31.5))
    assert len([i for i in items2 if isinstance(i, Polygon)]) == 1 and stats2["fill"] == "solid"
    assert stats2["thick_core_bodies"] == 0
