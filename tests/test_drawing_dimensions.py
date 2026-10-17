"""Dimension plan and rendering, QA round 2 (``src/drawing/dimensions.py``).

Issues (numbers of the QA round 2 triage): 1 coaxial bores of two walls stay
two features, 15 position reference circle and dimension side, 16
undimensioned disclosure (bodies, filled holes, rounds), 17 a value text
wider than its span goes outside, 26 inch-fraction values keep 3 decimals,
27 callouts avoid circles drawn only hidden, 29 a revolved outline gets one
diameter instead of two linear overall dims. Review fixes: a counterbore with
a conical shoulder (chamfered step, drill point) stays one feature; the circle
fit of issue 27 is pre-selected by the hole radii of the view.

Test discipline: every negative test asserts its precondition, and the
behaviours are pinned by pairs of inputs or by a negative control (a module
switch turned off) so a no-op implementation fails. Real OCCT shapes come
from BRepPrimAPI primitives; no corpus file is read.
"""
import math
import os
import sys

import numpy as np
import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)

from src.drawing import dimensions as dm  # noqa: E402
from src.drawing.dimensions import (  # noqa: E402
    format_value, linear_reach, linear_text_position, outside_text_offset, plan_dimensions, render_dimension)
from src.drawing.model import (  # noqa: E402
    BBox2D, Circle, CylFeature, Dimension, LineStyle, Polyline, ProjectedView, Segment, Text, text_corners)
from src.drawing.options import DrawingOptions  # noqa: E402
from src.drawing.projection import VIEWS  # noqa: E402
from src.drawing.style import DEFAULT_STYLE  # noqa: E402

ST = DEFAULT_STYLE


# ------------------------------------------------------------ fixtures --

def _rect(x0, y0, x1, y1, style=LineStyle.VISIBLE):
    return [Segment((x0, y0), (x1, y0), style), Segment((x1, y0), (x1, y1), style),
            Segment((x1, y1), (x0, y1), style), Segment((x0, y1), (x0, y0), style)]


def _box_views(ex, ey, ez, top_extra=(), top_hidden=()):
    return {
        "front": ProjectedView(VIEWS["front"], _rect(0, 0, ex, ez), [], BBox2D(0, 0, ex, ez), "hlr_exact"),
        "top": ProjectedView(VIEWS["top"], _rect(0, 0, ex, ey) + list(top_extra), list(top_hidden),
                             BBox2D(0, 0, ex, ey), "hlr_exact"),
        "right": ProjectedView(VIEWS["right"], _rect(0, 0, ey, ez), [], BBox2D(0, 0, ey, ez), "hlr_exact"),
    }


def _hole(x, y, r, z=7.5, length=15.0, axis=(0.0, 0.0, 1.0), kind="hole"):
    return CylFeature(np.array([x, y, z], float), np.array(axis, float), r, length, kind, True, 1)


def _linear(plan):
    return [d for d in plan.dims if d.kind == "linear"]


def _diam(plan):
    return [d for d in plan.dims if d.kind == "diameter"]


def _occ():
    pytest.importorskip("OCP")
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
    from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt
    return BRepAlgoAPI_Cut, BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder, gp_Ax2, gp_Dir, gp_Pnt


def _project(shape, names):
    from src.drawing.brep_views import project_brep
    return {n: project_brep(shape, VIEWS[n], exact=True, deflection=0.02) for n in names}


def _extents(shape):
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    box = Bnd_Box()
    BRepBndLib.Add_s(shape, box, False)
    x0, y0, z0, x1, y1, z1 = box.Get()
    return np.array([[x0, y0, z0], [x1, y1, z1]])


# ---- issue 26: inch-fraction values keep three decimals -----------------------

@pytest.mark.parametrize("v, text", [
    (3.175, "3.175"), (9.525, "9.525"), (15.875, "15.875"), (3.1750002, "3.175"), (3.1749998, "3.175"),
    (3.1749, "3.17"), (25.4, "25.4"), (6.35, "6.35"), (860.0000002, "860"),
    # metric thousandths stay on the 0.01 rule (pinned by test_drawing_layout.test_format_value)
    (12.345, "12.35"), (2.675, "2.68"), (1.375, "1.38"), (60.0, "60"), (-0.001, "0")])
def test_format_value_inch_fractions(v, text):
    assert format_value(v) == text


def test_format_value_inch_rule_is_not_a_no_op():
    # precondition: the plain 0.01 rounding of 3.175 is 3.18
    from decimal import ROUND_HALF_UP, Decimal
    assert str(Decimal("3.175").quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)) == "3.18"
    assert format_value(3.175) != "3.18"


# ---- issue 1: coaxial bores in two walls are two features ---------------------

def _two_wall_features(split_gap=90.0):
    """Two Ø52 bores on one X axis line, 10 mm long, ``split_gap`` apart, plus a counterbore."""
    ax = (1.0, 0.0, 0.0)
    feats = [_hole(5.0, 60.0, 26.0, z=60.0, length=10.0, axis=ax),
             _hole(15.0 + split_gap, 60.0, 26.0, z=60.0, length=10.0, axis=ax)]
    # counterbore in the near wall only: Ø10 x 10 (x 100..110) with Ø16 x 4 (x 106..110) touching
    feats += [_hole(105.0, 25.0, 5.0, z=25.0, length=10.0, axis=ax),
              _hole(108.0, 25.0, 8.0, z=25.0, length=4.0, axis=ax)]
    return feats


def _right_view_with_circles():
    vis = _rect(0, 0, 120, 120) + [Circle((60.0, 60.0), 26.0), Circle((25.0, 25.0), 5.0),
                                    Circle((25.0, 25.0), 8.0)]
    views = {"front": ProjectedView(VIEWS["front"], _rect(0, 0, 110, 120), [], BBox2D(0, 0, 110, 120), "hlr_exact"),
             "right": ProjectedView(VIEWS["right"], vis, [], BBox2D(0, 0, 120, 120), "hlr_exact")}
    return views


def test_coaxial_bores_of_two_walls_are_counted_twice():
    feats = _two_wall_features()
    plan = plan_dimensions(_right_view_with_circles(), feats, DrawingOptions(), ST, np.array([110.0, 120.0, 120.0]))
    texts = sorted(d.text for d in _diam(plan))
    assert texts == ["2 x Ø52", "Ø10 / Ø16 counterbore"], texts
    ref = next(d for d in _diam(plan) if d.text == "2 x Ø52")
    assert ref.count == 2 and ref.view == "right"
    assert plan.stats["coaxial_runs_split"] == 1
    assert not [u for u in plan.undimensioned if u["kind"] == "step_depth"]
    # one position pair for the two coaxial bores (same 2D centre): no duplicates
    keys = [(d.view, d.angle_deg, d.text) for d in _linear(plan)]
    assert len(keys) == len(set(keys))


def test_coaxial_split_negative_control(monkeypatch):
    feats = _two_wall_features()
    monkeypatch.setattr(dm, "_split_axial", False)
    plan = plan_dimensions(_right_view_with_circles(), feats, DrawingOptions(), ST, np.array([110.0, 120.0, 120.0]))
    texts = sorted(d.text for d in _diam(plan))
    assert "2 x Ø52" not in texts                  # without the split the two bores merge again
    assert plan.stats["coaxial_runs_split"] == 0


def test_touching_faces_stay_one_feature_and_gap_tolerance_scales():
    ax = (1.0, 0.0, 0.0)
    # a 0.04 mm gap (below COAXIAL_TOL) between two faces of one bore: still one feature
    feats = [_hole(5.0, 60.0, 26.0, z=60.0, length=10.0, axis=ax),       # x 0 .. 10
             _hole(15.04, 60.0, 26.0, z=60.0, length=10.0, axis=ax)]     # x 10.04 .. 20.04
    plan = plan_dimensions(_right_view_with_circles(), feats, DrawingOptions(), ST, np.array([110.0, 120.0, 120.0]))
    assert [d.text for d in _diam(plan)] == ["Ø52"]
    # control: a 5 mm gap splits
    feats[1] = _hole(25.0, 60.0, 26.0, z=60.0, length=10.0, axis=ax)
    plan = plan_dimensions(_right_view_with_circles(), feats, DrawingOptions(), ST, np.array([110.0, 120.0, 120.0]))
    assert [d.text for d in _diam(plan)] == ["2 x Ø52"]


def test_coaxial_bores_from_a_real_occt_hollow_box():
    Cut, Box, Cyl, Ax2, Dir, Pnt = _occ()
    from src.drawing.brep_features import cylinder_features
    shape = Box(Pnt(0, 0, 0), 110.0, 120.0, 120.0).Shape()
    shape = Cut(shape, Box(Pnt(10, 10, 10), 90.0, 100.0, 100.0).Shape()).Shape()
    shape = Cut(shape, Cyl(Ax2(Pnt(-1, 60, 60), Dir(1, 0, 0)), 26.0, 112.0).Shape()).Shape()
    shape = Cut(shape, Cyl(Ax2(Pnt(99, 25, 25), Dir(1, 0, 0)), 5.0, 12.0).Shape()).Shape()
    shape = Cut(shape, Cyl(Ax2(Pnt(106, 25, 25), Dir(1, 0, 0)), 8.0, 5.0).Shape()).Shape()
    feats = cylinder_features(shape)
    holes = [f for f in feats if f.kind == "hole"]
    assert sorted(round(f.radius, 6) for f in holes) == [5.0, 8.0, 26.0, 26.0]     # precondition: two Ø52 faces
    views = _project(shape, ["front", "top", "right"])
    b = _extents(shape)
    plan = plan_dimensions(views, feats, DrawingOptions(), ST, b[1] - b[0], model_bounds=b)
    texts = sorted(d.text for d in _diam(plan))
    assert texts == ["2 x Ø52", "Ø10 / Ø16 counterbore"], texts
    assert all(d.view == "right" for d in _diam(plan))


def _cone_counterbore(cone_len=3.0):
    """50 x 50 x 20 block: Ø16 x 6 counterbore, a conical shoulder ``cone_len`` long
    (Ø16 -> Ø10), Ø10 through. 3 mm = a 45 degree chamfered step; 1.8 mm ~ a 118
    degree drill point (gap = 3 / tan 59 deg)."""
    Cut, Box, Cyl, Ax2, Dir, Pnt = _occ()
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeCone
    top = 20.0 - 6.0
    shape = Box(Pnt(0, 0, 0), 50.0, 50.0, 20.0).Shape()
    shape = Cut(shape, Cyl(Ax2(Pnt(25, 25, top), Dir(0, 0, 1)), 8.0, 7.0).Shape()).Shape()
    shape = Cut(shape, BRepPrimAPI_MakeCone(Ax2(Pnt(25, 25, top - cone_len), Dir(0, 0, 1)),
                                            5.0, 8.0, cone_len).Shape()).Shape()
    return Cut(shape, Cyl(Ax2(Pnt(25, 25, -1), Dir(0, 0, 1)), 5.0, top - cone_len + 1.0).Shape()).Shape()


@pytest.mark.parametrize("cone_len", [3.0, 1.8])
def test_counterbore_with_a_conical_shoulder_stays_one_feature(cone_len, monkeypatch):
    # review of issue 1: the cone between the two cylinder faces is a gap of cone_len in the
    # CylFeature spans; the bore is still one counterbore (one callout + depth disclosure)
    from src.drawing.brep_features import cylinder_features
    shape = _cone_counterbore(cone_len)
    feats = cylinder_features(shape)
    holes = sorted((round(f.radius, 6), round(f.length, 6)) for f in feats if f.kind == "hole")
    assert holes == [(5.0, round(14.0 - cone_len, 6)), (8.0, 6.0)]                  # precondition
    b = _extents(shape)
    tol = max(dm.COAXIAL_TOL, 1e-3 * float(b[1, 2] - b[0, 2]))
    assert cone_len > 10.0 * tol                       # precondition: the gap is far above the split tol
    views = _project(shape, ["front", "top", "right"])
    plan = plan_dimensions(views, feats, DrawingOptions(), ST, b[1] - b[0], model_bounds=b)
    assert [d.text for d in _diam(plan)] == ["Ø10 / Ø16 counterbore"]
    assert plan.stats["coaxial_runs_split"] == 0
    assert [(u["kind"], u["count"]) for u in plan.undimensioned] == [("counterbore_depth", 1)]
    # negative control: without the bridge of a radius step the two faces split again
    monkeypatch.setattr(dm, "_STEP_GAP_FACTOR", 0.0)
    plan0 = plan_dimensions(views, feats, DrawingOptions(), ST, b[1] - b[0], model_bounds=b)
    assert sorted(d.text for d in _diam(plan0)) == ["Ø10", "Ø16"]


def test_radius_step_bridge_has_limits():
    n = np.array([1.0, 0.0, 0.0])
    ax = (1.0, 0.0, 0.0)

    def runs(feats):
        return [sorted(round(f.radius, 3) for f, _ in r)
                for r in dm._axial_runs([(f, np.zeros(2)) for f in feats], n, 0.05)]

    # Ø10 x 10 (x 0..10) and Ø16 x 4: gap 5 mm <= 2 x 3 -> one run; gap 7 mm > 6 -> two runs
    assert runs([_hole(5.0, 0, 5.0, z=0, length=10.0, axis=ax), _hole(17.0, 0, 8.0, z=0, length=4.0, axis=ax)]) \
        == [[5.0, 8.0]]
    assert runs([_hole(5.0, 0, 5.0, z=0, length=10.0, axis=ax), _hole(19.0, 0, 8.0, z=0, length=4.0, axis=ax)]) \
        == [[5.0], [8.0]]
    # equal radii: a 1 mm gap still splits (the bores of two thin walls)
    assert runs([_hole(5.0, 0, 5.0, z=0, length=10.0, axis=ax), _hole(16.0, 0, 5.0, z=0, length=10.0, axis=ax)]) \
        == [[5.0], [5.0]]
    # stepped set of two walls ~400 mm apart, different radii: two runs
    assert runs([_hole(5.0, 0, 5.05, z=0, length=10.0, axis=ax), _hole(415.0, 0, 6.0, z=0, length=10.0, axis=ax)]) \
        == [[5.05], [6.0]]
    # the bridge compares with the face that ends the run: Ø10 | Ø16 (touching) then a Ø16 bore 2 mm on
    # (equal radius to the run end) splits
    assert runs([_hole(5.0, 0, 5.0, z=0, length=10.0, axis=ax), _hole(12.0, 0, 8.0, z=0, length=4.0, axis=ax),
                 _hole(21.0, 0, 8.0, z=0, length=10.0, axis=ax)]) == [[5.0, 8.0], [8.0]]


# ---- issue 27: no callout on a circle drawn only hidden -----------------------

def _blind_from_back():
    Cut, Box, Cyl, Ax2, Dir, Pnt = _occ()
    shape = Box(Pnt(0, 0, 0), 50.0, 20.0, 30.0).Shape()
    # Ø5 x 12 deep from the back face (y = 20) toward -y: open toward the back view
    return Cut(shape, Cyl(Ax2(Pnt(25, 21, 15), Dir(0, -1, 0)), 2.5, 13.0).Shape()).Shape()


def test_blind_hole_seen_only_hidden_gets_no_callout():
    shape = _blind_from_back()
    from src.drawing.brep_features import cylinder_features
    feats = cylinder_features(shape)
    assert [round(f.radius, 6) for f in feats if f.kind == "hole"] == [2.5]
    views = _project(shape, ["front", "top"])
    idx = dm._circle_index(views["front"])
    c = np.array([25.0, 15.0])
    assert not dm._has_circle(idx["visible"], c, 2.5) and dm._has_circle(idx["hidden"], c, 2.5)   # precondition
    b = _extents(shape)
    plan = plan_dimensions(views, feats, DrawingOptions(), ST, b[1] - b[0], model_bounds=b)
    assert _diam(plan) == []
    u = [e for e in plan.undimensioned if e.get("reason") == "hidden_only"]
    assert len(u) == 1 and u[0]["kind"] == "hole_group" and u[0]["view"] == "front"
    assert u[0]["radius"] == pytest.approx(2.5) and u[0]["count"] == 1
    assert plan.stats["hidden_only"] == 1
    # control: with the back view drawn the callout goes there (the circle is visible)
    views_b = _project(shape, ["front", "top", "back"])
    plan_b = plan_dimensions(views_b, feats, DrawingOptions(), ST, b[1] - b[0], model_bounds=b)
    d = _diam(plan_b)
    assert [x.text for x in d] == ["Ø5"] and d[0].view == "back"
    assert not [e for e in plan_b.undimensioned if e.get("reason") == "hidden_only"]


def test_callout_prefers_the_visible_circle_of_a_group():
    # two Ø6 holes; the top view draws the one at (10, 10) visible and does not draw the other
    # (status unknown): the callout goes to the drawn one
    holes = [_hole(10.0, 10.0, 3.0), _hole(90.0, 50.0, 3.0)]
    views = _box_views(100.0, 60.0, 15.0, top_extra=[Circle((10.0, 10.0), 3.0)])
    d = _diam(plan_dimensions(views, holes, DrawingOptions(), ST, np.array([100.0, 60.0, 15.0])))
    assert len(d) == 1 and d[0].text == "2 x Ø6"
    assert d[0].center == pytest.approx((10.0, 10.0))
    # control: both visible -> the leader-cost choice is the top-right circle (short 45 degree leader)
    views2 = _box_views(100.0, 60.0, 15.0, top_extra=[Circle((10.0, 10.0), 3.0), Circle((90.0, 50.0), 3.0)])
    d2 = _diam(plan_dimensions(views2, holes, DrawingOptions(), ST, np.array([100.0, 60.0, 15.0])))
    assert d2[0].center == pytest.approx((90.0, 50.0))
    # a hole drawn only hidden is another feature (opens to the far side): its own disclosure
    views3 = _box_views(100.0, 60.0, 15.0, top_extra=[Circle((10.0, 10.0), 3.0)],
                        top_hidden=[Circle((90.0, 50.0), 3.0, LineStyle.HIDDEN)])
    p3 = plan_dimensions(views3, holes, DrawingOptions(), ST, np.array([100.0, 60.0, 15.0]))
    assert [(x.text, x.center) for x in _diam(p3)] == [("Ø6", (10.0, 10.0))]
    assert [(u["reason"], u["count"]) for u in p3.undimensioned] == [("hidden_only", 1)]


def test_circle_status_from_polylines_and_unknown():
    t = np.linspace(0.0, 2.0 * math.pi, 40, endpoint=False)
    poly = Polyline(np.column_stack([30 + 4 * np.cos(t), 20 + 4 * np.sin(t)]), closed=True)
    line = Polyline(np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0]]))
    v = ProjectedView(VIEWS["top"], [poly, line], [], BBox2D(0, 0, 50, 40), "hlr_poly")
    idx = dm._circle_index(v)
    assert idx["visible"].shape == (1, 3)                      # the straight polyline is no circle
    assert dm._circle_status(idx, np.array([30.0, 20.0]), [4.0]) == "visible"
    assert dm._circle_status(idx, np.array([10.0, 20.0]), [4.0]) == "unknown"


def test_circle_fit_prefilter_by_hole_radii(monkeypatch):
    # review of issue 27: only polylines sized like an arc (>= 90 deg) of a looked-up radius are fitted
    t = np.linspace(0.0, 2.0 * math.pi, 40, endpoint=False)
    full = Polyline(np.column_stack([30 + 4 * np.cos(t), 20 + 4 * np.sin(t)]), closed=True)
    q = np.linspace(math.radians(45.0), math.radians(137.0), 12)               # a 92 degree arc, r = 10
    arc = Polyline(np.column_stack([60 + 10 * np.cos(q), 20 + 10 * np.sin(q)]))
    v = ProjectedView(VIEWS["top"], [full, arc], [], BBox2D(0, 0, 80, 40), "hlr_poly")
    calls = []
    real = dm._fit_circle
    monkeypatch.setattr(dm, "_fit_circle", lambda pts: calls.append(1) or real(pts))
    idx = dm._circle_index(v)                                   # no hint: every polyline is fitted
    assert len(calls) == 2 and idx["visible"].shape == (2, 3)
    calls.clear()
    idx = dm._circle_index(v, [4.0, 10.0])                      # both sizes looked up: both kept
    assert len(calls) == 2 and idx["visible"].shape == (2, 3)
    assert dm._circle_status(idx, np.array([60.0, 20.0]), [10.0]) == "visible"
    calls.clear()
    idx = dm._circle_index(v, [10.0])                           # the r 4 circle is not fitted
    assert len(calls) == 1
    assert dm._circle_status(idx, np.array([30.0, 20.0]), [4.0]) == "unknown"
    assert dm._circle_status(idx, np.array([60.0, 20.0]), [10.0]) == "visible"
    calls.clear()
    idx = dm._circle_index(v, [25.0])                           # no size fits: nothing fitted
    assert calls == [] and idx["visible"].shape == (0, 3)
    # Circle primitives never need a fit and are always indexed
    vc = ProjectedView(VIEWS["top"], [Circle((5.0, 5.0), 2.0)], [], BBox2D(0, 0, 10, 10), "hlr_exact")
    assert dm._circle_index(vc, []).get("visible").shape == (1, 3)


# ---- issue 29: a revolved outline gets one diameter ---------------------------

def _washer():
    Cut, Box, Cyl, Ax2, Dir, Pnt = _occ()
    outer = Cyl(Ax2(Pnt(0, 0, 0), Dir(0, 0, 1)), 31.5, 3.0).Shape()
    return Cut(outer, Cyl(Ax2(Pnt(0, 0, -1), Dir(0, 0, 1)), 16.0, 5.0).Shape()).Shape()


def test_washer_outer_diameter_is_one_diameter_callout():
    from src.drawing.brep_features import cylinder_features
    shape = _washer()
    feats = cylinder_features(shape)
    assert [(f.kind, round(f.radius, 6)) for f in feats] == [("hole", 16.0), ("boss", 31.5)]   # precondition
    views = _project(shape, ["front", "top", "right"])
    b = _extents(shape)
    plan = plan_dimensions(views, feats, DrawingOptions(), ST, b[1] - b[0], model_bounds=b)
    od = [d for d in _diam(plan) if d.text == "Ø63"]
    assert len(od) == 1 and od[0].view == "top" and od[0].value == pytest.approx(63.0)
    assert od[0].radius == pytest.approx(31.5) and od[0].center == pytest.approx((0.0, 0.0), abs=1e-6)
    assert [round(d.value, 3) for d in _linear(plan)] == [3.0]          # only the thickness stays linear
    assert not [u for u in plan.undimensioned if u["kind"] == "boss_group"]
    assert sorted(d.text for d in _diam(plan)) == ["Ø32", "Ø63"]
    bore = next(d for d in _diam(plan) if d.text == "Ø32")
    assert bore.leader_angle_deg != od[0].leader_angle_deg              # distinct leader directions
    assert plan.stats["od_diameter"] == "top"


def test_square_plate_with_bore_keeps_its_linear_overall_dims():
    Cut, Box, Cyl, Ax2, Dir, Pnt = _occ()
    from src.drawing.brep_features import cylinder_features
    shape = Box(Pnt(-31.5, -31.5, 0), 63.0, 63.0, 3.0).Shape()
    shape = Cut(shape, Cyl(Ax2(Pnt(0, 0, -1), Dir(0, 0, 1)), 16.0, 5.0).Shape()).Shape()
    feats = cylinder_features(shape)
    views = _project(shape, ["front", "top", "right"])
    b = _extents(shape)
    plan = plan_dimensions(views, feats, DrawingOptions(), ST, b[1] - b[0], model_bounds=b)
    # the bore of a square plate keeps its two position dims (31.5)
    assert sorted(round(d.value, 3) for d in _linear(plan)) == [3.0, 31.5, 31.5, 63.0, 63.0]
    assert [d.text for d in _diam(plan)] == ["Ø32"]
    assert plan.stats["od_diameter"] is None


def test_circular_outline_without_a_full_boss_keeps_linear_dims():
    # synthetic washer outline but no boss feature (as in test_drawing_layout's concentric bore test)
    front = ProjectedView(VIEWS["front"], _rect(0, 0, 40, 10), [], BBox2D(0, 0, 40, 10), "hlr_exact")
    top = ProjectedView(VIEWS["top"], [Circle((20.0, 20.0), 20.0), Circle((20.0, 20.0), 5.0)], [],
                        BBox2D(0, 0, 40, 40), "hlr_exact")
    views = {"front": front, "top": top}
    partial = CylFeature(np.array([20.0, 20.0, 5.0]), np.array([0.0, 0.0, 1.0]), 20.0, 10.0, "boss", False, 1)
    plan = plan_dimensions(views, [_hole(20.0, 20.0, 5.0, z=5.0, length=10.0), partial], DrawingOptions(), ST,
                           np.array([40.0, 40.0, 10.0]))
    assert sorted(round(d.value, 3) for d in _linear(plan)) == [10.0, 40.0, 40.0]
    assert [u["kind"] for u in plan.undimensioned] == ["boss_group"]
    full = CylFeature(partial.center, partial.axis, 20.0, 10.0, "boss", True, 1)
    plan2 = plan_dimensions(views, [_hole(20.0, 20.0, 5.0, z=5.0, length=10.0), full], DrawingOptions(), ST,
                            np.array([40.0, 40.0, 10.0]))
    assert sorted(d.text for d in _diam(plan2)) == ["Ø10", "Ø40"]
    assert sorted(round(d.value, 3) for d in _linear(plan2)) == [10.0]


# ---- issue 16: undimensioned disclosure ---------------------------------------

def test_multi_body_extents_and_offsets_are_disclosed():
    views = _box_views(100.0, 60.0, 15.0)
    one = plan_dimensions(views, [], DrawingOptions(), ST, np.array([100.0, 60.0, 15.0]), n_bodies=1)
    assert one.undimensioned == []                                     # precondition: nothing for one body
    two = plan_dimensions(views, [], DrawingOptions(), ST, np.array([100.0, 60.0, 15.0]), n_bodies=3)
    by_kind = {u["kind"]: u for u in two.undimensioned}
    assert by_kind["body_extent"] == {"kind": "body_extent", "count": 3, "reason": "not_supported", "view": None}
    assert by_kind["body_position"]["count"] == 2 and by_kind["body_position"]["reason"] == "not_supported"
    default = plan_dimensions(views, [], DrawingOptions(), ST, np.array([100.0, 60.0, 15.0]))
    assert default.undimensioned == []


def test_multi_body_mesh_route_disclosure():
    trimesh = pytest.importorskip("trimesh")
    from src.drawing.mesh_views import project_mesh
    a = trimesh.creation.box(extents=(20.0, 10.0, 10.0))
    b = trimesh.creation.box(extents=(10.0, 10.0, 10.0))
    b.apply_translation((30.0, 0.0, 0.0))
    mesh = trimesh.util.concatenate([a, b])
    n = len(mesh.split(only_watertight=False))
    assert n == 2                                                       # precondition: two components
    views = {name: project_mesh(mesh, VIEWS[name], raster_px=400, feature_angle_deg=30.0, organic=False)
             for name in ("front", "top")}
    ext = mesh.bounds[1] - mesh.bounds[0]
    plan = plan_dimensions(views, [], DrawingOptions(), ST, ext, n_bodies=n)
    kinds = [u["kind"] for u in plan.undimensioned]
    assert "body_extent" in kinds and "body_position" in kinds


def test_filled_holes_and_rounds_are_disclosed():
    views = _box_views(100.0, 60.0, 15.0)
    feats = [_hole(20.0, 20.0, 4.0, kind="filled_hole"), _hole(80.0, 20.0, 4.0, kind="filled_hole"),
             CylFeature(np.array([0.0, 0.0, 7.5]), np.array([0.0, 0.0, 1.0]), 1.5, 3.0, "round", False, 1),
             CylFeature(np.array([100.0, 0.0, 7.5]), np.array([0.0, 0.0, 1.0]), 1.5, 3.0, "round", False, 1)]
    plan = plan_dimensions(views, feats, DrawingOptions(), ST, np.array([100.0, 60.0, 15.0]))
    assert _diam(plan) == []                                            # no callout on a fastener
    by_reason = {u["reason"]: u for u in plan.undimensioned}
    f = by_reason["filled_by_fastener"]
    assert f["kind"] == "hole_group" and f["count"] == 2 and f["radius"] == pytest.approx(4.0)
    r = by_reason["not_supported"]
    assert r["kind"] == "round_group" and r["count"] == 2 and r["radius"] == pytest.approx(1.5)


# ---- issue 15: position reference circle and side -----------------------------

def test_positions_reference_the_group_circle_nearest_the_datum_corner():
    holes = [_hole(10.0, 10.0, 3.0), _hole(90.0, 50.0, 3.0)]
    views = _box_views(100.0, 60.0, 15.0, top_extra=[Circle((10.0, 10.0), 3.0), Circle((90.0, 50.0), 3.0)])
    plan = plan_dimensions(views, holes, DrawingOptions(), ST, np.array([100.0, 60.0, 15.0]))
    d = _diam(plan)
    assert d[0].center == pytest.approx((90.0, 50.0))                  # precondition: callout far from datum
    pos = [x for x in _linear(plan) if x.view == "top" and round(x.value, 6) not in (100.0, 60.0)]
    assert sorted(x.text for x in pos) == ["10", "10"]                 # from (10, 10), not 90 / 50
    for x in pos:
        nrm = 1 if x.angle_deg == 0.0 else 0
        ext = max(abs(x.base[nrm] - x.p1[nrm]), abs(x.base[nrm] - x.p2[nrm]))
        assert ext < 0.3 * math.hypot(100.0, 60.0), (x.text, ext)
    assert plan.stats["position_refs_moved"] == 1


def test_position_side_avoids_crossing_another_circle():
    # the X extension line of the Ø4 at (52, 40) toward the nearer top edge (18 mm) runs through
    # the Ø6 at (50, 54); the bottom side (38 mm) is free. A crossing costs 0.25 x the view
    # diagonal (29 mm), so the bottom wins: 38 < 18 + 29.
    holes = [_hole(50.0, 54.0, 3.0), _hole(52.0, 40.0, 2.0)]
    views = _box_views(100.0, 60.0, 15.0, top_extra=[Circle((50.0, 54.0), 3.0), Circle((52.0, 40.0), 2.0)])
    seg, _ = dm._hole_ext_line(np.array([52.0, 40.0]), 2.0, 0, True, BBox2D(0, 0, 100, 60))
    assert dm._crossed_circles(seg, np.array([52.0, 40.0]), [(np.array([50.0, 54.0]), 3.0)]) == 1   # precondition
    plan = plan_dimensions(views, holes, DrawingOptions(), ST, np.array([100.0, 60.0, 15.0]))
    x52 = [(d, b) for d, b in zip(plan.dims, plan.bands)
           if d.kind == "linear" and d.view == "top" and d.angle_deg == 0.0 and d.text == "52"]
    assert len(x52) == 1 and x52[0][1][0] == "bottom"
    assert x52[0][0].p2 == pytest.approx((52.0, 38.0))                 # bottom quadrant of the Ø4
    x50 = [b for d, b in zip(plan.dims, plan.bands) if d.view == "top" and d.angle_deg == 0.0 and d.text == "50"]
    assert x50 and x50[0][0] == "top"                                  # its own line is free: nearer side
    assert plan.stats["position_sides_switched"] == 1


def test_overall_dim_moves_to_the_side_with_short_extension_lines():
    # a wedge: low on the left (10), tall on the right (50); the top extreme point is at the right
    outline = [Segment((0.0, 0.0), (100.0, 0.0)), Segment((100.0, 0.0), (100.0, 50.0)),
               Segment((100.0, 50.0), (90.0, 50.0)), Segment((90.0, 50.0), (0.0, 10.0)),
               Segment((0.0, 10.0), (0.0, 0.0))]
    views = {"front": ProjectedView(VIEWS["front"], outline, [], BBox2D(0, 0, 100, 50), "hlr_exact")}
    plan = plan_dimensions(views, [], DrawingOptions(), ST, np.array([100.0, 30.0, 50.0]))
    by = {d.angle_deg: (d, b) for d, b in zip(plan.dims, plan.bands)}
    assert by[90.0][1][0] == "right" and by[90.0][0].p2 == pytest.approx((100.0, 50.0))
    assert by[0.0][1][0] == "bottom"                                   # a tie keeps the rule's side
    assert plan.stats["overall_sides_switched"] == 1
    # control: a rectangle keeps the left side
    rect = {"front": ProjectedView(VIEWS["front"], _rect(0, 0, 100, 50), [], BBox2D(0, 0, 100, 50), "hlr_exact")}
    p2 = plan_dimensions(rect, [], DrawingOptions(), ST, np.array([100.0, 30.0, 50.0]))
    assert {b[0] for b in p2.bands} == {"bottom", "left"}


# ---- issue 17: a text wider than its span goes outside --------------------------

def _boxes_cross(seg, corners):
    """Segment vs axis-aligned box of ``corners`` (Liang-Barsky clip)."""
    b = BBox2D.from_points(corners)
    (x0, y0), (x1, y1) = seg.a, seg.b
    dx, dy = x1 - x0, y1 - y0
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, x0 - b.xmin), (dx, b.xmax - x0), (-dy, y0 - b.ymin), (dy, b.ymax - y0)):
        if abs(p) < 1e-15:
            if q < 0:
                return False
            continue
        t = q / p
        if p < 0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
        if t0 > t1:
            return False
    return True


def _parts(items):
    segs = [i for i in items if isinstance(i, Segment)]
    texts = [i for i in items if isinstance(i, Text)]
    return segs, texts


@pytest.mark.parametrize("angle, p2", [(0.0, (2.2, 0.0)), (90.0, (0.0, 2.2))])
def test_short_span_text_is_not_crossed_by_its_extension_lines(angle, p2):
    base = (1.1, -10.0) if angle == 0.0 else (-10.0, 1.1)
    d = Dimension("linear", "2.77", 2.77, p1=(0.0, 0.0), p2=p2, base=base, angle_deg=angle)
    items = render_dimension(d, ST)
    segs, texts = _parts(items)
    assert len(segs) == 3 and len(texts) == 1                         # same item count as before
    tc = text_corners(texts[0])
    tw = BBox2D.from_points(tc)
    assert max(tw.width, tw.height) + 2 * ST.dim_text_gap > 2.2        # precondition: the text does not fit
    ax = 0 if angle == 0.0 else 1
    ext = [s for s in segs if abs(s.a[ax] - s.b[ax]) < 1e-9]
    assert len(ext) == 2
    for s in ext:
        assert not _boxes_cross(s, tc), s
    # beyond the p2 end, on the dimension line extended under it
    assert tw.xmin > 2.2 if ax == 0 else tw.ymin > 2.2
    line = next(s for s in segs if s not in ext)
    hi = max(line.a[ax], line.b[ax])
    assert hi >= (tw.xmax if ax == 0 else tw.ymax) - 1e-9
    assert texts[0].position == pytest.approx(linear_text_position(d, ST))
    r1, r2 = linear_reach(2.2, "2.77", ST)
    assert hi == pytest.approx(2.2 + r2) and r1 == pytest.approx(ST.arrow_length + dm.SHORT_SPAN_EXTRA)


def test_text_side_avoids_the_extension_line_of_an_outer_dim():
    # TO-220 tab (QA 14): on the right of the view, the position 5.13 (band 0, datum bottom, p2 at
    # the hole) and the overall 10.25 (band 1, extension lines at y 0 and 10.25). At 2:1 the text
    # "5.13" does not fit its 10.26 mm span; past the hole it would lie across the 10.25 extension
    # line at the top edge, past the datum end it is free.
    bb = BBox2D(0.0, 0.0, 10.0, 10.25)
    pos = Dimension("linear", "5.13", 5.13, p1=(10.0, 0.0), p2=(9.1, 5.13), base=(10.0, 2.565), angle_deg=90.0,
                    view="front")
    tot = Dimension("linear", "10.25", 10.25, p1=(10.0, 0.0), p2=(10.0, 10.25), base=(10.0, 5.125),
                    angle_deg=90.0, view="front")
    sides = dm._text_sides([pos, tot], [("right", 0), ("right", 1)], {"front": bb}, ST, 2.0)
    assert outside_text_offset(2.0 * 5.13, "5.13", ST) > 0.0           # precondition: does not fit at 2:1
    assert sides == [-1, 1]
    # control: without the outer dim the p2 end wins (it sticks out of the view's extent less)
    assert dm._text_sides([pos], [("right", 0)], {"front": bb}, ST, 2.0) == [1]
    # the layout applies it in sheet space: the text then sits past the p1 end
    ds = Dimension("linear", "5.13", 5.13, p1=(20.0, 0.0), p2=(18.2, 10.26), base=(40.0, 5.13), angle_deg=90.0)
    moved = dm.apply_text_side(ds, -1, ST)
    assert moved.text_offset < 0.0 and dm.apply_text_side(ds, 1, ST) is ds
    tb = BBox2D.from_points(text_corners(_parts(render_dimension(moved, ST))[1][0]))
    assert tb.ymax < 0.0
    wide = Dimension("linear", "5.13", 5.13, p1=(20.0, 0.0), p2=(18.2, 40.0), base=(40.0, 20.0), angle_deg=90.0)
    assert dm.apply_text_side(wide, -1, ST) is wide                     # fits: nothing to move


def test_plan_reports_text_sides_parallel_to_dims():
    holes = [_hole(10.0, 10.0, 3.0)]
    views = _box_views(100.0, 60.0, 15.0, top_extra=[Circle((10.0, 10.0), 3.0)])
    plan = plan_dimensions(views, holes, DrawingOptions(), ST, np.array([100.0, 60.0, 15.0]))
    assert len(plan.text_sides) == len(plan.dims)
    for d, s in zip(plan.dims, plan.text_sides):
        assert (s is None) == (d.kind == "diameter")


def test_wide_span_text_stays_centred_and_explicit_offset_wins():
    d = Dimension("linear", "2.77", 2.77, p1=(0.0, 0.0), p2=(50.0, 0.0), base=(25.0, -10.0), angle_deg=0.0)
    assert outside_text_offset(50.0, "2.77", ST) == 0.0
    t = _parts(render_dimension(d, ST))[1][0]
    assert t.position == pytest.approx((25.0, -10.0 + ST.dim_text_gap))
    # a medium span (inside arrows) with a long text also goes outside
    m = Dimension("linear", "43.81", 43.81, p1=(0.0, 0.0), p2=(9.0, 0.0), base=(4.5, -10.0), angle_deg=0.0)
    assert 9.0 >= 2 * ST.arrow_length + 0.5                           # precondition: arrows inside
    tm = BBox2D.from_points(text_corners(_parts(render_dimension(m, ST))[1][0]))
    assert tm.xmin > 9.0
    # an explicit (layout) offset toward p1 wins over the default
    off = -outside_text_offset(2.2, "2.77", ST)
    s = Dimension("linear", "2.77", 2.77, p1=(0.0, 0.0), p2=(2.2, 0.0), base=(1.1, -10.0), angle_deg=0.0,
                  text_offset=off)
    ts = BBox2D.from_points(text_corners(_parts(render_dimension(s, ST))[1][0]))
    assert ts.xmax < 0.0
    assert linear_reach(2.2, "2.77", ST, text_offset=off)[0] > ST.arrow_length + dm.SHORT_SPAN_EXTRA
