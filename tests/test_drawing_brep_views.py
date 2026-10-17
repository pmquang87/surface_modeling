"""Tests for src/drawing/brep_views.py and src/drawing/brep_sections.py (task B2).

Fixtures are built here with BRepPrimAPI (no dependency on brep_load being
merged). Expected numbers come from the HLR and brep-analysis probes:

* 60x40x20 box with a vertical r5 through hole at (30,20): front view shows
  the 4 outline lines and no circle; the hole walls x = 25 / x = 35 are
  hidden; HCompound also repeats the box back edges and the edge-on mouths
  on the outline, which the clean-up must subtract. Top view: one visible
  full circle, the far mouth (hidden) lies on it and must be dropped.
* Box filleted r3 on all edges, front view: 4 corner arcs, the tangent
  lines x = 3 / 57 (Rg1) are not drawn, one duplicate arc is removed.
* ico2 r20 faceted shell + EncodeRegularity: PolyAlgo front view gives the
  24 mesh silhouette edges and a +-20 bbox.
* Plate 100x60x15 + boss d30x10 at (78,30) + 4 x d6.5 + d20 at (50,30):
  section y = 30 -> loops of area 600 and 900; plane x = 93 is tangent to
  the boss (OCCT returns one OPEN wire that also holds the middle loop).

Every negative test asserts its precondition first; the module flags
``_SUBTRACT_COVERED``, ``_SWAP_CW_ARCS`` and ``_RECOVER_OPEN`` are switched
off in negative controls to show the checks are load-bearing.
"""
import math
import os
import sys
import types

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    import OCP  # noqa: F401
    HAS_OCP = True
except ImportError:
    HAS_OCP = False

needs_ocp = pytest.mark.skipif(not HAS_OCP, reason="cadquery-ocp not installed")
pytestmark = needs_ocp

from src.drawing.model import Arc, Circle, CylFeature, Polyline, Segment  # noqa: E402
from src.drawing.projection import VIEWS, ViewSpec  # noqa: E402


# ----------------------------------------------------------------- fixtures

def _box(x, y, z, dx, dy, dz):
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.gp import gp_Pnt
    return BRepPrimAPI_MakeBox(gp_Pnt(x, y, z), dx, dy, dz).Shape()


def _cyl(p, d, r, h):
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
    from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt
    return BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(*p), gp_Dir(*d)), r, h).Shape()


def _cut(a, b):
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    return BRepAlgoAPI_Cut(a, b).Shape()


def _fuse(a, b):
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
    return BRepAlgoAPI_Fuse(a, b).Shape()


def _compound(*shapes):
    from OCP.BRep import BRep_Builder
    from OCP.TopoDS import TopoDS_Compound
    b = BRep_Builder()
    c = TopoDS_Compound()
    b.MakeCompound(c)
    for s in shapes:
        b.Add(c, s)
    return c


def _box_hole():
    return _cut(_box(0, 0, 0, 60, 40, 20), _cyl((30, 20, -5), (0, 0, 1), 5.0, 30.0))


def _fillet_box():
    from OCP.BRepFilletAPI import BRepFilletAPI_MakeFillet
    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS
    box = _box(0, 0, 0, 60, 40, 20)
    mf = BRepFilletAPI_MakeFillet(box)
    ex = TopExp_Explorer(box, TopAbs_EDGE)
    while ex.More():
        mf.Add(3.0, TopoDS.Edge_s(ex.Current()))
        ex.Next()
    mf.Build()
    assert mf.IsDone()
    return mf.Shape()


_TILT = math.radians(30.0)


def _tilted_hole():
    """60x40x20 box, r5 hole through (30,20,10) tilted 30 deg from Z in the XZ plane."""
    d = (math.sin(_TILT), 0.0, math.cos(_TILT))
    start = (30 - 30 * d[0], 20.0, 10 - 30 * d[2])
    return _cut(_box(0, 0, 0, 60, 40, 20), _cyl(start, d, 5.0, 60.0))


def _plate():
    """REPORT-brep-analysis.md fixture (own copy, see the plan's B1/B2 note)."""
    shape = _box(0, 0, 0, 100.0, 60.0, 15.0)
    shape = _fuse(shape, _cyl((78, 30, 15), (0, 0, 1), 15.0, 10.0))
    for (x, y), r in [((8, 8), 3.25), ((92, 8), 3.25), ((8, 52), 3.25), ((92, 52), 3.25),
                      ((50, 30), 10.0)]:
        shape = _cut(shape, _cyl((x, y, -5), (0, 0, 1), r, 40.0))
    return shape


def _plate_features():
    z = np.array([0.0, 0.0, 1.0])
    feats = [CylFeature(np.array([x, y, 7.5]), z, 3.25, 15.0, "hole", True, 1)
             for x, y in [(8, 8), (92, 8), (8, 52), (92, 52)]]
    feats.append(CylFeature(np.array([50.0, 30.0, 7.5]), z, 10.0, 15.0, "hole", True, 1))
    feats.append(CylFeature(np.array([78.0, 30.0, 20.0]), z, 15.0, 10.0, "boss", True, 1))
    return feats


def _plate_model(shape=None):
    return types.SimpleNamespace(shape=shape, census={"bbox_min": [0.0, 0.0, 0.0],
                                                      "bbox_max": [100.0, 60.0, 25.0]})


def _ico_shell(subdivisions=2, radius=20.0):
    """Faceted shell + EncodeRegularity built directly from src.io.faceted_step."""
    import trimesh
    from OCP.BRepLib import BRepLib
    from src.io.faceted_step import _build_direct, _shell_from_faces
    m = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    shell = _shell_from_faces(_build_direct(m.vertices, m.faces))
    BRepLib.EncodeRegularity_s(shell, 0.35)
    return m, shell


# ----------------------------------------------------------------- helpers

def _segs(prims):
    return [p for p in prims if isinstance(p, Segment)]


def _on(v, target, tol=1e-6):
    return abs(float(v) - target) <= tol


def _arc_mid(a):
    sweep = (a.end_deg - a.start_deg) % 360.0 or 360.0
    m = math.radians(a.start_deg + sweep / 2.0)
    return (a.center[0] + a.radius * math.cos(m), a.center[1] + a.radius * math.sin(m))


def _seg_on_outline(s, ys=(0.0, 20.0)):
    return any(_on(s.a[1], y) and _on(s.b[1], y) for y in ys)


# ----------------------------------------------------------------- views

def test_front_view_box_hole_outline_and_hidden_walls():
    from src.drawing.brep_views import project_brep
    pv = project_brep(_box_hole(), VIEWS["front"], exact=True, deflection=0.02)
    assert pv.algorithm == "hlr_exact"
    assert pv.spec == VIEWS["front"]
    assert not any(isinstance(p, (Circle, Arc)) for p in pv.visible)
    vis = _segs(pv.visible)
    assert len(vis) == 4 and len(pv.visible) == 4
    pts = np.array([[s.a, s.b] for s in vis]).reshape(-1, 2)
    assert pts[:, 0].min() == pytest.approx(0.0, abs=1e-6)
    assert pts[:, 0].max() == pytest.approx(60.0, abs=1e-6)
    assert pts[:, 1].min() == pytest.approx(0.0, abs=1e-6)
    assert pts[:, 1].max() == pytest.approx(20.0, abs=1e-6)
    # each outline edge spans the full side
    lengths = sorted(round(math.dist(s.a, s.b), 6) for s in vis)
    assert lengths == [20.0, 20.0, 60.0, 60.0]

    hid = _segs(pv.hidden)
    assert hid, "precondition: the hole walls must be hidden"
    walls = [s for s in hid if _on(s.a[0], s.b[0]) and abs(s.a[1] - s.b[1]) > 19.999]
    wall_x = sorted(round(s.a[0], 6) for s in walls)
    assert wall_x == [25.0, 35.0]
    assert not any(_seg_on_outline(s) for s in hid)
    assert pv.bbox is not None
    assert (pv.bbox.xmin, pv.bbox.xmax) == pytest.approx((0.0, 60.0), abs=1e-6)


def test_front_view_negative_control_without_subtraction(monkeypatch):
    """Without the clean-up the back edges / edge-on mouths overprint the outline."""
    import src.drawing.brep_views as bv
    monkeypatch.setattr(bv, "_SUBTRACT_COVERED", False)
    pv = bv.project_brep(_box_hole(), VIEWS["front"], exact=True, deflection=0.02)
    assert any(_seg_on_outline(s) for s in _segs(pv.hidden))


def test_top_view_hole_circle_and_far_mouth_dropped(monkeypatch):
    import src.drawing.brep_views as bv
    pv = bv.project_brep(_box_hole(), VIEWS["top"], exact=True, deflection=0.02)
    circles = [p for p in pv.visible if isinstance(p, Circle)]
    assert len(circles) == 1
    c = circles[0]
    assert c.center == pytest.approx((30.0, 20.0), abs=1e-6)
    assert c.radius == pytest.approx(5.0, abs=1e-6)
    assert not any(isinstance(p, Circle) for p in pv.hidden)
    # negative control: the far mouth IS in the raw hidden compound
    monkeypatch.setattr(bv, "_SUBTRACT_COVERED", False)
    raw = bv.project_brep(_box_hole(), VIEWS["top"], exact=True, deflection=0.02)
    assert any(isinstance(p, Circle) for p in raw.hidden)


def test_filleted_box_front_arcs(monkeypatch):
    import src.drawing.brep_views as bv
    shape = _fillet_box()
    pv = bv.project_brep(shape, VIEWS["front"], exact=True, deflection=0.02)
    # preconditions: OCCT does emit the tangent lines (Rg1) and a duplicate visible arc
    assert pv.stats["edge_counts"]["Rg1LineVCompound"] > 0
    assert pv.stats["raw_visible"] > len(pv.visible)
    with monkeypatch.context() as mp:
        mp.setattr(bv, "dedupe_primitives", lambda prims, tol: list(prims))
        undeduped = bv.project_brep(shape, VIEWS["front"], exact=True, deflection=0.02)
    assert len([p for p in undeduped.visible if isinstance(p, Arc)]) > 4
    with monkeypatch.context() as mp:
        mp.setattr(bv, "_VISIBLE_COMPOUNDS", bv._VISIBLE_COMPOUNDS + ("Rg1LineVCompound",))
        with_rg1 = bv.project_brep(shape, VIEWS["front"], exact=True, deflection=0.02)
    assert any(_on(s.a[0], 3.0) and _on(s.b[0], 3.0) for s in _segs(with_rg1.visible))
    arcs = [p for p in pv.visible if isinstance(p, Arc)]
    assert len(arcs) == 4          # the duplicate arc of VCompound is gone
    for a in arcs:
        assert a.radius == pytest.approx(3.0, abs=1e-6)
        sweep = (a.end_deg - a.start_deg) % 360.0
        assert sweep == pytest.approx(90.0, abs=1e-6)
        mx, my = _arc_mid(a)
        inside = 3.0 + 1e-6 < mx < 57.0 - 1e-6 and 3.0 + 1e-6 < my < 17.0 - 1e-6
        assert not inside, (a, (mx, my))
    for s in _segs(pv.visible):
        assert not (_on(s.a[0], 3.0) and _on(s.b[0], 3.0))
        assert not (_on(s.a[0], 57.0) and _on(s.b[0], 57.0))
        assert not (_on(s.a[1], 3.0) and _on(s.b[1], 3.0))
        assert not (_on(s.a[1], 17.0) and _on(s.b[1], 17.0))
    # the four straight outline pieces are there (between the fillets)
    assert len(_segs(pv.visible)) == 4
    # negative control: without the CW swap at least one arc is its complement
    monkeypatch.setattr(bv, "_SWAP_CW_ARCS", False)
    raw = bv.project_brep(shape, VIEWS["front"], exact=True, deflection=0.02)
    raw_arcs = [p for p in raw.visible if isinstance(p, Arc)]
    assert len(raw_arcs) >= 4
    assert any(3.0 < _arc_mid(a)[0] < 57.0 and 3.0 < _arc_mid(a)[1] < 17.0 for a in raw_arcs)


def _ellipse_distance(pts, cx, cy, a, b):
    """First-order distance of points to the ellipse ((x-cx)/a)^2 + ((y-cy)/b)^2 = 1."""
    x = (pts[:, 0] - cx)
    y = (pts[:, 1] - cy)
    f = (x / a) ** 2 + (y / b) ** 2 - 1.0
    g = np.hypot(2 * x / a ** 2, 2 * y / b ** 2)
    return np.abs(f) / np.maximum(g, 1e-12)


def test_tilted_hole_gives_discretized_ellipse():
    from src.drawing.brep_views import project_brep
    defl = 0.02
    pv = project_brep(_tilted_hole(), VIEWS["top"], exact=True, deflection=defl)
    polys = [p for p in pv.visible if isinstance(p, Polyline)]
    assert polys, [type(p).__name__ for p in pv.visible]
    a, b = 5.0 / math.cos(_TILT), 5.0
    centres = [(30 + 10 * math.tan(_TILT), 20.0), (30 - 10 * math.tan(_TILT), 20.0)]
    worst = 0.0
    for p in polys:
        pts = np.asarray(p.points, float)
        assert len(pts) >= 5
        seq = np.vstack([pts, pts[:1]]) if p.closed else pts
        mids = 0.5 * (seq[:-1] + seq[1:])
        # quarter points too: the chord sag peaks at the middle
        dev_v = np.minimum(*[_ellipse_distance(pts, cx, cy, a, b) for cx, cy in centres])
        assert dev_v.max() < 1e-3, "vertices must lie on one of the two mouth ellipses"
        dev_m = np.minimum(*[_ellipse_distance(mids, cx, cy, a, b) for cx, cy in centres])
        worst = max(worst, float(dev_m.max()))
    assert 0.0 < worst < defl


def test_hidden_false_returns_no_hidden_lines():
    from src.drawing.brep_views import project_brep
    with_h = project_brep(_box_hole(), VIEWS["front"], exact=True, deflection=0.02, hidden=True)
    assert with_h.hidden, "precondition: this view has hidden lines"
    no_h = project_brep(_box_hole(), VIEWS["front"], exact=True, deflection=0.02, hidden=False)
    assert no_h.hidden == []
    assert len(no_h.visible) == len(with_h.visible)


def test_faceted_shell_poly_hlr_silhouette():
    from src.drawing.brep_views import prepare_faceted_shell, project_brep
    m, shell = _ico_shell()
    assert np.allclose(np.linalg.norm(m.vertices, axis=1), 20.0), "precondition: icosphere r 20"
    shape = prepare_faceted_shell(shell, 0.05)
    assert shape is shell
    pv = project_brep(shape, VIEWS["front"], exact=False, deflection=0.05)
    assert pv.algorithm == "hlr_poly"
    assert len(_segs(pv.visible)) == 24
    assert len(pv.visible) == 24
    bb = pv.bbox
    assert (bb.xmin, bb.xmax, bb.ymin, bb.ymax) == pytest.approx((-20, 20, -20, 20), abs=1e-6)


def test_faceted_shape_from_brep_load_poly_hlr():
    try:
        from src.drawing.brep_load import faceted_shape_from_mesh
    except (ImportError, AttributeError):
        pytest.skip("brep_load.faceted_shape_from_mesh not merged yet (task B1)")
    import trimesh
    from src.drawing.brep_views import prepare_faceted_shell, project_brep
    m = trimesh.creation.icosphere(subdivisions=2, radius=20.0)
    assert np.allclose(np.linalg.norm(m.vertices, axis=1), 20.0)
    shape = faceted_shape_from_mesh(m, 0.35, lambda *a, **k: None)
    shape = prepare_faceted_shell(shape, 0.05)
    pv = project_brep(shape, VIEWS["front"], exact=False, deflection=0.05)
    assert pv.algorithm == "hlr_poly"
    assert len(_segs(pv.visible)) == 24
    bb = pv.bbox
    assert (bb.xmin, bb.xmax, bb.ymin, bb.ymax) == pytest.approx((-20, 20, -20, 20), abs=1e-6)


def test_exact_false_meshes_an_unmeshed_shape():
    """PolyAlgo on faces without a triangulation gives nothing; project_brep meshes first."""
    from src.drawing.brep_views import project_brep
    pv = project_brep(_box(0, 0, 0, 10, 10, 10), VIEWS["front"], exact=False, deflection=0.05)
    assert pv.algorithm == "hlr_poly"
    assert pv.stats["meshed"] is True
    assert len(_segs(pv.visible)) == 4


def _outline_sag(prims, r):
    """Max radial sag of the chord midpoints of an outline around the origin."""
    worst = 0.0
    for s in _segs(prims):
        mid = 0.5 * (np.asarray(s.a) + np.asarray(s.b))
        worst = max(worst, r - float(np.hypot(*mid)))
    return worst


def test_exact_false_remeshes_a_coarser_existing_triangulation():
    """A mesh left from an earlier step (coarser than requested) must not set the accuracy."""
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from src.drawing.brep_views import project_brep
    cyl = _cyl((0, 0, 0), (0, 0, 1), 10.0, 20.0)
    BRepMesh_IncrementalMesh(cyl, 0.5, False, 0.35, True)
    coarse = project_brep(cyl, VIEWS["top"], exact=False, deflection=0.5)
    assert coarse.stats["meshed"] is False          # the existing mesh is fine enough for 0.5
    # 36 chords of 10 deg (angular limit 0.35 rad): sag 0.038, well above the 0.01 asked next
    assert _outline_sag(coarse.visible, 10.0) > 0.02, "precondition: the existing mesh is coarse"
    fine = project_brep(cyl, VIEWS["top"], exact=False, deflection=0.01)
    assert fine.stats["meshed"] is True
    sag = _outline_sag(fine.visible, 10.0)
    assert 0.0 < sag <= 0.01 * 1.05
    assert len(_segs(fine.visible)) > len(_segs(coarse.visible))
    # an equally fine mesh is reused, not rebuilt
    again = project_brep(cyl, VIEWS["top"], exact=False, deflection=0.01)
    assert again.stats["meshed"] is False


def test_two_box_compound_counts_are_the_sum():
    from src.drawing.brep_views import project_brep
    a = _box(0, 0, 0, 10, 10, 10)
    b = _box(30, 0, 0, 20, 15, 5)
    comp = _compound(a, b)
    for name in ("front", "top", "iso"):
        pa = project_brep(a, VIEWS[name], exact=True, deflection=0.02)
        pb = project_brep(b, VIEWS[name], exact=True, deflection=0.02)
        pc = project_brep(comp, VIEWS[name], exact=True, deflection=0.02)
        assert len(pa.visible) > 0 and len(pb.visible) > 0
        assert len(pc.visible) == len(pa.visible) + len(pb.visible), name
        assert len(pc.hidden) == len(pa.hidden) + len(pb.hidden), name


def test_time_budget_raises_timeout():
    from src.drawing.brep_views import project_brep
    with pytest.raises(TimeoutError):
        project_brep(_box_hole(), VIEWS["front"], exact=True, deflection=0.02, time_budget_s=1e-9)
    with pytest.raises(TimeoutError):
        project_brep(_box_hole(), VIEWS["front"], exact=False, deflection=0.05, time_budget_s=1e-9)
    # a sane budget does not raise
    pv = project_brep(_box_hole(), VIEWS["front"], exact=True, deflection=0.02, time_budget_s=60.0)
    assert pv.visible


def test_edges_to_primitives_null_and_style():
    from src.drawing.brep_views import edges_to_primitives
    from src.drawing.model import LineStyle
    assert edges_to_primitives(None, 0.02, LineStyle.VISIBLE) == []
    from OCP.TopoDS import TopoDS_Compound
    assert edges_to_primitives(TopoDS_Compound(), 0.02, LineStyle.VISIBLE) == []
    # a planar face in z = 0 (what HLR returns): a 1 x 2 rectangle and a r 0.5 circle
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCP.gp import gp_Pln, gp_Pnt, gp_Dir
    face = BRepBuilderAPI_MakeFace(gp_Pln(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), 0.0, 1.0, 0.0, 2.0).Face()
    prims = edges_to_primitives(face, 0.02, LineStyle.HIDDEN)
    assert len(prims) == 4
    assert all(isinstance(p, Segment) and p.style == LineStyle.HIDDEN for p in prims)
    assert sorted(round(math.dist(p.a, p.b), 9) for p in prims) == [1.0, 1.0, 2.0, 2.0]
    disc = _cyl((3, 4, 0), (0, 0, 1), 0.5, 1.0)
    circles = [p for p in edges_to_primitives(disc, 0.02, LineStyle.VISIBLE) if isinstance(p, Circle)]
    assert circles and all(c.center == pytest.approx((3.0, 4.0)) and c.radius == pytest.approx(0.5)
                           for c in circles)
    with pytest.raises(ValueError):
        edges_to_primitives(face, 0.0, LineStyle.VISIBLE)


# ----------------------------------------------------------------- sections

def test_choose_section_plane_auto_through_largest_hole():
    from src.drawing.brep_sections import choose_section_plane_brep
    point, normal, marker = choose_section_plane_brep(_plate_model(), _plate_features(), "auto")
    assert point[1] == pytest.approx(30.0, abs=1e-9)
    assert point[0] == pytest.approx(50.0, abs=1e-9)
    assert np.allclose(normal, [0.0, -1.0, 0.0])
    assert marker == "top"
    assert np.allclose(normal, VIEWS["front"].n)


def test_choose_section_plane_fallbacks_and_modes():
    from src.drawing.brep_sections import choose_section_plane_brep
    # no hole perpendicular to Y (only the boss and a Y-axis hole) -> bbox centre
    feats = [CylFeature(np.array([20.0, 30.0, 7.5]), np.array([0.0, 1.0, 0.0]), 4.0, 60.0,
                        "hole", True, 1),
             CylFeature(np.array([78.0, 30.0, 20.0]), np.array([0.0, 0.0, 1.0]), 15.0, 10.0,
                        "boss", True, 1)]
    point, normal, marker = choose_section_plane_brep(_plate_model(), feats, "auto")
    assert np.allclose(point, [50.0, 30.0, 12.5])
    assert np.allclose(normal, [0.0, -1.0, 0.0]) and marker == "top"
    # mode top: horizontal plane, cutting line drawn in the front view
    point, normal, marker = choose_section_plane_brep(_plate_model(), _plate_features(), "top")
    assert np.allclose(normal, VIEWS["top"].n) and marker == "front"
    assert point[2] == pytest.approx(12.5)
    # mode right through the Y-axis hole (perpendicular to X): x of that hole
    point, normal, marker = choose_section_plane_brep(_plate_model(), feats, "right")
    assert np.allclose(normal, VIEWS["right"].n) and marker == "front"
    assert point[0] == pytest.approx(20.0)
    with pytest.raises(ValueError):
        choose_section_plane_brep(_plate_model(), feats, "diagonal")
    # bbox taken from the shape when the census has none
    plate = _plate()
    point, _, _ = choose_section_plane_brep(types.SimpleNamespace(shape=plate, census={}), [], "auto")
    assert np.allclose(point, [50.0, 30.0, 12.5], atol=1e-6)


def _areas(loops):
    from src.drawing.geometry2d import polygon_area
    return sorted(abs(polygon_area(L)) for L in loops)


def test_section_plate_through_big_hole():
    from src.drawing.brep_sections import section_brep
    plate = _plate()
    sv = section_brep(plate, (50.0, 30.0, 7.5), (0.0, -1.0, 0.0), VIEWS["front"], exact=True,
                      deflection=0.02, depth_extent=60.0)
    assert len(sv.loops) == 2
    assert _areas(sv.loops) == pytest.approx([600.0, 900.0], abs=0.5)
    for L in sv.loops:
        assert L.ndim == 2 and L.shape[1] == 2
        assert np.linalg.norm(L[0] - L[-1]) > 1e-9      # no repeated end point
    allpts = np.vstack(sv.loops)
    assert allpts[:, 0].min() == pytest.approx(0.0, abs=1e-6)
    assert allpts[:, 0].max() == pytest.approx(100.0, abs=1e-6)
    assert allpts[:, 1].max() == pytest.approx(25.0, abs=1e-6)    # view y = world Z
    assert sv.body is not None and sv.body.visible
    assert sv.stats["open_wires_dropped"] == 0
    assert sv.stats["nudged"] is False
    assert np.allclose(sv.plane_normal, VIEWS["front"].n)
    assert sv.spec == VIEWS["front"]
    assert sv.marker_view == "top"
    assert sv.stats["coplanar"] == 0
    # the body spans the full width
    assert sv.body.bbox.xmin == pytest.approx(0.0, abs=1e-6)
    assert sv.body.bbox.xmax == pytest.approx(100.0, abs=1e-6)
    # the KEPT half is the one behind the plane (y >= 30): the d20 hole walls are cut through
    # the axis, so x = 40 / 60 are visible vertical lines and NOT hidden (the other half shows
    # its uncut front face y = 0 and the walls x = 40 / 60 become hidden)
    vx, hx = _vertical_xs(sv.body.visible), _vertical_xs(sv.body.hidden)
    assert {40.0, 60.0} <= vx, sorted(vx)
    assert 40.0 not in hx and 60.0 not in hx, sorted(hx)


def _vertical_xs(prims):
    return {round(s.a[0], 3) for s in _segs(prims)
            if abs(s.a[0] - s.b[0]) < 1e-6 and abs(s.a[1] - s.b[1]) > 1.0}


def test_section_body_negative_control_wrong_half(monkeypatch):
    """Feeding the half-space the opposite side keeps the front half: the wall check fails."""
    import src.drawing.brep_sections as bs
    plate = _plate()
    good = bs.section_brep(plate, (50.0, 30.0, 7.5), (0.0, -1.0, 0.0), VIEWS["front"], exact=True,
                           deflection=0.02, depth_extent=60.0)
    assert {40.0, 60.0} <= _vertical_xs(good.body.visible), "precondition: correct half shows the walls"
    orig = bs._half_space_cut
    monkeypatch.setattr(bs, "_half_space_cut", lambda s, p, n: orig(s, p, -np.asarray(n)))
    bad = bs.section_brep(plate, (50.0, 30.0, 7.5), (0.0, -1.0, 0.0), VIEWS["front"], exact=True,
                          deflection=0.02, depth_extent=60.0)
    assert len(bad.loops) == 2                      # the loops do not depend on the kept half
    assert not ({40.0, 60.0} <= _vertical_xs(bad.body.visible))
    assert {40.0, 60.0} <= _vertical_xs(bad.body.hidden)


def test_section_hidden_false_drops_body_hidden_lines():
    from src.drawing.brep_sections import section_brep
    plate = _plate()
    with_h = section_brep(plate, (50.0, 30.0, 7.5), (0.0, -1.0, 0.0), VIEWS["front"], exact=True,
                          deflection=0.02, depth_extent=60.0)
    assert with_h.body.hidden, "precondition: the body has hidden lines (the far d6.5 holes)"
    no_h = section_brep(plate, (50.0, 30.0, 7.5), (0.0, -1.0, 0.0), VIEWS["front"], exact=True,
                        deflection=0.02, depth_extent=60.0, hidden=False)
    assert no_h.body.hidden == []
    assert len(no_h.body.visible) == len(with_h.body.visible)
    assert len(no_h.loops) == 2


def test_section_coplanar_with_top_face_is_nudged(monkeypatch):
    """z = 15 lies in the plate's top face: without the nudge the boss footprint is a hole."""
    import src.drawing.brep_sections as bs
    plate = _plate()
    boss = math.pi * 15.0 ** 2
    # negative control first (also the precondition: OCCT returns the face outline, not empty)
    monkeypatch.setattr(bs, "_NUDGE_COPLANAR", False)
    raw = bs.section_brep(plate, (50.0, 30.0, 15.0), (0.0, 0.0, 1.0), VIEWS["top"], exact=True,
                          deflection=0.02, depth_extent=25.0)
    assert raw.stats["nudged"] is False
    assert len(raw.loops) == 7
    assert any(abs(a - boss) < 2.0 for a in _areas(raw.loops)), "precondition: boss footprint loop"
    monkeypatch.setattr(bs, "_NUDGE_COPLANAR", True)
    sv = bs.section_brep(plate, (50.0, 30.0, 15.0), (0.0, 0.0, 1.0), VIEWS["top"], exact=True,
                         deflection=0.02, depth_extent=25.0)
    assert sv.stats["nudged"] is True and sv.stats["coplanar"] >= 1
    assert float(sv.plane_point[2]) == pytest.approx(14.75, abs=1e-9)   # toward the centre z 12.5
    holes = 4 * math.pi * 3.25 ** 2 + math.pi * 10.0 ** 2
    assert len(sv.loops) == 6                          # outer + 5 holes, no boss footprint
    assert not any(abs(a - boss) < 2.0 for a in _areas(sv.loops))
    assert max(_areas(sv.loops)) == pytest.approx(6000.0, abs=0.5)
    from src.drawing.geometry2d import polygon_area
    net = sum(polygon_area(L) for L in sv.loops)       # outers CCW, holes CW
    # inscribed hole polygons (sag 0.02) are ~1.9 mm2 smaller than the true circles
    assert 6000.0 - holes < abs(net) < 6000.0 - holes + 2.5
    # the boss top face (z = 25) alone: nudged down into the boss -> its full disc
    top = bs.section_brep(plate, (78.0, 30.0, 25.0), (0.0, 0.0, 1.0), VIEWS["top"], exact=True,
                          deflection=0.02, depth_extent=25.0)
    assert top.stats["coplanar"] >= 1 and top.stats["nudged"] is True
    assert float(top.plane_point[2]) == pytest.approx(24.75, abs=1e-9)
    assert len(top.loops) == 1 and _areas(top.loops)[0] == pytest.approx(boss, abs=2.0)


def test_section_missing_the_shape_is_nudged_then_empty():
    from src.drawing.brep_sections import section_brep
    plate = _plate()
    hit = section_brep(plate, (50.0, 30.0, 7.5), (0.0, -1.0, 0.0), VIEWS["front"], exact=True,
                       deflection=0.02, depth_extent=60.0)
    assert hit.loops and hit.body is not None, "precondition: the same call through the part works"
    sv = section_brep(plate, (50.0, 500.0, 7.5), (0.0, -1.0, 0.0), VIEWS["front"], exact=True,
                      deflection=0.02, depth_extent=60.0)
    assert sv.loops == []
    assert sv.body is None
    assert sv.stats["nudged"] is True and sv.stats["coplanar"] == 0
    # the 0.6 mm nudge (1 % of 60) goes toward the bbox centre (y = 30)
    assert float(sv.plane_point[1]) == pytest.approx(499.4, abs=1e-9)
    assert sv.stats["nudge_mm"] == pytest.approx(0.6, abs=1e-12)    # along n = -Y


def test_section_just_outside_is_nudged_into_the_part():
    """y = -0.3 misses; the 1 % nudge (0.6 mm) goes toward the part and finds the slab."""
    from src.drawing.brep_sections import section_brep
    sv = section_brep(_plate(), (50.0, -0.3, 7.5), (0.0, -1.0, 0.0), VIEWS["front"], exact=True,
                      deflection=0.02, depth_extent=60.0)
    assert sv.stats["nudged"] is True
    assert len(sv.loops) == 1
    assert _areas(sv.loops) == pytest.approx([1500.0], abs=0.5)
    assert float(sv.plane_point[1]) == pytest.approx(0.3, abs=1e-9)


def test_section_tangent_plane_drops_open_wire_and_recovers_loop(monkeypatch):
    import src.drawing.brep_sections as bs
    plate = _plate()
    sv = bs.section_brep(plate, (93.0, 30.0, 7.5), (1.0, 0.0, 0.0), VIEWS["right"], exact=True,
                         deflection=0.02, depth_extent=100.0)
    assert sv.stats["open_wires_dropped"] >= 1
    chord = math.sqrt(3.25 ** 2 - 1.0)
    side = (8.0 - chord) * 15.0
    mid = (60.0 - 2 * (8.0 + chord)) * 15.0
    assert len(sv.loops) == 3
    assert _areas(sv.loops) == pytest.approx([side, side, mid], abs=0.1)
    assert sv.stats["loops_recovered"] == 1 and sv.marker_view == "front"
    # negative control: without the recovery the middle slab is lost with the open wire
    monkeypatch.setattr(bs, "_RECOVER_OPEN", False)
    raw = bs.section_brep(plate, (93.0, 30.0, 7.5), (1.0, 0.0, 0.0), VIEWS["right"], exact=True,
                          deflection=0.02, depth_extent=100.0)
    assert raw.stats["open_wires_dropped"] >= 1
    assert len(raw.loops) == 2


def test_section_rejects_normal_not_matching_spec():
    from src.drawing.brep_sections import section_brep
    with pytest.raises(ValueError):
        section_brep(_plate(), (50.0, 30.0, 7.5), (1.0, 0.0, 0.0), VIEWS["front"], exact=True,
                     deflection=0.02, depth_extent=60.0)
