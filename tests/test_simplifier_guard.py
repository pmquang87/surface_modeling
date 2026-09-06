"""Tests for the measured guard around ShapeUpgrade_UnifySameDomain.

``NURBSSimplifier.simplify`` used to hand back whatever OCCT's
``ShapeUpgrade_UnifySameDomain`` produced, with ConcatBSplines on, no safe
input mode and a 0.1 rad (5.7 deg) angular tolerance -- and no post-condition
at all. A merge that quietly swallowed a fillet or inverted a face could not
be distinguished from a merge that only removed redundant triangle seams.

The discipline tested here is ported from BlinkingSun/stl2step (MIT),
``unifySameOnce`` in ``src/refit_prism_build.cpp``: run the unifier, then
MEASURE the result (volume + surface-type census) and keep the ORIGINAL shape
whenever the measurement moved.

Every gate below gets an input it must say "no" to, plus a pass-through
control so the guard cannot pass by being a blanket refusal:

  (a) faceted box   -> 12 planar faces merge to 6, volume unchanged, accepted
  (b) volume drift  -> injected 1.001x scale, rejected, original returned
  (c) pass-through  -> injected equal shape, accepted (not a blanket refusal)
  (d) cylinder      -> passes through, cylinder count unchanged
  (e) census drift  -> injected equal-volume shape with no cylinder, rejected
  (f) face growth   -> injected equal-volume shape with more faces, rejected
"""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import OCP  # noqa: F401
    HAS_OCP = True
except ImportError:
    HAS_OCP = False

needs_ocp = pytest.mark.skipif(not HAS_OCP, reason="cadquery-ocp not installed")

pytestmark = needs_ocp


# --------------------------------------------------------------------------
# measurement helpers -- deliberately independent of the module under test
# --------------------------------------------------------------------------

def _vol(shape):
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    from OCP.Precision import Precision
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, props, Precision.Confusion_s())
    return float(props.Mass())


def _faces(shape):
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_FACE
    n = 0
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        n += 1
        exp.Next()
    return n


def _types(shape):
    """{surface type name: count} straight from BRepAdaptor_Surface."""
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopoDS import TopoDS
    out = {}
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        t = BRepAdaptor_Surface(TopoDS.Face_s(exp.Current()), False).GetType()
        key = getattr(t, "name", None) or str(t)
        out[key] = out.get(key, 0) + 1
        exp.Next()
    return out


def _faceted_box(a=10.0):
    """A 12-triangle box solid: the exact input UnifySameDomain should fix."""
    from OCP.gp import gp_Pnt
    from OCP.BRepBuilderAPI import (BRepBuilderAPI_MakePolygon,
                                    BRepBuilderAPI_MakeFace,
                                    BRepBuilderAPI_Sewing,
                                    BRepBuilderAPI_MakeSolid)
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_SHELL
    from OCP.TopoDS import TopoDS

    v = [(0, 0, 0), (a, 0, 0), (a, a, 0), (0, a, 0),
         (0, 0, a), (a, 0, a), (a, a, a), (0, a, a)]
    quads = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
             (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    sew = BRepBuilderAPI_Sewing(1e-7)
    for q in quads:
        for tri in ((q[0], q[1], q[2]), (q[0], q[2], q[3])):
            poly = BRepBuilderAPI_MakePolygon()
            for i in tri:
                poly.Add(gp_Pnt(*v[i]))
            poly.Close()
            mf = BRepBuilderAPI_MakeFace(poly.Wire())
            assert mf.IsDone(), "test fixture: triangle face not built"
            sew.Add(mf.Face())
    sew.Perform()
    exp = TopExp_Explorer(sew.SewedShape(), TopAbs_SHELL)
    assert exp.More(), "test fixture: sewing produced no shell"
    solid = BRepBuilderAPI_MakeSolid(TopoDS.Shell_s(exp.Current())).Solid()
    assert _faces(solid) == 12, "test fixture: box is not 12 triangles"
    return solid


def _scaled(shape, factor):
    from OCP.gp import gp_Pnt, gp_Trsf
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    trsf = gp_Trsf()
    trsf.SetScale(gp_Pnt(0, 0, 0), factor)
    return BRepBuilderAPI_Transform(shape, trsf, True).Shape()


def _copy(shape):
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Copy
    return BRepBuilderAPI_Copy(shape).Shape()


def _cylinder(r=5.0, h=10.0):
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
    return BRepPrimAPI_MakeCylinder(r, h).Shape()


def _box_of_volume(target):
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    a = target ** (1.0 / 3.0)
    return BRepPrimAPI_MakeBox(a, a, a).Shape()


class _FakeUnifier:
    """Records how the simplifier drove the unifier; returns ``make(shape)``."""
    make = staticmethod(lambda s: s)
    last = None

    def __init__(self, shape, unify_edges, unify_faces, concat_bsplines):
        self.ctor = (shape, unify_edges, unify_faces, concat_bsplines)
        self.safe_input = None
        self.linear = None
        self.angular = None
        self.built = False
        type(self).last = self

    def SetLinearTolerance(self, value):
        self.linear = value

    def SetAngularTolerance(self, value):
        self.angular = value

    def SetSafeInputMode(self, value):
        self.safe_input = value

    def Build(self):
        self.built = True

    def Shape(self):
        assert self.built, "simplifier called Shape() before Build()"
        return type(self).make(self.ctor[0])


def _inject(monkeypatch, make):
    """Install a fake unifier and prove the injection actually took."""
    import src.nurbs.simplifier as mod
    real = mod.ShapeUpgrade_UnifySameDomain
    fake = type("_Injected", (_FakeUnifier,), {"make": staticmethod(make)})
    monkeypatch.setattr(mod, "ShapeUpgrade_UnifySameDomain", fake)
    assert mod.ShapeUpgrade_UnifySameDomain is fake, "monkeypatch did not take"
    assert mod.ShapeUpgrade_UnifySameDomain is not real, "fake equals the real class"
    return fake


# --------------------------------------------------------------------------
# constructor defaults
# --------------------------------------------------------------------------

def test_default_tolerances_are_conservative_and_explicit_args_still_win():
    from OCP.Precision import Precision
    from src.nurbs.simplifier import NURBSSimplifier

    d = NURBSSimplifier()
    assert d.angular_tol == pytest.approx(math.radians(0.001)), \
        "angular default must be 0.001 deg expressed in radians"
    assert d.angular_tol < math.radians(0.01), "default is not the old 0.1 rad"
    assert d.linear_tol == pytest.approx(Precision.Confusion_s())

    explicit = NURBSSimplifier(linear_tolerance=0.05, angular_tolerance=0.1)
    assert explicit.linear_tol == 0.05
    assert explicit.angular_tol == 0.1


def test_unifier_is_built_safely(monkeypatch):
    """ConcatBSplines OFF, SetSafeInputMode ON, tolerances forwarded."""
    from src.nurbs.simplifier import NURBSSimplifier

    box = _faceted_box()
    fake = _inject(monkeypatch, lambda s: s)
    NURBSSimplifier().simplify(box)

    rec = fake.last
    assert rec is not None, "the simplifier never constructed a unifier"
    shape_arg, unify_edges, unify_faces, concat = rec.ctor
    assert shape_arg is box
    assert (unify_edges, unify_faces) == (True, True)
    assert concat is False, "ConcatBSplines must be OFF (it rebuilds surfaces)"
    assert rec.safe_input is True, "SetSafeInputMode(True) was never called"
    assert rec.linear == pytest.approx(1e-7)
    assert rec.angular == pytest.approx(math.radians(0.001))
    assert rec.built is True


# --------------------------------------------------------------------------
# (a) the real unifier on a real faceted box
# --------------------------------------------------------------------------

def test_faceted_box_merges_to_six_faces_without_moving_volume():
    from src.nurbs.simplifier import NURBSSimplifier

    box = _faceted_box()
    v0 = _vol(box)
    assert _faces(box) == 12 and v0 == pytest.approx(1000.0, abs=1e-9)

    simp = NURBSSimplifier()
    out = simp.simplify(box)

    assert _faces(out) == 6, f"coplanar triangles were not merged: {_faces(out)}"
    assert _vol(out) == pytest.approx(v0, abs=1e-12)
    rep = simp.last_report
    assert rep["accepted"] is True, rep
    assert rep["faces_before"] == 12 and rep["faces_after"] == 6
    assert rep["non_planar_before"] == 0 and rep["non_planar_after"] == 0
    assert rep["volume_before"] == pytest.approx(v0, abs=1e-12)
    assert rep["volume_after"] == pytest.approx(v0, abs=1e-12)


# --------------------------------------------------------------------------
# (b) defect injection: the unifier moves the volume
# --------------------------------------------------------------------------

def test_volume_drift_is_rejected_and_original_is_returned(monkeypatch, caplog):
    import logging
    from src.nurbs.simplifier import NURBSSimplifier

    box = _faceted_box()
    v0 = _vol(box)

    # precondition: the injected defect is real and above the 1e-6 budget
    drifted = _scaled(box, 1.001)
    v_bad = _vol(drifted)
    assert abs(v_bad - v0) / abs(v0) > 1e-6, "the defect was not injected"
    assert _types(drifted) == _types(box), "scaling must not change the census"
    assert _faces(drifted) == _faces(box), "scaling must not change face count"

    _inject(monkeypatch, lambda s: _scaled(s, 1.001))
    simp = NURBSSimplifier()
    with caplog.at_level(logging.WARNING):
        out = simp.simplify(box)

    assert out is box, "the drifted shape was kept instead of the original"
    assert _vol(out) == pytest.approx(v0, abs=1e-12)
    rep = simp.last_report
    assert rep["accepted"] is False, rep
    assert rep["volume_before"] == pytest.approx(v0, abs=1e-9)
    assert rep["volume_after"] == pytest.approx(v_bad, abs=1e-9)
    assert "volume" in (rep["reason"] or "").lower(), rep

    msg = caplog.text
    assert f"{v0:.12g}" in msg, f"the warning does not name the old volume: {msg}"
    assert f"{v_bad:.12g}" in msg, f"the warning does not name the new volume: {msg}"


# --------------------------------------------------------------------------
# (c) pass-through control: the guard is not a blanket refusal
# --------------------------------------------------------------------------

def test_equal_shape_from_the_unifier_is_accepted(monkeypatch):
    from src.nurbs.simplifier import NURBSSimplifier

    box = _faceted_box()
    v0 = _vol(box)

    # precondition: the injected shape really is a different object, and equal
    probe = _copy(box)
    assert probe is not box and not probe.IsSame(box), "the copy is the original"
    assert _vol(probe) == pytest.approx(v0, abs=1e-12)
    assert _types(probe) == _types(box) and _faces(probe) == _faces(box)

    _inject(monkeypatch, _copy)
    simp = NURBSSimplifier()
    out = simp.simplify(box)

    assert simp.last_report["accepted"] is True, simp.last_report
    assert out is not box, "an equal-measure result must be taken, not discarded"
    assert _vol(out) == pytest.approx(v0, abs=1e-12)


# --------------------------------------------------------------------------
# (d) a curved body survives the round trip
# --------------------------------------------------------------------------

def test_cylinder_passes_through_with_its_cylinder_intact():
    from src.nurbs.simplifier import NURBSSimplifier

    cyl = _cylinder()
    v0, t0 = _vol(cyl), _types(cyl)
    assert t0.get("GeomAbs_Cylinder") == 1, f"fixture has no cylinder: {t0}"

    simp = NURBSSimplifier()
    out = simp.simplify(cyl)

    assert _vol(out) == pytest.approx(v0, rel=1e-12)
    assert _types(out).get("GeomAbs_Cylinder") == 1, _types(out)
    rep = simp.last_report
    assert rep["accepted"] is True, rep
    assert rep["non_planar_before"] == rep["non_planar_after"] == 1


# --------------------------------------------------------------------------
# (e) defect injection: same volume, but the curved face is gone
# --------------------------------------------------------------------------

def test_surface_census_change_is_rejected(monkeypatch):
    from src.nurbs.simplifier import NURBSSimplifier

    cyl = _cylinder()
    v0 = _vol(cyl)
    boxy = _box_of_volume(v0)

    # preconditions: volume passes the gate, the census does not
    assert abs(_vol(boxy) - v0) / abs(v0) <= 1e-6, "the volume gate would fire first"
    assert _types(boxy).get("GeomAbs_Cylinder", 0) == 0, "the defect was not injected"
    assert _types(cyl).get("GeomAbs_Cylinder") == 1

    _inject(monkeypatch, lambda s: boxy)
    simp = NURBSSimplifier()
    out = simp.simplify(cyl)

    assert out is cyl, "a shape that lost its cylinder was accepted"
    assert _types(out).get("GeomAbs_Cylinder") == 1
    rep = simp.last_report
    assert rep["accepted"] is False, rep
    assert rep["non_planar_before"] == 1 and rep["non_planar_after"] == 0
    assert "planar" in (rep["reason"] or "").lower(), rep


# --------------------------------------------------------------------------
# (f) defect injection: same volume and census, but more faces
# --------------------------------------------------------------------------

def test_face_count_increase_is_rejected(monkeypatch):
    from src.nurbs.simplifier import NURBSSimplifier

    faceted = _faceted_box()
    merged = NURBSSimplifier().simplify(faceted)   # 6 planar faces
    assert _faces(merged) == 6, "fixture: the 6-face box was not produced"

    # preconditions: equal volume, equal census kind, strictly more faces
    assert _vol(faceted) == pytest.approx(_vol(merged), abs=1e-12)
    assert _types(faceted).get("GeomAbs_Plane") == 12
    assert _faces(faceted) > _faces(merged), "the defect was not injected"

    _inject(monkeypatch, lambda s: faceted)
    simp = NURBSSimplifier()
    out = simp.simplify(merged)

    assert out is merged, "a shape with MORE faces was accepted as a simplification"
    assert _faces(out) == 6
    rep = simp.last_report
    assert rep["accepted"] is False, rep
    assert rep["faces_before"] == 6 and rep["faces_after"] == 12
    assert "face" in (rep["reason"] or "").lower(), rep


# --------------------------------------------------------------------------
# degenerate inputs must not crash and must still report
# --------------------------------------------------------------------------

def test_none_input_returns_none_and_reports_not_run():
    from src.nurbs.simplifier import NURBSSimplifier
    simp = NURBSSimplifier()
    assert simp.simplify(None) is None
    assert simp.last_report["ran"] is False


def test_unifier_exception_returns_the_original(monkeypatch):
    from src.nurbs.simplifier import NURBSSimplifier

    def _boom(shape):
        raise RuntimeError("injected OCCT failure")

    box = _faceted_box()
    _inject(monkeypatch, _boom)
    simp = NURBSSimplifier()
    out = simp.simplify(box)
    assert out is box
    assert simp.last_report["accepted"] is False
    assert "injected OCCT failure" in (simp.last_report["reason"] or "")


def test_null_shape_from_the_unifier_returns_the_original(monkeypatch):
    """``Shape()`` can hand back a null shape; that is a refusal, not a result."""
    from OCP.TopoDS import TopoDS_Shape
    from src.nurbs.simplifier import NURBSSimplifier

    # precondition: the injected result really is null
    assert TopoDS_Shape().IsNull(), "fixture: the injected shape is not null"

    box = _faceted_box()
    _inject(monkeypatch, lambda s: TopoDS_Shape())
    simp = NURBSSimplifier()
    out = simp.simplify(box)

    assert out is box, "a null shape was returned to the caller"
    assert _faces(out) == 12
    assert simp.last_report["accepted"] is False
    assert "null" in (simp.last_report["reason"] or "").lower(), simp.last_report


def test_none_from_the_unifier_returns_the_original(monkeypatch):
    from src.nurbs.simplifier import NURBSSimplifier

    box = _faceted_box()
    _inject(monkeypatch, lambda s: None)
    simp = NURBSSimplifier()
    out = simp.simplify(box)

    assert out is box, "None was returned to the caller"
    assert simp.last_report["accepted"] is False
    assert simp.last_report["reason"], "no reason was recorded"


def test_unimportable_step_audit_is_reported_not_raised(monkeypatch):
    """``simplify`` promises a report, never an exception.

    ``brep_volume`` is imported lazily from ``src.io.step_audit``. If that
    import is done outside the guard, a partial/circular import or a broken
    numpy escapes past every gate with ``last_report`` still empty -- the one
    failure mode this module exists to prevent.
    """
    import src.io.step_audit as step_audit
    from src.nurbs.simplifier import NURBSSimplifier

    box = _faceted_box()

    # positive control: with step_audit intact the guard runs and accepts
    ok = NURBSSimplifier()
    good = ok.simplify(box)
    assert ok.last_report["accepted"] is True, ok.last_report
    assert _faces(good) == 6

    # inject the defect: the measurement dependency loses brep_volume
    monkeypatch.delattr(step_audit, "brep_volume")
    with pytest.raises(ImportError):
        from src.io.step_audit import brep_volume  # noqa: F401
    assert not hasattr(step_audit, "brep_volume"), "the defect was not injected"

    simp = NURBSSimplifier()
    out = simp.simplify(box)          # must NOT raise

    assert out is box, "the original shape was not returned"
    assert simp.last_report["accepted"] is False
    assert simp.last_report["ran"] is False, "the unifier ran without a measurement"
    assert "brep_volume" in (simp.last_report["reason"] or ""), simp.last_report
