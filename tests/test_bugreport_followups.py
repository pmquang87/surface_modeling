"""Same-pattern follow-ups to the bug report ``bug-report.md`` (automated bug
hunt on commit 4e2241347de43175401df20c6722f38fa70820b4).

The first round fixed each bug where the report found it. Reviewers of that
round then found the same defect again in files outside the fixed group. Each
test here names its follow-up item and the original bug it mirrors:

* F1 (mirrors M15, cand 7/8) - ``gui.panels.face_normal`` kept the absolute
  ``norm > 1e-8`` degeneracy test and no centring, so the Properties panel
  showed a zero normal for a valid triangle with 1e-5 legs.
* F2 (mirrors S:nan_volume, the NaN volume gates fixed in
  ``faceted_step.unify_verdict`` and ``NURBSSimplifier.simplify``) -
  ``step_audit.verdict`` gated the B-Rep volume with ``vol <= 0``, the volume
  budget with ``abs(delta) > tol`` and the shell / solid counts with ``== 0`` /
  ``< 1``; every comparison with NaN is False, so NaN passed all four.
* F3 (mirrors S:nan_volume, unify's "None, not NaN" report rule) -
  ``export_step_faceted`` copied a NaN B-Rep volume into
  ``volume_brep_mm3`` / ``volume_delta_pct`` (and ``measure_shape`` into the
  embedded audit), so the report was no longer strict JSON.
* F4 (mirrors M11, cand 24) - ``step_audit`` failed the audit for a census
  with ``surfaces_consistent=False`` but passed one with
  ``curves_consistent=False``, the EDGE_CURVE half of the same invariant,
  which ``compare_expect`` already rejects.

Test discipline (the repo's vacuous-test rules): each test first asserts that
the defect / precondition really is in its input, every checker gets an input
it must accept (positive control) and one it must reject, and expected values
are exact.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.halfedge_mesh import HalfEdgeMesh  # noqa: E402


# --------------------------------------------------------------------------
# F1 (mirrors M15): the Properties panel normal of a small face
# --------------------------------------------------------------------------

# A nominally collinear triple far from the origin. Centred, its Newell vector
# is below the relative threshold (degenerate -> zero vector); uncentred, the
# rounding of P + Q leaves a noise vector above it (measured: |n| = 9.7e-19
# against a threshold of 2.0e-22), which normalises to a unit "normal".
_FAR_COLLINEAR = [
    [1182.8084516740714, 575.9326302015468, 105.62701902348016],
    [1182.808448309971, 575.9326348601165, 105.62701918277486],
    [1182.8084449458706, 575.9326395186863, 105.62701934206956],
]


def _single_face(points):
    mesh = HalfEdgeMesh.from_arrays(np.asarray(points, dtype=np.float64), [[0, 1, 2]])
    assert len(mesh.faces) == 1
    return mesh, mesh.faces[0]


def _core_normal(points):
    mesh, face = _single_face(points)
    mesh.compute_face_normals()
    return np.asarray(face.normal, dtype=np.float64)


@pytest.mark.parametrize("name,points,expected", [
    # positive control: a unit triangle, accepted by both the old and new rule
    ("unit", [[0, 0, 0], [1, 0, 0], [0, 1, 0]], [0.0, 0.0, 1.0]),
    # the F1 defect: a valid triangle with 1e-5 legs (a scan in metres)
    ("legs_1e-5", [[0, 0, 0], [1e-5, 0, 0], [0, 1e-5, 0]], [0.0, 0.0, 1.0]),
    # must be rejected: an exactly collinear (zero-area) face
    ("collinear", [[0, 0, 0], [1, 0, 0], [2, 0, 0]], [0.0, 0.0, 0.0]),
    # must be rejected: a collinear face far from the origin (needs centring)
    ("far_collinear", _FAR_COLLINEAR, [0.0, 0.0, 0.0]),
])
def test_f1_panel_face_normal_matches_the_m15_core_rule(name, points, expected):
    """F1 (mirrors M15): ``gui.panels.face_normal`` says it mirrors
    ``HalfEdgeMesh.compute_face_normals``; it must give the same normal as the
    fixed core routine, including a unit normal for a 1e-5-leg triangle and
    the zero vector for a zero-area face."""
    from src.gui.panels import face_normal

    expected = np.asarray(expected, dtype=np.float64)
    # precondition: the M15-fixed core routine gives the expected value
    core = _core_normal(points)
    assert np.allclose(core, expected, atol=1e-12), (name, core)
    if name == "legs_1e-5":
        # precondition: the old absolute threshold really zeroes this face
        P = np.asarray(points, dtype=np.float64)
        assert np.linalg.norm(np.cross(P[1] - P[0], P[2] - P[0])) < 1e-8

    mesh, face = _single_face(points)
    shown = face_normal(mesh, face)
    assert np.allclose(shown, expected, atol=1e-12), (name, shown)
    assert np.array_equal(shown, core), (name, shown, core)
    # the value is cached on the face, as before
    assert np.array_equal(np.asarray(face.normal), shown)


# --------------------------------------------------------------------------
# F2 (mirrors S:nan_volume): NaN pass-through in step_audit.verdict
# --------------------------------------------------------------------------

NAN = float("nan")
INF = float("inf")


def _clean_measurement(**overrides):
    """A measurement every gate of ``verdict`` accepts (as in test_step_audit)."""
    m = {
        "readable": True, "faces": 6, "solids": 1, "shells": 1,
        "closed_stored": [True], "closed_measured": [True],
        "free_edges": 0, "brepcheck_valid": True,
        "brep_volume_mm3": 1000.0, "mesh_volume_mm3": 1000.0,
        "mesh_watertight": True, "volume_delta_pct": 0.0,
        "bbox_max_diff_mm": 0.0, "expected_faces": None,
        "census_faces": None,
    }
    m.update(overrides)
    return m


def test_f2_verdict_positive_controls():
    """F2: the clean measurement passes, with and without a volume budget,
    and a finite delta inside the budget passes too."""
    from src.io.step_audit import verdict
    assert verdict(_clean_measurement()) == (0, [])
    assert verdict(_clean_measurement(), volume_tol_pct=1e-4) == (0, [])
    assert verdict(_clean_measurement(volume_delta_pct=5e-5),
                   volume_tol_pct=1e-4) == (0, [])
    # report-only (no budget): a non-finite delta is not gated, as documented
    assert verdict(_clean_measurement(volume_delta_pct=NAN)) == (0, [])
    # finite out-of-budget delta and non-positive volume still fail as before
    assert verdict(_clean_measurement(volume_delta_pct=5.0), volume_tol_pct=1.0) == (
        2, ["volume differs from the mesh by 5 % (budget 1 %)"])
    assert verdict(_clean_measurement(brep_volume_mm3=-1.0)) == (
        2, ["B-Rep volume not positive: -1.0"])


@pytest.mark.parametrize("vol", [NAN, INF, -INF])
def test_f2_verdict_rejects_a_non_finite_brep_volume(vol):
    """F2 (mirrors S:nan_volume): ``vol <= 0`` is False for NaN, so a failed
    volume integration passed the audit as a clean solid."""
    from src.io.step_audit import verdict
    assert not (NAN <= 0), "precondition: NaN compares False"
    code, reasons = verdict(_clean_measurement(brep_volume_mm3=vol,
                                               volume_delta_pct=None))
    assert code == 2
    assert reasons == [f"B-Rep volume is not finite: {vol!r}"], reasons


@pytest.mark.parametrize("delta", [NAN, INF, -INF])
def test_f2_verdict_rejects_a_non_finite_volume_delta(delta):
    """F2 (mirrors S:nan_volume): ``abs(delta) > budget`` is False for NaN, so
    an unmeasurable volume difference passed the faceted exporter's budget."""
    from src.io.step_audit import verdict
    assert not (abs(NAN) > 1e-4), "precondition: NaN compares False"
    code, reasons = verdict(_clean_measurement(volume_delta_pct=delta),
                            volume_tol_pct=1e-4)
    assert code == 2
    assert reasons == [f"volume difference from the mesh is not finite: {delta!r}"], \
        reasons


@pytest.mark.parametrize("key,value,reason", [
    ("shells", NAN, "no shell in the file"),
    ("solids", NAN, "no solid body (SolidWorks would import a surface body)"),
    ("shells", None, "no shell in the file"),
    ("solids", None, "no solid body (SolidWorks would import a surface body)"),
])
def test_f2_verdict_count_gates_reject_nan(key, value, reason):
    """F2: the shell and solid count gates were written as ``== 0`` / ``< 1``,
    which NaN also slips through; they must fail closed like the others.
    (free_edges / faces / census_faces use ``!=`` and already reject NaN.)"""
    from src.io.step_audit import verdict
    assert verdict(_clean_measurement(**{key: 0})) == (2, [reason])  # rejects 0
    code, reasons = verdict(_clean_measurement(**{key: value}))
    assert (code, reasons) == (2, [reason]), reasons


def test_f2_verdict_ne_gates_already_reject_nan():
    """F2 control: the ``!=`` gates were already NaN-safe; pin that."""
    from src.io.step_audit import verdict
    code, reasons = verdict(_clean_measurement(free_edges=NAN))
    assert code == 2 and reasons == ["free edges: nan"], reasons
    code, reasons = verdict(_clean_measurement(expected_faces=6, faces=NAN))
    assert code == 2 and reasons == ["6 faces expected but nan in the file"], reasons


# --------------------------------------------------------------------------
# F3 (mirrors S:nan_volume): the faceted export report must stay strict JSON
# --------------------------------------------------------------------------

try:
    import OCP  # noqa: F401
    HAS_OCP = True
except ImportError:
    HAS_OCP = False

needs_ocp = pytest.mark.skipif(not HAS_OCP, reason="cadquery-ocp not installed")


def _strict_json(obj):
    """Serialise as strict JSON (no NaN / Infinity tokens) or raise."""
    import json
    return json.dumps(obj, allow_nan=False, default=str)


def test_f3_strict_json_checker_rejects_nan():
    """F3 control: the checker the F3 tests rely on really rejects NaN and
    accepts None."""
    with pytest.raises(ValueError):
        _strict_json({"volume": NAN})
    with pytest.raises(ValueError):
        _strict_json({"volume": INF})
    assert _strict_json({"volume": None}) == '{"volume": null}'


@needs_ocp
@pytest.mark.parametrize("bad", [NAN, INF])
def test_f3_faceted_report_stores_none_for_a_non_finite_brep_volume(
        tmp_path, monkeypatch, bad):
    """F3 (mirrors S:nan_volume / unify's ``volume_rel_delta``): when the
    volume integrator returns a non-finite value, ``export_step_faceted``
    must store None (not NaN) in ``volume_brep_mm3`` / ``volume_delta_pct``,
    in the embedded audit as well, warn about it, and the whole report must
    serialise as strict JSON. The audit still refuses the file."""
    import trimesh
    import src.io.step_audit as step_audit
    from src.io.faceted_step import export_step_faceted

    box = trimesh.creation.box(extents=(10.0, 20.0, 30.0))

    # positive control: the real integrator gives a finite, exact report
    good = export_step_faceted(box, str(tmp_path / "good.step"))
    assert good["ok"] is True and good["audit_reasons"] == [], good
    assert good["volume_brep_mm3"] == pytest.approx(6000.0, rel=1e-9)
    assert abs(good["volume_delta_pct"]) < 1e-7
    _strict_json(good)

    calls = []

    def bad_volume(shape):
        calls.append(1)
        return bad
    monkeypatch.setattr(step_audit, "brep_volume", bad_volume)

    for verify in (False, True):
        calls.clear()
        report = export_step_faceted(box, str(tmp_path / f"bad_{verify}.step"),
                                     verify=verify)
        # precondition: the injected value reached the exporter
        assert calls, "the injected integrator was never called"
        assert report["ok"] is True, report      # a file was still written
        assert report["volume_mesh_mm3"] == pytest.approx(6000.0, rel=1e-12)
        assert report["volume_brep_mm3"] is None, report["volume_brep_mm3"]
        assert report["volume_delta_pct"] is None, report["volume_delta_pct"]
        assert any("volume" in w and "not finite" in w for w in report["warnings"]), \
            report["warnings"]
        if verify:
            audit = report["audit"]
            assert audit["brep_volume_mm3"] is None, audit["brep_volume_mm3"]
            assert audit["volume_delta_pct"] is None, audit["volume_delta_pct"]
            assert "B-Rep volume not positive: None" in report["audit_reasons"], \
                report["audit_reasons"]
        _strict_json(report)


# --------------------------------------------------------------------------
# F4 (mirrors M11): an inconsistent curve census must not pass the audit
# --------------------------------------------------------------------------

def _good_census(faces=6):
    return {"ok": True, "input": "x.step", "faces": faces, "edges": 12,
            "surfaces": {"plane": faces, "cylinder": 0, "cone": 0, "sphere": 0,
                         "torus": 0, "bspline": 0, "other": 0},
            "curves": {"line": 12, "circle": 0, "bspline": 0, "other": 0},
            "surfaces_consistent": True, "curves_consistent": True,
            "cylinder_radii": [], "cylinder_radii_unique": []}


def test_f4_verdict_flags_inconsistent_curves_like_surfaces():
    """F4 (mirrors M11, cand 24): ``curves_consistent=False`` is the
    EDGE_CURVE twin of ``surfaces_consistent`` (step_census docstring, items 2
    and 9): an argument list did not parse, so the census is not a
    trustworthy witness and the audit must say so, as it does for faces."""
    from src.io import step_census
    from src.io.step_audit import verdict

    # positive control: a consistent census that agrees with OCP passes clean
    assert verdict(_clean_measurement(census=_good_census(), census_faces=6)) == (0, [])
    # the surface twin is already rejected (M11) - the reference behaviour
    code, reasons = verdict(_clean_measurement(
        census=dict(_good_census(), surfaces_consistent=False), census_faces=6))
    assert code == 2 and len(reasons) == 1 and "surfaces_consistent" in reasons[0]
    # precondition: compare_expect already treats the curve flag as a failure
    assert step_census.compare_expect({"faces": 6},
                                      dict(_good_census(), curves_consistent=False))

    code, reasons = verdict(_clean_measurement(
        census=dict(_good_census(), curves_consistent=False), census_faces=6))
    assert code == 2, reasons
    assert reasons == ["text census failed: an EDGE_CURVE has no curve bucket "
                       "(curves_consistent)"], reasons


@needs_ocp
def test_f4_measure_step_carries_the_curve_flag_after_the_census_is_popped(
        tmp_path, monkeypatch):
    """F4, real route: ``measure_step`` sets the flat ``census_error`` that
    survives convert.py / faceted_step.py popping the census dict; a census
    with ``curves_consistent=False`` must reach ``verdict`` through it."""
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from src.io import step_audit, step_census
    from src.io.exporters import export_step
    from src.io.step_audit import verdict

    step = str(tmp_path / "box.step")
    export_step(BRepPrimAPI_MakeBox(10.0, 20.0, 30.0).Shape(), step)

    # positive control: the intact file audits clean with the census as witness
    m = step_audit.measure_step(step, expected_faces=6)
    assert m["census"]["curves_consistent"] is True and m["census_faces"] == 6
    m.pop("census")
    assert verdict(m) == (0, [])

    real = step_census.census_path

    def broken_curves(path):
        c = real(path)
        assert c["ok"] is True and c["edges"] == 12   # precondition: real census
        c["curves"]["line"] -= 1                      # one EDGE_CURVE "dropped"
        c["curves_consistent"] = False
        return c
    monkeypatch.setattr(step_census, "census_path", broken_curves)

    m = step_audit.measure_step(step, expected_faces=6)
    assert m["census_faces"] == 6, "precondition: the face witness still agrees"
    m.pop("census")
    code, reasons = verdict(m)
    assert code == 2, reasons
    assert reasons == ["text census failed: an EDGE_CURVE has no curve bucket "
                       "(curves_consistent)"], reasons
