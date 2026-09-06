"""Tests for the STEP read-back audit (src/io/step_audit.py).

The audit used to be one function that printed numbers and returned an exit
code, so nothing in the suite ever exercised its failure branch. It is now
split into ``measure_step`` (all the OpenCascade work, returns a dict) and
``verdict`` (pure function of that dict), which lets the verdict be tested
without OCP and lets every gate be driven by an injected defect.

Design borrowed from BlinkingSun/stl2step (MIT): closure and BRepCheck are
necessary but not sufficient, the volume is the cheap global falsifier, and
every gate needs an input it must say "no" to.
"""
import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.io.step_audit import verdict  # noqa: E402

try:
    import OCP  # noqa: F401
    HAS_OCP = True
except ImportError:
    HAS_OCP = False

needs_ocp = pytest.mark.skipif(not HAS_OCP, reason="cadquery-ocp not installed")


def _clean_measurement(**overrides):
    m = {
        "readable": True,
        "faces": 6,
        "solids": 1,
        "shells": 1,
        "closed_stored": [True],
        "closed_measured": [True],
        "free_edges": 0,
        "brepcheck_valid": True,
        "brep_volume_mm3": 1000.0,
        "mesh_volume_mm3": 1000.0,
        "mesh_watertight": True,
        "volume_delta_pct": 0.0,
        "bbox_max_diff_mm": 0.0,
        "expected_faces": None,
        "census_faces": None,
    }
    m.update(overrides)
    return m


# --------------------------------------------------------------------------
# verdict(): pure function, no OCP needed
# --------------------------------------------------------------------------

def test_verdict_clean_measurement_passes():
    code, reasons = verdict(_clean_measurement())
    assert code == 0
    assert reasons == []


def test_verdict_empty_shape_is_not_ok():
    """A STEP that re-reads as zero shells / zero solids must NOT audit clean.

    Regression: the old expression ``valid and free_edges == 0 and all(closed)``
    is True for ``closed == []`` (``all([])`` is True), an empty compound has
    no free bounds and BRepCheck_Analyzer calls it valid, so an empty file
    returned exit 0 with "fully closed valid solid".
    """
    m = _clean_measurement(faces=0, solids=0, shells=0, closed_stored=[],
                           closed_measured=[], free_edges=0,
                           brepcheck_valid=True, brep_volume_mm3=0.0,
                           volume_delta_pct=None)
    code, reasons = verdict(m)
    assert code == 2
    assert any("shell" in r or "solid" in r for r in reasons)


def test_verdict_no_solid_body_fails_even_if_shells_closed():
    # a closed shell that was never promoted: SolidWorks imports a surface body
    m = _clean_measurement(solids=0)
    code, reasons = verdict(m)
    assert code == 2
    assert any("solid" in r for r in reasons)


def test_verdict_free_edges_fail():
    code, reasons = verdict(_clean_measurement(free_edges=4))
    assert code == 2
    assert any("free" in r for r in reasons)


def test_verdict_uses_measured_closure_not_stored_flag():
    # stored flag lies "closed", measured closure says open -> must fail
    m = _clean_measurement(closed_stored=[True], closed_measured=[False])
    code, reasons = verdict(m)
    assert code == 2
    # and the reverse: a stale stored False on a shell that IS closed passes
    m = _clean_measurement(closed_stored=[False], closed_measured=[True])
    code, _ = verdict(m)
    assert code == 0


def test_verdict_brepcheck_invalid_fails():
    code, reasons = verdict(_clean_measurement(brepcheck_valid=False))
    assert code == 2
    assert any("BRepCheck" in r for r in reasons)


def test_verdict_negative_volume_fails():
    # inside-out solid: closure, free edges and BRepCheck can all still pass
    m = _clean_measurement(brep_volume_mm3=-1000.0, volume_delta_pct=-200.0)
    code, reasons = verdict(m)
    assert code == 2
    assert any("volume" in r for r in reasons)


def test_verdict_volume_gate_is_report_only_by_default():
    # a smooth NURBS fit legitimately differs from its chord mesh by ~0.1 %
    m = _clean_measurement(brep_volume_mm3=1001.0, volume_delta_pct=0.1)
    code, _ = verdict(m)
    assert code == 0


def test_verdict_volume_gate_fires_when_a_tolerance_is_given():
    m = _clean_measurement(brep_volume_mm3=1050.0, volume_delta_pct=5.0)
    code, reasons = verdict(m, volume_tol_pct=2.0)
    assert code == 2
    assert any("volume" in r for r in reasons)
    # inside the budget: passes
    code, _ = verdict(_clean_measurement(brep_volume_mm3=1010.0, volume_delta_pct=1.0),
                      volume_tol_pct=2.0)
    assert code == 0


def test_verdict_volume_gate_skipped_for_non_watertight_reference():
    # the mesh volume of an open STL is meaningless: do not gate on it
    m = _clean_measurement(mesh_watertight=False, mesh_volume_mm3=500.0,
                           volume_delta_pct=None)
    code, reasons = verdict(m, volume_tol_pct=1e-4)
    assert code == 0
    assert reasons == []


def test_verdict_unreadable_is_exit_1():
    m = _clean_measurement(readable=False)
    code, reasons = verdict(m)
    assert code == 1


def test_verdict_expected_face_count_mismatch():
    # 2152 patches fitted but fewer faces landed in the file: silently dropped
    m = _clean_measurement(faces=2000, expected_faces=2152)
    code, reasons = verdict(m)
    assert code == 2
    assert any("2152" in r and "2000" in r for r in reasons)
    code, _ = verdict(_clean_measurement(faces=2152, expected_faces=2152))
    assert code == 0


def test_verdict_census_disagreement_is_flagged():
    # the OCP reader can only drop faces, never invent them: census > OCP
    # means the written file has faces the kernel silently discarded on read
    m = _clean_measurement(faces=6, census_faces=7)
    code, reasons = verdict(m)
    assert code == 2
    assert any("census" in r for r in reasons)
    code, _ = verdict(_clean_measurement(faces=6, census_faces=6))
    assert code == 0


# --------------------------------------------------------------------------
# measure_step(): real OpenCascade round trips
# --------------------------------------------------------------------------

def _box_shape(dx=10.0, dy=20.0, dz=30.0):
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    return BRepPrimAPI_MakeBox(dx, dy, dz).Shape()


def _open_shell_of_box():
    """Five of the six faces of a box, as a shell (one face removed)."""
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopoDS import TopoDS, TopoDS_Shell
    from OCP.BRep import BRep_Builder
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Sewing
    faces = []
    exp = TopExp_Explorer(_box_shape(), TopAbs_FACE)
    while exp.More():
        faces.append(TopoDS.Face_s(exp.Current()))
        exp.Next()
    sew = BRepBuilderAPI_Sewing(1e-6)
    for f in faces[:5]:
        sew.Add(f)
    sew.Perform()
    return sew.SewedShape()


def _write_box_stl(path, dx=10.0, dy=20.0, dz=30.0):
    import trimesh
    m = trimesh.creation.box(extents=(dx, dy, dz))
    m.apply_translation((dx / 2, dy / 2, dz / 2))
    m.export(path)
    return m


@needs_ocp
def test_measure_step_closed_box_solid():
    from src.io.exporters import export_step
    from src.io.step_audit import measure_step
    with tempfile.TemporaryDirectory() as d:
        step = os.path.join(d, "box.step")
        stl = os.path.join(d, "box.stl")
        export_step(_box_shape(), step)
        _write_box_stl(stl)
        m = measure_step(step, reference_stl=stl, expected_faces=6)
        assert m["readable"] is True
        assert m["faces"] == 6
        assert m["solids"] == 1
        assert m["shells"] == 1
        assert m["closed_measured"] == [True]
        assert m["free_edges"] == 0
        assert m["brepcheck_valid"] is True
        assert m["brep_volume_mm3"] == pytest.approx(6000.0, rel=1e-9)
        assert m["mesh_volume_mm3"] == pytest.approx(6000.0, rel=1e-9)
        assert m["mesh_watertight"] is True
        assert abs(m["volume_delta_pct"]) < 1e-7
        assert m["bbox_max_diff_mm"] < 1e-6
        assert m["dev_max_mm"] < 1e-6
        code, reasons = verdict(m, volume_tol_pct=1e-6)
        assert code == 0, reasons


@needs_ocp
def test_measure_step_open_shell_is_condemned():
    from src.io.exporters import export_step
    from src.io.step_audit import measure_step
    with tempfile.TemporaryDirectory() as d:
        step = os.path.join(d, "open.step")
        export_step(_open_shell_of_box(), step)
        m = measure_step(step)
        assert m["faces"] == 5, "the defect was not injected"
        assert m["solids"] == 0
        assert m["free_edges"] == 4
        assert m["closed_measured"] == [False]
        code, reasons = verdict(m)
        assert code == 2
        assert any("free" in r for r in reasons)


@needs_ocp
def test_measure_shape_inverted_solid_has_negative_volume():
    """Closure, free edges and BRepCheck all pass on an inside-out solid;
    only the signed volume sees it. Measured in memory: the STEP writer /
    reader pair heals the orientation on the way through the file, so this
    is the check the faceted exporter runs BEFORE writing."""
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_SHELL
    from OCP.TopoDS import TopoDS, TopoDS_Solid
    from OCP.BRep import BRep_Builder
    from src.io.step_audit import measure_shape
    exp = TopExp_Explorer(_box_shape(), TopAbs_SHELL)
    shell = TopoDS.Shell_s(exp.Current())
    reversed_shell = TopoDS.Shell_s(shell.Reversed())
    solid = TopoDS_Solid()
    b = BRep_Builder()
    b.MakeSolid(solid)
    b.Add(solid, reversed_shell)
    m = measure_shape(solid)
    assert m["solids"] == 1 and m["free_edges"] == 0, "the defect was not injected"
    assert m["closed_measured"] == [True]
    assert m["brepcheck_valid"] is True, "BRepCheck was expected to be blind to this"
    assert m["brep_volume_mm3"] < 0
    code, reasons = verdict(m)
    assert code == 2
    assert any("volume" in r for r in reasons)


@needs_ocp
def test_measure_step_syntactically_empty_step_is_unreadable_not_a_traceback():
    from src.io.step_audit import measure_step
    with tempfile.TemporaryDirectory() as d:
        step = os.path.join(d, "empty.step")
        with open(step, "w") as f:
            f.write("ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\nENDSEC;\nEND-ISO-10303-21;\n")
        m = measure_step(step)
        assert m["readable"] is False
        code, _ = verdict(m)
        assert code == 1


@needs_ocp
def test_measure_step_deviation_sees_a_translated_reference():
    """The sampled STEP->STL deviation must report ~1 mm when the reference
    is shifted by 1 mm, so a checker returning 0 cannot pass."""
    from src.io.exporters import export_step
    from src.io.step_audit import measure_step
    import trimesh
    with tempfile.TemporaryDirectory() as d:
        step = os.path.join(d, "box.step")
        stl = os.path.join(d, "shifted.stl")
        export_step(_box_shape(), step)
        m_stl = trimesh.creation.box(extents=(10.0, 20.0, 30.0))
        m_stl.apply_translation((5.0, 10.0, 15.0 + 1.0))
        m_stl.export(stl)
        m = measure_step(step, reference_stl=stl)
        # the box moved 1 mm in z: the top/bottom faces are 1 mm off, the side
        # faces are 0 off, so the max is ~1 and the mean is in between
        assert m["dev_max_mm"] == pytest.approx(1.0, abs=0.05)
        assert m["dev_mean_mm"] > 0.05
        assert m["bbox_max_diff_mm"] == pytest.approx(1.0, abs=1e-6)


@needs_ocp
def test_measure_step_accepts_an_in_memory_reference_mesh():
    """The faceted exporter hands the welded trimesh straight to the audit."""
    import trimesh
    from src.io.exporters import export_step
    from src.io.step_audit import measure_step
    ref = trimesh.creation.box(extents=(10.0, 20.0, 30.0))
    ref.apply_translation((5.0, 10.0, 15.0))
    with tempfile.TemporaryDirectory() as d:
        step = os.path.join(d, "box.step")
        export_step(_box_shape(), step)
        m = measure_step(step, reference_stl=ref)
        assert m["mesh_watertight"] is True
        assert m["mesh_volume_mm3"] == pytest.approx(6000.0, rel=1e-9)
        assert abs(m["volume_delta_pct"]) < 1e-7
        assert m["dev_max_mm"] < 1e-6
        # an open reference mesh yields no volume comparison at all
        open_ref = trimesh.Trimesh(vertices=ref.vertices, faces=ref.faces[:-1], process=False)
        assert not open_ref.is_watertight, "the defect was not injected"
        m2 = measure_step(step, reference_stl=open_ref)
        assert m2["mesh_watertight"] is False
        assert m2["mesh_volume_mm3"] is None and m2["volume_delta_pct"] is None


@needs_ocp
def test_brep_volume_uses_adaptive_integration():
    from src.io.step_audit import brep_volume
    assert brep_volume(_box_shape()) == pytest.approx(6000.0, rel=1e-12)
