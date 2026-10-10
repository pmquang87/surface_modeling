"""Regression tests for the convert CLI and the STEP audit / census / canonical
form, from the automated bug hunt on commit
4e2241347de43175401df20c6722f38fa70820b4 (bug-report.md).

Covers:

* M2  (cand 1)  - ``convert()`` reported ``"ok": true`` in RESULT for a
  conversion that failed after the STEP was written (audit verdict 1, or the
  audit raised).
* M11 (cand 24) - ``step_audit.verdict`` passed clean when the text census
  itself had failed (``ok: False`` / ``error``) or was internally
  inconsistent, i.e. the second witness was silently dropped.
* cand 25 (suspicion) - ``step_canonical._string_mask`` did not know
  ``/* ... */`` comments, so an apostrophe in a comment flipped the string
  state and the DATA section was reported missing.
* compare_expect (suspicion) - ``step_census.compare_expect`` ignored the
  census keys ``edges`` and ``curves_consistent`` and compared the expected
  cylinder radii in file order against the sorted census list.
* measure_shape (suspicion) - ``step_audit.measure_shape`` crashed for
  ``deviation_samples=0``.

Every gate gets a positive control (intact input, gate ran and passed) and an
injected defect that trips exactly that gate.
"""
import json
import os
import sys

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)

from src.io import step_audit, step_census  # noqa: E402
from src.io.step_audit import verdict  # noqa: E402
from src.io.step_canonical import canonical_step_text, extract_data_section  # noqa: E402

try:
    import OCP  # noqa: F401
    HAS_OCP = True
except ImportError:
    HAS_OCP = False

needs_ocp = pytest.mark.skipif(not HAS_OCP, reason="cadquery-ocp not installed")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _clean_measurement(**overrides):
    """A measurement every gate of ``verdict`` accepts (same as test_step_audit)."""
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


def _good_census(faces=6):
    return {"ok": True, "input": "x.step", "faces": faces, "edges": 12,
            "surfaces": {"plane": faces, "cylinder": 0, "cone": 0, "sphere": 0,
                         "torus": 0, "bspline": 0, "other": 0},
            "curves": {"line": 12, "circle": 0, "bspline": 0, "other": 0},
            "surfaces_consistent": True, "curves_consistent": True,
            "cylinder_radii": [], "cylinder_radii_unique": []}


def _box_stl(path, extents=(10.0, 20.0, 30.0)):
    import trimesh
    m = trimesh.creation.box(extents=extents)
    m.export(path)
    return m


def _box_step(path):
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from src.io.exporters import export_step
    export_step(BRepPrimAPI_MakeBox(10.0, 20.0, 30.0).Shape(), path)
    assert os.path.getsize(path) > 0
    return path


def _deep_offset_chain(depth, first_id=900000):
    """An ADVANCED_FACE on a ``depth``-long OFFSET_SURFACE chain ending in a
    PLANE: legal Part 21, no cycle, but deeper than Python's recursion limit
    for the recursive ``resolve_surface``."""
    lines = [f"#{first_id} = ADVANCED_FACE('',(#{first_id - 1}),#{first_id + 1},.T.);"]
    for k in range(depth):
        lines.append(f"#{first_id + 1 + k} = OFFSET_SURFACE('',#{first_id + 2 + k},1.,.F.);")
    lines.append(f"#{first_id + 1 + depth} = PLANE('',#{first_id - 2});")
    return "\n".join(lines) + "\n"


def _run_main(argv, capsys):
    from src.convert import main
    code = main(argv)
    out = capsys.readouterr().out
    lines = [ln for ln in out.splitlines() if ln.strip()]
    assert lines and lines[-1].startswith("RESULT "), out
    return code, json.loads(lines[-1][len("RESULT "):])


# --------------------------------------------------------------------------
# M2 (cand 1): RESULT JSON must say ok: false when convert() fails
# --------------------------------------------------------------------------

@needs_ocp
def test_m2_convert_audit_verdict_1_reports_ok_false(tmp_path, monkeypatch, capsys):
    """M2 (cand 1): a written STEP whose audit verdict is 1 (unusable) must
    give exit 1 AND ``"ok": false`` with the verdict reasons as ``error``."""
    stl = str(tmp_path / "box.stl")
    step = str(tmp_path / "box.step")
    _box_stl(stl)

    # positive control: the same conversion unpatched is clean
    code, res = _run_main([stl, step, "--faceted", "--quiet"], capsys)
    assert code == 0 and res["ok"] is True and res["exit_code"] == 0, res
    assert res["audit_reasons"] == []
    os.remove(step)

    # injected defect: the audit condemns the file (e.g. OCCT cannot re-read it)
    calls = []

    def condemn(m, volume_tol_pct=None):
        calls.append(m)
        return 1, ["written STEP cannot be re-read as a shape"]
    monkeypatch.setattr(step_audit, "verdict", condemn)
    code, res = _run_main([stl, step, "--faceted", "--quiet"], capsys)
    assert len(calls) == 1, "the injected verdict was not used"
    assert os.path.exists(step), "precondition: a STEP was written before the audit"
    assert code == 1
    assert res["exit_code"] == 1
    assert res["ok"] is False, res
    assert res["error"] == "written STEP cannot be re-read as a shape"


@needs_ocp
def test_m2_convert_audit_raising_reports_ok_false(tmp_path, monkeypatch, capsys):
    """M2 (cand 1): an exception inside the audit (after the STEP was written)
    must give exit 1 AND ``"ok": false`` with the exception as ``error``."""
    stl = str(tmp_path / "box.stl")
    step = str(tmp_path / "box.step")
    _box_stl(stl)

    def boom(*args, **kwargs):
        raise RuntimeError("injected audit crash")
    monkeypatch.setattr(step_audit, "measure_step", boom)
    code, res = _run_main([stl, step, "--faceted", "--quiet"], capsys)
    assert os.path.exists(step), "precondition: a STEP was written before the audit"
    assert code == 1
    assert res["exit_code"] == 1
    assert res["ok"] is False, res
    assert res["error"] == "RuntimeError: injected audit crash"

    # the same direct call through convert() agrees with the CLI
    from src.convert import convert
    code2, res2 = convert(stl, step, faceted=True, quiet=True)
    assert (code2, res2["ok"], res2["exit_code"]) == (1, False, 1), res2


# --------------------------------------------------------------------------
# M11 (cand 24): a failed / inconsistent text census must not pass the audit
# --------------------------------------------------------------------------

def test_m11_verdict_census_failure_is_not_a_clean_pass():
    """M11 (cand 24): ``census={'ok': False, 'error': ...}`` with
    ``census_faces=None`` must give code 2 with a census reason, not (0, [])."""
    # positive control: a good census that agrees with OCP passes clean
    assert verdict(_clean_measurement(census=_good_census(), census_faces=6)) == (0, [])
    # and no census at all (module unavailable) stays report-only
    assert verdict(_clean_measurement(census=None)) == (0, [])

    failed = {"ok": False, "input": "x.step",
              "error": "RecursionError: maximum recursion depth exceeded"}
    code, reasons = verdict(_clean_measurement(census=failed, census_faces=None))
    assert code == 2
    assert len(reasons) == 1, reasons
    assert "census" in reasons[0] and "RecursionError" in reasons[0], reasons

    # the step_audit._census envelope form: {"error": ...} without "ok"
    code, reasons = verdict(_clean_measurement(census={"error": "boom"}))
    assert code == 2 and len(reasons) == 1 and "boom" in reasons[0], reasons


def test_m11_verdict_census_inconsistent_surfaces_is_flagged():
    """M11 (cand 24): ``surfaces_consistent=False`` means an ADVANCED_FACE did
    not parse; the witness is not trustworthy, so the audit must say so."""
    bad = _good_census()
    bad["surfaces_consistent"] = False
    code, reasons = verdict(_clean_measurement(census=bad, census_faces=6))
    assert code == 2
    assert len(reasons) == 1 and "census" in reasons[0], reasons


def test_m11_verdict_sees_census_failure_after_the_census_dict_is_popped():
    """M11 (cand 24): convert.py and faceted_step.py pop the verbose ``census``
    dict before calling ``verdict``; the flat ``census_error`` key that
    ``measure_step`` sets must still carry the failure."""
    code, reasons = verdict(_clean_measurement(census_error="RecursionError: x"))
    assert code == 2 and len(reasons) == 1 and "RecursionError" in reasons[0], reasons
    assert verdict(_clean_measurement(census_error=None)) == (0, [])


@needs_ocp
def test_m11_measure_step_deep_offset_chain_census_failure_fails_audit(tmp_path):
    """M11 (cand 24), real route: a 2,000-deep OFFSET_SURFACE chain makes the
    census hit RecursionError; the audit of that file must not be clean."""
    good = _box_step(str(tmp_path / "box.step"))
    # positive control: the intact file audits clean with the census as witness
    m = step_audit.measure_step(good, expected_faces=6)
    assert m["census_faces"] == 6
    assert verdict(m) == (0, [])

    with open(good, encoding="utf-8") as fh:
        text = fh.read()
    marker = "ENDSEC;\nEND-ISO-10303-21;"
    assert text.count(marker) == 1, "unexpected STEP trailer"
    deep = tmp_path / "deep.step"
    deep.write_text(text.replace(marker, _deep_offset_chain(2000) + marker),
                    encoding="utf-8")

    c = step_census.census_path(str(deep))
    assert c["ok"] is False and "RecursionError" in c["error"], \
        "precondition: the deep chain must break the census"
    m = step_audit.measure_step(str(deep), expected_faces=6)
    assert m["readable"] is True and m["faces"] == 6, \
        "precondition: OCCT still reads the box"
    assert m["census_faces"] is None
    code, reasons = verdict(m)
    assert code == 2, reasons
    assert any("census" in r and "RecursionError" in r for r in reasons), reasons


@needs_ocp
def test_m11_convert_census_failure_reaches_the_result(tmp_path, monkeypatch, capsys):
    """M11 (cand 24) through convert(): convert pops the census dict before
    the verdict, so a census failure must still turn exit 0 into exit 2."""
    stl = str(tmp_path / "box.stl")
    step = str(tmp_path / "box.step")
    _box_stl(stl)
    code, res = _run_main([stl, step, "--faceted", "--quiet"], capsys)
    assert code == 0 and res["audit_reasons"] == [], res   # positive control

    monkeypatch.setattr(step_census, "census_path",
                        lambda p: {"ok": False, "input": str(p),
                                   "error": "RecursionError: injected"})
    code, res = _run_main([stl, step, "--faceted", "--quiet"], capsys)
    assert code == 2, res
    assert res["ok"] is True and res["exit_code"] == 2   # a file WAS written
    assert len(res["audit_reasons"]) == 1, res["audit_reasons"]
    assert "census" in res["audit_reasons"][0]
    assert "RecursionError: injected" in res["audit_reasons"][0]


# --------------------------------------------------------------------------
# cand 25: /* ... */ comments in the canonical-form slicer
# --------------------------------------------------------------------------

_STEP_WITH_COMMENT = """ISO-10303-21;
HEADER;
/* {comment} */
FILE_DESCRIPTION(('a model'),'2;1');
FILE_NAME('model','2026-10-10T10:00:00',('Author'),(''),'pp','sys','');
FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));
ENDSEC;
DATA;
#10 = CARTESIAN_POINT('',(1.0,2.0,3.0));
ENDSEC;
END-ISO-10303-21;
"""


def test_cand25_apostrophe_in_a_comment_does_not_hide_the_data_section():
    """cand 25: an apostrophe inside ``/* ... */`` is not a string delimiter;
    the DATA section must still be found and sliced."""
    plain = _STEP_WITH_COMMENT.format(comment="Bobs file")
    bobs = _STEP_WITH_COMMENT.format(comment="Bob's file")
    assert bobs.count("'") % 2 == 1, "precondition: an odd number of apostrophes"
    # positive control: without the apostrophe the file slices
    assert "#10 = CARTESIAN_POINT" in canonical_step_text(plain)

    data = extract_data_section(bobs)
    assert data.startswith("DATA;") and "#10 = CARTESIAN_POINT" in data
    canon = canonical_step_text(bobs)
    assert "Bob's file" in canon, "the comment must stay in the canonical HEADER"
    assert "2026-10-10T10:00:00" not in canon
    # the canonical form still discriminates on this file
    other = bobs.replace("(1.0,2.0,3.0)", "(1.0,2.0,3.5)")
    assert canonical_step_text(other) != canon


def test_cand25_keywords_inside_a_comment_are_not_section_keywords():
    """cand 25: a ``DATA;`` line inside a comment is not the DATA section, and
    a ``/*`` inside a string does not open a comment."""
    decoy = ("/*\nDATA;\n#99 = FAKE_POINT('',(9.,9.,9.));\nENDSEC;\n*/")
    text = _STEP_WITH_COMMENT.replace("/* {comment} */", decoy)
    assert text.find("\nDATA;\n") < text.rfind("\nDATA;\n"), \
        "precondition: the decoy DATA; comes first"
    data = extract_data_section(text)
    assert "FAKE_POINT" not in data and "CARTESIAN_POINT" in data, data

    # a "/*" inside a string literal must not swallow the rest of the file
    in_string = _STEP_WITH_COMMENT.format(comment="x").replace(
        "'a model'", "'path/*.step'")
    assert "'path/*.step'" in in_string
    data = extract_data_section(in_string)
    assert data.startswith("DATA;") and "CARTESIAN_POINT" in data


# --------------------------------------------------------------------------
# compare_expect (suspicion): edges, curves_consistent, radii order
# --------------------------------------------------------------------------

_TWO_CYL_STEP = """ISO-10303-21;
HEADER;
ENDSEC;
DATA;
#1 = ADVANCED_FACE('',(#9),#20,.T.);
#2 = ADVANCED_FACE('',(#9),#21,.T.);
#20 = CYLINDRICAL_SURFACE('',#30,10.);
#21 = CYLINDRICAL_SURFACE('',#30,5.);
#40 = EDGE_CURVE('',#50,#51,#60,.T.);
#60 = LINE('',#70,#71);
ENDSEC;
END-ISO-10303-21;
"""


def test_compare_expect_checks_edges_from_a_census_produced_truth():
    """compare_expect suspicion: a truth file written from the census itself
    carries ``edges``; a file with a different edge count must not pass."""
    actual = step_census.census_text(_TWO_CYL_STEP)
    assert actual["ok"] and actual["edges"] == 1 and actual["faces"] == 2
    truth = json.loads(json.dumps(actual))
    assert step_census.compare_expect(truth, actual) == []   # positive control

    truth["edges"] = 2
    errs = step_census.compare_expect(truth, actual)
    assert errs == ["edges: got 1 want 2"], errs


def test_compare_expect_flags_inconsistent_curves():
    """compare_expect suspicion: ``curves_consistent=False`` is the EDGE_CURVE
    twin of ``surfaces_consistent`` and must fail the expectation too."""
    actual = step_census.census_text(_TWO_CYL_STEP)
    assert actual["curves_consistent"] is True
    assert step_census.compare_expect({"faces": 2}, actual) == []
    broken = dict(actual, curves_consistent=False)
    errs = step_census.compare_expect({"faces": 2}, broken)
    assert len(errs) == 1 and errs[0].startswith("curves_consistent"), errs


def test_compare_expect_radii_are_compared_as_a_sorted_list():
    """compare_expect suspicion: the census sorts ``cylinder_radii``; an
    expectation listing the same radii in another order must pass, and a
    different radius must still fail."""
    actual = step_census.census_text(_TWO_CYL_STEP)
    assert actual["cylinder_radii"] == [5, 10], actual["cylinder_radii"]
    assert step_census.compare_expect({"cylinder_radii": [5, 10]}, actual) == []
    assert step_census.compare_expect({"cylinder_radii": [10, 5]}, actual) == []
    errs = step_census.compare_expect({"cylinder_radii": [10, 6]}, actual)
    assert errs == ["cylinder_radii: got [5, 10] want [6, 10]"], errs


# --------------------------------------------------------------------------
# measure_shape (suspicion): deviation_samples=0
# --------------------------------------------------------------------------

@needs_ocp
def test_measure_shape_with_zero_deviation_samples_skips_the_deviation():
    """measure_shape suspicion: ``deviation_samples=0`` must not crash; it
    measures everything else and leaves the deviation unmeasured (None)."""
    import trimesh
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    shape = BRepPrimAPI_MakeBox(10.0, 20.0, 30.0).Shape()
    ref = trimesh.creation.box(extents=(10.0, 20.0, 30.0))
    ref.apply_translation((5.0, 10.0, 15.0))

    # positive control: with samples the deviation is measured (~0)
    m = step_audit.measure_shape(shape, reference_stl=ref, deviation_samples=50)
    assert m["dev_max_mm"] is not None and m["dev_max_mm"] < 1e-6

    m0 = step_audit.measure_shape(shape, reference_stl=ref, deviation_samples=0)
    assert m0["dev_mean_mm"] is None and m0["dev_p95_mm"] is None
    assert m0["dev_max_mm"] is None
    assert m0["faces"] == 6
    assert m0["brep_volume_mm3"] == pytest.approx(6000.0, rel=1e-9)
    assert m0["bbox_max_diff_mm"] < 1e-6
