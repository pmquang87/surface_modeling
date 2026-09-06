"""Tests for the OCP-free STEP Part 21 text census (src/io/step_census.py).

The census is the *second witness* in the audit: it reads the bytes that were
written, shares no code with the OpenCascade writer or reader, and is compared
against the reader's face count in ``src/io/step_audit.verdict``. A witness
that agrees with the kernel by accident is worthless, so every test here
carries a defect-injected negative control:

* the two parser bugs that were fixed while vendoring are reproduced by
  ``_upstream_split_args`` (a verbatim copy of the upstream for-loop) and by
  censusing the same text with that function monkey-patched in - if the fix
  were a no-op, those controls would pass and fail the test;
* the real-file tests delete or rewrite an entity in a STEP file OpenCascade
  wrote, so the numbers cannot come from a constant.
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from src.io import step_census  # noqa: E402
from src.io.step_census import census_path, census_text, split_args  # noqa: E402

try:
    import OCP  # noqa: F401
    HAS_OCP = True
except ImportError:
    HAS_OCP = False

needs_ocp = pytest.mark.skipif(not HAS_OCP, reason="cadquery-ocp not installed")


# --------------------------------------------------------------------------
# The upstream (unfixed) argument splitter, kept as a negative control.
# Verbatim from BlinkingSun/stl2step tests/tools/step_census.py:120-146. The
# for-loop can only ``continue`` on a doubled quote, so it steps onto the
# second quote of the escape and inverts its own in_str state.
# --------------------------------------------------------------------------
def _upstream_split_args(args: str):
    out = []
    depth = 0
    in_str = False
    start = 0
    for i, c in enumerate(args):
        if in_str:
            if c == "'" and i + 1 < len(args) and args[i + 1] == "'":
                continue
            if c == "'":
                in_str = False
            continue
        if c == "'":
            in_str = True
            continue
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        elif c == "," and depth == 0:
            out.append(args[start:i].strip())
            start = i + 1
    tail = args[start:].strip()
    if tail:
        out.append(tail)
    return out


ESCAPED = "'It''s a quote',(#81),#70,.T."


# --------------------------------------------------------------------------
# (a) unit level: split_args and the complex-instance classifier
# --------------------------------------------------------------------------

def test_split_args_ordinary_argument_list():
    """Positive control: the shape OpenCascade actually writes."""
    assert split_args("'',(#18),#32,.F.") == ["''", "(#18)", "#32", ".F."]
    assert split_args("'',5,5,((#1,#2),(#3,#4)),.T.") == [
        "''", "5", "5", "((#1,#2),(#3,#4))", ".T."]
    assert split_args("") == []


def test_split_args_survives_the_doubled_quote_escape():
    """A ' inside a Part 21 string is written '' and must not end the string.

    Upstream this inverted the string state and swallowed the rest of the
    argument list, which made ADVANCED_FACE lose its face_geometry reference.
    """
    parts = split_args(ESCAPED)
    assert parts == ["'It''s a quote'", "(#81)", "#70", ".T."], parts
    assert len(parts) == 4

    # negative control: the upstream loop on the very same text
    upstream_parts = _upstream_split_args(ESCAPED)
    assert len(upstream_parts) == 1, (
        "the upstream defect is not what this test assumes it is; "
        f"upstream returned {upstream_parts!r}")
    assert upstream_parts != parts


def test_split_args_handles_an_even_number_of_escapes():
    # OCCT doubles each quote, so a name holding '' arrives as '''' in the file
    assert split_args("'Gegenk''''oerper 1','',#7") == [
        "'Gegenk''''oerper 1'", "''", "#7"]


# One rational B-spline face, written the way OCCT writes weights != 1:
# an AND-combined (complex) instance with an empty type name.
COMPLEX_SURFACE_STEP = """ISO-10303-21;
HEADER;
ENDSEC;
DATA;
#17 = ADVANCED_FACE('',(#18),#35,.T.);
#35 = ( BOUNDED_SURFACE() B_SPLINE_SURFACE(1,1,(
    (#36,#37)
    ,(#38,#39
)),.UNSPECIFIED.,.F.,.F.,.F.) B_SPLINE_SURFACE_WITH_KNOTS((2,2),(2,2),(
    0.,1.),(0.,1.),.PIECEWISE_BEZIER_KNOTS.)
GEOMETRIC_REPRESENTATION_ITEM() RATIONAL_B_SPLINE_SURFACE((
    (1.,1.)
,(1.,2.))) REPRESENTATION_ITEM('') SURFACE() );
ENDSEC;
END-ISO-10303-21;
"""


def test_complex_instance_surface_is_bucketed_as_bspline():
    c = census_text(COMPLEX_SURFACE_STEP)
    assert c["ok"] is True
    assert c["faces"] == 1
    assert c["surfaces"]["bspline"] == 1, c["surfaces"]
    assert c["surfaces"]["other"] == 0, c["surfaces"]
    assert c["surfaces_consistent"] is True


def test_complex_instance_classifier_is_not_hardwired_to_bspline():
    """Negative controls for the classifier: a plain surface and an unknown
    one must not be reported as bspline."""
    plain = COMPLEX_SURFACE_STEP.replace(
        COMPLEX_SURFACE_STEP[COMPLEX_SURFACE_STEP.index("#35 = ("):
                             COMPLEX_SURFACE_STEP.index("SURFACE() );") + len("SURFACE() );")],
        "#35 = PLANE('',#40);")
    assert "BOUNDED_SURFACE" not in plain, "the defect was not injected"
    c = census_text(plain)
    assert c["faces"] == 1
    assert c["surfaces"]["plane"] == 1 and c["surfaces"]["bspline"] == 0, c["surfaces"]

    weird = plain.replace("PLANE('',#40)", "SOME_FUTURE_SURFACE('',#40)")
    c = census_text(weird)
    assert c["surfaces"]["other"] == 1, c["surfaces"]
    assert c["surfaces"]["plane"] == 0 and c["surfaces"]["bspline"] == 0


def test_complex_instance_bucketing_needs_the_fix():
    """Defect injection: without the complex-instance scan the face lands in
    'other'. Proves the assertion above is not true by accident."""
    original = step_census.complex_sub_names
    try:
        step_census.complex_sub_names = lambda args: []      # the upstream behaviour
        c = census_text(COMPLEX_SURFACE_STEP)
    finally:
        step_census.complex_sub_names = original
    assert c["surfaces"]["other"] == 1, (
        "the injected defect did not change the census; the complex-instance "
        f"scan is not being used: {c['surfaces']}")
    assert c["surfaces"]["bspline"] == 0


def test_census_of_a_non_step_file_reports_not_ok():
    c = census_text("hello, this is not a STEP file\n", name="junk.txt")
    assert c["ok"] is False
    assert "error" in c


# --------------------------------------------------------------------------
# (a2) reference cycles: resolve_surface follows OFFSET_SURFACE, so it needs
# the same ``seen`` guard resolve_curve always had.
# --------------------------------------------------------------------------

CYCLIC_OFFSET_STEP = """ISO-10303-21;
HEADER;
ENDSEC;
DATA;
#17 = ADVANCED_FACE('',(#18),#35,.T.);
#35 = OFFSET_SURFACE('',#36,1.,.F.);
#36 = OFFSET_SURFACE('',#35,1.,.F.);
ENDSEC;
END-ISO-10303-21;
"""


def test_a_cyclic_offset_surface_does_not_blow_the_stack():
    """A cycle in the surface chain must terminate, not recurse forever.

    ``resolve_curve`` has always carried a ``seen`` set; ``resolve_surface``
    did not, so ``#35 -> #36 -> #35`` recursed until RecursionError escaped
    ``census_path`` - which promises a report dict, and whose caller
    ``src/io/step_audit._census`` then logs a warning and silently drops the
    second witness. An unclassifiable chain is ``other``, not a crash.
    """
    # precondition: the cycle really is in the text under test
    assert "#35 = OFFSET_SURFACE('',#36" in CYCLIC_OFFSET_STEP
    assert "#36 = OFFSET_SURFACE('',#35" in CYCLIC_OFFSET_STEP
    c = census_text(CYCLIC_OFFSET_STEP)
    assert c["ok"] is True, c
    assert c["faces"] == 1
    assert c["surfaces"]["other"] == 1, c["surfaces"]
    assert c["surfaces_consistent"] is True


def test_a_plain_offset_surface_still_resolves_to_its_base_surface():
    """Positive control for the cycle guard: breaking the cycle must restore
    the resolution, so the guard cannot be 'always return other'."""
    text = CYCLIC_OFFSET_STEP.replace(
        "#36 = OFFSET_SURFACE('',#35,1.,.F.);", "#36 = PLANE('',#40);")
    assert "OFFSET_SURFACE('',#35" not in text, "the cycle was not removed"
    c = census_text(text)
    assert c["surfaces"]["plane"] == 1, c["surfaces"]
    assert c["surfaces"]["other"] == 0, c["surfaces"]


def test_the_cycle_guard_is_not_shared_between_faces():
    """Two faces on one surface entity, plus an offset onto that same entity.

    The classic way to break a recursion guard is to let the ``seen`` set
    outlive one top-level call (a mutable default argument): face 2 would then
    find #50 already visited and fall into ``other``.
    """
    shared = """ISO-10303-21;
DATA;
#1 = ADVANCED_FACE('',(#10),#50,.T.);
#2 = ADVANCED_FACE('',(#11),#50,.T.);
#3 = ADVANCED_FACE('',(#12),#51,.T.);
#50 = PLANE('',#60);
#51 = OFFSET_SURFACE('',#50,1.,.F.);
ENDSEC;
"""
    assert shared.count("#50,.T.") == 2, "both faces must share one surface"
    c = census_text(shared)
    assert c["faces"] == 3
    assert c["surfaces"]["plane"] == 3, c["surfaces"]
    assert c["surfaces"]["other"] == 0, c["surfaces"]


def test_census_path_reports_a_parser_crash_instead_of_raising(tmp_path):
    """``census_path`` promises a report dict for anything it is handed.

    Defect injection: make the parser raise. Without the envelope the
    exception escapes and ``step_audit`` loses ``census_faces`` entirely -
    the audit then passes with no second witness at all.
    """
    p = tmp_path / "x.step"
    p.write_text(COMPLEX_SURFACE_STEP, encoding="utf-8")
    assert census_path(str(p))["ok"] is True, "precondition: this file censuses"

    original = step_census.parse_entities
    try:
        def boom(text):
            raise RecursionError("maximum recursion depth exceeded")
        step_census.parse_entities = boom
        assert step_census.parse_entities is not original, "the defect was not injected"
        c = census_path(str(p))
    finally:
        step_census.parse_entities = original
    assert c["ok"] is False, c
    assert "recursion" in c.get("error", "").lower(), c
    assert census_path(str(p))["ok"] is True, "the defect was not restored"


# --------------------------------------------------------------------------
# (b) real OpenCascade round trips
# --------------------------------------------------------------------------

def _export(shape, path, **kw):
    from src.io.exporters import export_step
    export_step(shape, path, **kw)
    assert os.path.getsize(path) > 0
    return path


@needs_ocp
def test_census_of_a_box(tmp_path):
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    step = _export(BRepPrimAPI_MakeBox(10.0, 20.0, 30.0).Shape(),
                   str(tmp_path / "box.step"), product_name="box")
    c = census_path(step)
    assert c["ok"] is True
    assert c["faces"] == 6
    assert c["surfaces"]["plane"] == 6
    assert c["edges"] == 12
    assert c["curves"]["line"] == 12
    assert c["surfaces_consistent"] is True
    assert c["curves_consistent"] is True
    assert c["cylinder_radii"] == []


@needs_ocp
def test_census_of_a_cylinder(tmp_path):
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
    step = _export(BRepPrimAPI_MakeCylinder(5.0, 10.0).Shape(),
                   str(tmp_path / "cyl.step"))
    c = census_path(step)
    assert c["faces"] == 3
    assert c["surfaces"]["cylinder"] == 1
    assert c["surfaces"]["plane"] == 2
    assert c["cylinder_radii"] == pytest.approx([5.0])
    assert c["curves"]["circle"] == 2
    # a second geometry pins ``edges``: the box's 12 is the only other value
    # asserted anywhere, and a hard-wired 12 would pass that test alone
    assert c["edges"] == 3, c
    assert c["curves"]["line"] == 1, c["curves"]
    assert c["surfaces_consistent"] is True
    assert c["curves_consistent"] is True


def _bezier_face(weights=None):
    """A 6x6 Bezier patch built exactly like src/nurbs/converter.py:168-177."""
    from OCP.TColgp import TColgp_Array2OfPnt
    from OCP.gp import gp_Pnt
    from OCP.Geom import Geom_BezierSurface
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    pts = TColgp_Array2OfPnt(1, 6, 1, 6)
    for i in range(6):
        for j in range(6):
            pts.SetValue(i + 1, j + 1,
                         gp_Pnt(float(i), float(j), float(0.3 * i * j)))
    if weights is None:
        surf = Geom_BezierSurface(pts)
    else:
        surf = Geom_BezierSurface(pts, weights)
    mk = BRepBuilderAPI_MakeFace(surf, 1e-6)
    assert mk.IsDone()
    return mk.Face()


@needs_ocp
def test_census_of_a_bezier_patch(tmp_path):
    step = _export(_bezier_face(), str(tmp_path / "bez.step"))
    c = census_path(step)
    assert c["faces"] == 1
    assert c["surfaces"]["bspline"] == 1, c["surfaces"]
    assert c["surfaces"]["other"] == 0
    assert c["surfaces_consistent"] is True


@needs_ocp
def test_census_of_a_rational_bezier_patch_written_as_a_complex_instance(tmp_path):
    """Weights != 1 make OCCT write ``#n = ( BOUNDED_SURFACE() ... )``.

    This is the file-level form of the complex-instance bug: upstream put the
    face in ``other`` because the entity type name is empty.
    """
    from OCP.TColStd import TColStd_Array2OfReal
    w = TColStd_Array2OfReal(1, 6, 1, 6)
    for i in range(1, 7):
        for j in range(1, 7):
            w.SetValue(i, j, 2.0 if (i == 3 and j == 3) else 1.0)
    step = _export(_bezier_face(w), str(tmp_path / "rational.step"))
    text = Path(step).read_text(encoding="utf-8", errors="replace")
    assert "BOUNDED_SURFACE()" in text, (
        "OCCT did not write a complex instance; this test is not exercising "
        "the bug it was written for")
    c = census_path(step)
    assert c["faces"] == 1
    assert c["surfaces"]["bspline"] == 1, c["surfaces"]
    assert c["surfaces"]["other"] == 0, c["surfaces"]


@needs_ocp
def test_census_path_accepts_a_plain_string(tmp_path):
    """step_audit calls ``census_path(step_path)`` with a str; upstream took
    only a Path and blew up on ``str.read_text``."""
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    step = _export(BRepPrimAPI_MakeBox(1.0, 1.0, 1.0).Shape(),
                   str(tmp_path / "unit.step"))
    from_str = census_path(step)
    from_path = census_path(Path(step))
    assert from_str["faces"] == 6 and from_path["faces"] == 6
    assert from_str == from_path


# --------------------------------------------------------------------------
# (c) mutation: the census counts the text, not a constant
# --------------------------------------------------------------------------

@needs_ocp
def test_deleting_one_advanced_face_is_seen_by_census_and_audit(tmp_path):
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from src.io.step_audit import measure_step, verdict
    good = _export(BRepPrimAPI_MakeBox(10.0, 20.0, 30.0).Shape(),
                   str(tmp_path / "box.step"), product_name="box")
    assert census_path(good)["faces"] == 6, "the unmutated file must census 6"

    lines = Path(good).read_text(encoding="utf-8").splitlines(keepends=True)
    idx = [i for i, ln in enumerate(lines) if "ADVANCED_FACE" in ln]
    assert len(idx) == 6, f"expected 6 ADVANCED_FACE lines, found {len(idx)}"
    bad = tmp_path / "box_missing_face.step"
    bad.write_text("".join(lines[:idx[0]] + lines[idx[0] + 1:]), encoding="utf-8")
    assert "ADVANCED_FACE" in lines[idx[0]], "the defect was not injected"
    assert bad.read_text(encoding="utf-8").count("ADVANCED_FACE") == 5

    c = census_path(str(bad))
    assert c["faces"] == 5, f"census still reports {c['faces']} faces"
    assert c["surfaces"]["plane"] == 5
    assert c["surfaces_consistent"] is True

    m = measure_step(str(bad))
    assert m["census_faces"] == 5, (
        f"step_audit did not pick the census up: census_faces={m['census_faces']}")
    code, reasons = verdict(m)
    census_flagged = any("census" in r for r in reasons)
    ocp_dropped = (m["faces"] == 5 and (m["free_edges"] or 0) > 0)
    detail = (f"faces={m['faces']} census_faces={m['census_faces']} "
              f"solids={m['solids']} free_edges={m['free_edges']} "
              f"closed={m['closed_measured']} code={code} reasons={reasons}")
    assert census_flagged or ocp_dropped, "the audit missed the deleted face: " + detail
    # measured on OCP 7.9 / OCCT 7.9: the reader drops the face too, so the
    # census AGREES with the kernel (5 == 5) and the open shell is what fails.
    assert not census_flagged, "unexpected census/OCP disagreement: " + detail
    assert m["faces"] == 5 and m["solids"] == 0 and m["free_edges"] == 4, detail
    assert code == 2, detail
    assert any("free" in r for r in reasons), detail


# --------------------------------------------------------------------------
# (d) string-escape regression on a file OpenCascade wrote
# --------------------------------------------------------------------------

@needs_ocp
def test_escaped_quote_in_a_face_name_does_not_lose_the_surface(tmp_path):
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    good = _export(BRepPrimAPI_MakeBox(10.0, 20.0, 30.0).Shape(),
                   str(tmp_path / "box.step"), product_name="box")
    text = Path(good).read_text(encoding="utf-8")
    assert "ADVANCED_FACE(''," in text, "unexpected ADVANCED_FACE spelling"
    # one face gets a name holding a single apostrophe, escaped as '' -> the
    # odd number of quotes is what flips the upstream splitter's string state
    patched = text.replace("ADVANCED_FACE('',", "ADVANCED_FACE('Gegenk''oerper',", 1)
    assert patched != text, "the defect was not injected"
    assert patched.count("Gegenk''oerper") == 1
    step = tmp_path / "named.step"
    step.write_text(patched, encoding="utf-8")

    c = census_path(str(step))
    assert c["faces"] == 6, c
    assert c["surfaces"]["plane"] == 6, c["surfaces"]
    assert c["surfaces_consistent"] is True

    # negative control: the same file censused with the upstream splitter
    original = step_census.split_args
    try:
        step_census.split_args = _upstream_split_args
        broken = census_text(patched)
    finally:
        step_census.split_args = original
    assert broken["faces"] == 6, "faces are counted by type name, not by args"
    assert sum(broken["surfaces"].values()) == 5, (
        "the upstream splitter was expected to drop exactly one surface, got "
        f"{broken['surfaces']}")
    assert broken["surfaces_consistent"] is False, (
        "surfaces_consistent did not notice the dropped surface")


@needs_ocp
def test_escaped_quote_in_an_edge_name_is_caught_by_curves_consistent(tmp_path):
    """``curves_consistent`` is the EDGE_CURVE half of the same invariant.

    Without this control ``curves_consistent`` could be hard-wired to True and
    every other test in the file would still pass.
    """
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    good = _export(BRepPrimAPI_MakeBox(10.0, 20.0, 30.0).Shape(),
                   str(tmp_path / "box.step"), product_name="box")
    text = Path(good).read_text(encoding="utf-8")
    assert "EDGE_CURVE(''," in text, "unexpected EDGE_CURVE spelling"
    patched = text.replace("EDGE_CURVE('',", "EDGE_CURVE('Kante''1',", 1)
    assert patched != text, "the defect was not injected"
    assert patched.count("Kante''1") == 1

    c = census_text(patched)
    assert c["edges"] == 12, c
    assert c["curves"]["line"] == 12, c["curves"]
    assert c["curves_consistent"] is True

    # negative control: the upstream splitter drops exactly that one curve
    original = step_census.split_args
    try:
        step_census.split_args = _upstream_split_args
        broken = census_text(patched)
    finally:
        step_census.split_args = original
    assert broken["edges"] == 12, "edges are counted by type name, not by args"
    assert sum(broken["curves"].values()) == 11, (
        "the upstream splitter was expected to drop exactly one curve, got "
        f"{broken['curves']}")
    assert broken["curves_consistent"] is False, (
        "curves_consistent did not notice the dropped curve")


@needs_ocp
def test_escaped_quote_in_the_product_name_round_trips(tmp_path):
    """The realistic route: a product name with an apostrophe. OCCT doubles
    every quote on write, so the file holds '' inside a string."""
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    step = _export(BRepPrimAPI_MakeBox(10.0, 20.0, 30.0).Shape(),
                   str(tmp_path / "gk.step"), product_name="Gegenk''oerper")
    text = Path(step).read_text(encoding="utf-8")
    assert "''" in text.split("DATA;", 1)[1], "no escaped quote landed in the file"
    c = census_path(step)
    assert c["faces"] == 6
    assert c["surfaces_consistent"] is True

    # and the PRODUCT argument list itself must still split: name, name,
    # description, context list. OCCT doubles each quote on write, so the
    # name arrives as 'Gegenk''''oerper 1'.
    product = next(ln for ln in text.splitlines() if ln.startswith("#")
                   and " = PRODUCT(" in ln)
    args = product[product.index("PRODUCT(") + len("PRODUCT("):product.rindex(")")]
    parts = split_args(args)
    assert len(parts) == 4, f"PRODUCT args split into {parts!r}"
    assert parts[0].startswith("'Gegenk") and parts[0].endswith("'"), parts[0]
    assert parts[3].startswith("(#")


# --------------------------------------------------------------------------
# (e) CLI
# --------------------------------------------------------------------------

@needs_ocp
def test_cli_prints_json_on_stdout(tmp_path):
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    step = _export(BRepPrimAPI_MakeBox(10.0, 20.0, 30.0).Shape(),
                   str(tmp_path / "box.step"), product_name="box")
    proc = subprocess.run(
        [sys.executable, "-m", "src.io.step_census", step],
        cwd=REPO_ROOT, capture_output=True, text=True, encoding="utf-8")
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["faces"] == 6, payload
    assert payload["ok"] is True
    assert payload["surfaces"]["plane"] == 6


def test_cli_reports_a_missing_file_without_a_traceback(tmp_path):
    missing = str(tmp_path / "nope.step")
    proc = subprocess.run(
        [sys.executable, "-m", "src.io.step_census", missing],
        cwd=REPO_ROOT, capture_output=True, text=True, encoding="utf-8")
    assert "Traceback" not in proc.stderr, proc.stderr
    assert proc.returncode == 1
    payload = json.loads(proc.stdout)
    assert payload["ok"] is False


# A one-face STEP built by hand, so the CLI tests need no OCP.
MINIMAL_STEP = """ISO-10303-21;
HEADER;
ENDSEC;
DATA;
#17 = ADVANCED_FACE('',(#18),#35,.T.);
#35 = PLANE('',#40);
ENDSEC;
END-ISO-10303-21;
"""


def _run_cli(*argv):
    return subprocess.run([sys.executable, "-m", "src.io.step_census", *argv],
                          cwd=REPO_ROOT, capture_output=True, text=True,
                          encoding="utf-8")


def test_cli_survives_a_cyclic_offset_surface_without_a_traceback(tmp_path):
    step = tmp_path / "cyclic.step"
    step.write_text(CYCLIC_OFFSET_STEP, encoding="utf-8")
    proc = _run_cli(str(step))
    assert "Traceback" not in proc.stderr, proc.stderr
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["faces"] == 1


def test_cli_expect_accepts_the_truth_and_rejects_a_wrong_number(tmp_path):
    """``--expect`` is the gate the census exists to feed; it had no test."""
    step = tmp_path / "one.step"
    step.write_text(MINIMAL_STEP, encoding="utf-8")

    good = tmp_path / "good.json"
    good.write_text(json.dumps({"faces": 1, "surfaces": {"plane": 1}}),
                    encoding="utf-8")
    ok = _run_cli(str(step), "--expect", str(good))
    assert ok.returncode == 0, ok.stderr
    assert "EXPECT OK" in ok.stderr, ok.stderr

    # negative control: the same file against a ground truth it does not meet
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"faces": 1, "surfaces": {"plane": 99}}),
                   encoding="utf-8")
    assert json.loads(bad.read_text(encoding="utf-8"))["surfaces"]["plane"] == 99, \
        "the defect was not injected"
    fail = _run_cli(str(step), "--expect", str(bad))
    assert fail.returncode == 1, fail.stderr
    assert "surfaces.plane: got 1 want 99" in fail.stderr, fail.stderr


def test_cli_expect_does_not_traceback_on_a_missing_or_invalid_json(tmp_path):
    step = tmp_path / "one.step"
    step.write_text(MINIMAL_STEP, encoding="utf-8")
    for name, content in (("gone.json", None), ("broken.json", "{not json")):
        truth = tmp_path / name
        if content is not None:
            truth.write_text(content, encoding="utf-8")
        assert truth.exists() is (content is not None), "wrong precondition"
        proc = _run_cli(str(step), "--expect", str(truth))
        assert "Traceback" not in proc.stderr, (name, proc.stderr)
        assert proc.returncode == 2, (name, proc.returncode, proc.stderr)
        assert "cannot read --expect" in proc.stderr, (name, proc.stderr)


def test_cli_diff_table_compares_two_files(tmp_path):
    """The multi-file mode and the two consistency rows added while vendoring."""
    a = tmp_path / "a.step"
    a.write_text(MINIMAL_STEP, encoding="utf-8")
    b = tmp_path / "b.step"
    b.write_text(MINIMAL_STEP.replace("PLANE('',#40)",
                                      "CYLINDRICAL_SURFACE('',#40,7.)"),
                 encoding="utf-8")
    assert "CYLINDRICAL_SURFACE" in b.read_text(encoding="utf-8"), \
        "the difference was not injected"

    proc = _run_cli(str(a), str(b))
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert [f["faces"] for f in payload["files"]] == [1, 1], payload

    table = proc.stderr
    assert "surfaces_consistent" in table and "curves_consistent" in table, table

    def cells(prefix):
        line = next(ln for ln in table.splitlines() if ln.startswith(prefix))
        return [c.strip() for c in line.split("|")[2:4]]

    # the table must show the two files DIFFERING, not one constant column
    assert cells("| surface.plane") == ["1", "0"], table
    assert cells("| surface.cylinder") == ["0", "1"], table
    assert cells("| cylinder_radii (n)") == ["0", "1"], table
