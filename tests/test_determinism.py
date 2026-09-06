"""Determinism of the STL -> STEP pipeline, and of the tools that check it.

Why this file exists
--------------------
``src/nurbs/g3_fitter.py`` used to decorate two arithmetic kernels with
``@njit(fastmath=True)`` behind a silent ``try: from numba import njit /
except ImportError: <no-op decorator>``.  numba is not in
``requirements.txt`` but *is* installed on the maintainer's machine, so the
documented install and the maintainer's runs executed different arithmetic:
``fastmath`` reassociates the Coons blend, the two kernels drift by ~1e-13,
and the ill-conditioned ``lsqr`` solve downstream amplifies that to ~0.35 mm
on interior control points.  Two people running the same command on the same
STL got STEP files that differ in the third decimal of the reported
deviation.

So the fitter is now plain numpy, and this file holds the tests that keep it
that way:

* a canonical comparator for STEP files (``src/io/step_canonical.py``) that
  is itself mutation-tested, so "the two files are identical" is a claim that
  can fail;
* the retopology stage (``QuadWrapper``) must be bit-reproducible;
* the whole ``src.convert.convert`` pipeline must produce byte-identical
  STEP output twice in a row;
* a guard that fails if numba is reintroduced.

Test discipline: every behaviour below has a positive control *and* a
defect-injected negative control, and preconditions ("the defect really was
injected") are asserted before the assertion that matters.  A comparator that
always says "equal" and a comparator that always says "different" both fail
this file.
"""
import inspect
import os
import re
import sys
import tempfile
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FITTER_PATH = os.path.join(REPO, "src", "nurbs", "g3_fitter.py")

from src.io.step_canonical import (  # noqa: E402
    canonical_step_digest,
    canonical_step_text,
    extract_data_section,
    step_files_equal,
)

try:
    import OCP  # noqa: F401
    HAS_OCP = True
except ImportError:
    HAS_OCP = False

needs_ocp = pytest.mark.skipif(not HAS_OCP, reason="cadquery-ocp not installed")


# --------------------------------------------------------------------------
# STEP text fixtures
# --------------------------------------------------------------------------

_HEADER_TMPL = """ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('Open CASCADE Model'),'2;1');
FILE_NAME('Open CASCADE Shape Model','{ts}',('Author'),('Open CASCADE'),
  'Open CASCADE STEP processor 7.9','1','Unknown');
FILE_SCHEMA(('AUTOMOTIVE_DESIGN {{ 1 0 10303 214 -1 1 5 4 }}'));
ENDSEC;
"""

_DATA_TMPL = """DATA;
#1 = APPLICATION_PROTOCOL_DEFINITION('international standard',
  'automotive_design',2000,#2);
#2 = APPLICATION_CONTEXT('core data for automotive mechanical design');
#10 = CARTESIAN_POINT('',({x},2.0000000000000000,3.0000000000000000));
#11 = DIRECTION('',(0.0000000000000000,0.0000000000000000,1.0000000000000000));
ENDSEC;
END-ISO-10303-21;
"""


def _step_text(ts="2026-09-05T09:15:00", x="1.0000000000000000",
               header_extra="", eol="\n"):
    """A small but structurally real STEP file.

    ``ts`` is the FILE_NAME timestamp (HEADER, run-variant), ``x`` the first
    coordinate of #10 (DATA, must never be normalised away).
    """
    header = _HEADER_TMPL.format(ts=ts)
    if header_extra:
        # inserted just before the HEADER's ENDSEC;
        header = header.replace("ENDSEC;\n", header_extra + "ENDSEC;\n", 1)
    text = header + _DATA_TMPL.format(x=x)
    if eol != "\n":
        text = text.replace("\n", eol)
    return text


def _renumbered(text, offset=1000):
    """Shift every ``#N`` entity id by ``offset`` (a topology-visible change)."""
    return re.sub(r"#(\d+)", lambda m: "#%d" % (int(m.group(1)) + offset), text)


def _write(path, text):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)
    return path


# --------------------------------------------------------------------------
# (a) MUTATION CONTROL on the canonicaliser
# --------------------------------------------------------------------------

def test_canonicaliser_reports_a_changed_coordinate_digit_as_different():
    """Positive control: a one-digit change in DATA must survive canonicalisation.

    This is the test a constant/over-normalising canonicaliser fails.
    """
    a = _step_text(x="1.0000000000000000")
    b = _step_text(x="1.0000000000000001")
    # precondition: the defect really is a single digit in the DATA section
    assert a != b, "fixture bug: the two texts are identical"
    assert sum(1 for ca, cb in zip(a, b) if ca != cb) == 1, \
        "fixture bug: expected exactly one differing character"

    assert canonical_step_text(a) != canonical_step_text(b)

    with tempfile.TemporaryDirectory() as d:
        pa = _write(os.path.join(d, "a.step"), a)
        pb = _write(os.path.join(d, "b.step"), b)
        equal, diff = step_files_equal(pa, pb)
        assert equal is False
        assert diff, "reported different but produced no diff lines"
        assert canonical_step_digest(pa) != canonical_step_digest(pb)


def test_canonicaliser_reports_a_changed_file_name_timestamp_as_equal():
    """Negative control on the identity function: the wall-clock timestamp is
    the one HEADER field two runs of the same binary are allowed to differ in."""
    a = _step_text(ts="2026-09-05T09:15:00")
    b = _step_text(ts="2031-01-31T23:59:59")
    # precondition: the variation really was injected, and only there
    assert a != b, "fixture bug: timestamps are identical"
    assert a.replace("2026-09-05T09:15:00", "2031-01-31T23:59:59") == b, \
        "fixture bug: the two texts differ somewhere other than the timestamp"

    assert canonical_step_text(a) == canonical_step_text(b)

    with tempfile.TemporaryDirectory() as d:
        pa = _write(os.path.join(d, "a.step"), a)
        pb = _write(os.path.join(d, "b.step"), b)
        equal, diff = step_files_equal(pa, pb)
        assert equal is True, "\n".join(diff)
        assert diff == []
        assert canonical_step_digest(pa) == canonical_step_digest(pb)


def test_canonicaliser_keeps_the_header_apart_from_the_timestamp():
    """The HEADER is kept (schema/preprocessor changes are real changes)."""
    canon = canonical_step_text(_step_text())
    assert "FILE_SCHEMA" in canon
    assert "AUTOMOTIVE_DESIGN" in canon
    assert "FILE_NAME" in canon
    # ...but the timestamp itself is gone
    assert "2026-09-05T09:15:00" not in canon

    # and a different SCHEMA is still a difference
    other = _step_text().replace("AUTOMOTIVE_DESIGN", "CONFIG_CONTROL_DESIGN")
    assert canonical_step_text(other) != canon


def test_canonicaliser_never_renumbers_entity_ids():
    """Renumbering can mask a topology change, so shifted ids must diff."""
    a = _step_text()
    b = _renumbered(a, offset=1000)
    # precondition: the ids really were shifted and nothing else moved
    assert "#10 = CARTESIAN_POINT" in a and "#1010 = CARTESIAN_POINT" in b
    assert a != b

    assert canonical_step_text(a) != canonical_step_text(b)
    assert "#10 = CARTESIAN_POINT" in canonical_step_text(a), \
        "entity ids were rewritten by the canonicaliser"


def test_canonicaliser_normalises_line_endings():
    crlf = _step_text(eol="\r\n")
    lf = _step_text(eol="\n")
    # precondition: the fixture really is CRLF
    assert "\r\n" in crlf and "\r" not in lf
    assert crlf != lf
    assert canonical_step_text(crlf) == canonical_step_text(lf)
    assert "\r" not in canonical_step_text(crlf)


def test_data_section_slicing_is_token_aware():
    """``DATA;`` inside a string literal, and ``FOODATA;`` as a suffix, must
    not be mistaken for the start of the DATA section.

    The decoy below is a HEADER string that spans lines and contains a line
    which *is* exactly ``DATA;`` followed by a fake entity and a fake
    ``ENDSEC;``.  A slicer that only anchors on line starts slices the decoy
    and never sees the real geometry.
    """
    decoy = (
        "FILE_DESCRIPTION(('a description that spans lines\n"
        "DATA;\n"
        "#99 = FAKE_POINT((9.0,9.0,9.0));\n"
        "ENDSEC;\n"
        "end of description'),'2;1');\n"
        "FOODATA;\n"
    )
    text = _step_text(header_extra=decoy)

    # preconditions: the decoys really are in the text, they come first, and
    # the decoy block really is inside one string literal
    assert "FAKE_POINT" in text
    assert text.count("'a description that spans lines") == 1
    assert "end of description'" in text
    assert "\nFOODATA;\n" in text
    first_data = text.find("\nDATA;\n")
    real_data = text.rfind("\nDATA;\n")
    assert first_data != -1 and first_data < real_data, \
        "fixture bug: the decoy DATA; line is not before the real one"

    data = extract_data_section(text)
    assert data.startswith("DATA;")
    assert "CARTESIAN_POINT" in data, "sliced the decoy, not the real section"
    assert "FAKE_POINT" not in data
    assert data.rstrip().endswith("ENDSEC;")

    # the canonical form still contains the real geometry
    canon = canonical_step_text(text)
    assert "#10 = CARTESIAN_POINT" in canon


def test_section_slicing_survives_a_non_ascii_header():
    """Regression: uppercasing is not length-preserving in Unicode.

    A German 'ss' uppercases to two characters, so a slicer that searches in
    ``text.upper()`` and then indexes back into ``text`` walks off by one per
    such character. With ten of them ahead of the keyword, the real ``DATA;``
    was reported missing on a perfectly well-formed file, i.e. the comparator
    raised instead of comparing.
    """
    name = "Gehaeuse Fussplatte " + "ß" * 10
    text = _step_text(header_extra="FILE_PART('%s');\n" % name)

    # precondition: the fixture really does defeat a .upper()-based search
    assert len(text.upper()) > len(text), \
        "fixture bug: this text uppercases to the same length"
    assert text.index(name) < text.index("\nDATA;"), \
        "fixture bug: the non-ASCII run must sit ahead of the DATA; keyword"

    data = extract_data_section(text)
    assert data.startswith("DATA;") and "CARTESIAN_POINT" in data
    canon = canonical_step_text(text)
    assert "#10 = CARTESIAN_POINT" in canon
    assert name in canon, "the HEADER text was mangled"

    # defect-injected control: the slicer must still be discriminating on
    # this file, not just 'not raising'
    other = _step_text(header_extra="FILE_PART('%s');\n" % name,
                       x="1.0000000000000001")
    assert canonical_step_text(other) != canon


def test_extract_data_section_rejects_a_file_without_one():
    """Negative control: a comparator that returns '' for a broken file would
    call two broken files 'identical'."""
    broken = "ISO-10303-21;\nHEADER;\nENDSEC;\nEND-ISO-10303-21;\n"
    assert "DATA;" not in broken  # precondition
    with pytest.raises(ValueError):
        extract_data_section(broken)
    with pytest.raises(ValueError):
        canonical_step_text(broken)


def test_unterminated_data_section_is_rejected():
    truncated = _step_text()
    cut = truncated.rfind("ENDSEC;")
    truncated = truncated[:cut]
    assert "DATA;" in truncated  # precondition: DATA; is there, ENDSEC; is not
    assert truncated.count("ENDSEC;") == 1, "only the HEADER's ENDSEC; may remain"
    with pytest.raises(ValueError):
        extract_data_section(truncated)


# --------------------------------------------------------------------------
# (b) QuadWrapper determinism
# --------------------------------------------------------------------------

def _icosphere_cage(target_faces=120, radius=10.0, subdivisions=2):
    import trimesh
    from src.core.halfedge_mesh import HalfEdgeMesh
    from src.reverse_engineering.quad_wrap import QuadWrapper

    tm = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    mesh = HalfEdgeMesh.from_trimesh(tm)
    wrapper = QuadWrapper(target_face_count=target_faces, smoothing_weight=0.5)
    return wrapper.wrap(mesh).to_arrays()


def test_quadwrapper_is_bit_reproducible():
    a = _icosphere_cage()
    b = _icosphere_cage()

    # precondition: the cage is non-trivial, so array_equal is not comparing
    # two empty arrays (which would make this test vacuous)
    assert a["vertices"].shape[0] > 20, f"cage too small: {a['vertices'].shape}"
    assert len(a["faces"]) > 20
    assert a["vertices"].shape == b["vertices"].shape

    assert np.array_equal(a["vertices"], b["vertices"]), (
        "QuadWrapper is not deterministic: max |dv| = "
        f"{np.abs(a['vertices'] - b['vertices']).max()!r}"
    )
    assert a["faces"] == b["faces"]


def test_quadwrapper_comparison_can_fail():
    """Defect-injected control for the test above: a different reference mesh
    must produce a different cage, proving np.array_equal is discriminating
    here and not comparing two degenerate arrays."""
    a = _icosphere_cage(radius=10.0)
    c = _icosphere_cage(radius=7.0)
    assert a["vertices"].shape[0] > 20 and c["vertices"].shape[0] > 20
    if a["vertices"].shape == c["vertices"].shape:
        assert not np.array_equal(a["vertices"], c["vertices"])
    # a 7 mm sphere cannot have the same cage vertices as a 10 mm one
    assert abs(np.abs(a["vertices"]).max() - np.abs(c["vertices"]).max()) > 1.0


# --------------------------------------------------------------------------
# (c) End-to-end: two runs of convert() must write the same STEP
# --------------------------------------------------------------------------

_VOLATILE_RESULT_KEYS = ("seconds", "output", "input")

# OpenCascade's STEP writer appends a running transfer index to the PRODUCT
# name ("sphere 1", "sphere 2", ...). The counter is a static inside the
# writer, so it restarts at 1 in every new process but keeps counting when
# convert() is called twice inside ONE process. It is the only line that
# differs between two in-process runs; everything geometric is bit-identical.
_PRODUCT_LINE = re.compile(r"^#\d+ = PRODUCT\('[^']*','[^']*','',\(#\d+\)\);$")


def _offending(differing):
    """Differing lines that are NOT the permitted OCC PRODUCT counter.

    Both sides must look like a PRODUCT line: excusing a pair because the
    *left* one happens to be a PRODUCT line would let a drifted right-hand
    line through at the same line index.
    """
    return [t for t in differing
            if not (_PRODUCT_LINE.match(t[1]) and _PRODUCT_LINE.match(t[2]))]


def _stable_result(result):
    out = {k: v for k, v in result.items() if k not in _VOLATILE_RESULT_KEYS}
    audit = out.get("audit")
    if isinstance(audit, dict):
        audit = dict(audit)
        audit.pop("step_path", None)
        out["audit"] = audit
    return out


def _make_sphere_stl(path, subdivisions=3, radius=10.0):
    import trimesh
    trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius).export(path)
    return path


def _run_cli(stl, step, timeout=900):
    """One `python -m src.convert` process; returns (exit_code, RESULT dict)."""
    import json
    import subprocess
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(
        [sys.executable, "-m", "src.convert", stl, step,
         "--target-faces", "120", "--quiet"],
        cwd=REPO, capture_output=True, text=True, encoding="utf-8",
        errors="replace", env=env, timeout=timeout)
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    assert lines and lines[-1].startswith("RESULT "), \
        f"no RESULT line; stderr tail:\n{proc.stderr[-2000:]}"
    return proc.returncode, json.loads(lines[-1][len("RESULT "):])


@needs_ocp
def test_convert_is_byte_reproducible_across_processes(capsys):
    """The real-world claim: two invocations of the CLI on the same STL write
    the SAME STEP, byte for byte in canonical form (only the FILE_NAME
    wall-clock timestamp is allowed to move). No normalisation beyond that."""
    with tempfile.TemporaryDirectory() as d:
        stl = _make_sphere_stl(os.path.join(d, "sphere.stl"))
        # same basename in two directories: the output name feeds the PRODUCT
        # name, so it must not be part of what we vary
        os.makedirs(os.path.join(d, "r1"))
        os.makedirs(os.path.join(d, "r2"))
        out1 = os.path.join(d, "r1", "sphere.step")
        out2 = os.path.join(d, "r2", "sphere.step")

        t0 = time.time()
        code1, res1 = _run_cli(stl, out1)
        t1 = time.time()
        code2, res2 = _run_cli(stl, out2)
        t2 = time.time()

        # preconditions: both runs actually produced a real STEP file
        assert code1 in (0, 2), res1.get("error")
        assert code2 in (0, 2), res2.get("error")
        assert os.path.getsize(out1) > 100_000
        assert os.path.getsize(out2) > 100_000

        with capsys.disabled():
            print(f"\n[determinism] CLI process 1: {t1 - t0:.2f} s, "
                  f"process 2: {t2 - t1:.2f} s, "
                  f"{os.path.getsize(out1) / 1e6:.3f} MB each")

        equal, diff = step_files_equal(out1, out2, max_diff_lines=40)
        if not equal:
            pytest.fail("\n".join(
                ["convert() is not reproducible across processes",
                 f"  run1 sha256={canonical_step_digest(out1)}",
                 f"  run2 sha256={canonical_step_digest(out2)}"] + diff))

        assert _stable_result(res1) == _stable_result(res2)


@needs_ocp
def test_convert_twice_in_one_process_differs_only_in_the_occ_product_counter(capsys):
    """convert() called twice in ONE process, as a library.

    Everything the pipeline computes must be identical; the single permitted
    difference is OpenCascade's PRODUCT transfer index, which is a static in
    the STEP writer. Asserting that *only* that line moves is what makes this
    a test rather than an excuse: any drift in a CARTESIAN_POINT, a knot
    vector, or the entity order fails it.
    """
    from src.convert import convert

    with tempfile.TemporaryDirectory() as d:
        stl = _make_sphere_stl(os.path.join(d, "sphere.stl"))
        os.makedirs(os.path.join(d, "a"))
        os.makedirs(os.path.join(d, "b"))
        out1 = os.path.join(d, "a", "sphere.step")
        out2 = os.path.join(d, "b", "sphere.step")

        t0 = time.time()
        code1, res1 = convert(stl, out1, target_faces=120, quiet=True)
        t1 = time.time()
        code2, res2 = convert(stl, out2, target_faces=120, quiet=True)
        t2 = time.time()

        assert code1 in (0, 2), res1.get("error")
        assert code2 in (0, 2), res2.get("error")
        with capsys.disabled():
            print(f"[determinism] in-process run 1: {t1 - t0:.2f} s, "
                  f"run 2: {t2 - t1:.2f} s")

        a = canonical_step_text(open(out1, encoding="utf-8").read()).splitlines()
        b = canonical_step_text(open(out2, encoding="utf-8").read()).splitlines()
        assert len(a) == len(b), (
            f"line counts differ: {len(a)} vs {len(b)} - that is a real "
            "geometry/topology drift, not the PRODUCT counter")

        differing = [(i, x, y) for i, (x, y) in enumerate(zip(a, b)) if x != y]
        offending = _offending(differing)
        assert offending == [], (
            "convert() drifted in-process beyond the OCC PRODUCT counter:\n"
            + "\n".join(f"  line {i}:\n    -{x}\n    +{y}"
                        for i, x, y in offending[:10]))
        # precondition on the precondition: the PRODUCT line really is there,
        # so `offending == []` is not passing because nothing was compared
        assert any(_PRODUCT_LINE.match(ln) for ln in a), \
            "no PRODUCT line in the canonical text - the filter is vacuous"
        assert len(differing) <= 1, differing[:5]

        assert _stable_result(res1) == _stable_result(res2)


@needs_ocp
def test_reproducibility_check_fails_on_a_fastmath_scale_kernel_drift(monkeypatch,
                                                                     capsys):
    """Defect injection at exactly the place numba used to sit.

    ``@njit(fastmath=True)`` reassociates the arithmetic in the two kernels,
    which moves their results by ~1e-13 relative. Perturb the edge kernel by
    that much on the second run only: the ill-conditioned ``lsqr`` solve in
    ``fit_surface`` amplifies it, and the two STEP files must come out
    different. If this test passes trivially, the reproducibility tests above
    are measuring nothing.
    """
    import src.nurbs.g3_fitter as g3
    from src.convert import convert

    original = g3._numba_compute_edge_control_points
    state = {"calls": 0, "run": 1}

    def drifting(*args, **kwargs):
        out = original(*args, **kwargs)
        state["calls"] += 1
        if state["run"] == 2:
            out[1:5] += 1e-13 * np.abs(out[1:5])
        return out

    monkeypatch.setattr(g3, "_numba_compute_edge_control_points", drifting)

    with tempfile.TemporaryDirectory() as d:
        stl = _make_sphere_stl(os.path.join(d, "sphere.stl"))
        os.makedirs(os.path.join(d, "a"))
        os.makedirs(os.path.join(d, "b"))
        outs = []
        for run, sub in ((1, "a"), (2, "b")):
            state["run"] = run
            out = os.path.join(d, sub, "sphere.step")
            code, res = convert(stl, out, target_faces=120, quiet=True)
            assert code in (0, 2), res.get("error")
            outs.append(out)

        # precondition: the patched kernel really was the one in use
        assert state["calls"] > 100, f"kernel called only {state['calls']} times"

        a = canonical_step_text(open(outs[0], encoding="utf-8").read()).splitlines()
        b = canonical_step_text(open(outs[1], encoding="utf-8").read()).splitlines()
        differing = [(i, x, y) for i, (x, y) in enumerate(zip(a, b)) if x != y]
        offending = _offending(differing)
        with capsys.disabled():
            print(f"[determinism] injected 1e-13 edge drift -> "
                  f"{len(offending)} changed geometry lines")
        assert offending, (
            "a 1e-13 drift in the edge kernel produced an identical STEP - "
            "the reproducibility check is not sensitive to the fitter at all")


@needs_ocp
def test_step_digest_notices_an_injected_coordinate_change():
    """Defect-injected control for the two tests above: prove the digest of a
    REAL exported STEP moves when one coordinate digit does.  Without this, a
    canonicaliser that collapsed every real file to the same text would make
    the reproducibility tests pass vacuously."""
    from src.convert import convert

    with tempfile.TemporaryDirectory() as d:
        stl = _make_sphere_stl(os.path.join(d, "sphere.stl"))
        out = os.path.join(d, "run.step")
        code, res = convert(stl, out, target_faces=120, quiet=True)
        assert code in (0, 2), res.get("error")

        with open(out, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        m = re.search(r"CARTESIAN_POINT\('',\(-?\d+\.(\d)", text)
        assert m is not None, "no CARTESIAN_POINT in the exported STEP"
        pos = m.start(1)
        mutated = text[:pos] + ("9" if text[pos] != "9" else "1") + text[pos + 1:]
        # precondition: the mutation changed exactly one character in DATA
        assert mutated != text
        assert len(mutated) == len(text)
        assert sum(1 for ca, cb in zip(text, mutated) if ca != cb) == 1

        bad = os.path.join(d, "mutated.step")
        _write(bad, mutated)
        assert canonical_step_digest(bad) != canonical_step_digest(out)
        equal, diff = step_files_equal(out, bad)
        assert equal is False and diff


# --------------------------------------------------------------------------
# (d) Guard: numba must not come back
# --------------------------------------------------------------------------

def _numba_offenders(source):
    """Reasons why ``source`` still depends on numba - empty list = clean.

    The check runs on the parsed AST, not on the raw text, for two reasons:
    the kernels keep their legacy ``_numba_compute_*`` names (other code may
    import them), and the module docstring has to be allowed to *explain*
    why numba was removed. Only executable code counts.

    Note the decorator rule is deliberately blunt: *any* decorator on a
    function or class in this file is reported, not just numba's. A
    hand-rolled JIT wrapper would otherwise slip past the name and import
    rules. The price is that adding an innocent ``@staticmethod`` to
    ``g3_fitter.py`` also trips it - reword the rule then, do not delete it.
    """
    import ast

    tree = ast.parse(source)
    reasons = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "numba":
                    reasons.append("import numba (line %d)" % node.lineno)
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] == "numba":
                reasons.append("from numba import ... (line %d)" % node.lineno)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.decorator_list:
                reasons.append("decorator on %s (line %d)" % (node.name, node.lineno))
        elif isinstance(node, ast.Name):
            if node.id in ("njit", "jit", "vectorize", "guvectorize"):
                reasons.append("reference to %s (line %d)" % (node.id, node.lineno))
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, str) and node.value.strip() == "numba":
                reasons.append("string 'numba' used as code (line %d)" % node.lineno)
    return reasons


def _fitter_source():
    with open(FITTER_PATH, "r", encoding="utf-8") as fh:
        return fh.read()


def test_g3_fitter_has_no_numba_dependency():
    src = _fitter_source()
    # precondition: we are reading the right file
    assert "_numba_compute_interior_control_points" in src, \
        f"{FITTER_PATH} does not look like the fitter"

    offenders = _numba_offenders(src)
    assert offenders == [], (
        "numba / njit is back in src/nurbs/g3_fitter.py. It makes the "
        "maintainer's runs differ from a clean `pip install -r "
        "requirements.txt` (numba is not in it), and fastmath moves interior "
        "control points by ~0.35 mm through the lsqr solve. (Any decorator "
        "counts - see _numba_offenders.) Offenders: " + repr(offenders)
    )


def test_numba_guard_can_fire():
    """Mutation control for the guard: it must reject a source that DOES use
    numba, and must NOT fire on the legacy names or on the docstring that
    explains the removal. Without this, the test above passes against
    anything."""
    bad = ("from numba import njit\n"
           "\n"
           "@njit(fastmath=True, nogil=True)\n"
           "def f():\n"
           "    pass\n")
    assert _numba_offenders(bad), "guard does not fire on a numba-using source"

    bad2 = "import numba\n\ndef f():\n    return numba\n"
    assert _numba_offenders(bad2)

    ok = ('"""Was njit(fastmath=True) from numba; removed for determinism."""\n'
          "import numpy as np\n"
          "\n"
          "def _numba_compute_interior_control_points(ctrl_pts):\n"
          "    return np.asarray(ctrl_pts)\n")
    assert _numba_offenders(ok) == [], \
        "guard fires on the docstring / the legacy kernel names"


def test_g3_fitter_kernels_are_plain_python_functions():
    from src.nurbs.g3_fitter import (
        _numba_compute_edge_control_points,
        _numba_compute_interior_control_points,
    )

    for fn in (_numba_compute_edge_control_points,
               _numba_compute_interior_control_points):
        assert inspect.isfunction(fn), f"{fn!r} is not a plain Python function"
        # numba's Dispatcher exposes the original under .py_func and carries
        # .inspect_llvm / .signatures; a plain function has none of them.
        for attr in ("py_func", "inspect_llvm", "signatures", "nopython_signatures"):
            assert not hasattr(fn, attr), f"{fn.__name__} looks JIT-wrapped ({attr})"
        assert inspect.getsourcefile(fn) == FITTER_PATH


def test_g3_fitter_kernels_still_compute_the_right_thing():
    """The kernels must keep working after the decorators come off - a guard
    that only checks for the absence of numba would pass on a gutted file."""
    from src.nurbs.g3_fitter import (
        _numba_compute_edge_control_points,
        _numba_compute_interior_control_points,
    )

    p0 = np.array([0.0, 0.0, 0.0])
    p1 = np.array([5.0, 0.0, 0.0])
    d1 = np.array([5.0, 0.0, 0.0])
    z = np.zeros(3)
    edge = _numba_compute_edge_control_points(p0, p1, d1, d1, z)
    assert edge.shape == (6, 3)
    assert np.allclose(edge[0], p0) and np.allclose(edge[5], p1)
    # uniform tangents -> evenly spaced Bezier control points
    assert np.allclose(edge[:, 0], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])

    ctrl = np.zeros((6, 6, 3))
    for i in range(6):
        for j in range(6):
            ctrl[i, j] = [i, j, 0.0]
    interior = _numba_compute_interior_control_points(ctrl.copy())
    # a bilinear boundary reproduces itself through the Coons blend
    assert np.allclose(interior, ctrl)

    # defect-injected control: a non-planar boundary must move the interior,
    # so the assertion above is not passing on a no-op kernel
    bumped = ctrl.copy()
    bumped[0, 3, 2] = 7.0
    out = _numba_compute_interior_control_points(bumped.copy())
    assert not np.allclose(out[1:5, 1:5], bumped[1:5, 1:5])
