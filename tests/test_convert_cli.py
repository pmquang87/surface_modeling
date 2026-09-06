"""End-to-end tests of the ``python -m src.convert`` command line.

Contract (pattern from BlinkingSun/stl2step, MIT): the LAST line on stdout
is ``RESULT {json}`` and it is the only machine-readable stdout content;
human progress goes to stderr. Exit codes: 0 clean, 2 STEP written but the
audit found something, 1 failed and no output written (usage errors too).
"""
import json
import os
import subprocess
import sys
import tempfile

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

try:
    import OCP  # noqa: F401
    HAS_OCP = True
except ImportError:
    HAS_OCP = False

needs_ocp = pytest.mark.skipif(not HAS_OCP, reason="cadquery-ocp not installed")


def _run(args, cwd=REPO, timeout=600):
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run([sys.executable, "-m", "src.convert", *args],
                          cwd=cwd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", env=env,
                          timeout=timeout)
    return proc


def _result_line(stdout):
    lines = [ln for ln in stdout.splitlines() if ln.strip()]
    assert lines, "no stdout at all"
    last = lines[-1]
    assert last.startswith("RESULT "), f"last stdout line is not RESULT: {last!r}"
    return json.loads(last[len("RESULT "):]), lines


def _write_icosphere(path):
    import trimesh
    m = trimesh.creation.icosphere(subdivisions=3, radius=10.0)
    m.export(path)


@needs_ocp
def test_cli_success_emits_single_result_line_and_exit_0():
    with tempfile.TemporaryDirectory() as d:
        stl = os.path.join(d, "sphere.stl")
        step = os.path.join(d, "sphere.step")
        _write_icosphere(stl)
        proc = _run([stl, step, "--target-faces", "120"])
        assert proc.returncode == 0, proc.stderr[-2000:]
        result, lines = _result_line(proc.stdout)
        # stdout carries ONLY the contract line: no OCCT banner, no progress
        assert len(lines) == 1, f"stdout polluted:\n{proc.stdout}"
        assert result["ok"] is True
        assert result["exit_code"] == 0
        assert result["output"].endswith("sphere.step")
        assert result["triangles"] == 1280
        audit = result["audit"]
        for key in ("faces", "solids", "shells", "free_edges", "brepcheck_valid",
                    "brep_volume_mm3", "mesh_volume_mm3", "volume_delta_pct",
                    "dev_mean_mm", "dev_p95_mm", "dev_max_mm", "bbox_max_diff_mm"):
            assert key in audit, key
        assert audit["solids"] == 1
        assert audit["free_edges"] == 0
        assert audit["faces"] == result["patches"]
        assert result["seconds"] > 0
        # progress went to stderr
        assert "loading" in proc.stderr
        assert os.path.exists(step)


@needs_ocp
def test_cli_missing_input_is_exit_1_with_error_and_no_output():
    with tempfile.TemporaryDirectory() as d:
        missing = os.path.join(d, "nope.stl")
        step = os.path.join(d, "out.step")
        proc = _run([missing, step])
        assert proc.returncode == 1
        result, lines = _result_line(proc.stdout)
        assert len(lines) == 1
        assert result["ok"] is False
        assert "nope.stl" in result["error"]
        assert not os.path.exists(step)
        assert "Traceback" not in proc.stderr


def test_cli_usage_error_is_exit_1_without_result_line():
    # argparse's default exit 2 collided with "written with warnings"
    proc = _run(["--bogus-flag", "a.stl", "b.step"])
    assert proc.returncode == 1
    assert "RESULT" not in proc.stdout
    assert "usage" in proc.stderr.lower()


@needs_ocp
def test_cli_quiet_suppresses_progress_but_keeps_result():
    with tempfile.TemporaryDirectory() as d:
        stl = os.path.join(d, "sphere.stl")
        step = os.path.join(d, "sphere.step")
        _write_icosphere(stl)
        proc = _run([stl, step, "--target-faces", "120", "--quiet"])
        assert proc.returncode == 0, proc.stderr[-2000:]
        result, lines = _result_line(proc.stdout)
        assert len(lines) == 1
        assert result["ok"] is True
        assert proc.stderr.strip() == "", f"--quiet leaked to stderr:\n{proc.stderr}"


@needs_ocp
def test_cli_volume_tolerance_flag_can_fail_a_good_fit():
    """--volume-tol is a real gate: an impossibly tight budget turns a good
    conversion into exit 2 with the volume named in the reasons."""
    with tempfile.TemporaryDirectory() as d:
        stl = os.path.join(d, "sphere.stl")
        step = os.path.join(d, "sphere.step")
        _write_icosphere(stl)
        proc = _run([stl, step, "--target-faces", "120", "--volume-tol", "1e-12"])
        assert proc.returncode == 2, proc.stderr[-2000:]
        result, _ = _result_line(proc.stdout)
        assert result["ok"] is True  # a file WAS written
        assert result["exit_code"] == 2
        assert any("volume" in r for r in result["audit_reasons"])
        assert os.path.exists(step)
