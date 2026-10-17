"""Sandbox worker of ``src.drawing``: job object, limits, capped pipes, RESULT.

Every child here is a tiny ``python -c`` program run through
``run_sandboxed`` with the sanitized environment and a fresh work dir
(pytest ``tmp_path``). Test discipline: every limit test asserts its
precondition (the grandchild really ran, the planted module really gets
imported without the guard, ...) and the clean run asserts that every
breach flag is False, so a constant or no-op implementation fails.

The Windows tests need a real job object (they run on this machine inside
an outer job; nested jobs work on Windows 11). The POSIX branch is only
unit-tested (written, not run on Linux).
"""
import json
import os
import sys
import time

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)

from src.drawing import sandbox  # noqa: E402
from src.drawing.sandbox import (  # noqa: E402
    SandboxResult, WindowsJob, CappedReader, run_sandboxed, sanitized_env)

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:  # pragma: no cover - psutil is installed on this machine
    psutil = None
    HAS_PSUTIL = False

windows_only = pytest.mark.skipif(os.name != "nt", reason="job object path is Windows only")
needs_psutil = pytest.mark.skipif(not HAS_PSUTIL, reason="psutil not installed")

PY = sys.executable


def _run(code, work_dir, **kw):
    args = dict(timeout_s=30.0, memory_limit_mb=1024, cpu_limit_s=60.0, output_cap_mb=200)
    args.update(kw)
    return run_sandboxed([PY, "-c", code], str(work_dir), **args)


def _alive_with_marker(pid, marker):
    """True when ``pid`` exists AND is still our process (guards PID reuse)."""
    if not psutil.pid_exists(pid):
        return False
    try:
        return any(marker in part for part in psutil.Process(pid).cmdline())
    except psutil.Error:
        return False


# ---------------------------------------------------------- environment --

def test_sanitized_env_contents(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONHOME", r"C:\nowhere")
    monkeypatch.setenv("SECRET_TOKEN_FOR_TEST", "x")
    assert os.environ.get("PYTHONHOME")  # precondition: the parent has it
    wd = str(tmp_path)
    env = sanitized_env(wd, REPO)
    assert env["OPENBLAS_NUM_THREADS"] == "1"
    assert env["OMP_NUM_THREADS"] == "1"
    assert env["PYTHONSAFEPATH"] == "1"
    assert env["PYTHONPATH"] == REPO
    assert env["TEMP"] == wd and env["TMP"] == wd
    assert "PYTHONHOME" not in env
    assert "SECRET_TOKEN_FOR_TEST" not in env
    if os.name == "nt":
        assert env["USERPROFILE"] == wd
        assert env["APPDATA"] == os.environ["APPDATA"]
        assert os.path.dirname(PY) in env["PATH"].split(os.pathsep)
    else:
        assert env["HOME"] == wd
    # two different work dirs give two different envs (not a constant)
    other = sanitized_env(os.path.join(wd, "b"), REPO)
    assert other["TEMP"] != env["TEMP"]


@windows_only
def test_planted_module_in_work_dir_is_not_imported(tmp_path):
    marker = tmp_path / "PLANTED_MARKER"
    (tmp_path / "json.py").write_text(
        "open(r'%s', 'w').write('imported')\n" % str(marker), encoding="utf-8")
    code = "import json; print('RESULT {}')"
    # precondition: without PYTHONSAFEPATH the cwd is on sys.path and the plant runs
    env_unsafe = sanitized_env(str(tmp_path), REPO)
    del env_unsafe["PYTHONSAFEPATH"]
    r0 = _run(code, tmp_path, env=env_unsafe)
    assert r0.exit_code == 0, r0.stderr_tail
    assert marker.exists(), "precondition failed: planted json.py was not imported without the guard"
    marker.unlink()
    # with the sanitized env it is not
    r1 = _run(code, tmp_path)
    assert r1.exit_code == 0, r1.stderr_tail
    assert r1.result == {}
    assert not marker.exists()


# ------------------------------------------------------- clean contract --

@windows_only
def test_result_line_parsed_and_no_breach_flags(tmp_path):
    code = "import json; print('progress', flush=True); print('RESULT ' + json.dumps({'ok': True, 'x': 1}))"
    r = _run(code, tmp_path)
    assert isinstance(r, SandboxResult)
    assert r.exit_code == 0
    assert r.result["x"] == 1 and r.result["ok"] is True
    s = r.sandbox
    assert s["enabled"] is True
    assert s["isolation"] == "job_object"
    assert s["wall_s"] > 0
    assert s["peak_rss_mb"] > 5
    assert s["peak_commit_mb"] is not None and s["peak_commit_mb"] > 0
    assert s["cpu_s"] is not None and s["cpu_s"] >= 0
    for flag in ("timed_out", "memory_exceeded", "cpu_exceeded", "output_exceeded", "killed",
                 "stdout_truncated", "stderr_truncated"):
        assert s[flag] is False, flag
    assert s["result_error"] is None
    assert s["exit_code"] == 0
    assert s["timeout_s"] == 30.0 and s["memory_limit_mb"] == 1024 and s["cpu_limit_s"] == 60.0
    assert r.contract_exit_code == 0
    assert "progress" in r.stdout_tail
    # json round trip of the section (it goes into the report file)
    json.dumps(s)


@windows_only
def test_child_exit_code_two_passes_through(tmp_path):
    code = "import sys; print('RESULT {\"ok\": true}'); sys.exit(2)"
    r = _run(code, tmp_path)
    assert r.exit_code == 2
    assert r.result == {"ok": True}
    assert r.contract_exit_code == 2


@windows_only
def test_no_result_line(tmp_path):
    r = _run("print('hello, no contract here')", tmp_path)
    assert r.exit_code == 0  # precondition: the child itself was fine
    assert r.result is None
    assert r.sandbox["result_error"] == "no RESULT line"
    assert r.contract_exit_code == 1


@windows_only
def test_bad_result_json(tmp_path):
    r = _run("print('RESULT {not json')", tmp_path)
    assert r.exit_code == 0
    assert r.result is None
    assert r.sandbox["result_error"].startswith("bad RESULT json")
    assert r.contract_exit_code == 1


@windows_only
def test_stdout_flood_is_capped_and_result_survives(tmp_path):
    code = ("import sys\n"
            "chunk = ('x' * 1023 + '\\n') * 1024\n"
            "for i in range(50): sys.stdout.write(chunk)\n"
            "sys.stdout.write('RESULT {\"ok\": true}\\n')\n")
    t0 = time.perf_counter()
    r = _run(code, tmp_path)
    assert r.exit_code == 0, r.stderr_tail
    assert r.sandbox["stdout_truncated"] is True
    assert r.result == {"ok": True}
    assert time.perf_counter() - t0 < 20


def test_capped_reader_keeps_head_and_tail():
    import io
    data = b"".join(b"line %06d\n" % i for i in range(20000)) + b"RESULT {}\n"
    rd = CappedReader(io.BufferedReader(io.BytesIO(data)), cap=1000, tail=200)
    rd.run()  # synchronous for the unit test
    assert rd.total == len(data)
    assert rd.truncated is True
    assert bytes(rd.head) == data[:1000]
    assert bytes(rd.tail) == data[-200:]
    assert rd.last_line() == "RESULT {}"
    small = CappedReader(io.BufferedReader(io.BytesIO(b"a\nb\n")), cap=1000, tail=200)
    small.run()
    assert small.truncated is False and small.last_line() == "b"


# --------------------------------------------------------------- limits --

@windows_only
@needs_psutil
def test_timeout_kills_child_and_grandchild(tmp_path):
    grand = "import time,os; open('g.pid','w').write(str(os.getpid())); time.sleep(60)"
    code = ("import time,subprocess,sys\n"
            "p = subprocess.Popen([sys.executable, '-c', %r])\n"
            "print(p.pid, flush=True)\n"
            "time.sleep(60)\n" % grand)
    t0 = time.perf_counter()
    r = _run(code, tmp_path, timeout_s=3.0)
    wall = time.perf_counter() - t0
    gpid_file = tmp_path / "g.pid"
    assert gpid_file.exists(), "precondition failed: the grandchild never ran"
    gpid = int(gpid_file.read_text())
    assert r.sandbox["timed_out"] is True
    assert r.sandbox["killed"] is True
    assert r.exit_code != 0
    assert r.contract_exit_code == 1
    assert r.sandbox["result_error"] == "no RESULT line"
    assert wall < 15
    time.sleep(0.2)
    assert not _alive_with_marker(gpid, "g.pid")
    assert not psutil.pid_exists(r.sandbox["pid"]) or not _alive_with_marker(r.sandbox["pid"], "g.pid")


@windows_only
def test_memory_limit_refuses_allocation(tmp_path):
    r = _run("b = bytearray(1_500_000_000)", tmp_path, memory_limit_mb=512)
    assert r.sandbox["memory_exceeded"] is True
    assert "PROCESS_MEMORY_LIMIT" in r.sandbox["job_messages"]
    assert r.exit_code == 1
    assert r.sandbox["killed"] is False
    assert "MemoryError" in r.stderr_tail
    assert r.contract_exit_code == 1


@windows_only
def test_memory_breach_detected_from_port_when_child_swallows_it(tmp_path):
    code = ("try:\n    b = bytearray(1_500_000_000)\nexcept MemoryError:\n    b = None\n"
            "print('RESULT {\"ok\": true}')\n")
    r = _run(code, tmp_path, memory_limit_mb=512)
    assert r.exit_code == 0 and r.result == {"ok": True}  # the child hid the breach
    assert "MemoryError" not in r.stderr_tail             # so stderr cannot tell
    assert r.sandbox["memory_exceeded"] is True           # only the port message can
    assert r.contract_exit_code == 1


LOG_LINE_CHILD = (
    "import sys\n"
    "sys.stderr.write('WARNING mesh: decimation hit MemoryError, using fallback\\n')\n"
    "sys.stderr.write('INFO numpy said: Unable to allocate 2 GiB, retried smaller\\n')\n"
    "sys.stderr.flush()\n"
    "print('RESULT {\"ok\": true}')\n")


@windows_only
def test_memory_word_in_a_log_line_is_not_a_breach(tmp_path):
    r = _run(LOG_LINE_CHILD, tmp_path)
    assert "MemoryError" in r.stderr_tail       # precondition: the words are in stderr
    assert r.exit_code == 0 and r.result == {"ok": True}
    assert r.sandbox["isolation"] == "job_object"
    assert r.sandbox["memory_exceeded"] is False
    assert r.contract_exit_code == 0


@windows_only
def test_job_path_uses_port_message_not_stderr_text(tmp_path):
    # a MemoryError that the job limit did not cause (no port message) is the
    # child's own failure, not a sandbox memory breach
    r = _run("raise MemoryError('not from the job limit')", tmp_path, memory_limit_mb=1024)
    assert "MemoryError" in r.stderr_tail and r.exit_code == 1   # precondition
    assert "PROCESS_MEMORY_LIMIT" not in r.sandbox["job_messages"]
    assert r.sandbox["memory_exceeded"] is False
    assert r.contract_exit_code == 1            # still a failed run (no RESULT)


@windows_only
@needs_psutil
def test_fallback_memory_heuristic_is_anchored_and_respects_ok(tmp_path, monkeypatch):
    def broken(*a, **k):
        raise OSError("simulated")
    monkeypatch.setattr(sandbox, "WindowsJob", broken)
    # a log line that mentions MemoryError, child reports ok: no breach
    r = _run(LOG_LINE_CHILD, tmp_path)
    assert r.sandbox["isolation"] == "psutil_tree"   # precondition: no port available
    assert "MemoryError" in r.stderr_tail
    assert r.result == {"ok": True}
    assert r.sandbox["memory_exceeded"] is False
    assert r.contract_exit_code == 0
    # a real traceback (no RESULT): without a port the stderr text is the signal
    r2 = _run("raise MemoryError('refused')", tmp_path)
    assert r2.sandbox["isolation"] == "psutil_tree"
    assert r2.exit_code == 1 and r2.result is None
    assert r2.sandbox["memory_exceeded"] is True
    # the numpy flavour of the traceback line also counts
    r3 = _run("import sys; sys.stderr.write('numpy._core._exceptions._ArrayMemoryError: "
              "Unable to allocate 1.46 GiB\\n'); sys.exit(1)", tmp_path)
    assert r3.sandbox["memory_exceeded"] is True
    # and so does a memory word in the child's own RESULT error when ok is false
    r4 = _run("import sys; print('RESULT {\"ok\": false, \"error\": \"MemoryError in HLR\"}'); "
              "sys.exit(1)", tmp_path)
    assert r4.result["ok"] is False
    assert r4.sandbox["memory_exceeded"] is True


@windows_only
def test_cpu_backstop(tmp_path):
    t0 = time.perf_counter()
    r = _run("while True: pass", tmp_path, cpu_limit_s=2.0, timeout_s=30.0)
    wall = time.perf_counter() - t0
    assert r.sandbox["timed_out"] is False  # precondition: the wall clock did not fire
    assert (r.exit_code & 0xFFFFFFFF) == 0xC0000044
    assert r.sandbox["cpu_exceeded"] is True
    # Windows enforces PROCESS_TIME coarsely: a 2 s (and even a 1 s) limit fired
    # after 4.4-8.1 s of job CPU / 4.5-8.5 s wall on this machine, so the bound
    # is loose; it still proves the backstop fired long before the 30 s timeout
    assert 2.0 <= r.sandbox["cpu_s"] < 12.0
    assert wall < 15
    assert r.contract_exit_code == 1


@windows_only
def test_output_cap_kills_writer(tmp_path):
    code = ("import time\nf = open('big.bin', 'wb')\n"
            "for i in range(300):\n    f.write(b'0' * 2**20); f.flush(); time.sleep(0.005)\n")
    r = _run(code, tmp_path, output_cap_mb=50)
    assert r.sandbox["output_exceeded"] is True
    assert r.sandbox["killed"] is True
    size_mb = os.path.getsize(tmp_path / "big.bin") / 2**20
    assert 50 <= size_mb < 300  # it crossed the cap and was stopped well before the end


@windows_only
def test_output_cap_measures_growth_not_preexisting_files(tmp_path):
    # the CLI copies the input into <work>/in before the spawn: a file larger
    # than the cap that was there before the child must not count
    (tmp_path / "in").mkdir()
    pre = tmp_path / "in" / "input.stl"
    with open(pre, "wb") as fh:
        fh.seek(60 * 2**20 - 1)
        fh.write(b"\0")
    assert os.path.getsize(pre) > 50 * 2**20  # precondition: pre-placed file exceeds the cap
    r = _run("print('RESULT {\"ok\": true}')", tmp_path, output_cap_mb=50)
    assert r.sandbox["output_exceeded"] is False
    assert r.sandbox["killed"] is False
    assert r.result == {"ok": True}
    assert r.contract_exit_code == 0
    assert r.sandbox["output_baseline_mb"] >= 60.0
    # growth on top of the baseline is still capped, and only growth counts:
    # the writer gets past 50 MB of its own before the kill (a total-size cap
    # would kill it at once, with almost nothing written)
    code = ("import time\nf = open('big.bin', 'wb')\n"
            "for i in range(300):\n    f.write(b'0' * 2**20); f.flush(); time.sleep(0.005)\n")
    r2 = _run(code, tmp_path, output_cap_mb=50)
    assert r2.sandbox["output_exceeded"] is True and r2.sandbox["killed"] is True
    size_mb = os.path.getsize(tmp_path / "big.bin") / 2**20
    assert 50 <= size_mb < 300


@windows_only
def test_extra_watch_dir_counts_toward_output_cap(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    wd = tmp_path / "work"
    wd.mkdir()
    code = ("import time\nf = open(r'%s', 'wb')\n"
            "for i in range(300):\n    f.write(b'0' * 2**20); f.flush(); time.sleep(0.005)\n"
            % str(out / "big.bin"))
    r = run_sandboxed([PY, "-c", code], str(wd), timeout_s=30, memory_limit_mb=1024,
                      cpu_limit_s=60, output_cap_mb=50, watch_dirs=[str(out)])
    assert r.sandbox["output_exceeded"] is True
    assert os.path.getsize(out / "big.bin") < 300 * 2**20


@windows_only
def test_native_crash_code_is_mapped(tmp_path):
    code = "import ctypes; ctypes.windll.kernel32.ExitProcess(ctypes.c_uint(0xC0000005))"
    r = _run(code, tmp_path)
    assert (r.exit_code & 0xFFFFFFFF) == 0xC0000005  # precondition
    assert r.result is None
    assert r.sandbox["result_error"] == "native crash (input unreadable by OpenCASCADE?)"
    assert r.contract_exit_code == 1
    # negative control: an ordinary exit 3 without RESULT is not called a crash
    r2 = _run("import sys; sys.exit(3)", tmp_path)
    assert r2.exit_code == 3
    assert r2.sandbox["result_error"] == "no RESULT line"


@windows_only
def test_child_is_in_job_and_below_normal_priority(tmp_path):
    code = ("import ctypes, psutil\n"
            "from ctypes import wintypes\n"
            "k = ctypes.WinDLL('kernel32')\n"
            "k.GetCurrentProcess.restype = wintypes.HANDLE\n"
            "k.IsProcessInJob.argtypes = [wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL)]\n"
            "res = wintypes.BOOL()\n"
            "k.IsProcessInJob(k.GetCurrentProcess(), None, ctypes.byref(res))\n"
            "import json; print('RESULT ' + json.dumps({'in_job': bool(res.value), 'nice': psutil.Process().nice()}))\n")
    r = _run(code, tmp_path)
    assert r.exit_code == 0, r.stderr_tail
    assert r.result["in_job"] is True
    if HAS_PSUTIL:
        assert r.result["nice"] == psutil.BELOW_NORMAL_PRIORITY_CLASS


@windows_only
def test_cad_imports_work_under_sanitized_env_and_512mb_cap(tmp_path):
    pytest.importorskip("trimesh")
    pytest.importorskip("ezdxf")
    pytest.importorskip("OCP")
    # user site-packages (APPDATA), a home dir for ezdxf (USERPROFILE) and the
    # single OpenBLAS thread are all needed for this to pass under 512 MB
    code = ("import numpy, trimesh, ezdxf\nfrom OCP.BRep import BRep_Tool\n"
            "import json, sys; print('RESULT ' + json.dumps({'ok': True, 'cwd_on_path': '' in sys.path}))\n")
    r = _run(code, tmp_path, memory_limit_mb=512, timeout_s=60.0)
    assert r.exit_code == 0, r.stderr_tail
    assert r.result == {"ok": True, "cwd_on_path": False}
    assert r.sandbox["memory_exceeded"] is False
    assert 50 < r.sandbox["peak_commit_mb"] < 512


# ------------------------------------------------------ psutil fallback --

@windows_only
@needs_psutil
def test_fallback_without_job_object_still_kills_tree(tmp_path, monkeypatch):
    def broken(*a, **k):
        raise OSError("simulated: job objects forbidden")
    monkeypatch.setattr(sandbox, "WindowsJob", broken)
    grand = "import time,os; open('g.pid','w').write(str(os.getpid())); time.sleep(60)"
    code = ("import time,subprocess,sys\n"
            "p = subprocess.Popen([sys.executable, '-c', %r])\n"
            "time.sleep(60)\n" % grand)
    r = _run(code, tmp_path, timeout_s=3.0)
    assert (tmp_path / "g.pid").exists(), "precondition failed: the grandchild never ran"
    gpid = int((tmp_path / "g.pid").read_text())
    assert r.sandbox["isolation"] == "psutil_tree"
    assert "simulated" in (r.sandbox["isolation_error"] or "")
    assert r.sandbox["timed_out"] is True and r.sandbox["killed"] is True
    time.sleep(0.2)
    assert not _alive_with_marker(gpid, "g.pid")


@windows_only
@needs_psutil
def test_fallback_reaps_orphaned_grandchild(tmp_path, monkeypatch):
    def broken(*a, **k):
        raise OSError("simulated")
    monkeypatch.setattr(sandbox, "WindowsJob", broken)
    grand = "import time,os; open('g.pid','w').write(str(os.getpid())); time.sleep(60)"
    # the child exits normally after 1.5 s and leaves its grandchild orphaned
    code = ("import time,subprocess,sys\n"
            "p = subprocess.Popen([sys.executable, '-c', %r], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
            "time.sleep(1.5)\n"
            "print('RESULT {\"ok\": true}')\n" % grand)
    r = _run(code, tmp_path)
    assert (tmp_path / "g.pid").exists(), "precondition failed: the grandchild never ran"
    gpid = int((tmp_path / "g.pid").read_text())
    assert r.exit_code == 0 and r.result == {"ok": True}
    assert r.sandbox["isolation"] == "psutil_tree"
    time.sleep(0.2)
    assert not _alive_with_marker(gpid, "g.pid")


# --------------------------------------------------------------- POSIX --

def test_posix_rlimit_env_round_trip():
    text = sandbox.posix_rlimit_env(memory_limit_mb=700, cpu_limit_s=12.5, output_cap_mb=33)
    plan = sandbox.parse_rlimits(text)
    assert plan["as_bytes"] == 700 * 2**20
    assert plan["cpu_s"] == 13          # rounded up, never below the requested seconds
    assert plan["fsize_bytes"] == 33 * 2**20
    other = sandbox.parse_rlimits(sandbox.posix_rlimit_env(memory_limit_mb=0, cpu_limit_s=0, output_cap_mb=0))
    assert other["as_bytes"] is None and other["cpu_s"] is None and other["fsize_bytes"] is None
    assert sandbox.parse_rlimits("") == {}
    assert sandbox.parse_rlimits("{garbage") == {}


class _FakeResource:
    """Stand-in for the POSIX ``resource`` module that records setrlimit calls."""
    RLIMIT_AS, RLIMIT_CPU, RLIMIT_FSIZE, RLIMIT_CORE = "AS", "CPU", "FSIZE", "CORE"

    def __init__(self):
        self.calls = {}

    def setrlimit(self, which, limits):
        self.calls[which] = tuple(limits)


def test_apply_rlimits_from_env_maps_to_setrlimit(monkeypatch):
    fake = _FakeResource()
    monkeypatch.setitem(sys.modules, "resource", fake)
    env = {sandbox.RLIMIT_ENV: sandbox.posix_rlimit_env(512, 12.5, 50)}
    applied = sandbox.apply_rlimits_from_env(env)
    assert fake.calls["AS"] == (512 * 2**20, 512 * 2**20)
    assert fake.calls["CPU"] == (13, 18)          # ceil(cpu), ceil(cpu) + 5
    assert fake.calls["FSIZE"] == (50 * 2**20, 50 * 2**20)
    assert fake.calls["CORE"] == (0, 0)
    assert applied == {"as_bytes": 512 * 2**20, "cpu_s": 13, "fsize_bytes": 50 * 2**20}
    # zero limits mean unlimited: only the core dump limit is set
    fake2 = _FakeResource()
    monkeypatch.setitem(sys.modules, "resource", fake2)
    assert sandbox.apply_rlimits_from_env({sandbox.RLIMIT_ENV: sandbox.posix_rlimit_env(0, 0, 0)}) == {}
    assert fake2.calls == {"CORE": (0, 0)}
    # no variable: nothing is touched
    fake3 = _FakeResource()
    monkeypatch.setitem(sys.modules, "resource", fake3)
    assert sandbox.apply_rlimits_from_env({}) == {}
    assert fake3.calls == {}


@pytest.mark.skipif(os.name != "nt", reason="resource exists on POSIX")
def test_apply_rlimits_without_resource_module_does_not_raise(monkeypatch):
    monkeypatch.setitem(sys.modules, "resource", None)   # import resource -> ImportError
    env = {sandbox.RLIMIT_ENV: sandbox.posix_rlimit_env(512, 10, 50)}
    assert sandbox.apply_rlimits_from_env(env) == {}


# ============================================================ QA round 1 (F5)
# Issue 20 (font cache / fontTools stderr), 17 (progress of a killed child),
# 22/21 (flag table, level), using report.py as the parent's failure report.

import logging  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import types  # noqa: E402

from src.drawing import report as report_mod  # noqa: E402
from src.drawing.report import ESCALATABLE_FLAGS, FLAGS, Report  # noqa: E402

_CACHE_PROBE = (
    "import json, logging, os, sys, time\n"
    "logging.basicConfig(level=logging.ERROR)\n"           # like the CLI with --quiet
    "f = os.path.join(os.environ['XDG_CACHE_HOME'], 'ezdxf', 'font_manager_cache.json')\n"
    "before = os.path.exists(f); m0 = os.path.getmtime(f) if before else None\n"
    "t = time.perf_counter()\n"
    "import ezdxf\n"
    "doc = ezdxf.new('R2010', setup=True)\n"
    "dt = time.perf_counter() - t\n"
    "print('RESULT ' + json.dumps({'ok': True, 'cache_at_start': before, 'import_s': dt,\n"
    "      'rebuilt': os.path.exists(f) and (m0 is None or os.path.getmtime(f) != m0)}))\n")


def test_sanitized_env_points_xdg_cache_into_work_dir(tmp_path):
    env = sanitized_env(str(tmp_path), REPO)
    assert env["XDG_CACHE_HOME"] == os.path.join(str(tmp_path), ".cache")
    assert sandbox._inside(env["XDG_CACHE_HOME"], str(tmp_path))
    assert not sandbox._inside(str(tmp_path), env["XDG_CACHE_HOME"])     # not a constant True


def test_font_cache_seeded_and_never_rebuilt_by_the_child(tmp_path):
    """Issue 20: the child finds ezdxf's font cache and does not rescan the system fonts."""
    pytest.importorskip("ezdxf")
    persistent = str(tmp_path / "persistent")
    # control: without seeding the child starts with no cache and rebuilds it
    wd0 = tmp_path / "w0"
    r0 = _run(_CACHE_PROBE, wd0, font_cache=False, timeout_s=90.0)
    assert r0.exit_code == 0, r0.stderr_tail
    assert r0.sandbox["font_cache"] == "disabled"
    assert r0.result["cache_at_start"] is False and r0.result["rebuilt"] is True
    # seeded: the persistent cache is built once (trusted child), copied into the work dir
    wd1 = tmp_path / "w1"
    r1 = _run(_CACHE_PROBE, wd1, font_cache=persistent, timeout_s=90.0)
    assert r1.exit_code == 0, r1.stderr_tail
    assert r1.sandbox["font_cache"] == "seeded"
    assert r1.result["cache_at_start"] is True and r1.result["rebuilt"] is False
    assert "fontTools" not in r1.stderr_tail and "ERROR" not in r1.stderr_tail
    src = os.path.join(persistent, "ezdxf", sandbox.FONT_CACHE_FILE)
    assert sandbox._valid_font_cache(src)
    m_src = os.path.getmtime(src)
    # a second job reuses the persistent file (no rebuild) and the child cannot touch it
    r2 = _run(_CACHE_PROBE, tmp_path / "w2", font_cache=persistent, timeout_s=90.0)
    assert r2.result["cache_at_start"] is True and r2.result["rebuilt"] is False
    assert os.path.getmtime(src) == m_src
    print("ezdxf import+new: unseeded %.3f s, seeded %.3f / %.3f s"
          % (r0.result["import_s"], r1.result["import_s"], r2.result["import_s"]))


def test_font_cache_build_failure_is_not_fatal(tmp_path, monkeypatch):
    calls = []

    def fake_run(*a, **kw):
        calls.append(a)
        return types.SimpleNamespace(returncode=1)

    monkeypatch.setattr(sandbox.subprocess, "run", fake_run)
    monkeypatch.setattr(sandbox, "_FONT_CACHE_FAILED", {})
    cache = str(tmp_path / "broken")
    assert sandbox.ensure_font_cache(cache) is None
    assert sandbox.ensure_font_cache(cache) is None
    assert len(calls) == 1                      # the failed build is not retried per job
    monkeypatch.undo()
    r = _run("import json; print('RESULT ' + json.dumps({'ok': True}))", tmp_path / "w",
             font_cache=cache)
    # undo() restored _FONT_CACHE_FAILED too; a real build into 'broken' may now succeed
    assert r.result == {"ok": True}
    assert r.sandbox["font_cache"] in ("seeded", "unavailable")


def _box_step(path):
    pytest.importorskip("OCP")
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
    w = STEPControl_Writer()
    w.Transfer(BRepPrimAPI_MakeBox(60.0, 40.0, 20.0).Shape(), STEPControl_AsIs)
    assert w.Write(str(path)) == IFSelect_RetDone
    return str(path)


@windows_only
def test_cli_sandboxed_quiet_run_has_empty_stderr_tail(tmp_path):
    """Issue 20: a normal sandboxed --quiet run leaves nothing in stderr_tail."""
    step = _box_step(tmp_path / "box.step")
    out = tmp_path / "out"
    cache_dir = tmp_path / "font_cache"                         # hermetic: not %LOCALAPPDATA%
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1",
               **{sandbox.FONT_CACHE_DIR_ENV: str(cache_dir)})
    r = subprocess.run([PY, "-m", "src.drawing", step, str(out), "--quiet"], cwd=REPO, env=env,
                       capture_output=True, timeout=240)
    line = [ln for ln in r.stdout.decode("utf-8", "replace").splitlines()
            if ln.startswith("RESULT ")]
    assert line, r.stderr.decode("utf-8", "replace")[-2000:]
    res = json.loads(line[-1][len("RESULT "):])
    assert res["sandbox"]["enabled"] is True                   # precondition: it WAS sandboxed
    assert res["ok"] is True, res.get("error")
    assert res["sandbox"]["font_cache"] == "seeded"
    assert res["sandbox"]["stderr_tail"] == ""
    # the persistent cache was built in the temp dir (the override took effect)
    assert any(cache_dir.iterdir()), "font cache not built under the override dir"


def test_default_font_cache_dir_env_override(tmp_path, monkeypatch):
    monkeypatch.delenv(sandbox.FONT_CACHE_DIR_ENV, raising=False)
    default = sandbox.default_font_cache_dir()
    assert os.path.normcase(str(tmp_path)) not in os.path.normcase(default)   # precondition
    monkeypatch.setenv(sandbox.FONT_CACHE_DIR_ENV, str(tmp_path / "fc"))
    assert sandbox.default_font_cache_dir() == os.path.abspath(str(tmp_path / "fc"))
    monkeypatch.setenv(sandbox.FONT_CACHE_DIR_ENV, "  ")
    assert sandbox.default_font_cache_dir() == default


# ------------------------------------------------------------- progress --

_PROGRESS_CHILD = (
    "import os, sys, time\n"
    "from src.drawing.report import Report, write_progress, progress_path_from_env\n"
    "rep = Report('box.step', '.', None)\n"
    "rep.data['input']['source_unit'] = 'mm'\n"
    "rep.data['geometry']['extents'] = [60.0, 40.0, 20.0]\n"
    "rep.flag('approval_blank', None, 'blank')\n"
    "rep.data['timing']['load_s'] = 0.25\n"
    "assert write_progress(progress_path_from_env(), rep.progress_snapshot('load'))\n"
    "assert write_progress(progress_path_from_env(), rep.progress_snapshot('views', view='front'))\n"
    "time.sleep(60)\n")


def test_progress_of_a_killed_child_reaches_the_fail_report(tmp_path):
    """Issue 17: timeout -> SandboxResult.progress -> merge_progress -> a useful report."""
    r = _run(_PROGRESS_CHILD, tmp_path, timeout_s=4.0, progress_file="progress.json")
    assert r.sandbox["timed_out"] is True and r.result is None         # precondition
    assert os.path.isfile(tmp_path / "progress.json")
    assert isinstance(r.progress, dict) and r.progress["stage"] == "views"
    assert r.sandbox["last_stage"] == "views" and r.sandbox["progress_error"] is None
    rep = Report(str(tmp_path / "box.step"), str(tmp_path), None)
    rep.data["sandbox"] = dict(r.sandbox)
    rep.merge_progress(r.progress, r.sandbox)
    code, data = rep.finish(False, "sandbox: " + r.failure_reason())
    assert code == 1 and data["quality"]["level"] == "none"
    flags = {f["code"]: f for f in data["quality"]["flags"]}
    assert "sandbox_timeout" in flags and "views" in flags["sandbox_timeout"]["detail"]
    assert "front" in flags["sandbox_timeout"]["detail"]
    assert flags["approval_blank"]["severity"] == "info"              # the child's own flags
    assert data["quality"]["last_stage"] == "views"
    assert data["input"]["source_unit"] == "mm"
    assert data["geometry"]["extents"] == [60.0, 40.0, 20.0]
    assert data["timing"]["load_s"] == 0.25
    assert data["timing"]["total_s"] == pytest.approx(r.sandbox["wall_s"]) and \
        data["timing"]["total_s"] >= 4.0


def test_progress_ignored_when_result_present_and_stale_file_removed(tmp_path):
    # a stale snapshot from an earlier job is removed before the spawn
    (tmp_path / "progress.json").write_text(json.dumps({"stage": "stale"}), encoding="utf-8")
    r = _run("import time; time.sleep(30)", tmp_path, timeout_s=2.0, progress_file="progress.json")
    assert r.sandbox["timed_out"] is True
    assert r.progress is None and r.sandbox["progress_error"] == "no progress file"
    # a child that reports a RESULT: the progress file is not read
    code = ("import json, os\n"
            "open(os.environ['SRC_DRAWING_PROGRESS_FILE'], 'w').write(json.dumps({'stage': 'x'}))\n"
            "print('RESULT ' + json.dumps({'ok': True}))\n")
    r2 = _run(code, tmp_path / "b", progress_file=str(tmp_path / "b" / "p.json"))
    assert os.path.isfile(tmp_path / "b" / "p.json")                 # precondition: it was written
    assert r2.result == {"ok": True} and r2.progress is None
    assert sandbox.PROGRESS_ENV == report_mod.PROGRESS_ENV


def test_progress_file_garbage_is_rejected(tmp_path):
    p = tmp_path / "p.json"
    p.write_text("[1, 2, 3]", encoding="utf-8")
    assert sandbox.read_progress(str(p))[0] is None
    p.write_text("{not json", encoding="utf-8")
    assert sandbox.read_progress(str(p))[0] is None
    p.write_text(json.dumps({"stage": "x" * 100}), encoding="utf-8")
    assert sandbox.read_progress(str(p))[0] == {"stage": "x" * 100}   # control: valid object
    assert sandbox.read_progress(str(p), max_bytes=10)[0] is None


def test_merge_progress_takes_only_plain_known_data(tmp_path):
    rep = Report(str(tmp_path / "x.step"), str(tmp_path), None)
    hostile = {"stage": "s" * 5000, "source_unit": 5,
               "geometry": {"extents": [1.0, 2.0, 3.0], "volume_mm3": {"a": {"b": {"c": 1}}},
                            "not_a_key": 1},
               "flags": [{"code": "ok_code", "severity": "warning", "detail": "d" * 5000},
                         {"code": "bad code!", "severity": "warning"},
                         {"code": "x", "severity": "fatal"}, "junk"],
               "warnings": ["w"] + [1, 2], "timing": {"load_s": float("nan"), "views_s": 1.5}}
    rep.merge_progress(hostile, {"timed_out": True, "killed": True, "wall_s": 9.5,
                                 "timeout_s": 9.0})
    code, d = rep.finish(False, "x")
    assert len(d["quality"]["last_stage"]) == 500
    assert d["input"]["source_unit"] is None
    assert d["geometry"]["extents"] == [1.0, 2.0, 3.0] and d["geometry"]["volume_mm3"] is None
    assert "not_a_key" not in d["geometry"]
    codes = [f["code"] for f in d["quality"]["flags"]]
    assert codes.count("ok_code") == 1 and "bad code!" not in codes and "x" not in codes
    assert len(next(f for f in d["quality"]["flags"] if f["code"] == "ok_code")["detail"]) == 500
    assert codes.count("sandbox_timeout") == 1 and "sandbox_killed" not in codes
    assert d["warnings"] == ["w"]
    assert d["timing"]["load_s"] is None and d["timing"]["views_s"] == 1.5
    assert d["timing"]["total_s"] == 9.5
    # no progress at all: the breach flag still names the missing file
    rep2 = Report(str(tmp_path / "y.step"), str(tmp_path), None)
    rep2.merge_progress(None, {"memory_exceeded": True, "killed": True, "memory_limit_mb": 512})
    f = rep2.data["quality"]["flags"][0]
    assert f["code"] == "sandbox_memory_exceeded" and "no progress file" in f["detail"]


# ------------------------------------------------------------ flag table --

_NEW_CODES = {
    "loose_curves_ignored": "info", "supplemental_shells_ignored": "info",
    "dimension_extent_mismatch": "warning", "unit_assumed_mm": "info", "unit_suspect": "info",
    "step_tessellated_only": "warning", "sandbox_timeout": "warning",
    "hidden_lines_dropped": "warning", "not_a_solid": "warning", "mesh_non_manifold": "warning",
    "section_fallback": "info", "interior_silhouettes_omitted": "info",
    "multiple_bodies": "warning", "approval_blank": "info", "faceted_audit_mismatch": "warning",
    "view_failed": "warning", "section_failed": "warning", "dimensions_failed": "warning",
    "writer_failed": "warning",
}


def test_flag_table_has_every_code_with_its_severity():
    for code, sev in _NEW_CODES.items():
        assert FLAGS.get(code) == sev, code
    assert set(FLAGS.values()) == {"info", "warning"}
    assert report_mod.WARNING_FLAGS | report_mod.INFO_FLAGS == set(FLAGS)
    assert not report_mod.WARNING_FLAGS & report_mod.INFO_FLAGS
    assert "none" in report_mod.LEVELS


def test_every_flag_code_in_the_source_is_in_the_table():
    """Each ``.flag("code", ...)`` literal in src/drawing is in FLAGS (typo guard)."""
    pat = re.compile(r"\.flag\(\s*[\"']([A-Za-z0-9_]+)[\"']")
    found = {}
    src = os.path.join(REPO, "src", "drawing")
    for dp, _dn, files in os.walk(src):
        for fn in files:
            if fn.endswith(".py"):
                text = open(os.path.join(dp, fn), encoding="utf-8").read()
                for m in pat.finditer(text):
                    found.setdefault(m.group(1), fn)
    assert len(found) >= 20                                    # precondition: the scan works
    missing = {c: fn for c, fn in found.items() if c not in FLAGS}
    assert not missing, missing


def test_every_tuple_emitted_flag_code_is_in_the_table_with_its_severity():
    """Review F5: layout/dimensions/pipeline emit flags as ``("code", "warning"|"info",
    detail)`` tuples that reach ``Report.flag`` later; each code must be in FLAGS
    and its literal severity must match the table (or be an allowed escalation)."""
    pat = re.compile(r"\(\s*[\"']([a-z0-9_]+)[\"']\s*,\s*[\"'](warning|info)[\"']")
    found = {}
    src = os.path.join(REPO, "src", "drawing")
    for dp, _dn, files in os.walk(src):
        for fn in files:
            if fn.endswith(".py") and fn != "report.py":
                text = open(os.path.join(dp, fn), encoding="utf-8").read()
                for m in pat.finditer(text):
                    found.setdefault((m.group(1), m.group(2)), fn)
    codes = {c for c, _s in found}
    # precondition: the scan sees the layout and dimensions tuple paths
    assert {"sheet_not_iso5457", "views_overflow", "dimension_extent_mismatch"} <= codes
    missing = {c: fn for (c, _s), fn in found.items() if c not in FLAGS}
    assert not missing, missing
    wrong = {c: (s, FLAGS[c]) for (c, s) in found
             if s != FLAGS[c] and not (c in ESCALATABLE_FLAGS and s == "warning")}
    assert not wrong, wrong


def test_flag_default_severity_unknown_code_and_escalation(tmp_path, caplog):
    rep = Report(str(tmp_path / "p.step"), str(tmp_path), None)
    rep.flag("not_a_solid", detail="50 open components")
    rep.flag("approval_blank")
    sev = {f["code"]: f["severity"] for f in rep.data["quality"]["flags"]}
    assert sev == {"not_a_solid": "warning", "approval_blank": "info"}
    with pytest.raises(ValueError):
        rep.flag("no_such_code_xyz")                         # unknown and no severity
    with caplog.at_level(logging.WARNING, logger="src.drawing.report"):
        rep.flag("no_such_code_xyz", "info", "typo?")
        assert any("no_such_code_xyz" in r.getMessage() for r in caplog.records)
        caplog.clear()
        rep.flag("unit_assumed_mm", "warning", "5 x 2.5 x 1.375 looks like inch")
        assert not caplog.records                            # allowed escalation
        rep.flag("section_fallback", "warning", "wrong")
        assert any("section_fallback" in r.getMessage() for r in caplog.records)
    assert rep.has_flag("no_such_code_xyz")


def _full_report(tmp_path, **flags):
    rep = Report(str(tmp_path / "p.step"), str(tmp_path), types.SimpleNamespace(hidden_lines=True))
    rep.data["route"]["pipeline"] = "brep"
    rep.data["views"] = [{"name": "front", "algorithm": "hlr_exact"}]
    rep.data["section"] = {"loops": 2}
    rep.data["dimensions"] = [{"kind": "linear", "value": 60.0}]
    for code in flags:
        rep.flag(code)
    return rep


def test_not_a_solid_caps_level_at_minimal(tmp_path):
    assert _full_report(tmp_path).finish(True)[1]["quality"]["level"] == "full"   # control
    code, d = _full_report(tmp_path, not_a_solid=1).finish(True)
    assert d["quality"]["level"] == "minimal" and code == 2
    assert _full_report(tmp_path, not_a_solid=1).finish(False, "x")[1]["quality"]["level"] == "none"


def test_spawn_failure_sets_the_progress_keys(tmp_path):
    """Review F5: the spawn-failed path reports the same sandbox keys as the others."""
    missing_exe = str(tmp_path / "no_such_interpreter.exe")
    assert not os.path.exists(missing_exe)                      # precondition: spawn must fail
    r = run_sandboxed([missing_exe, "-c", "pass"], str(tmp_path / "w"), timeout_s=5.0,
                      memory_limit_mb=256, cpu_limit_s=5.0, output_cap_mb=10,
                      progress_file="progress.json", font_cache=False)
    assert r.exit_code == 1 and r.result is None
    assert str(r.sandbox["result_error"]).startswith("spawn failed")
    assert r.progress is None
    assert "last_stage" in r.sandbox and r.sandbox["last_stage"] is None
    assert r.sandbox["progress_error"] == "spawn failed"
