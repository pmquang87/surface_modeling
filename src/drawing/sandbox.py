"""Sandbox worker: run one child process under resource limits and parse its RESULT.

``run_sandboxed(argv, work_dir, ...)`` spawns ``argv`` with cwd = ``work_dir``,
``stdin`` closed, a sanitized environment (``sanitized_env``) and, on Windows,
``BELOW_NORMAL_PRIORITY_CLASS | CREATE_NO_WINDOW``. It enforces a wall-clock
timeout, a committed-memory cap, a CPU-time backstop and a byte cap on the
work dir, kills the whole process tree on a breach, captures stdout/stderr
through two capped reader threads and parses the child's ``RESULT {json}``
line (the stdout contract of ``src/convert.py``). The result carries the
``sandbox`` section of the quality report (spec 3.8 / 3.9).

The module is standalone: it imports nothing from the package and no
geometry library, so the CLI parent can use it before numpy/OCP are loaded.
``psutil`` is optional and imported lazily.

Measured facts it relies on (sandbox probe, Windows 11 26200, Python 3.14):

* Job objects through ctypes work even though the worker itself already runs
  inside a job (Claude Code, terminals): nested jobs are allowed. The child
  cannot break away (CREATE_BREAKAWAY_FROM_JOB -> access denied).
* The job memory limit counts committed memory. A breach does NOT kill the
  child: the allocation is refused and Python raises ``MemoryError`` (a
  1.5 GB ``bytearray`` under 512 MB exits 1 with a traceback in 0.1 s). The
  job's peak counters never show a refused allocation (peak stayed at
  10 MB), so on the job-object path ``memory_exceeded`` comes from the
  completion-port message ``PROCESS_MEMORY_LIMIT`` / ``JOB_MEMORY_LIMIT``
  alone (also posted when the child catches the error and exits 0), never
  from the peak and never from stderr text (a log line that merely mentions
  ``MemoryError`` must not fail a good run). Without a job object (psutil
  fallback, POSIX) there is no port: there a traceback line starting with
  ``...MemoryError`` or ``OpenBLAS error`` in stderr, or a memory word in
  ``RESULT.error``, counts, but only when the child did not report ``ok: true``.
* The output cap measures GROWTH: the bytes already in the watched dirs
  just before the spawn (the CLI copies the input into ``<work>/in`` first)
  are the baseline and do not count toward ``output_cap_mb``.
* ``TerminateJobObject`` kills child and grandchildren in about 3 ms;
  ``KILL_ON_JOB_CLOSE`` reaps orphans when the handle closes.
* ``JOB_OBJECT_LIMIT_PROCESS_TIME`` is enforced coarsely: a 2 s limit gave
  exit 0xC0000044 after 4.4-8.1 s of job CPU (4.5-8.5 s wall), a 1 s limit
  after 7 s, so it is a backstop only; the wall clock is the main limit.
* numpy's OpenBLAS commits about 628 MB at import with 20 threads; with
  ``OPENBLAS_NUM_THREADS=1`` numpy commits 17.5 MB and OCP + trimesh 192 MB.
  Without it the child dies with "OpenBLAS error" under any cap below about
  650 MB before any Python code runs.
* All CAD packages live in the user site-packages: ``-I``/``-s`` break every
  import and ``APPDATA`` must be passed; ezdxf (and so cadquery) raise
  "Could not determine home directory" without ``USERPROFILE``.
  ``PYTHONSAFEPATH=1`` keeps the cwd off ``sys.path`` (a planted ``json.py``
  in the work dir is imported otherwise).
* Two daemon reader threads keeping a 1 MiB head and a 64 KiB tail read a
  50 MB stdout flood in 0.32 s without deadlock, and the RESULT line is
  recovered from the tail.
* The psutil-only fallback (no job object) misses grandchildren whose parent
  already exited; here every descendant seen while polling is remembered
  with its create time and killed at the end if it still runs.
* Return code 0xC0000005 is a native access violation (for example
  ``TransferRoots`` after a failed STEP read); it is reported as unreadable
  input.

The POSIX branch (``_posix_run``) is written but UNVERIFIED (no Linux host):
it starts the child in its own session, passes the rlimits in the
environment variable ``RLIMIT_ENV`` (the child applies them itself with
``apply_rlimits_from_env``; ``preexec_fn`` is unsafe with reader threads)
and kills the process group.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

IS_WINDOWS = os.name == "nt"

STREAM_HEAD_CAP = 1 << 20          # bytes of stdout/stderr kept from the start
STREAM_TAIL_CAP = 64 * 1024        # rolling tail; the RESULT line is recovered from it
TEXT_TAIL_CHARS = 2000             # length of stdout_tail / stderr_tail in the result

STATUS_ACCESS_VIOLATION = 0xC0000005
STATUS_QUOTA_EXCEEDED = 0xC0000044   # exit code of a JOB_OBJECT_LIMIT_PROCESS_TIME breach
NATIVE_CRASH_CODES = (STATUS_ACCESS_VIOLATION,)
NATIVE_CRASH_ERROR = "native crash (input unreadable by OpenCASCADE?)"
NO_RESULT_ERROR = "no RESULT line"
BAD_RESULT_ERROR = "bad RESULT json"

RLIMIT_ENV = "SRC_DRAWING_RLIMITS"   # POSIX: JSON limits the child applies to itself

# Fallback-only heuristics (no completion port). The stderr pattern is anchored
# to the start of a line, so it matches a traceback's last line
# ("MemoryError", "numpy._core._exceptions._ArrayMemoryError: Unable to
# allocate ...") and OpenBLAS's own message, not a log line that mentions them.
_STDERR_MEMORY_PATTERN = re.compile(r"^[\w.]*MemoryError\b|^OpenBLAS error", re.M)
_RESULT_MEMORY_PATTERN = re.compile(r"MemoryError|Unable to allocate|OpenBLAS error")
_DEFAULT_REPO_ROOT = str(Path(__file__).resolve().parents[2])


# ------------------------------------------------------------- result --

@dataclass
class SandboxResult:
    exit_code: int                 # raw child return code (0xC0000044 etc. kept as is)
    result: Optional[dict]         # parsed RESULT json, None when missing or unparseable
    sandbox: dict                  # the report's "sandbox" section
    stdout_tail: str
    stderr_tail: str

    def failure_reason(self) -> Optional[str]:
        """Why the run counts as failed, or None when the child's own verdict stands."""
        s = self.sandbox
        if s.get("timed_out"):
            return "timed out after %.0f s" % float(s.get("timeout_s") or 0.0)
        if s.get("memory_exceeded"):
            return "memory limit of %s MB exceeded" % s.get("memory_limit_mb")
        if s.get("cpu_exceeded"):
            return "CPU time limit of %.0f s exceeded" % float(s.get("cpu_limit_s") or 0.0)
        if s.get("output_exceeded"):
            return "output cap of %s MB exceeded" % s.get("output_cap_mb")
        if s.get("killed"):
            return "child killed"
        if self.result is None:
            return s.get("result_error") or NO_RESULT_ERROR
        if self.exit_code not in (0, 2):
            return "child exit code %d" % self.exit_code
        return None

    @property
    def contract_exit_code(self) -> int:
        """Exit code of the parent under the RESULT contract: the child's 0 / 2,
        else 1 (killed, timed out, limit breached, no RESULT, crash, ok false)."""
        if self.failure_reason() is not None:
            return 1
        if isinstance(self.result, dict) and self.result.get("ok") is False:
            return 1
        return int(self.exit_code)


# ---------------------------------------------------------- environment --

def sanitized_env(work_dir: str, repo_root: str) -> dict:
    """The child's whole environment (nothing else is inherited).

    Measured as the minimal working set for numpy/OCP/trimesh/ezdxf/cadquery
    imports from an empty temp cwd; see the module docstring for why each
    variable is there. ``PYTHONHOME`` and every secret of the parent are
    dropped because the dict is built from scratch.
    """
    work_dir = str(work_dir)
    env = {
        "PYTHONPATH": str(repo_root),
        "PYTHONSAFEPATH": "1",            # cwd never on sys.path
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
        "OPENBLAS_NUM_THREADS": "1",      # 20 OpenBLAS threads commit ~630 MB at numpy import
        "OMP_NUM_THREADS": "1",
        "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": work_dir,
        "TEMP": work_dir,
        "TMP": work_dir,
    }
    if IS_WINDOWS:
        systemroot = os.environ.get("SYSTEMROOT", r"C:\Windows")
        env["SYSTEMROOT"] = systemroot
        if os.environ.get("APPDATA"):
            env["APPDATA"] = os.environ["APPDATA"]   # user site-packages live under it
        env["USERPROFILE"] = work_dir                 # ezdxf needs a home dir; a fake one is fine
        env["PATH"] = os.pathsep.join([os.path.dirname(sys.executable),
                                       os.path.join(systemroot, "System32")])
    else:
        env["HOME"] = work_dir
        env["TMPDIR"] = work_dir
        env["PATH"] = os.pathsep.join([os.path.dirname(sys.executable), "/usr/bin", "/bin"])
    return env


# ------------------------------------------------------- capped reader --

class CappedReader(threading.Thread):
    """Drain a pipe concurrently; keep the first ``cap`` bytes and a rolling tail.

    Without concurrent draining ``Popen.wait`` deadlocks once the pipe buffer
    is full; ``communicate`` would buffer everything with no cap.
    """

    def __init__(self, pipe, cap: int = STREAM_HEAD_CAP, tail: int = STREAM_TAIL_CAP):
        super().__init__(daemon=True)
        self.pipe, self.cap, self.tail_cap = pipe, int(cap), int(tail)
        self.head = bytearray()
        self.tail = bytearray()
        self.total = 0

    def run(self):
        read = self.pipe.read1 if hasattr(self.pipe, "read1") else self.pipe.read
        try:
            while True:
                b = read(65536)
                if not b:
                    break
                self.total += len(b)
                room = self.cap - len(self.head)
                if room > 0:
                    self.head += b[:room]
                self.tail += b
                if len(self.tail) > self.tail_cap:
                    del self.tail[: len(self.tail) - self.tail_cap]
        except (OSError, ValueError) as exc:     # pipe closed under us
            logger.debug("capped reader stopped: %s", exc)
        finally:
            try:
                self.pipe.close()
            except Exception:
                pass

    @property
    def truncated(self) -> bool:
        return self.total > self.cap

    def text(self) -> str:
        """Everything when nothing was dropped, else the tail."""
        data = self.head if not self.truncated else self.tail
        return bytes(data).decode("utf-8", "replace")

    def last_line(self) -> str:
        lines = bytes(self.tail).rstrip(b"\r\n").splitlines()
        return lines[-1].decode("utf-8", "replace").rstrip("\r") if lines else ""

    def tail_text(self, n_chars: int = TEXT_TAIL_CHARS) -> str:
        return bytes(self.tail).decode("utf-8", "replace")[-n_chars:]


# ----------------------------------------------------------- job object --

if IS_WINDOWS:
    import ctypes
    from ctypes import wintypes

    JOB_OBJECT_LIMIT_PROCESS_TIME = 0x00000002
    JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100
    JOB_OBJECT_LIMIT_JOB_MEMORY = 0x00000200
    JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION = 0x00000400
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
    _JobObjectBasicAccountingInformation = 1
    _JobObjectAssociateCompletionPortInformation = 7
    _JobObjectExtendedLimitInformation = 9

    class _IO_COUNTERS(ctypes.Structure):
        _fields_ = [(n, ctypes.c_ulonglong) for n in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong),
                    ("PerJobUserTimeLimit", ctypes.c_longlong),
                    ("LimitFlags", wintypes.DWORD),
                    ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t),
                    ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t),
                    ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):   # 144 bytes on x64
        _fields_ = [("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
                    ("IoInfo", _IO_COUNTERS),
                    ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]

    class _JOBOBJECT_BASIC_ACCOUNTING_INFORMATION(ctypes.Structure):
        _fields_ = [("TotalUserTime", ctypes.c_longlong),
                    ("TotalKernelTime", ctypes.c_longlong),
                    ("ThisPeriodTotalUserTime", ctypes.c_longlong),
                    ("ThisPeriodTotalKernelTime", ctypes.c_longlong),
                    ("TotalPageFaultCount", wintypes.DWORD),
                    ("TotalProcesses", wintypes.DWORD),
                    ("ActiveProcesses", wintypes.DWORD),
                    ("TotalTerminatedProcesses", wintypes.DWORD)]

    class _JOBOBJECT_ASSOCIATE_COMPLETION_PORT(ctypes.Structure):
        _fields_ = [("CompletionKey", wintypes.LPVOID), ("CompletionPort", wintypes.HANDLE)]

    class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _k32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
    _k32.CreateJobObjectW.restype = wintypes.HANDLE
    _k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]
    _k32.SetInformationJobObject.restype = wintypes.BOOL
    _k32.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID,
                                               wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
    _k32.QueryInformationJobObject.restype = wintypes.BOOL
    _k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    _k32.AssignProcessToJobObject.restype = wintypes.BOOL
    _k32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
    _k32.TerminateJobObject.restype = wintypes.BOOL
    _k32.CloseHandle.argtypes = [wintypes.HANDLE]
    _k32.CloseHandle.restype = wintypes.BOOL
    _k32.CreateIoCompletionPort.argtypes = [wintypes.HANDLE, wintypes.HANDLE, ctypes.c_size_t, wintypes.DWORD]
    _k32.CreateIoCompletionPort.restype = wintypes.HANDLE
    _k32.GetQueuedCompletionStatus.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD),
                                               ctypes.POINTER(ctypes.c_size_t),
                                               ctypes.POINTER(wintypes.LPVOID), wintypes.DWORD]
    _k32.GetQueuedCompletionStatus.restype = wintypes.BOOL
    _k32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PROCESS_MEMORY_COUNTERS),
                                             wintypes.DWORD]
    _k32.K32GetProcessMemoryInfo.restype = wintypes.BOOL
    _INVALID_HANDLE_VALUE = wintypes.HANDLE(-1).value

_PORT_MESSAGES = {1: "END_OF_JOB_TIME", 2: "END_OF_PROCESS_TIME", 3: "ACTIVE_PROCESS_LIMIT",
                  4: "ACTIVE_PROCESS_ZERO", 6: "NEW_PROCESS", 7: "EXIT_PROCESS",
                  8: "ABNORMAL_EXIT_PROCESS", 9: "PROCESS_MEMORY_LIMIT", 10: "JOB_MEMORY_LIMIT"}
_ROUTINE_MESSAGES = {"NEW_PROCESS", "EXIT_PROCESS", "ACTIVE_PROCESS_ZERO"}
_MEMORY_MESSAGES = {"PROCESS_MEMORY_LIMIT", "JOB_MEMORY_LIMIT"}
_CPU_MESSAGES = {"END_OF_PROCESS_TIME", "END_OF_JOB_TIME"}


class WindowsJob:
    """A Windows job object with a completion port (ctypes, kernel32).

    Limits: process + job committed memory, per-process user CPU time,
    KILL_ON_JOB_CLOSE, DIE_ON_UNHANDLED_EXCEPTION (no WER dialog). Raises
    ``OSError`` when the job cannot be created or configured; the caller then
    falls back to the psutil tree.
    """

    def __init__(self, memory_limit_bytes: Optional[int] = None, cpu_limit_s: Optional[float] = None):
        if not IS_WINDOWS:
            raise OSError("job objects exist on Windows only")
        self.handle = None
        self.port = None
        h = _k32.CreateJobObjectW(None, None)
        if not h:
            raise ctypes.WinError(ctypes.get_last_error())
        self.handle = h
        try:
            info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
            flags = JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION | JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if memory_limit_bytes:
                flags |= JOB_OBJECT_LIMIT_PROCESS_MEMORY | JOB_OBJECT_LIMIT_JOB_MEMORY
                info.ProcessMemoryLimit = int(memory_limit_bytes)
                info.JobMemoryLimit = int(memory_limit_bytes)
            if cpu_limit_s:
                flags |= JOB_OBJECT_LIMIT_PROCESS_TIME
                info.BasicLimitInformation.PerProcessUserTimeLimit = int(cpu_limit_s * 10_000_000)
            info.BasicLimitInformation.LimitFlags = flags
            if not _k32.SetInformationJobObject(h, _JobObjectExtendedLimitInformation,
                                                ctypes.byref(info), ctypes.sizeof(info)):
                raise ctypes.WinError(ctypes.get_last_error())
            port = _k32.CreateIoCompletionPort(_INVALID_HANDLE_VALUE, None, 0, 1)
            if not port:
                raise ctypes.WinError(ctypes.get_last_error())
            self.port = port
            assoc = _JOBOBJECT_ASSOCIATE_COMPLETION_PORT(None, port)
            if not _k32.SetInformationJobObject(h, _JobObjectAssociateCompletionPortInformation,
                                                ctypes.byref(assoc), ctypes.sizeof(assoc)):
                raise ctypes.WinError(ctypes.get_last_error())
        except BaseException:
            self.close()
            raise

    def assign(self, popen) -> None:
        if not _k32.AssignProcessToJobObject(self.handle, int(popen._handle)):
            raise ctypes.WinError(ctypes.get_last_error())

    def drain(self, timeout_ms: int = 0) -> List[Tuple[str, Optional[int]]]:
        """Messages queued on the port: (name, pid-or-None). Non-blocking by default."""
        msgs = []
        if not self.port:
            return msgs
        code, key, ov = wintypes.DWORD(), ctypes.c_size_t(), wintypes.LPVOID()
        wait = int(timeout_ms)
        while _k32.GetQueuedCompletionStatus(self.port, ctypes.byref(code), ctypes.byref(key),
                                             ctypes.byref(ov), wait):
            msgs.append((_PORT_MESSAGES.get(code.value, str(code.value)), ov.value))
            wait = 0
        return msgs

    def query_extended(self):
        info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        if not _k32.QueryInformationJobObject(self.handle, _JobObjectExtendedLimitInformation,
                                              ctypes.byref(info), ctypes.sizeof(info), None):
            raise ctypes.WinError(ctypes.get_last_error())
        return info

    def query_accounting(self):
        info = _JOBOBJECT_BASIC_ACCOUNTING_INFORMATION()
        if not _k32.QueryInformationJobObject(self.handle, _JobObjectBasicAccountingInformation,
                                              ctypes.byref(info), ctypes.sizeof(info), None):
            raise ctypes.WinError(ctypes.get_last_error())
        return info

    def terminate(self, exit_code: int = 1) -> None:
        if self.handle and not _k32.TerminateJobObject(self.handle, int(exit_code)):
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self) -> None:
        if self.port:
            _k32.CloseHandle(self.port)
            self.port = None
        if self.handle:
            _k32.CloseHandle(self.handle)   # KILL_ON_JOB_CLOSE reaps whatever is left
            self.handle = None


def _peak_working_set_bytes(popen) -> int:
    """Peak working set of the (possibly exited) root process from its open handle."""
    if not IS_WINDOWS:
        return 0
    try:
        pmc = _PROCESS_MEMORY_COUNTERS()
        pmc.cb = ctypes.sizeof(pmc)
        if _k32.K32GetProcessMemoryInfo(int(popen._handle), ctypes.byref(pmc), pmc.cb):
            return int(pmc.PeakWorkingSetSize)
    except Exception as exc:
        logger.debug("GetProcessMemoryInfo failed: %s", exc)
    return 0


# ------------------------------------------------------------- helpers --

def _import_psutil():
    try:
        import psutil
        return psutil
    except ImportError:
        return None


def _dir_bytes(dirs: List[str]) -> int:
    total = 0
    for d in dirs:
        for dp, _dn, files in os.walk(d):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(dp, f))
                except OSError:
                    pass
    return total


def _parse_result(text: str) -> Tuple[Optional[dict], Optional[str]]:
    for line in reversed(text.rstrip("\r\n").splitlines()):
        line = line.rstrip("\r")
        if line.startswith("RESULT "):
            try:
                obj = json.loads(line[len("RESULT "):])
            except json.JSONDecodeError as exc:
                logger.warning("child RESULT line is not JSON: %s", exc)
                return None, BAD_RESULT_ERROR + ": " + str(exc)
            if not isinstance(obj, dict):
                return None, BAD_RESULT_ERROR + ": not an object"
            return obj, None
    return None, NO_RESULT_ERROR


# ---------------------------------------------------------------- POSIX --

def posix_rlimit_env(memory_limit_mb, cpu_limit_s, output_cap_mb) -> str:
    """JSON for ``RLIMIT_ENV``: the limits the POSIX child applies to itself."""
    return json.dumps({"as_mb": int(memory_limit_mb or 0),
                       "cpu_s": float(cpu_limit_s or 0.0),
                       "fsize_mb": int(output_cap_mb or 0)})


def parse_rlimits(text: str) -> dict:
    """``{as_bytes, cpu_s, fsize_bytes}`` (None = unlimited); ``{}`` for empty/garbage."""
    if not text:
        return {}
    try:
        raw = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {}
    if not isinstance(raw, dict):
        return {}

    def _num(key):
        try:
            v = float(raw.get(key) or 0)
        except (TypeError, ValueError):
            return 0.0
        return v if v > 0 else 0.0

    as_mb, cpu, fs_mb = _num("as_mb"), _num("cpu_s"), _num("fsize_mb")
    return {"as_bytes": int(as_mb * 2**20) if as_mb else None,
            "cpu_s": int(math.ceil(cpu)) if cpu else None,
            "fsize_bytes": int(fs_mb * 2**20) if fs_mb else None}


def apply_rlimits_from_env(environ: Optional[dict] = None) -> dict:
    """POSIX child side (UNVERIFIED): apply ``RLIMIT_ENV`` limits to this process.

    The CLI child should call this first thing. Returns the applied plan,
    ``{}`` when the variable is absent or ``resource`` does not exist
    (Windows, where the job object does the work).
    """
    environ = os.environ if environ is None else environ
    plan = parse_rlimits(environ.get(RLIMIT_ENV, ""))
    if not plan:
        return {}
    try:
        import resource
    except ImportError:
        return {}
    applied = {}
    try:
        if plan["as_bytes"]:
            resource.setrlimit(resource.RLIMIT_AS, (plan["as_bytes"], plan["as_bytes"]))
            applied["as_bytes"] = plan["as_bytes"]
        if plan["cpu_s"]:
            resource.setrlimit(resource.RLIMIT_CPU, (plan["cpu_s"], plan["cpu_s"] + 5))
            applied["cpu_s"] = plan["cpu_s"]
        if plan["fsize_bytes"]:
            resource.setrlimit(resource.RLIMIT_FSIZE, (plan["fsize_bytes"], plan["fsize_bytes"]))
            applied["fsize_bytes"] = plan["fsize_bytes"]
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except (ValueError, OSError) as exc:
        logger.warning("setrlimit failed: %s", exc)
    return applied


def _posix_run(argv, work_dir, *, timeout_s, memory_limit_mb, cpu_limit_s, output_cap_mb,
               env=None, poll_s=0.1, watch_dirs=None) -> SandboxResult:
    """POSIX sandbox (UNVERIFIED: written, never run on Linux/macOS).

    Own session (``killpg`` reaches the tree unless a descendant calls setsid
    itself), rlimits through ``RLIMIT_ENV`` applied by the child, wall clock and
    output cap polled here. Note RLIMIT_AS limits virtual size, which OCC and
    numpy reserve generously.
    """
    return _run_core(argv, work_dir, posix=True, timeout_s=timeout_s, memory_limit_mb=memory_limit_mb,
                     cpu_limit_s=cpu_limit_s, output_cap_mb=output_cap_mb, env=env, poll_s=poll_s,
                     watch_dirs=watch_dirs)


# ------------------------------------------------------------ the worker --

def run_sandboxed(argv: List[str], work_dir: str, *, timeout_s: float, memory_limit_mb: int,
                  cpu_limit_s: float, output_cap_mb: int, env: Optional[dict] = None,
                  poll_s: float = 0.1, watch_dirs: Optional[List[str]] = None) -> SandboxResult:
    """Run ``argv`` with cwd ``work_dir`` under the limits; never raises for child failures.

    ``env`` defaults to ``sanitized_env(work_dir, <repo root>)``. ``watch_dirs``
    are extra directories (for example the output dir when it is outside the
    work dir) watched together with the work dir. ``output_cap_mb`` caps the
    GROWTH of all watched dirs: their total size just before the spawn is the
    baseline, so pre-placed input files and older outputs do not count. A
    limit of 0 / None disables that limit.
    """
    if not IS_WINDOWS:
        return _posix_run(argv, work_dir, timeout_s=timeout_s, memory_limit_mb=memory_limit_mb,
                          cpu_limit_s=cpu_limit_s, output_cap_mb=output_cap_mb, env=env,
                          poll_s=poll_s, watch_dirs=watch_dirs)
    return _run_core(argv, work_dir, posix=False, timeout_s=timeout_s, memory_limit_mb=memory_limit_mb,
                     cpu_limit_s=cpu_limit_s, output_cap_mb=output_cap_mb, env=env, poll_s=poll_s,
                     watch_dirs=watch_dirs)


def _new_section(timeout_s, memory_limit_mb, cpu_limit_s, output_cap_mb) -> dict:
    return {"enabled": True, "isolation": "none", "isolation_error": None,
            "timeout_s": float(timeout_s or 0.0), "memory_limit_mb": int(memory_limit_mb or 0),
            "cpu_limit_s": float(cpu_limit_s or 0.0), "output_cap_mb": int(output_cap_mb or 0),
            "peak_rss_mb": 0.0, "peak_commit_mb": None, "wall_s": 0.0, "cpu_s": None,
            "timed_out": False, "memory_exceeded": False, "cpu_exceeded": False,
            "output_exceeded": False, "killed": False, "exit_code": None, "pid": None,
            "stdout_truncated": False, "stderr_truncated": False, "result_error": None,
            "job_messages": [], "stderr_tail": ""}


def _run_core(argv, work_dir, *, posix, timeout_s, memory_limit_mb, cpu_limit_s, output_cap_mb,
              env, poll_s, watch_dirs) -> SandboxResult:
    work_dir = os.path.abspath(str(work_dir))
    os.makedirs(work_dir, exist_ok=True)
    sb = _new_section(timeout_s, memory_limit_mb, cpu_limit_s, output_cap_mb)
    if env is None:
        env = sanitized_env(work_dir, _DEFAULT_REPO_ROOT)
    env = dict(env)
    mem_bytes = int(memory_limit_mb) * 2**20 if memory_limit_mb else 0
    cap_bytes = int(output_cap_mb) * 2**20 if output_cap_mb else 0
    dirs = [work_dir] + [os.path.abspath(str(d)) for d in (watch_dirs or [])]
    psutil = _import_psutil()

    job = None
    popen_kw = dict(cwd=work_dir, env=env, stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if posix:
        env[RLIMIT_ENV] = posix_rlimit_env(memory_limit_mb, cpu_limit_s, output_cap_mb)
        popen_kw["start_new_session"] = True
    else:
        popen_kw["creationflags"] = subprocess.BELOW_NORMAL_PRIORITY_CLASS | subprocess.CREATE_NO_WINDOW
        try:
            job = WindowsJob(memory_limit_bytes=mem_bytes or None, cpu_limit_s=cpu_limit_s or None)
        except Exception as exc:     # OSError from ctypes, or anything odd
            sb["isolation_error"] = str(exc)
            logger.warning("job object unavailable, falling back to the psutil tree: %s", exc)
            job = None

    # the cap measures what the child adds, not what was there before it (the
    # CLI copies the input into <work>/in before the spawn)
    baseline_bytes = _dir_bytes(dirs) if cap_bytes else 0
    sb["output_baseline_mb"] = round(baseline_bytes / 2**20, 1)
    t0 = time.perf_counter()
    try:
        p = subprocess.Popen(list(argv), **popen_kw)
    except (OSError, ValueError) as exc:
        if job is not None:
            job.close()
        sb["result_error"] = "spawn failed: %s" % exc
        sb["exit_code"] = 1
        sb["wall_s"] = round(time.perf_counter() - t0, 3)
        return SandboxResult(exit_code=1, result=None, sandbox=sb, stdout_tail="", stderr_tail="")
    sb["pid"] = p.pid

    if posix:
        sb["isolation"] = "posix_rlimit"
    elif job is not None:
        try:
            job.assign(p)
            sb["isolation"] = "job_object"
        except Exception as exc:
            sb["isolation_error"] = str(exc)
            logger.warning("AssignProcessToJobObject failed, psutil fallback: %s", exc)
            job.close()
            job = None
    if not posix and job is None:
        sb["isolation"] = "psutil_tree" if psutil is not None else "none"

    ro, re_ = CappedReader(p.stdout), CappedReader(p.stderr)
    ro.start()
    re_.start()

    root = None
    if psutil is not None:
        try:
            root = psutil.Process(p.pid)
        except psutil.Error:
            root = None
    seen: Dict[int, float] = {}       # descendant pid -> create_time (orphan reaping)
    cpu_seen: Dict[int, float] = {}
    messages: List[str] = []
    peak_rss = 0

    def kill_tree():
        if job is not None:
            try:
                job.terminate(1)
                return
            except Exception as exc:
                logger.warning("TerminateJobObject failed: %s", exc)
        if posix:
            try:
                import signal
                os.killpg(p.pid, signal.SIGKILL)
                return
            except Exception as exc:
                logger.warning("killpg failed: %s", exc)
        if root is not None:
            _psutil_kill(psutil, root, seen)
        else:
            try:
                p.kill()
            except OSError:
                pass

    def drain(timeout_ms=0):
        if job is None:
            return
        try:
            for name, _pid in job.drain(timeout_ms):
                messages.append(name)
        except Exception as exc:
            logger.debug("port drain failed: %s", exc)

    while True:
        elapsed = time.perf_counter() - t0
        if p.poll() is not None:
            break
        rss = 0
        if root is not None:
            try:
                procs = [root] + root.children(recursive=True)
            except psutil.Error:
                procs = []
            for pr in procs:
                try:
                    if pr.pid != root.pid and pr.pid not in seen:
                        seen[pr.pid] = pr.create_time()
                    rss += pr.memory_info().rss
                    ct = pr.cpu_times()
                    cpu_seen[pr.pid] = ct.user + ct.system
                except psutil.Error:
                    pass
        peak_rss = max(peak_rss, rss)
        drain(0)
        reason = None
        if timeout_s and elapsed > timeout_s:
            reason = "timed_out"
        elif job is None and not posix and mem_bytes and rss > mem_bytes:
            reason = "memory_exceeded"       # fallback: polled RSS, no job to refuse allocations
        elif cap_bytes and _dir_bytes(dirs) - baseline_bytes > cap_bytes:
            reason = "output_exceeded"
        if reason:
            sb[reason] = True
            sb["killed"] = True
            logger.warning("sandbox: %s after %.1f s, killing the process tree", reason, elapsed)
            kill_tree()
            break
        try:
            p.wait(timeout=poll_s)
        except subprocess.TimeoutExpired:
            pass

    try:
        p.wait(timeout=10)
    except subprocess.TimeoutExpired:
        kill_tree()
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logger.error("child %d did not exit after kill", p.pid)
    sb["wall_s"] = round(time.perf_counter() - t0, 3)
    rc = p.returncode if p.returncode is not None else 1
    peak_rss = max(peak_rss, _peak_working_set_bytes(p))

    if job is not None:
        try:
            info, acc = job.query_extended(), job.query_accounting()
            sb["peak_commit_mb"] = round(info.PeakJobMemoryUsed / 2**20, 1)
            sb["cpu_s"] = round((acc.TotalUserTime + acc.TotalKernelTime) / 1e7, 3)
        except Exception as exc:
            logger.warning("job query failed: %s", exc)
        try:
            job.terminate(1)          # reaps orphaned grandchildren; harmless when empty
        except Exception as exc:
            logger.debug("final TerminateJobObject: %s", exc)
        t_end = time.perf_counter() + 0.5
        while "ACTIVE_PROCESS_ZERO" not in messages and time.perf_counter() < t_end:
            drain(50)
        job.close()
    else:
        if root is not None:
            _reap_orphans(psutil, seen)
        elif posix:
            try:
                import signal
                os.killpg(p.pid, signal.SIGKILL)
            except Exception:
                pass
        sb["cpu_s"] = round(sum(cpu_seen.values()), 3) if cpu_seen else None

    # the root has exited and the tree was terminated / reaped, so the pipes
    # close at once; a missed grandchild that inherited them would otherwise
    # hold each join for its full timeout
    ro.join(2.0)
    re_.join(2.0)
    if ro.is_alive() or re_.is_alive():
        logger.warning("a pipe is still held open by a leftover process; output may be incomplete")

    sb["exit_code"] = rc
    sb["peak_rss_mb"] = round(peak_rss / 2**20, 1)
    sb["stdout_truncated"] = ro.truncated
    sb["stderr_truncated"] = re_.truncated
    limit_msgs = []
    for m in messages:
        if m not in _ROUTINE_MESSAGES and m not in limit_msgs:
            limit_msgs.append(m)
    sb["job_messages"] = limit_msgs

    stderr_text = bytes(re_.head).decode("utf-8", "replace") + bytes(re_.tail).decode("utf-8", "replace")
    result, result_error = _parse_result(ro.text())
    if result is None and (rc & 0xFFFFFFFF) in NATIVE_CRASH_CODES:
        result_error = NATIVE_CRASH_ERROR
    sb["result_error"] = result_error

    if _MEMORY_MESSAGES.intersection(messages):
        sb["memory_exceeded"] = True
    elif sb["isolation"] != "job_object":
        # no completion port: fall back to text, but never against a child
        # that reported success (its log lines may mention MemoryError)
        child_ok = isinstance(result, dict) and result.get("ok") is True
        if not child_ok and (
                _STDERR_MEMORY_PATTERN.search(stderr_text)
                or (isinstance(result, dict)
                    and _RESULT_MEMORY_PATTERN.search(str(result.get("error") or "")))):
            sb["memory_exceeded"] = True
    if (rc & 0xFFFFFFFF) == STATUS_QUOTA_EXCEEDED or _CPU_MESSAGES.intersection(messages):
        sb["cpu_exceeded"] = True
    if posix:
        import signal
        if rc == -getattr(signal, "SIGXCPU", -10_000):
            sb["cpu_exceeded"] = True
        if rc == -getattr(signal, "SIGXFSZ", -10_000):
            sb["output_exceeded"] = True

    stderr_tail = re_.tail_text()
    sb["stderr_tail"] = stderr_tail
    logger.info("sandbox: exit %s, wall %.2f s, isolation %s, result_error %s",
                rc, sb["wall_s"], sb["isolation"], result_error)
    return SandboxResult(exit_code=int(rc), result=result, sandbox=sb,
                         stdout_tail=ro.tail_text(), stderr_tail=stderr_tail)


def _psutil_kill(psutil, root, seen: Dict[int, float]) -> None:
    """Kill root, its live descendants, and every remembered descendant still running."""
    procs = []
    try:
        procs = root.children(recursive=True)
    except psutil.Error:
        pass
    procs.append(root)
    procs.extend(_remembered_alive(psutil, seen, exclude={pr.pid for pr in procs}))
    for pr in procs:
        try:
            pr.kill()
        except psutil.Error:
            pass
    try:
        psutil.wait_procs(procs, timeout=5)
    except Exception:
        pass


def _remembered_alive(psutil, seen: Dict[int, float], exclude=()) -> list:
    alive = []
    for pid, ctime in seen.items():
        if pid in exclude:
            continue
        try:
            pr = psutil.Process(pid)
            if abs(pr.create_time() - ctime) < 1e-3:     # same process, not a reused PID
                alive.append(pr)
        except psutil.Error:
            pass
    return alive


def _reap_orphans(psutil, seen: Dict[int, float]) -> None:
    procs = _remembered_alive(psutil, seen)
    if not procs:
        return
    logger.warning("killing %d orphaned descendant(s) of the sandboxed child", len(procs))
    for pr in procs:
        try:
            pr.kill()
        except psutil.Error:
            pass
    try:
        psutil.wait_procs(procs, timeout=5)
    except Exception:
        pass
