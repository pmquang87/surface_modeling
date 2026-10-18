"""Command line of the drawing module: ``python -m src.drawing INPUT OUT_DIR [options]``.

Contract (the ``src/convert.py`` pattern):

* stdout carries exactly one line, ``RESULT {json}`` (the quality report),
  on success and on failure; progress and library output go to stderr
  (``--quiet`` silences progress except one ``src.drawing.stage`` line per
  stage boundary; sandboxed, the parent forwards only those lines of the
  child's stderr tail).
* exit 0 = drawing written, no warning flag; 2 = drawing written with
  warning flags; 1 = failed (usage errors included: ``_Parser`` turns
  argparse's exit 2 into 1, prints the usage on stderr and nothing on
  stdout).

Two modes:

* default (sandboxed): the parent sniffs the input and applies the size and
  triangle caps WITHOUT importing any geometry library, copies the input
  into a fresh ``tempfile.mkdtemp()`` work directory, writes the options to
  ``options.json`` there and runs ``python -m src.drawing <copy>
  <work>/out --no-sandbox --options-json <work>/options.json`` through
  ``sandbox.run_sandboxed`` (job object, wall timeout, memory limit, CPU
  backstop = 2 x timeout, output cap on the work directory, sanitized
  environment). The child's outputs are moved to OUT_DIR, the ``sandbox``
  section is merged into the result and into ``<stem>.report.json``, and
  the parent exits with the child's code (1 when the child was killed,
  timed out, breached a limit or printed no RESULT line). The child writes
  ``<work>/progress.json`` at every stage boundary
  (``report.PROGRESS_FILENAME``); a child without RESULT gets a failure
  report with its last stage, unit, geometry and flags merged in, the
  breach flag (``sandbox_timeout`` ...) and its wall time (QA issue 17); a
  STEP child that left no unit (killed before its first snapshot) gets the
  parent's ``brep_load.step_unit_sniff`` of the Part 21 text.
  Empty and truncated STL inputs are rejected in the parent with their own
  error texts (``pipeline.input_error_text`` = ``sniff.unreadable_reason``,
  QA issue 25).
* ``--no-sandbox``: ``pipeline.run`` in this process; the report says
  ``sandbox.enabled false``.

Writing the child's outputs into the work directory (not straight into
OUT_DIR) is what makes the work-directory output cap of the sandbox cover
the drawing files.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import shutil
import sys
import tempfile
from typing import List, Optional

from .options import (FORMATS, ORIENTS, PROJECTIONS, SECTIONS, SHEET_CHOICES, STL_TO_STEP,
                      STL_UNITS, UPS, VIEW_NAMES, DrawingOptions)

logger = logging.getLogger(__name__)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
#: logger of the pipeline's stage lines (``pipeline.stage_logger``), INFO even under --quiet
STAGE_LOGGER = "src.drawing.stage"


class _Parser(argparse.ArgumentParser):
    """argparse exits 2 on usage errors, which collides with the 'written
    with warnings' tier; a usage error means nothing was written -> 1."""

    def error(self, message):
        self.print_usage(sys.stderr)
        print(f"{self.prog}: error: {message}", file=sys.stderr)
        sys.exit(1)


def _csv(allowed):
    def parse(text: str):
        items = tuple(t.strip() for t in str(text).split(",") if t.strip())
        bad = [t for t in items if t not in allowed]
        if not items or bad:
            raise argparse.ArgumentTypeError(
                f"invalid entry {bad[0] if bad else text!r}; choose from {', '.join(allowed)}")
        return items
    return parse


_NO_CAP = "none"


def _cap(text: str):
    """``--max-hidden-primitives``: a non-negative integer, or 'none' (no cap)."""
    t = str(text).strip().lower()
    if t in ("none", "off"):
        return _NO_CAP
    try:
        n = int(t)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected an integer or 'none', got {text!r}") from None
    if n < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0 or 'none', got {n}")
    return n


def build_parser() -> _Parser:
    p = _Parser(prog="python -m src.drawing",
                description="STEP / STL to a 2D engineering drawing (PDF, SVG, DXF) plus a "
                            "JSON quality report.",
                epilog="stdout: exactly one line 'RESULT {json}'. exit 0 clean, 2 written "
                       "with quality warnings, 1 failed.")
    p.add_argument("input", help="input STEP (.step/.stp) or STL file (detected by content)")
    p.add_argument("out_dir", help="output directory")
    p.add_argument("--formats", type=_csv(FORMATS), default=None,
                   help="comma list of pdf,svg,dxf (default all three)")
    p.add_argument("--views", type=_csv(VIEW_NAMES), default=None,
                   help="comma list of views (default front,top,right,iso)")
    p.add_argument("--section", choices=SECTIONS, default=None,
                   help="section A-A plane (default auto: through the largest hole axis; none "
                        "for large assemblies, where front/top/right still draw one)")
    p.add_argument("--sheet", choices=SHEET_CHOICES, default=None, help="sheet size (default A3)")
    p.add_argument("--projection", choices=PROJECTIONS, default=None,
                   help="first (ISO, default) or third angle")
    p.add_argument("--orient", choices=ORIENTS, default=None,
                   help="auto: longest extent = width, shortest = depth (default); fixed: file axes")
    p.add_argument("--up", choices=UPS, default=None, help="up axis of the file (default z)")
    p.add_argument("--stl-to-step", dest="stl_to_step", choices=STL_TO_STEP, default=None,
                   help="STL route: auto (faceted STEP for prismatic meshes), faceted, nurbs, off")
    p.add_argument("--stl-unit", dest="stl_unit", choices=STL_UNITS, default=None,
                   help="unit of an STL file (STL carries none; default: read as mm and "
                        "flagged unit_assumed_mm); the mesh is scaled to mm")
    p.add_argument("--max-hidden-primitives", dest="max_hidden_primitives", type=_cap,
                   default=None, metavar="N",
                   help="drop the hidden lines of a B-rep view with more than N raw hidden "
                        "primitives (flag hidden_lines_dropped; default 20000; 'none' = no cap)")
    p.add_argument("--no-hidden", dest="no_hidden", action="store_true", help="no hidden lines")
    p.add_argument("--hidden-lines-force", dest="hidden_lines_force", action="store_true",
                   help="keep hidden lines on large assemblies (more than 20 solids or 50000 "
                        "raw hidden edges per view; default: omitted, flag "
                        "hidden_lines_omitted_assembly)")
    p.add_argument("--no-hole-positions", dest="no_hole_positions", action="store_true",
                   help="no hole position dimensions")
    p.add_argument("--title", default=None, help="drawing title (default: file stem)")
    p.add_argument("--owner", default=None, help="legal owner field of the title block")
    p.add_argument("--timeout", dest="timeout_s", type=float, default=None,
                   help="sandbox wall-clock limit in seconds (default 180, minimum 15)")
    p.add_argument("--memory-limit-mb", dest="memory_limit_mb", type=int, default=None,
                   help="sandbox memory limit in MB (default 2048)")
    p.add_argument("--no-sandbox", dest="no_sandbox", action="store_true",
                   help="run in this process (no child, no limits)")
    p.add_argument("--options-json", dest="options_json", default=None, metavar="FILE",
                   help="DrawingOptions as JSON; explicit flags override it")
    p.add_argument("--quiet", action="store_true",
                   help="no progress on stderr except one 'src.drawing.stage' line per stage")
    return p


def options_from_args(args, parser: _Parser) -> DrawingOptions:
    base = DrawingOptions()
    if args.options_json:
        try:
            with open(args.options_json, "r", encoding="utf-8") as fh:
                base = DrawingOptions.from_json(fh.read())
        except (OSError, ValueError) as exc:
            parser.error(f"--options-json: {exc}")
    kw = {}
    for name in ("formats", "views", "section", "sheet", "projection", "orient", "up",
                 "stl_to_step", "title", "owner", "timeout_s", "memory_limit_mb", "stl_unit",
                 "max_hidden_primitives"):
        v = getattr(args, name)
        if v is not None:
            kw[name] = None if (name == "max_hidden_primitives" and v == _NO_CAP) else v
    if args.no_hidden:
        kw["hidden_lines"] = False
    if args.no_hole_positions:
        kw["hole_positions"] = False
    if args.hidden_lines_force:
        kw["hidden_lines_force"] = True
    if args.quiet:
        kw["quiet"] = True
    try:
        opts = base.with_overrides(**kw)
        opts.validate()
    except ValueError as exc:
        parser.error(str(exc))
    return opts


def _setup_logging(quiet: bool) -> None:
    root = logging.getLogger()
    if not any(getattr(h, "_drawing_cli", False) for h in root.handlers):
        h = logging.StreamHandler(sys.stderr)
        h._drawing_cli = True                    # type: ignore[attr-defined]
        h.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        root.addHandler(h)
    root.setLevel(logging.ERROR if quiet else logging.INFO)
    # one line per pipeline stage boundary even under --quiet: the sandbox keeps
    # the child's stderr tail, which then shows where a killed run died (F7-perf)
    logging.getLogger(STAGE_LOGGER).setLevel(logging.INFO)


@contextlib.contextmanager
def _stdout_to_stderr(quiet: bool):
    """Libraries may ``print``; stdout is reserved for the RESULT line."""
    if quiet:
        with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
            yield
    else:
        with contextlib.redirect_stdout(sys.stderr):
            yield


def _emit(result: dict, stdout) -> None:
    stdout.write("RESULT " + json.dumps(result, ensure_ascii=False, default=str) + "\n")
    stdout.flush()


def _parent_step_unit(input_path: str, sniffed, current) -> Optional[str]:
    """The STEP unit for a killed child's report when its progress gave none.

    Only for a sniffed STEP input whose ``source_unit`` is still unset
    (a child that died before its first progress snapshot, e.g. during the
    library imports): ``brep_load.step_unit_sniff`` (Part 21 text, OCP-free;
    the module pulls numpy, so it is imported only on this failure path).
    None when not applicable or when the sniff fails (QA issue 17 step 2).
    """
    if current not in (None, "", "unknown") or getattr(sniffed, "kind", None) != "step":
        return None
    try:
        from .brep_load import step_unit_sniff
        return step_unit_sniff(input_path)
    except Exception as exc:                   # never lose the failure report over it
        logger.error("STEP unit sniff failed: %s", exc)
        return None


def _fail_report(input_path: str, out_dir: str, opts: DrawingOptions, error: str,
                 sniffed=None, sandbox: Optional[dict] = None, progress: Optional[dict] = None,
                 merge_progress: bool = False):
    """A failure report built in the parent (no geometry library imported).

    ``merge_progress`` (a sandboxed child that printed no RESULT): the
    child's last progress snapshot ``progress`` (None when it wrote none) is
    merged with ``Report.merge_progress``: stage, unit, geometry, flags,
    the breach flag (``sandbox_timeout`` ...) with the last stage, and
    ``timing.total_s`` = the child's wall time (QA issue 17).
    """
    from .report import Report
    rep = Report(input_path, out_dir, opts)
    if sniffed is not None:
        rep.data["input"].update(detected_format=sniffed.kind, size_bytes=sniffed.size_bytes,
                                 sha256=sniffed.sha256 or None,
                                 triangles_declared=sniffed.triangles,
                                 extension=sniffed.extension or None)
    if sandbox is not None:
        rep.data["sandbox"] = dict(sandbox)
    if merge_progress:
        try:
            rep.merge_progress(progress, sandbox)
        except Exception as exc:               # never lose the failure report over it
            logger.error("progress merge failed: %s", exc)
        unit = _parent_step_unit(input_path, sniffed, rep.data["input"].get("source_unit"))
        if unit is not None:
            rep.data["input"]["source_unit"] = unit
    code, data = rep.finish(False, error)
    try:
        rep.write(os.path.join(out_dir, f"{rep.stem}.report.json"))
        data = rep.data
    except Exception as exc:
        logger.error("report could not be written: %s", exc)
    return code, data


def _move_outputs(result: dict, out_dir: str, staging: str) -> None:
    """Move the child's files from the staging dir to ``out_dir``; fix the paths."""
    os.makedirs(out_dir, exist_ok=True)
    outputs = result.get("outputs") or {}
    staging_abs = os.path.abspath(staging)
    for key, path in list(outputs.items()):
        if not path:
            continue
        src = os.path.abspath(str(path))
        if os.path.dirname(src) != staging_abs:
            continue                           # never move anything outside the staging dir
        dst = os.path.abspath(os.path.join(out_dir, os.path.basename(src)))
        try:
            if os.path.isfile(src):
                shutil.move(src, dst)
                outputs[key] = dst
            else:
                outputs[key] = None
        except OSError as exc:
            logger.error("could not move %s to %s: %s", src, dst, exc)
            outputs[key] = None


_BREACHES = (("timed_out", "sandbox_timeout"), ("memory_exceeded", "sandbox_memory_exceeded"),
             ("cpu_exceeded", "sandbox_cpu_exceeded"), ("output_exceeded", "sandbox_output_exceeded"),
             ("killed", "sandbox_killed"))


def _add_breach_flag(quality: dict, box: dict) -> None:
    """A child that printed RESULT but breached a limit: add the breach flag once."""
    from .report import FLAGS
    flags = quality.setdefault("flags", [])
    if not isinstance(flags, list):
        return
    for key, code in _BREACHES:
        if box.get(key):
            if not any(isinstance(f, dict) and f.get("code") == code for f in flags):
                flags.append({"code": code, "severity": FLAGS.get(code, "warning"),
                              "detail": f"limit breached after the RESULT line ({key})"})
            return


def _stderr_to_forward(tail: Optional[str], quiet: bool) -> str:
    """The part of the child's stderr tail the parent prints: all of it, or under
    ``--quiet`` only the ``src.drawing.stage`` lines (QA round 3 issue 30: the docs
    promise one stage line per stage boundary on stderr under --quiet, and a sandboxed
    --quiet run printed none; they stayed in ``sandbox.stderr_tail``)."""
    if not tail:
        return ""
    if not quiet:
        return str(tail)
    keep = [ln for ln in str(tail).splitlines() if STAGE_LOGGER in ln]
    return "\n".join(keep)


def _run_sandboxed(input_path: str, out_dir: str, opts: DrawingOptions, stderr) -> tuple:
    from .pipeline import input_error_text          # standard library only at import
    from .report import PROGRESS_FILENAME, write_report_json
    from .sniff import check_limits, sniff

    if not os.path.isfile(input_path):
        return _fail_report(input_path, out_dir, opts, f"input file not found: {input_path}")
    try:
        s = sniff(input_path, max_bytes=int(opts.max_input_mb * 2 ** 20))
    except OSError as exc:
        return _fail_report(input_path, out_dir, opts, f"input unreadable: {exc}")
    # empty / truncated STL get their own text (QA issue 25), the rest the generic one
    err = check_limits(s, opts.max_input_mb, opts.max_triangles) or input_error_text(s)
    if err:
        return _fail_report(input_path, out_dir, opts, err, sniffed=s)

    from . import sandbox as sb

    work = tempfile.mkdtemp(prefix="drawing_")
    try:
        in_dir = os.path.join(work, "in")
        staging = os.path.join(work, "out")
        os.makedirs(in_dir)
        os.makedirs(staging)
        local_in = os.path.join(in_dir, os.path.basename(input_path))
        shutil.copyfile(input_path, local_in)
        opts_path = os.path.join(work, "options.json")
        with open(opts_path, "w", encoding="utf-8") as fh:
            fh.write(opts.to_json())
        argv = [sys.executable, "-m", "src.drawing", local_in, staging, "--no-sandbox",
                "--options-json", opts_path]
        if opts.quiet:
            argv.append("--quiet")
        env = sb.sanitized_env(work, REPO_ROOT)
        res = sb.run_sandboxed(argv, work, timeout_s=float(opts.timeout_s),
                               memory_limit_mb=int(opts.memory_limit_mb),
                               cpu_limit_s=2.0 * float(opts.timeout_s),
                               output_cap_mb=int(opts.output_cap_mb), env=env,
                               progress_file=PROGRESS_FILENAME)
        box = dict(res.sandbox or {})
        box["enabled"] = True
        tail = _stderr_to_forward(res.stderr_tail, opts.quiet)
        if tail:
            stderr.write(tail if tail.endswith("\n") else tail + "\n")
            stderr.flush()
        result = res.result if isinstance(res.result, dict) else None
        reason = res.failure_reason()
        # a clean child failure (RESULT ok false, exit 1) keeps the child's own error
        if (reason is not None and result is not None and res.exit_code == 1
                and result.get("ok") is False and not any(
                    box.get(k) for k in ("timed_out", "memory_exceeded", "cpu_exceeded",
                                         "output_exceeded", "killed"))):
            reason = None
        if result is None:
            # the child was killed or crashed: keep what its progress file says
            return _fail_report(input_path, out_dir, opts, f"sandbox: {reason or 'no RESULT'}",
                                sniffed=s, sandbox=box, progress=getattr(res, "progress", None),
                                merge_progress=True)
        _move_outputs(result, out_dir, staging)
        result.setdefault("input", {})["path"] = os.path.abspath(input_path)
        result["sandbox"] = box
        code = int(res.exit_code)
        if reason is not None:                 # limit breach or odd exit code: the run failed
            child_err = result.get("error")
            result["ok"] = False
            result["error"] = f"sandbox: {reason}" + (f"; child: {child_err}" if child_err else "")
            result["exit_code"] = code = 1
            q = result.setdefault("quality", {})
            q["level"] = "none"
            _add_breach_flag(q, box)
        elif code != int(result.get("exit_code", code)):
            logger.warning("child exit code %s differs from its RESULT exit_code %s",
                           code, result.get("exit_code"))
        stem = os.path.splitext(os.path.basename(input_path))[0] or "drawing"
        report_path = os.path.abspath(os.path.join(out_dir, f"{stem}.report.json"))
        result.setdefault("outputs", {})["report"] = report_path
        write_report_json(result, report_path)
        return code, result
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main(argv: Optional[List[str]] = None) -> int:
    real_stdout = sys.stdout
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
    parser = build_parser()
    args = parser.parse_args(argv)
    opts = options_from_args(args, parser)
    _setup_logging(opts.quiet)
    input_path = os.path.abspath(args.input)
    out_dir = os.path.abspath(args.out_dir)
    stderr = sys.stderr

    with _stdout_to_stderr(opts.quiet):
        try:
            if args.no_sandbox:
                from .pipeline import run

                def say(msg):
                    if not opts.quiet:
                        print(f"[drawing] {msg}", file=stderr, flush=True)

                code, result = run(input_path, out_dir, opts, log=say)
            else:
                code, result = _run_sandboxed(input_path, out_dir, opts, stderr)
        except Exception as exc:  # noqa: BLE001 - the contract is RESULT, not a traceback
            logger.exception("drawing CLI failed")
            code, result = _fail_report(input_path, out_dir, opts, f"{type(exc).__name__}: {exc}")
    _emit(result, real_stdout)
    return int(code)


__all__ = ["main", "build_parser", "options_from_args"]
