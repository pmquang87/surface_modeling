"""Quality report of one drawing run (schema ``surface_modeling.drawing.report/1``).

``Report`` holds the JSON document of spec 3.8, pre-filled with nulls so
every key exists even when a run fails early, and computes the exit code
and the quality level at the end:

* exit code (the ``src/convert.py`` contract): 0 = drawing written, no
  warning flag; 2 = drawing written, at least one ``warning`` flag (look at
  it); 1 = failed (``ok`` False, ``error`` set).
* quality level: ``full`` = B-rep route with hidden lines, a section with
  loops and at least one linear (overall) dimension, and no view drawn by
  the polygonal HLR; ``reduced`` = mesh route, or a B-rep route that lost
  the hidden lines, the section or the dimensions, or used the poly HLR;
  ``minimal`` = views only (no dimension, no section); ``none`` = the run
  failed and no drawing exists (an addition to the spec's three levels: a
  failed run has no views, so even ``minimal`` would overstate it).

Severities: ``FLAGS`` maps every flag code to its severity (spec 3.8 plus
this implementation's additions); ``WARNING_FLAGS`` / ``INFO_FLAGS`` are
derived from it. ``flag(code)`` takes the table severity when none is
passed; an unknown code needs an explicit severity and is logged, and a
severity that disagrees with the table is logged (a typo would otherwise
change the exit code silently), except the info -> warning raise allowed
for ``ESCALATABLE_FLAGS``. ``approval_blank`` is listed as a warning in
the spec but is ``info`` here: the approval field of an auto-generated
drawing is ALWAYS blank, so a warning would make every run exit 2 and the
exit code would carry no information. ``not_a_solid`` caps the level at
``minimal``.

Progress (sandboxed runs): the child writes ``progress_snapshot(stage)``
with ``write_progress`` to the path in ``PROGRESS_ENV`` at each stage
boundary; when it is killed, the parent's failure report takes the last
snapshot with ``merge_progress`` (stage, unit, geometry, flags, the
``sandbox_timeout``-type flag and the child's wall time).

Import cost: standard library only (json, os, time), so the sandbox parent
can build a failure report before any geometry library is imported.
"""
from __future__ import annotations

import copy
import json
import logging
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

SCHEMA = "surface_modeling.drawing.report/1"

#: Every quality flag code with its severity: the single source of truth.
#: ``warning`` = look at the drawing (exit code 2 when the run succeeded);
#: ``info`` = disclosed, the exit code stays 0.
FLAGS: Dict[str, str] = {
    # ---- spec 3.8 ----
    "mesh_not_watertight": "warning",
    "multiple_bodies": "warning",          # STEP and STL routes
    "brepcheck_invalid": "warning",
    "open_shell": "warning",
    "inverted_solid": "warning",
    "fallback_to_mesh": "warning",
    "hlr_timeout": "warning",
    "section_empty": "warning",
    "section_open_wires": "warning",
    "views_overflow": "warning",
    "dimensions_capped": "warning",
    "unit_not_mm": "warning",
    "sheet_not_iso5457": "warning",
    "holes_not_detected_mesh_route": "info",
    "organic_classified": "info",
    "too_many_triangles": "info",
    "extension_mismatch": "info",
    "facet_edges_suppressed": "info",
    "hlr_poly_used": "info",
    "section_nudged": "info",
    "degenerate_faces_removed": "info",
    # the spec lists it as a warning; the approval field of an auto-generated
    # drawing is ALWAYS blank, so a warning would make every run exit 2
    "approval_blank": "info",
    # ---- additions of this implementation: a part of the drawing is missing ----
    "view_failed": "warning",
    "section_failed": "warning",
    "writer_failed": "warning",
    "dimensions_failed": "warning",
    "faceted_audit_mismatch": "warning",
    # ---- QA round 1 ----
    "loose_curves_ignored": "info",        # free edges/vertices (sketch/PMI curve sets) not drawn
    "supplemental_shells_ignored": "info",  # free shells/faces next to solids not drawn
    "supplemental_geometry_ignored": "info",
    "loose_curves_drawn": "info",          # pure wireframe / surface model keeps them
    "dimension_extent_mismatch": "warning",
    "unit_assumed_mm": "info",             # STL carries no unit; may be raised to warning
    "unit_suspect": "info",                # sizes look like another unit; may be raised
    "views_tiny": "warning",
    "step_tessellated_only": "warning",
    "hidden_lines_dropped": "warning",
    "not_a_solid": "warning",              # level minimal
    "mesh_non_manifold": "warning",
    "section_fallback": "info",
    "interior_silhouettes_omitted": "info",
    "layout_text_collision": "info",       # layout.py: labels/dimension texts overlap
    # sandbox breaches (the run failed; exit 1 regardless of the severity)
    "sandbox_timeout": "warning",
    "sandbox_memory_exceeded": "warning",
    "sandbox_cpu_exceeded": "warning",
    "sandbox_output_exceeded": "warning",
    "sandbox_killed": "warning",
}
#: Codes whose default severity is ``info`` but which a caller may raise to
#: ``warning`` (for example ``unit_assumed_mm`` when the size is implausible).
ESCALATABLE_FLAGS = frozenset({"unit_assumed_mm", "unit_suspect"})
WARNING_FLAGS = frozenset(c for c, s in FLAGS.items() if s == "warning")
INFO_FLAGS = frozenset(c for c, s in FLAGS.items() if s == "info")
SEVERITIES = ("info", "warning")
LEVELS = ("full", "reduced", "minimal", "none")
#: Flags that cap the level at ``minimal`` (the drawing is not of a solid part).
MINIMAL_LEVEL_FLAGS = frozenset({"not_a_solid"})

#: Progress file of a sandboxed child (written at each stage boundary; read by
#: the parent when the child produced no RESULT). ``sandbox.run_sandboxed``
#: passes its absolute path to the child in this environment variable.
PROGRESS_ENV = "SRC_DRAWING_PROGRESS_FILE"
PROGRESS_FILENAME = "progress.json"
_BREACH_FLAGS = (("timed_out", "sandbox_timeout"), ("memory_exceeded", "sandbox_memory_exceeded"),
                 ("cpu_exceeded", "sandbox_cpu_exceeded"),
                 ("output_exceeded", "sandbox_output_exceeded"), ("killed", "sandbox_killed"))
_MAX_PROGRESS_FLAGS = 200
_MAX_TEXT = 500


def flag_severity(code: str) -> Optional[str]:
    """The table severity of ``code``, or None for an unknown code."""
    return FLAGS.get(str(code))


def _sandbox_disabled() -> Dict[str, Any]:
    return {"enabled": False, "isolation": "none", "timeout_s": None, "memory_limit_mb": None,
            "cpu_limit_s": None, "peak_rss_mb": None, "peak_commit_mb": None, "wall_s": None,
            "cpu_s": None, "timed_out": False, "memory_exceeded": False, "cpu_exceeded": False,
            "output_exceeded": False, "killed": False, "exit_code": None,
            "stdout_truncated": False, "stderr_truncated": False, "result_error": None,
            "stderr_tail": ""}


def _versions() -> Dict[str, Optional[str]]:
    """Package versions from the installed metadata (nothing is imported)."""
    from importlib import metadata

    def ver(*dists):
        for d in dists:
            try:
                return metadata.version(d)
            except Exception:
                continue
        return None

    try:
        from . import __version__ as module_version
    except Exception:  # pragma: no cover - package always has it
        module_version = None
    return {"module": module_version, "ocp": ver("cadquery-ocp", "cadquery_ocp", "OCP"),
            "trimesh": ver("trimesh"), "ezdxf": ver("ezdxf")}


def _jsonable(value):
    """Plain JSON types: numpy scalars/arrays -> Python, NaN/inf -> None, tuples -> lists."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in value]
    tolist = getattr(value, "tolist", None)       # numpy arrays and scalars
    if callable(tolist):
        return _jsonable(tolist())
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _jsonable(item())
        except Exception:
            pass
    return str(value)


def _short(value, n: int = _MAX_TEXT) -> Optional[str]:
    """A capped string, or None for anything that is not a string."""
    return value[:n] if isinstance(value, str) else None


def _plain(value, depth: int = 0) -> bool:
    """True for small plain JSON data (numbers, strings, bools, short lists/dicts of them)."""
    if value is None or isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    if isinstance(value, str):
        return len(value) <= _MAX_TEXT
    if depth >= 2:
        return False
    if isinstance(value, list):
        return len(value) <= 64 and all(_plain(v, depth + 1) for v in value)
    if isinstance(value, dict):
        return len(value) <= 64 and all(isinstance(k, str) and _plain(v, depth + 1)
                                        for k, v in value.items())
    return False


def write_progress(path: Optional[str], snapshot: dict) -> bool:
    """Atomically write a progress snapshot (``Report.progress_snapshot``).

    For the pipeline at each stage boundary; ``path`` usually comes from
    ``progress_path_from_env()``. Returns False (and logs at debug level)
    instead of raising: progress is best effort and must never fail a run.
    """
    if not path:
        return False
    try:
        path = os.path.abspath(str(path))
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(_jsonable(snapshot), fh, ensure_ascii=False)
        os.replace(tmp, path)
        return True
    except Exception as exc:
        log.debug("progress file %s not written: %s", path, exc)
        return False


def progress_path_from_env(environ: Optional[dict] = None) -> Optional[str]:
    """The progress file path the sandbox parent passed (``PROGRESS_ENV``), or None."""
    environ = os.environ if environ is None else environ
    value = environ.get(PROGRESS_ENV) or ""
    return value or None


class Report:
    """The quality report of one run; ``data`` is the JSON document."""

    def __init__(self, input_path: str, out_dir: str, options) -> None:
        self.t0 = time.perf_counter()
        self._total_s_override: Optional[float] = None
        self.options = options
        stem = os.path.splitext(os.path.basename(str(input_path)))[0] or "drawing"
        self.stem = stem
        self.out_dir = str(out_dir)
        mode = getattr(options, "stl_to_step", None)
        orient = getattr(options, "orient", None)
        self.data: Dict[str, Any] = {
            "schema": SCHEMA,
            "ok": False, "exit_code": 1, "error": None,
            "input": {"path": os.path.abspath(str(input_path)), "detected_format": None,
                      "extension": os.path.splitext(str(input_path))[1].lower() or None,
                      "size_bytes": None, "sha256": None, "triangles_declared": None,
                      "source_unit": None},
            "route": {"pipeline": None, "source": None,
                      "classifier": {"verdict": "n/a", "planar_big": None, "sharp_turn": None,
                                     "threshold": getattr(options, "classifier_threshold", None)},
                      "stl_to_step": {"attempted": False, "mode": mode, "ok": None, "seconds": None,
                                      "skipped_reason": None, "error": None, "audit": {}},
                      "fallbacks": [],
                      "orientation": {"permutation": None, "flips": None, "mode": orient}},
            "geometry": {"bbox_min": None, "bbox_max": None, "extents": None,
                         "drawing_bbox_min": None, "drawing_bbox_max": None,
                         "volume_mm3": None, "area_mm2": None, "solids": None, "shells": None,
                         "faces": None, "edges": None, "face_types": None,
                         "brepcheck_valid": None, "free_edges": None, "closed": None,
                         "triangles": None, "watertight": None, "components": None,
                         "features": {"holes": None, "bosses": None, "fillets": None}},
            "views": [],
            "dimensions": [],
            "undimensioned": [],
            "section": None,
            "sheet": {"size": None, "orientation": None, "scale": None, "projection": None,
                      "iso5457": None},
            "outputs": {"pdf": None, "svg": None, "dxf": None, "step": None, "report": None},
            "quality": {"level": None, "flags": [], "last_stage": None},
            "warnings": [],
            "timing": {"total_s": None, "load_s": None, "views_s": None, "section_s": None,
                       "write_s": None},
            "sandbox": _sandbox_disabled(),
            "version": _versions(),
        }

    # ------------------------------------------------------------ events --
    def flag(self, code: str, severity: Optional[str] = None, detail: str = "") -> None:
        """Add a quality flag; an exact duplicate (code, severity, detail) is ignored.

        ``severity`` defaults to the ``FLAGS`` table entry. An unknown code
        needs an explicit severity (ValueError otherwise) and is logged, so a
        typo is visible without failing the drawing run; a severity that
        disagrees with the table is kept but logged (except the allowed
        ``ESCALATABLE_FLAGS`` info -> warning raise).
        """
        code = str(code)
        expected = FLAGS.get(code)
        if severity is None:
            if expected is None:
                raise ValueError(f"unknown flag code {code!r} and no severity given")
            severity = expected
        if severity not in SEVERITIES:
            raise ValueError(f"flag severity must be one of {SEVERITIES}, got {severity!r}")
        if expected is None:
            log.warning("report: flag code %r is not in report.FLAGS", code)
        elif expected != severity and not (code in ESCALATABLE_FLAGS and severity == "warning"):
            log.warning("report: flag %s raised as %s, the table says %s", code, severity, expected)
        entry = {"code": str(code), "severity": severity, "detail": str(detail or "")}
        if entry not in self.data["quality"]["flags"]:
            self.data["quality"]["flags"].append(entry)
            log.info("flag %s (%s) %s", code, severity, detail)

    def warn(self, text: str) -> None:
        """Human-readable warning (does not change the exit code by itself)."""
        text = str(text)
        if text not in self.data["warnings"]:
            self.data["warnings"].append(text)
            log.warning("%s", text)

    def fallback(self, frm: str, to: str, reason: str) -> None:
        self.data["route"]["fallbacks"].append({"from": str(frm), "to": str(to),
                                                "reason": str(reason)})
        log.warning("fallback %s -> %s: %s", frm, to, reason)

    def has_flag(self, code: str) -> bool:
        return any(f["code"] == code for f in self.data["quality"]["flags"])

    # ---------------------------------------------------------- progress --
    def progress_snapshot(self, stage: str, view: Optional[str] = None) -> dict:
        """The child's state at a stage boundary, for ``write_progress``.

        Keys: ``stage``, ``view``, ``t_s`` (seconds since the report
        started), ``source_unit``, ``detected_format``, ``route``
        (pipeline/source), ``geometry``, ``flags``, ``warnings``, ``timing``.
        """
        d = self.data
        return _jsonable({
            "stage": str(stage), "view": None if view is None else str(view),
            "t_s": round(time.perf_counter() - self.t0, 3),
            "source_unit": d["input"].get("source_unit"),
            "detected_format": d["input"].get("detected_format"),
            "route": {"pipeline": d["route"].get("pipeline"), "source": d["route"].get("source")},
            "geometry": d.get("geometry"),
            "flags": d["quality"]["flags"], "warnings": d.get("warnings"),
            "timing": d.get("timing"),
        })

    def merge_progress(self, progress: Optional[dict], sandbox: Optional[dict] = None) -> None:
        """Merge a killed child's last progress snapshot into this (failure) report.

        ``progress`` is the dict ``sandbox.run_sandboxed`` returns as
        ``SandboxResult.progress`` (untrusted: only known keys with plain
        types are taken, strings are capped). ``sandbox`` is the report's
        sandbox section; when a limit was breached the matching flag
        (``sandbox_timeout`` and so on) is added with the last stage in the
        detail, and ``timing.total_s`` becomes the child's wall time.
        Sets ``quality.last_stage``. Never raises on malformed input.
        """
        p = progress if isinstance(progress, dict) else {}
        d = self.data
        stage = _short(p.get("stage"))
        view = _short(p.get("view"))
        d["quality"]["last_stage"] = stage
        for key in ("source_unit", "detected_format"):
            if isinstance(p.get(key), str) and d["input"].get(key) in (None, "", "unknown"):
                d["input"][key] = _short(p[key])
        route = p.get("route")
        if isinstance(route, dict):
            for key in ("pipeline", "source"):
                if isinstance(route.get(key), str) and not d["route"].get(key):
                    d["route"][key] = _short(route[key])
        geom = p.get("geometry")
        if isinstance(geom, dict):
            for key, val in geom.items():
                if key in d["geometry"] and d["geometry"][key] is None and _plain(val):
                    d["geometry"][key] = val
        timing = p.get("timing")
        if isinstance(timing, dict):
            for key in ("load_s", "views_s", "section_s", "write_s"):
                val = timing.get(key)
                if isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(val):
                    d["timing"][key] = float(val)
        flags = p.get("flags")
        if isinstance(flags, list):
            for f in flags[:_MAX_PROGRESS_FLAGS]:
                if not isinstance(f, dict):
                    continue
                code, sev = f.get("code"), f.get("severity")
                if (isinstance(code, str) and 0 < len(code) <= 64 and sev in SEVERITIES
                        and all(c.isalnum() or c == "_" for c in code)):
                    self.flag(code, sev, _short(f.get("detail")) or "")
        warns = p.get("warnings")
        if isinstance(warns, list):
            for w in warns[:50]:
                if isinstance(w, str):
                    self.warn(_short(w))
        sb = sandbox if isinstance(sandbox, dict) else {}
        where = f"last stage: {stage or 'unknown'}" + (f", view {view}" if view else "")
        if not p:
            where += " (no progress file)"
        for key, code in _BREACH_FLAGS:
            if sb.get(key):
                limit = {"timed_out": sb.get("timeout_s"), "memory_exceeded": sb.get("memory_limit_mb"),
                         "cpu_exceeded": sb.get("cpu_limit_s"),
                         "output_exceeded": sb.get("output_cap_mb")}.get(key)
                self.flag(code, None, where + (f"; limit {limit}" if limit else ""))
                break                                  # "killed" accompanies every breach
        wall = sb.get("wall_s")
        if isinstance(wall, (int, float)) and not isinstance(wall, bool) and math.isfinite(wall) \
                and wall > 0:
            self._total_s_override = round(float(wall), 3)

    # ------------------------------------------------------------ result --
    def _level(self, ok: bool) -> str:
        d = self.data
        if not ok:
            return "none"
        views = d.get("views") or []
        if any(self.has_flag(c) for c in MINIMAL_LEVEL_FLAGS):
            return "minimal"
        section = d.get("section") or {}
        has_section = bool(section) and int(section.get("loops") or 0) > 0
        has_dims = any(x.get("kind") == "linear" for x in (d.get("dimensions") or []))
        if not has_section and not d.get("dimensions"):
            return "minimal"
        hidden = bool(getattr(self.options, "hidden_lines", True))
        poly = any(v.get("algorithm") == "hlr_poly" for v in views)
        if (d["route"].get("pipeline") == "brep" and hidden and has_section and has_dims
                and not poly and views):
            return "full"
        return "reduced"

    def finish(self, ok: bool, error: Optional[str] = None) -> Tuple[int, dict]:
        """Set ok / error / exit code / level / total time; returns ``(exit_code, data)``.

        A warning flag gives exit 2 only when ``ok``; a failed run is always 1.
        """
        d = self.data
        d["ok"] = bool(ok)
        if not ok:
            d["error"] = str(error) if error else (d.get("error") or "drawing failed")
            code = 1
        else:
            d["error"] = None
            warn = any(f["severity"] == "warning" for f in d["quality"]["flags"])
            code = 2 if warn else 0
        d["exit_code"] = code
        d["quality"]["level"] = self._level(bool(ok))
        total = round(time.perf_counter() - self.t0, 3)
        if self._total_s_override is not None:
            total = self._total_s_override         # the sandboxed child's wall time
        d["timing"]["total_s"] = total
        self.data = _jsonable(d)
        return code, self.data

    def to_json(self) -> str:
        return json.dumps(_jsonable(self.data), ensure_ascii=False, indent=2, sort_keys=False)

    def write(self, path: str) -> None:
        """Write the report as UTF-8 JSON (atomic replace); sets ``outputs.report``."""
        path = os.path.abspath(str(path))
        self.data["outputs"]["report"] = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(self.to_json())
            fh.write("\n")
        os.replace(tmp, path)

    def copy_data(self) -> dict:
        return copy.deepcopy(self.data)


def write_report_json(data: dict, path: str) -> None:
    """Write an already finished report dict (used by the sandbox parent)."""
    path = os.path.abspath(str(path))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(_jsonable(data), fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    os.replace(tmp, path)


__all__ = ["Report", "SCHEMA", "FLAGS", "ESCALATABLE_FLAGS", "WARNING_FLAGS", "INFO_FLAGS",
           "LEVELS", "MINIMAL_LEVEL_FLAGS", "PROGRESS_ENV", "PROGRESS_FILENAME", "flag_severity",
           "progress_path_from_env", "write_progress", "write_report_json"]
