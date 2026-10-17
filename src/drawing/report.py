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

Severities follow spec 3.8. ``WARNING_FLAGS`` and ``INFO_FLAGS`` list the
codes the pipeline emits; ``flag`` accepts any code but warns in the log
when the severity disagrees with the table (a typo would otherwise change
the exit code silently). ``approval_blank`` is listed as a warning in the
spec but is emitted as ``info`` here: the approval field of an
auto-generated drawing is ALWAYS blank, so a warning would make every run
exit 2 and the exit code would carry no information.

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

WARNING_FLAGS = frozenset({
    "mesh_not_watertight", "multiple_bodies", "brepcheck_invalid", "open_shell",
    "inverted_solid", "fallback_to_mesh", "hlr_timeout", "section_empty",
    "section_open_wires", "views_overflow", "dimensions_capped", "unit_not_mm",
    "sheet_not_iso5457",
    # additions of this implementation (a part of the drawing is missing)
    "view_failed", "section_failed", "writer_failed", "dimensions_failed",
    "faceted_audit_mismatch",
})
INFO_FLAGS = frozenset({
    "holes_not_detected_mesh_route", "organic_classified", "too_many_triangles",
    "extension_mismatch", "facet_edges_suppressed", "hlr_poly_used", "section_nudged",
    "degenerate_faces_removed", "approval_blank",
})
SEVERITIES = ("info", "warning")
LEVELS = ("full", "reduced", "minimal", "none")


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


class Report:
    """The quality report of one run; ``data`` is the JSON document."""

    def __init__(self, input_path: str, out_dir: str, options) -> None:
        self.t0 = time.perf_counter()
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
            "quality": {"level": None, "flags": []},
            "warnings": [],
            "timing": {"total_s": None, "load_s": None, "views_s": None, "section_s": None,
                       "write_s": None},
            "sandbox": _sandbox_disabled(),
            "version": _versions(),
        }

    # ------------------------------------------------------------ events --
    def flag(self, code: str, severity: str, detail: str = "") -> None:
        """Add a quality flag; an exact duplicate (code, severity, detail) is ignored."""
        if severity not in SEVERITIES:
            raise ValueError(f"flag severity must be one of {SEVERITIES}, got {severity!r}")
        expected = ("warning" if code in WARNING_FLAGS else
                    "info" if code in INFO_FLAGS else None)
        if expected is not None and expected != severity:
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

    # ------------------------------------------------------------ result --
    def _level(self, ok: bool) -> str:
        d = self.data
        if not ok:
            return "none"
        views = d.get("views") or []
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
        d["timing"]["total_s"] = round(time.perf_counter() - self.t0, 3)
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


__all__ = ["Report", "SCHEMA", "WARNING_FLAGS", "INFO_FLAGS", "LEVELS", "write_report_json"]
