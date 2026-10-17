"""``DrawingOptions``: every knob of the drawing module in one dataclass.

The CLI parses into it, the sandbox parent hands it to the child as JSON
(``--options-json``), and the pipeline reads it. ``validate`` raises
``ValueError`` whose message starts with the offending field name, so the
CLI can turn it into a usage error (exit 1).

Measured facts the limits rely on (sandbox probe):

* library imports cost 4-5 s of wall time per run, so ``timeout_s`` below
  15 s would produce false timeouts -> floor 15 s;
* OCP + trimesh import commits about 192 MB with OPENBLAS_NUM_THREADS=1, so
  ``memory_limit_mb`` below 256 cannot even import -> floor 256 MB;
* ASCII STL facet scan runs at about 380 MB/s, so the 200 MB input cap costs
  well under a second in the pre-spawn sniff.

The view names are repeated here (not imported from ``projection.py``) so
this module stays import-free; ``tests/test_drawing_core.py`` pins them.
"""
from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass
from typing import Optional, Tuple

FORMATS = ("pdf", "svg", "dxf")
VIEW_NAMES = ("front", "back", "top", "bottom", "right", "left", "iso")
SECTIONS = ("auto", "front", "top", "right", "off")
SHEET_CHOICES = ("A3", "A4", "A4L", "auto")
PROJECTIONS = ("first", "third")
ORIENTS = ("auto", "fixed")
UPS = ("z", "y")
STL_TO_STEP = ("auto", "faceted", "nurbs", "off")
#: ``stl_unit`` choices (STL carries no unit) and their factor to mm.
STL_UNITS = ("mm", "cm", "m", "in")
STL_UNIT_FACTORS = {"mm": 1.0, "cm": 10.0, "m": 1000.0, "in": 25.4}

MIN_TIMEOUT_S = 15.0
MIN_MEMORY_MB = 256

_TUPLE_FIELDS = ("formats", "views")
_FLOAT_FIELDS = ("text_height", "facet_smooth_angle_deg", "feature_angle_deg",
                 "hlr_time_budget_s", "classifier_threshold", "max_input_mb", "timeout_s")
_INT_FIELDS = ("max_hole_callouts", "faceted_max_triangles", "raster_px", "exact_hlr_max_faces",
               "max_triangles", "memory_limit_mb", "output_cap_mb", "max_hidden_primitives")
_BOOL_FIELDS = ("hidden_lines", "hole_positions", "quiet", "hidden_lines_force")


@dataclass
class DrawingOptions:
    formats: Tuple[str, ...] = ("pdf", "svg", "dxf")
    views: Tuple[str, ...] = ("front", "top", "right", "iso")
    section: str = "auto"              # auto | front | top | right | off
    sheet: str = "A3"                  # A3 | A4 | A4L | auto
    projection: str = "first"          # first | third
    orient: str = "auto"               # auto | fixed
    up: str = "z"                      # z | y
    text_height: float = 3.5
    hidden_lines: bool = True
    hole_positions: bool = True
    max_hole_callouts: int = 6
    stl_to_step: str = "auto"          # auto | faceted | nurbs | off
    faceted_max_triangles: int = 30_000
    facet_smooth_angle_deg: float = 20.0
    feature_angle_deg: float = 30.0
    raster_px: int = 1024
    exact_hlr_max_faces: int = 4000
    hlr_time_budget_s: float = 60.0
    classifier_threshold: float = 0.10
    max_input_mb: float = 200.0
    max_triangles: int = 2_000_000
    timeout_s: float = 180.0
    memory_limit_mb: int = 2048
    output_cap_mb: int = 200
    title: Optional[str] = None
    owner: str = ""
    quiet: bool = False
    # QA round 1: the unit of an STL file (None = not given: read as mm and
    # flagged ``unit_assumed_mm``); the mesh is scaled by STL_UNIT_FACTORS
    stl_unit: Optional[str] = None
    # raw hidden primitives per B-rep view above which the hidden lines of
    # that view are dropped (flag ``hidden_lines_dropped``); None = no cap.
    # Measured raw counts: at most 1393 per view on the QA corpus parts
    # (NIST CTC/FTC, exact HLR), 185k-193k on the 10665-face RC buggy (poly
    # HLR), whose hidden clean-up alone cost about 14 s per view
    max_hidden_primitives: Optional[int] = 20_000
    # F7-perf: large assemblies are drawn without hidden lines (more than 20
    # solid instances, or a view with more than 50000 raw hidden HLR edges;
    # flag ``hidden_lines_omitted_assembly``). True keeps them (the
    # ``max_hidden_primitives`` cap and the time budget still apply).
    hidden_lines_force: bool = False

    def __post_init__(self):
        # lists (from JSON or callers) become tuples so options stay hashable-ish
        for name in _TUPLE_FIELDS:
            v = getattr(self, name)
            if isinstance(v, list):
                setattr(self, name, tuple(v))

    # ------------------------------------------------------------ checks --
    def validate(self) -> None:
        """Raise ``ValueError("<field>: ...")`` on the first invalid field."""
        def bad(name, why):
            raise ValueError(f"{name}: {why} (got {getattr(self, name)!r})")

        def choice(name, allowed):
            if getattr(self, name) not in allowed:
                bad(name, "must be one of " + ", ".join(allowed))

        def number(name, lo=None, hi=None, lo_open=False, integer=False):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                bad(name, "must be a number")
            if integer and not isinstance(v, int):
                bad(name, "must be an integer")
            if v != v:  # NaN
                bad(name, "must not be NaN")
            if isinstance(v, float) and not math.isfinite(v):  # inf would switch a safety limit off
                bad(name, "must be finite")
            if lo is not None and (v <= lo if lo_open else v < lo):
                bad(name, f"must be {'>' if lo_open else '>='} {lo}")
            if hi is not None and v > hi:
                bad(name, f"must be <= {hi}")

        def names(name, allowed):
            v = getattr(self, name)
            if isinstance(v, str) or not isinstance(v, (tuple, list)):
                bad(name, "must be a tuple of names")
            if not v:
                bad(name, "must not be empty")
            for item in v:
                if item not in allowed:
                    bad(name, f"unknown entry {item!r}; allowed: " + ", ".join(allowed))
            if len(set(v)) != len(v):
                bad(name, "has duplicate entries")

        names("formats", FORMATS)
        names("views", VIEW_NAMES)
        choice("section", SECTIONS)
        choice("sheet", SHEET_CHOICES)
        choice("projection", PROJECTIONS)
        choice("orient", ORIENTS)
        choice("up", UPS)
        number("text_height", 1.0, 20.0)
        for b in _BOOL_FIELDS:
            if not isinstance(getattr(self, b), bool):
                bad(b, "must be true or false")
        number("max_hole_callouts", 0, 100, integer=True)
        choice("stl_to_step", STL_TO_STEP)
        number("faceted_max_triangles", 1, integer=True)
        number("facet_smooth_angle_deg", 0.0, 180.0)
        number("feature_angle_deg", 0.0, 180.0, lo_open=True)
        number("raster_px", 64, 16384, integer=True)
        number("exact_hlr_max_faces", 0, integer=True)
        number("hlr_time_budget_s", 0.0, lo_open=True)
        number("classifier_threshold", 0.0, 1.0)
        number("max_input_mb", 0.0, lo_open=True)
        number("max_triangles", 1, integer=True)
        number("timeout_s", MIN_TIMEOUT_S)
        number("memory_limit_mb", MIN_MEMORY_MB, integer=True)
        number("output_cap_mb", 1, integer=True)
        if self.title is not None and not isinstance(self.title, str):
            bad("title", "must be a string or null")
        if not isinstance(self.owner, str):
            bad("owner", "must be a string")
        if self.stl_unit is not None and self.stl_unit not in STL_UNITS:
            bad("stl_unit", "must be null or one of " + ", ".join(STL_UNITS))
        if self.max_hidden_primitives is not None:
            number("max_hidden_primitives", 0, integer=True)

    # -------------------------------------------------------------- JSON --
    def to_json(self) -> str:
        d = dataclasses.asdict(self)
        for name in _TUPLE_FIELDS:
            d[name] = list(d[name])
        return json.dumps(d, ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "DrawingOptions":
        """Parse ``to_json`` output; missing keys take defaults, unknown keys raise."""
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"options JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("options JSON: top level must be an object")
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"{unknown[0]}: unknown option")
        kw = dict(data)
        for name in _TUPLE_FIELDS:
            if name in kw and isinstance(kw[name], list):
                kw[name] = tuple(kw[name])
        for name in _FLOAT_FIELDS:
            v = kw.get(name)
            if isinstance(v, int) and not isinstance(v, bool):
                kw[name] = float(v)
        for name in _INT_FIELDS:
            v = kw.get(name)
            if isinstance(v, float) and v.is_integer():
                kw[name] = int(v)
        return cls(**kw)

    def with_overrides(self, **kw) -> "DrawingOptions":
        """A copy with the given fields replaced; ``self`` is not modified."""
        known = {f.name for f in dataclasses.fields(self)}
        for k in kw:
            if k not in known:
                raise ValueError(f"{k}: unknown option")
        return dataclasses.replace(self, **kw)
