"""The router of the drawing module: input -> geometry source -> sheet -> files + report.

``run(input_path, out_dir, options)`` never raises: every stage is wrapped,
failures become ``error`` / ``warnings`` / ``fallbacks`` / quality flags in
the report, and it returns ``(exit_code, report_dict)`` with the
``src/convert.py`` exit codes (0 clean, 2 written with warning flags,
1 failed). The report is also written to ``<out_dir>/<stem>.report.json``.

Order (spec 3.2-3.4): sniff -> limits -> load (B-rep or mesh) -> orientation
(``projection.auto_orientation`` on the file bbox, then the depth sign: a
180-degree turn about Z when the far side carries clearly more features,
QA issue 14) applied with ``transform_shape`` / ``transform_mesh`` -> bbox
of the oriented model (STEP: the exact file box mapped by the signed axis
permutation, ``_permuted_box``; no second ``AddOptimal``) -> cylinder features (true B-rep only) -> the
requested views -> section (unless ``off``) -> ``plan_dimensions`` ->
``compose`` -> writers -> report.

Routing:

* STEP -> ``brep_load.load_step`` (the DRAWN shape: the solids when there
  are any; loose curve sets and supplemental shells are left out and
  flagged ``loose_curves_ignored`` / ``supplemental_shells_ignored``) ->
  exact HLR (``HLRBRep_Algo``) up to ``exact_hlr_max_faces`` faces, poly
  HLR above (flag ``hlr_poly_used``). A view that fails or exceeds
  ``hlr_time_budget_s`` in the native HLR is skipped and flagged
  (``view_failed`` / ``hlr_timeout``); a budget overrun later (extraction,
  clean-up) keeps the view without hidden lines (``hidden_lines_dropped``),
  and so does a view with more than ``max_hidden_primitives`` (default
  20000) raw hidden primitives; no view at all -> exit 1. A tessellated-only STEP (``TessellatedOnlyError``)
  goes to the mesh route (``route.source`` ``step_tessellated_mesh``, flag
  ``step_tessellated_only``).
* STL -> ``mesh_load.load_mesh_model`` (census + prismatic/organic/not_solid
  classifier). The unit is ``stl_unit`` when given (the mesh is scaled to
  mm), else assumed mm (``source_unit`` ``assumed_mm``, flag
  ``unit_assumed_mm``). ``stl_to_step``: ``auto`` converts prismatic meshes
  with at most ``faceted_max_triangles`` triangles to a faceted STEP
  (``src.io.faceted_step.export_step_faceted(verify=False)`` into
  ``<out_dir>/<stem>.step``) and draws the faceted body with the poly HLR
  (``brep_load.faceted_shape_from_mesh`` -> shell / solid with the smooth
  facet edges coded C1, ``project_brep(exact=False)``); otherwise the mesh
  route with ``stl_to_step.skipped_reason`` ``organic`` / ``not_solid`` /
  ``too_many_triangles``. ``faceted`` forces the conversion, ``nurbs`` runs
  the smooth converter of ``src/convert.py`` (QuadWrapper +
  SubDToNURBSConverter, G1, reference fit) and draws that B-rep with exact
  HLR, ``off`` always uses the mesh route. ANY exception of the B-rep stage
  on an STL input (conversion, empty body, a failed or timed-out view, the
  section) falls back to the mesh route (``route.fallbacks``, flag
  ``fallback_to_mesh``). A ``not_solid`` mesh (triangle soup) is drawn on
  the mesh route without section and overall dimensions (flag
  ``not_a_solid``, level ``minimal``).
* Mesh route: ``mesh_views.project_mesh`` (depth buffer) and
  ``mesh_sections``; no hole features (flag ``holes_not_detected_mesh_route``).

Unit plausibility (QA issue 5): ``mesh_load.unit_plausibility`` on the
model extents in mm, with the declared STEP unit (or the STL unit when
given); a suspicious size raises ``unit_suspect`` (warning) with the
alternative reading in the detail. After the layout, ``views_tiny``
(warning) when the sheet scale is the largest standard scale and the view
geometry covers less than 10 % of the drawing frame (``_views_tiny``).

Mesh ``inverted_solid`` follows the census verdict ``inverted_solid``
(``mesh_load.inverted_solid_verdict``: inverted with no winding conflict, a
few, or a negative raw signed sum matching the repaired volume; QA round 2
issue 8) with the signed raw sum in the detail; for a badly wound mesh the
census "inverted" verdict only describes its dominant piece, so one warning
text about the inconsistent winding is written instead. Any winding conflict
with a repaired volume adds ``winding_repaired`` (info).

Large assemblies (more than ``_ASSEMBLY_MIN_SOLIDS`` solid instances, QA
round 2 issues 5 and 11): no hidden lines (below), no hole callouts and no
position dimensions (``dimensions_omitted_assembly``, undimensioned reason
``assembly`` per radius group) and no auto section
(``section_skipped_assembly``; an explicit ``--section`` plane is drawn). The
section is also skipped when its estimated cost (``_SECTION_COST_VIEWS`` x
the mean view time) exceeds the time left before the write reserve.

The faceted body that is drawn is built from the ORIENTED mesh by
``faceted_shape_from_mesh`` (one face per triangle, smooth edges coded C1:
the measured requirement of the poly HLR), while the STEP file written for
the user is ``export_step_faceted`` of the mesh in its FILE coordinates
(coplanar faces merged). ``export_step_faceted`` does not return its shape,
and the merged faces would lose the regularity coding the poly HLR needs.
Its volume is audited against the winding-repaired mesh volume of the
census (``mesh_load.mesh_volume_repaired``), and the drawn B-rep volume is
the single ``geometry.volume_mm3`` of that route.

HLR discretization: ``deflection = 0.05 / scale_guess`` mm, where
``scale_guess`` is ``layout.choose_scale`` on the bbox-only view boxes of
the oriented model before any HLR (A3 when ``sheet=auto``: its scale is at
least the A4 one, so the deflection is never coarser than needed), clamped
to [0.001, 5] mm. The same ``scale_guess`` is passed to the views as
``paper_scale`` (paper-mm tolerances, dash chaining) and to the dimension
planner as ``scale_hint``.

Progress (sandboxed runs): at every stage boundary the pipeline writes
``Report.progress_snapshot`` (stage, view, flags, unit, geometry, timing,
plus ``counts``; the stage times so far are copied into the report's
``timing`` first, ``views_s`` updated before every view) to the file named
by ``report.PROGRESS_ENV`` (no-op when
the variable is unset); the CLI parent merges the last snapshot into the
failure report when the child is killed (QA issue 17).

Large assemblies and the run deadline (F7-perf): the STEP census goes per
unique solid above ``exact_hlr_max_faces``; more than 20 solids or 50000
raw hidden edges in a view draw that model / view without hidden lines
(``hidden_lines_omitted_assembly``, unless ``hidden_lines_force``); the run
plans to finish by 0.8 x ``timeout_s`` (view budgets, section skip
``section_skipped_budget``, DXF audit gate ``writers.dxf_audited``); the poly
HLR meshes the shape once per run (``_Ctx.mesh_cache``). Every stage boundary
logs one ``src.drawing.stage`` INFO line (kept by the CLI under --quiet).

No output to stdout (the CLI owns it): progress goes to ``logging`` or the
``log`` callable; the NURBS converter (which prints) runs under
``redirect_stdout(sys.stderr)``.
"""
from __future__ import annotations

import contextlib
import logging
import math
import os
import sys
import time
import types
from typing import Any, Callable, Dict, List, Optional, Tuple

from .options import STL_UNIT_FACTORS, DrawingOptions
from .report import Report, progress_path_from_env, write_progress

logger = logging.getLogger(__name__)

SOURCE_NOTES = {"step": "STEP", "stl_faceted_step": "STL->faceted STEP",
                "stl_nurbs_step": "STL->NURBS STEP", "stl_mesh": "STL mesh projection",
                "step_tessellated_mesh": "tessellated STEP mesh projection"}
_SECTION_VIEWS = {"auto": "front", "front": "front", "top": "top", "right": "right"}
_DEFLECTION_PER_SCALE = 0.05
_DEFLECTION_RANGE = (1e-3, 5.0)

# depth sign (QA issue 14): B-rep faces facing each side (brep_features.score_depth_sign)
_DEPTH_FLIP_RATIO = 1.25          # flip when the far side scores more than 1.25 x ...
_DEPTH_FLIP_MIN_DIFF = 2.0        # ... and at least 2 (about two faces) more
_DEPTH_SIGN_MAX_FACES = 5000      # above: skipped (7.5 s on the 10665-face RC buggy)
_DEPTH_MESH_MARGIN = 1.2          # mesh route: mesh_load.choose_depth_sign_mesh margin

# flags the B-rep stage of an STL input produces; forgotten on the mesh fallback
_BREP_STAGE_FLAGS = ("hlr_poly_used", "section_nudged", "section_open_wires", "section_empty",
                     "holes_not_detected_mesh_route", "facet_edges_suppressed",
                     "hidden_lines_dropped", "section_fallback", "hidden_lines_omitted_assembly",
                     "section_skipped_budget", "section_skipped_assembly",
                     "section_gaps_bridged")

# F7-perf: large assemblies and the cooperative run deadline
_ASSEMBLY_MIN_SOLIDS = 20         # more solid instances: no hidden lines unless hidden_lines_force
_MAX_HIDDEN_EDGES = 50_000        # raw hidden HLR edges of one view above which it has none
_DEADLINE_FRACTION = 0.8          # run deadline = 0.8 x timeout_s after run() started
_RESERVE_FRACTION = 0.25          # of the run budget kept for section + sheet while drawing views
_WRITE_RESERVE_FRACTION = 0.1     # of the run budget kept for the sheet and writers
_MIN_RESERVE_S = 5.0
_MIN_VIEW_BUDGET_S = 2.0          # a view never gets less (its visible lines are still drawn)
_DXF_AUDIT_MAX_ITEMS = 100_000    # sheet items above which the DXF is not re-read for the audit
_DXF_AUDIT_MIN_REMAINING_S = 10.0  # ... and when less time than this is left before the deadline
# QA round 2 issue 11: the section's cost is estimated as this many mean view times
# (measured on the 211-solid RC buggy: section 37.0 s vs a mean view of 10.1 s; on the
# NIST parts the section costs 0.2-1.5 x a view); skipped when the time left before
# the write reserve is below the estimate
_SECTION_COST_VIEWS = 1.0
# QA round 2 issue 21: section filled solid over more than this share of the cut area
_SECTION_BLACKENED_SHARE = 0.5
# ... and an open-mesh bridge chord longer than this share of the section extent: warning
_BRIDGE_WARN_SHARE = 0.05
# QA round 2 issue 30: the strip note when the unit is doubtful (two lines at 2.5 mm or
# more, layout.strip_items; at most 64 characters with a '(' part)
UNIT_NOTE_ASSUMED = "Dimensions in mm (unit assumed, see report)"
UNIT_NOTE_SUSPECT = "Dimensions in mm (unit doubtful, see report)"

stage_logger = logging.getLogger("src.drawing.stage")


def _rss_mb() -> Optional[float]:
    try:
        import psutil
        return float(psutil.Process().memory_info().rss) / 2 ** 20
    except Exception:
        return None


class _Failure(Exception):
    """A stage failure that ends the run with exit 1 (message = report error)."""


class _BrepStageError(Exception):
    """A B-rep stage failure on an STL input: fall back to the mesh route."""


class _Ctx:
    """Mutable state of one run."""

    def __init__(self, rep: Report, input_path: str, out_dir: str, options: DrawingOptions,
                 say: Callable[[str], None]):
        self.rep = rep
        self.input_path = input_path
        self.out_dir = out_dir
        self.options = options
        self.say = say
        self.stem = rep.stem
        # write_s is the whole sheet stage; dimensions_s + layout_s + writers_s split it.
        # features_s: B-rep cylinder feature detection (review R5: it sat in no bucket)
        self.timing: Dict[str, float] = {"load_s": 0.0, "features_s": 0.0, "views_s": 0.0,
                                         "section_s": 0.0, "write_s": 0.0, "dimensions_s": 0.0,
                                         "layout_s": 0.0, "writers_s": 0.0}
        self.views: Dict[str, Any] = {}
        self.section = None
        self.features: List[Any] = []
        self.extents = None             # oriented model extents (3,)
        self.bounds = None              # oriented model box (2, 3): lo, hi
        self.scale_guess: Optional[float] = None
        self.source: Optional[str] = None
        self.skip_section: Optional[str] = None      # reason, e.g. "not_a_solid"
        self.skip_overall_dims: Optional[str] = None
        # QA round 2: number of solids (B-rep) or mesh components (planner's n_bodies)
        # and the large-assembly verdict (no hole dims, no auto section; issues 5, 11)
        self.n_bodies: Optional[int] = None
        self.assembly: bool = False
        self.progress_path = progress_path_from_env()
        # cooperative run deadline (F7-perf item D): the sandbox kills the child at
        # timeout_s; the run plans to finish by 0.8 x timeout_s after this point
        self.t_start = time.perf_counter()
        try:                                         # validate() runs later, inside _run
            self.budget_s = _DEADLINE_FRACTION * float(options.timeout_s)
            if not (math.isfinite(self.budget_s) and self.budget_s > 0.0):
                raise ValueError(self.budget_s)
        except Exception:
            self.budget_s = _DEADLINE_FRACTION * 180.0
        self.deadline = self.t_start + self.budget_s
        # shapes meshed for the poly HLR in this run (brep_views._mesh_once, item F)
        self.mesh_cache: Dict[int, Any] = {}

    def remaining(self) -> float:
        """Seconds left before the run deadline (negative when passed)."""
        return self.deadline - time.perf_counter()

    def reserve(self, fraction: float) -> float:
        return max(_MIN_RESERVE_S, fraction * self.budget_s)

    def stage_line(self, stage: str, view: Optional[str] = None) -> None:
        """One INFO line per stage boundary on ``src.drawing.stage`` (kept by the
        CLI under --quiet, so a killed child's stderr tail shows where it was)."""
        try:
            rss = _rss_mb()
            stage_logger.info("stage %s%s elapsed %.1f s remaining %.1f s rss %s", stage,
                              f" view {view}" if view else "",
                              time.perf_counter() - self.t_start, self.remaining(),
                              "n/a" if rss is None else f"{rss:.0f} MB")
        except Exception:                            # never fail a run over a log line
            pass

    def progress(self, stage: str, view: Optional[str] = None) -> bool:
        """Log the stage line and write the progress snapshot (the snapshot is a
        no-op without ``report.PROGRESS_ENV``)."""
        self.stage_line(stage, view)
        if not self.progress_path:
            return False
        try:
            # the stage times so far: a killed child's report keeps them (QA issue 17)
            for k, v in self.timing.items():
                self.rep.data["timing"][k] = round(float(v), 3)
            snap = self.rep.progress_snapshot(stage, view)
            snap["counts"] = {"views": len(self.views),
                              "section_loops": (len(self.section.loops)
                                                if self.section is not None else 0),
                              "dimensions": len(self.rep.data.get("dimensions") or []),
                              "flags": len(self.rep.data["quality"]["flags"])}
            return write_progress(self.progress_path, snap)
        except Exception as exc:                     # progress is best effort
            logger.debug("progress snapshot failed: %s", exc)
            return False


# ---------------------------------------------------------------- helpers --

def _out_path(ctx: _Ctx, ext: str) -> str:
    return os.path.abspath(os.path.join(ctx.out_dir, f"{ctx.stem}.{ext}"))


def _is_identity(matrix) -> bool:
    import numpy as np
    return bool(np.allclose(np.asarray(matrix, float), np.eye(3), atol=1e-12))


def _orientation(ctx: _Ctx, extents):
    from .projection import auto_orientation
    o = auto_orientation(extents, mode=ctx.options.orient, up=ctx.options.up)
    ctx.rep.data["route"]["orientation"] = o.to_dict()
    return o


def _record_orientation(ctx: _Ctx, o, depth: dict) -> None:
    d = o.to_dict()
    d.update(depth)
    ctx.rep.data["route"]["orientation"] = d


def _depth_sign_brep(ctx: _Ctx, shape, o, faces: Optional[int], bbox=None):
    """``(orientation, info)``: keep or turn the part 180 deg about Z (QA issue 14).

    ``brep_features.choose_depth_sign_brep`` on the FILE shape with the
    front viewer's direction mapped back (``o.matrix.T @ VIEWS["front"].n``;
    QA round 2 issue 4): the open-cavity rule (``brep_cavity``) decides when
    a side holds a clear cavity, else the face rule (``brep_faces``: turned
    when the far side has more than 1.25 x the faces and at least 2 more);
    the cavity rule yields to the face rule for an open housing (the deeper
    side's mean recess above 0.4 of the bbox depth). ``bbox``: the census
    exact ``(lo, hi)`` in file coordinates (saves the scorer its own box).
    Only for ``orient=auto``.
    """
    import numpy as np
    from .projection import VIEWS, flip_depth

    info = {"depth_flipped": False, "depth_method": "brep_faces", "depth_scores": None,
            "depth_reason": None}
    if ctx.options.orient != "auto":
        info["depth_reason"] = "orient fixed: the file axes are kept"
        return o, info
    if faces is not None and faces > _DEPTH_SIGN_MAX_FACES:
        info["depth_reason"] = f"not scored: {faces} faces > {_DEPTH_SIGN_MAX_FACES}"
        return o, info
    try:
        from .brep_features import choose_depth_sign_brep
        n_file = np.asarray(o.matrix, float).T @ np.asarray(VIEWS["front"].n, float)
        r = choose_depth_sign_brep(shape, n_file, bbox=bbox, ratio=_DEPTH_FLIP_RATIO,
                                   min_diff=_DEPTH_FLIP_MIN_DIFF)
    except Exception as exc:
        info["depth_reason"] = f"scoring failed: {type(exc).__name__}: {exc}"
        logger.warning("depth sign scoring failed: %s", exc)
        return o, info
    flip = bool(r.get("flip"))
    info["depth_method"] = r.get("method") or "brep_faces"
    info["depth_scores"] = {k: (bool(r.get(src)) if k == "housing" else
                                (None if r.get(src) is None else float(r.get(src))))
                            for k, src in (("kept", "score_kept"), ("flipped", "score_flipped"),
                                           ("cavity_kept", "cavity_kept"),
                                           ("cavity_flipped", "cavity_flipped"),
                                           ("faces_kept", "faces_kept"),
                                           ("faces_flipped", "faces_flipped"),
                                           ("mean_recess_kept", "mean_recess_kept"),
                                           ("mean_recess_flipped", "mean_recess_flipped"),
                                           ("housing", "housing"))}
    info["depth_reason"] = r.get("reason") or None      # says "turned 180 deg" itself
    if flip:
        o = flip_depth(o)
        info["depth_flipped"] = True
    return o, info


def _depth_sign_mesh(ctx: _Ctx, mesh, o):
    """``(orientation, info)`` from ``mesh_load.choose_depth_sign_mesh`` (orient auto only)."""
    import numpy as np
    from .projection import flip_depth

    info = {"depth_flipped": False, "depth_method": "mesh_feature_edges", "depth_scores": None,
            "depth_reason": None}
    if ctx.options.orient != "auto":
        info["depth_reason"] = "orient fixed: the file axes are kept"
        return o, info
    try:
        from .mesh_load import choose_depth_sign_mesh
        r = choose_depth_sign_mesh(mesh, o.matrix, margin=_DEPTH_MESH_MARGIN)
    except Exception as exc:
        info["depth_reason"] = f"scoring failed: {type(exc).__name__}: {exc}"
        logger.warning("mesh depth sign scoring failed: %s", exc)
        return o, info
    info["depth_method"] = r.get("method") or "mesh_feature_edges"
    info["depth_scores"] = {"kept": float(r.get("score_kept") or 0.0),
                            "flipped": float(r.get("score_flipped") or 0.0)}
    for key in ("cavity_kept", "cavity_flipped", "edges_kept", "edges_flipped"):
        if r.get(key) is not None:
            info["depth_scores"][key] = float(r[key])
    info["depth_reason"] = r.get("reason")
    if r.get("flip"):
        o2 = flip_depth(o)
        if not np.allclose(np.asarray(r.get("matrix"), float), o2.matrix, atol=1e-9):
            logger.warning("mesh depth flip matrix disagrees with projection.flip_depth")
        o = o2
        info["depth_flipped"] = True
    return o, info


def _corners(lo, hi):
    import numpy as np
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                     for z in (lo[2], hi[2])])


def _permuted_box(matrix, lo, hi):
    """``(lo', hi')`` of the box ``[lo, hi]`` under ``world' = matrix @ world``
    when ``matrix`` is a signed axis permutation (every ``Orientation``), else
    None. Exact: such a matrix maps an axis-aligned box onto one."""
    import numpy as np
    m = np.asarray(matrix, float)
    if m.shape != (3, 3) or not np.allclose(np.abs(m), np.round(np.abs(m)), atol=1e-12) \
            or not np.allclose(np.abs(m).sum(axis=0), 1.0) \
            or not np.allclose(np.abs(m).sum(axis=1), 1.0):
        return None
    pts = _corners(lo, hi) @ np.round(m).T
    return pts.min(axis=0), pts.max(axis=0)


def _scale_guess(ctx: _Ctx, lo, hi) -> float:
    """``choose_scale`` on the bbox-only boxes of the requested principal views."""
    from .layout import choose_scale
    from .model import BBox2D
    from .projection import PRINCIPAL, VIEWS
    from .style import DEFAULT_STYLE, SHEETS

    pts = _corners(lo, hi)
    boxes = {}
    for name in ctx.options.views:
        if name in PRINCIPAL:
            boxes[name] = BBox2D.from_points(VIEWS[name].to_view(pts)[:, :2])
    if not boxes:                           # iso only: estimate with the front box
        boxes["front"] = BBox2D.from_points(VIEWS["front"].to_view(pts)[:, :2])
    sheet = SHEETS["A3" if ctx.options.sheet == "auto" else ctx.options.sheet]
    try:
        _s, info = choose_scale(boxes, {}, sheet, DEFAULT_STYLE, ctx.options.projection,
                                ctx.options.section != "off", "iso" in ctx.options.views)
        return float(info.get("scale") or 1.0)
    except Exception as exc:                 # the guess only sets the HLR accuracy
        logger.warning("scale guess failed (%s); using 1:1", exc)
        return 1.0


def _deflection(scale: float) -> float:
    lo, hi = _DEFLECTION_RANGE
    d = _DEFLECTION_PER_SCALE / max(float(scale), 1e-12)
    return float(min(max(d, lo), hi))


def _view_entry(name: str, pv, seconds: float) -> dict:
    st = pv.stats or {}
    e = {"name": name, "algorithm": pv.algorithm, "visible": len(pv.visible),
         "hidden": len(pv.hidden), "seconds": round(float(seconds), 4)}
    if st.get("hidden_dropped"):
        e["hidden_dropped"] = st.get("hidden_dropped_reason") or True
    if st.get("hidden_omitted"):
        e["hidden_omitted"] = st.get("hidden_omitted")
    if st.get("time_budget_s") is not None:
        e["time_budget_s"] = st.get("time_budget_s")
    if st.get("mesh_reused"):
        e["mesh_reused"] = True
    # QA round 2 (R1 clean-ups): mesh route chain welding / hairpins / dropped loops and
    # slivers, B-rep route hidden curves under visible ones (issues 19, 25, 18)
    for key in _VIEW_COUNT_KEYS:
        v = st.get(key)
        if v is not None:
            try:
                e[key] = int(v)
            except (TypeError, ValueError):
                pass
    return e


_VIEW_COUNT_KEYS = ("welded_ends", "hairpins_removed", "closed_loops_dropped", "dropped_chains",
                    "n_sliver_pairs", "hidden_curves_dropped", "hidden_curves_trimmed")


def _view_flags(ctx: _Ctx, name: str, pv) -> None:
    """Report flags a projected view asks for (QA issues 3 and 13)."""
    st = pv.stats or {}
    if st.get("hidden_dropped") and st.get("hidden_dropped_reason") == "max_hidden_edges":
        ctx.rep.flag("hidden_lines_omitted_assembly", "info",
                     f"view {name}: {st.get('raw_hidden_edges')} raw hidden edges > "
                     f"{_MAX_HIDDEN_EDGES}: drawn without hidden lines (--hidden-lines-force "
                     f"keeps them)")
    elif st.get("hidden_dropped"):
        ctx.rep.flag("hidden_lines_dropped", "warning",
                     f"view {name}: hidden lines omitted ({st.get('hidden_dropped_reason')})")
    if st.get("interior_silhouettes_omitted"):
        ctx.rep.flag("interior_silhouettes_omitted", "info",
                     "organic mesh too large for silhouette candidates: interior and hidden "
                     "silhouettes (through holes, overlapping lobes) are not drawn")


def _section_spec(normal):
    import numpy as np
    from .projection import VIEWS
    n = np.asarray(normal, float)
    for name in ("front", "top", "right", "back", "bottom", "left"):
        if float(np.dot(n, VIEWS[name].n)) > 1.0 - 1e-9:
            return VIEWS[name]
    raise ValueError(f"section normal {n.tolist()} is not a principal view direction")


def _section_report(ctx: _Ctx, sec, info: Optional[dict] = None) -> None:
    st = sec.stats or {}
    info = info or {}
    d = {"label": sec.label, "view": sec.spec.name, "marker_view": sec.marker_view,
         "plane_point": [float(v) for v in sec.plane_point],
         "plane_normal": [float(v) for v in sec.plane_normal],
         "loops": len(sec.loops), "open_wires_dropped": int(st.get("open_wires_dropped", 0) or 0),
         "hatch_segments": 0, "nudged": bool(st.get("nudged", False)),
         "seconds": st.get("seconds"), "reason": info.get("reason"),
         "hole_kind": info.get("hole_kind"),
         "retried_at_body": bool(st.get("retried_at_body", False))}
    for key in ("kept_length_ratio", "kept_area_ratio", "bridged_gaps", "bridge_max_gap",
                "bridge_rejected", "excess_bodies", "half_space_excess_mm",
                "half_space_excess_tol_mm", "fasteners_unsectioned", "bodies_skipped"):
        if key in st:
            d[key] = st.get(key)
    ctx.rep.data["section"] = d
    n_bridged = int(st.get("bridged_gaps", 0) or 0)
    if n_bridged > 0:
        # QA round 2 issue 21: synthetic chords across mesh holes are disclosed; a chord
        # longer than 5 % of the section extent is a warning
        gap = float(st.get("bridge_max_gap", 0.0) or 0.0)
        extent = 0.0
        try:
            import numpy as np
            pts = np.vstack([np.asarray(lp, float).reshape(-1, 2)[:, :2] for lp in sec.loops
                             if np.size(lp)])
            extent = float(np.ptp(pts, axis=0).max())
        except Exception:
            extent = 0.0
        share = gap / extent if extent > 0.0 else float("inf")
        sev = "warning" if share > _BRIDGE_WARN_SHARE else "info"
        ctx.rep.flag("section_gaps_bridged", sev,
                     f"{n_bridged} open section gap(s) of the mesh closed by straight chords; "
                     f"largest {gap:.3g} mm"
                     + (f" ({100.0 * share:.3g} % of the section extent {extent:.4g} mm)"
                        if extent > 0.0 else ""))
    if st.get("nudged"):
        ctx.rep.flag("section_nudged", "info", "the plane gave no closed loop (or lay in a "
                     "planar face) and was moved by 1 % of the depth")
    if int(st.get("open_wires_dropped", 0) or 0) > 0:
        ctx.rep.flag("section_open_wires", "warning",
                     f"{st.get('open_wires_dropped')} open section wire(s) dropped")
    if not sec.loops:
        ctx.rep.flag("section_empty", "warning", "the section plane cuts no closed loop; "
                     "no section view drawn")
    if int(st.get("excess_bodies", 0) or 0) > 0:
        # report.FLAGS has no code for an inexact half-space cut yet: a warning text
        ctx.rep.warn(f"section cut inexact: {st.get('excess_bodies')} kept body(ies) reach "
                     f"{st.get('half_space_excess_mm')} mm past the plane (tolerance "
                     f"{st.get('half_space_excess_tol_mm')} mm)")


# ------------------------------------------------------------ B-rep stage --

def _brep_stage(ctx: _Ctx, shape, lo, hi, *, exact_ok: bool, features_on: bool,
                strict: bool, solids: Optional[int] = None,
                roots: Optional[int] = None) -> None:
    """Features, views and section of an oriented OCCT shape.

    ``strict`` (STL inputs): any failure raises ``_BrepStageError`` so the
    caller falls back to the mesh route. Otherwise a failed view is skipped
    and flagged and a failed section is flagged. ``solids``: the census
    solid count (an auto section is skipped for a model without solid).

    F7-perf: more than ``_ASSEMBLY_MIN_SOLIDS`` solids, or a view with more
    than ``_MAX_HIDDEN_EDGES`` raw hidden HLR edges, is drawn without hidden
    lines (flag ``hidden_lines_omitted_assembly``, info) unless
    ``hidden_lines_force``; ``roots`` is only reported. Each view gets a
    share of the time left before the run deadline (``ctx.remaining()``
    minus a section/sheet reserve) as its ``time_budget_s``; the section is
    skipped (``section_skipped_budget``) when less than the sheet reserve is
    left, and its body projection gets the time left before that reserve.
    """
    import numpy as np
    from . import brep_views
    from .brep_features import cylinder_features
    from .brep_sections import choose_section_plane_brep_info, section_brep
    from .projection import VIEWS

    opts = ctx.options
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    ctx.extents = hi - lo
    ctx.bounds = np.array([lo, hi])
    scale = _scale_guess(ctx, lo, hi)
    ctx.scale_guess = scale
    defl = _deflection(scale)
    faces = None
    try:
        from .brep_load import _count_unique
        from OCP.TopAbs import TopAbs_FACE
        faces = _count_unique(shape, TopAbs_FACE)
    except Exception as exc:
        logger.debug("face count failed: %s", exc)
    if faces == 0:
        raise (_BrepStageError if strict else _Failure)("the B-rep has no faces")
    exact = bool(exact_ok and (faces is None or faces <= opts.exact_hlr_max_faces))
    if exact_ok and not exact:
        ctx.rep.flag("hlr_poly_used", "info", f"{faces} faces > exact_hlr_max_faces "
                     f"{opts.exact_hlr_max_faces}: polygonal HLR on a meshed copy")
    ctx.say(f"B-rep views: {'exact' if exact else 'poly'} HLR, deflection {defl:.4g} mm "
            f"(scale guess {scale:g})")

    if features_on:
        tf = time.perf_counter()
        try:
            ctx.features = cylinder_features(shape)
        except Exception as exc:
            ctx.rep.warn(f"cylinder feature detection failed: {exc}")
            ctx.features = []
        ctx.timing["features_s"] = ctx.timing.get("features_s", 0.0) + (time.perf_counter() - tf)
        kinds = [f.kind for f in ctx.features]
        ctx.rep.data["geometry"]["features"] = {"holes": kinds.count("hole"),
                                                "bosses": kinds.count("boss"),
                                                "fillets": kinds.count("fillet")}
        if kinds.count("filled_hole"):
            ctx.rep.data["geometry"]["features"]["filled_holes"] = kinds.count("filled_hole")
    else:
        # detection skipped (faceted body): "unknown", not "none" - same as the mesh route
        ctx.features = []
        ctx.rep.data["geometry"]["features"] = {"holes": None, "bosses": None, "fillets": None}

    # large assembly (F7-perf item C): no hidden lines unless forced. The STEP root
    # count is not used: Inventor and STEPControl assemblies arrive as ONE root
    # (the 211-solid RC buggy suspension has roots == 1)
    hidden_on = bool(opts.hidden_lines)
    # QA round 2 issues 5 / 11: the same gate also drops the hole dimensions and the
    # auto section (_sheet_and_write, below)
    ctx.assembly = bool(solids is not None and int(solids) > _ASSEMBLY_MIN_SOLIDS)
    if hidden_on and not opts.hidden_lines_force and ctx.assembly:
        hidden_on = False
        ctx.rep.flag("hidden_lines_omitted_assembly", "info",
                     f"{int(solids)} solids > {_ASSEMBLY_MIN_SOLIDS}"
                     + (f" ({roots} root(s))" if roots is not None else "")
                     + ": large assembly drawn without hidden lines (--hidden-lines-force "
                       "keeps them)")
    max_hidden_edges = None if opts.hidden_lines_force else _MAX_HIDDEN_EDGES

    t0 = time.perf_counter()
    views_base = ctx.timing["views_s"]
    for i, name in enumerate(opts.views):
        spec = VIEWS[name]
        ctx.timing["views_s"] = views_base + (time.perf_counter() - t0)   # live for progress
        ctx.progress("views", view=name)
        tv = time.perf_counter()
        # run deadline (item D): this view's share of the time left before the
        # section / sheet reserve; never above hlr_time_budget_s, and a native HLR
        # step between the share and hlr_time_budget_s keeps the visible lines
        n_left = len(opts.views) - i
        share = (ctx.remaining() - ctx.reserve(_RESERVE_FRACTION)) / n_left
        budget = float(min(opts.hlr_time_budget_s, max(share, _MIN_VIEW_BUDGET_S)))
        try:
            pv = brep_views.project_brep(shape, spec, exact=exact, deflection=defl,
                                         hidden=bool(hidden_on and name != "iso"),
                                         time_budget_s=budget,
                                         paper_scale=scale,
                                         max_hidden_primitives=opts.max_hidden_primitives,
                                         max_hidden_edges=max_hidden_edges,
                                         hlr_raise_after_s=opts.hlr_time_budget_s,
                                         mesh_cache=ctx.mesh_cache)
            if pv.stats is not None:
                pv.stats["time_budget_s"] = round(budget, 3)
                if opts.hidden_lines and not hidden_on and name != "iso":
                    pv.stats["hidden_omitted"] = "assembly"
        except TimeoutError as exc:
            if strict:
                raise _BrepStageError(f"view {name}: {exc}") from exc
            ctx.rep.flag("hlr_timeout", "warning", f"view {name}: {exc}")
            continue
        except Exception as exc:
            if strict:
                raise _BrepStageError(f"view {name}: {type(exc).__name__}: {exc}") from exc
            ctx.rep.flag("view_failed", "warning", f"view {name}: {type(exc).__name__}: {exc}")
            continue
        if pv.bbox is None:
            if strict:
                raise _BrepStageError(f"view {name}: HLR returned no edges")
            ctx.rep.flag("view_failed", "warning", f"view {name}: HLR returned no edges")
            continue
        ctx.views[name] = pv
        _view_flags(ctx, name, pv)
        ctx.rep.data["views"].append(_view_entry(name, pv, time.perf_counter() - tv))
    ctx.timing["views_s"] = views_base + (time.perf_counter() - t0)
    if not ctx.views:
        raise (_BrepStageError if strict else _Failure)("no view could be drawn")

    if opts.section == "off":
        return
    if opts.section == "auto" and ctx.assembly:
        # QA round 2 issue 11: the auto section of a large assembly cost 30-37 s on the
        # 211-solid RC buggy and came out all black; an explicit plane still draws it
        ctx.rep.flag("section_skipped_assembly", "info",
                     f"{int(solids)} solids > {_ASSEMBLY_MIN_SOLIDS}: no automatic section of a "
                     f"large assembly (--section front|top|right draws one)")
        ctx.rep.data["section"] = None
        ctx.section = None
        return
    ctx.progress("section")
    t1 = time.perf_counter()
    write_reserve = ctx.reserve(_WRITE_RESERVE_FRACTION)
    if ctx.remaining() < write_reserve:
        # run deadline (item D): the drawing without the section beats a killed run
        ctx.rep.flag("section_skipped_budget", "warning",
                     f"{max(ctx.remaining(), 0.0):.1f} s left before the run deadline "
                     f"(0.8 x timeout {opts.timeout_s:g} s) < {write_reserve:.1f} s kept for the "
                     f"sheet: no section view drawn")
        ctx.rep.data["section"] = None
        ctx.section = None
        return
    # QA round 2 issue 11: the section's own cost, estimated from the mean view time
    mean_view_s = (ctx.timing["views_s"] - views_base) / max(len(opts.views), 1)
    estimate = _SECTION_COST_VIEWS * mean_view_s
    if ctx.remaining() - write_reserve < estimate:
        ctx.rep.flag("section_skipped_budget", "warning",
                     f"estimated section cost {estimate:.1f} s ({_SECTION_COST_VIEWS:g} x the mean "
                     f"view time {mean_view_s:.1f} s) > {max(ctx.remaining() - write_reserve, 0.0):.1f}"
                     f" s left before the sheet reserve (run deadline 0.8 x timeout "
                     f"{opts.timeout_s:g} s): no section view drawn")
        ctx.rep.data["section"] = None
        ctx.section = None
        return
    try:
        census = {"bbox_min": lo.tolist(), "bbox_max": hi.tolist()}
        if solids is not None:
            census["solids"] = int(solids)
        model = types.SimpleNamespace(census=census, shape=shape)
        point, normal, marker, info = choose_section_plane_brep_info(model, ctx.features,
                                                                     opts.section)
        if opts.section == "auto" and info.get("skip_hint") == "no_solid":
            ctx.rep.flag("section_fallback", "info", "no solid: an open surface model has no "
                         "closed section; no section view drawn")
            ctx.rep.data["section"] = None
            ctx.section = None
        else:
            spec = _section_spec(normal)
            depth = float(abs(np.dot(hi - lo, np.asarray(normal, float))))
            # features: the fasteners (filled-hole partners) cut lengthwise stay whole
            # (QA round 2 issue 28; section_brep picks them from the plane normal)
            sec = section_brep(shape, point, normal, spec, exact=exact, deflection=defl,
                               depth_extent=depth, hidden=False,
                               time_budget_s=max(ctx.remaining() - write_reserve,
                                                 _MIN_VIEW_BUDGET_S),
                               features=ctx.features or None)
            sec.marker_view = marker
            _section_report(ctx, sec, info)
            if info.get("reason") == "bbox_centre_fallback":
                ctx.rep.flag("section_fallback", "info",
                             "no hole axis fits the section mode: the plane goes through the "
                             "bbox centre")
            elif info.get("reason") == "thin_part":
                ctx.rep.flag("section_fallback", "info",
                             "no hole axis fits the section mode and the part is thin: the top "
                             "plane goes through the bbox centre")
            ctx.section = sec if sec.loops else None
    except Exception as exc:
        if strict:
            raise _BrepStageError(f"section: {type(exc).__name__}: {exc}") from exc
        ctx.rep.flag("section_failed", "warning", f"{type(exc).__name__}: {exc}")
        ctx.section = None
    ctx.timing["section_s"] += time.perf_counter() - t1


# ------------------------------------------------------------- mesh stage --

def _mesh_stage(ctx: _Ctx, mesh, organic: bool) -> None:
    import numpy as np
    from .mesh_sections import choose_section_plane_mesh, section_mesh
    from .mesh_views import project_mesh
    from .projection import VIEWS

    opts = ctx.options
    bounds = np.asarray(mesh.bounds, float)
    ctx.extents = bounds[1] - bounds[0]
    ctx.bounds = bounds.copy()
    scale = _scale_guess(ctx, bounds[0], bounds[1])
    ctx.scale_guess = scale
    ctx.features = []
    ctx.views = {}
    ctx.section = None
    ctx.assembly = False                     # no hole features on this route anyway
    ctx.rep.data["views"] = []
    ctx.rep.data["section"] = None
    ctx.rep.data["geometry"]["features"] = {"holes": None, "bosses": None, "fillets": None}
    ctx.rep.flag("holes_not_detected_mesh_route", "info",
                 "the mesh route has no hole features: no diameter callouts")
    ctx.say(f"mesh views: depth buffer {opts.raster_px} px, organic={organic}, "
            f"scale guess {scale:g}")
    t0 = time.perf_counter()
    views_base = ctx.timing["views_s"]
    for name in opts.views:
        ctx.timing["views_s"] = views_base + (time.perf_counter() - t0)   # live for progress
        ctx.progress("views", view=name)
        tv = time.perf_counter()
        try:
            pv = project_mesh(mesh, VIEWS[name], raster_px=opts.raster_px,
                              feature_angle_deg=opts.feature_angle_deg, organic=organic,
                              hidden=bool(opts.hidden_lines and name != "iso"),
                              paper_scale=scale, time_budget_s=opts.hlr_time_budget_s)
        except Exception as exc:
            ctx.rep.flag("view_failed", "warning", f"view {name}: {type(exc).__name__}: {exc}")
            continue
        if pv.bbox is None:
            ctx.rep.flag("view_failed", "warning", f"view {name}: nothing projected")
            continue
        ctx.views[name] = pv
        _view_flags(ctx, name, pv)
        ctx.rep.data["views"].append(_view_entry(name, pv, time.perf_counter() - tv))
    ctx.timing["views_s"] = views_base + (time.perf_counter() - t0)
    if not ctx.views:
        raise _Failure("no view could be drawn from the mesh")

    if opts.section == "off":
        return
    if ctx.skip_section:
        logger.info("section skipped: %s", ctx.skip_section)
        return
    ctx.progress("section")
    t1 = time.perf_counter()
    try:
        point, normal, marker = choose_section_plane_mesh(mesh, opts.section)
        spec = _section_spec(normal)
        depth = float(abs(np.dot(ctx.extents, np.asarray(normal, float))))
        sec = section_mesh(mesh, point, normal, spec, raster_px=opts.raster_px,
                           feature_angle_deg=opts.feature_angle_deg, organic=organic,
                           depth_extent=depth)
        sec.marker_view = marker
        _section_report(ctx, sec, {"reason": "bbox_centre"})
        ctx.section = sec if sec.loops else None
    except Exception as exc:
        ctx.rep.flag("section_failed", "warning", f"{type(exc).__name__}: {exc}")
        ctx.section = None
    ctx.timing["section_s"] += time.perf_counter() - t1


# ----------------------------------------------------------------- inputs --

def _unit_check(ctx: _Ctx, extents_mm, declared: Optional[str]) -> None:
    """``unit_plausibility`` on the extents in mm; ``unit_suspect`` when implausible."""
    try:
        from .mesh_load import unit_plausibility
        verdict, detail, alt = unit_plausibility(list(map(float, extents_mm)),
                                                 declared_unit=declared)
    except Exception as exc:
        logger.debug("unit plausibility failed: %s", exc)
        return
    ctx.rep.data["input"]["unit_plausibility"] = {"verdict": verdict, "detail": detail,
                                                  "alternative": alt}
    if verdict == "suspicious":
        ctx.rep.flag("unit_suspect", "warning", detail)


def _run_step(ctx: _Ctx) -> None:
    import numpy as np
    from .brep_load import TessellatedOnlyError, load_step, transform_shape, exact_bbox

    t0 = time.perf_counter()
    try:
        # the OCP-free text sniff first, so a child killed during the load still
        # reports a unit (progress file, QA issue 17); load_step replaces it
        from .brep_load import step_unit_sniff
        ctx.rep.data["input"]["source_unit"] = step_unit_sniff(ctx.input_path)
    except Exception as exc:
        logger.debug("unit sniff failed: %s", exc)
    ctx.progress("load")
    try:
        # above the exact-HLR face gate: per-unique-solid BRepCheck / volume and the
        # bounded exact bbox (brep_load._census, F7-perf item B)
        model = load_step(ctx.input_path,
                          fast_census_above_faces=ctx.options.exact_hlr_max_faces)
    except TessellatedOnlyError as exc:
        ctx.timing["load_s"] += time.perf_counter() - t0
        _run_step_tessellated(ctx, exc)
        return
    except ValueError as exc:
        raise _Failure(str(exc)) from exc
    rep, c = ctx.rep, model.census
    rep.data["input"]["source_unit"] = model.source_unit
    rep.data["input"]["unit_factor"] = model.source_unit_factor
    rep.data["route"].update(pipeline="brep", source="step")
    ctx.source = "step"
    g = rep.data["geometry"]
    for key in ("bbox_min", "bbox_max", "volume_mm3", "area_mm2", "solids", "shells", "faces",
                "edges", "face_types", "brepcheck_valid", "free_edges", "closed",
                "loose_faces", "volume_scope"):
        g[key] = c.get(key)
    for key in ("census_mode", "bbox_source", "solids_unique", "solid_instances",
                "volume_method"):
        if c.get(key) is not None:
            g[key] = c.get(key)
    bodies = dict(model.bodies or {})
    if bodies:
        g["bodies"] = bodies
    for w in model.warnings:
        rep.warn(w)
    dropped = bodies.get("dropped") or {}
    n_edges, n_verts = int(dropped.get("loose_edges", 0)), int(dropped.get("loose_vertices", 0))
    if n_edges or n_verts:
        rep.flag("loose_curves_ignored", "info",
                 f"{n_edges} loose edge(s), {n_verts} loose vertex(es) (curve sets, sketch or "
                 f"PMI geometry) not drawn and not part of the bbox")
    supp = {k: int(dropped.get(k, 0)) for k in ("free_shells", "free_faces", "tessellated_bodies")
            if dropped.get(k)}
    if supp:
        rep.flag("supplemental_shells_ignored", "info",
                 ", ".join(f"{n} {k.replace('_', ' ')}" for k, n in supp.items())
                 + f" next to the {bodies.get('drawn') or 'drawn bodies'} not drawn")
    if (c.get("solids") or 0) > 1:
        rep.flag("multiple_bodies", "warning",
                 f"{model.roots} root(s), {c.get('solids')} solid(s); drawn as one part")
    if c.get("brepcheck_valid") is False:
        rep.flag("brepcheck_invalid", "warning", "BRepCheck_Analyzer reports the shape invalid")
    if (c.get("solids") or 0) == 0 or (c.get("free_edges") or 0) > 0:
        rep.flag("open_shell", "warning",
                 f"{c.get('solids')} solid(s), {c.get('free_edges')} free edge(s)")
    if c.get("inverted"):
        rep.flag("inverted_solid", "warning", f"signed volume {c.get('volume_mm3')}")
    if model.source_unit != "mm":
        rep.flag("unit_not_mm", "warning", f"source length unit {model.source_unit}; OCCT "
                 f"converted the geometry to mm on read")
    if c.get("bbox_min") is None:
        raise _Failure("the shape has no bounding box")
    lo, hi = np.asarray(c["bbox_min"], float), np.asarray(c["bbox_max"], float)
    g["extents"] = (hi - lo).tolist()
    _unit_check(ctx, hi - lo, model.source_unit)

    if c.get("solids") is not None:
        ctx.n_bodies = int(c.get("solids") or 0)
    o = _orientation(ctx, hi - lo)
    shape = model.shape
    o, depth = _depth_sign_brep(ctx, shape, o, c.get("faces"), bbox=(lo.copy(), hi.copy()))
    _record_orientation(ctx, o, depth)
    if not _is_identity(o.matrix):
        try:
            shape = transform_shape(shape, o.matrix)
        except ValueError as exc:
            raise _Failure(f"orientation failed: {exc}") from exc
        mapped = _permuted_box(o.matrix, lo, hi)
        if mapped is not None:
            # a signed axis permutation maps the exact file box onto the exact
            # oriented box: no second AddOptimal (29 s on the 10665-face RC buggy)
            lo, hi = mapped
        else:
            bb = exact_bbox(shape)
            if bb is None:
                raise _Failure("the oriented shape has no bounding box")
            lo, hi = bb[0], bb[1]
    g["drawing_bbox_min"], g["drawing_bbox_max"] = list(map(float, lo)), list(map(float, hi))
    ctx.timing["load_s"] += time.perf_counter() - t0
    ctx.say(f"STEP loaded: {c.get('faces')} faces, unit {model.source_unit}"
            + (", depth turned 180 deg" if depth.get("depth_flipped") else ""))
    ctx.progress("loaded")
    _brep_stage(ctx, shape, lo, hi, exact_ok=True, features_on=True, strict=False,
                solids=c.get("solids"), roots=model.roots)


def _run_step_tessellated(ctx: _Ctx, exc) -> None:
    """A tessellated-only STEP (QA issue 16): draw its triangles on the mesh route."""
    import numpy as np
    from .mesh_load import classify, mesh_census, transform_mesh

    rep, opts = ctx.rep, ctx.options
    rep.data["input"]["source_unit"] = getattr(exc, "source_unit", None)
    rep.data["input"]["unit_factor"] = getattr(exc, "source_unit_factor", None)
    mesh = getattr(exc, "mesh", None)
    if mesh is None or len(getattr(mesh, "faces", [])) == 0:
        raise _Failure(f"{exc}; no triangles could be extracted")
    t0 = time.perf_counter()
    rep.flag("step_tessellated_only", "warning",
             f"{getattr(exc, 'faces', 0)} tessellated face(s), {getattr(exc, 'triangles', 0)} "
             f"triangle(s) without B-rep surfaces: drawn from the triangles (mesh route)")
    rep.fallback("brep", "mesh", "tessellated-only STEP (no B-rep surfaces)")
    if getattr(exc, "source_unit", "mm") not in ("mm", None):
        rep.flag("unit_not_mm", "warning", f"source length unit {exc.source_unit}; OCCT "
                 f"converted the geometry to mm on read")
    # QA round 2 issue 24: what the triangles leave out, as on the B-rep route
    rep.data["route"]["stl_to_step"]["skipped_reason"] = "tessellated_step"
    bodies = dict(getattr(exc, "bodies", None) or {})
    dropped = {k: int(v) for k, v in (getattr(exc, "dropped", None) or {}).items()
               if isinstance(v, (int, float)) and int(v) > 0}
    bodies["dropped"] = dropped
    rep.data["geometry"]["bodies"] = bodies
    n_shells = int(dropped.get("tessellated_shells_outside", 0) or 0)
    supp = {k: int(dropped.get(k, 0)) for k in ("free_shells", "free_faces", "tessellated_bodies")
            if dropped.get(k)}
    if n_shells or supp:
        parts = [f"{n_shells} tessellated solid(s) / shell(s) outside the drawn part"] \
            if n_shells else []
        parts += [f"{n} {k.replace('_', ' ')}" for k, n in supp.items() if not n_shells]
        rep.flag("supplemental_shells_ignored", "info", ", ".join(parts) + " not drawn")
    n_ann = int(dropped.get("annotation_occurrences", 0) or 0)
    n_edges = int(dropped.get("loose_edges", 0) or 0)
    n_verts = int(dropped.get("loose_vertices", 0) or 0)
    if n_ann or n_edges or n_verts:
        rep.flag("loose_curves_ignored", "info",
                 ", ".join(t for t in (
                     f"{n_ann} tessellated PMI annotation occurrence(s)" if n_ann else "",
                     f"{n_edges} loose edge(s)" if n_edges else "",
                     f"{n_verts} loose vertex(es)" if n_verts else "") if t)
                 + " not drawn and not part of the bbox")
    census = mesh_census(mesh)
    cls = classify(mesh, opts.classifier_threshold, census=census)
    rep.data["route"]["classifier"] = dict(cls)
    _mesh_geometry(ctx, census)
    organic, not_solid = _mesh_flags(ctx, census, cls)
    _unit_check(ctx, census.get("extents") or [0, 0, 0], getattr(exc, "source_unit", None))
    bounds = np.asarray(mesh.bounds, float)
    o = _orientation(ctx, bounds[1] - bounds[0])
    o, depth = _depth_sign_mesh(ctx, mesh, o)
    _record_orientation(ctx, o, depth)
    mesh_o = mesh if _is_identity(o.matrix) else transform_mesh(mesh, o.matrix)
    ob = np.asarray(mesh_o.bounds, float)
    rep.data["geometry"]["drawing_bbox_min"] = ob[0].tolist()
    rep.data["geometry"]["drawing_bbox_max"] = ob[1].tolist()
    rep.data["route"].update(pipeline="mesh", source="step_tessellated_mesh")
    ctx.source = "step_tessellated_mesh"
    ctx.timing["load_s"] += time.perf_counter() - t0
    ctx.progress("loaded")
    _mesh_stage(ctx, mesh_o, organic)


def _mesh_geometry(ctx: _Ctx, census: dict) -> None:
    g = ctx.rep.data["geometry"]
    for key in ("bbox_min", "bbox_max", "extents", "volume_mm3", "area_mm2", "triangles",
                "watertight", "components", "volume_approximate", "volume_source"):
        g[key] = census.get(key)
    if census.get("boundary_edges") is not None:
        g["closed"] = int(census.get("boundary_edges") or 0) == 0


def _mesh_flags(ctx: _Ctx, census: dict, cls: dict) -> Tuple[bool, bool]:
    """Census / classifier flags of a mesh; returns ``(organic, not_solid)``.

    ``organic`` is True for ``organic`` AND ``not_solid`` verdicts (the mesh
    route without conversion); ``not_solid`` also skips the section and the
    overall dimensions and caps the level at minimal (flag ``not_a_solid``).
    """
    from .mesh_load import watertight_detail
    rep = ctx.rep
    if not census.get("watertight"):
        rep.flag("mesh_not_watertight", "warning", watertight_detail(census))
    if int(census.get("nonmanifold_edges") or 0) > 0:
        rep.flag("mesh_non_manifold", "warning",
                 f"{census.get('nonmanifold_edges')} non-manifold edge(s) (more than two "
                 f"faces on one edge)")
    def g(v):
        return "n/a" if v is None else f"{float(v):.6g}"

    conflicts = int(census.get("winding_conflicts") or 0)
    raw_signed = census.get("volume_raw_signed_mm3")
    if "inverted_solid" in census:
        inverted_solid = bool(census.get("inverted_solid"))
    else:                                    # an older census: the round-1 rule
        inverted_solid = bool(census.get("inverted")) and conflicts == 0
    if inverted_solid:
        # QA round 2 issue 8: mesh_load.inverted_solid_verdict (no conflict, a few, or a
        # negative raw sum matching the repaired volume: 28's 2 conflicts in 451k edges)
        rep.flag("inverted_solid", "warning",
                 f"the mesh is inside out (raw signed volume {g(raw_signed)} mm3, "
                 f"{conflicts} winding conflict(s)); |volume| {g(census.get('volume_mm3'))} mm3 "
                 f"reported")
    elif census.get("inverted") and conflicts > 0:
        # the "inverted" verdict of a badly wound mesh only says its dominant piece is
        # wound inward: the defect is the inconsistent winding, not an inside-out solid
        rep.warn(f"face winding is inconsistent ({conflicts} edge(s) whose two faces run it "
                 f"in the same direction): the volume {g(census.get('volume_mm3'))} mm3 "
                 f"comes from a winding-repaired copy (raw signed sum {g(raw_signed)} mm3)")
    if conflicts > 0 and census.get("volume_repaired"):
        rep.flag("winding_repaired", "info",
                 f"{conflicts} winding conflict(s): volume {g(census.get('volume_mm3'))} mm3 "
                 f"from a winding-repaired copy (raw signed sum {g(raw_signed)} mm3)")
    comps = int(census.get("components") or 0)
    if comps > 0:
        ctx.n_bodies = comps
    if comps > 1:
        rep.flag("multiple_bodies", "warning", f"{comps} mesh components (bodies of the vertex "
                 f"graph); drawn as one part")
    verdict = cls.get("verdict")
    not_solid = verdict == "not_solid"
    organic = verdict in ("organic", "not_solid")
    if verdict == "organic":
        rep.flag("organic_classified", "info",
                 f"planar_big {cls.get('planar_big', 0.0):.3f} < threshold {cls.get('threshold')}")
    if not_solid:
        reason = cls.get("not_solid_reason") or "no closed component"
        rep.flag("not_a_solid", "warning", f"{reason}: no section, no overall dimensions")
        ctx.skip_section = "not_a_solid"
        ctx.skip_overall_dims = "not_a_solid"
    return organic, not_solid


def _scaled_census(census: dict, f: float) -> dict:
    """The census of a mesh scaled by ``f`` (lengths f, areas f^2, volumes f^3)."""
    c = dict(census)
    for key in ("bbox_min", "bbox_max", "extents"):
        if c.get(key) is not None:
            c[key] = [float(v) * f for v in c[key]]
    if c.get("area_mm2") is not None:
        c["area_mm2"] = float(c["area_mm2"]) * f * f
    for key in ("volume_mm3", "volume_raw_mm3", "volume_raw_signed_mm3"):
        if c.get(key) is not None:
            c[key] = float(c[key]) * f ** 3
    return c


def _faceted(ctx: _Ctx, mesh_file, mesh_o, census: Optional[dict] = None) -> Tuple[Any, Any, Any]:
    """Write the faceted STEP (file coordinates) and build the drawn body (oriented)."""
    import numpy as np
    from src.io.faceted_step import export_step_faceted
    from .brep_load import faceted_shape_from_mesh

    st = ctx.rep.data["route"]["stl_to_step"]
    st["attempted"] = True
    step_path = _out_path(ctx, "step")
    t0 = time.perf_counter()
    report = export_step_faceted(mesh_file, step_path, verify=False)
    st["seconds"] = round(time.perf_counter() - t0, 3)
    st["audit"] = {k: report.get(k) for k in ("faces", "solids", "shells", "open_shells",
                                              "components", "skipped_degenerate",
                                              "inverted_components", "volume_mesh_mm3",
                                              "volume_brep_mm3", "volume_delta_pct",
                                              "brepcheck_valid", "file_bytes")}
    for w in report.get("warnings", []) or []:
        ctx.rep.warn(f"faceted STEP: {w}")
    if not report.get("ok"):
        st["ok"] = False
        st["error"] = report.get("error") or "faceted export failed"
        raise _BrepStageError(f"faceted STEP export failed: {st['error']}")
    st["ok"] = True
    ctx.rep.data["outputs"]["step"] = step_path
    g = ctx.rep.data["geometry"]
    g["faces"], g["solids"], g["shells"] = report.get("faces"), report.get("solids"), report.get("shells")
    g["brepcheck_valid"] = report.get("brepcheck_valid")
    if report.get("faces") is not None:
        g["face_types"] = {"Plane": int(report.get("faces") or 0)}
    if report.get("skipped_degenerate"):
        ctx.rep.flag("degenerate_faces_removed", "info",
                     f"{report['skipped_degenerate']} degenerate triangle(s) skipped")

    def warn(msg):
        logger.info("faceted body: %s", msg)

    shape = faceted_shape_from_mesh(mesh_o, math.radians(ctx.options.facet_smooth_angle_deg), warn)
    _audit_drawn_body(ctx, shape, mesh_o, census)
    drawn = st["audit"].get("drawn_body") or {}
    if (drawn.get("solids") or 0) > 1 and not ctx.rep.has_flag("multiple_bodies"):
        ctx.rep.flag("multiple_bodies", "warning",
                     f"faceted body: {drawn.get('solids')} solids; drawn as one part")
    if "error" not in drawn:
        if drawn.get("free_edges") is not None:
            g["free_edges"] = int(drawn["free_edges"])
            g["closed"] = bool(drawn["free_edges"] == 0 and (drawn.get("solids") or 0) >= 1)
        if drawn.get("brep_volume_mm3") is not None:
            # the single authoritative volume of this route: the drawn B-rep
            g["volume_mm3"] = abs(float(drawn["brep_volume_mm3"]))
            g["volume_source"] = "faceted_brep"
    bounds = np.asarray(mesh_o.bounds, float)
    ctx.rep.flag("facet_edges_suppressed", "info",
                 f"facet edges below {ctx.options.facet_smooth_angle_deg:g} deg coded smooth "
                 f"(not drawn)")
    return shape, bounds[0], bounds[1]


_AUDIT_VOLUME_TOL_PCT = 1e-3     # the faceted body reproduces the mesh volume to round-off


def _audit_drawn_body(ctx: _Ctx, shape, mesh, census: Optional[dict] = None) -> None:
    """Independent check of the faceted body that is drawn (spec 3.2 in-process audit).

    ``export_step_faceted(verify=False)`` reports on itself only; this runs
    ``src.io.step_audit.measure_shape`` on the OCCT body handed to the HLR
    and compares its volume with the WINDING-REPAIRED mesh volume
    (``census["volume_mm3"]`` = ``mesh_load.mesh_volume_repaired``; the raw
    signed sum of a badly wound multi-body mesh is meaningless, QA issue
    12). Measured cost: 0.45 s for the 1544-face plate, 6.9 s for a
    20480-face icosphere (BRepCheck dominates). A failed measurement is
    logged and leaves ``None`` values; a body that disagrees with a closed
    mesh is flagged ``faceted_audit_mismatch`` (warning): the topology
    criteria need a watertight mesh, the volume criterion a reference
    volume that is not approximate. Never raises.
    """
    st = ctx.rep.data["route"]["stl_to_step"]
    t0 = time.perf_counter()
    try:
        from src.io.step_audit import measure_shape
        m = measure_shape(shape, deviation_samples=0)
    except Exception as exc:
        logger.warning("audit of the faceted body failed: %s", exc)
        st["audit"]["drawn_body"] = {"error": f"{type(exc).__name__}: {exc}"}
        return
    audit = {k: m.get(k) for k in ("readable", "faces", "solids", "shells", "free_edges",
                                   "brepcheck_valid", "brep_volume_mm3")}
    if census is None:
        try:
            from .mesh_load import mesh_census
            census = mesh_census(mesh)
        except Exception as exc:
            logger.warning("mesh census for the audit failed: %s", exc)
            census = {}
    watertight = bool(census.get("watertight", getattr(mesh, "is_watertight", False)))
    ref = census.get("volume_mm3")
    mesh_vol = abs(float(ref)) if ref is not None else None
    approximate = bool(census.get("volume_approximate"))
    audit["mesh_volume_mm3"] = mesh_vol
    audit["mesh_volume_source"] = census.get("volume_source")
    vol = m.get("brep_volume_mm3")
    audit["volume_delta_pct"] = (100.0 * (abs(vol) - mesh_vol) / mesh_vol
                                 if vol is not None and mesh_vol else None)
    audit["seconds"] = round(time.perf_counter() - t0, 3)
    st["audit"]["drawn_body"] = audit
    problems = []
    if watertight:
        if not (audit["solids"] or 0) >= 1:
            problems.append(f"{audit['solids']} solid(s)")
        if audit["free_edges"] is None or audit["free_edges"] != 0:
            problems.append(f"free edges {audit['free_edges']}")
        if not audit["brepcheck_valid"]:
            problems.append("BRepCheck invalid")
    if mesh_vol is not None and not approximate:
        d = audit["volume_delta_pct"]
        if d is None or not abs(d) <= _AUDIT_VOLUME_TOL_PCT:
            problems.append(f"volume delta {d} %")
    if problems:
        ctx.rep.flag("faceted_audit_mismatch", "warning",
                     "drawn faceted body vs closed mesh: " + "; ".join(problems))


def _nurbs(ctx: _Ctx, mesh_file, o) -> Tuple[Any, Any, Any]:
    """Smooth NURBS conversion (the calls of ``src.convert.convert``)."""
    from src.core.halfedge_mesh import HalfEdgeMesh
    from src.io.exporters import export_step
    from src.nurbs.converter import SubDToNURBSConverter
    from src.reverse_engineering.quad_wrap import QuadWrapper
    from .brep_load import exact_bbox, transform_shape

    st = ctx.rep.data["route"]["stl_to_step"]
    st["attempted"] = True
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(sys.stderr):
        dense = HalfEdgeMesh.from_trimesh(mesh_file)
        cage = QuadWrapper(target_face_count=2000, smoothing_weight=0.5).wrap(dense)
        # the cage checks of src/convert.py: no quad -> fail, boundary edges -> warn
        n_quads = sum(1 for f in cage.faces if len(cage.get_face_vertices(f)) == 4)
        n_boundary = sum(1 for e in cage.edges if cage.is_boundary_edge(e))
        st["audit"]["cage"] = {"vertices": len(cage.vertices), "faces": len(cage.faces),
                      "quads": n_quads, "boundary_edges": n_boundary}
        if len(cage.faces) == 0 or n_quads == 0:
            st["ok"] = False
            st["error"] = "quad wrap produced no quads"
            raise _BrepStageError(st["error"])
        if n_boundary:
            ctx.rep.warn(f"NURBS cage has {n_boundary} boundary edges; the STEP cannot close")
        conv = SubDToNURBSConverter(continuity="G1", tolerance=1e-4).convert(
            cage, reference_mesh=dense)
    shape = conv.get("shape")
    if shape is None:
        st["ok"] = False
        st["error"] = "NURBS conversion produced no shape"
        raise _BrepStageError(st["error"])
    step_path = _out_path(ctx, "step")
    export_step(shape, step_path)
    st["seconds"] = round(time.perf_counter() - t0, 3)
    st["ok"] = True
    st["audit"]["patches"] = len(conv.get("patches") or [])
    ctx.rep.data["outputs"]["step"] = step_path
    if not _is_identity(o.matrix):
        shape = transform_shape(shape, o.matrix)
    bb = exact_bbox(shape)
    if bb is None:
        raise _BrepStageError("the NURBS shape has no bounding box")
    return shape, bb[0], bb[1]


def _run_stl(ctx: _Ctx) -> None:
    import numpy as np
    from .mesh_load import load_mesh_model, transform_mesh

    opts, rep = ctx.options, ctx.rep
    t0 = time.perf_counter()
    ctx.progress("load")
    try:
        mm = load_mesh_model(ctx.input_path, opts.classifier_threshold)
    except ValueError as exc:
        raise _Failure(str(exc)) from exc
    census, cls = mm.census, mm.classifier
    # STL carries no unit: the given stl_unit (mesh scaled to mm) or assumed mm
    if opts.stl_unit:
        factor = float(STL_UNIT_FACTORS[opts.stl_unit])
        rep.data["input"]["source_unit"] = opts.stl_unit
        if factor != 1.0:
            mm.mesh.apply_scale(factor)
            census = _scaled_census(census, factor)
            mm.census = census
    else:
        factor = 1.0
        rep.data["input"]["source_unit"] = "assumed_mm"
    rep.data["input"]["unit_factor"] = factor
    rep.data["input"]["stl_unit"] = opts.stl_unit
    rep.data["route"]["classifier"] = dict(cls)
    _mesh_geometry(ctx, census)
    conflicts = int(census.get("winding_conflicts") or 0)
    for w in mm.warnings:
        if conflicts and w.startswith("volume computed on a winding-repaired copy"):
            continue                 # the winding_repaired flag of _mesh_flags carries it
        rep.warn(w)
    if not opts.stl_unit:
        rep.flag("unit_assumed_mm", "info", "STL carries no unit: read as mm (--stl-unit "
                 "sets it)")
    _unit_check(ctx, census.get("extents") or [0.0, 0.0, 0.0], opts.stl_unit)
    organic, not_solid = _mesh_flags(ctx, census, cls)
    bounds = np.asarray(mm.mesh.bounds, float)
    o = _orientation(ctx, bounds[1] - bounds[0])
    o, depth = _depth_sign_mesh(ctx, mm.mesh, o)
    _record_orientation(ctx, o, depth)
    mesh_o = mm.mesh if _is_identity(o.matrix) else transform_mesh(mm.mesh, o.matrix)
    ob = np.asarray(mesh_o.bounds, float)
    rep.data["geometry"]["drawing_bbox_min"] = ob[0].tolist()
    rep.data["geometry"]["drawing_bbox_max"] = ob[1].tolist()

    st = rep.data["route"]["stl_to_step"]
    mode = opts.stl_to_step
    tri = int(census.get("triangles") or 0)
    convert = None
    if not_solid:
        st["skipped_reason"] = "not_solid"            # no conversion of a triangle soup
    elif mode == "off":
        st["skipped_reason"] = "off"
    elif mode == "auto":
        if organic:
            st["skipped_reason"] = "organic"
        elif tri > opts.faceted_max_triangles:
            st["skipped_reason"] = "too_many_triangles"
            rep.flag("too_many_triangles", "info",
                     f"{tri} triangles > faceted_max_triangles {opts.faceted_max_triangles}")
        else:
            convert = "faceted"
    else:
        convert = mode                               # "faceted" | "nurbs" (forced)
    ctx.timing["load_s"] += time.perf_counter() - t0
    ctx.progress("loaded")

    if convert is not None:
        t1 = time.perf_counter()
        try:
            if convert == "faceted":
                shape, lo, hi = _faceted(ctx, mm.mesh, mesh_o, census)
                source, exact, features_on = "stl_faceted_step", False, False
            else:
                shape, lo, hi = _nurbs(ctx, mm.mesh, o)
                source, exact, features_on = "stl_nurbs_step", True, True
            ctx.timing["load_s"] += time.perf_counter() - t1
            rep.data["route"].update(pipeline="brep", source=source)
            ctx.source = source
            ctx.say(f"STL converted ({convert}); drawing the B-rep")
            if not features_on:
                rep.flag("holes_not_detected_mesh_route", "info",
                         "faceted body: planar facets only, no hole features")
            _brep_stage(ctx, shape, lo, hi, exact_ok=exact, features_on=features_on,
                        strict=True)
            return
        except Exception as exc:                     # noqa: BLE001 - any failure -> mesh route
            reason = f"{type(exc).__name__}: {exc}" if not isinstance(exc, _BrepStageError) else str(exc)
            if st.get("ok") is None:
                st["ok"] = False
                st["error"] = reason
            rep.fallback("brep", "mesh", reason)
            rep.flag("fallback_to_mesh", "warning", reason)
            rep.data["outputs"]["step"] = rep.data["outputs"].get("step")
            # forget everything the B-rep stage produced
            ctx.views, ctx.section, ctx.features = {}, None, []
            ctx.rep.data["quality"]["flags"] = [
                f for f in ctx.rep.data["quality"]["flags"] if f["code"] not in _BREP_STAGE_FLAGS]
            ctx.timing["load_s"] += time.perf_counter() - t1

    rep.data["route"].update(pipeline="mesh", source="stl_mesh")
    ctx.source = "stl_mesh"
    _mesh_stage(ctx, mesh_o, organic)


# ------------------------------------------------------------------ sheet --

def _assembly_hole_groups(features) -> List[dict]:
    """``undimensioned`` entries ``{kind: hole_group, radius, count, reason}``, one per
    hole radius (0.01 mm groups), for the holes of a large assembly (issue 5): reason
    ``assembly`` for the holes, ``filled_by_fastener`` for the holes a fastener of
    another solid fills (the gate drops both kinds before the planner, which would
    otherwise list the filled ones; review R5)."""
    reasons = {"hole": "assembly", "filled_hole": "filled_by_fastener"}
    groups: Dict[Tuple[str, float], int] = {}
    for f in features or []:
        reason = reasons.get(getattr(f, "kind", None))
        if reason is None:
            continue
        try:
            r = round(float(f.radius), 2)
        except (TypeError, ValueError):
            continue
        groups[(reason, r)] = groups.get((reason, r), 0) + 1
    order = {"assembly": 0, "filled_by_fastener": 1}
    return [{"kind": "hole_group", "radius": r, "count": n, "reason": reason, "view": None}
            for (reason, r), n in sorted(groups.items(), key=lambda kv: (order[kv[0][0]],
                                                                         kv[0][1]))]


def _report_dimensions(rep: Report, plan) -> None:
    rep.data["dimensions"] = [{"kind": d.kind, "view": d.view, "value_mm": float(d.value),
                               "text": d.text, "count": int(d.count)} for d in plan.dims]
    rep.data["undimensioned"] = [dict(u) for u in plan.undimensioned]


def _unit_note(rep: Report) -> Optional[str]:
    """The strip's unit note when the unit is doubtful (QA round 2 issue 30), else None:
    ``unit_assumed_mm`` or ``unit_suspect`` raised as a warning."""
    doubtful = {f["code"] for f in rep.data["quality"]["flags"]
                if f["code"] in ("unit_assumed_mm", "unit_suspect") and f["severity"] == "warning"}
    if not doubtful:
        return None
    src = rep.data["input"].get("source_unit")
    return UNIT_NOTE_ASSUMED if src == "assumed_mm" else UNIT_NOTE_SUSPECT


def _report_section_fill(rep: Report, sl: dict, scale: Optional[float] = None) -> None:
    """Copy the layout's section fill decision into ``report.section`` (issue 21) and
    flag ``section_blackened`` when more than half the cut area is filled solid.

    The layout's ``body_thickness`` is 2A/P measured ON THE SHEET: it goes to
    ``body_thickness_paper_mm``; with the sheet ``scale`` (paper / model) known it is
    also given in model mm (``body_thickness_model_mm``). Review R5: a part whose
    thin bodies are thinner than the fill limit in MODEL mm is thin-walled by design
    (filled at 1:1 too, ISO 128-50): ``thin_walled_by_design`` True and no
    ``section_blackened`` flag; the flag stays for walls the scale blackened."""
    from .style import section_fill_below
    sec = rep.data["section"]
    sec["hatch_segments"] = int(sl.get("hatch_segments") or 0)
    sec["hatch_spacing_mm"] = sl.get("hatch_spacing")
    sec["fill"] = sl.get("fill")
    sec["filled_regions"] = int(sl.get("filled_regions") or 0)
    try:
        frac = float(sl.get("filled_area_fraction") or 0.0)
    except (TypeError, ValueError):
        frac = 0.0
    sec["filled_area_fraction"] = round(frac, 4)
    for key in ("bodies", "thick_core_bodies"):
        if sl.get(key) is not None:
            sec[key] = int(sl.get(key) or 0)
    thin_by_design = False
    try:
        paper = [float(t) for t in (sl.get("body_thickness") or [])]
    except (TypeError, ValueError):
        paper = []
    try:
        s = float(scale) if scale is not None else None
    except (TypeError, ValueError):
        s = None
    if s is not None and not (math.isfinite(s) and s > 0.0):
        s = None
    if paper:
        sec["body_thickness_paper_mm"] = [round(t, 3) for t in paper[:64]]
        if s is not None:
            sec["body_thickness_model_mm"] = [round(t / s, 3) for t in paper[:64]]
            limit = float(section_fill_below())
            thin = [t for t in paper if t < limit]
            thin_by_design = bool(thin) and all(t / s < limit for t in thin)
            sec["thin_walled_by_design"] = thin_by_design
    if frac > _SECTION_BLACKENED_SHARE and not thin_by_design:
        rep.flag("section_blackened", "info",
                 f"{100.0 * frac:.3g} % of the cut area is filled solid (walls thinner than "
                 f"the fill limit at this scale; fill {sl.get('fill')})")


def _sheet_and_write(ctx: _Ctx) -> None:
    from . import dimensions
    from .layout import compose
    from .style import DEFAULT_STYLE

    opts, rep = ctx.options, ctx.rep
    t0 = time.perf_counter()
    ctx.progress("dimensions")
    if ctx.skip_overall_dims:
        # not a solid (triangle soup): no overall dimensions (the mesh route has no others)
        plan = dimensions.DimensionPlan(dims=[], undimensioned=[
            {"kind": "overall", "reason": ctx.skip_overall_dims, "view": None}])
    else:
        features, plan_opts, omitted = ctx.features, opts, []
        if ctx.assembly:
            # QA round 2 issue 5: no hole callouts or position dims across the parts of a
            # large assembly (coaxial bores of different solids 0.1 mm apart gave six
            # position dims for one hub); the omission is disclosed per radius group
            omitted = _assembly_hole_groups(ctx.features)
            features = [f for f in ctx.features if f.kind not in ("hole", "filled_hole")]
            plan_opts = opts.with_overrides(hole_positions=False)
        try:
            plan = dimensions.plan_dimensions(ctx.views, features, plan_opts, DEFAULT_STYLE,
                                              ctx.extents, model_bounds=ctx.bounds,
                                              scale_hint=ctx.scale_guess,
                                              n_bodies=ctx.n_bodies)
        except Exception as exc:      # spec 3.10: the views exist, so the sheet is still written
            logger.exception("dimension planning failed")
            reason = f"{type(exc).__name__}: {exc}"
            rep.warn(f"dimension planning failed: {reason}")
            rep.flag("dimensions_failed", "warning",
                     f"drawing written without dimensions: {reason}")
            plan = dimensions.DimensionPlan(dims=[], undimensioned=[])
        if ctx.assembly:
            plan.undimensioned = list(omitted) + list(plan.undimensioned)
            n_holes = sum(int(u.get("count") or 0) for u in omitted)
            n_filled = sum(int(u.get("count") or 0) for u in omitted
                           if u.get("reason") == "filled_by_fastener")
            filled_txt = (f"; {n_filled} filled by a fastener, reason filled_by_fastener"
                          if n_filled else "")
            rep.flag("dimensions_omitted_assembly", "info",
                     f"{ctx.n_bodies} solids > {_ASSEMBLY_MIN_SOLIDS}: no hole callouts and no "
                     f"position dimensions ({n_holes} hole feature(s) in {len(omitted)} radius "
                     f"group(s), undimensioned reason assembly{filled_txt})")
        hidden_only = int((getattr(plan, "stats", None) or {}).get("hidden_only") or 0)
        if hidden_only > 0:
            # R3: callouts are placed on visible circles only (QA round 2 issue 27)
            rep.flag("dimension_hidden_only", "info",
                     f"{hidden_only} hole feature(s) seen only as hidden circles in their axial "
                     f"view: no callout (see undimensioned)")
    t_dims = time.perf_counter()
    ctx.timing["dimensions_s"] = ctx.timing.get("dimensions_s", 0.0) + (t_dims - t0)
    _report_dimensions(rep, plan)
    capped = [u for u in plan.undimensioned if u.get("reason") == "cap"]
    if capped:
        rep.flag("dimensions_capped", "warning",
                 f"{len(capped)} hole group(s) above max_hole_callouts {opts.max_hole_callouts}")

    ctx.progress("layout")
    meta = {"ident": ctx.stem, "title": opts.title or ctx.stem,
            "input_file": os.path.basename(ctx.input_path),
            "source_note": SOURCE_NOTES.get(ctx.source or "", ""),
            "supplementary_title": SOURCE_NOTES.get(ctx.source or "", "")}
    unit_note = _unit_note(rep)
    if unit_note:
        meta["unit_note"] = unit_note
    lay = compose(ctx.views, ctx.section, plan, opts, DEFAULT_STYLE, meta)
    # the layout may drop position dims to keep the scale (undimensioned reason
    # layout_space, flag layout_dims_dropped): the report lists what the sheet shows
    if getattr(lay, "plan", None) is not None and lay.plan is not plan:
        _report_dimensions(rep, lay.plan)
    # compose already forwards plan.flags (dimension_extent_mismatch): one loop only
    for code, severity, detail in lay.flags:
        rep.flag(code, severity, detail)
    if not lay.sheet.meta.get("approval"):
        rep.flag("approval_blank", "info", "the approval field is blank (auto-generated drawing)")
    m = lay.sheet.meta
    tiny = _views_tiny(ctx, lay)
    if tiny is not None:
        rep.flag("views_tiny", "warning", tiny)
    rep.data["sheet"] = {"size": lay.sheet_size.name, "orientation": lay.sheet_size.orientation,
                         "scale": lay.scale_text, "projection": m.get("projection"),
                         "iso5457": bool(lay.sheet_size.iso5457),
                         "iso_scale": (None if m.get("iso_scale") is None
                                       else _scale_text(m.get("iso_scale"))),
                         "views_placed": sorted(lay.placements)}
    if isinstance(m.get("text_collisions"), dict):
        rep.data["sheet"]["text_collisions"] = dict(m["text_collisions"])
    if unit_note:
        rep.data["sheet"]["unit_note"] = unit_note
    if m.get("scale_candidates"):
        rep.data["sheet"]["scale_candidates"] = [list(c) for c in m["scale_candidates"]]
    if rep.data.get("section") is not None:
        _report_section_fill(rep, m.get("section_layout") or {},
                             scale=getattr(lay, "scale", None))
    t_lay = time.perf_counter()
    ctx.timing["layout_s"] = ctx.timing.get("layout_s", 0.0) + (t_lay - t_dims)

    from . import writers
    ctx.progress("write")
    fns = {"pdf": writers.write_pdf, "svg": writers.write_svg, "dxf": writers.write_dxf}
    written = 0
    os.makedirs(ctx.out_dir, exist_ok=True)
    for fmt in opts.formats:
        path = _out_path(ctx, fmt)
        kw = {}
        if fmt == "dxf":
            # item E: the re-read audit is skipped for huge sheets or near the deadline
            n_items = len(getattr(lay.sheet, "items", None) or [])
            left = ctx.remaining()
            audit = n_items <= _DXF_AUDIT_MAX_ITEMS and left >= _DXF_AUDIT_MIN_REMAINING_S
            kw["audit"] = audit
            w = rep.data.setdefault("writers", {})
            w["dxf_audited"] = False
            w["dxf_items"] = n_items
            if not audit:
                w["dxf_audit_skipped_reason"] = (
                    f"{n_items} sheet items > {_DXF_AUDIT_MAX_ITEMS}"
                    if n_items > _DXF_AUDIT_MAX_ITEMS else
                    f"{max(left, 0.0):.1f} s left before the run deadline < "
                    f"{_DXF_AUDIT_MIN_REMAINING_S:g} s")
        try:
            info = fns[fmt](lay.sheet, path, **kw)
        except Exception as exc:
            rep.flag("writer_failed", "warning", f"{fmt}: {type(exc).__name__}: {exc}")
            continue
        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            rep.flag("writer_failed", "warning", f"{fmt}: no file written")
            continue
        rep.data["outputs"][fmt] = path
        written += 1
        if isinstance(info, dict) and info.get("dash_adapted") is not None:
            # R1 (QA round 2 issue 2): hidden elements whose dash pattern was fitted to
            # their length (DXF: entities with their own ltscale)
            try:
                rep.data.setdefault("writers", {}).setdefault("dash_adapted", {})[fmt] = \
                    int(info["dash_adapted"])
            except (TypeError, ValueError):
                pass
        if fmt == "dxf" and isinstance(info, dict):
            rep.data.setdefault("writers", {})["dxf_audited"] = bool(info.get("audited"))
        if fmt == "dxf" and (info.get("audit_errors") or info.get("audit_fixes")):
            rep.warn(f"dxf audit after re-read: {info.get('audit_errors')} error(s), "
                     f"{info.get('audit_fixes')} fix(es)")
    t_end = time.perf_counter()
    # write_s stays the whole sheet stage (dimensions + layout + writers; review R5)
    ctx.timing["writers_s"] = ctx.timing.get("writers_s", 0.0) + (t_end - t_lay)
    ctx.timing["write_s"] += t_end - t0
    if written == 0:
        raise _Failure("no output file could be written")


_VIEWS_TINY_FILL = 0.10          # placed view geometry / drawing frame area (QA issue 5)


def _views_tiny(ctx: _Ctx, lay) -> Optional[str]:
    """Detail text when the views are tiny on the sheet, else None (QA issue 5 step 5).

    Tiny: the sheet scale is the largest standard scale
    (``style.STANDARD_SCALES[0]``, 50:1) AND the geometry of the placed
    views (their model bboxes at the sheet scale, the iso view at its own
    scale, the section at the sheet scale; dimension bands not counted)
    covers less than ``_VIEWS_TINY_FILL`` of the drawing frame. A part that
    small is usually a unit error (a metre or inch file read as mm).
    """
    try:
        from .layout import _frame_rect
        from .style import DEFAULT_STYLE, STANDARD_SCALES
        top = float(STANDARD_SCALES[0])
        scale = float(lay.scale)
        if scale < top * (1.0 - 1e-9):
            return None
        iso_s = lay.sheet.meta.get("iso_scale")
        iso_s = scale if iso_s is None else float(iso_s)
        area = 0.0
        for name in lay.placements:
            if name == "section":
                body = getattr(ctx.section, "body", None)
                bb = getattr(body, "bbox", None)
                if bb is None and ctx.section is not None and ctx.section.loops:
                    import numpy as np
                    pts = np.vstack([np.asarray(lp, float)[:, :2] for lp in ctx.section.loops])
                    w, h = np.ptp(pts[:, 0]), np.ptp(pts[:, 1])
                    area += float(w * h) * scale * scale
                    continue
            else:
                pv = ctx.views.get(name)
                bb = getattr(pv, "bbox", None)
            if bb is None:
                continue
            s = iso_s if name == "iso" else scale
            area += float(bb.xmax - bb.xmin) * float(bb.ymax - bb.ymin) * s * s
        f = _frame_rect(lay.sheet_size, DEFAULT_STYLE)
        frame = float(f.xmax - f.xmin) * float(f.ymax - f.ymin)
        if not frame > 0.0:
            return None
        fill = area / frame
    except Exception as exc:                   # a missing flag, never a failed run
        logger.debug("views_tiny check failed: %s", exc)
        return None
    if fill >= _VIEWS_TINY_FILL:
        return None
    ext = ctx.extents
    size = ("" if ext is None else
            " (model " + " x ".join(f"{float(v):.4g}" for v in ext) + " mm)")
    return (f"views fill {100.0 * fill:.2g} % of the drawing area at the largest standard "
            f"scale {_scale_text(scale)}{size}: check the unit")


def _scale_text(s) -> Optional[str]:
    from .style import scale_text
    try:
        return scale_text(float(s))
    except Exception:
        return None


# -------------------------------------------------------------------- run --

def input_error_text(s) -> Optional[str]:
    """Error text for a sniffed input that cannot be drawn, else None (QA issue 25).

    The single source of the texts is ``sniff.unreadable_reason``:
    ``"empty STL (0 facets)"``, ``"binary STL truncated: header declares N
    triangles (X bytes), file is Y bytes"`` and the generic "unsupported or
    corrupt input" text; any other kind that is neither STEP nor STL gets
    the generic text too. Kept under this name for backward compatibility.
    Standard library only (the sandbox parent calls it).
    """
    from .sniff import unreadable_reason
    reason = unreadable_reason(s)
    if reason is None and getattr(s, "kind", "unknown") not in ("step", "stl_binary", "stl_ascii"):
        return "unsupported or corrupt input (neither STEP nor STL by content)"
    return reason


def _finish(ctx: _Ctx, ok: bool, error: Optional[str] = None) -> Tuple[int, dict]:
    for k, v in ctx.timing.items():
        ctx.rep.data["timing"][k] = round(float(v), 3)
    code, data = ctx.rep.finish(ok, error)
    try:
        ctx.rep.write(_out_path(ctx, "report.json"))
        data = ctx.rep.data
    except Exception as exc:                          # out_dir not writable: RESULT still says so
        logger.error("report could not be written: %s", exc)
        ctx.rep.data["warnings"].append(f"report could not be written: {exc}")
        data = ctx.rep.data
    ctx.progress("done" if ok else "failed")
    return code, data


def _run(ctx: _Ctx) -> Tuple[int, dict]:
    from .sniff import check_limits, sniff

    opts, rep = ctx.options, ctx.rep
    opts.validate()
    os.makedirs(ctx.out_dir, exist_ok=True)
    t0 = time.perf_counter()
    if not os.path.isfile(ctx.input_path):
        raise _Failure(f"input file not found: {ctx.input_path}")
    s = sniff(ctx.input_path, max_bytes=int(opts.max_input_mb * 2 ** 20))
    rep.data["input"].update(detected_format=s.kind, extension=s.extension or None,
                             size_bytes=s.size_bytes, sha256=s.sha256 or None,
                             triangles_declared=s.triangles)
    ctx.progress("sniff")
    if s.extension_mismatch:
        rep.flag("extension_mismatch", "info", f"content is {s.kind}, extension {s.extension!r}")
    err = check_limits(s, opts.max_input_mb, opts.max_triangles) or input_error_text(s)
    if err:
        raise _Failure(err)
    ctx.timing["load_s"] += time.perf_counter() - t0
    if s.kind == "step":
        ctx.say("input: STEP")
        _run_step(ctx)
    else:
        ctx.say(f"input: {s.kind}, {s.triangles} triangles")
        _run_stl(ctx)
    _sheet_and_write(ctx)
    return _finish(ctx, True)


def run(input_path: str, out_dir: str, options: DrawingOptions, log=None) -> Tuple[int, dict]:
    """Make the drawing of ``input_path`` in ``out_dir``; never raises.

    Returns ``(exit_code, report_dict)``: 0 clean, 2 written with warning
    flags, 1 failed. ``log`` is an optional ``callable(str)`` for progress.
    """
    say = log if callable(log) else (lambda msg: logger.info("%s", msg))
    options = options if options is not None else DrawingOptions()
    rep = Report(input_path, out_dir, options)
    ctx = _Ctx(rep, str(input_path), str(out_dir), options, say)
    try:
        from src.io.occt_utils import quiet_occt  # cheap when OCP is missing: returns False
        quiet_occt()
    except Exception as exc:
        logger.debug("quiet_occt unavailable: %s", exc)
    try:
        return _run(ctx)
    except _Failure as exc:
        say(f"ERROR: {exc}")
        return _finish(ctx, False, str(exc))
    except Exception as exc:  # noqa: BLE001 - the contract is a report, not a traceback
        logger.exception("drawing failed")
        return _finish(ctx, False, f"{type(exc).__name__}: {exc}")


__all__ = ["run", "SOURCE_NOTES", "input_error_text"]
