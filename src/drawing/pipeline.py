"""The router of the drawing module: input -> geometry source -> sheet -> files + report.

``run(input_path, out_dir, options)`` never raises: every stage is wrapped,
failures become ``error`` / ``warnings`` / ``fallbacks`` / quality flags in
the report, and it returns ``(exit_code, report_dict)`` with the
``src/convert.py`` exit codes (0 clean, 2 written with warning flags,
1 failed). The report is also written to ``<out_dir>/<stem>.report.json``.

Order (spec 3.2-3.4): sniff -> limits -> load (B-rep or mesh) -> orientation
(``projection.auto_orientation`` on the file bbox, applied with
``transform_shape`` / ``transform_mesh``) -> bbox of the oriented model ->
cylinder features (true B-rep only) -> the requested views -> section
(unless ``off``) -> ``plan_dimensions`` -> ``compose`` -> writers -> report.

Routing:

* STEP -> ``brep_load.load_step`` -> exact HLR (``HLRBRep_Algo``) up to
  ``exact_hlr_max_faces`` faces, poly HLR above (flag ``hlr_poly_used``). A
  view that fails or exceeds ``hlr_time_budget_s`` is skipped and flagged
  (``view_failed`` / ``hlr_timeout``); no view at all -> exit 1.
* STL -> ``mesh_load.load_mesh_model`` (census + prismatic/organic
  classifier). ``stl_to_step``: ``auto`` converts prismatic meshes with at
  most ``faceted_max_triangles`` triangles to a faceted STEP
  (``src.io.faceted_step.export_step_faceted(verify=False)`` into
  ``<out_dir>/<stem>.step``) and draws the faceted body with the poly HLR
  (``brep_load.faceted_shape_from_mesh`` -> shell / solid with the smooth
  facet edges coded C1, ``project_brep(exact=False)``); otherwise the mesh
  route with ``stl_to_step.skipped_reason`` ``organic`` /
  ``too_many_triangles``. ``faceted`` forces the conversion, ``nurbs`` runs
  the smooth converter of ``src/convert.py`` (QuadWrapper +
  SubDToNURBSConverter, G1, reference fit) and draws that B-rep with exact
  HLR, ``off`` always uses the mesh route. ANY exception of the B-rep stage
  on an STL input (conversion, empty body, a failed or timed-out view, the
  section) falls back to the mesh route (``route.fallbacks``, flag
  ``fallback_to_mesh``).
* Mesh route: ``mesh_views.project_mesh`` (depth buffer) and
  ``mesh_sections``; no hole features (flag ``holes_not_detected_mesh_route``).

The faceted body that is drawn is built from the ORIENTED mesh by
``faceted_shape_from_mesh`` (one face per triangle, smooth edges coded C1:
the measured requirement of the poly HLR), while the STEP file written for
the user is ``export_step_faceted`` of the mesh in its FILE coordinates
(coplanar faces merged). ``export_step_faceted`` does not return its shape,
and the merged faces would lose the regularity coding the poly HLR needs.

HLR discretization: ``deflection = 0.05 / scale_guess`` mm, where
``scale_guess`` is ``layout.choose_scale`` on the bbox-only view boxes of
the oriented model before any HLR (A3 when ``sheet=auto``: its scale is at
least the A4 one, so the deflection is never coarser than needed), clamped
to [0.001, 5] mm.

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

from .options import DrawingOptions
from .report import Report

logger = logging.getLogger(__name__)

SOURCE_NOTES = {"step": "STEP", "stl_faceted_step": "STL->faceted STEP",
                "stl_nurbs_step": "STL->NURBS STEP", "stl_mesh": "STL mesh projection"}
_SECTION_VIEWS = {"auto": "front", "front": "front", "top": "top", "right": "right"}
_DEFLECTION_PER_SCALE = 0.05
_DEFLECTION_RANGE = (1e-3, 5.0)


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
        self.timing: Dict[str, float] = {"load_s": 0.0, "views_s": 0.0, "section_s": 0.0,
                                         "write_s": 0.0}
        self.views: Dict[str, Any] = {}
        self.section = None
        self.features: List[Any] = []
        self.extents = None             # oriented model extents (3,)
        self.source: Optional[str] = None


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


def _corners(lo, hi):
    import numpy as np
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                     for z in (lo[2], hi[2])])


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
    return {"name": name, "algorithm": pv.algorithm, "visible": len(pv.visible),
            "hidden": len(pv.hidden), "seconds": round(float(seconds), 4)}


def _section_spec(normal):
    import numpy as np
    from .projection import VIEWS
    n = np.asarray(normal, float)
    for name in ("front", "top", "right", "back", "bottom", "left"):
        if float(np.dot(n, VIEWS[name].n)) > 1.0 - 1e-9:
            return VIEWS[name]
    raise ValueError(f"section normal {n.tolist()} is not a principal view direction")


def _section_report(ctx: _Ctx, sec) -> None:
    st = sec.stats or {}
    ctx.rep.data["section"] = {
        "label": sec.label, "view": sec.spec.name, "marker_view": sec.marker_view,
        "plane_point": [float(v) for v in sec.plane_point],
        "plane_normal": [float(v) for v in sec.plane_normal],
        "loops": len(sec.loops), "open_wires_dropped": int(st.get("open_wires_dropped", 0) or 0),
        "hatch_segments": 0, "nudged": bool(st.get("nudged", False)),
        "seconds": st.get("seconds")}
    if st.get("nudged"):
        ctx.rep.flag("section_nudged", "info", "the plane gave no closed loop (or lay in a "
                     "planar face) and was moved by 1 % of the depth")
    if int(st.get("open_wires_dropped", 0) or 0) > 0:
        ctx.rep.flag("section_open_wires", "warning",
                     f"{st.get('open_wires_dropped')} open section wire(s) dropped")
    if not sec.loops:
        ctx.rep.flag("section_empty", "warning", "the section plane cuts no closed loop; "
                     "no section view drawn")


# ------------------------------------------------------------ B-rep stage --

def _brep_stage(ctx: _Ctx, shape, lo, hi, *, exact_ok: bool, features_on: bool,
                strict: bool) -> None:
    """Features, views and section of an oriented OCCT shape.

    ``strict`` (STL inputs): any failure raises ``_BrepStageError`` so the
    caller falls back to the mesh route. Otherwise a failed view is skipped
    and flagged and a failed section is flagged.
    """
    import numpy as np
    from . import brep_views
    from .brep_features import cylinder_features
    from .brep_sections import choose_section_plane_brep, section_brep
    from .projection import VIEWS

    opts = ctx.options
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    ctx.extents = hi - lo
    scale = _scale_guess(ctx, lo, hi)
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
        try:
            ctx.features = cylinder_features(shape)
        except Exception as exc:
            ctx.rep.warn(f"cylinder feature detection failed: {exc}")
            ctx.features = []
        kinds = [f.kind for f in ctx.features]
        ctx.rep.data["geometry"]["features"] = {"holes": kinds.count("hole"),
                                                "bosses": kinds.count("boss"),
                                                "fillets": kinds.count("fillet")}
    else:
        # detection skipped (faceted body): "unknown", not "none" - same as the mesh route
        ctx.features = []
        ctx.rep.data["geometry"]["features"] = {"holes": None, "bosses": None, "fillets": None}

    t0 = time.perf_counter()
    for name in opts.views:
        spec = VIEWS[name]
        tv = time.perf_counter()
        try:
            pv = brep_views.project_brep(shape, spec, exact=exact, deflection=defl,
                                         hidden=bool(opts.hidden_lines and name != "iso"),
                                         time_budget_s=opts.hlr_time_budget_s)
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
        ctx.rep.data["views"].append(_view_entry(name, pv, time.perf_counter() - tv))
    ctx.timing["views_s"] += time.perf_counter() - t0
    if not ctx.views:
        raise (_BrepStageError if strict else _Failure)("no view could be drawn")

    if opts.section == "off":
        return
    t1 = time.perf_counter()
    try:
        model = types.SimpleNamespace(census={"bbox_min": lo.tolist(), "bbox_max": hi.tolist()},
                                      shape=shape)
        point, normal, marker = choose_section_plane_brep(model, ctx.features, opts.section)
        spec = _section_spec(normal)
        depth = float(abs(np.dot(hi - lo, np.asarray(normal, float))))
        sec = section_brep(shape, point, normal, spec, exact=exact, deflection=defl,
                           depth_extent=depth, hidden=False)
        sec.marker_view = marker
        _section_report(ctx, sec)
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
    ctx.features = []
    ctx.views = {}
    ctx.section = None
    ctx.rep.data["views"] = []
    ctx.rep.data["section"] = None
    ctx.rep.data["geometry"]["features"] = {"holes": None, "bosses": None, "fillets": None}
    ctx.rep.flag("holes_not_detected_mesh_route", "info",
                 "the mesh route has no hole features: no diameter callouts")
    ctx.say(f"mesh views: depth buffer {opts.raster_px} px, organic={organic}")
    t0 = time.perf_counter()
    for name in opts.views:
        tv = time.perf_counter()
        try:
            pv = project_mesh(mesh, VIEWS[name], raster_px=opts.raster_px,
                              feature_angle_deg=opts.feature_angle_deg, organic=organic,
                              hidden=bool(opts.hidden_lines and name != "iso"))
        except Exception as exc:
            ctx.rep.flag("view_failed", "warning", f"view {name}: {type(exc).__name__}: {exc}")
            continue
        if pv.bbox is None:
            ctx.rep.flag("view_failed", "warning", f"view {name}: nothing projected")
            continue
        ctx.views[name] = pv
        ctx.rep.data["views"].append(_view_entry(name, pv, time.perf_counter() - tv))
    ctx.timing["views_s"] += time.perf_counter() - t0
    if not ctx.views:
        raise _Failure("no view could be drawn from the mesh")

    if opts.section == "off":
        return
    t1 = time.perf_counter()
    try:
        point, normal, marker = choose_section_plane_mesh(mesh, opts.section)
        spec = _section_spec(normal)
        depth = float(abs(np.dot(ctx.extents, np.asarray(normal, float))))
        sec = section_mesh(mesh, point, normal, spec, raster_px=opts.raster_px,
                           feature_angle_deg=opts.feature_angle_deg, organic=organic,
                           depth_extent=depth)
        sec.marker_view = marker
        _section_report(ctx, sec)
        ctx.section = sec if sec.loops else None
    except Exception as exc:
        ctx.rep.flag("section_failed", "warning", f"{type(exc).__name__}: {exc}")
        ctx.section = None
    ctx.timing["section_s"] += time.perf_counter() - t1


# ----------------------------------------------------------------- inputs --

def _run_step(ctx: _Ctx) -> None:
    import numpy as np
    from .brep_load import exact_bbox, load_step, transform_shape

    t0 = time.perf_counter()
    try:
        model = load_step(ctx.input_path)
    except ValueError as exc:
        raise _Failure(str(exc)) from exc
    rep, c = ctx.rep, model.census
    rep.data["input"]["source_unit"] = model.source_unit
    rep.data["route"].update(pipeline="brep", source="step")
    ctx.source = "step"
    g = rep.data["geometry"]
    for key in ("bbox_min", "bbox_max", "volume_mm3", "area_mm2", "solids", "shells", "faces",
                "edges", "face_types", "brepcheck_valid", "free_edges", "closed"):
        g[key] = c.get(key)
    for w in model.warnings:
        rep.warn(w)
    if model.roots > 1 or (c.get("solids") or 0) > 1:
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

    o = _orientation(ctx, hi - lo)
    shape = model.shape
    if not _is_identity(o.matrix):
        try:
            shape = transform_shape(shape, o.matrix)
        except ValueError as exc:
            raise _Failure(f"orientation failed: {exc}") from exc
        bb = exact_bbox(shape)
        if bb is None:
            raise _Failure("the oriented shape has no bounding box")
        lo, hi = bb[0], bb[1]
    g["drawing_bbox_min"], g["drawing_bbox_max"] = list(map(float, lo)), list(map(float, hi))
    ctx.timing["load_s"] += time.perf_counter() - t0
    ctx.say(f"STEP loaded: {c.get('faces')} faces, unit {model.source_unit}")
    _brep_stage(ctx, shape, lo, hi, exact_ok=True, features_on=True, strict=False)


def _mesh_geometry(ctx: _Ctx, census: dict) -> None:
    g = ctx.rep.data["geometry"]
    for key in ("bbox_min", "bbox_max", "extents", "volume_mm3", "area_mm2", "triangles",
                "watertight", "components"):
        g[key] = census.get(key)


def _faceted(ctx: _Ctx, mesh_file, mesh_o) -> Tuple[Any, Any, Any]:
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
    if report.get("skipped_degenerate"):
        ctx.rep.flag("degenerate_faces_removed", "info",
                     f"{report['skipped_degenerate']} degenerate triangle(s) skipped")

    def warn(msg):
        logger.info("faceted body: %s", msg)

    shape = faceted_shape_from_mesh(mesh_o, math.radians(ctx.options.facet_smooth_angle_deg), warn)
    _audit_drawn_body(ctx, shape, mesh_o)
    bounds = np.asarray(mesh_o.bounds, float)
    ctx.rep.flag("facet_edges_suppressed", "info",
                 f"facet edges below {ctx.options.facet_smooth_angle_deg:g} deg coded smooth "
                 f"(not drawn)")
    return shape, bounds[0], bounds[1]


_AUDIT_VOLUME_TOL_PCT = 1e-3     # the faceted body reproduces the mesh volume to round-off


def _audit_drawn_body(ctx: _Ctx, shape, mesh) -> None:
    """Independent check of the faceted body that is drawn (spec 3.2 in-process audit).

    ``export_step_faceted(verify=False)`` reports on itself only; this runs
    ``src.io.step_audit.measure_shape`` on the OCCT body handed to the HLR
    and compares its volume with the mesh volume. Measured cost: 0.45 s for
    the 1544-face plate, 6.9 s for a 20480-face icosphere (BRepCheck
    dominates). A failed measurement is logged and leaves ``None`` values;
    a body that disagrees with a watertight mesh is flagged
    ``faceted_audit_mismatch`` (warning). Never raises.
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
    watertight = bool(getattr(mesh, "is_watertight", False))
    # magnitudes: the body is built outward-oriented even from an inside-out mesh
    mesh_vol = abs(float(mesh.volume)) if watertight else None
    audit["mesh_volume_mm3"] = mesh_vol
    vol = m.get("brep_volume_mm3")
    audit["volume_delta_pct"] = (100.0 * (abs(vol) - mesh_vol) / mesh_vol
                                 if vol is not None and mesh_vol else None)
    audit["seconds"] = round(time.perf_counter() - t0, 3)
    st["audit"]["drawn_body"] = audit
    if not watertight:
        return                                       # open mesh: already flagged, no reference
    problems = []
    if not (audit["solids"] or 0) >= 1:
        problems.append(f"{audit['solids']} solid(s)")
    if audit["free_edges"] is None or audit["free_edges"] != 0:
        problems.append(f"free edges {audit['free_edges']}")
    if not audit["brepcheck_valid"]:
        problems.append("BRepCheck invalid")
    d = audit["volume_delta_pct"]
    if d is None or not abs(d) <= _AUDIT_VOLUME_TOL_PCT:
        problems.append(f"volume delta {d} %")
    if problems:
        ctx.rep.flag("faceted_audit_mismatch", "warning",
                     "drawn faceted body vs watertight mesh: " + "; ".join(problems))


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
    try:
        mm = load_mesh_model(ctx.input_path, opts.classifier_threshold)
    except ValueError as exc:
        raise _Failure(str(exc)) from exc
    census, cls = mm.census, mm.classifier
    rep.data["input"]["source_unit"] = "unknown"     # STL carries no unit; read as mm
    rep.data["route"]["classifier"] = dict(cls)
    _mesh_geometry(ctx, census)
    for w in mm.warnings:
        rep.warn(w)
    if not census.get("watertight"):
        rep.flag("mesh_not_watertight", "warning",
                 f"{census.get('boundary_edges')} boundary edge(s)")
    organic = cls.get("verdict") == "organic"
    if organic:
        rep.flag("organic_classified", "info",
                 f"planar_big {cls.get('planar_big', 0.0):.3f} < threshold {cls.get('threshold')}")
    bounds = np.asarray(mm.mesh.bounds, float)
    o = _orientation(ctx, bounds[1] - bounds[0])
    mesh_o = mm.mesh if _is_identity(o.matrix) else transform_mesh(mm.mesh, o.matrix)
    ob = np.asarray(mesh_o.bounds, float)
    rep.data["geometry"]["drawing_bbox_min"] = ob[0].tolist()
    rep.data["geometry"]["drawing_bbox_max"] = ob[1].tolist()

    st = rep.data["route"]["stl_to_step"]
    mode = opts.stl_to_step
    tri = int(census.get("triangles") or 0)
    convert = None
    if mode == "off":
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

    if convert is not None:
        t1 = time.perf_counter()
        try:
            if convert == "faceted":
                shape, lo, hi = _faceted(ctx, mm.mesh, mesh_o)
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
                f for f in ctx.rep.data["quality"]["flags"]
                if f["code"] not in ("hlr_poly_used", "section_nudged", "section_open_wires",
                                     "section_empty", "holes_not_detected_mesh_route",
                                     "facet_edges_suppressed")]
            ctx.timing["load_s"] += time.perf_counter() - t1

    rep.data["route"].update(pipeline="mesh", source="stl_mesh")
    ctx.source = "stl_mesh"
    _mesh_stage(ctx, mesh_o, organic)


# ------------------------------------------------------------------ sheet --

def _sheet_and_write(ctx: _Ctx) -> None:
    from . import dimensions
    from .layout import compose
    from .style import DEFAULT_STYLE

    opts, rep = ctx.options, ctx.rep
    t0 = time.perf_counter()
    try:
        plan = dimensions.plan_dimensions(ctx.views, ctx.features, opts, DEFAULT_STYLE,
                                          ctx.extents)
    except Exception as exc:          # spec 3.10: the views exist, so the sheet is still written
        logger.exception("dimension planning failed")
        reason = f"{type(exc).__name__}: {exc}"
        rep.warn(f"dimension planning failed: {reason}")
        rep.flag("dimensions_failed", "warning", f"drawing written without dimensions: {reason}")
        plan = dimensions.DimensionPlan(dims=[], undimensioned=[])
    rep.data["dimensions"] = [{"kind": d.kind, "view": d.view, "value_mm": float(d.value),
                               "text": d.text, "count": int(d.count)} for d in plan.dims]
    rep.data["undimensioned"] = [dict(u) for u in plan.undimensioned]
    capped = [u for u in plan.undimensioned if u.get("reason") == "cap"]
    if capped:
        rep.flag("dimensions_capped", "warning",
                 f"{len(capped)} hole group(s) above max_hole_callouts {opts.max_hole_callouts}")

    meta = {"ident": ctx.stem, "title": opts.title or ctx.stem,
            "input_file": os.path.basename(ctx.input_path),
            "source_note": SOURCE_NOTES.get(ctx.source or "", ""),
            "supplementary_title": SOURCE_NOTES.get(ctx.source or "", "")}
    lay = compose(ctx.views, ctx.section, plan, opts, DEFAULT_STYLE, meta)
    for code, severity, detail in lay.flags:
        rep.flag(code, severity, detail)
    if not lay.sheet.meta.get("approval"):
        rep.flag("approval_blank", "info", "the approval field is blank (auto-generated drawing)")
    m = lay.sheet.meta
    rep.data["sheet"] = {"size": lay.sheet_size.name, "orientation": lay.sheet_size.orientation,
                         "scale": lay.scale_text, "projection": m.get("projection"),
                         "iso5457": bool(lay.sheet_size.iso5457),
                         "iso_scale": (None if m.get("iso_scale") is None
                                       else _scale_text(m.get("iso_scale"))),
                         "views_placed": sorted(lay.placements)}
    if rep.data.get("section") is not None:
        sl = m.get("section_layout") or {}
        rep.data["section"]["hatch_segments"] = int(sl.get("hatch_segments") or 0)
        rep.data["section"]["hatch_spacing_mm"] = sl.get("hatch_spacing")

    from . import writers
    fns = {"pdf": writers.write_pdf, "svg": writers.write_svg, "dxf": writers.write_dxf}
    written = 0
    os.makedirs(ctx.out_dir, exist_ok=True)
    for fmt in opts.formats:
        path = _out_path(ctx, fmt)
        try:
            info = fns[fmt](lay.sheet, path)
        except Exception as exc:
            rep.flag("writer_failed", "warning", f"{fmt}: {type(exc).__name__}: {exc}")
            continue
        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            rep.flag("writer_failed", "warning", f"{fmt}: no file written")
            continue
        rep.data["outputs"][fmt] = path
        written += 1
        if fmt == "dxf" and (info.get("audit_errors") or info.get("audit_fixes")):
            rep.warn(f"dxf audit after re-read: {info.get('audit_errors')} error(s), "
                     f"{info.get('audit_fixes')} fix(es)")
    ctx.timing["write_s"] += time.perf_counter() - t0
    if written == 0:
        raise _Failure("no output file could be written")


def _scale_text(s) -> Optional[str]:
    from .style import scale_text
    try:
        return scale_text(float(s))
    except Exception:
        return None


# -------------------------------------------------------------------- run --

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
    if s.extension_mismatch:
        rep.flag("extension_mismatch", "info", f"content is {s.kind}, extension {s.extension!r}")
    err = check_limits(s, opts.max_input_mb, opts.max_triangles)
    if err:
        raise _Failure(err)
    ctx.timing["load_s"] += time.perf_counter() - t0
    if s.kind == "step":
        ctx.say("input: STEP")
        _run_step(ctx)
    elif s.kind in ("stl_binary", "stl_ascii"):
        ctx.say(f"input: {s.kind}, {s.triangles} triangles")
        _run_stl(ctx)
    else:
        raise _Failure("unsupported or corrupt input (neither STEP nor STL by content)")
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


__all__ = ["run", "SOURCE_NOTES"]
