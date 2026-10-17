"""DXF writer for a ``Sheet`` (ezdxf, R2010, model space at sheet scale).

Everything is written in model space, 1 unit = 1 sheet mm, frame and title
block included (paper space with viewports adds a second coordinate system
for no gain on a fixed auto-generated sheet). ``ezdxf`` is imported lazily
inside ``write_dxf``.

Facts this module relies on (writers probe, ezdxf 1.4.4, re-read with
``ezdxf.readfile`` + ``audit``):

* ``ezdxf.new("R2010", setup=True)`` has no ``HIDDEN`` linetype; a layer
  that asks for it saves fine and ``audit`` then silently replaces it by
  CONTINUOUS (1 fix). Hidden lines therefore use ``DASHED``; ``write_dxf``
  re-reads the file and reports ``audit_errors`` / ``audit_fixes`` so an
  undefined linetype cannot pass unnoticed.
* The ``EZDXF`` dimstyle is a cm style (``dimlfac=100``, tick arrows) and
  ``ISO-25`` does not exist (silent fallback). Dimensions use
  ``dimstyle="Standard"`` + overrides (text 3.5, arrow 3.5, ``dimexe`` 2,
  ``dimexo`` 1, ``dimtad`` 1, ``dimgap`` 1, ``dimdsep`` 46 - the Standard
  separator is a comma otherwise).
* An explicit dimension ``text`` is used verbatim (measured: ``"4 x %%c6.5"``
  stays as is), while ezdxf prefixes ``%%c`` only to the ``<>`` measurement.
  The writer passes the sheet's ``Dimension.text`` with ``Ø ± °`` turned into
  ``%%c %%p %%d``, so the DIMENSION shows exactly the value the PDF/SVG show
  (``%%c`` once); ``dimlfac`` = 1/scale keeps the stored measurement in model
  millimetres.
* ARC angles are CCW degrees, as in ``model.Arc``. ``ANSI31`` is a 45 degree
  pattern with 3.175 spacing at scale 1 (measured: offset length 3.175,
  ``angle=a`` gives a line angle of 45 + a), so a ``HatchLines`` item with
  boundary loops becomes one HATCH entity whose scale and angle reproduce
  the explicit lines' spacing and direction; ``hatch_style`` 0 (odd parity)
  leaves holes unhatched.

Layers (spec 3.7): VISIBLE 50, HIDDEN (DASHED) 25, SECTION 50, HATCH 25,
CENTER (CENTER) 25, CUTTING (CENTER) 50, DIM 25, TEXT 25, FRAME 70 (the
0.7 mm ISO 5457 frame of ``SheetStyle``), TITLE 25. FRAME 70 and the extra
CUTTING layer deviate from the spec text (FRAME 50, no CUTTING) on purpose.
``$LTSCALE`` is set so the DASHED dash is the ISO hidden dash (3 mm; DASHED
is 1.27 long at scale 1); the gaps (0.6 mm) and the CENTER pattern
(7.5/1.5/1.5/1.5 mm, also used for CUTTING) only approximate the ISO
patterns the PDF/SVG draw exactly.

Text height: ``Text.height`` is the em size in the PDF/SVG; DXF TEXT height
is the capital height, so TEXT entities and dimension text (``dimtxt``) get
``CAP_HEIGHT`` x height and the DXF letters match the PDF/SVG letters.

Dimension duplicates: a ``Sheet`` carries both the rendered dimension
primitives (in ``items``) and the semantic ``Dimension`` objects. The DXF
writes the semantic ones as DIMENSION entities and drops the rendered
copies of every dimension it wrote:

* exact - when ``sheet.meta["dimension_items"]`` is a list (one entry per
  ``sheet.dimensions`` entry) of lists of indices into ``sheet.items``,
  exactly those items are dropped;
* geometric fallback - otherwise a ``LineStyle.DIMENSION`` Segment /
  Polyline / Polygon is a copy only when all its points lie within half an
  arrow length of ONE construction axis of the dimension (linear: the
  dimension line and the two extension lines; diameter: the leader ray from
  the circle outwards), or when it is a horizontal shelf starting on the
  leader ray; a Text is a copy when its string equals the dimension text and
  it lies near the dimension. A filled polygon elsewhere near a dimension
  (section-marker arrowhead, symbol fill) is kept.

A ``Dimension`` with a non-finite coordinate or no extent is not written
(counted under ``dimensions_skipped`` and ``skipped``) and its rendered
primitives are kept.
"""
from __future__ import annotations

import logging
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..model import (Arc, Circle, Dimension, HatchLines, LineStyle, Polygon, Polyline, Segment,
                     Sheet, Text)
from ..style import DEFAULT_STYLE, SheetStyle
from . import CAP_HEIGHT, arc_span, clean_text, valid_item

log = logging.getLogger(__name__)

# name, ACI colour, linetype, lineweight (1/100 mm)
LAYERS: Dict[LineStyle, Tuple[str, int, str, int]] = {
    LineStyle.VISIBLE: ("VISIBLE", 7, "CONTINUOUS", 50),
    LineStyle.HIDDEN: ("HIDDEN", 8, "DASHED", 25),
    LineStyle.SECTION: ("SECTION", 7, "CONTINUOUS", 50),
    LineStyle.HATCH: ("HATCH", 7, "CONTINUOUS", 25),
    LineStyle.CENTER: ("CENTER", 1, "CENTER", 25),
    LineStyle.CUTTING: ("CUTTING", 1, "CENTER", 50),
    LineStyle.DIMENSION: ("DIM", 3, "CONTINUOUS", 25),
    LineStyle.TEXT: ("TEXT", 7, "CONTINUOUS", 25),
    LineStyle.FRAME: ("FRAME", 7, "CONTINUOUS", 70),
    LineStyle.TITLE: ("TITLE", 7, "CONTINUOUS", 25),
}
DASHED_DASH = 1.27          # DASHED dash length at LTSCALE 1 (ezdxf setup linetype)
ANSI31_SPACING = 3.175      # ANSI31 line spacing at pattern scale 1 (measured)
DIAMETER_REGION_REACH = 45.0   # default leader reach (mm beyond the circle) searched for copies
SHELF_TOL = 0.01               # mm, a shelf must start on the leader ray
HORIZONTAL_TOL = 1e-3          # mm, |dy| of a horizontal shelf segment


def _layer(style) -> str:
    return LAYERS.get(LineStyle(style), LAYERS[LineStyle.VISIBLE])[0]


def dxf_text(text: str) -> str:
    """Sheet text -> DXF control codes for DIMENSION text (Ø ± ° -> %%c %%p %%d)."""
    return (clean_text(text).replace("Ø", "%%c").replace("±", "%%p")
            .replace("°", "%%d"))


def _align(anchor: str, valign: str):
    from ezdxf.enums import TextEntityAlignment as A
    if valign == "middle":
        return {"start": A.MIDDLE_LEFT, "end": A.MIDDLE_RIGHT}.get(anchor, A.MIDDLE_CENTER)
    return {"start": A.LEFT, "end": A.RIGHT}.get(anchor, A.CENTER)


def _hatch_geometry(segs: np.ndarray, fallback: float) -> Tuple[float, float]:
    """(spacing, line angle deg in [0, 180)) of parallel hatch segments."""
    d = segs[:, 1] - segs[:, 0]
    lengths = np.hypot(d[:, 0], d[:, 1])
    i = int(np.argmax(lengths))
    theta = math.atan2(d[i, 1], d[i, 0])
    n = np.array([-math.sin(theta), math.cos(theta)])
    mids = 0.5 * (segs[:, 0] + segs[:, 1])
    offs = np.sort(mids @ n)
    diffs = np.diff(offs)
    diffs = diffs[diffs > 1e-4]
    spacing = float(diffs.min()) if diffs.size else float(fallback)
    return spacing, math.degrees(theta) % 180.0


def _finite(*vals) -> bool:
    try:
        a = np.asarray(vals, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return a.size > 0 and bool(np.all(np.isfinite(a)))


def _linear_measure(dim: Dimension) -> float:
    a = math.radians(float(dim.angle_deg))
    u = np.array([math.cos(a), math.sin(a)])
    return abs(float((np.asarray(dim.p2, float) - np.asarray(dim.p1, float)) @ u))


def valid_dimension(dim: Dimension) -> bool:
    """True when ``dim`` can become a DIMENSION entity (finite, with extent)."""
    try:
        if dim.kind == "linear":
            return (_finite(dim.p1, dim.p2, dim.base) and _finite(float(dim.angle_deg))
                    and _linear_measure(dim) > 1e-9)
        if dim.kind == "diameter":
            return (_finite(dim.center) and _finite(float(dim.radius), float(dim.leader_angle_deg))
                    and float(dim.radius) > 0)
    except (TypeError, ValueError):
        return False
    return False


def _scale_of(sheet: Sheet) -> Optional[float]:
    """Sheet mm per model mm: ``meta["scale"]``, else derived from a dimension."""
    s = sheet.meta.get("scale") if isinstance(sheet.meta, dict) else None
    try:
        s = float(s)
    except (TypeError, ValueError):
        s = None
    if s is not None and math.isfinite(s) and s > 0:
        return s
    for kind in ("linear", "diameter"):   # fall back to measured length / value
        for dim in sheet.dimensions:
            if dim.kind != kind or not valid_dimension(dim):
                continue
            try:
                v = float(dim.value)
            except (TypeError, ValueError):
                continue
            if not (math.isfinite(v) and v > 0):
                continue
            m = _linear_measure(dim) if kind == "linear" else 2.0 * abs(float(dim.radius))
            return m / v
    return None


def _unit(v) -> Optional[np.ndarray]:
    v = np.asarray(v, float)
    n = float(np.hypot(v[0], v[1]))
    return v / n if n > 1e-9 else None


class _DimZone:
    """Where the rendered primitives of one written dimension can lie."""

    def __init__(self, axes, shelf_axis, text: str, text_box, tol: float):
        self.axes = [(np.asarray(a, float), np.asarray(b, float)) for a, b in axes]
        self.shelf_axis = shelf_axis
        self.text = text
        self.text_box = text_box
        self.tol = tol


def _dist_to_segment(pts: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    l2 = float(ab @ ab)
    t = np.zeros(len(pts)) if l2 < 1e-18 else np.clip(((pts - a) @ ab) / l2, 0.0, 1.0)
    return np.hypot(*(pts - (a + t[:, None] * ab)).T)


def _box(points, margin: float):
    p = np.asarray(points, float).reshape(-1, 2)
    return (p[:, 0].min() - margin, p[:, 1].min() - margin,
            p[:, 0].max() + margin, p[:, 1].max() + margin)


def _zone(dim: Dimension, style: SheetStyle) -> Optional[_DimZone]:
    """Construction axes of a dimension (see the module docstring)."""
    tol = 0.5 * float(style.arrow_length)
    h = float(style.text_height)
    text = clean_text(dim.text)
    text_reach = h + float(style.dim_text_gap) + 0.75 * h * len(text) + tol
    if dim.kind == "linear":
        a = math.radians(float(dim.angle_deg))
        d = np.array([math.cos(a), math.sin(a)])
        n = np.array([-d[1], d[0]])
        p1, p2 = np.asarray(dim.p1, float), np.asarray(dim.p2, float)
        s = float(np.asarray(dim.base, float) @ n)
        q1, q2 = float(p1 @ d) * d + s * n, float(p2 @ d) * d + s * n
        u = _unit(q2 - q1)
        if u is None:
            return None
        ext = float(style.arrow_length) + 2.0 + tol      # arrows-outside case extends the line
        axes = [(q1 - ext * u, q2 + ext * u)]
        over = float(style.ext_overshoot) + tol
        for p, q in ((p1, q1), (p2, q2)):
            e = _unit(q - p)
            if e is None:
                axes.append((q - over * n, q + over * n))
            else:
                axes.append((p, q + over * e))
        pts = np.array([a_ for ax in axes for a_ in ax])
        return _DimZone(axes, None, text, _box(pts, text_reach), tol)
    if dim.kind == "diameter":
        a = math.radians(float(dim.leader_angle_deg))
        u = np.array([math.cos(a), math.sin(a)])
        c = np.asarray(dim.center, float)
        r = abs(float(dim.radius))
        reach = DIAMETER_REGION_REACH
        if _finite(dim.base):
            reach = max(reach, float((np.asarray(dim.base, float) - c) @ u) - r + tol)
        axis = (c + (r - tol) * u, c + (r + reach) * u)
        return _DimZone([axis], axis, text, _box([c - r, c + r, axis[1]], text_reach), tol)
    return None


def _geom_points(item) -> Optional[np.ndarray]:
    if isinstance(item, Segment):
        return np.array([item.a, item.b], float)
    if isinstance(item, (Polyline, Polygon)):
        return np.asarray(item.points, float).reshape(-1, 2)
    return None       # circles, arcs, hatches are never rendered dimension parts


def _is_dimension_copy(item, zones: List[_DimZone]) -> bool:
    if not zones:
        return False
    if isinstance(item, Text):
        txt = clean_text(item.text)
        x, y = float(item.position[0]), float(item.position[1])
        return any(txt == z.text and z.text_box[0] <= x <= z.text_box[2]
                   and z.text_box[1] <= y <= z.text_box[3] for z in zones)
    if item.style != LineStyle.DIMENSION:
        return False
    pts = _geom_points(item)
    if pts is None:
        return False
    for z in zones:
        if any(bool(np.all(_dist_to_segment(pts, a, b) <= z.tol)) for a, b in z.axes):
            return True
        if z.shelf_axis is not None and isinstance(item, Segment):
            if abs(pts[0, 1] - pts[1, 1]) <= HORIZONTAL_TOL:
                d = _dist_to_segment(pts, *z.shelf_axis)
                if float(d.min()) <= SHELF_TOL:
                    return True
    return False


def _exact_copies(sheet: Sheet, written: List[bool]) -> Optional[set]:
    """Item indices named by ``meta["dimension_items"]`` for the written dims, or None."""
    spec = sheet.meta.get("dimension_items") if isinstance(sheet.meta, dict) else None
    if spec is None:
        return None
    try:
        groups = [[int(i) for i in g] for g in spec]
    except (TypeError, ValueError):
        log.warning("dxf: meta['dimension_items'] is malformed; geometric matching used")
        return None
    n = len(sheet.items)
    if len(groups) != len(sheet.dimensions) or any(not 0 <= i < n for g in groups for i in g):
        log.warning("dxf: meta['dimension_items'] does not match the sheet (%d groups for %d "
                    "dimensions, %d items); geometric matching used",
                    len(groups), len(sheet.dimensions), n)
        return None
    drop = set()
    for g, ok in zip(groups, written):
        if ok:
            drop.update(g)
    return drop


def _add_item(msp, item, sheet_style: SheetStyle) -> int:
    """Add one item; returns the number of entities created."""
    layer = _layer(item.style)
    attribs = {"layer": layer}
    if isinstance(item, Segment):
        msp.add_line(tuple(map(float, item.a)), tuple(map(float, item.b)), dxfattribs=attribs)
        return 1
    if isinstance(item, Polyline):
        pts = [tuple(p) for p in np.asarray(item.points, float).reshape(-1, 2)]
        msp.add_lwpolyline(pts, close=bool(item.closed), dxfattribs=attribs)
        return 1
    if isinstance(item, Circle):
        msp.add_circle(tuple(map(float, item.center)), float(item.radius), dxfattribs=attribs)
        return 1
    if isinstance(item, Arc):
        span = arc_span(item.start_deg, item.end_deg)
        if span >= 360.0 - 1e-9:
            msp.add_circle(tuple(map(float, item.center)), float(item.radius), dxfattribs=attribs)
            return 1
        start = float(item.start_deg) % 360.0
        msp.add_arc(tuple(map(float, item.center)), float(item.radius), start_angle=start,
                    end_angle=(start + span) % 360.0, dxfattribs=attribs)
        return 1
    if isinstance(item, Polygon):
        pts = [tuple(p) for p in np.asarray(item.points, float).reshape(-1, 2)]
        if len(pts) == 3:
            msp.add_solid(pts, dxfattribs=attribs)
        elif len(pts) == 4:
            msp.add_solid([pts[0], pts[1], pts[3], pts[2]], dxfattribs=attribs)   # SOLID zigzag order
        else:
            h = msp.add_hatch(color=256, dxfattribs=attribs)
            h.paths.add_polyline_path(pts, is_closed=True)
            h.set_solid_fill(color=256)
        return 1
    if isinstance(item, HatchLines):
        segs = np.asarray(item.segments, float).reshape(-1, 2, 2)
        loops = [np.asarray(lp, float).reshape(-1, 2) for lp in item.loops]
        loops = [lp for lp in loops if lp.shape[0] >= 3]
        if not loops:
            for a, b in segs:
                msp.add_line(tuple(a), tuple(b), dxfattribs=attribs)
            return int(segs.shape[0])
        spacing, angle = _hatch_geometry(segs, sheet_style.hatch_spacing_mid)
        h = msp.add_hatch(color=256, dxfattribs=attribs)
        for lp in loops:
            h.paths.add_polyline_path([tuple(p) for p in lp], is_closed=True)
        # style=0: odd parity (set_pattern_fill defaults to 1 = outer and overrides hatch_style)
        h.set_pattern_fill("ANSI31", color=256, angle=angle - 45.0,
                           scale=spacing / ANSI31_SPACING, style=0)
        return 1
    if isinstance(item, Text):
        t = msp.add_text(clean_text(item.text), height=CAP_HEIGHT * float(item.height),
                         rotation=float(item.rotation_deg), dxfattribs=attribs)
        t.set_placement(tuple(map(float, item.position)), align=_align(item.anchor, item.valign))
        return 1
    raise TypeError(f"dxf: unsupported item {type(item).__name__}")


def _add_dimension(msp, dim: Dimension, override: dict) -> bool:
    attribs = {"layer": LAYERS[LineStyle.DIMENSION][0]}
    text = dxf_text(dim.text) if dim.text else "<>"
    if dim.kind == "linear":
        d = msp.add_linear_dim(base=tuple(map(float, dim.base)), p1=tuple(map(float, dim.p1)),
                               p2=tuple(map(float, dim.p2)), angle=float(dim.angle_deg),
                               text=text, dimstyle="Standard", override=override,
                               dxfattribs=attribs)
    elif dim.kind == "diameter":
        d = msp.add_diameter_dim(center=tuple(map(float, dim.center)), radius=float(dim.radius),
                                 angle=float(dim.leader_angle_deg), text=text,
                                 dimstyle="Standard", override=override, dxfattribs=attribs)
    else:
        log.warning("dxf: dimension kind %r not supported, skipped", dim.kind)
        return False
    d.render()
    return True


def write_dxf(sheet: Sheet, path: str, sheet_style: SheetStyle = DEFAULT_STYLE) -> dict:
    """Write ``sheet`` as an R2010 DXF in model space (sheet mm).

    Returns ``{"bytes", "entities", "layers": {name: count}, "dimensions",
    "dimensions_skipped", "dimension_items_replaced", "dimension_match",
    "skipped", "audit_errors", "audit_fixes"}``; ``dimension_match`` is
    ``"exact"`` (``meta["dimension_items"]``) or ``"geometric"``; ``skipped``
    counts skipped items plus skipped dimensions; the audit numbers come from
    re-reading the written file.
    """
    import ezdxf

    doc = ezdxf.new("R2010", setup=True)
    doc.units = 4
    doc.header["$INSUNITS"] = 4
    doc.header["$MEASUREMENT"] = 1
    doc.header["$LWDISPLAY"] = 1
    doc.header["$LTSCALE"] = round(float(sheet_style.dash_hidden[0]) / DASHED_DASH, 4)
    for name, color, lt, lw in LAYERS.values():
        if name not in doc.layers:
            doc.layers.add(name, color=color, linetype=lt, lineweight=lw)
    msp = doc.modelspace()

    scale = _scale_of(sheet)
    override = {"dimtxt": CAP_HEIGHT * float(sheet_style.text_height),
                "dimasz": float(sheet_style.arrow_length),
                "dimexe": float(sheet_style.ext_overshoot), "dimexo": float(sheet_style.ext_gap),
                "dimdec": 1, "dimtad": 1, "dimgap": float(sheet_style.dim_text_gap),
                "dimdsep": ord("."), "dimlfac": 1.0 / scale if scale else 1.0}

    zones: List[_DimZone] = []
    written: List[bool] = []
    n_dims = dims_skipped = 0
    for dim in sheet.dimensions:
        ok = False
        if not valid_dimension(dim):
            dims_skipped += 1
            log.warning("dxf: dimension %r is non-finite or degenerate; not written, rendered "
                        "primitives kept", getattr(dim, "text", None))
        else:
            try:
                ok = _add_dimension(msp, dim, override)
            except Exception as exc:  # keep the rendered primitives of that dimension
                log.warning("dxf: dimension %r failed (%s); rendered primitives kept", dim.text, exc)
        written.append(ok)
        if ok:
            n_dims += 1
            z = _zone(dim, sheet_style)
            if z is not None:
                zones.append(z)

    exact = _exact_copies(sheet, written)
    skipped = 0
    replaced = 0
    for idx, item in enumerate(sheet.items):
        if not valid_item(item):
            skipped += 1
            continue
        copy = (idx in exact) if exact is not None else _is_dimension_copy(item, zones)
        if copy:
            replaced += 1
            continue
        _add_item(msp, item, sheet_style)
    if skipped:
        log.warning("dxf: skipped %d degenerate or non-finite items", skipped)

    W, H = float(sheet.width), float(sheet.height)
    doc.header["$EXTMIN"] = (0.0, 0.0, 0.0)
    doc.header["$EXTMAX"] = (W, H, 0.0)
    try:
        doc.set_modelspace_vport(height=H * 1.05, center=(W / 2.0, H / 2.0))
    except Exception as exc:  # cosmetic only
        log.debug("dxf: set_modelspace_vport failed: %s", exc)
    doc.saveas(path)

    layers: Dict[str, int] = {}
    entities = 0
    for e in msp:
        entities += 1
        layers[e.dxf.layer] = layers.get(e.dxf.layer, 0) + 1
    audit_errors = audit_fixes = None
    try:
        aud = ezdxf.readfile(path).audit()
        audit_errors, audit_fixes = len(aud.errors), len(aud.fixes)
        if audit_errors or audit_fixes:
            log.warning("dxf: audit after re-read: %d errors, %d fixes: %s", audit_errors,
                        audit_fixes, [f.message for f in aud.fixes][:5])
    except Exception as exc:
        log.warning("dxf: re-read for audit failed: %s", exc)
    return {"bytes": os.path.getsize(path), "entities": entities, "layers": layers,
            "dimensions": n_dims, "dimensions_skipped": dims_skipped,
            "dimension_items_replaced": replaced,
            "dimension_match": "exact" if exact is not None else "geometric",
            "skipped": skipped + dims_skipped,
            "audit_errors": audit_errors, "audit_fixes": audit_fixes}
