"""Automatic dimension plan and the rendering of dimensions into sheet items.

Spec section 3.6. ``plan_dimensions`` works in VIEW coordinates (model mm,
not shifted) and never repeats a value across views; ``render_dimension``
turns one dimension in SHEET coordinates into ``Segment`` (extension,
dimension and leader lines), filled ``Polygon`` arrowheads and ``Text``.
The layout (``layout.py``) moves the planned dimensions onto the sheet and
decides the band offsets.

The plan (ISO 129 method 1 for the rendering; the placement rules are this
module's conventions):

1. Overall extents of the oriented model, each exactly once: width (world
   X) under the front view (else under the top / back / bottom view), height
   (world Z) left of the front view (else the right / left / back view),
   depth (world Y) left of the top view (else under the right view, left of
   the bottom view, under the left view). Values come from the view bbox.
2. Holes (``CylFeature.kind == "hole"``; bosses and fillets are not
   dimensioned): each hole goes to the first DRAWN principal view of
   ``top, front, right, bottom, back, left`` whose normal is parallel to the
   hole axis (``|dot| > 0.999``); none -> ``undimensioned`` with reason
   ``no_axial_view``. Per view the holes are grouped by radius (0.01 mm),
   ordered by decreasing count then radius, and the first
   ``max_hole_callouts`` groups get one diameter callout (``4 x Ø6.5`` /
   ``Ø20``, 45 degree leader) on the circle nearest the view's top-right
   corner; the rest -> ``undimensioned`` with reason ``cap``.
3. Hole positions (``hole_positions``): the callout circle of each kept
   group gets an X and a Y linear dimension from the view bbox minimum
   corner. The measured point is the circle's bottom (X) or left (Y)
   quadrant so the extension line starts at the hole edge; the value is to
   the centre.
4. Bands: per (view, side) the linear dimensions are sorted by value, the
   shortest nearest the outline (band 0), so extension lines of the long
   dimensions do not cross the short ones. ``DimensionPlan.bands[i]`` is
   ``(side, band)`` for linear dims and ``None`` for diameter callouts;
   side is ``left | bottom | right | top``. The layout puts band ``k`` at
   ``dim_first_offset + k * dim_pitch`` sheet mm from the view outline. In
   the plan, a linear dim's ``base`` lies ON the outline (placeholder); a
   diameter dim's ``base`` equals its centre (= default leader).

Rendering conventions (``style.SheetStyle``; the numeric values are DIN 406
teaching practice, not ISO, see ``style.py``): closed filled arrowhead with
15 degree included angle, 3.5 mm long; extension line from ``ext_gap``
beyond the measured point to ``ext_overshoot`` past the dimension line;
value ``dim_text_gap`` above the line, centred, reading from the bottom or
from the right (vertical dimensions are rotated 90 degrees, text on the
left of the line); when the span is shorter than two arrows plus 0.5 mm the
arrows go outside, pointing in. Diameter: arrow tip on the circle pointing
at the centre, leader along ``leader_angle_deg`` (``LEADER_LEN`` mm, or out
to ``base`` when ``base`` lies further out along the leader than one arrow
length beyond the circle), a horizontal shelf under the text, text anchored
at the leader end.

Values: ``format_value`` rounds half-up to 0.01 mm (``Decimal`` on the
shortest repr, so 12.345 -> "12.35" and 2.675 -> "2.68" although the binary
floats lie just below) and drops trailing zeros; no unit symbol (the sheet
says "Dimensions in mm").
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from typing import Dict, List, Optional, Tuple

import numpy as np

from .model import (TEXT_ADVANCE, BBox2D, CylFeature, Dimension, LineStyle, Polygon,
                    ProjectedView, Segment, SheetItem, Text)
from .options import DrawingOptions
from .style import SheetStyle

log = logging.getLogger(__name__)

DIAMETER_SIGN = "Ø"
LEADER_LEN = 8.0              # sheet mm, default diameter leader length beyond the circle
LEADER_ANGLE_DEG = 45.0
AXIS_PARALLEL_DOT = 0.999     # |dot(axis, view n)| above this = hole seen end-on
RADIUS_GROUP_TOL = 0.01       # mm
SIDES = ("left", "bottom", "right", "top")
HOLE_VIEW_ORDER = ("top", "front", "right", "bottom", "back", "left")

# world axis -> candidate (view, 2D axis 0=x 1=y, side), first drawn wins
_EXTENT_RULES: Tuple[Tuple[str, Tuple[Tuple[str, int, str], ...]], ...] = (
    ("width", (("front", 0, "bottom"), ("top", 0, "bottom"), ("back", 0, "bottom"),
               ("bottom", 0, "bottom"))),
    ("height", (("front", 1, "left"), ("right", 1, "left"), ("left", 1, "left"),
                ("back", 1, "left"))),
    ("depth", (("top", 1, "left"), ("right", 0, "bottom"), ("bottom", 1, "left"),
               ("left", 0, "bottom"))),
)
_WORLD_AXIS = {"width": 0, "depth": 1, "height": 2}
_MIN_EXTENT = 1e-9


@dataclass
class DimensionPlan:
    dims: List[Dimension]          # view coords, Dimension.view set
    undimensioned: List[dict]
    # parallel to dims: (side, band index) for linear dims, None for diameters.
    # Band offsets in sheet mm are applied by layout.compose.
    bands: List[Optional[Tuple[str, int]]] = field(default_factory=list)


# ------------------------------------------------------------------ values --

def format_value(v: float) -> str:
    """Model mm -> dimension text: half-up to 0.01, no trailing zeros, "-0" -> "0"."""
    x = float(v)
    if not math.isfinite(x):
        raise ValueError(f"format_value: not a finite number: {v!r}")
    d = Decimal(repr(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    s = format(d, "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    if s in ("-0", ""):
        s = "0"
    return s


def _diameter_text(radius: float, count: int) -> str:
    base = f"{DIAMETER_SIGN}{format_value(2.0 * radius)}"
    return f"{count} x {base}" if count > 1 else base


# -------------------------------------------------------------------- plan --

def _linear_dim(view: str, bb: BBox2D, axis: int, side: str, p2_override=None,
                value: Optional[float] = None) -> Dimension:
    """Linear dim from the bbox minimum corner; axis 0 -> horizontal, 1 -> vertical."""
    p1 = (bb.xmin, bb.ymin)
    if axis == 0:
        p2 = p2_override if p2_override is not None else (bb.xmax, bb.ymin)
        val = abs(p2[0] - p1[0]) if value is None else value
        y_line = bb.ymin if side == "bottom" else bb.ymax
        base = (0.5 * (p1[0] + p2[0]), y_line)
        angle = 0.0
    else:
        p2 = p2_override if p2_override is not None else (bb.xmin, bb.ymax)
        val = abs(p2[1] - p1[1]) if value is None else value
        x_line = bb.xmin if side == "left" else bb.xmax
        base = (x_line, 0.5 * (p1[1] + p2[1]))
        angle = 90.0
    return Dimension(kind="linear", text=format_value(val), value=float(val),
                     p1=(float(p1[0]), float(p1[1])), p2=(float(p2[0]), float(p2[1])),
                     base=(float(base[0]), float(base[1])), angle_deg=angle, view=view)


def _view_bbox(v: ProjectedView) -> Optional[BBox2D]:
    if v.bbox is not None:
        return v.bbox
    return BBox2D.of_primitives(list(v.visible) + list(v.hidden))


def _group_by_radius(items: List[Tuple[CylFeature, np.ndarray]]) -> List[List[Tuple[CylFeature, np.ndarray]]]:
    items = sorted(items, key=lambda t: float(t[0].radius))
    groups: List[List[Tuple[CylFeature, np.ndarray]]] = []
    for it in items:
        if groups and abs(float(it[0].radius) - float(groups[-1][0][0].radius)) <= RADIUS_GROUP_TOL:
            groups[-1].append(it)
        else:
            groups.append([it])
    return groups


def _unit(v) -> Optional[np.ndarray]:
    a = np.asarray(v, dtype=float).reshape(-1)
    if a.shape != (3,) or not np.all(np.isfinite(a)):
        return None
    n = float(np.linalg.norm(a))
    return a / n if n > 1e-12 else None


def plan_dimensions(views: Dict[str, ProjectedView], features: List[CylFeature], options: DrawingOptions,
                    style: SheetStyle, model_extents: np.ndarray) -> DimensionPlan:
    """Plan the overall, hole callout and hole position dimensions (module docstring)."""
    dims: List[Dimension] = []
    sides: List[Optional[str]] = []
    undimensioned: List[dict] = []
    boxes = {name: _view_bbox(v) for name, v in views.items()}
    ext = np.asarray(model_extents, dtype=float).reshape(-1) if model_extents is not None else None

    # 1. overall extents, once each
    for what, rules in _EXTENT_RULES:
        placed = False
        for view, axis, side in rules:
            bb = boxes.get(view)
            if bb is None:
                continue
            value = bb.width if axis == 0 else bb.height
            if value <= _MIN_EXTENT:
                log.debug("plan_dimensions: %s in view %s is degenerate (%.3g)", what, view, value)
                break
            if ext is not None and ext.shape == (3,):
                m = float(ext[_WORLD_AXIS[what]])
                if m > 0 and abs(m - value) > 0.01 * m + 1e-6:
                    log.warning("plan_dimensions: %s from view %s is %.4g, model extent %.4g",
                                what, view, value, m)
            dims.append(_linear_dim(view, bb, axis, side))
            sides.append(side)
            placed = True
            break
        if not placed:
            log.debug("plan_dimensions: no view for the overall %s", what)

    # 2. hole callouts
    drawn = [v for v in HOLE_VIEW_ORDER if v in views and boxes.get(v) is not None]
    per_view: Dict[str, List[Tuple[CylFeature, np.ndarray]]] = {}
    orphans: List[Tuple[CylFeature, np.ndarray]] = []
    for f in features or []:
        if getattr(f, "kind", None) != "hole":
            continue
        axis = _unit(f.axis)
        target = None
        if axis is not None:
            for name in drawn:
                if abs(float(np.dot(axis, views[name].spec.n))) > AXIS_PARALLEL_DOT:
                    target = name
                    break
        if target is None:
            orphans.append((f, np.zeros(2)))
            continue
        c2 = views[target].spec.to_view(np.asarray(f.center, dtype=float))[0, :2]
        per_view.setdefault(target, []).append((f, c2))

    for grp in _group_by_radius(orphans):
        undimensioned.append({"kind": "hole_group", "radius": float(grp[0][0].radius), "count": len(grp),
                              "reason": "no_axial_view", "view": None})

    cap = int(options.max_hole_callouts)
    for name in drawn:
        items = per_view.get(name)
        if not items:
            continue
        bb = boxes[name]
        groups = _group_by_radius(items)
        groups.sort(key=lambda g: (-len(g), -float(g[0][0].radius)))
        corner = np.array([bb.xmax, bb.ymax])
        for gi, grp in enumerate(groups):
            r = float(np.mean([float(f.radius) for f, _ in grp]))
            if gi >= cap:
                undimensioned.append({"kind": "hole_group", "radius": r, "count": len(grp),
                                      "reason": "cap", "view": name})
                continue
            _, c = min(grp, key=lambda t: float(np.linalg.norm(t[1] - corner)))
            cx, cy = float(c[0]), float(c[1])
            dims.append(Dimension(kind="diameter", text=_diameter_text(r, len(grp)), value=2.0 * r,
                                  center=(cx, cy), radius=r, leader_angle_deg=LEADER_ANGLE_DEG,
                                  base=(cx, cy), view=name, count=len(grp)))
            sides.append(None)
            # 3. positions of the callout circle from the bbox minimum corner
            if options.hole_positions:
                vx, vy = cx - bb.xmin, cy - bb.ymin
                if vx > _MIN_EXTENT:
                    dims.append(_linear_dim(name, bb, 0, "bottom", p2_override=(cx, cy - r), value=vx))
                    sides.append("bottom")
                if vy > _MIN_EXTENT:
                    dims.append(_linear_dim(name, bb, 1, "left", p2_override=(cx - r, cy), value=vy))
                    sides.append("left")

    # 4. bands: per (view, side), shortest value nearest the outline
    bands: List[Optional[Tuple[str, int]]] = [None] * len(dims)
    keyed: Dict[Tuple[str, str], List[int]] = {}
    for i, (d, side) in enumerate(zip(dims, sides)):
        if side is not None:
            keyed.setdefault((d.view, side), []).append(i)
    for (_, side), idx in keyed.items():
        for band, i in enumerate(sorted(idx, key=lambda j: (dims[j].value, j))):
            bands[i] = (side, band)
    log.debug("plan_dimensions: %d dims, %d undimensioned", len(dims), len(undimensioned))
    return DimensionPlan(dims=dims, undimensioned=undimensioned, bands=bands)


# --------------------------------------------------------------- rendering --

def arrowhead(tip, direction, length: float, included_angle_deg: float,
              style: LineStyle = LineStyle.DIMENSION) -> Polygon:
    """Filled triangle with its tip at ``tip`` pointing along ``direction``."""
    u = np.asarray(direction, dtype=float)
    n = float(np.linalg.norm(u))
    u = u / n if n > 1e-12 else np.array([1.0, 0.0])
    t = np.asarray(tip, dtype=float)
    half = math.tan(math.radians(included_angle_deg) / 2.0) * length
    base = t - length * u
    perp = np.array([-u[1], u[0]])
    return Polygon(np.array([t, base + half * perp, base - half * perp]), style)


def _pt(a) -> Tuple[float, float]:
    return (float(a[0]), float(a[1]))


def _render_linear(dim: Dimension, style: SheetStyle) -> List[SheetItem]:
    a = math.radians(float(dim.angle_deg))
    d = np.array([math.cos(a), math.sin(a)])
    nrm = np.array([-d[1], d[0]])
    p1 = np.asarray(dim.p1, dtype=float)
    p2 = np.asarray(dim.p2, dtype=float)
    base = np.asarray(dim.base, dtype=float)
    t1, t2 = float(p1 @ d), float(p2 @ d)
    s = float(base @ nrm)
    q1 = t1 * d + s * nrm
    q2 = t2 * d + s * nrm
    items: List[SheetItem] = []
    for p, q in ((p1, q1), (p2, q2)):
        off = s - float(p @ nrm)
        e = nrm if off >= 0 else -nrm
        items.append(Segment(_pt(p + style.ext_gap * e), _pt(q + style.ext_overshoot * e), LineStyle.DIMENSION))
    span = abs(t2 - t1)
    L = style.arrow_length
    u = (q2 - q1) / span if span > 1e-12 else d
    if span >= 2.0 * L + 0.5:
        items.append(Segment(_pt(q1), _pt(q2), LineStyle.DIMENSION))
        items.append(arrowhead(q1, -u, L, style.arrow_angle_deg))
        items.append(arrowhead(q2, u, L, style.arrow_angle_deg))
    else:     # no room: arrows outside pointing in, line extended past them
        ext = L + 2.0
        items.append(Segment(_pt(q1 - ext * u), _pt(q2 + ext * u), LineStyle.DIMENSION))
        items.append(arrowhead(q1, u, L, style.arrow_angle_deg))
        items.append(arrowhead(q2, -u, L, style.arrow_angle_deg))
    # reading direction: from the bottom or from the right -> angle in (-90, 90]
    rot = float(dim.angle_deg) % 360.0
    up = nrm
    if rot > 180.0:
        rot -= 360.0
    if rot > 90.0 + 1e-9 or rot <= -90.0 + 1e-9:
        rot = rot - 180.0 if rot > 0 else rot + 180.0
        up = -nrm
    mid = 0.5 * (q1 + q2)
    items.append(Text(_pt(mid + style.dim_text_gap * up), dim.text, style.text_height,
                      rotation_deg=rot + 0.0, anchor="middle", valign="baseline"))
    return items


def _render_diameter(dim: Dimension, style: SheetStyle) -> List[SheetItem]:
    a = math.radians(float(dim.leader_angle_deg))
    u = np.array([math.cos(a), math.sin(a)])
    c = np.asarray(dim.center, dtype=float)
    r = abs(float(dim.radius))
    tip = c + r * u
    beyond = float((np.asarray(dim.base, dtype=float) - c) @ u) - r
    length = beyond if beyond > style.arrow_length else LEADER_LEN
    end = tip + length * u
    h = style.text_height
    width = TEXT_ADVANCE * h * len(dim.text)
    right = u[0] >= -1e-12
    sx = 1.0 if right else -1.0
    gap = style.dim_text_gap
    items: List[SheetItem] = [
        Segment(_pt(tip), _pt(end), LineStyle.DIMENSION),
        Segment(_pt(end), _pt(end + np.array([sx * (width + 2.0 * gap), 0.0])), LineStyle.DIMENSION),
        arrowhead(tip, -u, style.arrow_length, style.arrow_angle_deg),
        Text(_pt(end + np.array([sx * gap, gap])), dim.text, h, rotation_deg=0.0,
             anchor="start" if right else "end", valign="baseline"),
    ]
    return items


def render_dimension(dim: Dimension, style: SheetStyle) -> List[SheetItem]:
    """One dimension in sheet coordinates -> Segments, arrowhead Polygons and one Text."""
    if dim.kind == "linear":
        return _render_linear(dim, style)
    if dim.kind == "diameter":
        return _render_diameter(dim, style)
    raise ValueError(f"render_dimension: unknown kind {dim.kind!r}")


def callout_extent(text: str, style: SheetStyle, leader_angle_deg: float = LEADER_ANGLE_DEG,
                   leader_len: float = LEADER_LEN) -> Tuple[float, float]:
    """(dx, dy) sheet mm a diameter callout reaches beyond its leader start.

    Used by the layout to reserve room on the right/top of a view: the
    leader (``leader_len`` along the angle), the shelf with the text and
    the text height above the shelf.
    """
    a = math.radians(leader_angle_deg)
    h = style.text_height
    width = TEXT_ADVANCE * h * len(text)
    dx = abs(math.cos(a)) * leader_len + width + 2.0 * style.dim_text_gap
    dy = abs(math.sin(a)) * leader_len + style.dim_text_gap + h
    return dx, dy


def dims_by_view(plan: DimensionPlan) -> Dict[str, List[int]]:
    """view name -> indices into plan.dims."""
    out: Dict[str, List[int]] = {}
    for i, d in enumerate(plan.dims):
        out.setdefault(d.view, []).append(i)
    return out


__all__ = ["DimensionPlan", "plan_dimensions", "render_dimension", "format_value", "arrowhead",
           "callout_extent", "dims_by_view", "DIAMETER_SIGN", "LEADER_LEN", "SIDES"]
