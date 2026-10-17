"""Automatic dimension plan and the rendering of dimensions into sheet items.

Spec section 3.6. ``plan_dimensions`` works in VIEW coordinates (model mm,
not shifted); ``render_dimension`` turns one dimension in SHEET
coordinates into ``Segment`` (extension, dimension and leader lines),
filled ``Polygon`` arrowheads and ``Text``. The layout (``layout.py``)
moves the planned dimensions onto the sheet and decides the band offsets.

The plan (ISO 129 method 1 for the rendering; the placement rules are this
module's conventions):

1. Overall extents of the oriented model, each exactly once: width (world
   X) under the front view (else under the top / back / bottom view), height
   (world Z) left of the front view (else the right / left / back view),
   depth (world Y) left of the top view (else under the right view, left of
   the bottom view, under the left view). The VALUE is the exact model
   extent projected on the view axis (``model_extents``, the oriented
   OCCT / mesh bounding box); the view bbox only places the dimension. When
   the view bbox and the model extent differ by more than
   ``EXTENT_MISMATCH_TOL`` (0.5 mm) the plan returns the flag
   ``dimension_extent_mismatch`` (warning) in ``DimensionPlan.flags``.
   With ``model_bounds`` (the oriented (lo, hi) box) the reference edges
   of every view are the exact box projected into the view (datum box);
   without it they are the view bbox edges.
   The extension lines start at real geometry: the end points are the
   projected view points (segment ends, polyline vertices, arc ends and
   quadrant points) that attain the extreme coordinate, the one nearest
   the dimension side (QA round 1 issue 23: bbox corners float away from
   multi-body and chamfered outlines).
2. Holes (``CylFeature.kind == "hole"``): each hole goes to the first
   DRAWN principal view of ``top, front, right, bottom, back, left`` whose
   normal is parallel to the hole axis (``|dot| > 0.999``); none ->
   ``undimensioned`` with reason ``no_axial_view``. Per view, holes on one
   axis line (2D centres closer than ``COAXIAL_TOL``) form ONE feature: a
   counterbore (two radii, the larger one at exactly one end of the axial
   span) or a stepped bore (any other multi-radius cluster). The features
   are grouped by their radius signature (0.01 mm) and kind, ordered by
   decreasing count then radius, and the first ``max_hole_callouts``
   groups get one diameter callout (``4 x Ø6.5``, ``Ø20``,
   ``4 x Ø10.5 / Ø15.9 counterbore``, ``Ø8 / Ø12 / Ø20 stepped``); the
   rest -> ``undimensioned`` with reason ``cap``. Counterbore and step
   depths are not dimensioned (``undimensioned``: ``counterbore_depth`` /
   ``step_depth``, reason ``not_supported``). Each callout gets its own
   circle and a leader angle from ``45, 135, -45, -135``: per group the
   (circle, angle) pair with the shortest leader out of the view box wins,
   plus a penalty of ``ANGLE_REUSE_WEIGHT`` x the view diagonal for an
   angle already used by a callout nearby (falling to 0 at half the
   diagonal) and ``LEADER_CROSS_WEIGHT`` x the diagonal per crossing of an
   already planned leader or of another hole's circle, plus the view mm the
   text would stick out sideways beyond the view (``_text_overhang``, at
   ``scale_hint`` or an estimated scale: a long counterbore text then runs
   over the view rather than widening the sheet); ties go to the
   angle order, then to the circle nearest the
   corner the angle points to (QA issue 8: every group used to sit on the
   top-right circle at 45 degrees). The arrow sits on the largest circle of
   the feature; the value is the smallest diameter. ``layout.compose``
   moves a callout whose text still collides.
   Bosses and fillets are not dimensioned: one ``undimensioned`` entry per
   radius group (``boss_group`` / ``fillet_group``, reason
   ``not_supported``).
3. Hole positions (``hole_positions``): the callout circle of each kept
   group gets an X and a Y linear dimension from ONE reference edge per
   view and axis (left or right, bottom or top): the edge nearer to the
   majority of the view's hole circles, left / bottom on a tie
   (``_position_datums``; mixing datums per hole inside one view is harder
   to make and inspect). The dimension sits on the side nearer the hole, so the
   extension line starts at the quadrant of the largest circle facing that
   side and does not cross the view. The value is to the centre. A
   concentric bore (centre at the datum box centre of a view whose outline
   is a circle of that centre) gets no position dims. A position equal to
   one already planned in the same view, along the same axis, from the
   same edge (``(view, axis, edge, text)``) is dropped: one extension
   line already gives it.
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
value ``dim_text_gap`` above the line, centred (shifted along the line by
``Dimension.text_offset``, the dimension line extended under a text that
leaves the span), reading from the bottom or from the right (vertical
dimensions are rotated 90 degrees, text on the left of the line); when the
span is shorter than two arrows plus 0.5 mm the arrows go outside,
pointing in. Diameter: arrow tip on the circle pointing at the centre,
leader along ``leader_angle_deg`` (``LEADER_LEN`` mm, or out to ``base``
when ``base`` lies further out along the leader than one arrow length
beyond the circle), a horizontal shelf under the text, text anchored at
the leader end.

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

from .model import (TEXT_ADVANCE, Arc, BBox2D, Circle, CylFeature, Dimension, LineStyle, Polygon,
                    Polyline, ProjectedView, Segment, SheetItem, Text, arc_extreme_points)
from .options import DrawingOptions
from .style import SheetStyle

log = logging.getLogger(__name__)

DIAMETER_SIGN = "Ø"
LEADER_LEN = 8.0              # sheet mm, default diameter leader length beyond the circle
LEADER_ANGLE_DEG = 45.0
LEADER_ANGLES = (45.0, 135.0, -45.0, -135.0)   # callout leader directions, in preference order
AXIS_PARALLEL_DOT = 0.999     # |dot(axis, view n)| above this = hole seen end-on
RADIUS_GROUP_TOL = 0.01       # mm
COAXIAL_TOL = 0.05            # mm: hole centres closer than this in the axial view share one axis line
EXTENT_MISMATCH_TOL = 0.5     # mm: view bbox vs exact model extent beyond this -> flag
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
    # (code, severity, detail) for the report, e.g. dimension_extent_mismatch
    flags: List[Tuple[str, str, str]] = field(default_factory=list)


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


def hole_callout_text(radii, kind: str, count: int) -> str:
    """Callout text of a hole feature group: ``4 x Ø6.5``, ``Ø10.5 / Ø15.9 counterbore``.

    ``radii`` ascending; ``kind`` "hole" | "counterbore" | "stepped".
    """
    parts = " / ".join(f"{DIAMETER_SIGN}{format_value(2.0 * float(r))}" for r in radii)
    base = parts if kind == "hole" else f"{parts} {kind}"
    return f"{count} x {base}" if count > 1 else base


# -------------------------------------------------------------------- plan --

def _linear_dim(view: str, bb: BBox2D, axis: int, side: str, p2_override=None,
                value: Optional[float] = None) -> Dimension:
    """Linear dim from the bbox minimum corner; axis 0 -> horizontal, 1 -> vertical.

    Kept for callers of the first version; ``plan_dimensions`` anchors its
    dimensions at real geometry (``_make_linear``).
    """
    p1 = (bb.xmin, bb.ymin)
    if axis == 0:
        p2 = p2_override if p2_override is not None else (bb.xmax, bb.ymin)
    else:
        p2 = p2_override if p2_override is not None else (bb.xmin, bb.ymax)
    val = abs(p2[axis] - p1[axis]) if value is None else value
    return _make_linear(view, bb, axis, side, p1, p2, float(val))


def _make_linear(view: str, bb: BBox2D, axis: int, side: str, p1, p2, value: float) -> Dimension:
    if axis == 0:
        y_line = bb.ymin if side == "bottom" else bb.ymax
        base = (0.5 * (p1[0] + p2[0]), y_line)
        angle = 0.0
    else:
        x_line = bb.xmin if side == "left" else bb.xmax
        base = (x_line, 0.5 * (p1[1] + p2[1]))
        angle = 90.0
    return Dimension(kind="linear", text=format_value(value), value=float(value),
                     p1=(float(p1[0]), float(p1[1])), p2=(float(p2[0]), float(p2[1])),
                     base=(float(base[0]), float(base[1])), angle_deg=angle, view=view)


def _view_bbox(v: ProjectedView) -> Optional[BBox2D]:
    if v.bbox is not None:
        return v.bbox
    return BBox2D.of_primitives(list(v.visible) + list(v.hidden))


def _prim_points(prims) -> np.ndarray:
    """Anchor candidates of view primitives: segment ends, polyline vertices,
    arc ends and quadrant points, circle quadrant points."""
    chunks: List[np.ndarray] = []
    for p in prims:
        try:
            if isinstance(p, Segment):
                chunks.append(np.array([p.a, p.b], dtype=float).reshape(-1, 2))
            elif isinstance(p, Arc):
                chunks.append(arc_extreme_points(p.center, p.radius, p.start_deg, p.end_deg))
            elif isinstance(p, Circle):
                cx, cy, r = float(p.center[0]), float(p.center[1]), abs(float(p.radius))
                chunks.append(np.array([[cx + r, cy], [cx, cy + r], [cx - r, cy], [cx, cy - r]]))
            elif isinstance(p, Polyline):
                chunks.append(np.asarray(p.points, dtype=float).reshape(-1, 2))
        except Exception:                         # malformed primitive: no anchor from it
            continue
    chunks = [c for c in chunks if c.size]
    if not chunks:
        return np.zeros((0, 2))
    pts = np.vstack(chunks)
    return pts[np.all(np.isfinite(pts), axis=1)]


def _anchor_points(v: ProjectedView) -> np.ndarray:
    pts = _prim_points(v.visible)
    if pts.shape[0] == 0:
        pts = _prim_points(v.hidden)
    return pts


def _edge_point(pts: np.ndarray, axis: int, high: bool, side_axis: int, side_high: bool,
                tol: float) -> Optional[np.ndarray]:
    """The view point attaining the min / max ``axis`` coordinate (within
    ``tol``) that lies nearest the dimension side (min / max ``side_axis``)."""
    if pts.shape[0] == 0:
        return None
    c = pts[:, axis]
    ext = c.max() if high else c.min()
    cand = pts[np.abs(c - ext) <= tol]
    if cand.shape[0] == 0:
        return None
    k = int(np.argmax(cand[:, side_axis]) if side_high else np.argmin(cand[:, side_axis]))
    return cand[k]


def _exact_box(spec, bounds: np.ndarray) -> Optional[BBox2D]:
    lo, hi = bounds[0], bounds[1]
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    try:
        return BBox2D.from_points(spec.to_view(corners)[:, :2])
    except Exception:
        return None


def _group_by_radius(items: List[Tuple[CylFeature, np.ndarray]]) -> List[List[Tuple[CylFeature, np.ndarray]]]:
    items = sorted(items, key=lambda t: float(t[0].radius))
    groups: List[List[Tuple[CylFeature, np.ndarray]]] = []
    for it in items:
        if groups and abs(float(it[0].radius) - float(groups[-1][0][0].radius)) <= RADIUS_GROUP_TOL:
            groups[-1].append(it)
        else:
            groups.append([it])
    return groups


def _group_features_by_radius(feats: List[CylFeature]) -> List[List[CylFeature]]:
    return [[f for f, _ in g] for g in _group_by_radius([(f, np.zeros(2)) for f in feats])]


def _unit(v) -> Optional[np.ndarray]:
    a = np.asarray(v, dtype=float).reshape(-1)
    if a.shape != (3,) or not np.all(np.isfinite(a)):
        return None
    n = float(np.linalg.norm(a))
    return a / n if n > 1e-12 else None


@dataclass
class _Cluster:
    """Holes on one axis line in one view (a plain hole, counterbore or stepped bore)."""
    centre: np.ndarray            # 2D, view coords
    feats: List[CylFeature]
    radii: List[float]            # ascending, merged within RADIUS_GROUP_TOL
    kind: str                     # "hole" | "counterbore" | "stepped"


def _merged_radii(feats: List[CylFeature]) -> List[float]:
    out: List[float] = []
    for r in sorted(float(f.radius) for f in feats):
        if out and abs(r - out[-1]) <= RADIUS_GROUP_TOL:
            continue
        out.append(r)
    return out


def _cluster_kind(feats: List[CylFeature], radii: List[float], n: np.ndarray) -> str:
    if len(radii) == 1:
        return "hole"
    if len(radii) == 2:
        spans = {0: [], 1: []}
        for f in feats:
            k = 0 if abs(float(f.radius) - radii[0]) <= RADIUS_GROUP_TOL else 1
            t = float(np.dot(np.asarray(f.center, dtype=float), n))
            h = 0.5 * abs(float(f.length))
            spans[k].append((t - h, t + h))
        lo = min(a for s in spans.values() for a, _ in s)
        hi = max(b for s in spans.values() for _, b in s)
        tol = max(0.01, 1e-3 * (hi - lo))
        big = spans[1]
        at_lo = any(abs(a - lo) <= tol for a, _ in big)
        at_hi = any(abs(b - hi) <= tol for _, b in big)
        small_lo = any(abs(a - lo) <= tol for a, _ in spans[0])
        small_hi = any(abs(b - hi) <= tol for _, b in spans[0])
        if at_lo != at_hi and ((at_lo and small_hi) or (at_hi and small_lo)):
            return "counterbore"
    return "stepped"


def _clusters(items: List[Tuple[CylFeature, np.ndarray]], n: np.ndarray) -> List[_Cluster]:
    """Coaxial holes of one view (2D centres within COAXIAL_TOL) -> one cluster each."""
    out: List[List[Tuple[CylFeature, np.ndarray]]] = []
    for f, c in items:
        for cl in out:
            if float(np.linalg.norm(cl[0][1] - c)) <= COAXIAL_TOL:
                cl.append((f, c))
                break
        else:
            out.append([(f, c)])
    res = []
    for cl in out:
        feats = [f for f, _ in cl]
        radii = _merged_radii(feats)
        centre = np.mean(np.array([c for _, c in cl]), axis=0)
        res.append(_Cluster(centre, feats, radii, _cluster_kind(feats, radii, n)))
    return res


def _same_signature(a: _Cluster, b: _Cluster) -> bool:
    return (a.kind == b.kind and len(a.radii) == len(b.radii)
            and all(abs(x - y) <= RADIUS_GROUP_TOL for x, y in zip(a.radii, b.radii)))


def _group_clusters(clusters: List[_Cluster]) -> List[List[_Cluster]]:
    groups: List[List[_Cluster]] = []
    for cl in sorted(clusters, key=lambda c: (c.kind, len(c.radii), c.radii)):
        for g in groups:
            if _same_signature(g[0], cl):
                g.append(cl)
                break
        else:
            groups.append([cl])
    groups.sort(key=lambda g: (-len(g), -g[0].radii[-1], -g[0].radii[0]))
    return groups


def _leader_exit(centre: np.ndarray, r: float, angle: float, bb: BBox2D) -> float:
    """Leader length (view mm) from the circle along ``angle`` to the view box edge."""
    a = math.radians(angle)
    u = np.array([math.cos(a), math.sin(a)])
    p = np.asarray(centre, dtype=float) + abs(float(r)) * u
    ts = []
    for axis, lo, hi in ((0, bb.xmin, bb.xmax), (1, bb.ymin, bb.ymax)):
        if abs(u[axis]) > 1e-12:
            ts.append(((hi if u[axis] > 0 else lo) - p[axis]) / u[axis])
    return max(0.0, min(ts)) if ts else 0.0


ANGLE_REUSE_WEIGHT = 0.25     # x view diagonal: cost of reusing a leader angle right next to another callout
LEADER_CROSS_WEIGHT = 0.5     # x view diagonal: cost of a leader crossing an already planned leader


def _leader_segment(centre: np.ndarray, r: float, angle: float, exit_len: float):
    a = math.radians(angle)
    u = np.array([math.cos(a), math.sin(a)])
    p = np.asarray(centre, dtype=float) + abs(float(r)) * u
    return p, p + max(float(exit_len), 1e-6) * u


def _segment_point_distance(seg, p) -> float:
    a, b = np.asarray(seg[0], dtype=float), np.asarray(seg[1], dtype=float)
    p = np.asarray(p, dtype=float)
    d = b - a
    L2 = float(d @ d)
    t = 0.0 if L2 < 1e-24 else min(1.0, max(0.0, float((p - a) @ d) / L2))
    return float(np.linalg.norm(a + t * d - p))


def _segments_cross(s1, s2) -> bool:
    """Proper crossing of two segments (shared end points do not count)."""
    (a, b), (c, d) = s1, s2

    def orient(p, q, r):
        return float((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))
    d1, d2 = orient(c, d, a), orient(c, d, b)
    d3, d4 = orient(a, b, c), orient(a, b, d)
    return (d1 * d2 < -1e-12) and (d3 * d4 < -1e-12)


def _angle_penalty(angle: float, centre: np.ndarray, placed: List[Tuple[float, np.ndarray]], diag: float) -> float:
    """Cost of a leader direction already used by callouts nearby (falls to 0 at half the diagonal)."""
    pen = 0.0
    for a, c in placed:
        if abs(a - angle) < 1e-9:
            d = float(np.linalg.norm(np.asarray(c) - np.asarray(centre)))
            pen += max(0.0, 1.0 - d / (0.5 * diag))
    return ANGLE_REUSE_WEIGHT * diag * pen


_SCALE_STEPS = (50.0, 20.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01)
_VIEW_BUDGET_MM = 150.0       # about the sheet room one principal view gets on A3


def _scale_estimate(bb: BBox2D, hint: Optional[float]) -> float:
    if hint is not None:
        try:
            h = float(hint)
            if math.isfinite(h) and h > 0:
                return h
        except (TypeError, ValueError):
            pass
    size = max(bb.width, bb.height, 1e-9)
    for s in _SCALE_STEPS:
        if size * s <= _VIEW_BUDGET_MM:
            return s
    return _SCALE_STEPS[-1]


def _text_overhang(exit_pt: np.ndarray, angle: float, text: str, bb: BBox2D, s_est: float,
                   style: SheetStyle) -> float:
    """View mm a callout text at the leader end would stick out beyond the view box sideways.

    The shelf and text run right for a leader pointing right, left
    otherwise (``_render_diameter``); the text width is sheet mm, so it is
    divided by the estimated scale.
    """
    w = (TEXT_ADVANCE * style.text_height * len(text or "") + 3.0 * style.dim_text_gap + 3.0) / max(s_est, 1e-9)
    x = float(exit_pt[0])
    if math.cos(math.radians(angle)) >= -1e-12:
        return max(0.0, x + w - bb.xmax)
    return max(0.0, bb.xmin - (x - w))


def _circular_outline(v: ProjectedView, db: BBox2D) -> bool:
    """True when the view's outline is a circle centred on the datum box centre."""
    R = 0.5 * db.width
    if R <= _MIN_EXTENT or abs(db.width - db.height) > max(0.01, 0.01 * R):
        return False
    c = np.array([0.5 * (db.xmin + db.xmax), 0.5 * (db.ymin + db.ymax)])
    tol = max(0.01, 1e-3 * R)
    for p in v.visible:
        if isinstance(p, (Circle, Arc)):
            if (float(np.linalg.norm(np.asarray(p.center, dtype=float) - c)) <= tol
                    and abs(abs(float(p.radius)) - R) <= max(tol, 0.01 * R)):
                return True
    return False


def _position_datums(centres: List[np.ndarray], db: BBox2D) -> Dict[int, bool]:
    """One reference edge per axis for the hole positions of one view.

    ``{axis: high}``: ``high`` is True when more than half of ``centres``
    lie nearer the right (axis 0) or top (axis 1) edge of the datum box
    than the opposite edge; a tie or no centres keeps the left / bottom
    edge. Every position in the view along that axis is measured from that
    one edge (QA round 1 issue 23: one datum per view, not the nearer edge
    per hole).
    """
    out: Dict[int, bool] = {}
    for axis in (0, 1):
        lo_e, hi_e = (db.xmin, db.xmax) if axis == 0 else (db.ymin, db.ymax)
        n_hi = sum(1 for c in centres if (float(c[axis]) - lo_e) > (hi_e - float(c[axis])) + 1e-9)
        out[axis] = 2 * n_hi > len(centres)
    return out


def plan_dimensions(views: Dict[str, ProjectedView], features: List[CylFeature], options: DrawingOptions,
                    style: SheetStyle, model_extents: np.ndarray, *,
                    model_bounds: Optional[np.ndarray] = None,
                    scale_hint: Optional[float] = None) -> DimensionPlan:
    """Plan the overall, hole callout and hole position dimensions (module docstring).

    ``model_extents``: (3,) exact extents of the oriented model (the values
    of the overall dims). ``model_bounds``: optional (2, 3) oriented
    ``(lo, hi)`` box; when given, the reference edges of the overall and
    position dims are that box projected into each view instead of the
    view bbox. ``scale_hint``: optional expected drawing scale (sheet mm
    per model mm, e.g. the pipeline's scale guess); it converts the callout
    text width to view units when the callout circles are chosen (a long
    ``... counterbore`` text placed over the view instead of beyond it).
    Without it the scale is estimated from the view size
    (``_scale_estimate``).
    """
    dims: List[Dimension] = []
    sides: List[Optional[str]] = []
    undimensioned: List[dict] = []
    flags: List[Tuple[str, str, str]] = []
    boxes = {name: _view_bbox(v) for name, v in views.items()}
    ext = np.asarray(model_extents, dtype=float).reshape(-1) if model_extents is not None else None
    if ext is not None and (ext.shape != (3,) or not np.all(np.isfinite(ext))):
        ext = None
    bounds = None
    if model_bounds is not None:
        try:
            b = np.asarray(model_bounds, dtype=float).reshape(2, 3)
            if np.all(np.isfinite(b)) and np.all(b[1] >= b[0]):
                bounds = b
        except Exception:
            bounds = None
    datum: Dict[str, BBox2D] = {}
    for name, v in views.items():
        bb = boxes.get(name)
        if bb is None:
            continue
        db = _exact_box(v.spec, bounds) if bounds is not None else None
        datum[name] = db if db is not None else bb
    anchors: Dict[str, np.ndarray] = {}

    def pts_of(name: str) -> np.ndarray:
        if name not in anchors:
            anchors[name] = _anchor_points(views[name])
        return anchors[name]

    def tol_of(bb: BBox2D) -> float:
        return max(1e-3, 2e-3 * max(bb.width, bb.height))

    def anchor(name: str, axis: int, high: bool, side: str, edge_value: float) -> Tuple[float, float]:
        """Point on the edge ``axis = edge_value`` at real geometry nearest ``side``."""
        bb = boxes[name]
        side_axis = 1 - axis
        side_high = side in ("top", "right")
        p = _edge_point(pts_of(name), axis, high, side_axis, side_high, tol_of(bb))
        if p is None:
            other = (bb.ymax if side_high else bb.ymin) if axis == 0 else (bb.xmax if side_high else bb.xmin)
        else:
            other = float(p[side_axis])
        return (edge_value, other) if axis == 0 else (other, edge_value)

    seen_keys = set()

    # 1. overall extents, once each; values from the exact model extents
    for what, rules in _EXTENT_RULES:
        placed = False
        for view, axis, side in rules:
            bb = boxes.get(view)
            if bb is None:
                continue
            vval = bb.width if axis == 0 else bb.height
            if vval <= _MIN_EXTENT:
                log.debug("plan_dimensions: %s in view %s is degenerate (%.3g)", what, view, vval)
                break
            db = datum[view]
            spec = views[view].spec
            value = db.width if axis == 0 else db.height
            if ext is not None:
                vec = spec.x if axis == 0 else spec.y
                m = float(np.abs(np.asarray(vec, dtype=float)) @ ext)
                if m > _MIN_EXTENT:
                    value = m
            lo_e, hi_e = (db.xmin, db.xmax) if axis == 0 else (db.ymin, db.ymax)
            vlo, vhi = (bb.xmin, bb.xmax) if axis == 0 else (bb.ymin, bb.ymax)
            dev = max(abs(value - vval), abs(lo_e - vlo), abs(hi_e - vhi))
            if dev > EXTENT_MISMATCH_TOL:
                detail = (f"{what} in view {view}: view bbox {vval:.3f} mm "
                          f"[{vlo:.3f}, {vhi:.3f}], model extent {value:.3f} mm [{lo_e:.3f}, {hi_e:.3f}]")
                flags.append(("dimension_extent_mismatch", "warning", detail))
                log.info("plan_dimensions: %s", detail)
            p1 = anchor(view, axis, False, side, lo_e)
            p2 = anchor(view, axis, True, side, hi_e)
            dims.append(_make_linear(view, bb, axis, side, p1, p2, value))
            sides.append(side)
            seen_keys.add((view, axis, "overall", dims[-1].text))
            placed = True
            break
        if not placed:
            log.debug("plan_dimensions: no view for the overall %s", what)

    # 2. hole callouts
    drawn = [v for v in HOLE_VIEW_ORDER if v in views and boxes.get(v) is not None]
    per_view: Dict[str, List[Tuple[CylFeature, np.ndarray]]] = {}
    orphans: List[Tuple[CylFeature, np.ndarray]] = []
    others: Dict[str, List[CylFeature]] = {}
    for f in features or []:
        kind = getattr(f, "kind", None)
        if kind in ("boss", "fillet"):
            others.setdefault(kind, []).append(f)
            continue
        if kind != "hole":
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
    for kind in ("boss", "fillet"):
        for grp in _group_features_by_radius(others.get(kind, [])):
            undimensioned.append({"kind": f"{kind}_group",
                                  "radius": float(np.mean([float(f.radius) for f in grp])),
                                  "count": len(grp), "reason": "not_supported", "view": None})

    cap = int(options.max_hole_callouts)
    for name in drawn:
        items = per_view.get(name)
        if not items:
            continue
        bb, db = boxes[name], datum[name]
        s_est = _scale_estimate(bb, scale_hint)
        beyond = (3.0 + style.text_height + 2.0 * style.dim_text_gap) / s_est
        groups = _group_clusters(_clusters(items, np.asarray(views[name].spec.n, dtype=float)))
        corners = {a: np.array([bb.xmax if math.cos(math.radians(a)) > 0 else bb.xmin,
                                bb.ymax if math.sin(math.radians(a)) > 0 else bb.ymin])
                   for a in LEADER_ANGLES}
        placed: List[Tuple[float, np.ndarray]] = []
        leaders: List[Tuple[np.ndarray, np.ndarray]] = []
        all_circles = [(c.centre, c.radii[-1]) for g in groups for c in g]
        diag = max(math.hypot(bb.width, bb.height), 1e-9)
        circular = _circular_outline(views[name], db)
        dc = np.array([0.5 * (db.xmin + db.xmax), 0.5 * (db.ymin + db.ymax)])
        ctol_v = max(0.01, 1e-3 * max(db.width, db.height))
        high_edge = _position_datums(
            [c.centre for g in groups[:max(cap, 0)] for c in g
             if not (circular and float(np.linalg.norm(c.centre - dc)) <= ctol_v)], db)
        for gi, grp in enumerate(groups):
            first = grp[0]
            r_small, r_big = first.radii[0], first.radii[-1]
            if gi >= cap:
                undimensioned.append({"kind": "hole_group", "radius": float(r_small), "count": len(grp),
                                      "reason": "cap", "view": name})
                continue
            best = None
            text = hole_callout_text(first.radii, first.kind, len(grp))
            for ai, a in enumerate(LEADER_ANGLES):
                for cl in grp:
                    exit_len = _leader_exit(cl.centre, cl.radii[-1], a, bb)
                    # the leader runs on past the outline (exit margin + text) before its shelf
                    seg = _leader_segment(cl.centre, cl.radii[-1], a, exit_len + beyond)
                    crossings = sum(1 for s in leaders if _segments_cross(seg, s))
                    crossings += sum(1 for oc, orad in all_circles
                                     if float(np.linalg.norm(oc - cl.centre)) > COAXIAL_TOL
                                     and _segment_point_distance(seg, oc) < orad)
                    cost = (exit_len + _angle_penalty(a, cl.centre, placed, diag)
                            + LEADER_CROSS_WEIGHT * diag * crossings
                            + _text_overhang(seg[0] + exit_len * (seg[1] - seg[0]) / max(exit_len + beyond, 1e-9),
                                             a, text, bb, s_est, style))
                    score = (round(cost, 6), ai, round(float(np.linalg.norm(cl.centre - corners[a])), 6))
                    if best is None or score < best[0]:
                        best = (score, a, cl, seg)
            _, ang, cl, seg = best
            placed.append((ang, cl.centre))
            leaders.append(seg)
            r_arrow = max(float(np.mean([float(f.radius) for f in cl.feats
                                         if abs(float(f.radius) - r_big) <= RADIUS_GROUP_TOL])), 0.0)
            cx, cy = float(cl.centre[0]), float(cl.centre[1])
            dims.append(Dimension(kind="diameter", text=text, value=2.0 * r_small, center=(cx, cy), radius=r_arrow,
                                  leader_angle_deg=ang, base=(cx, cy), view=name, count=len(grp)))
            sides.append(None)
            if first.kind in ("counterbore", "stepped"):
                undimensioned.append({"kind": "counterbore_depth" if first.kind == "counterbore" else "step_depth",
                                      "radius": float(r_big), "count": len(grp), "reason": "not_supported",
                                      "view": name})
            # 3. positions of the callout circle from one reference edge per view and axis
            #    (the edge nearer to the majority of the view's holes, see _position_datums)
            if not options.hole_positions:
                continue
            if circular and float(np.linalg.norm(cl.centre - dc)) <= ctol_v:
                log.debug("plan_dimensions: concentric bore in view %s: no position dims", name)
                continue
            for axis in (0, 1):
                c_ax, c_other = (cx, cy) if axis == 0 else (cy, cx)
                lo_e, hi_e = (db.xmin, db.xmax) if axis == 0 else (db.ymin, db.ymax)
                olo, ohi = (db.ymin, db.ymax) if axis == 0 else (db.xmin, db.xmax)
                high = high_edge[axis]
                edge = hi_e if high else lo_e
                value = abs(c_ax - edge)
                if value <= _MIN_EXTENT:
                    continue
                side_high = (c_other - olo) > (ohi - c_other) + 1e-9
                side = ("top" if side_high else "bottom") if axis == 0 else ("right" if side_high else "left")
                key = (name, axis, ("max" if high else "min"), format_value(value))
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                q = r_arrow if side_high else -r_arrow
                hole_pt = (cx, cy + q) if axis == 0 else (cx + q, cy)
                datum_pt = anchor(name, axis, high, side, edge)
                dims.append(_make_linear(name, bb, axis, side, datum_pt, hole_pt, value))
                sides.append(side)

    # 4. bands: per (view, side), shortest value nearest the outline
    bands: List[Optional[Tuple[str, int]]] = [None] * len(dims)
    keyed: Dict[Tuple[str, str], List[int]] = {}
    for i, (d, side) in enumerate(zip(dims, sides)):
        if side is not None:
            keyed.setdefault((d.view, side), []).append(i)
    for (_, side), idx in keyed.items():
        for band, i in enumerate(sorted(idx, key=lambda j: (dims[j].value, j))):
            bands[i] = (side, band)
    log.debug("plan_dimensions: %d dims, %d undimensioned, %d flags", len(dims), len(undimensioned), len(flags))
    return DimensionPlan(dims=dims, undimensioned=undimensioned, bands=bands, flags=flags)


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


def _linear_frame(dim: Dimension):
    a = math.radians(float(dim.angle_deg))
    d = np.array([math.cos(a), math.sin(a)])
    nrm = np.array([-d[1], d[0]])
    p1 = np.asarray(dim.p1, dtype=float)
    p2 = np.asarray(dim.p2, dtype=float)
    s = float(np.asarray(dim.base, dtype=float) @ nrm)
    t1, t2 = float(p1 @ d), float(p2 @ d)
    q1 = t1 * d + s * nrm
    q2 = t2 * d + s * nrm
    span = abs(t2 - t1)
    u = (q2 - q1) / span if span > 1e-12 else d
    rot = float(dim.angle_deg) % 360.0
    up = nrm
    if rot > 180.0:
        rot -= 360.0
    if rot > 90.0 + 1e-9 or rot <= -90.0 + 1e-9:
        rot = rot - 180.0 if rot > 0 else rot + 180.0
        up = -nrm
    return d, nrm, p1, p2, s, q1, q2, span, u, rot, up


def linear_text_position(dim: Dimension, style: SheetStyle) -> Tuple[float, float]:
    """Anchor (middle of the baseline) of a linear dimension's value text, sheet mm.

    The centre of the dimension line, ``dim_text_gap`` off it on the
    reading side, shifted by ``dim.text_offset`` along the line (p1 -> p2).
    """
    _, _, _, _, _, q1, q2, _, u, _, up = _linear_frame(dim)
    mid = 0.5 * (q1 + q2) + float(dim.text_offset or 0.0) * u
    return _pt(mid + style.dim_text_gap * up)


def _render_linear(dim: Dimension, style: SheetStyle) -> List[SheetItem]:
    d, nrm, p1, p2, s, q1, q2, span, u, rot, up = _linear_frame(dim)
    items: List[SheetItem] = []
    for p, q in ((p1, q1), (p2, q2)):
        off = s - float(p @ nrm)
        e = nrm if off >= 0 else -nrm
        items.append(Segment(_pt(p + style.ext_gap * e), _pt(q + style.ext_overshoot * e), LineStyle.DIMENSION))
    L = style.arrow_length
    shift = float(dim.text_offset or 0.0)
    half_w = 0.5 * TEXT_ADVANCE * style.text_height * len(dim.text or "")
    # line ends as offsets along u from q1 (0 .. span)
    if span >= 2.0 * L + 0.5:
        a_end, b_end = 0.0, span
        arrows = (arrowhead(q1, -u, L, style.arrow_angle_deg), arrowhead(q2, u, L, style.arrow_angle_deg))
    else:     # no room: arrows outside pointing in, line extended past them
        ext = L + 2.0
        a_end, b_end = -ext, span + ext
        arrows = (arrowhead(q1, u, L, style.arrow_angle_deg), arrowhead(q2, -u, L, style.arrow_angle_deg))
    if shift != 0.0:      # a shifted text that leaves the line gets the line extended under it
        c = 0.5 * span + shift
        a_end = min(a_end, c - half_w)
        b_end = max(b_end, c + half_w)
    items.append(Segment(_pt(q1 + a_end * u), _pt(q1 + b_end * u), LineStyle.DIMENSION))
    items.extend(arrows)
    mid = 0.5 * (q1 + q2) + shift * u
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
           "callout_extent", "dims_by_view", "hole_callout_text", "linear_text_position",
           "DIAMETER_SIGN", "LEADER_LEN", "LEADER_ANGLES", "SIDES", "EXTENT_MISMATCH_TOL"]
