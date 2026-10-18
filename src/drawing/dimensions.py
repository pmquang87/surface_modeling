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
   multi-body and chamfered outlines). The dim moves to the opposite side
   (right instead of left, top instead of bottom) when its two extension
   lines are shorter there by more than ``OVERALL_SIDE_MARGIN`` (10 %) of
   the view extent across (QA round 2 issue 15).
   A revolved outline (QA round 2 issue 29): when a principal view's
   outline is a circle on its datum box centre and a FULL boss of that
   radius is seen end-on there (``_outer_diameter``), the two overall dims
   of that view's world axes are replaced by one diameter callout
   ``Ø<2R>`` in that view (leader angle not used by a hole callout there),
   and that boss is not listed as ``boss_group``.
2. Holes (``CylFeature.kind == "hole"``): each hole goes to the first
   DRAWN principal view of ``top, front, right, bottom, back, left`` whose
   normal is parallel to the hole axis (``|dot| > 0.999``); none ->
   ``undimensioned`` with reason ``no_axial_view``. Holes on one axis line
   (2D centres closer than ``COAXIAL_TOL``) are split into axial runs
   (QA round 2 issue 1: spans ``t +- length / 2`` along the view normal,
   a gap above ``max(COAXIAL_TOL, 1e-3 x the model extent along it)``
   starts a new run), so the bores of two opposite walls are two features.
   Each run forms ONE feature: a
   counterbore (two radii, the larger one at exactly one end of the axial
   span) or a stepped bore (any other multi-radius cluster). Of the axial
   views, a feature goes to the first whose circles of that feature are
   drawn visible (``_circle_status``: visible, then not drawn at all, then
   partly hidden); a feature drawn only hidden in every axial view gets no
   callout but ``undimensioned`` reason ``hidden_only`` (QA round 2 issue
   27). Inside a group, the callout circle prefers a visible one. The features
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
   the feature; the value is the smallest diameter. A counterbore on the
   face away from the viewer (QA round 3 issue 3, ``_far_side``: its large
   circle drawn only hidden while the bore is visible, or, without drawn
   circles, the large cylinder behind the bore along the view normal) gets
   its arrow on the visible bore circle and the text ``4 x Ø6 / Ø10 cbore
   far side`` (``stats["far_side_callouts"]``); near and far
   counterbores of one size are separate groups. ``layout.compose``
   moves a callout whose text still collides.
   Bosses, fillets and rounds are not dimensioned: one ``undimensioned``
   entry per radius group (``boss_group`` / ``fillet_group`` /
   ``round_group``, reason ``not_supported``). Holes a fastener of another
   solid fills (``filled_hole``): ``hole_group`` reason
   ``filled_by_fastener``. With ``n_bodies`` > 1: ``body_extent`` (count
   n) and ``body_position`` (count n - 1), reason ``not_supported``.
3. Hole positions (``hole_positions``): each kept group gets an X and a Y
   linear dimension from ONE reference edge per
   view and axis (left or right, bottom or top): the edge nearer to the
   majority of the view's hole circles, left / bottom on a tie
   (``_position_datums``; mixing datums per hole inside one view is harder
   to make and inspect). The measured circle is the group's circle nearest
   that datum corner, plus the cost of its extension lines (QA round 2
   issue 15; the callout circle wins a tie). Each dimension sits on the
   side whose extension line, from the quadrant of the largest circle
   facing that side to the outline, is shorter, each other circle it runs
   through counting ``POSITION_CROSS_WEIGHT`` x the view diagonal (the
   nearer side on a tie). The value is to the centre. A
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
   ``DimensionPlan.text_sides[i]`` names the end (+1 p2, -1 p1) an
   outside value text should go past (``_text_sides``: the end whose text
   crosses fewer extension lines of the dims further out on that side);
   the layout applies it with ``apply_text_side``. ``DimensionPlan.stats``
   counts the QA round 2 decisions.

Rendering conventions (``style.SheetStyle``; the numeric values are DIN 406
teaching practice, not ISO, see ``style.py``): closed filled arrowhead with
15 degree included angle, 3.5 mm long; extension line from ``ext_gap``
beyond the measured point to ``ext_overshoot`` past the dimension line;
value ``dim_text_gap`` above the line, centred (shifted along the line by
``Dimension.text_offset``, the dimension line extended under a text that
leaves the span), reading from the bottom or from the right (vertical
dimensions are rotated 90 degrees, text on the left of the line); when the
span is shorter than two arrows plus 0.5 mm the arrows go outside,
pointing in. A value text wider than the span minus two ``dim_text_gap``
goes outside (``outside_text_offset``): past the p2 end by default,
``arrow_length + SHORT_SPAN_EXTRA`` beyond the extension line, on the
extended dimension line; a non-zero ``text_offset`` wins (QA round 2
issue 17). ``linear_reach`` gives the reach past both ends for the
layout. Diameter: arrow tip on the circle pointing at the centre,
leader along ``leader_angle_deg`` (``LEADER_LEN`` mm, or out to ``base``
when ``base`` lies further out along the leader than one arrow length
beyond the circle), a horizontal shelf under the text, text anchored at
the leader end.

Values: ``format_value`` rounds half-up to 0.01 mm (``Decimal`` on the
shortest repr, so 12.345 -> "12.35" and 2.675 -> "2.68" although the binary
floats lie just below) and drops trailing zeros; no unit symbol (the sheet
says "Dimensions in mm"). Inch-fraction values (an exact 0.001 mm value
that is also a whole number of thousandths of an inch, 3.175 = 1/8 in)
keep three decimals (QA round 2 issue 26).
"""
from __future__ import annotations

import dataclasses
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
    # counters of the planning decisions (QA round 2), JSON-ready:
    # coaxial_runs_split (extra hole features from splitting an axis line into axial runs),
    # hidden_only (hole features drawn only hidden in every axial view: no callout),
    # od_diameter (view name of the outer-diameter callout that replaced two overall dims,
    # or None), position_refs_moved (position dims taken from another circle of the group
    # than the callout circle), position_sides_switched (position dims put on the farther
    # side because the nearer extension line crosses another circle),
    # overall_sides_switched (overall dims moved to the opposite side: shorter extension lines)
    stats: dict = field(default_factory=dict)
    # parallel to dims: for a linear dim, the end a value text goes past when it does not fit
    # between the extension lines (+1 beyond p2, -1 beyond p1; ``apply_text_side``), None for
    # diameters; empty for plans built by hand (the renderer then uses +1)
    text_sides: List[Optional[int]] = field(default_factory=list)


# ------------------------------------------------------------------ values --

INCH_MM = 25.4
_THOU_TOL_MM = 1e-5           # mm: a value this close to an exact thousandth is that thousandth


def _inch_thousandth(x: float) -> bool:
    """True when ``x`` mm is an exact 3-decimal mm value AND a whole number of
    thousandths of an inch (3.175 = 0.125 in, 9.525 = 0.375 in, 15.875 = 0.625 in).

    The inch condition keeps metric values such as 12.345 or 2.675 on the
    0.01 rule (QA round 2 issue 26: only inch-fraction nominals lost their
    last digit; a bare "exact thousandth" rule would also change 12.345 ->
    "12.345", which the sheet convention rounds to 12.35).
    """
    k = x * 1000.0
    if abs(k - round(k)) > _THOU_TOL_MM * 1000.0:
        return False
    thou = round(k) / INCH_MM                  # thousandths of an inch
    return abs(thou - round(thou)) <= 1e-6 * max(1.0, abs(thou))


def format_value(v: float) -> str:
    """Model mm -> dimension text: half-up to 0.01, no trailing zeros, "-0" -> "0".

    Exception (QA round 2 issue 26): an exact thousandth of a mm that is also
    a whole number of thousandths of an inch (inch-fraction nominals such as
    3.175 = 1/8 in) keeps its three decimals when the 0.01 rounding would
    change it: 3.175 -> "3.175", 3.1749 -> "3.17", 12.345 -> "12.35".
    """
    x = float(v)
    if not math.isfinite(x):
        raise ValueError(f"format_value: not a finite number: {v!r}")
    d = Decimal(repr(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    if _inch_thousandth(x):
        d3 = Decimal(repr(x)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
        if d3 != d:
            d = d3
    s = format(d, "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    if s in ("-0", ""):
        s = "0"
    return s


def _diameter_text(radius: float, count: int) -> str:
    base = f"{DIAMETER_SIGN}{format_value(2.0 * radius)}"
    return f"{count} x {base}" if count > 1 else base


#: QA round 3 issue 3: the kind word of a far-side counterbore callout. Compact on purpose:
#: the first wording ('... counterbore (cbore far side)', 17 characters more than the near-side
#: text) took the whole top band of NIST FTC 10 and pushed '8 x Ø2.5' to a 93 mm leader
#: across the view; 'cbore far side' is 3 characters more than 'counterbore'.
FAR_SIDE_NOTE = "cbore far side"


def hole_callout_text(radii, kind: str, count: int, far_side: bool = False) -> str:
    """Callout text of a hole feature group: ``4 x Ø6.5``, ``Ø10.5 / Ø15.9 counterbore``.

    ``radii`` ascending; ``kind`` "hole" | "counterbore" | "stepped". ``far_side``
    (QA round 3 issue 3): the counterbore lies on the face away from the viewer; the kind
    word is ``FAR_SIDE_NOTE`` instead (``4 x Ø6 / Ø10 cbore far side``).
    """
    parts = " / ".join(f"{DIAMETER_SIGN}{format_value(2.0 * float(r))}" for r in radii)
    if far_side and kind == "counterbore":
        base = f"{parts} {FAR_SIDE_NOTE}"
    else:
        base = parts if kind == "hole" else f"{parts} {kind}"
    return f"{count} x {base}" if count > 1 else base


def _far_side(cl: "_Cluster", index: Dict[str, np.ndarray], n: np.ndarray) -> bool:
    """True when the counterbore of ``cl`` (two radii) lies on the face away from the viewer
    of the view with normal ``n`` (toward the viewer) and index ``index`` (QA round 3 issue 3).

    The view decides when it draws the circles: the large one visible -> near side; the
    large one only hidden while the small one is visible -> far side. Otherwise (a
    synthetic view without circles) the axial positions decide: the large cylinder's
    centre behind the small one's along ``n``.
    """
    if cl.kind != "counterbore" or len(cl.radii) != 2:
        return False
    r_small, r_big = cl.radii[0], cl.radii[-1]
    big_vis = _has_circle(index["visible"], cl.centre, r_big)
    if big_vis:
        return False
    if _has_circle(index["hidden"], cl.centre, r_big) and \
            _has_circle(index["visible"], cl.centre, r_small):
        return True
    try:
        t = {0: [], 1: []}
        for f in cl.feats:
            k = 1 if abs(float(f.radius) - r_big) <= RADIUS_GROUP_TOL else 0
            t[k].append(float(np.dot(np.asarray(f.center, dtype=float), n)))
        if not t[0] or not t[1]:
            return False
        return float(np.mean(t[1])) < float(np.mean(t[0])) - 1e-9
    except Exception:
        return False


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
    status: str = "unknown"       # how the view draws its circles: visible | partial | hidden | unknown
    far: bool = False             # counterbore on the face away from the viewer (QA round 3 issue 3)


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


# Split the holes of one axis line into axial runs (QA round 2 issue 1). Module constant so
# the tests can switch it off as a negative control (the bores of two opposite walls then
# merge into one "stepped" feature again).
_split_axial = True

# A gap between two coaxial faces of DIFFERENT radius up to this factor x |r_a - r_b| is a
# conical shoulder (a chamfered counterbore step or a drill point: CylFeature lists only the
# cylinders, so the cone shows as a gap of |dr| / tan(half angle), 0.6 x |dr| at 118 deg,
# 1.0 x |dr| at 90 deg) and does not split the bore. 0 switches the bridge off (test control).
_STEP_GAP_FACTOR = 2.0


def _axial_runs(cl: List[Tuple[CylFeature, np.ndarray]], n: np.ndarray, tol: float
                ) -> List[List[Tuple[CylFeature, np.ndarray]]]:
    """Holes of one axis line -> runs of touching / overlapping axial spans.

    Each hole spans ``t +- |length| / 2`` with ``t = centre . n``; a new run
    starts where the next span (sorted by start) begins more than ``tol``
    after the end of the current run. A counterbore (two faces that touch)
    stays one run; the bores of two opposite walls become two. A gap to a
    face of another radius (vs the face that ends the run) up to
    ``_STEP_GAP_FACTOR x |dr|`` is a conical step and keeps the run.
    """
    if not _split_axial or len(cl) < 2:
        return [cl]
    spans = []
    for f, c in cl:
        try:
            t = float(np.dot(np.asarray(f.center, dtype=float), n))
            h = 0.5 * abs(float(f.length))
        except Exception:
            return [cl]
        if not (math.isfinite(t) and math.isfinite(h)):
            return [cl]
        spans.append((t - h, t + h, f, c))
    spans.sort(key=lambda s: (s[0], s[1]))
    runs: List[List[Tuple[CylFeature, np.ndarray]]] = []
    end = -math.inf
    end_r = 0.0                                   # radius of the face that ends the current run
    for lo, hi, f, c in spans:
        try:
            r = abs(float(f.radius))
        except Exception:
            r = end_r
        dr = abs(r - end_r)
        bridge = (runs and dr > RADIUS_GROUP_TOL
                  and lo - end <= _STEP_GAP_FACTOR * dr + tol)
        if runs and (lo <= end + tol or bridge):
            runs[-1].append((f, c))
            if hi > end:
                end, end_r = hi, r
        else:
            runs.append([(f, c)])
            end, end_r = hi, r
    return runs


def _clusters(items: List[Tuple[CylFeature, np.ndarray]], n: np.ndarray,
              axial_tol: Optional[float] = None, stats: Optional[dict] = None) -> List[_Cluster]:
    """Coaxial holes of one view (2D centres within COAXIAL_TOL), split into
    axial runs (``_axial_runs``, gap above ``axial_tol``, default
    ``COAXIAL_TOL``) -> one cluster per run. ``stats["coaxial_runs_split"]``
    (when given) counts the extra clusters the split made."""
    out: List[List[Tuple[CylFeature, np.ndarray]]] = []
    for f, c in items:
        for cl in out:
            if float(np.linalg.norm(cl[0][1] - c)) <= COAXIAL_TOL:
                cl.append((f, c))
                break
        else:
            out.append([(f, c)])
    tol = COAXIAL_TOL if axial_tol is None else max(float(axial_tol), 0.0)
    nn = np.asarray(n, dtype=float).reshape(-1)
    res = []
    for line in out:
        runs = _axial_runs(line, nn, tol)
        if stats is not None and len(runs) > 1:
            stats["coaxial_runs_split"] = stats.get("coaxial_runs_split", 0) + len(runs) - 1
        for cl in runs:
            feats = [f for f, _ in cl]
            radii = _merged_radii(feats)
            centre = np.mean(np.array([c for _, c in cl]), axis=0)
            res.append(_Cluster(centre, feats, radii, _cluster_kind(feats, radii, nn)))
    return res


# ------------------------------------------------- circle visibility (issue 27) --

_STATUS_RANK = {"visible": 0.0, "unknown": 0.5, "partial": 1.0, "hidden": 3.0}
_FIT_MIN_POINTS = 5
_FIT_MIN_SWEEP_DEG = 90.0


def _fit_circle(pts: np.ndarray) -> Optional[Tuple[float, float, float]]:
    """Least-squares circle (Kasa) through a polyline, or None when the points
    do not lie on one circle (1 % of r) or sweep less than 90 degrees."""
    p = np.asarray(pts, dtype=float).reshape(-1, 2)
    if p.shape[0] < _FIT_MIN_POINTS or not np.all(np.isfinite(p)):
        return None
    x, y = p[:, 0], p[:, 1]
    a = np.column_stack([x, y, np.ones_like(x)])
    try:
        sol = np.linalg.lstsq(a, x * x + y * y, rcond=None)[0]
    except Exception:
        return None
    cx, cy = 0.5 * float(sol[0]), 0.5 * float(sol[1])
    r2 = float(sol[2]) + cx * cx + cy * cy
    if not math.isfinite(r2) or r2 <= 0.0:
        return None
    r = math.sqrt(r2)
    d = np.hypot(x - cx, y - cy)
    if float(np.max(np.abs(d - r))) > max(0.01, 0.01 * r):
        return None
    ang = np.sort(np.mod(np.arctan2(y - cy, x - cx), 2.0 * math.pi))
    gaps = np.diff(np.concatenate([ang, ang[:1] + 2.0 * math.pi]))
    if math.degrees(2.0 * math.pi - float(np.max(gaps))) < _FIT_MIN_SWEEP_DEG:
        return None
    return cx, cy, r


# A polyline arc of radius r sweeping >= _FIT_MIN_SWEEP_DEG (90 deg) has a bbox diagonal
# between its chord r * sqrt(2) and the full-circle 2 * sqrt(2) * r; with the 1 % fit
# tolerance these bounds (plus a small absolute margin) pre-select the polylines worth a fit.
_FIT_DIAG_LO = math.sqrt(2.0) * 0.98
_FIT_DIAG_HI = 2.0 * math.sqrt(2.0) * 1.02
_FIT_DIAG_EPS = 0.05


def _fit_candidate(pts: np.ndarray, radii: Optional[np.ndarray]) -> bool:
    """Cheap prefilter before ``_fit_circle``: the polyline has enough points and
    (when ``radii`` is given, sorted ascending) a bbox diagonal an arc of >= 90
    degrees of one of those radii can have."""
    p = pts if isinstance(pts, np.ndarray) and pts.ndim == 2 else np.asarray(pts, dtype=float).reshape(-1, 2)
    if p.shape[0] < _FIT_MIN_POINTS:
        return False
    if radii is None:
        return True
    if radii.size == 0:
        return False
    w, h = (p.max(axis=0) - p.min(axis=0)).tolist()[:2]
    d = math.hypot(w, h)
    if not math.isfinite(d):
        return False
    lo = (d - _FIT_DIAG_EPS) / _FIT_DIAG_HI          # smallest radius that fits this diagonal
    hi = (d + _FIT_DIAG_EPS) / _FIT_DIAG_LO          # largest
    i = int(np.searchsorted(radii, lo, side="left"))
    return i < radii.size and float(radii[i]) <= hi


def _circle_rows(prims, radii: Optional[np.ndarray] = None) -> np.ndarray:
    rows = []
    for p in prims:
        try:
            if isinstance(p, (Circle, Arc)):
                rows.append((float(p.center[0]), float(p.center[1]), abs(float(p.radius))))
            elif isinstance(p, Polyline):
                if not _fit_candidate(p.points, radii):
                    continue
                fit = _fit_circle(p.points)
                if fit is not None:
                    rows.append(fit)
        except Exception:                         # malformed primitive: not a circle
            continue
    arr = np.asarray(rows, dtype=float).reshape(-1, 3)
    return arr[np.all(np.isfinite(arr), axis=1)]


def _circle_index(v: ProjectedView, radii=None) -> Dict[str, np.ndarray]:
    """``{"visible": (k, 3), "hidden": (m, 3)}`` rows ``(cx, cy, r)`` of the
    view's circles, arcs and circular polylines.

    ``radii``: optional radii of the holes looked up in this view; when given,
    only polylines whose size fits an arc of one of them are circle-fitted
    (``_fit_candidate``; review of issue 27: the fit costs ~65 us per polyline).
    Circle and Arc primitives are always indexed."""
    rr = None
    if radii is not None:
        try:
            rr = np.sort(np.abs(np.asarray(list(radii), dtype=float).reshape(-1)))
            rr = rr[np.isfinite(rr)]
        except Exception:
            rr = None
    return {"visible": _circle_rows(v.visible, rr), "hidden": _circle_rows(v.hidden, rr)}


def _has_circle(rows: np.ndarray, c: np.ndarray, r: float) -> bool:
    if rows.shape[0] == 0:
        return False
    tol = max(0.02, 2e-3 * abs(float(r)))
    d = np.hypot(rows[:, 0] - float(c[0]), rows[:, 1] - float(c[1]))
    return bool(np.any((d <= tol) & (np.abs(rows[:, 2] - abs(float(r))) <= tol)))


def _circle_status(index: Dict[str, np.ndarray], c: np.ndarray, radii: List[float]) -> str:
    """How a view draws a hole feature (centre ``c``, ``radii``):

    ``visible``: every radius found drawn visible, or some visible and the
    rest not drawn at all; ``partial``: some visible, some only hidden;
    ``hidden``: drawn, but only hidden; ``unknown``: no circle of these
    radii there (a synthetic view, an unfitted polyline).
    """
    n_vis = n_hid = 0
    for r in radii:
        if _has_circle(index["visible"], c, r):
            n_vis += 1
        elif _has_circle(index["hidden"], c, r):
            n_hid += 1
    if n_vis and not n_hid:
        return "visible"
    if n_vis:
        return "partial"
    if n_hid:
        return "hidden"
    return "unknown"


def _same_signature(a: _Cluster, b: _Cluster) -> bool:
    return (a.kind == b.kind and len(a.radii) == len(b.radii) and a.far == b.far
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
STATUS_WEIGHT = 2.0           # x view diagonal x _STATUS_RANK: callouts / positions prefer visible circles
POSITION_CROSS_WEIGHT = 0.25  # x view diagonal: cost per other circle a position extension line crosses


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


_OPPOSITE = {"bottom": "top", "top": "bottom", "left": "right", "right": "left"}
OVERALL_SIDE_MARGIN = 0.1     # x view extent across: an overall dim changes side only when that much shorter
_VIEW_AXES = {0: "width", 1: "depth", 2: "height"}


def _ext_sum(bb: BBox2D, axis: int, side: str, p1, p2) -> float:
    """Summed length of the two extension lines of an overall dim from its anchors to the outline on ``side``."""
    oa = 1 - axis
    if axis == 0:
        line = bb.ymax if side == "top" else bb.ymin
    else:
        line = bb.xmax if side == "right" else bb.xmin
    return abs(line - float(p1[oa])) + abs(line - float(p2[oa]))


def _hole_ext_line(c: np.ndarray, r: float, axis: int, side_high: bool, bb: BBox2D):
    """Extension line (segment, length) of a position dim from the circle quadrant to the outline.

    ``axis`` 0 (an X position, horizontal dim): vertical line at ``x = cx``
    to the top / bottom edge; axis 1: horizontal line to the right / left.
    """
    cx, cy = float(c[0]), float(c[1])
    q = abs(float(r)) if side_high else -abs(float(r))
    if axis == 0:
        a = (cx, cy + q)
        b = (cx, bb.ymax if side_high else bb.ymin)
    else:
        a = (cx + q, cy)
        b = (bb.xmax if side_high else bb.xmin, cy)
    return (np.array(a), np.array(b)), float(math.hypot(b[0] - a[0], b[1] - a[1]))


def _circle_arrays(circles) -> Tuple[np.ndarray, np.ndarray]:
    """``[(centre, r), ...]`` -> ``(centres (k, 2), radii (k,))``; arrays pass through."""
    if isinstance(circles, tuple) and len(circles) == 2 and isinstance(circles[0], np.ndarray):
        return circles
    if not circles:
        return np.zeros((0, 2)), np.zeros(0)
    return (np.array([np.asarray(c, dtype=float).reshape(2) for c, _ in circles]),
            np.array([abs(float(r)) for _, r in circles]))


def _crossed_circles(seg, own: np.ndarray, circles) -> int:
    """Other circles (centre farther than COAXIAL_TOL from ``own``) that ``seg`` passes through.

    ``circles``: ``[(centre, r), ...]`` or ``_circle_arrays`` output (vectorised)."""
    cen, rad = _circle_arrays(circles)
    if cen.shape[0] == 0:
        return 0
    a, b = np.asarray(seg[0], dtype=float), np.asarray(seg[1], dtype=float)
    d = b - a
    L2 = float(d @ d)
    t = np.zeros(cen.shape[0]) if L2 < 1e-24 else np.clip((cen - a) @ d / L2, 0.0, 1.0)
    dist = np.linalg.norm(a + t[:, None] * d - cen, axis=1)
    other = np.linalg.norm(cen - np.asarray(own, dtype=float).reshape(2), axis=1) > COAXIAL_TOL
    return int(np.count_nonzero(other & (dist < rad)))


def _side_cost(c: np.ndarray, r: float, axis: int, side_high: bool, bb: BBox2D, circles, diag: float) -> float:
    """Cost of a position dim's extension line on one side: its length (view mm) plus
    ``POSITION_CROSS_WEIGHT`` x ``diag`` per other circle it runs through (QA round 2 issue 15)."""
    seg, length = _hole_ext_line(c, r, axis, side_high, bb)
    return length + POSITION_CROSS_WEIGHT * diag * _crossed_circles(seg, np.asarray(c, dtype=float), circles)


def _outer_diameter(views: Dict[str, ProjectedView], datum: Dict[str, BBox2D],
                    features: Optional[List[CylFeature]]) -> Optional[dict]:
    """The revolved outline of QA round 2 issue 29, or None.

    The first principal view (``front, top, right, back, bottom, left``)
    whose outline is a circle on its datum box centre (``_circular_outline``)
    and that sees a FULL boss of that radius end-on, centred there (a D shape
    or a partial cylinder does not count). Returns ``{"view", "radius",
    "centre", "feats", "axes"}``; ``axes`` are the overall-extent names
    (width / depth / height) of the view's two in-plane world axes.
    """
    bosses = [f for f in features or [] if getattr(f, "kind", None) == "boss" and bool(getattr(f, "full_circle", False))]
    if not bosses:
        return None
    for name in ("front", "top", "right", "back", "bottom", "left"):
        v, db = views.get(name), datum.get(name)
        if v is None or db is None:
            continue
        try:
            if not _circular_outline(v, db):
                continue
            n = np.asarray(v.spec.n, dtype=float)
            R = 0.5 * db.width
            dc = np.array([0.5 * (db.xmin + db.xmax), 0.5 * (db.ymin + db.ymax)])
            tol = max(0.01, 2e-3 * R)
            match = []
            for f in bosses:
                ax = _unit(f.axis)
                if ax is None or abs(float(np.dot(ax, n))) <= AXIS_PARALLEL_DOT:
                    continue
                if abs(float(f.radius) - R) > tol:
                    continue
                c2 = v.spec.to_view(np.asarray(f.center, dtype=float))[0, :2]
                if float(np.linalg.norm(c2 - dc)) > tol:
                    continue
                match.append(f)
            if not match:
                continue
            axes = tuple(_VIEW_AXES[int(np.argmax(np.abs(np.asarray(vec, dtype=float))))]
                         for vec in (v.spec.x, v.spec.y))
        except Exception as exc:                  # malformed view / feature: no OD callout
            log.debug("plan_dimensions: outer diameter check of view %s failed: %s", name, exc)
            continue
        return {"view": name, "radius": float(np.mean([float(f.radius) for f in match])), "centre": dc,
                "feats": match, "axes": axes}
    return None


def plan_dimensions(views: Dict[str, ProjectedView], features: List[CylFeature], options: DrawingOptions,
                    style: SheetStyle, model_extents: np.ndarray, *,
                    model_bounds: Optional[np.ndarray] = None,
                    scale_hint: Optional[float] = None,
                    n_bodies: Optional[int] = None) -> DimensionPlan:
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
    (``_scale_estimate``). ``n_bodies``: optional number of solids / mesh
    components; above 1 the body sizes and the offsets between the bodies
    are disclosed as ``undimensioned`` (``body_extent`` count n,
    ``body_position`` count n - 1, reason ``not_supported``).
    """
    dims: List[Dimension] = []
    sides: List[Optional[str]] = []
    undimensioned: List[dict] = []
    flags: List[Tuple[str, str, str]] = []
    stats = {"coaxial_runs_split": 0, "hidden_only": 0, "od_diameter": None, "position_refs_moved": 0,
             "position_sides_switched": 0, "overall_sides_switched": 0, "far_side_callouts": 0}
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

    # 0. a revolved outline (QA round 2 issue 29): a principal view whose outline is a
    #    circle on the datum box centre AND a full boss of that radius seen end-on -> one
    #    "Ø" callout replaces the two overall dims of that view's world axes
    od = _outer_diameter(views, datum, features)
    skip_overall = set()
    if od is not None:
        skip_overall = set(od["axes"])
        stats["od_diameter"] = od["view"]

    # 1. overall extents, once each; values from the exact model extents
    for what, rules in _EXTENT_RULES:
        if what in skip_overall:
            log.debug("plan_dimensions: overall %s given by the outer diameter in view %s", what, od["view"])
            continue
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
            # issue 15: the opposite side when its extension lines are clearly shorter
            alt = _OPPOSITE[side]
            q1 = anchor(view, axis, False, alt, lo_e)
            q2 = anchor(view, axis, True, alt, hi_e)
            across = bb.height if axis == 0 else bb.width
            if (_ext_sum(bb, axis, alt, q1, q2)
                    < _ext_sum(bb, axis, side, p1, p2) - max(OVERALL_SIDE_MARGIN * across, tol_of(bb))):
                side, p1, p2 = alt, q1, q2
                stats["overall_sides_switched"] += 1
            dims.append(_make_linear(view, bb, axis, side, p1, p2, value))
            sides.append(side)
            seen_keys.add((view, axis, "overall", dims[-1].text))
            placed = True
            break
        if not placed:
            log.debug("plan_dimensions: no view for the overall %s", what)

    # 2. hole callouts
    drawn = [v for v in HOLE_VIEW_ORDER if v in views and boxes.get(v) is not None]
    by_cands: Dict[Tuple[str, ...], List[Tuple[CylFeature, np.ndarray]]] = {}
    orphans: List[Tuple[CylFeature, np.ndarray]] = []
    others: Dict[str, List[CylFeature]] = {}
    filled: List[CylFeature] = []
    od_ids = {id(f) for f in od["feats"]} if od is not None else set()
    for f in features or []:
        kind = getattr(f, "kind", None)
        if kind in ("boss", "fillet", "round"):
            if id(f) not in od_ids:              # the OD boss is dimensioned by its Ø callout
                others.setdefault(kind, []).append(f)
            continue
        if kind == "filled_hole":
            filled.append(f)
            continue
        if kind != "hole":
            continue
        axis = _unit(f.axis)
        cands: Tuple[str, ...] = ()
        if axis is not None:
            cands = tuple(name for name in drawn
                          if abs(float(np.dot(axis, views[name].spec.n))) > AXIS_PARALLEL_DOT)
        if not cands:
            orphans.append((f, np.zeros(2)))
            continue
        c2 = views[cands[0]].spec.to_view(np.asarray(f.center, dtype=float))[0, :2]
        by_cands.setdefault(cands, []).append((f, c2))

    for grp in _group_by_radius(orphans):
        undimensioned.append({"kind": "hole_group", "radius": float(grp[0][0].radius), "count": len(grp),
                              "reason": "no_axial_view", "view": None})
    for grp in _group_features_by_radius(filled):
        undimensioned.append({"kind": "hole_group", "radius": float(np.mean([float(f.radius) for f in grp])),
                              "count": len(grp), "reason": "filled_by_fastener", "view": None})
    for kind in ("boss", "fillet", "round"):
        for grp in _group_features_by_radius(others.get(kind, [])):
            undimensioned.append({"kind": f"{kind}_group",
                                  "radius": float(np.mean([float(f.radius) for f in grp])),
                                  "count": len(grp), "reason": "not_supported", "view": None})
    if n_bodies is not None:
        try:
            nb = int(n_bodies)
        except (TypeError, ValueError):
            nb = 0
        if nb > 1:                                # QA round 2 issue 16: multi-body disclosure
            undimensioned.append({"kind": "body_extent", "count": nb, "reason": "not_supported", "view": None})
            undimensioned.append({"kind": "body_position", "count": nb - 1, "reason": "not_supported",
                                  "view": None})

    # holes of one axis line -> features (axial runs), each to the axial view that draws it
    # visible (QA round 2 issues 1 and 27)
    indexes: Dict[str, Dict[str, np.ndarray]] = {}
    view_radii: Dict[str, List[float]] = {}
    for cands, items in by_cands.items():
        for name in cands:
            view_radii.setdefault(name, []).extend(float(f.radius) for f, _ in items)

    def index_of(name: str) -> Dict[str, np.ndarray]:
        if name not in indexes:
            indexes[name] = _circle_index(views[name], view_radii.get(name, []))
        return indexes[name]

    per_view: Dict[str, List[_Cluster]] = {}
    hidden_only: Dict[str, List[_Cluster]] = {}
    for cands, items in by_cands.items():
        n0 = np.asarray(views[cands[0]].spec.n, dtype=float)
        extent_n = float(np.abs(n0) @ ext) if ext is not None else 0.0
        for cl in _clusters(items, n0, max(COAXIAL_TOL, 1e-3 * extent_n), stats=stats):
            c3 = np.mean(np.array([np.asarray(f.center, dtype=float) for f in cl.feats]), axis=0)
            best = None
            for oi, name in enumerate(cands):
                c_v = views[name].spec.to_view(c3)[0, :2]
                st = _circle_status(index_of(name), c_v, cl.radii)
                key = (_STATUS_RANK[st], oi)
                if best is None or key < best[0]:
                    best = (key, name, c_v, st)
            _, name, c_v, st = best
            if st == "hidden":
                hidden_only.setdefault(cands[0], []).append(cl)
                continue
            placed_cl = _Cluster(c_v, cl.feats, cl.radii, cl.kind, st)
            placed_cl.far = _far_side(placed_cl, index_of(name),
                                      np.asarray(views[name].spec.n, dtype=float))
            per_view.setdefault(name, []).append(placed_cl)
    for name, cls in hidden_only.items():
        for grp in _group_clusters(cls):
            stats["hidden_only"] += len(grp)
            undimensioned.append({"kind": "hole_group", "radius": float(grp[0].radii[0]), "count": len(grp),
                                  "reason": "hidden_only", "view": name})

    cap = int(options.max_hole_callouts)
    view_leaders: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {}
    view_angles: Dict[str, List[Tuple[float, np.ndarray]]] = {}
    for name in drawn:
        clusters = per_view.get(name)
        if not clusters:
            continue
        bb, db = boxes[name], datum[name]
        s_est = _scale_estimate(bb, scale_hint)
        beyond = (3.0 + style.text_height + 2.0 * style.dim_text_gap) / s_est
        groups = _group_clusters(clusters)
        corners = {a: np.array([bb.xmax if math.cos(math.radians(a)) > 0 else bb.xmin,
                                bb.ymax if math.sin(math.radians(a)) > 0 else bb.ymin])
                   for a in LEADER_ANGLES}
        placed: List[Tuple[float, np.ndarray]] = view_angles.setdefault(name, [])
        leaders: List[Tuple[np.ndarray, np.ndarray]] = view_leaders.setdefault(name, [])
        all_circles = [(c.centre, c.radii[-1]) for g in groups for c in g]
        circle_arr = _circle_arrays(all_circles)
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
            text = hole_callout_text(first.radii, first.kind, len(grp), far_side=first.far)
            # QA round 3 issue 3: a counterbore on the far face is called out on its visible
            # bore circle (the large circle is dashed there) and says so in the text
            r_lead = r_small if first.far else r_big
            if first.far:
                stats["far_side_callouts"] += 1
            for ai, a in enumerate(LEADER_ANGLES):
                for cl in grp:
                    exit_len = _leader_exit(cl.centre, r_lead, a, bb)
                    # the leader runs on past the outline (exit margin + text) before its shelf
                    seg = _leader_segment(cl.centre, r_lead, a, exit_len + beyond)
                    crossings = sum(1 for s in leaders if _segments_cross(seg, s))
                    crossings += _crossed_circles(seg, cl.centre, circle_arr)
                    cost = (exit_len + _angle_penalty(a, cl.centre, placed, diag)
                            + LEADER_CROSS_WEIGHT * diag * crossings
                            + STATUS_WEIGHT * diag * _STATUS_RANK.get(cl.status, 0.5)
                            + _text_overhang(seg[0] + exit_len * (seg[1] - seg[0]) / max(exit_len + beyond, 1e-9),
                                             a, text, bb, s_est, style))
                    score = (round(cost, 6), ai, round(float(np.linalg.norm(cl.centre - corners[a])), 6))
                    if best is None or score < best[0]:
                        best = (score, a, cl, seg)
            _, ang, cl, seg = best
            placed.append((ang, cl.centre))
            leaders.append(seg)
            r_arrow = max(float(np.mean([float(f.radius) for f in cl.feats
                                         if abs(float(f.radius) - r_lead) <= RADIUS_GROUP_TOL])), 0.0)
            cx, cy = float(cl.centre[0]), float(cl.centre[1])
            dims.append(Dimension(kind="diameter", text=text, value=2.0 * r_small, center=(cx, cy), radius=r_arrow,
                                  leader_angle_deg=ang, base=(cx, cy), view=name, count=len(grp)))
            sides.append(None)
            if first.kind in ("counterbore", "stepped"):
                undimensioned.append({"kind": "counterbore_depth" if first.kind == "counterbore" else "step_depth",
                                      "radius": float(r_big), "count": len(grp), "reason": "not_supported",
                                      "view": name})
            # 3. positions from one reference edge per view and axis (the edge nearer to the
            #    majority of the view's holes, see _position_datums), measured to the circle of
            #    the group nearest that datum corner whose extension lines cross the fewest
            #    other circles (QA round 2 issue 15), on the shorter free side per dim
            if not options.hole_positions:
                continue
            corner = np.array([db.xmax if high_edge[0] else db.xmin, db.ymax if high_edge[1] else db.ymin])

            def ref_score(c: _Cluster, _corner=corner, _bb=bb, _circles=circle_arr, _cl=cl, _diag=diag):
                r = c.radii[-1]
                ext_cost = sum(min(_side_cost(c.centre, r, ax, sh, _bb, _circles, _diag) for sh in (False, True))
                               for ax in (0, 1))
                cost = (float(np.linalg.norm(c.centre - _corner)) + ext_cost
                        + STATUS_WEIGHT * _diag * _STATUS_RANK.get(c.status, 0.5))
                return (round(cost, 6), 0 if c is _cl else 1)

            ref = min(grp, key=ref_score)
            if ref is not cl:
                stats["position_refs_moved"] += 1
            if circular and float(np.linalg.norm(ref.centre - dc)) <= ctol_v:
                log.debug("plan_dimensions: concentric bore in view %s: no position dims", name)
                continue
            rx, ry = float(ref.centre[0]), float(ref.centre[1])
            r_ref = max(float(np.mean([float(f.radius) for f in ref.feats
                                       if abs(float(f.radius) - r_lead) <= RADIUS_GROUP_TOL] or [r_lead])), 0.0)
            for axis in (0, 1):
                c_ax, c_other = (rx, ry) if axis == 0 else (ry, rx)
                lo_e, hi_e = (db.xmin, db.xmax) if axis == 0 else (db.ymin, db.ymax)
                olo, ohi = (db.ymin, db.ymax) if axis == 0 else (db.xmin, db.xmax)
                high = high_edge[axis]
                edge = hi_e if high else lo_e
                value = abs(c_ax - edge)
                if value <= _MIN_EXTENT:
                    continue
                key = (name, axis, ("max" if high else "min"), format_value(value))
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                nearer_high = (c_other - olo) > (ohi - c_other) + 1e-9
                # the side with the shorter extension line, each other circle it crosses
                # counting POSITION_CROSS_WEIGHT x the view diagonal; a tie keeps the nearer side
                side_high, best_side = nearer_high, None
                for sh in (nearer_high, not nearer_high):
                    sc = _side_cost(ref.centre, r_ref, axis, sh, bb, circle_arr, diag)
                    if best_side is None or sc < best_side - 1e-9:
                        best_side, side_high = sc, sh
                if side_high != nearer_high:
                    stats["position_sides_switched"] += 1
                side = ("top" if side_high else "bottom") if axis == 0 else ("right" if side_high else "left")
                q = r_ref if side_high else -r_ref
                hole_pt = (rx, ry + q) if axis == 0 else (rx + q, ry)
                datum_pt = anchor(name, axis, high, side, edge)
                dims.append(_make_linear(name, bb, axis, side, datum_pt, hole_pt, value))
                sides.append(side)

    # 2b. the outer-diameter callout (issue 29) in its view, on a leader angle not used there
    if od is not None and od["view"] in boxes:
        name = od["view"]
        bb = boxes[name]
        c, R = od["centre"], float(od["radius"])
        used = view_angles.get(name, [])
        best = None
        for ai, a in enumerate(LEADER_ANGLES):
            seg = _leader_segment(c, R, a, 0.5 * max(bb.width, bb.height))
            crossings = sum(1 for s in view_leaders.get(name, []) if _segments_cross(seg, s))
            reuse = sum(1 for ua, _ in used if abs(ua - a) < 1e-9)
            score = (crossings, reuse, ai)
            if best is None or score < best[0]:
                best = (score, a)
        value = 2.0 * R
        dims.append(Dimension(kind="diameter", text=f"{DIAMETER_SIGN}{format_value(value)}", value=value,
                              center=(float(c[0]), float(c[1])), radius=R, leader_angle_deg=best[1],
                              base=(float(c[0]), float(c[1])), view=name, count=1))
        sides.append(None)

    # 4. bands: per (view, side), shortest value nearest the outline
    bands: List[Optional[Tuple[str, int]]] = [None] * len(dims)
    keyed: Dict[Tuple[str, str], List[int]] = {}
    for i, (d, side) in enumerate(zip(dims, sides)):
        if side is not None:
            keyed.setdefault((d.view, side), []).append(i)
    for (_, side), idx in keyed.items():
        for band, i in enumerate(sorted(idx, key=lambda j: (dims[j].value, j))):
            bands[i] = (side, band)
    text_sides = _text_sides(dims, bands, boxes, style, scale_hint)
    log.debug("plan_dimensions: %d dims, %d undimensioned, %d flags", len(dims), len(undimensioned), len(flags))
    return DimensionPlan(dims=dims, undimensioned=undimensioned, bands=bands, flags=flags, stats=stats,
                         text_sides=text_sides)


def _text_sides(dims: List[Dimension], bands: List[Optional[Tuple[str, int]]], boxes: Dict[str, BBox2D],
                style: SheetStyle, scale_hint: Optional[float]) -> List[Optional[int]]:
    """Per linear dim the end (+1 p2, -1 p1) an outside value text goes past (QA round 2 issue 17).

    At the hinted / estimated scale, the text beyond each end spans
    ``arrow_length + SHORT_SPAN_EXTRA .. + text width`` sheet mm along the
    line. The end whose text interval crosses fewer extension lines of the
    dims further out on the same side of the view (their p1 / p2
    coordinates) wins, then the one that stays inside the view's extent
    along the line, then p2 (the hole end of a position dim).
    """
    out: List[Optional[int]] = [None] * len(dims)
    for i, (d, b) in enumerate(zip(dims, bands)):
        if d.kind != "linear" or b is None or d.view not in boxes:
            continue
        axis = 0 if abs(math.sin(math.radians(float(d.angle_deg)))) < 1e-6 else 1
        bb = boxes[d.view]
        s = _scale_estimate(bb, scale_hint)
        a1, a2 = float(d.p1[axis]), float(d.p2[axis])
        tw = TEXT_ADVANCE * style.text_height * len(d.text or "")
        if tw + 2.0 * style.dim_text_gap <= s * abs(a2 - a1):
            out[i] = 1                            # fits at the expected scale; +1 if it does not after all
            continue
        near = (style.arrow_length + SHORT_SPAN_EXTRA) / s
        far = near + tw / s
        lines = [float(p[axis]) for e, eb in zip(dims, bands)
                 if e is not d and e.kind == "linear" and eb is not None and e.view == d.view
                 and eb[0] == b[0] and eb[1] > b[1] for p in (e.p1, e.p2)]
        lo_e, hi_e = (bb.xmin, bb.xmax) if axis == 0 else (bb.ymin, bb.ymax)
        best = None
        for sgn, end, other in ((1, a2, a1), (-1, a1, a2)):
            u = 1.0 if end >= other else -1.0
            t0, t1 = sorted((end + u * near, end + u * far))
            cross = sum(1 for x in lines if t0 - 1e-9 <= x <= t1 + 1e-9)
            outside = max(0.0, lo_e - t0) + max(0.0, t1 - hi_e)
            score = (cross, round(outside, 9), 0 if sgn == 1 else 1)
            if best is None or score < best[0]:
                best = (score, sgn)
        out[i] = best[1]
    return out


def apply_text_side(ds: Dimension, side: Optional[int], style: SheetStyle) -> Dimension:
    """A sheet-space linear dim with the text placed past the p1 end when ``side`` is -1.

    For the layout, after ``_dim_to_sheet``: when ``ds.text_offset`` is 0,
    the text does not fit between the extension lines at this scale and
    ``side`` is -1, returns a copy with ``text_offset`` =
    ``-outside_text_offset(span)``; otherwise ``ds`` unchanged (the
    renderer's default is the p2 end).
    """
    if ds.kind != "linear" or side != -1 or float(ds.text_offset or 0.0) != 0.0:
        return ds
    off = outside_text_offset(_linear_frame(ds)[7], ds.text, style)
    if off == 0.0:
        return ds
    return dataclasses.replace(ds, text_offset=-off)


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


SHORT_SPAN_EXTRA = 2.0        # sheet mm the line runs on past an outside arrow (and the text gap beyond it)


def _arrows_outside(span: float, style: SheetStyle) -> bool:
    return span < 2.0 * style.arrow_length + 0.5


def outside_text_offset(span: float, text: str, style: SheetStyle) -> float:
    """Default ``text_offset`` (sheet mm, toward p2) of a linear dim of sheet length ``span``.

    0 when the value text fits between the extension lines (text width +
    2 x ``dim_text_gap`` <= span); otherwise the text goes outside, beyond
    the arrow at the p2 end, its near edge ``arrow_length +
    SHORT_SPAN_EXTRA`` past the p2 extension line, on the dimension line
    extended under it (QA round 2 issue 17: a centred text wider than the
    span was crossed by both extension lines). p2 is the hole end of a
    position dim and the high end of an overall dim. A layout that wants
    the p1 side sets ``Dimension.text_offset`` to the negated value; any
    non-zero ``text_offset`` wins over this default.
    """
    tw = TEXT_ADVANCE * style.text_height * len(text or "")
    if tw + 2.0 * style.dim_text_gap <= float(span):
        return 0.0
    return 0.5 * float(span) + style.arrow_length + SHORT_SPAN_EXTRA + 0.5 * tw


def linear_text_offset(dim: Dimension, style: SheetStyle) -> float:
    """The text shift along the line that ``render_dimension`` uses for a linear dim (sheet mm)."""
    shift = float(dim.text_offset or 0.0)
    if shift != 0.0:
        return shift
    span = _linear_frame(dim)[7]
    return outside_text_offset(span, dim.text, style)


def _line_ends(span: float, text: str, style: SheetStyle, shift: float) -> Tuple[float, float]:
    """Dimension line ends as offsets along p1 -> p2 from the p1 end (0 .. span when nothing sticks out)."""
    L = style.arrow_length
    if _arrows_outside(span, style):
        ext = L + SHORT_SPAN_EXTRA
        a_end, b_end = -ext, span + ext
    else:
        a_end, b_end = 0.0, span
    if shift != 0.0:      # a shifted text that leaves the line gets the line extended under it
        half_w = 0.5 * TEXT_ADVANCE * style.text_height * len(text or "")
        c = 0.5 * span + shift
        a_end = min(a_end, c - half_w)
        b_end = max(b_end, c + half_w)
    return a_end, b_end


def linear_reach(span: float, text: str, style: SheetStyle, text_offset: float = 0.0) -> Tuple[float, float]:
    """``(beyond_p1, beyond_p2)``: sheet mm a rendered linear dim of sheet length ``span``
    reaches past its p1 and its p2 measured point along the dimension direction.

    Outside arrows, the extended line and a text outside the span
    (``outside_text_offset``, or ``text_offset`` when non-zero) all count;
    the layout reserves this room (``layout._short_span_room``).
    """
    span = float(span)
    shift = float(text_offset or 0.0) or outside_text_offset(span, text, style)
    a_end, b_end = _line_ends(span, text, style, shift)
    return max(0.0, -a_end), max(0.0, b_end - span)


def linear_text_position(dim: Dimension, style: SheetStyle) -> Tuple[float, float]:
    """Anchor (middle of the baseline) of a linear dimension's value text, sheet mm.

    The centre of the dimension line, ``dim_text_gap`` off it on the
    reading side, shifted along the line (p1 -> p2) by ``dim.text_offset``,
    or, when that is 0 and the text does not fit, by
    ``outside_text_offset`` (the text beyond the p2 arrow).
    """
    _, _, _, _, _, q1, q2, _, u, _, up = _linear_frame(dim)
    mid = 0.5 * (q1 + q2) + linear_text_offset(dim, style) * u
    return _pt(mid + style.dim_text_gap * up)


def _render_linear(dim: Dimension, style: SheetStyle) -> List[SheetItem]:
    d, nrm, p1, p2, s, q1, q2, span, u, rot, up = _linear_frame(dim)
    items: List[SheetItem] = []
    for p, q in ((p1, q1), (p2, q2)):
        off = s - float(p @ nrm)
        e = nrm if off >= 0 else -nrm
        items.append(Segment(_pt(p + style.ext_gap * e), _pt(q + style.ext_overshoot * e), LineStyle.DIMENSION))
    L = style.arrow_length
    shift = linear_text_offset(dim, style)
    # line ends as offsets along u from q1 (0 .. span)
    if not _arrows_outside(span, style):
        arrows = (arrowhead(q1, -u, L, style.arrow_angle_deg), arrowhead(q2, u, L, style.arrow_angle_deg))
    else:     # no room: arrows outside pointing in, line extended past them
        arrows = (arrowhead(q1, u, L, style.arrow_angle_deg), arrowhead(q2, -u, L, style.arrow_angle_deg))
    a_end, b_end = _line_ends(span, dim.text, style, shift)
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
           "linear_text_offset", "linear_reach", "outside_text_offset", "apply_text_side",
           "DIAMETER_SIGN", "LEADER_LEN", "LEADER_ANGLES", "SIDES", "EXTENT_MISMATCH_TOL", "SHORT_SPAN_EXTRA"]
