"""Sheet layout: scale, view arrangement, frame, title block, strip, symbol, section marker.

``compose`` turns the projected views, the optional section and the
dimension plan into a ``Sheet`` of primitives in sheet millimetres (origin
bottom-left, Y up). Spec section 3.6; every number comes from
``style.SheetStyle`` (sourced in the standards probe, REPORT-standards.md).

Sourced rules (ISO texts read through their BIS adoptions):

* ISO 5457: frame 20 mm left / 10 mm elsewhere, 0.7 mm; centring marks at
  the four side midpoints from 5 mm outside the frame to 10 mm inside (a
  mark that would run into the title block keeps only its outside part).
  A4 landscape (``A4L``) is not allowed by 4.1 -> flag ``sheet_not_iso5457``.
* ISO 7200: title block 180 x 36 mm (4 rows of 9 mm) at the bottom right of
  the frame, columns at 0 | 32 | 74.2 | 129 | 180 mm, the bottom row's last
  column split into Rev 7 | Date of issue 25 | Lang 10 | Sheet 9. Scale,
  projection symbol and "Dimensions in mm" sit OUTSIDE the block, in a
  180 x 9 strip directly above it (ISO 7200 clause 4).
* ISO 5456-2: first angle = top view below the front view, right-side view
  on its LEFT; third angle mirrored (``projection.grid_cell``). Projection
  symbol at h = 3.5: circles Ø3.5 and Ø7, cone 7 long with ends 3.5 and 7,
  gap 1.05; the cone's small end points left in both; first angle = circles
  RIGHT of the cone, third angle = circles LEFT.
* ISO 5455 scales (``style.STANDARD_SCALES``); the largest one at which
  every view box plus its dimension bands plus the 20 mm view gaps fits the
  drawing space (5 mm clear of the frame) without touching the title block
  and strip (5 mm clear) wins. Nothing fits at 1:100 -> 1:100 is used and
  ``views_overflow`` is flagged. ``LayoutResult.flags`` starts with the
  plan's own flags (``DimensionPlan.flags``, e.g. ``dimension_extent_mismatch``).
* ISO 128-40: cutting line = wide long-dashed dotted (04.2) dashes of
  ``section_dash_len`` (12 mm) at both ends, from ``MARKER_GAP`` (1.5 mm)
  OUTSIDE the view outline to 13.5 mm (QA round 2 issue 13: they started
  3 mm inside and covered part edges); arrows = continuous wide shaft (CUTTING style,
  so it lands on the cutting-line layer, not among the model edges; the
  shaft is shorter than the first 12 mm dash and renders solid), filled 30
  degree head 5 mm long, ``section_arrow_len`` (10 mm) overall, pointing in
  the VIEWING direction of the section (= -plane normal: the section view
  looks from the +normal side toward the retained half); letter 5 mm beside
  each arrow; label ``A-A`` (5 mm) immediately above the section view.

Conventions of this module (no standard behind them):

* Clearance: views (with their bands) keep ``CLEARANCE`` = 5 mm from the
  frame and the title zone. REPORT-standards.md suggests 10 mm (itself a
  convention, not an ISO number); 5 mm is chosen on purpose because every
  item a view draws is reserved inside its placement rect (bands, callouts,
  labels, marker, short-span arrows), so the 5 mm is free paper on top of
  that, and the smaller value lets larger parts keep the bigger scale.
  (Measured: the synthetic plate gets the same scales with 10 mm, A3 1:1
  and A4 1:5.)
* Dimension bands: ``k`` bands on a side reserve ``dim_first_offset +
  k * dim_pitch`` mm (10 + 7k, spec 3.6); band ``i`` puts the dimension line
  at ``dim_first_offset + i * dim_pitch`` from the view outline.
* Extra room, added to the bands: the section view and the iso view
  reserve the label height above them. The marker view reserves, per
  scale, the exact reach of the rendered cutting-line marker beyond the
  outline (``_marker_room``, round 2; it replaced the fixed 12 mm along
  and 11 mm across). Diameter callouts reserve, per scale, the
  exact reach of the rendered callout beyond the outline
  (``_callout_room``), taken as the MAX with the bands, not added to them
  (QA round 1 issue 7: the old fixed reach from the circle, added to the
  bands, cost about 25 mm per side; the synthetic plate on A4 fell from
  1:5 to 1:10 once bands appeared on four sides).
* Text collisions (``_resolve_text_collisions``, after the dimensions are
  rendered): a linear dimension text crossed by the section marker (wide
  dash, arrows, letters) is shifted along its line
  (``Dimension.text_offset``, QA issue 4); a callout whose text overlaps
  another text (``TEXT_CLEARANCE`` = 1 mm), sits on the marker, or whose
  leader or shelf runs through another text or touches the marker (round 2
  issue 10) tries the four leader angles and
  longer leaders; the shortest leader that stays in its placement rect,
  off view outlines and off foreign dimension lines wins (QA issue 8).
  Texts that still overlap afterwards give the flag
  ``layout_text_collision`` (WARNING since round 2 issue 10: overprinted
  dimensions), and so do (round 3 issue 8) texts crossed by a line of
  another dimension and section-marker items touching a dimension line
  (``text_collisions(..., groups=...)``); texts the pass moved give
  ``layout_text_moved`` (info); ``sheet.meta["text_collisions"]`` counts the
  moves and the collisions left (``remaining``, ``text_overlaps``,
  ``text_line_crossings``, ``marker_dimension_crossings``).
* Scale-dependent room (``min_mm_fn`` of ``choose_scale``): a linear
  dimension whose sheet span is shorter than two arrows + 0.5 mm is drawn
  with the arrows outside and its line ``arrow_length + 2`` mm longer at
  both ends (``dimensions._render_linear``); a value text that does not
  fit between the extension lines is placed outside, beyond the arrow, by
  the layout itself (``_outside_high``: the end with no extension line of
  this or an outer band between the span and the far end of the text, then
  the end with more room inside the view; round 2 issue 17, round 3 issue
  24). That reach, the rendered section marker
  (``_marker_room``) and the rendered callouts (``_callout_room``) are
  reserved at each candidate scale.
* Section marker and dimension bands (round 2 issue 14): the marker never
  crosses a dimension line. With bands on the side the cutting line leaves
  through, the wide dash ends ``MARKER_BAND_CLEAR`` before the first
  dimension line, the arrow stays ``MARKER_ARROW_AT`` out and the letter
  goes beyond the arrow head, at both ends alike (round 3 issue 4); when
  that arrow or letter would touch an extension or dimension line, the end
  moves past the bands as below (round 3 issue 5, ``_marker_past_sides``)
  (round 1 moved the arrow past the bands and
  joined it with a narrow chain across every dimension and extension
  line). When an extension line of that side lies on the cutting line
  (the plane through a dimensioned hole centre), that end moves past the
  outermost dimension line, its overshoot and its text; flipping or
  re-datuming the dimension cannot help, because the extension line at the
  hole centre is the same on both sides and from both datums.
* Position-dimension bands (round 2 issue 7): when capping the position
  dims at ``POSITION_BAND_CAP`` (3) bands per view side wins a scale step,
  the excess ones (outermost first, never an overall dim) are dropped,
  listed in ``LayoutResult.plan.undimensioned`` with reason
  ``layout_space`` and flagged ``layout_dims_dropped`` (info).
* Iso view label: ``ISO (1:n)`` with the iso view's own scale (5 mm, spec
  3.6 "labelled with its scale") ``LABEL_GAP`` above the iso view.
* Grid: cells from ``projection.grid_cell`` (empty columns/rows are
  collapsed). The section view goes to the free middle-row cell on the side
  opposite the right-side view, else to the row below F, else above; the
  iso view to (1, 1), else the next free corner cell. When that does not
  fit a scale, the section tries the free cells of the front column (below
  / above the top-bottom stack, ``section_column_cells``) and then floats
  in free space like the iso view, before the scale drops (round 2 issue
  6: a thin top-profile section in the side cell doubled the grid width;
  measured on 80 box sizes x A3/A4, 78 of 160 cases get a larger scale and
  none a smaller one, choose_scale at most 117 ms). Inside one column the
  views whose 2D x axis is the same world direction share one x origin
  (front and top share X); inside one row the views with the same 2D y axis
  share one y origin (front, right, front-direction section share Z); other
  views are centred in their cell.
* Iso view: at each scale the grid WITH the iso cell is tried first; when
  that does not fit, the iso view (aligned to nothing) floats into free
  space at least ``view_gap`` clear of every grid view, top-right corner
  first (``info["iso_mode"]`` = "cell" | "free"): eight fixed positions,
  then a vectorised sweep of up to 61 x 61 positions (2.5 mm steps) over
  the drawing space, nearest the top-right corner first
  (``_float_spots``). Measured on the synthetic plate (100 x 60 x 15, iso,
  section): the iso row above F pushes the top view into the title block
  at 1:1, so without the float A3 would drop to 1:2; with it the plate is
  drawn at 1:1. When the floating iso view still finds no room, it may be
  drawn up to ``ISO_SCALE_STEPS`` (2) standard scales smaller than the
  sheet before the sheet scale drops (spec 3.6: the iso view is labelled
  with its scale). Measured (QA round 1 issue 7): with the eight fixed
  positions only the plate's iso view went to 1:2 on A3; with the sweep it
  is 1:1, and 45 of 700 synthetic box sizes on A3/A4 got a larger iso or
  sheet scale (choose_scale at most 43 ms).
* Placement of the whole grid: six preferred positions (centred, centred
  above the title zone, centred left of it, top-left, top-right,
  bottom-left), then a sweep of at most 41 x 41 positions ordered by
  distance from the centre; the first one where every view rectangle stays
  5 mm inside the frame, 5 mm clear of the title zone and 2 mm clear of the
  inner parts of the centring marks wins (measured: under 16 ms per
  ``compose`` on shapes from 5 x 3 x 1 to 60 x 40 x 20 m).
* ``--sheet auto``: the first of A4 portrait, A3 landscape on which the
  drawing fits at 1:1 or larger; otherwise A3 with its best reduction
  (overflow on A3 if nothing fits).
* Hidden lines are not drawn in the iso view; the section body shows its
  visible edges only. Hatch: one ``HatchLines`` per region (outer loop
  with its holes), ONE spacing per body (QA round 3 issue 1): ``hatch_spacing``
  of the section's shorter extent ON THE SHEET, or ``t / 3`` (at least
  0.7 mm) of its narrow regions when those hold more than 20 % of the body's
  area (``_body_hatch_spacing``); fill or hatch is decided per BODY
  (``SectionView.loop_body``) from its mean thickness ``t = 2A/P``: below
  2.1 mm the body is filled (keyhole ``Polygon``, holes stay open), so one
  wall is never half black, half hatched (round 2 issue 3). Body k is
  hatched at 45 / 135 degrees alternately, with a half-spacing phase
  shift every second pair (issue 28); lines are phase-locked to the sheet
  origin (``_section_items``).
* Strip: ``meta["unit_note"]`` replaces "Dimensions in mm" (round 2 issue
  30, e.g. "Dimensions in mm (unit assumed, see report)").
"""
from __future__ import annotations

import dataclasses
import datetime
import logging
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .dimensions import (LEADER_ANGLES, LEADER_LEN, SIDES, DimensionPlan, arrowhead, render_dimension)
from .geometry2d import hatch_lines, nest_loops
from .model import (TEXT_ADVANCE, BBox2D, Circle, Dimension, HatchLines, LineStyle, Point,
                    Polygon, Polyline, ProjectedView, SectionView, Segment, Sheet, SheetItem, Text,
                    transform_primitive)
from .options import DrawingOptions
from .projection import PRINCIPAL, VIEWS, ViewSpec, grid_cell
from .style import (HATCH_MIN_SEGMENTS, MIN_LETTERING_HEIGHT, SHEETS, STANDARD_SCALES, THICK_CORE_FRACTION,
                    SheetSize, SheetStyle, hatch_spacing, region_fill_below, scale_text,
                    section_fill_below, thin_region_spacing)

log = logging.getLogger(__name__)

SECTION_KEY = "section"
CLEARANCE = 5.0               # views stay this far from the frame and the title zone
MARK_CLEARANCE = 2.0          # views stay this far from the inner part of a centring mark
LEADER_EXIT_MARGIN = 3.0      # a callout leader ends this far outside the view outline
LABEL_GAP = 3.0               # A-A baseline above the section view
MARKER_GAP = 1.5              # the wide cutting dash starts this far OUTSIDE the outline (round 2 issue 13)
MARKER_INSIDE = -MARKER_GAP   # (round 1 name) negative: the dash no longer starts inside the outline
MARKER_ARROW_AT = 6.0         # arrow shaft this far outside the outline, along the line
MARKER_LETTER_AT = 4.0        # letter this far beyond the shaft, along the line
MARKER_LETTER_GAP = 1.0       # letter this far beyond the arrow tip (bands on that side)
MARKER_ALONG = MARKER_GAP + 12.0   # (informative) marker reach along the line without bands; see _marker_room
MARKER_PERP = 11.0            # (informative, round 1) room on the arrow side; _marker_room measures it
MARKER_BAND_CLEAR = 2.0       # wide dash ends this far before the first dimension line
SHORT_SPAN_EXTRA = 2.0        # dimensions._render_linear extends a short span's line by L + this
ISO_SCALE_STEPS = 2           # a floating iso view may drop this many standard scales below the sheet's
_SECTION_ALTERNATIVES = True  # QA round 2 issue 6: try front-column / floating section placements
POSITION_BAND_CAP: Optional[int] = 3   # QA round 2 issue 7: position-dim bands per side when it wins a scale
STRIP_COLS = (0.0, 40.0, 62.0, 112.0, 180.0)   # SCALE | symbol | Dimensions in mm | source
_EPS = 1e-9

Sides = Tuple[float, float, float, float]   # left, bottom, right, top (sheet mm)


@dataclass
class LayoutResult:
    sheet: Sheet
    scale: float
    scale_text: str
    sheet_size: SheetSize
    placements: Dict[str, Tuple[Point, BBox2D]]      # view -> (offset, sheet bbox incl. dimension bands)
    overflow: bool
    flags: List[Tuple[str, str, str]]                 # (code, severity, detail)
    # the dimension plan actually drawn: the input plan, or (QA round 2 issue 7) a copy with the
    # excess position dims removed and listed in undimensioned with reason "layout_space"
    plan: Optional[DimensionPlan] = None


# ------------------------------------------------------------------ bands --

def band_extent(style: SheetStyle, k: int) -> float:
    """Sheet mm reserved by ``k`` dimension bands on one side: 0, or 10 + 7k."""
    return 0.0 if k <= 0 else float(style.dim_first_offset + k * style.dim_pitch)


def _resolved_bands(plan: DimensionPlan) -> List[Optional[Tuple[str, int]]]:
    """``plan.bands``, or inferred ones (horizontal -> bottom, vertical -> left) when missing."""
    out: List[Optional[Tuple[str, int]]] = []
    seen: Dict[Tuple[str, str], int] = {}
    bands = list(plan.bands) if plan.bands else []
    for i, d in enumerate(plan.dims):
        b = bands[i] if i < len(bands) else None
        if d.kind != "linear":
            out.append(None)
            continue
        if b is None:
            side = "bottom" if abs(math.sin(math.radians(d.angle_deg))) < 0.5 else "left"
            k = seen.get((d.view, side), 0)
            seen[(d.view, side)] = k + 1
            b = (side, k)
        out.append((str(b[0]), int(b[1])))
    return out


def band_counts(plan: DimensionPlan) -> Dict[str, Tuple[int, int, int, int]]:
    """view -> number of dimension bands on (left, bottom, right, top)."""
    counts: Dict[str, List[int]] = {}
    for d, b in zip(plan.dims, _resolved_bands(plan)):
        if b is None:
            continue
        side, k = b
        c = counts.setdefault(d.view, [0, 0, 0, 0])
        i = SIDES.index(side)
        c[i] = max(c[i], k + 1)
    return {v: (c[0], c[1], c[2], c[3]) for v, c in counts.items()}


OVERALL_SPAN_FRACTION = 0.98   # review R4: an outermost-band dim spanning >= 98 % of the extent is overall


def _dim_role(d: Dimension) -> Optional[str]:
    """The planner's role marker of a dim (``Dimension.role``: "overall", "hole_position", ...), or None."""
    r = getattr(d, "role", None)
    return str(r) if r else None


def _is_overall(d: Dimension, boxes: Dict[str, BBox2D], outermost: bool = False) -> bool:
    """A linear dim that spans its view's whole extent along its direction (an overall dim).

    A role marker from the planner (``Dimension.role``) decides when set.
    Otherwise: the span (or value) matches the view bbox extent within
    max(0.01, 1e-3 ext), or (review R4, ``outermost``: the dim sits in the
    outermost band of its side) the span is at least OVERALL_SPAN_FRACTION
    of the extent; the planner takes overall values from the model extents,
    so a view bbox a few tenths of a mm larger must not demote it.
    """
    role = _dim_role(d)
    if role is not None:
        return role == "overall"
    bb = boxes.get(d.view)
    if bb is None or d.kind != "linear":
        return False
    a = math.radians(float(d.angle_deg))
    axis = 0 if abs(math.sin(a)) < 0.5 else 1
    ext = bb.width if axis == 0 else bb.height
    span = abs(float(d.p2[axis]) - float(d.p1[axis]))
    tol = max(0.01, 1e-3 * ext)
    if abs(span - ext) <= tol or float(d.value) >= ext - tol:
        return True
    return bool(outermost) and max(span, float(d.value)) >= OVERALL_SPAN_FRACTION * ext


def _cap_bands(plan: DimensionPlan, boxes: Dict[str, BBox2D], max_rows: int
               ) -> Tuple[DimensionPlan, List[dict]]:
    """``(plan with at most max_rows position-dim bands per view side, dropped entries)``.

    QA round 2 issue 7: deep position-dimension stacks (one band each) force
    the scale down. Per (view, side) the overall dims are always kept; of the
    position dims the ``max_rows`` nearest the outline (lowest band index =
    shortest values) are kept and the rest dropped, the largest band index
    first. The kept dims are renumbered 0..n-1 per side in their old order.
    Dropped dims come back as ``undimensioned`` entries with reason
    ``layout_space`` and kind ``Dimension.role`` (when the planner sets it)
    or ``"linear"``. Returns the input plan itself when nothing is dropped.
    """
    resolved = _resolved_bands(plan)
    per_side: Dict[Tuple[str, str], List[int]] = {}
    for i, (d, b) in enumerate(zip(plan.dims, resolved)):
        if b is not None:
            per_side.setdefault((d.view, b[0]), []).append(i)
    drop: set = set()
    for _key, idx in per_side.items():
        k_out = max(resolved[i][1] for i in idx)
        pos = sorted((i for i in idx if not _is_overall(plan.dims[i], boxes, resolved[i][1] == k_out)),
                     key=lambda i: (resolved[i][1], i))
        drop.update(pos[max(0, int(max_rows)):])
    if not drop:
        return plan, []
    dims: List[Dimension] = []
    bands: List[Optional[Tuple[str, int]]] = []
    for i, (d, b) in enumerate(zip(plan.dims, resolved)):
        if i not in drop:
            dims.append(d)
            bands.append(b)
    renum: Dict[Tuple[str, str], List[int]] = {}
    for j, (d, b) in enumerate(zip(dims, bands)):
        if b is not None:
            renum.setdefault((d.view, b[0]), []).append(j)
    for (_, side), idx in renum.items():
        for k, j in enumerate(sorted(idx, key=lambda j: (bands[j][1], j))):
            bands[j] = (side, k)
    # review R4: the entry kind is the planner's role marker when it sets one, else the generic
    # "linear" (the text says which dim); a fixed "hole_position" would mislabel step dims
    dropped = [{"kind": _dim_role(plan.dims[i]) or "linear", "view": plan.dims[i].view,
                "value": float(plan.dims[i].value),
                "text": plan.dims[i].text, "side": resolved[i][0], "count": 1, "radius": None,
                "reason": "layout_space"} for i in sorted(drop)]
    try:
        new = dataclasses.replace(plan, dims=dims, undimensioned=list(plan.undimensioned) + dropped, bands=bands)
    except Exception:                 # not a dataclass (duck-typed plan): the base fields only
        new = DimensionPlan(dims=dims, undimensioned=list(plan.undimensioned) + dropped, bands=bands,
                            flags=list(getattr(plan, "flags", None) or []))
    return new, dropped


# ------------------------------------------------------------------- grid --

def section_column_cells(view_names: Sequence[str], projection: str) -> List[Tuple[int, int]]:
    """Free cells of the front column for the section view: below / above the top-bottom stack.

    QA round 2 issue 6: a section seen as the top-view profile is as wide as
    the front view but only a few mm tall; in the side cell of the front row
    it doubles the grid width. In the front column it shares the front's x
    origin (same 2D x axis) and costs only its height plus one view gap.
    Order: below the lowest view of the column, above the highest, then the
    remaining free cells next to the front view.
    """
    taken = {grid_cell(n, projection) for n in view_names if n in PRINCIPAL}
    rows = sorted(r for c, r in taken if c == 0) or [0]
    out: List[Tuple[int, int]] = []
    for cand in ((0, rows[0] - 1), (0, rows[-1] + 1), (0, -1), (0, 1), (0, -2), (0, 2)):
        if cand not in taken and cand not in out:
            out.append(cand)
    return out


def grid_cells(view_names: Sequence[str], projection: str, has_section: bool = False,
               section_cell: Optional[Tuple[int, int]] = None) -> Dict[str, Tuple[int, int]]:
    """(col, row) per view relative to front at (0, 0); adds ``section`` and moves ``iso`` when taken.

    ``section_cell``: preferred cell of the section view (used when free),
    else the default order (side cell of the front row first).
    """
    cells: Dict[str, Tuple[int, int]] = {}
    for name in view_names:
        if name in PRINCIPAL:
            cells[name] = grid_cell(name, projection)
    taken = set(cells.values())
    if has_section:
        side = grid_cell("right", projection)[0]
        other = -side
        pref = [tuple(section_cell)] if section_cell is not None else []
        for cand in pref + [(other, 0), (side, 0), (other, -1), (side, -1), (other, 1), (side, 1),
                            (2 * other, 0), (2 * side, 0)]:
            if cand not in taken:
                cells[SECTION_KEY] = cand
                taken.add(cand)
                break
    if "iso" in view_names:
        first = grid_cell("iso", projection)
        for cand in (first, (-first[0], 1), (first[0], -1), (-first[0], -1), (2, 1), (-2, 1),
                     (2, -1), (-2, -1), (0, 2), (0, -2)):
            if cand not in taken:
                cells["iso"] = cand
                taken.add(cand)
                break
    return cells


def _default_spec(name: str) -> ViewSpec:
    return VIEWS["front"] if name == SECTION_KEY else VIEWS[name]


def _axis_key(v: np.ndarray) -> Tuple[float, ...]:
    return tuple(round(float(c), 9) + 0.0 for c in v)


@dataclass
class _Arrangement:
    origins: Dict[str, Tuple[float, float]]
    rects: Dict[str, BBox2D]
    width: float
    height: float


def _arrange(boxes: Dict[str, BBox2D], cells: Dict[str, Tuple[int, int]], specs: Dict[str, ViewSpec],
             sides: Dict[str, Sides], scale: float, gap: float) -> _Arrangement:
    """Origins (sheet offset of view coordinate 0) with the grid's bottom-left at (0, 0)."""
    rel: Dict[str, Tuple[float, float, float, float]] = {}
    for n in cells:
        bb = boxes[n]
        l, b, r, t = sides.get(n, (0.0, 0.0, 0.0, 0.0))
        rel[n] = (scale * bb.xmin - l, scale * bb.ymin - b, scale * bb.xmax + r, scale * bb.ymax + t)
    if not cells:
        return _Arrangement({}, {}, 0.0, 0.0)
    ox: Dict[str, float] = {}
    x = 0.0
    for c in sorted({cr[0] for cr in cells.values()}):
        groups: Dict[tuple, List[str]] = {}
        for n in cells:
            if cells[n][0] == c:
                groups.setdefault(_axis_key(specs[n].x), []).append(n)
        ext = {g: (min(rel[n][0] for n in ns), max(rel[n][2] for n in ns)) for g, ns in groups.items()}
        w = max(hi - lo for lo, hi in ext.values())
        for g, ns in groups.items():
            lo, hi = ext[g]
            for n in ns:
                ox[n] = x + 0.5 * (w - (hi - lo)) - lo
        x += w + gap
    width = x - gap
    oy: Dict[str, float] = {}
    y = 0.0                                   # top edge of the current row, going down
    for r in sorted({cr[1] for cr in cells.values()}, reverse=True):
        groups = {}
        for n in cells:
            if cells[n][1] == r:
                groups.setdefault(_axis_key(specs[n].y), []).append(n)
        ext = {g: (min(rel[n][1] for n in ns), max(rel[n][3] for n in ns)) for g, ns in groups.items()}
        h = max(hi - lo for lo, hi in ext.values())
        for g, ns in groups.items():
            lo, hi = ext[g]
            for n in ns:
                oy[n] = (y - h) + 0.5 * (h - (hi - lo)) - lo
        y -= h + gap
    height = -(y + gap)
    origins = {n: (ox[n], oy[n] + height) for n in cells}
    rects = {n: BBox2D(origins[n][0] + rel[n][0], origins[n][1] + rel[n][1],
                       origins[n][0] + rel[n][2], origins[n][1] + rel[n][3]) for n in cells}
    return _Arrangement(origins, rects, width, height)


def _frame_rect(sheet: SheetSize, style: SheetStyle) -> BBox2D:
    return BBox2D(style.margin_left, style.margin, sheet.width - style.margin, sheet.height - style.margin)


def _title_zone(sheet: SheetSize, style: SheetStyle) -> BBox2D:
    f = _frame_rect(sheet, style)
    return BBox2D(f.xmax - style.title_block_w, f.ymin, f.xmax, f.ymin + style.title_block_h + style.strip_h)


def _obstacles(sheet: SheetSize, style: SheetStyle) -> List[BBox2D]:
    """Areas views must not touch: the title zone and the inner parts of the centring marks."""
    f = _frame_rect(sheet, style)
    tz = _title_zone(sheet, style)
    i = style.centring_mark_inside
    xm, ym = sheet.width / 2.0, sheet.height / 2.0
    w = MARK_CLEARANCE
    obs = [tz.expanded(CLEARANCE),
           BBox2D(xm - w, f.ymax - i - w, xm + w, f.ymax),
           BBox2D(f.xmin, ym - w, f.xmin + i + w, ym + w),
           BBox2D(f.xmax - i - w, ym - w, f.xmax, ym + w)]
    if not (tz.xmin - _EPS <= xm <= tz.xmax + _EPS):
        obs.append(BBox2D(xm - w, f.ymin, xm + w, f.ymin + i + w))
    return obs


def _overlaps(a: BBox2D, b: BBox2D) -> bool:
    return a.xmin < b.xmax - _EPS and a.xmax > b.xmin + _EPS and a.ymin < b.ymax - _EPS and a.ymax > b.ymin + _EPS


def _inside(a: BBox2D, b: BBox2D) -> bool:
    return a.xmin >= b.xmin - _EPS and a.ymin >= b.ymin - _EPS and a.xmax <= b.xmax + _EPS and a.ymax <= b.ymax + _EPS


def _shift(r: BBox2D, dx: float, dy: float) -> BBox2D:
    return BBox2D(r.xmin + dx, r.ymin + dy, r.xmax + dx, r.ymax + dy)


_SWEEP_STEPS = 40          # grid-position sweep resolution per axis (at least 2.5 mm per step)
_ALT_SWEEP_STEPS = 12      # coarser sweep for the alternative section placements (round 2 issue 6)


def _grid_candidates(W: float, H: float, inner: BBox2D, tz: BBox2D,
                     steps: int = _SWEEP_STEPS) -> List[Tuple[float, float]]:
    """Grid bottom-left positions: six preferred ones, then a sweep ordered by distance from the centre."""
    cx = inner.xmin + 0.5 * (inner.width - W)
    cy = inner.ymin + 0.5 * (inner.height - H)
    cands = [(cx, cy),
             (cx, tz.ymax + 0.5 * (inner.ymax - tz.ymax - H)),
             (inner.xmin + 0.5 * (tz.xmin - inner.xmin - W), cy),
             (inner.xmin, inner.ymax - H),
             (inner.xmax - W, inner.ymax - H),
             (inner.xmin, inner.ymin)]
    xr, yr = inner.width - W, inner.height - H
    nx = int(min(steps, max(1, math.floor(xr / 2.5)))) if xr > _EPS else 0
    ny = int(min(steps, max(1, math.floor(yr / 2.5)))) if yr > _EPS else 0
    xs = [inner.xmin + xr * i / nx for i in range(nx + 1)] if nx else [inner.xmin]
    ys = [inner.ymin + yr * j / ny for j in range(ny + 1)] if ny else [inner.ymin]
    sweep = [(x, y) for x in xs for y in ys]
    sweep.sort(key=lambda p: (round((p[0] - cx) ** 2 + (p[1] - cy) ** 2, 9), -p[1], p[0]))
    return cands + sweep


def _free_candidates(w: float, h: float, grid: BBox2D, inner: BBox2D, tz: BBox2D,
                     gap: float) -> List[Tuple[float, float]]:
    """Lower-left corners to try for a floating view (iso), top-right first."""
    return [(inner.xmax - w, inner.ymax - h),
            (grid.xmax + gap, grid.ymax - h),
            (grid.xmax - w, grid.ymax + gap),
            (inner.xmax - w, tz.ymax),
            (inner.xmin, inner.ymax - h),
            (grid.xmin - gap - w, grid.ymax - h),
            (grid.xmin, grid.ymin - gap - h),
            (inner.xmin, inner.ymin)]


def _place(arr: _Arrangement, sheet: SheetSize, style: SheetStyle,
           float_size: Optional[Tuple[float, float]] = None,
           float_sizes: Optional[Sequence[Tuple[float, float]]] = None,
           sweep_steps: int = _SWEEP_STEPS):
    """``(dx, dy, float_ll)`` for the grid's bottom-left corner, or None when it does not fit.

    With ``float_size = (w, h)`` a floating rectangle (the iso view) must be
    placed too, at least ``view_gap`` clear of every grid rectangle;
    ``float_ll`` is its lower-left corner (None without a floating view).
    With ``float_sizes`` (several floating rectangles, e.g. the iso view and
    the section, QA round 2 issue 6) they are placed one after the other,
    each ``view_gap`` clear of the grid and of the ones placed before, and
    ``float_ll`` is the list of their lower-left corners.
    """
    many = float_sizes is not None
    sizes = list(float_sizes) if many else ([float_size] if float_size is not None else [])
    inner = _frame_rect(sheet, style).expanded(-CLEARANCE)
    tz = _title_zone(sheet, style).expanded(CLEARANCE)
    obstacles = _obstacles(sheet, style)
    W, H = arr.width, arr.height
    if W > inner.width + _EPS or H > inner.height + _EPS:
        return None
    if any(w > inner.width + _EPS or h > inner.height + _EPS for w, h in sizes):
        return None
    rects = list(arr.rects.values())
    gap = style.view_gap

    def free(r: BBox2D) -> bool:
        return _inside(r, inner) and not any(_overlaps(r, o) for o in obstacles)

    spots = [_float_spots(w, h, inner, obstacles) for w, h in sizes]
    for dx, dy in _grid_candidates(W, H, inner, tz, sweep_steps):
        shifted = [_shift(r, dx, dy) for r in rects]
        if not all(free(s) for s in shifted):
            continue
        if not sizes:
            return dx, dy, ([] if many else None)
        grid = BBox2D(dx, dy, dx + W, dy + H)
        placed: List[BBox2D] = []
        lls: List[Tuple[float, float]] = []
        for (w, h), sp in zip(sizes, spots):
            hit = None
            for fx, fy in _free_candidates(w, h, grid, inner, tz, gap):
                fr = BBox2D(fx, fy, fx + w, fy + h)
                if not free(fr):
                    continue
                if any(_overlaps(fr.expanded(gap - 1e-6), s) for s in shifted + placed):
                    continue
                hit = (fx, fy)
                break
            if hit is None:
                hit = _first_clear_spot(sp, w, h, shifted + placed, gap - 1e-6)
            if hit is None:
                break
            lls.append(hit)
            placed.append(BBox2D(hit[0], hit[1], hit[0] + w, hit[1] + h))
        if len(lls) == len(sizes):
            return dx, dy, (lls if many else lls[0])
    return None


_FLOAT_STEP = 2.5           # sheet mm between the floating-view sweep positions
_FLOAT_MAX_STEPS = 60       # at most 61 x 61 sweep positions


def _float_spots(w: float, h: float, inner: BBox2D, obstacles: List[BBox2D]):
    """(xs, ys) lower-left corners of a floating ``w x h`` view that stay inside
    ``inner`` and off the obstacles, nearest the top-right corner first.

    QA round 1 issue 7: the eight fixed corner positions of
    ``_free_candidates`` missed free space, so the iso view dropped up to two
    scales on a mostly empty sheet. The sweep is vectorised (numpy), so one
    test per grid position costs one pass over the spots.
    """
    xr, yr = inner.width - w, inner.height - h
    if xr < -_EPS or yr < -_EPS:
        return None
    nx = int(min(_FLOAT_MAX_STEPS, max(1, math.floor(xr / _FLOAT_STEP)))) if xr > _EPS else 0
    ny = int(min(_FLOAT_MAX_STEPS, max(1, math.floor(yr / _FLOAT_STEP)))) if yr > _EPS else 0
    xs = np.linspace(inner.xmin, inner.xmin + max(xr, 0.0), nx + 1)
    ys = np.linspace(inner.ymin, inner.ymin + max(yr, 0.0), ny + 1)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    X, Y = X.ravel(), Y.ravel()
    ok = np.ones(X.shape, dtype=bool)
    for o in obstacles:
        ok &= ~((X < o.xmax - _EPS) & (X + w > o.xmin + _EPS) & (Y < o.ymax - _EPS) & (Y + h > o.ymin + _EPS))
    X, Y = X[ok], Y[ok]
    if X.size == 0:
        return None
    d2 = (X - (inner.xmax - w)) ** 2 + (Y - (inner.ymax - h)) ** 2
    order = np.lexsort((X, -Y, np.round(d2, 9)))
    return X[order], Y[order]


def _first_clear_spot(spots, w: float, h: float, rects: List[BBox2D], gap: float):
    """First spot whose ``w x h`` rectangle keeps ``gap`` from every rect, or None."""
    if spots is None:
        return None
    X, Y = spots
    ok = np.ones(X.shape, dtype=bool)
    for r in rects:
        ok &= ~((X - gap < r.xmax - _EPS) & (X + w + gap > r.xmin + _EPS)
                & (Y - gap < r.ymax - _EPS) & (Y + h + gap > r.ymin + _EPS))
    idx = np.flatnonzero(ok)
    if idx.size == 0:
        return None
    k = int(idx[0])
    return float(X[k]), float(Y[k])


def _estimate_iso_box(boxes: Dict[str, BBox2D]) -> Optional[BBox2D]:
    def pick(*cands):
        for name, attr in cands:
            b = boxes.get(name)
            if b is not None:
                return getattr(b, attr)
        return None
    ex = pick(("front", "width"), ("top", "width"), ("back", "width"), ("bottom", "width"))
    ey = pick(("top", "height"), ("right", "width"), ("bottom", "height"), ("left", "width"))
    ez = pick(("front", "height"), ("right", "height"), ("back", "height"), ("left", "height"))
    known = [v for v in (ex, ey, ez) if v is not None]
    if not known:
        return None
    fill = max(known)
    ex, ey, ez = (fill if v is None else v for v in (ex, ey, ez))
    corners = np.array([[x, y, z] for x in (0.0, ex) for y in (0.0, ey) for z in (0.0, ez)])
    return BBox2D.from_points(VIEWS["iso"].to_view(corners)[:, :2])


def choose_scale(view_boxes: Dict[str, BBox2D], bands: Dict[str, Tuple[int, int, int, int]], sheet: SheetSize,
                 style: SheetStyle, projection: str, has_section: bool, has_iso: bool, *,
                 specs: Optional[Dict[str, ViewSpec]] = None,
                 extra_mm: Optional[Dict[str, Sides]] = None,
                 min_mm_fn: Optional[Callable[[float], Dict[str, Sides]]] = None
                 ) -> Tuple[Optional[float], dict]:
    """Largest ISO 5455 scale at which the arrangement fits ``sheet``.

    ``view_boxes``: view-coordinate boxes of the principal views, plus
    optionally ``iso`` and ``section`` (missing ones are estimated: iso from
    the principal extents, section = the front box). ``bands``: dimension
    band counts (left, bottom, right, top) per view. ``extra_mm``: fixed
    sheet-mm room added on each side (callouts, labels, section marker).
    ``min_mm_fn(scale)``: per view the minimum sheet-mm room on each side at
    that scale (short-span dimension arrows); a side gets the larger of
    bands + extra and this minimum.
    Returns ``(scale, info)``; ``scale`` is None when nothing fits even at
    the smallest scale, and then ``info["overflow"]`` is True and
    ``info["scale"]`` the smallest scale. ``info`` also carries ``cells``,
    ``origins``, ``rects`` (sheet mm, bands included), ``tried``,
    ``iso_mode`` and ``iso_scale`` (the scale the iso view is drawn at; it
    equals ``scale`` unless the floating iso view had to drop up to
    ``ISO_SCALE_STEPS`` standard scales; None without an iso view).
    QA round 2 issue 6: at every scale the section view tries the default
    cell, then the free cells of the front column (``section_column_cells``),
    then floating in free space like the iso view, before the scale drops;
    ``info["section_mode"]`` = "cell" | "free" | None and
    ``info["scale_candidates"]`` = ``[(scale, fitted, detail)]`` per scale
    tried (detail: the placement that fitted, or why the closest one failed).
    """
    boxes: Dict[str, BBox2D] = {n: b for n, b in view_boxes.items()
                                if b is not None and (n in PRINCIPAL or n in ("iso", SECTION_KEY))}
    if has_iso and "iso" not in boxes:
        est = _estimate_iso_box(boxes)
        if est is not None:
            boxes["iso"] = est
    if not has_iso:
        boxes.pop("iso", None)
    if has_section and SECTION_KEY not in boxes:
        est = boxes.get("front") or next((boxes[n] for n in PRINCIPAL if n in boxes), None)
        if est is not None:
            boxes[SECTION_KEY] = est
    if not has_section:
        boxes.pop(SECTION_KEY, None)
    names = [n for n in boxes if n != SECTION_KEY]
    has_sec = SECTION_KEY in boxes
    has_iso_box = "iso" in boxes
    cells = grid_cells(names, projection, has_section=has_sec)
    # section placements, tried in this order at every scale (QA round 2 issue 6): the default
    # cell, the free cells of the front column, then floating like the iso view
    variants: List[Tuple[str, Dict[str, Tuple[int, int]], bool]] = [("cell", cells, False)]
    if has_sec and _SECTION_ALTERNATIVES:
        for cc in section_column_cells(names, projection):
            if cc == cells.get(SECTION_KEY):
                continue
            cv = grid_cells(names, projection, has_section=True, section_cell=cc)
            if cv.get(SECTION_KEY) == cc:
                variants.append(("cell", cv, False))
        variants.append(("free", grid_cells(names, projection, has_section=False), True))
    spec_map = {n: ((specs or {}).get(n) or _default_spec(n)) for n in boxes}
    extra = extra_mm or {}
    base_sides: Dict[str, Sides] = {}
    for n in boxes:
        k = bands.get(n, (0, 0, 0, 0))
        e = extra.get(n, (0.0, 0.0, 0.0, 0.0))
        base_sides[n] = tuple(band_extent(style, int(k[i])) + float(e[i]) for i in range(4))   # type: ignore[assignment]
    sides: Dict[str, Sides] = dict(base_sides)
    info: dict = {"cells": cells, "tried": [], "overflow": False, "sheet": sheet.name, "iso_mode": None,
                  "section_mode": None, "scale_candidates": []}
    if not cells:
        info.update(scale=1.0, origins={}, rects={}, grid=(0.0, 0.0))
        return 1.0, info
    inner = _frame_rect(sheet, style).expanded(-CLEARANCE)
    mins_cache: Dict[float, Dict[str, Sides]] = {}

    def mins_at(s: float) -> Dict[str, Sides]:
        # min_mm_fn renders callouts / the marker: evaluate it once per scale
        key = float(s)
        if key not in mins_cache:
            mins_cache[key] = min_mm_fn(key) if min_mm_fn is not None else {}
        return mins_cache[key]

    def set_sides(s: float) -> None:
        mins = mins_at(s)
        for n, bs in base_sides.items():
            m = mins.get(n)
            sides[n] = bs if m is None else tuple(max(float(bs[i]), float(m[i])) for i in range(4))  # type: ignore[assignment]

    def own_rel(name: str, s: float) -> Tuple[float, float, float, float]:
        # room of a floating view at its own scale (the iso view may differ from the sheet scale)
        bb = boxes[name]
        bs = base_sides[name]
        m = mins_at(s).get(name)
        l, b, r, t = bs if m is None else tuple(max(float(bs[i]), float(m[i])) for i in range(4))
        return (s * bb.xmin - l, s * bb.ymin - b, s * bb.xmax + r, s * bb.ymax + t)

    def try_one(var, s: float, iso_s: float, iso_float: bool):
        _, cv, sec_float = var
        use = {n: c for n, c in cv.items() if not (iso_float and n == "iso")}
        arr = _arrange(boxes, use, spec_map, sides, float(s), style.view_gap)
        floats: List[Tuple[str, Tuple[float, float, float, float]]] = []
        if iso_float:
            floats.append(("iso", own_rel("iso", float(iso_s))))
        if sec_float:
            floats.append((SECTION_KEY, own_rel(SECTION_KEY, float(s))))
        # the alternative section placements sweep a coarser grid (measured: the full 41 x 41 sweep
        # with two floating views took up to 0.53 s per choose_scale on extreme boxes)
        steps = _SWEEP_STEPS if var is variants[0] else _ALT_SWEEP_STEPS
        if floats:
            pos = _place(arr, sheet, style, float_sizes=[(r[2] - r[0], r[3] - r[1]) for _, r in floats],
                         sweep_steps=steps)
        else:
            pos = _place(arr, sheet, style, sweep_steps=steps)
            if pos is not None:
                pos = (pos[0], pos[1], [])
        return arr, use, floats, pos

    def accept(s, arr, use, floats, pos, var, iso_float, iso_s):
        dx, dy, lls = pos
        origins = {n: (o[0] + dx, o[1] + dy) for n, o in arr.origins.items()}
        rects = {n: _shift(r, dx, dy) for n, r in arr.rects.items()}
        for (name, rl), ll in zip(floats, lls):
            origins[name] = (ll[0] - rl[0], ll[1] - rl[1])
            rects[name] = BBox2D(ll[0], ll[1], ll[0] + rl[2] - rl[0], ll[1] + rl[3] - rl[1])
        iso_mode = "free" if iso_float else ("cell" if "iso" in use else None)
        sec_mode = None if not has_sec else ("free" if var[2] else "cell")
        info.update(scale=s, grid=(arr.width, arr.height), origins=origins, rects=rects, iso_mode=iso_mode,
                    section_mode=sec_mode, cells=use, iso_scale=(iso_s if "iso" in origins else None))
        desc = []
        if has_sec:
            desc.append(f"section {sec_mode}" + (f" {use[SECTION_KEY]}" if SECTION_KEY in use else ""))
        if "iso" in origins:
            desc.append(f"iso {iso_mode} {scale_text(iso_s)}")
        return ", ".join(desc) or "grid"

    for idx, s in enumerate(STANDARD_SCALES):
        set_sides(s)
        order = []
        for var in variants:
            if not has_iso_box or "iso" in var[1]:
                order.append((var, s, False))
            if has_iso_box:
                # the iso view does not fit its grid cell: float it into free space, at the
                # sheet scale or up to ISO_SCALE_STEPS standard scales smaller (labelled)
                order.append((var, s, True))
        if has_iso_box:
            for iso_s in STANDARD_SCALES[idx + 1: idx + 1 + ISO_SCALE_STEPS]:
                order.extend((var, iso_s, True) for var in variants)
        worst = None
        for var, iso_s, iso_float in order:
            arr, use, floats, pos = try_one(var, s, iso_s, iso_float)
            if pos is not None:
                info["tried"].append((s, True))
                desc = accept(s, arr, use, floats, pos, var, iso_float, iso_s)
                info["scale_candidates"].append((s, True, desc))
                return s, info
            over = max(arr.width - inner.width, arr.height - inner.height)
            if worst is None or over < worst[0]:
                worst = (over, arr.width, arr.height)
        info["tried"].append((s, False))
        if worst is not None and worst[0] > _EPS:
            reason = (f"smallest grid {worst[1]:.1f} x {worst[2]:.1f} mm exceeds the drawing space "
                      f"{inner.width:.1f} x {inner.height:.1f} mm")
        else:
            reason = "no position clear of the title block, the centring marks and the floating views"
        info["scale_candidates"].append((s, False, reason))
    s = STANDARD_SCALES[-1]
    set_sides(s)
    use = cells
    arr = _arrange(boxes, use, spec_map, sides, float(s), style.view_gap)
    accept(s, arr, use, [], (inner.xmin, inner.ymax - arr.height, []), variants[0], False, s)
    info["overflow"] = True
    log.warning("layout: views do not fit %s even at %s", sheet.name, scale_text(s))
    return None, info


# --------------------------------------------------------------- helpers --

def _view_bbox(v: ProjectedView) -> Optional[BBox2D]:
    if v.bbox is not None:
        return v.bbox
    return BBox2D.of_primitives(list(v.visible) + list(v.hidden))


def _section_bbox(section: SectionView) -> Optional[BBox2D]:
    bb = None
    pts = [np.asarray(lp, dtype=float).reshape(-1, 2) for lp in section.loops if np.size(lp)]
    if pts:
        bb = BBox2D.from_points(np.vstack(pts))
    if section.body is not None:
        b2 = _view_bbox(section.body)
        if b2 is not None:
            bb = b2 if bb is None else bb.union(b2)
    return bb


def _restyle(item, style: LineStyle):
    return dataclasses.replace(item, style=style)


def _geom_box(bb: BBox2D, origin: Point, scale: float) -> BBox2D:
    return BBox2D(origin[0] + scale * bb.xmin, origin[1] + scale * bb.ymin,
                  origin[0] + scale * bb.xmax, origin[1] + scale * bb.ymax)


def _clip_line(p: np.ndarray, d: np.ndarray, box: BBox2D):
    """Infinite line p + t d against a box -> (t0, side0, t1, side1) or None."""
    t0, t1 = -math.inf, math.inf
    s0 = s1 = None
    for axis, lo, hi, lo_name, hi_name in ((0, box.xmin, box.xmax, "left", "right"),
                                           (1, box.ymin, box.ymax, "bottom", "top")):
        if abs(d[axis]) < 1e-12:
            if p[axis] < lo - 1e-9 or p[axis] > hi + 1e-9:
                return None
            continue
        ta, tb = (lo - p[axis]) / d[axis], (hi - p[axis]) / d[axis]
        na, nb = lo_name, hi_name
        if ta > tb:
            ta, tb, na, nb = tb, ta, nb, na
        if ta > t0:
            t0, s0 = ta, na
        if tb < t1:
            t1, s1 = tb, nb
    if s0 is None or s1 is None or t0 > t1 + 1e-12:
        return None
    return t0, s0, t1, s1


def _ray_exit(p: np.ndarray, u: np.ndarray, box: BBox2D) -> float:
    ts = []
    for axis, lo, hi in ((0, box.xmin, box.xmax), (1, box.ymin, box.ymax)):
        if abs(u[axis]) > 1e-12:
            ts.append(((hi if u[axis] > 0 else lo) - p[axis]) / u[axis])
    return max(0.0, min(ts)) if ts else 0.0


def _dominant_side(v: np.ndarray) -> str:
    if abs(v[0]) >= abs(v[1]):
        return "right" if v[0] > 0 else "left"
    return "top" if v[1] > 0 else "bottom"


def _marker_geometry(section: SectionView, views: Dict[str, ProjectedView], boxes: Dict[str, BBox2D]):
    """(view name, point2d, line dir2d, view dir2d, clip) of the cutting line, or None."""
    n = np.asarray(section.plane_normal, dtype=float).reshape(-1)
    if n.shape != (3,) or not np.all(np.isfinite(n)) or np.linalg.norm(n) < 1e-12:
        return None
    n = n / np.linalg.norm(n)
    order = [section.marker_view] + [v for v in ("top", "front", "right", "bottom", "back", "left")
                                     if v != section.marker_view]
    for name in order:
        if name not in views or name not in PRINCIPAL or boxes.get(name) is None:
            continue
        spec = views[name].spec
        if abs(float(np.dot(spec.n, n))) > 1e-6:
            continue                              # plane not seen edge-on in this view
        d3 = np.cross(n, spec.n)
        d2 = np.array([float(d3 @ spec.x), float(d3 @ spec.y)])
        a2 = np.array([float(-n @ spec.x), float(-n @ spec.y)])
        if np.linalg.norm(d2) < 1e-9 or np.linalg.norm(a2) < 1e-9:
            continue
        d2 /= np.linalg.norm(d2)
        a2 /= np.linalg.norm(a2)
        p2 = spec.to_view(np.asarray(section.plane_point, dtype=float))[0, :2]
        clip = _clip_line(p2, d2, boxes[name])
        if clip is None:
            log.info("layout: the section plane misses view %s; no cutting-line marker", name)
            return None
        if name != section.marker_view:
            log.info("layout: cutting-line marker moved from %s to %s", section.marker_view, name)
        return name, p2, d2, a2, clip
    log.info("layout: no drawn principal view shows the section plane edge-on; no marker")
    return None


def _fit_text(text: str, width: float, height: float, min_height: float = 1.8) -> Tuple[str, float]:
    """Shrink (then truncate) ``text`` so its estimated width fits ``width``."""
    if not text:
        return text, height
    h = height
    est = TEXT_ADVANCE * h * len(text)
    if est > width:
        h = max(min_height, width / (TEXT_ADVANCE * len(text)))
    while text and TEXT_ADVANCE * h * len(text) > width + 1e-9:
        text = text[:-2] + "~" if len(text) > 2 else text[:-1]
    return text, h


# ------------------------------------------------------------ sheet parts --

def frame_items(sheet: SheetSize, style: SheetStyle) -> List[SheetItem]:
    """Frame (4 FRAME segments) and the ISO 5457 centring marks."""
    f = _frame_rect(sheet, style)
    items: List[SheetItem] = [
        Segment((f.xmin, f.ymin), (f.xmax, f.ymin), LineStyle.FRAME),
        Segment((f.xmax, f.ymin), (f.xmax, f.ymax), LineStyle.FRAME),
        Segment((f.xmax, f.ymax), (f.xmin, f.ymax), LineStyle.FRAME),
        Segment((f.xmin, f.ymax), (f.xmin, f.ymin), LineStyle.FRAME),
    ]
    tz = _title_zone(sheet, style)
    o, i = style.centring_mark_outside, style.centring_mark_inside
    xm, ym = sheet.width / 2.0, sheet.height / 2.0
    bottom_in = f.ymin if tz.xmin - _EPS <= xm <= tz.xmax + _EPS else f.ymin + i
    marks = [((xm, f.ymin - o), (xm, bottom_in)), ((xm, f.ymax + o), (xm, f.ymax - i)),
             ((f.xmin - o, ym), (f.xmin + i, ym)), ((f.xmax + o, ym), (f.xmax - i, ym))]
    for a, b in marks:
        items.append(Segment(a, b, LineStyle.FRAME))
    return items


def projection_symbol(center: Point, h: float, projection: str) -> List[SheetItem]:
    """ISO 5456-2 symbol: 2 concentric circles (Ø h, Ø 2h) + closed 4-point cone (small end left)."""
    H = 2.0 * h
    gap = 0.3 * h                                 # 3 d with d = h / 10
    total = 2.0 * H + gap
    xl = float(center[0]) - total / 2.0
    cy = float(center[1])
    if projection in ("first", "first_angle"):
        cone_x, circ_x = xl, xl + H + gap + H / 2.0
    else:
        circ_x, cone_x = xl + H / 2.0, xl + H + gap
    cone = np.array([[cone_x, cy - h / 2.0], [cone_x + H, cy - H / 2.0],
                     [cone_x + H, cy + H / 2.0], [cone_x, cy + h / 2.0]])
    return [Polyline(cone, True, LineStyle.TITLE),
            Circle((circ_x, cy), h / 2.0, LineStyle.TITLE),
            Circle((circ_x, cy), H / 2.0, LineStyle.TITLE)]


# (row from the top 0..3, x0, x1, label, field key)
_TITLE_CELLS = (
    (0, 0.0, 32.0, "Resp. dept.", "department"),
    (0, 32.0, 74.2, "Technical reference", "technical_reference"),
    (0, 74.2, 129.0, "Document type", "document_type"),
    (0, 129.0, 180.0, "Document status", "document_status"),
    (1, 0.0, 32.0, "Created by", "creator"),
    (1, 32.0, 74.2, "Approved by", "approval"),
    (1, 74.2, 129.0, "Title", "title"),
    (1, 129.0, 180.0, "Identification number", "ident"),
    (2, 0.0, 32.0, "Size", "size"),
    (2, 32.0, 74.2, "Legal owner", "owner"),
    (2, 74.2, 129.0, "Supplementary title", "supplementary_title"),
    (2, 129.0, 180.0, "Input file", "input_file"),
    (3, 0.0, 32.0, "Classification", "classification"),
    (3, 32.0, 74.2, "Generator", "generator"),
    (3, 74.2, 129.0, "Keywords", "keywords"),
    (3, 129.0, 136.0, "Rev", "revision"),
    (3, 136.0, 161.0, "Date of issue", "date"),
    (3, 161.0, 171.0, "Lang", "language"),
    (3, 171.0, 180.0, "Sheet", "sheet_number"),
)
_BIG_FIELDS = ("title", "ident")


def title_block_items(sheet: SheetSize, style: SheetStyle, fields: Dict[str, str]) -> List[SheetItem]:
    """ISO 7200 block (180 x 36, 4 rows) at the frame's bottom-right corner."""
    f = _frame_rect(sheet, style)
    x0, y0 = f.xmax - style.title_block_w, f.ymin
    rh = style.title_row_h
    top = y0 + style.title_block_h
    items: List[SheetItem] = [Segment((x0, top), (f.xmax, top), LineStyle.FRAME),
                              Segment((x0, y0), (x0, top), LineStyle.FRAME)]
    rows = int(round(style.title_block_h / rh))
    for k in range(1, rows):
        items.append(Segment((x0, y0 + k * rh), (f.xmax, y0 + k * rh), LineStyle.TITLE))
    for row in range(rows):
        ytop = top - row * rh
        bounds = sorted({c[1] for c in _TITLE_CELLS if c[0] == row} - {0.0})
        for bx in bounds:
            items.append(Segment((x0 + bx, ytop - rh), (x0 + bx, ytop), LineStyle.TITLE))
    for row, cx0, cx1, label, key in _TITLE_CELLS:
        ytop = top - row * rh
        w = cx1 - cx0 - 2.0
        big = key in _BIG_FIELDS
        lh = 1.8 if big else style.title_label_height
        ltxt, lh = _fit_text(label, w, lh)
        items.append(Text((x0 + cx0 + 1.0, ytop - 0.6 - lh), ltxt, lh, anchor="start"))
        value = str(fields.get(key, "") or "")
        if value:
            vh = style.label_height if big else style.text_height
            vh = min(vh, rh - lh - 2.0) if not big else vh
            vtxt, vh = _fit_text(value, w, vh)
            items.append(Text((x0 + cx0 + 1.0, ytop - rh + 1.4), vtxt, vh, anchor="start"))
    return items


NOTE_LINE_GAP = 1.0           # strip: gap between and around the two lines of a wrapped note


def _note_lines(text: str, cx: float, cy: float, width: float, cell_h: float, height: float) -> List[Text]:
    """A note centred in a ``width`` x ``cell_h`` cell, on one or two lines, never below 2.5 mm.

    Review R4: shrinking the 43-character unit caveat to one line gave 1.9 mm
    text, below the lettering minimum (MIN_LETTERING_HEIGHT). One line is kept
    when it fits at ``MIN_LETTERING_HEIGHT`` or more; otherwise the words are
    split in two lines (before a word that opens a parenthesis when both lines
    then fit, else the most balanced split), each as high as the cell allows
    (``(cell_h - 3 gaps) / 2``, at most ``height``).
    """
    if not text:
        return []
    one = width / (TEXT_ADVANCE * len(text))
    words = text.split()
    if one >= min(height, MIN_LETTERING_HEIGHT) - 1e-9 or len(words) < 2:
        t, h = _fit_text(text, width, height, min_height=min(height, MIN_LETTERING_HEIGHT))
        return [Text((cx, cy), t, h, anchor="middle", valign="middle")]
    h2 = min(height, 0.5 * (cell_h - 3.0 * NOTE_LINE_GAP))
    splits = [(" ".join(words[:k]), " ".join(words[k:])) for k in range(1, len(words))]
    fits = [s for s in splits if TEXT_ADVANCE * MIN_LETTERING_HEIGHT * max(len(s[0]), len(s[1])) <= width + 1e-9]
    paren = [s for k, s in enumerate(splits, start=1) if words[k].startswith("(") and s in fits]
    a, b = paren[0] if paren else min(splits, key=lambda s: max(len(s[0]), len(s[1])))
    h = max(MIN_LETTERING_HEIGHT, min(h2, width / (TEXT_ADVANCE * max(len(a), len(b)))))
    dy = 0.5 * (h + NOTE_LINE_GAP)
    out = []
    for txt, y in ((a, cy + dy), (b, cy - dy)):
        t, hh = _fit_text(txt, width, h, min_height=MIN_LETTERING_HEIGHT)
        out.append(Text((cx, y), t, hh, anchor="middle", valign="middle"))
    return out


def strip_items(sheet: SheetSize, style: SheetStyle, scale_label: str, projection: str,
                source_note: str, unit_note: Optional[str] = None) -> List[SheetItem]:
    """180 x 9 strip above the title block: SCALE | projection symbol | Dimensions in mm | source.

    ``unit_note`` replaces "Dimensions in mm" when given (QA round 2 issue
    30: an STL whose unit was assumed prints the caveat on the sheet, e.g.
    "Dimensions in mm (unit assumed, see report)"; set on two lines rather
    than shrunk below 2.5 mm, review R4, see ``_note_lines``).
    """
    f = _frame_rect(sheet, style)
    x0 = f.xmax - style.title_block_w
    y0 = f.ymin + style.title_block_h
    y1 = y0 + style.strip_h
    ym = 0.5 * (y0 + y1)
    items: List[SheetItem] = [Segment((x0, y1), (f.xmax, y1), LineStyle.TITLE),
                              Segment((x0, y0), (x0, y1), LineStyle.TITLE)]
    for c in STRIP_COLS[1:-1]:
        items.append(Segment((x0 + c, y0), (x0 + c, y1), LineStyle.TITLE))
    cells = list(zip(STRIP_COLS[:-1], STRIP_COLS[1:]))
    units = str(unit_note).strip() if unit_note else "Dimensions in mm"
    for (c0, c1), txt in ((cells[0], f"SCALE {scale_label}"), (cells[3], source_note)):
        if not txt:
            continue
        t, h = _fit_text(txt, c1 - c0 - 2.0, style.text_height)
        items.append(Text((x0 + 0.5 * (c0 + c1), ym), t, h, anchor="middle", valign="middle"))
    c0, c1 = cells[2]
    items += _note_lines(units, x0 + 0.5 * (c0 + c1), ym, c1 - c0 - 2.0, style.strip_h, style.text_height)
    c0, c1 = cells[1]
    items += projection_symbol((x0 + 0.5 * (c0 + c1), ym), style.symbol_h, projection)
    return items


def _min_width(points: np.ndarray) -> float:
    """Smallest caliper width of a 2D point set (min over its convex hull edges), sheet mm.

    Unlike the shorter bounding-box side it does not depend on the region's
    orientation: a thin strip at 45 degrees is as narrow as the same strip
    lying flat. Degenerate input (fewer than 3 distinct points, collinear)
    gives 0.
    """
    pts = np.unique(np.asarray(points, dtype=float).reshape(-1, 2), axis=0)
    if len(pts) < 3:
        return 0.0

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: List[np.ndarray] = []
    for p in pts:                                   # Andrew's monotone chain (pts sorted by x, y)
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0.0:
            lower.pop()
        lower.append(p)
    upper: List[np.ndarray] = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0.0:
            upper.pop()
        upper.append(p)
    hull = np.array(lower[:-1] + upper[:-1])
    if len(hull) < 3:
        return 0.0
    edges = np.roll(hull, -1, axis=0) - hull
    lens = np.linalg.norm(edges, axis=1)
    keep = lens > 1e-12
    nrm = np.column_stack([-edges[keep, 1], edges[keep, 0]]) / lens[keep, None]
    proj = hull @ nrm.T                             # (n_hull, n_edges)
    return float(np.min(np.ptp(proj, axis=0)))


def _loop_area_perimeter(lp: np.ndarray) -> Tuple[float, float]:
    """(|shoelace area|, perimeter) of one closed loop (no repeated end point)."""
    p = np.asarray(lp, dtype=float).reshape(-1, 2)
    if len(p) < 3:
        return 0.0, 0.0
    q = np.roll(p, -1, axis=0)
    area = 0.5 * abs(float(np.sum(p[:, 0] * q[:, 1] - q[:, 0] * p[:, 1])))
    return area, float(np.sum(np.linalg.norm(q - p, axis=1)))


def _region_area_perimeter(outer: np.ndarray, holes: Sequence[np.ndarray]) -> Tuple[float, float]:
    """Material area (outer minus holes) and the summed perimeter of all loops of a region."""
    a, p = _loop_area_perimeter(outer)
    for h in holes:
        ha, hp = _loop_area_perimeter(h)
        a -= ha
        p += hp
    return max(a, 0.0), p


def _keyhole(outer: np.ndarray, holes: Sequence[np.ndarray]) -> np.ndarray:
    """One simple fill polygon of ``outer`` with its ``holes`` cut out through zero-width bridges.

    ``Polygon`` has no holes, so a filled region with holes would be filled
    across them (a thin ring drawn as a black disc). Each hole (CW from
    ``nest_loops``, outer CCW) is joined to the nearest vertex of the polygon
    built so far: ``... P_i, H_j ... H_j, P_i ...``. The slit has zero width,
    so even-odd and non-zero fills both leave the hole open.
    """
    poly = np.asarray(outer, dtype=float).reshape(-1, 2)
    for h in sorted((np.asarray(x, dtype=float).reshape(-1, 2) for x in holes),
                    key=lambda x: -float(x[:, 0].max())):
        if len(h) < 3:
            continue
        d = np.linalg.norm(poly[:, None, :] - h[None, :, :], axis=2)
        i, j = np.unravel_index(int(np.argmin(d)), d.shape)
        ring = np.vstack([h[j:], h[:j + 1]])
        poly = np.vstack([poly[:i + 1], ring, poly[i:]])
    return poly.copy()


_CORE_SAMPLES = 3000          # sample points per body / region for the thick-core test
_CORE_WORK = 2.0e8            # cap on points x edges of the distance computation


def _thick_core_fraction(nested: Sequence[Tuple[np.ndarray, Sequence[np.ndarray]]], depth: float) -> float:
    """Area fraction of the material of ``nested`` (outer, holes) deeper than ``depth`` (sheet mm).

    Review R4: the mean thickness 2A/P of a finned body (a heat sink) is pulled
    below the fill limit by its fins although its base is thick. The material
    is sampled on horizontal scan-line chords (``hatch_lines`` at 0 degrees,
    even-odd, so holes are excluded) with about ``_CORE_SAMPLES`` points, each
    weighted by its share of chord length; a point is deep when its distance
    to every loop edge exceeds ``depth``. 0.0 on degenerate input or failure.
    """
    try:
        loops = [np.asarray(lp, dtype=float).reshape(-1, 2) for outer, holes in nested
                 for lp in [outer] + list(holes)]
        loops = [lp for lp in loops if len(lp) >= 3]
        if not loops:
            return 0.0
        area = sum(_region_area_perimeter(o, h)[0] for o, h in nested)
        if area <= _EPS:
            return 0.0
        a = np.concatenate(loops)
        b = np.concatenate([np.roll(lp, -1, axis=0) for lp in loops])
        n_e = len(a)
        step = max(0.2, math.sqrt(area / _CORE_SAMPLES), math.sqrt(area * n_e / _CORE_WORK))
        chords = [hatch_lines([o] + list(h), step, 0.0, 0.5 * step) for o, h in nested]
        chords = [c for c in chords if len(c)]
        if not chords:
            return 0.0
        segs = np.concatenate(chords)
        x0 = np.minimum(segs[:, 0, 0], segs[:, 1, 0])
        length = np.abs(segs[:, 1, 0] - segs[:, 0, 0])
        n = np.maximum(1, np.round(length / step).astype(int))
        rep = np.repeat(np.arange(len(segs)), n)
        k = np.arange(int(n.sum())) - np.repeat(np.cumsum(n) - n, n)
        px = x0[rep] + (k + 0.5) * (length[rep] / n[rep])
        pts = np.column_stack([px, segs[rep, 0, 1]])
        w = length[rep] / n[rep]
        tot = float(w.sum())
        # the chord ends lie on the boundary: a point within depth of them is shallow (cheap
        # pre-filter; the exact edge distance runs on the survivors only)
        cand = np.flatnonzero((px - x0[rep] > depth) & (x0[rep] + length[rep] - px > depth))
        if not len(cand) or tot <= _EPS:
            return 0.0
        d = b - a
        dd = np.maximum(np.einsum("ij,ij->i", d, d), 1e-300)
        e_lo = np.minimum(a, b)
        e_hi = np.maximum(a, b)
        deep = np.zeros(len(pts), dtype=bool)
        cand = cand[np.argsort(pts[cand, 1], kind="stable")]
        chunk = 256
        for i0 in range(0, len(cand), chunk):
            ids = cand[i0:i0 + chunk]
            p = pts[ids]
            lo = p.min(axis=0) - depth
            hi = p.max(axis=0) + depth
            near = np.flatnonzero(np.all(e_hi >= lo, axis=1) & np.all(e_lo <= hi, axis=1))
            if not len(near):                 # no edge within depth of the chunk: all deep
                deep[ids] = True
                continue
            an, dn = a[near], d[near]
            rel = p[:, None, :] - an[None, :, :]
            t = np.clip(np.einsum("mnj,nj->mn", rel, dn) / dd[near][None, :], 0.0, 1.0)
            r = rel - t[:, :, None] * dn[None, :, :]
            deep[ids] = np.einsum("mnj,mnj->mn", r, r).min(axis=1) > depth * depth
        return float(w[deep].sum() / tot)
    except Exception as exc:          # a guard only: never lose the section over it
        log.warning("layout: thick-core test failed (%s)", exc)
        return 0.0


def _section_bodies(section: SectionView, loops: List[np.ndarray]) -> List[Tuple[object, List[np.ndarray]]]:
    """``[(body id, loops)]`` from ``section.loop_body`` (parallel to ``section.loops``), sorted by id.

    No ``loop_body`` (or one that does not match the loops) -> one body. The
    loops of each body are nested on their own, so a hole always belongs to
    an outer loop of its own body.
    """
    raw = [lp for lp in section.loops]
    ids = getattr(section, "loop_body", None)
    if ids is None:
        return [(0, loops)]
    try:
        ids = list(ids)
    except Exception:
        return [(0, loops)]
    if len(ids) != len(raw):
        log.info("layout: section.loop_body has %d entries for %d loops; one body assumed", len(ids), len(raw))
        return [(0, loops)]
    kept = [ids[i] for i, lp in enumerate(raw) if np.size(lp)]
    groups: Dict[object, List[np.ndarray]] = {}
    for b, lp in zip(kept, loops):
        groups.setdefault(b, []).append(lp)
    try:
        order = sorted(groups)
    except Exception:
        order = list(groups)
    return [(b, groups[b]) for b in order]


#: QA round 3 issue 1: the narrow regions of a hatched body (2A/P below two section
#: spacings, at least ``region_fill_below``) lower the body's ONE spacing only when they
#: hold more than this share of its area; otherwise they are hatched at the section spacing
#: (and filled when that leaves fewer than ``HATCH_MIN_SEGMENTS`` lines)
BODY_NARROW_SHARE = 0.2


def _body_hatch_spacing(spacing: float, thicknesses: Sequence[float], areas: Sequence[float]) -> float:
    """The one hatch spacing of a hatched body (QA round 3 issue 1, ISO 128-50: one part, one hatch).

    ``thicknesses`` / ``areas``: 2A/P and material area of each region (sheet mm). The
    section ``spacing``, unless the narrow regions (``region_fill_below() <= t < 2 x
    spacing``; thinner ones are filled anyway) hold more than ``BODY_NARROW_SHARE`` of
    the body's area: then the smallest ``thin_region_spacing`` of those regions (t / 3,
    at least ``HATCH_MIN_REGION``). Round 2 chose the spacing per region, so a narrow
    strip of one solid got 0.75-1.95 mm next to its 3 mm hatch.
    """
    s = float(spacing)
    total = float(sum(max(float(a), 0.0) for a in areas))
    narrow = [(float(t), max(float(a), 0.0)) for t, a in zip(thicknesses, areas)
              if region_fill_below() <= float(t) < 2.0 * s]
    if not narrow or total <= _EPS:
        return s
    if sum(a for _, a in narrow) <= BODY_NARROW_SHARE * total:
        return s
    return float(min(thin_region_spacing(s, t) for t, _ in narrow))


def _section_items(section: SectionView, origin: Point, scale: float, style: SheetStyle,
                   geom: BBox2D) -> Tuple[List[SheetItem], dict]:
    """Section outlines, hatch per region (outer loop with its holes), body edges, A-A label.

    QA round 2 issue 3: the fill decision is made per BODY (``section.loop_body``,
    the solid each loop comes from; missing -> one body), from the body's mean
    wall thickness on the sheet ``t = 2 A / P`` over all its loops (A = outer
    areas minus hole areas, P = all perimeters; a strip of width w gives about
    w whatever its shape, an annulus its radial width). A body with ``t`` below
    ``style.section_fill_below()`` (2.1 mm) is filled solid (ISO 128-50
    blackened thin section); otherwise every region of the body is hatched
    (one part, one treatment: a wall is never half black, half hatched). QA
    round 3 issue 1: a hatched body has ONE spacing (``_body_hatch_spacing``):
    the section spacing ``hatch_spacing`` of the shorter extent of all loops,
    or ``thin_region_spacing`` (t / 3, at least 0.7 mm) of its narrow regions
    when those hold more than ``BODY_NARROW_SHARE`` of the body's area (a thin
    ring alone gets t / 3); ``stats["body_spacing"]`` lists it per body (None
    for a filled body) and ``stats["hatch_spacing"]`` is the drawn spacing
    (the smallest when bodies differ). Round 1 (per region,
    bbox and hull caliper) filled the straight piece of a wall and hatched its
    L piece, and kept 3 mm hatching on a 0.8 mm ring.
    Review R4 guards: a thin body is still hatched when more than
    ``THICK_CORE_FRACTION`` of its area lies deeper than the fill limit inside
    the material (``_thick_core_fraction``: fins pull 2A/P down, the thick base
    must not go black; counted in ``stats["thick_core_bodies"]``); a region of
    a hatched body with its own t below ``region_fill_below()`` (1.4 mm), or
    with fewer than ``HATCH_MIN_SEGMENTS`` hatch lines at the body's spacing, is
    filled (a 0.8 mm island with one hatch line reads as an empty pocket),
    giving ``"mixed"``.
    Issue 28: bodies are ranked by id; body k is hatched at 45 degrees (k even)
    or 135 degrees (k odd), with a half-spacing phase shift for k = 2, 3 mod 4,
    so adjacent parts of an assembly read apart.
    """
    items: List[SheetItem] = []
    stats = {"hatch_segments": 0, "hatch_spacing": None, "loops": 0, "hatch_spacings": [],
             "filled_regions": 0, "fill": None, "filled_area_fraction": 0.0, "bodies": 0,
             "body_thickness": [], "hatch_angles": [], "thick_core_bodies": 0, "body_spacing": []}
    off = np.array([float(origin[0]), float(origin[1])])
    loops = [np.asarray(lp, dtype=float).reshape(-1, 2) * scale + off for lp in section.loops if np.size(lp)]
    bodies = [(b, nest_loops(lps)) for b, lps in _section_bodies(section, loops)] if loops else []
    bodies = [(b, nested) for b, nested in bodies if nested]
    ordered: List[np.ndarray] = []
    for _, nested in bodies:
        for outer, holes in nested:
            ordered.append(outer)
            ordered.extend(holes)
    for lp in ordered:
        items.append(Polyline(lp, True, LineStyle.SECTION))
    if ordered:
        bb = BBox2D.from_points(np.vstack(ordered))
        spacing = hatch_spacing(style, min(bb.width, bb.height))
        fill_below = section_fill_below()
        n_segs = 0
        area_all = area_filled = 0.0
        n_hatched = 0
        for rank, (_, nested) in enumerate(bodies):
            metrics = [_region_area_perimeter(outer, holes) for outer, holes in nested]
            a_sum = sum(m[0] for m in metrics)
            p_sum = sum(m[1] for m in metrics)
            t_body = 2.0 * a_sum / p_sum if p_sum > _EPS else 0.0
            stats["body_thickness"].append(float(t_body))
            area_all += a_sum
            thick_core = False
            if t_body < fill_below:
                # review R4: thin fins pull 2A/P down; a body with a thick core is hatched
                thick_core = _thick_core_fraction(nested, fill_below) > THICK_CORE_FRACTION
                stats["thick_core_bodies"] += int(thick_core)
            if t_body < fill_below and not thick_core:
                for outer, holes in nested:
                    # holes stay open: one keyhole polygon (a filled 0.8 mm ring is a ring, not a disc)
                    items.append(Polygon(_keyhole(outer, holes), LineStyle.SECTION))
                    stats["filled_regions"] += 1
                area_filled += a_sum
                stats["body_spacing"].append(None)
                continue
            angle = 45.0 if rank % 2 == 0 else 135.0
            ts = [2.0 * a_r / p_r if p_r > _EPS else 0.0 for a_r, p_r in metrics]
            s = _body_hatch_spacing(spacing, ts, [m[0] for m in metrics])
            phase = 0.5 * s if (rank // 2) % 2 == 1 else 0.0
            body_hatched = False
            for (outer, holes), (a_r, _p_r), t in zip(nested, metrics, ts):
                region = [outer] + list(holes)
                try:
                    segs = hatch_lines(region, s, angle, phase)
                except Exception as exc:              # pragma: no cover - cap on absurd inputs
                    log.warning("layout: hatch failed (%s)", exc)
                    segs = np.zeros((0, 2, 2))
                # review R4: a tiny island of a hatched body (t < 1.4 mm, or fewer than 2 hatch
                # lines at the body's spacing) reads as an empty pocket when hatched; it is
                # filled unless it has a thick core (fins on a thick base)
                if (t < region_fill_below() or len(segs) < HATCH_MIN_SEGMENTS) \
                        and _thick_core_fraction([(outer, holes)], fill_below) <= THICK_CORE_FRACTION:
                    items.append(Polygon(_keyhole(outer, holes), LineStyle.SECTION))
                    stats["filled_regions"] += 1
                    area_filled += a_r
                    continue
                items.append(HatchLines(segs, [lp.copy() for lp in region], LineStyle.HATCH))
                stats["hatch_spacings"].append(float(s))
                stats["hatch_angles"].append(float(angle))
                n_segs += int(len(segs))
                n_hatched += 1
                body_hatched = True
            stats["body_spacing"].append(float(s) if body_hatched else None)
        fill = ("solid" if n_hatched == 0 else "hatch" if stats["filled_regions"] == 0 else "mixed")
        drawn = [v for v in stats["body_spacing"] if v is not None]
        # QA round 3 issue 1: the report gives the DRAWN spacing (the single value when every
        # hatched body agrees, else the smallest); None when nothing is hatched
        stats.update(hatch_segments=n_segs, hatch_spacing=(min(drawn) if drawn else None),
                     nominal_hatch_spacing=spacing, loops=len(ordered), fill=fill,
                     bodies=len(bodies),
                     filled_area_fraction=float(area_filled / area_all) if area_all > _EPS else 0.0)
    if section.body is not None:
        for p in section.body.visible:
            items.append(_restyle(transform_primitive(p, origin, scale), LineStyle.VISIBLE))
    label = f"{section.label}-{section.label}"
    items.append(Text((0.5 * (geom.xmin + geom.xmax), geom.ymax + LABEL_GAP), label,
                      style.label_height, anchor="middle"))
    return items, stats


def iso_label(scale: float) -> str:
    """Label of the iso view: ``ISO (1:2)`` (spec 3.6: labelled with its scale)."""
    return f"ISO ({scale_text(scale)})"


def _marker_end(style: SheetStyle, k: int, collinear: bool) -> Tuple[float, float, float, bool]:
    """``(dash start, dash end, arrow at, letter beside the arrow tip)`` along one end of the cutting
    line, sheet mm from the view outline, with ``k`` dimension bands on that side.

    QA round 2 issues 13 / 14: the wide dash starts ``MARKER_GAP`` outside the
    outline (it used to start 3 mm inside and covered part edges), and the
    marker never crosses a dimension line: with bands on that side the dash
    ends ``MARKER_BAND_CLEAR`` before the first dimension line, the arrow
    stays at ``MARKER_ARROW_AT`` and the letter goes beyond the arrow head
    (round 1 moved the arrow beyond the bands and joined it with a narrow
    chain across every dimension and extension line). When an extension line
    of that side lies on the cutting line (``collinear``: the plane passes
    through a dimensioned hole centre; flipping or re-datuming the dimension
    cannot help, the extension line at the hole centre is the same on both
    sides and from both datums), the whole end moves past the outermost
    dimension line, its overshoot and a text on its far side.
    """
    dash = float(style.section_dash_len)
    if k > 0 and collinear:
        # past the outermost dimension line, its extension-line overshoot and a value text
        # drawn on the far side of it
        start = (style.dim_first_offset + (k - 1) * style.dim_pitch
                 + max(style.ext_overshoot, style.dim_text_gap + style.text_height) + MARKER_GAP)
        return start, start + dash, start + (MARKER_ARROW_AT - MARKER_GAP), False
    if k > 0:
        end = max(MARKER_GAP + 1.0, min(MARKER_GAP + dash, style.dim_first_offset - MARKER_BAND_CLEAR))
        return MARKER_GAP, end, min(MARKER_ARROW_AT, end), True
    return MARKER_GAP, MARKER_GAP + dash, MARKER_ARROW_AT, False


def marker_arrow_at(style: SheetStyle, k: int, collinear: bool = False) -> float:
    """Distance of the section arrow from the outline along the cutting line, given ``k`` bands there."""
    return _marker_end(style, int(k), bool(collinear))[2]


def _pt2(v) -> Point:
    return (float(v[0]), float(v[1]))


def _collinear_sides(geo, plan: DimensionPlan, scale: float, tol: float = 0.5) -> set:
    """Sides of the marker view whose dimension extension lines lie on the cutting line.

    An extension line of a linear dim is perpendicular to the dimension and
    runs from a measured point toward the dim's band side. It lies on the
    cutting line when the cutting line is parallel to it and the measured
    point is within ``tol`` sheet mm of it (QA round 2 issue 14: the plane
    through a hole centre and the position dim of that hole).
    """
    name, p2, d2, _a2, (_t0, s0, _t1, s1) = geo
    out: set = set()
    if not plan.dims:
        return out
    for d, b in zip(plan.dims, _resolved_bands(plan)):
        if b is None or d.kind != "linear" or d.view != name or b[0] not in (s0, s1):
            continue
        a = math.radians(float(d.angle_deg))
        ext = np.array([-math.sin(a), math.cos(a)])
        if abs(float(ext[0] * d2[1] - ext[1] * d2[0])) > 1e-6:
            continue                                  # the cutting line is not parallel to the ext lines
        for p in (d.p1, d.p2):
            r = np.asarray(p, dtype=float) - p2
            if abs(float(r[0] * d2[1] - r[1] * d2[0])) * scale < tol:
                out.add(b[0])
                break
    return out


def _marker_items(geo, origin: Point, scale: float, style: SheetStyle, label: str,
                  counts: Tuple[int, int, int, int] = (0, 0, 0, 0),
                  collinear: Sequence[str] = ()) -> List[SheetItem]:
    """Cutting-line marker: per end a wide dash, an arrow (shaft + head) and the letter.

    See ``_marker_end`` for the positions along the line; ``collinear``: the
    sides whose end goes past the dimension bands (``_marker_past_sides``: an
    extension line on the cutting line, or one the arrow or letter would
    cross). The letter is placed the same way at both ends (QA round 3 issue
    4): beside the arrow tip when either end has it there, else above the
    arrow.
    """
    _, p2, d2, a2, (t0, s0, t1, s1) = geo
    off = np.array([float(origin[0]), float(origin[1])])
    items: List[SheetItem] = []
    h = style.label_height
    ends = []
    for t, side, outward in ((t0, s0, -d2), (t1, s1, d2)):
        e = off + scale * (p2 + t * d2)
        ends.append((e, outward) + _marker_end(style, int(counts[SIDES.index(side)]), side in collinear))
    # QA round 3 issue 4: ONE letter placement for the marker - beside the arrow tip when
    # either end needs it (bands on that side), else above the arrow - so the two ends of
    # one cutting line read alike
    beside = any(end[5] for end in ends)
    for e, outward, start, end, at, _own in ends:
        items.append(Segment(_pt2(e + start * outward), _pt2(e + end * outward), LineStyle.CUTTING))
        s = e + at * outward
        tip = s + style.section_arrow_len * a2
        neck = tip - style.section_head_len * a2
        items.append(Segment(_pt2(s), _pt2(neck), LineStyle.CUTTING))
        items.append(arrowhead(tip, a2, style.section_head_len, style.section_head_angle_deg, LineStyle.CUTTING))
        if beside:
            half = 0.5 * (TEXT_ADVANCE * h * len(label) if abs(a2[0]) >= abs(a2[1]) else h)
            c = tip + (MARKER_LETTER_GAP + half) * a2
        else:
            c = s + MARKER_LETTER_AT * outward + 0.5 * style.section_arrow_len * a2
        items.append(Text(_pt2(c), label, h, anchor="middle", valign="middle"))
    return items


def _view_dimension_segments(name: str, plan: DimensionPlan, boxes: Dict[str, BBox2D], scale: float,
                              style: SheetStyle) -> List[Segment]:
    """Extension and dimension line ``Segment`` s of the linear dims of view ``name``, rendered as
    ``compose`` renders them (band offsets, outside texts) with the view origin at (0, 0)."""
    bb = boxes.get(name)
    if bb is None or not plan.dims:
        return []
    geom = BBox2D(scale * bb.xmin, scale * bb.ymin, scale * bb.xmax, scale * bb.ymax)
    ext_index = _ext_index(plan)
    out: List[Segment] = []
    for d, band in zip(plan.dims, _resolved_bands(plan)):
        if d.view != name or d.kind != "linear":
            continue
        try:
            ds = _dim_to_sheet(d, band, (0.0, 0.0), scale, geom, style)
            go_high = _outside_high(d, band, bb, scale, style, ext_index)
            off = _outside_text_offset(ds, geom, style, go_high) if go_high is not None else 0.0
            off = float(ds.text_offset or 0.0) or off
            if off != float(ds.text_offset or 0.0):
                ds = dataclasses.replace(ds, text_offset=off)
            out += [it for it in render_dimension(ds, style) if isinstance(it, Segment)]
        except Exception as exc:              # a check only: never lose the sheet over it
            log.debug("layout: dimension segments of %r failed (%s)", d.text, exc)
    return out


def _marker_past_sides(geo, plan: DimensionPlan, boxes: Dict[str, BBox2D], style: SheetStyle, label: str,
                       scale: float, counts: Tuple[int, int, int, int]) -> set:
    """Sides whose marker end goes past the dimension bands (``_marker_end`` collinear branch).

    ``_collinear_sides`` (an extension line on the cutting line) plus, QA round 3
    issue 5, every side with bands whose marker end (dash, arrow shaft, head or
    letter) touches an extension or dimension line of the marker view's linear
    dims: the arrow at ``MARKER_ARROW_AT`` runs across the first band zone,
    parallel to the band side, and crossed the extension lines of a dim whose
    measured point lies near the cutting line. ``compose`` and ``_marker_room``
    both use it, so the room ``choose_scale`` reserves is the drawn marker.
    """
    sides = set(_collinear_sides(geo, plan, scale))
    name = geo[0]
    segs = _view_dimension_segments(name, plan, boxes, scale, style)
    if not segs:
        return sides
    _, _p2, _d2, _a2, (_t0, s0, _t1, s1) = geo
    for _round in range(2):                   # the letter mode follows both ends (issue 4)
        items = _marker_items(geo, (0.0, 0.0), scale, style, label, counts, sides)
        half = len(items) // 2
        added = False
        for side, part in ((s0, items[:half]), (s1, items[half:])):
            if side in sides or int(counts[SIDES.index(side)]) <= 0:
                continue
            if any(_item_hits_segment(it, s.a, s.b) for it in part for s in segs):
                sides.add(side)
                added = True
        if not added:
            break
    return sides


def _marker_room(marker, plan: DimensionPlan, boxes: Dict[str, BBox2D], style: SheetStyle, label: str
                 ) -> Callable[[float], Dict[str, Sides]]:
    """``scale -> {marker view: sides}``: sheet mm the rendered marker reaches beyond the view outline."""
    if marker is None or boxes.get(marker[0]) is None:
        return lambda scale: {}
    name = marker[0]
    counts = band_counts(plan).get(name, (0, 0, 0, 0))

    def room(scale: float) -> Dict[str, Sides]:
        bb = boxes[name]
        geom = BBox2D(scale * bb.xmin, scale * bb.ymin, scale * bb.xmax, scale * bb.ymax)
        items = _marker_items(marker, (0.0, 0.0), scale, style, label, counts,
                              _marker_past_sides(marker, plan, boxes, style, label, scale, counts))
        box = BBox2D.of_primitives(items)
        if box is None:
            return {}
        reach = (geom.xmin - box.xmin, geom.ymin - box.ymin, box.xmax - geom.xmax, box.ymax - geom.ymax)
        return {name: tuple(max(0.0, float(r)) for r in reach)}      # type: ignore[dict-item]

    return room


def _dim_to_sheet(d: Dimension, band: Optional[Tuple[str, int]], origin: Point, scale: float,
                  geom: BBox2D, style: SheetStyle) -> Dimension:
    o = np.array([float(origin[0]), float(origin[1])])

    def tp(p) -> Point:
        q = o + scale * np.asarray(p, dtype=float)
        return (float(q[0]), float(q[1]))

    if d.kind == "linear":
        p1, p2 = tp(d.p1), tp(d.p2)
        side, k = band if band is not None else ("bottom", 0)
        off = style.dim_first_offset + k * style.dim_pitch
        mx, my = 0.5 * (p1[0] + p2[0]), 0.5 * (p1[1] + p2[1])
        base = {"bottom": (mx, geom.ymin - off), "top": (mx, geom.ymax + off),
                "left": (geom.xmin - off, my), "right": (geom.xmax + off, my)}[side]
        return dataclasses.replace(d, p1=p1, p2=p2, base=(float(base[0]), float(base[1])))
    c = np.asarray(tp(d.center))
    r = scale * float(d.radius)
    a = math.radians(float(d.leader_angle_deg))
    u = np.array([math.cos(a), math.sin(a)])
    tip = c + r * u
    length = _callout_leader_length(tip, u, geom, d.text, style)
    end = tip + length * u
    return dataclasses.replace(d, center=(float(c[0]), float(c[1])), radius=r,
                               base=(float(end[0]), float(end[1])))


def _callout_leader_length(tip: np.ndarray, u: np.ndarray, geom: BBox2D, text: str, style: SheetStyle,
                           extra: float = 0.0) -> float:
    """Leader length from the circle: out of the outline by ``LEADER_EXIT_MARGIN`` (at least
    ``LEADER_LEN``), plus ``extra``, and longer when the text above the shelf would still sit on
    the view outline (a leader leaving through the bottom edge: the text is drawn ABOVE the
    shelf, back toward the view)."""
    length = max(LEADER_LEN, _ray_exit(tip, u, geom) + LEADER_EXIT_MARGIN) + float(extra)
    gap, h = style.dim_text_gap, style.text_height
    right = u[0] >= -1e-12
    sx = 1.0 if right else -1.0
    for _ in range(int(4.0 * h) + 8):
        end = tip + length * u
        t = Text((float(end[0] + sx * gap), float(end[1] + gap)), text, h,
                 anchor="start" if right else "end", valign="baseline")
        tb = BBox2D.of_primitives([t])
        if tb is None or not _overlaps(tb, geom):
            break
        length += 1.0
    return float(length)


# ------------------------------------------------------- text collisions --

def _box_of(item) -> Optional[BBox2D]:
    try:
        return BBox2D.of_primitives([item])
    except Exception:
        return None


def _segment_hits_box(a, b, box: BBox2D) -> bool:
    """Liang-Barsky: does segment a-b pass through the (open) box?"""
    x0, y0 = float(a[0]), float(a[1])
    dx, dy = float(b[0]) - x0, float(b[1]) - y0
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, x0 - box.xmin), (dx, box.xmax - x0), (-dy, y0 - box.ymin), (dy, box.ymax - y0)):
        if abs(p) < 1e-12:
            if q <= 1e-9:
                return False
            continue
        t = q / p
        if p < 0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
        if t0 > t1 - 1e-12:
            return False
    return True


def _segments_intersect(a, b, c, d, eps: float = 1e-9) -> bool:
    """Do the closed segments a-b and c-d touch or cross (collinear overlap counts)?"""
    p, r = np.asarray(a, dtype=float), np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    q, s = np.asarray(c, dtype=float), np.asarray(d, dtype=float) - np.asarray(c, dtype=float)
    rxs = float(r[0] * s[1] - r[1] * s[0])
    qp = q - p
    qpxr = float(qp[0] * r[1] - qp[1] * r[0])
    if abs(rxs) < eps:
        if abs(qpxr) > eps * max(1.0, float(np.linalg.norm(r))):
            return False                              # parallel, apart
        rr = float(r @ r)
        if rr < eps:                                  # a-b is a point: on c-d?
            ss = float(s @ s)
            if ss < eps:
                return float(np.linalg.norm(qp)) < 1e-6
            w = float((p - q) @ s) / ss
            return -eps <= w <= 1.0 + eps and float(np.linalg.norm(q + w * s - p)) < 1e-6
        t0 = float(qp @ r) / rr
        t1 = t0 + float(s @ r) / rr
        lo, hi = min(t0, t1), max(t0, t1)
        return hi >= -eps and lo <= 1.0 + eps
    t = float(qp[0] * s[1] - qp[1] * s[0]) / rxs
    u = qpxr / rxs
    return -eps <= t <= 1.0 + eps and -eps <= u <= 1.0 + eps


def _item_hits_segment(item, a, b) -> bool:
    """Does the segment a-b touch a marker item (Segment exactly, other items by their box)?"""
    if isinstance(item, Segment):
        return _segments_intersect(a, b, item.a, item.b)
    box = _box_of(item)
    return box is not None and _segment_hits_box(a, b, box)


def _item_hits_box(item, box: BBox2D) -> bool:
    if isinstance(item, Segment):
        return _segment_hits_box(item.a, item.b, box)
    b = _box_of(item)
    return b is not None and _overlaps(b, box)


def _render_group(ds: Dimension, style: SheetStyle):
    items = render_dimension(ds, style)
    text = next((it for it in items if isinstance(it, Text)), None)
    return items, text


def _callout_candidate(ds: Dimension, angle: float, extra: float, geom: BBox2D,
                       style: SheetStyle) -> Dimension:
    """Sheet-space diameter callout with leader ``angle`` leaving ``geom`` plus ``extra`` mm."""
    a = math.radians(float(angle))
    u = np.array([math.cos(a), math.sin(a)])
    c = np.asarray(ds.center, dtype=float)
    tip = c + abs(float(ds.radius)) * u
    length = _callout_leader_length(tip, u, geom, ds.text, style, extra)
    end = tip + length * u
    return dataclasses.replace(ds, leader_angle_deg=float(angle), base=(float(end[0]), float(end[1])))


CALLOUT_EXTRA_STEPS = 12       # leader lengthenings tried per angle when a callout text collides
TEXT_CLEARANCE = 1.0           # sheet mm a moved text keeps from every other text (adjacent texts read as one)


def _resolve_text_collisions(items: List[SheetItem], sheet: Sheet, groups: List[List[int]],
                             marker_idx: List[int], fixed_texts: List[int], geoms: Dict[str, BBox2D],
                             rects: Dict[str, BBox2D], allowed: BBox2D, blocked: List[BBox2D],
                             style: SheetStyle) -> dict:
    """Move dimension texts off the section marker and callout texts off each other.

    QA round 1 issue 4: a linear dimension text crossed by the cutting line
    (wide dash, narrow connector, arrow, letter) is shifted along its line by
    ``k * (text width / 2 + text height)``, toward the longer free half
    first. Issue 8: a diameter callout whose text (or leader) overlaps
    another text, or whose text sits on the marker, tries the leader
    angles 45 / 135 / -45 / -135 and longer leaders; a candidate must keep
    its text off every other text and the marker and inside the frame
    (``allowed``, off ``blocked``); among those the ranking is (QA round 3
    issue 8): out of every other view's placement rect, off other dimension
    lines, inside its own view's rect, off every view outline, then the
    shortest leader. ``items`` and ``sheet.dimensions`` are
    updated in place (same item count per dimension). Returns counts.
    """
    stats = {"linear_text_shifted": 0, "callouts_moved": 0, "unresolved": 0}
    marker = [items[i] for i in marker_idx]
    h = style.text_height

    def text_index(g: List[int]) -> Optional[int]:
        return next((i for i in g if isinstance(items[i], Text)), None)

    def other_text_boxes(skip: int) -> List[BBox2D]:
        out = []
        for k, g in enumerate(groups):
            if k == skip:
                continue
            ti = text_index(g)
            if ti is not None:
                b = _box_of(items[ti])
                if b is not None:
                    out.append(b)
        for i in fixed_texts:
            b = _box_of(items[i])
            if b is not None:
                out.append(b)
        return out

    def hits_marker(box: BBox2D) -> bool:
        return any(_item_hits_box(m, box) for m in marker)

    def replace(k: int, ds: Dimension, rendered: List[SheetItem]) -> None:
        for i, it in zip(groups[k], rendered):
            items[i] = it
        sheet.dimensions[k] = ds

    # 1. linear texts on the cutting line: shift along the dimension line
    for k, ds in enumerate(sheet.dimensions):
        if ds.kind != "linear" or not marker:
            continue
        ti = text_index(groups[k])
        if ti is None:
            continue
        box = _box_of(items[ti])
        if box is None or not hits_marker(box):
            continue
        a = math.radians(float(ds.angle_deg))
        d = np.array([math.cos(a), math.sin(a)])
        p1, p2 = np.asarray(ds.p1, dtype=float), np.asarray(ds.p2, dtype=float)
        u = d if float((p2 - p1) @ d) >= 0 else -d
        mid = 0.5 * float(p1 @ u + p2 @ u)
        lo, hi = min(float(p1 @ u), float(p2 @ u)), max(float(p1 @ u), float(p2 @ u))
        crossings = [float(np.mean([np.asarray(m.a, float) @ u, np.asarray(m.b, float) @ u]))
                     for m in marker if isinstance(m, Segment) and _segment_hits_box(m.a, m.b, box)]
        tc = float(np.mean(crossings)) if crossings else mid
        first = 1.0 if (hi - tc) >= (tc - lo) else -1.0
        step = 0.5 * TEXT_ADVANCE * h * len(ds.text or "") + h
        others = other_text_boxes(k)
        rect = rects.get(ds.view)
        best = None
        for m in range(1, 7):
            for sgn in (first, -first):
                cand = dataclasses.replace(ds, text_offset=sgn * m * step)
                rendered, t = _render_group(cand, style)
                tb = _box_of(t)
                if tb is None or hits_marker(tb) or any(_overlaps(tb, o.expanded(TEXT_CLEARANCE)) for o in others):
                    continue
                inside = rect is None or all(_inside(b, rect.expanded(1e-6)) for b in
                                             (_box_of(it) for it in rendered) if b is not None)
                score = (0 if inside else 1, m)
                if best is None or score < best[0]:
                    best = (score, cand, rendered)
            if best is not None and best[0][0] == 0:
                break
        if best is not None:
            replace(k, best[1], best[2])
            stats["linear_text_shifted"] += 1

    # 2. diameter callouts: no text on another text, the marker or a foreign leader
    dim_segments = {k: [i for i in g if isinstance(items[i], Segment)] for k, g in enumerate(groups)}

    def callout_clash(k: int, rendered: List[SheetItem], t: Text, others: List[BBox2D]) -> bool:
        tb = _box_of(t)
        if tb is None:
            return True
        if any(_overlaps(tb, o.expanded(TEXT_CLEARANCE)) for o in others) or hits_marker(tb):
            return True
        for it in rendered:          # own leader / shelf across another text or the section marker
            if not isinstance(it, Segment):
                continue
            if any(_segment_hits_box(it.a, it.b, o.expanded(-0.2)) for o in others):
                return True
            if any(_item_hits_segment(m, it.a, it.b) for m in marker):     # round 2 issue 10
                return True
        return False

    outlines = list(geoms.values())

    def callout_score(k: int, rendered: List[SheetItem], t: Text, others: List[BBox2D], rect, e: int, ai: int):
        """None when the placement is not allowed, else (into another view's rect, on lines,
        out of its own rect, on outlines, leader length, ai).

        QA round 3 issue 8: a text crossed by another dimension's line is a collision of the
        sheet (layout_text_collision); it now ranks before leaving the own placement rect
        into free paper (measured on NIST FTC 10: the only in-rect spot for '8 x Ø2.5' put
        it on the 27 extension line below the view; a clean spot past the view's lower right
        corner, outside the rect, is taken instead), but never before running into another
        view's rect."""
        if callout_clash(k, rendered, t, others):
            return None
        tb = _box_of(t)
        if not _inside(tb, allowed) or any(_overlaps(tb, b) for b in blocked):
            return None
        boxes_r = [b for b in (_box_of(it) for it in rendered) if b is not None]
        out_rect = 0 if (rect is None or all(_inside(b, rect.expanded(1e-6)) for b in boxes_r)) else 1
        foreign_rects = [r for name, r in rects.items() if name != sheet.dimensions[k].view]
        into_other = 1 if out_rect and any(_overlaps(b, r) for b in boxes_r for r in foreign_rects) else 0
        on_outline = sum(1 for o in outlines if _overlaps(tb, o))
        foreign = [items[i] for kk, segs in dim_segments.items() if kk != k for i in segs]
        inner = tb.expanded(-LINE_CROSS_SHRINK)
        on_lines = sum(1 for s in foreign if _segment_hits_box(s.a, s.b, inner))
        leader = next((it for it in rendered if isinstance(it, Segment)), None)
        length = math.hypot(leader.b[0] - leader.a[0], leader.b[1] - leader.a[1]) if leader is not None else 0.0
        return (into_other, on_lines, out_rect, on_outline, round(length, 3), ai)

    for _pass in range(2):
        for k, ds in enumerate(sheet.dimensions):
            if ds.kind != "diameter" or ds.view not in geoms:
                continue
            g = groups[k]
            ti = text_index(g)
            if ti is None:
                continue
            others = other_text_boxes(k)
            rect = rects.get(ds.view)
            now = callout_score(k, [items[i] for i in g], items[ti], others, rect, 0, 0)
            if now is not None and now[:4] == (0, 0, 0, 0):
                continue                      # clean as planned
            geom = geoms[ds.view]
            angles = [float(ds.leader_angle_deg)] + [a for a in LEADER_ANGLES
                                                     if abs(a - float(ds.leader_angle_deg)) > 1e-9]
            best = None
            for e in range(CALLOUT_EXTRA_STEPS + 1):
                extra = e * (h + 2.0 * style.dim_text_gap)
                for ai, ang in enumerate(angles):
                    cand = _callout_candidate(ds, ang, extra, geom, style)
                    rendered, t = _render_group(cand, style)
                    score = callout_score(k, rendered, t, others, rect, e, ai)
                    if score is not None and (best is None or score < best[0]):
                        best = (score, cand, rendered)
                if best is not None and best[0][:4] == (0, 0, 0, 0) and extra > best[0][4]:
                    break                     # every longer leader is longer than the clean best
            if best is None:
                if now is None:
                    stats["unresolved"] += 1
                continue
            if now is not None and best[0] >= now:
                continue
            replace(k, best[1], best[2])
            stats["callouts_moved"] += 1
    return stats


#: sheet mm a text box is shrunk by before a dimension line counts as crossing it (the box
#: estimate is generous; the callout pass uses the same margin)
LINE_CROSS_SHRINK = 0.2


def _collision_lists(items: Sequence[SheetItem], texts_idx: Sequence[int], marker_idx: Sequence[int],
                     groups: Optional[Sequence[Sequence[int]]] = None,
                     marker_dim_idx: Optional[Sequence[int]] = None) -> Dict[str, List[str]]:
    """The collisions left on a sheet, by kind (QA round 3 issue 8).

    ``texts``: overlapping text pairs and texts crossed by a marker item
    (``marker_idx``); with ``groups`` (the item indices of each dimension):
    ``text_line_crossings``, texts crossed by a ``Segment`` (extension,
    dimension, leader or shelf line) of ANOTHER dimension (box shrunk by
    ``LINE_CROSS_SHRINK``), and ``marker_dimension_crossings``, items of
    ``marker_dim_idx`` (default ``marker_idx``; the letters included) that
    touch a dimension ``Segment``.
    """
    texts: List[str] = []
    boxes = [(i, _box_of(items[i])) for i in texts_idx]
    boxes = [(i, b) for i, b in boxes if b is not None]
    for a in range(len(boxes)):
        ia, ba = boxes[a]
        for b in range(a + 1, len(boxes)):
            ib, bb = boxes[b]
            if _overlaps(ba, bb):
                texts.append(f"{items[ia].text!r} overlaps {items[ib].text!r}")
        for m in marker_idx:
            if m != ia and _item_hits_box(items[m], ba):
                texts.append(f"{items[ia].text!r} crossed by the section marker")
                break
    lines: List[str] = []
    marks: List[str] = []
    if groups:
        owner = {i: k for k, g in enumerate(groups) for i in g}
        segs = [(k, items[i]) for k, g in enumerate(groups) for i in g if isinstance(items[i], Segment)]
        mk = list(marker_idx if marker_dim_idx is None else marker_dim_idx)
        skip = set(mk)
        for i, b in boxes:
            if i in skip:
                continue                       # a marker letter: counted with the marker below
            inner = b.expanded(-LINE_CROSS_SHRINK)
            if inner.xmax <= inner.xmin or inner.ymax <= inner.ymin:
                continue
            own = owner.get(i)
            if any(k != own and _segment_hits_box(s.a, s.b, inner) for k, s in segs):
                lines.append(f"{items[i].text!r} crossed by a dimension line")
        for m in mk:
            if any(owner.get(m) != k and _item_hits_segment(items[m], s.a, s.b) for k, s in segs):
                what = f"letter {items[m].text!r}" if isinstance(items[m], Text) else "line"
                marks.append(f"section marker ({what}) crosses a dimension line")
    return {"texts": texts, "text_line_crossings": lines, "marker_dimension_crossings": marks}


def text_collisions(items: Sequence[SheetItem], texts_idx: Sequence[int], marker_idx: Sequence[int],
                    groups: Optional[Sequence[Sequence[int]]] = None,
                    marker_dim_idx: Optional[Sequence[int]] = None) -> List[str]:
    """Descriptions of every pair of overlapping texts and every text crossed by a marker item;
    with ``groups`` (QA round 3 issue 8) also every text crossed by another dimension's line
    and every marker item touching a dimension line (``_collision_lists``)."""
    c = _collision_lists(items, texts_idx, marker_idx, groups, marker_dim_idx)
    return c["texts"] + c["text_line_crossings"] + c["marker_dimension_crossings"]


def _extra_allowances(plan: DimensionPlan, boxes: Dict[str, BBox2D], counts: Dict[str, Tuple[int, int, int, int]],
                      marker, style: SheetStyle) -> Dict[str, Sides]:
    extra: Dict[str, List[float]] = {}

    def add(view: str, side: str, mm: float):
        e = extra.setdefault(view, [0.0, 0.0, 0.0, 0.0])
        i = SIDES.index(side)
        e[i] = max(e[i], mm)

    # diameter callouts: scale-dependent exact reach, see _callout_room (QA round 1 issue 7:
    # the fixed reach measured from the circle was ADDED to the bands, about 25 mm per side)
    for labelled in (SECTION_KEY, "iso"):
        if labelled in boxes:
            add(labelled, "top", LABEL_GAP + style.label_height + 1.0)
    # the section marker: scale-dependent exact reach, see _marker_room (round 2 issues 13 / 14)
    return {v: (e[0], e[1], e[2], e[3]) for v, e in extra.items()}


def _callout_room(plan: DimensionPlan, boxes: Dict[str, BBox2D], style: SheetStyle
                  ) -> Callable[[float], Dict[str, Sides]]:
    """``scale -> {view: sides}`` sheet mm the diameter callouts reach beyond the view outline.

    Renders each callout exactly as ``compose`` does at that scale (leader
    out of the outline by ``LEADER_EXIT_MARGIN``, shelf, text) and measures
    its box against the geometry box. ``choose_scale`` takes the larger of
    this and the bands on each side: the callout text sits in the band zone
    next to the outline, the collision pass in ``compose`` keeps it off the
    dimension texts.
    """
    calls = [d for d in plan.dims if d.kind == "diameter" and d.view in boxes]

    def room(scale: float) -> Dict[str, Sides]:
        out: Dict[str, List[float]] = {}
        for d in calls:
            bb = boxes[d.view]
            geom = BBox2D(scale * bb.xmin, scale * bb.ymin, scale * bb.xmax, scale * bb.ymax)
            ds = dataclasses.replace(d, center=(scale * float(d.center[0]), scale * float(d.center[1])),
                                     radius=scale * float(d.radius))
            ds = _callout_candidate(ds, float(d.leader_angle_deg), 0.0, geom, style)
            box = BBox2D.of_primitives(render_dimension(ds, style))
            if box is None:
                continue
            e = out.setdefault(d.view, [0.0, 0.0, 0.0, 0.0])
            reach = (geom.xmin - box.xmin, geom.ymin - box.ymin, box.xmax - geom.xmax, box.ymax - geom.ymax)
            for i in range(4):
                e[i] = max(e[i], float(reach[i]))
        return {v: (e[0], e[1], e[2], e[3]) for v, e in out.items()}

    return room


def _max_room(*fns: Callable[[float], Dict[str, Sides]]) -> Callable[[float], Dict[str, Sides]]:
    def room(scale: float) -> Dict[str, Sides]:
        out: Dict[str, List[float]] = {}
        for fn in fns:
            for v, s in fn(scale).items():
                e = out.setdefault(v, [0.0, 0.0, 0.0, 0.0])
                for i in range(4):
                    e[i] = max(e[i], float(s[i]))
        return {v: (e[0], e[1], e[2], e[3]) for v, e in out.items()}
    return room


def _outside_goes_high(room_lo: float, room_hi: float) -> bool:
    """The outside text goes past the high end when that end has at least as much room."""
    return room_hi >= room_lo - 1e-6


def _linear_axis(d: Dimension) -> Optional[int]:
    a = math.radians(float(d.angle_deg))
    if abs(math.sin(a)) < 1e-6:
        return 0
    if abs(math.cos(a)) < 1e-6:
        return 1
    return None


def _ext_index(plan: DimensionPlan) -> Dict[Tuple[str, str], List[Tuple[int, int, float]]]:
    """(view, side) -> [(band, axis, measured coordinate along the dim axis)] of the linear dims."""
    out: Dict[Tuple[str, str], List[Tuple[int, int, float]]] = {}
    for d, b in zip(plan.dims, _resolved_bands(plan)):
        axis = _linear_axis(d) if d.kind == "linear" else None
        if b is None or axis is None:
            continue
        for p in (d.p1, d.p2):
            out.setdefault((d.view, b[0]), []).append((int(b[1]), axis, float(p[axis])))
    return out


def _outside_high(d: Dimension, band: Optional[Tuple[str, int]], bb: BBox2D, scale: float,
                  style: SheetStyle, ext_index) -> Optional[bool]:
    """Which end an outside value text of the view-space linear dim ``d`` goes past (True = high
    coordinate end), or None when the text fits between the extension lines.

    The end with fewer extension lines of the same side's dims in this band or
    further out (those lines run through this band's zone) between its own
    extension line and the far end of the text wins, then the end with more
    room inside the view's extent. QA round 3 issue 24: round 2 counted only
    lines inside the text box, so a line between the span and its text (the
    35.15 next to the 12.93 of NIST FTC 07) did not count and the text read
    as belonging to the neighbouring span. A text that leaves the view's
    extent into a corner where the perpendicular side has bands counts one
    more (it meets those dims' lines and texts there). ``compose`` and
    ``_short_span_room`` both use it, so the reserved room is what is drawn.
    """
    axis = _linear_axis(d)
    if axis is None or d.p1 is None or d.p2 is None:
        return None
    c1, c2 = float(d.p1[axis]), float(d.p2[axis])
    lo, hi = min(c1, c2), max(c1, c2)
    span = scale * (hi - lo)
    tw = TEXT_ADVANCE * style.text_height * len(d.text or "")
    if tw + 2.0 * style.dim_text_gap <= span:
        return None
    reach = style.arrow_length + SHORT_SPAN_EXTRA
    side, k = band if band is not None else ("bottom" if axis == 0 else "left", 0)
    lines = [scale * c for j, ax, c in (ext_index or {}).get((d.view, side), []) if j >= k and ax == axis]
    crossings = []
    eps = 1e-6                     # the dim's own extension lines (and lines on them) do not count
    bmin, bmax = (bb.xmin, bb.xmax) if axis == 0 else (bb.ymin, bb.ymax)
    # a text that leaves the view's extent runs into the corner of the perpendicular side's bands
    # (their dimension lines, extended under their own outside texts): occupied when that side
    # has bands (the 12.93 / 12.93 pair of NIST FTC 07 met there)
    ends = (("right" if axis == 0 else "top", scale * hi + eps, scale * hi + reach + tw + 0.5,
             scale * hi + reach + tw > scale * bmax + 1e-9),
            ("left" if axis == 0 else "bottom", scale * lo - reach - tw - 0.5, scale * lo - eps,
             scale * lo - reach - tw < scale * bmin - 1e-9))
    for perp, a0, a1, leaves in ends:
        n = sum(1 for c in lines if a0 <= c <= a1)
        if leaves and (ext_index or {}).get((d.view, perp)):
            n += 1
        crossings.append(n)
    if crossings[0] != crossings[1]:
        return crossings[0] < crossings[1]
    return _outside_goes_high(scale * (lo - bmin), scale * (bmax - hi))


def _outside_text_offset(d: Dimension, geom: BBox2D, style: SheetStyle,
                         go_high: Optional[bool] = None) -> float:
    """``text_offset`` for a sheet-space linear dim whose value text does not fit its span.

    QA round 2 issue 17: a text wider than the gap between the extension
    lines is put outside, beyond the arrow (``span / 2 + arrow_length +
    SHORT_SPAN_EXTRA + width / 2`` from the centre, the dimension line
    extended under it), at the end with more room inside the view's extent
    (for a position dim that is the hole end, so the text does not sit on
    the overall dim's extension line at the view edge). The layout sets it
    explicitly so the room reserved by ``_short_span_room`` is exactly what
    is drawn, whatever default ``render_dimension`` uses. An offset already
    set is kept; 0 when the text fits.
    """
    if d.kind != "linear" or float(d.text_offset or 0.0) != 0.0:
        return float(d.text_offset or 0.0)
    a = math.radians(float(d.angle_deg))
    if abs(math.sin(a)) < 1e-6:
        axis = 0
    elif abs(math.cos(a)) < 1e-6:
        axis = 1
    else:
        return 0.0
    c1, c2 = float(d.p1[axis]), float(d.p2[axis])
    span = abs(c2 - c1)
    tw = TEXT_ADVANCE * style.text_height * len(d.text or "")
    if tw + 2.0 * style.dim_text_gap <= span:
        return 0.0
    if go_high is None:
        gmin, gmax = (geom.xmin, geom.xmax) if axis == 0 else (geom.ymin, geom.ymax)
        lo, hi = min(c1, c2), max(c1, c2)
        go_high = _outside_goes_high(lo - gmin, gmax - hi)
    mag = 0.5 * span + style.arrow_length + SHORT_SPAN_EXTRA + 0.5 * tw
    return mag if go_high == (c2 >= c1) else -mag


def _short_span_room(plan: DimensionPlan, boxes: Dict[str, BBox2D], style: SheetStyle,
                     labels: Optional[Dict[str, Callable[[float], str]]] = None
                     ) -> Callable[[float], Dict[str, Sides]]:
    """``scale -> {view: (left, bottom, right, top)}`` sheet mm the linear dims reach along their direction.

    Mirrors ``dimensions._render_linear``: a span shorter than
    ``2 * arrow_length + 0.5`` on the sheet gets its line extended by
    ``arrow_length + SHORT_SPAN_EXTRA`` at both ends; a value text that does
    not fit between the extension lines (width + 2 x ``dim_text_gap`` > span)
    is placed outside by ``compose`` (``_outside_text_offset``, QA round 2
    issue 17) and reaches ``arrow_length + SHORT_SPAN_EXTRA + width`` past
    the end with more room inside the view's extent. The reach is
    measured from the view's geometry box (the bands lie across the
    dimension direction and are reserved separately). ``labels``: view ->
    ``scale -> label text`` of a centred view label (A-A, ISO (1:n)); a label
    wider than the view reaches half the difference past its left and right.
    """
    L = float(style.arrow_length)
    h = float(style.text_height)
    lin = []
    ext_index = _ext_index(plan)
    for d, band in zip(plan.dims, _resolved_bands(plan)):
        if d.kind != "linear" or d.view not in boxes or d.p1 is None or d.p2 is None:
            continue
        axis = _linear_axis(d)
        if axis is None:
            continue                      # the plan only makes horizontal and vertical dims
        lo = min(float(d.p1[axis]), float(d.p2[axis]))
        hi = max(float(d.p1[axis]), float(d.p2[axis]))
        lin.append((d, band, d.view, axis, lo, hi, TEXT_ADVANCE * h * len(d.text or "")))

    def room(scale: float) -> Dict[str, Sides]:
        out: Dict[str, List[float]] = {}
        for d, band, view, axis, lo, hi, tw in lin:
            bb = boxes[view]
            span = scale * (hi - lo)
            bmin, bmax = (bb.xmin, bb.xmax) if axis == 0 else (bb.ymin, bb.ymax)
            room_lo, room_hi = scale * (lo - bmin), scale * (bmax - hi)
            reach_lo = reach_hi = L + SHORT_SPAN_EXTRA if span < 2.0 * L + 0.5 else 0.0
            go_high = _outside_high(d, band, bb, scale, style, ext_index)
            if go_high is not None:
                # the text goes outside, beyond the arrow (_outside_high / _outside_text_offset)
                out_reach = L + SHORT_SPAN_EXTRA + tw
                if go_high:
                    reach_hi = max(reach_hi, out_reach)
                else:
                    reach_lo = max(reach_lo, out_reach)
            if reach_lo <= 0.0 and reach_hi <= 0.0:
                continue
            need_lo = reach_lo - room_lo
            need_hi = reach_hi - room_hi
            e = out.setdefault(view, [0.0, 0.0, 0.0, 0.0])
            i_lo, i_hi = (0, 2) if axis == 0 else (1, 3)
            e[i_lo] = max(e[i_lo], need_lo)
            e[i_hi] = max(e[i_hi], need_hi)
        for view, text_at in (labels or {}).items():
            bb = boxes.get(view)
            if bb is None:
                continue
            over = 0.5 * (TEXT_ADVANCE * style.label_height * len(text_at(scale)) - scale * bb.width)
            if over > 0.0:
                e = out.setdefault(view, [0.0, 0.0, 0.0, 0.0])
                e[0] = max(e[0], over)
                e[2] = max(e[2], over)
        return {v: (e[0], e[1], e[2], e[3]) for v, e in out.items()}

    return room


# ---------------------------------------------------------------- compose --

def compose(views: Dict[str, ProjectedView], section: Optional[SectionView], plan: DimensionPlan,
            options: DrawingOptions, style: SheetStyle, meta: dict) -> LayoutResult:
    """Arrange views, section, dimensions, frame, title block and strip on one sheet."""
    meta = dict(meta or {})
    if abs(float(options.text_height) - style.text_height) > 1e-12:
        style = dataclasses.replace(style, text_height=float(options.text_height))
    boxes: Dict[str, BBox2D] = {}
    specs: Dict[str, ViewSpec] = {}
    for name, v in views.items():
        if v is None or not (name in PRINCIPAL or name == "iso"):
            if v is not None:
                log.warning("layout: view %r is not a known view name; skipped", name)
            continue
        bb = _view_bbox(v)
        if bb is None:
            log.info("layout: view %s is empty; skipped", name)
            continue
        boxes[name] = bb
        specs[name] = v.spec
    if section is not None:
        sb = _section_bbox(section)
        if sb is not None:
            boxes[SECTION_KEY] = sb
            specs[SECTION_KEY] = section.spec
        else:
            log.info("layout: the section has no loops and no body; not drawn")
    has_section = SECTION_KEY in boxes
    has_iso = "iso" in boxes
    marker = _marker_geometry(section, views, boxes) if has_section else None
    labels: Dict[str, Callable[[float], str]] = {}
    sec_letter = str(section.label) if section is not None else "A"
    if has_section:
        sec_label = f"{section.label}-{section.label}"
        labels[SECTION_KEY] = lambda s: sec_label
    if has_iso:
        labels["iso"] = lambda s: iso_label(s)

    def fit(size: SheetSize, p: DimensionPlan):
        counts_p = band_counts(p)
        extra = _extra_allowances(p, boxes, counts_p, marker, style)
        room = _max_room(_short_span_room(p, boxes, style, labels), _callout_room(p, boxes, style),
                         _marker_room(marker, p, boxes, style, sec_letter))
        return choose_scale(boxes, counts_p, size, style, options.projection, has_section, has_iso,
                            specs=specs, extra_mm=extra, min_mm_fn=room)

    # QA round 2 issue 7: when capping the position-dimension bands wins a scale step, the
    # excess position dims are dropped (undimensioned, reason layout_space)
    capped, dropped = (plan, [])
    if POSITION_BAND_CAP is not None:
        try:
            capped, dropped = _cap_bands(plan, boxes, int(POSITION_BAND_CAP))
        except Exception as exc:      # never lose the sheet over the cap
            log.warning("layout: band cap failed (%s)", exc)
            capped, dropped = plan, []

    def attempt(size: SheetSize, full=None):
        s_full, info_full = full if full is not None else fit(size, plan)
        if not dropped:
            return s_full, info_full, plan, []
        s_cap, info_cap = fit(size, capped)
        if s_cap is not None and (s_full is None or s_cap > s_full * (1.0 + 1e-9)):
            return s_cap, info_cap, capped, dropped
        return s_full, info_full, plan, []

    if options.sheet == "auto":
        # review R4: the sheet is chosen from the UNCAPPED plan (A4 when it holds every dim at
        # >= 1:1, else A3); the cap is tried on the chosen sheet only. Capping on A4 first kept
        # A4 by dropping dims that A3 shows in full, often at a larger scale.
        # When the cap wins a step on A4, A3 is still preferred if it shows every dim at that
        # scale or larger (measured: a 30 x 20 part with 8 position dims, A4 capped 2:1 with 5
        # dims dropped against A3 5:1 with all dims).
        names = ("A4", "A3")
        fulls: Dict[str, tuple] = {}
        pick = names[-1]
        for name in names:
            fulls[name] = fit(SHEETS[name], plan)
            if fulls[name][0] is not None and fulls[name][0] >= 1.0 - 1e-12:
                pick = name
                break
        size = SHEETS[pick]
        s, info, p_used, p_dropped = attempt(size, fulls[pick])
        if p_dropped and pick != names[-1]:
            big = names[-1]
            if big not in fulls:
                fulls[big] = fit(SHEETS[big], plan)
            s_big, info_big = fulls[big]
            if s_big is not None and s_big >= s * (1.0 - 1e-9):
                size, s, info, p_used, p_dropped = SHEETS[big], s_big, info_big, plan, []
        chosen = (size, s, info, p_used, p_dropped)
    else:
        size = SHEETS[options.sheet]
        s, info, p_used, p_dropped = attempt(size)
        chosen = (size, s, info, p_used, p_dropped)
    size, s, info, plan, dropped = chosen
    counts = band_counts(plan)
    overflow = s is None
    scale = float(info["scale"])
    stext = scale_text(scale)
    origins: Dict[str, Tuple[float, float]] = info.get("origins", {})
    rects: Dict[str, BBox2D] = info.get("rects", {})

    iso_scale = float(info.get("iso_scale") or scale)
    view_scale = {n: (iso_scale if n == "iso" else scale) for n in origins}

    sheet = Sheet(size.name, float(size.width), float(size.height))
    items = sheet.items
    items += frame_items(size, style)
    geoms = {n: _geom_box(boxes[n], origins[n], view_scale[n]) for n in origins}
    for name in origins:
        if name == SECTION_KEY:
            continue
        v = views[name]
        vs = view_scale[name]
        for p in v.visible:
            items.append(_restyle(transform_primitive(p, origins[name], vs), LineStyle.VISIBLE))
        if options.hidden_lines and name != "iso":
            for p in v.hidden:
                items.append(_restyle(transform_primitive(p, origins[name], vs), LineStyle.HIDDEN))
        if name == "iso":
            g = geoms[name]
            items.append(Text((0.5 * (g.xmin + g.xmax), g.ymax + LABEL_GAP), iso_label(vs),
                              style.label_height, anchor="middle"))
    section_stats: dict = {}
    marker_idx: List[int] = []
    if has_section:
        sec_items, section_stats = _section_items(section, origins[SECTION_KEY], scale, style, geoms[SECTION_KEY])
        items += sec_items
        if marker is not None:
            m_counts = counts.get(marker[0], (0, 0, 0, 0))
            mk = _marker_items(marker, origins[marker[0]], scale, style, section.label, m_counts,
                               _marker_past_sides(marker, plan, boxes, style, section.label, scale,
                                                  m_counts))
            marker_idx = list(range(len(items), len(items) + len(mk)))
            items += mk
    label_idx = [i for i, it in enumerate(items) if isinstance(it, Text) and i not in marker_idx]
    dimension_items: List[List[int]] = []
    ext_index = _ext_index(plan)
    for d, band in zip(plan.dims, _resolved_bands(plan)):
        if d.view not in origins or d.view == SECTION_KEY:
            log.debug("layout: dimension %r of undrawn view %r skipped", d.text, d.view)
            continue
        ds = _dim_to_sheet(d, band, origins[d.view], view_scale[d.view], geoms[d.view], style)
        if ds.kind == "linear":                                         # round 2 issue 17
            go_high = _outside_high(d, band, boxes[d.view], view_scale[d.view], style, ext_index)
            off = _outside_text_offset(ds, geoms[d.view], style, go_high) if go_high is not None else 0.0
            off = float(ds.text_offset or 0.0) or off
            if off != float(ds.text_offset or 0.0):
                ds = dataclasses.replace(ds, text_offset=off)
        rendered = render_dimension(ds, style)
        dimension_items.append(list(range(len(items), len(items) + len(rendered))))
        items += rendered
        sheet.dimensions.append(ds)
    try:
        collision_stats = _resolve_text_collisions(
            items, sheet, dimension_items, marker_idx, label_idx, geoms, rects,
            _frame_rect(size, style).expanded(-1.0), _obstacles(size, style), style)
    except Exception as exc:          # the pass only moves texts; never lose the sheet over it
        log.warning("layout: text collision pass failed (%s)", exc)
        collision_stats = {"error": f"{type(exc).__name__}: {exc}"}
    texts_idx = label_idx + [i for i in marker_idx if isinstance(items[i], Text)] + \
        [i for g in dimension_items for i in g if isinstance(items[i], Text)]
    # QA round 3 issue 8: texts crossed by another dimension's lines and marker items touching a
    # dimension line count as collisions too (the warning layout_text_collision)
    by_kind = _collision_lists(items, texts_idx, [i for i in marker_idx if not isinstance(items[i], Text)],
                               dimension_items, marker_idx)
    collisions = by_kind["texts"] + by_kind["text_line_crossings"] + by_kind["marker_dimension_crossings"]
    collision_stats["remaining"] = len(collisions)
    collision_stats["text_overlaps"] = len(by_kind["texts"])
    collision_stats["text_line_crossings"] = len(by_kind["text_line_crossings"])
    collision_stats["marker_dimension_crossings"] = len(by_kind["marker_dimension_crossings"])

    projection = "first_angle" if options.projection in ("first", "first_angle") else "third_angle"
    fields = {
        "document_type": meta.get("document_type", "Auto-generated drawing"),
        "document_status": meta.get("document_status", "Unchecked"),
        # spec 3.6 creator "surface_modeling drawing module" is split over two cells: the
        # 31 characters need 33.5 mm at the 1.8 mm minimum text height and the ISO 7200
        # "Created by" cell has 30 mm, so the full string would be truncated
        "creator": meta.get("creator", "drawing module"),
        "approval": meta.get("approval", ""),
        "title": options.title or meta.get("title") or meta.get("ident") or "",
        "ident": meta.get("ident", ""),
        "size": size.name,
        "owner": options.owner or meta.get("owner", ""),
        "supplementary_title": meta.get("supplementary_title", ""),
        "input_file": meta.get("input_file", ""),
        "generator": meta.get("generator", "surface_modeling"),
        "revision": meta.get("revision", "-"),
        "date": meta.get("date") or datetime.date.today().isoformat(),
        "language": meta.get("language", "en"),
        "sheet_number": meta.get("sheet_number", "1/1"),
        "department": meta.get("department", ""),
        "technical_reference": meta.get("technical_reference", ""),
        "classification": meta.get("classification", ""),
        "keywords": meta.get("keywords", ""),
    }
    items += title_block_items(size, style, fields)
    items += strip_items(size, style, stext, options.projection, str(meta.get("source_note", "")),
                         unit_note=meta.get("unit_note") or None)

    sheet.meta.update(meta)
    sheet.meta.update({"size": size.name, "orientation": size.orientation, "iso5457": size.iso5457,
                       "scale": scale, "scale_text": stext, "projection": projection,
                       "overflow": overflow, "dimension_items": dimension_items,
                       "title": fields["title"], "owner": fields["owner"],
                       "date": fields["date"], "section_layout": section_stats,
                       "iso_scale": (iso_scale if "iso" in origins else None),
                       "marker_view": marker[0] if marker is not None else None,
                       "text_collisions": collision_stats,
                       "scale_candidates": [(float(c[0]), bool(c[1]), str(c[2]))
                                            for c in info.get("scale_candidates", [])],
                       "section_mode": info.get("section_mode"),
                       "dims_dropped": [dict(u) for u in dropped]})
    # the planner's flags (dimension_extent_mismatch, QA issue 2) travel with the layout's, so the
    # pipeline's existing loop over LayoutResult.flags reports them with no extra step
    flags: List[Tuple[str, str, str]] = [tuple(f) for f in (getattr(plan, "flags", None) or [])]
    if not size.iso5457:
        flags.append(("sheet_not_iso5457", "warning",
                      f"{size.name} ({size.orientation}) is not an ISO 5457 sheet (A4 is portrait only)"))
    if overflow:
        flags.append(("views_overflow", "warning",
                      f"the views do not fit {size.name} even at {stext}"))
    if dropped:
        flags.append(("layout_dims_dropped", "info",
                      f"{len(dropped)} position dimension(s) dropped to keep {stext} (at most "
                      f"{POSITION_BAND_CAP} position bands per side; listed in undimensioned, reason layout_space)"))
    # QA round 2 issue 10: overlaps left after the resolution pass are a warning (overprinted
    # dimensions); texts the pass moved are reported as info
    if collisions:
        flags.append(("layout_text_collision", "warning",
                      f"{len(collisions)} text collision(s) left ({len(by_kind['texts'])} text overlap(s), "
                      f"{len(by_kind['text_line_crossings'])} text(s) crossed by dimension lines, "
                      f"{len(by_kind['marker_dimension_crossings'])} marker / dimension line clash(es)): "
                      + "; ".join(collisions[:5])))
    moved = int(collision_stats.get("linear_text_shifted", 0) or 0) + int(collision_stats.get("callouts_moved", 0) or 0)
    if moved:
        flags.append(("layout_text_moved", "info",
                      f"{moved} dimension text(s) moved off the section marker or other texts"))
    placements = {n: ((float(origins[n][0]), float(origins[n][1])), rects[n]) for n in origins}
    log.debug("layout: %s at %s, %d items, %d dims", size.name, stext, len(items), len(sheet.dimensions))
    return LayoutResult(sheet=sheet, scale=scale, scale_text=stext, sheet_size=size,
                        placements=placements, overflow=overflow, flags=flags, plan=plan)


__all__ = ["LayoutResult", "compose", "choose_scale", "grid_cells", "band_counts", "band_extent",
           "frame_items", "title_block_items", "strip_items", "projection_symbol", "marker_arrow_at",
           "SECTION_KEY", "text_collisions", "section_column_cells", "POSITION_BAND_CAP"]
