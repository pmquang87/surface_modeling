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
  ``section_dash_len`` (12 mm) at both ends, from 3 mm inside the view
  outline to 9 mm outside; arrows = continuous wide shaft (CUTTING style,
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
  reserve the label height above them; the marker view reserves 12 mm
  (plus the bands it has to clear, see below) on the sides the cutting line
  leaves through and 11 mm on the side the arrows point to (when the bands
  do not already cover it). Diameter callouts reserve, per scale, the
  exact reach of the rendered callout beyond the outline
  (``_callout_room``), taken as the MAX with the bands, not added to them
  (QA round 1 issue 7: the old fixed reach from the circle, added to the
  bands, cost about 25 mm per side; the synthetic plate on A4 fell from
  1:5 to 1:10 once bands appeared on four sides).
* Text collisions (``_resolve_text_collisions``, after the dimensions are
  rendered): a linear dimension text crossed by the section marker (wide
  dash, narrow connector, arrows, letters) is shifted along its line
  (``Dimension.text_offset``, QA issue 4); a callout whose text overlaps
  another text (``TEXT_CLEARANCE`` = 1 mm), sits on the marker, or whose
  leader runs through another text tries the four leader angles and
  longer leaders; the shortest leader that stays in its placement rect,
  off view outlines and off foreign dimension lines wins (QA issue 8).
  Texts that still overlap afterwards give the flag
  ``layout_text_collision`` (info); ``sheet.meta["text_collisions"]``
  counts the moves.
* Scale-dependent room (``min_mm_fn`` of ``choose_scale``): a linear
  dimension whose sheet span is shorter than two arrows + 0.5 mm is drawn
  with the arrows outside and its line ``arrow_length + 2`` mm longer at
  both ends (``dimensions._render_linear``); a value text wider than the
  span sticks out by half the difference. That reach is reserved along the
  dimension direction at each candidate scale.
* Section marker and dimension bands: when the side the cutting line leaves
  through carries dimension bands, the arrow and the letter move beyond the
  outermost band (``band_extent + MARKER_BAND_CLEAR``) so they do not sit
  on the first dimension line; the wide dash stays at the outline and a
  narrow long-dash-dot (CENTER) piece joins it to the arrow (ISO 128-40:
  the cutting plane is narrow 04.1 with wide ends).
* Iso view label: ``ISO (1:n)`` with the iso view's own scale (5 mm, spec
  3.6 "labelled with its scale") ``LABEL_GAP`` above the iso view.
* Grid: cells from ``projection.grid_cell`` (empty columns/rows are
  collapsed). The section view goes to the free middle-row cell on the side
  opposite the right-side view, else to the row below F, else above; the
  iso view to (1, 1), else the next free corner cell. Inside one column the
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
  with its holes), ``hatch_spacing`` of the section's shorter extent ON THE
  SHEET for every region it crosses with two or more lines, a finer
  ``style.region_hatch_spacing`` for narrower regions, and a filled
  ``Polygon`` for regions under ``HATCH_FILL_BELOW`` (2 mm); 45 degrees,
  phase-locked to the sheet origin (``_section_items``).
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
from .style import (HATCH_FILL_BELOW, SHEETS, STANDARD_SCALES, SheetSize, SheetStyle, hatch_spacing,
                    region_hatch_spacing, scale_text)

log = logging.getLogger(__name__)

SECTION_KEY = "section"
CLEARANCE = 5.0               # views stay this far from the frame and the title zone
MARK_CLEARANCE = 2.0          # views stay this far from the inner part of a centring mark
LEADER_EXIT_MARGIN = 3.0      # a callout leader ends this far outside the view outline
LABEL_GAP = 3.0               # A-A baseline above the section view
MARKER_INSIDE = 3.0           # cutting dash starts this far inside the outline
MARKER_ARROW_AT = 6.0         # arrow shaft this far outside the outline, along the line
MARKER_LETTER_AT = 4.0        # letter this far beyond the shaft, along the line
MARKER_ALONG = 12.0           # room the marker needs beyond the outline along the line (no bands)
MARKER_PERP = 11.0            # room on the side the arrows point to
MARKER_BAND_CLEAR = 2.0       # arrow shaft this far beyond the outermost dimension band
SHORT_SPAN_EXTRA = 2.0        # dimensions._render_linear extends a short span's line by L + this
ISO_SCALE_STEPS = 2           # a floating iso view may drop this many standard scales below the sheet's
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


# ------------------------------------------------------------------- grid --

def grid_cells(view_names: Sequence[str], projection: str, has_section: bool = False) -> Dict[str, Tuple[int, int]]:
    """(col, row) per view relative to front at (0, 0); adds ``section`` and moves ``iso`` when taken."""
    cells: Dict[str, Tuple[int, int]] = {}
    for name in view_names:
        if name in PRINCIPAL:
            cells[name] = grid_cell(name, projection)
    taken = set(cells.values())
    if has_section:
        side = grid_cell("right", projection)[0]
        other = -side
        for cand in ((other, 0), (side, 0), (other, -1), (side, -1), (other, 1), (side, 1),
                     (2 * other, 0), (2 * side, 0)):
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


def _grid_candidates(W: float, H: float, inner: BBox2D, tz: BBox2D) -> List[Tuple[float, float]]:
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
    nx = int(min(_SWEEP_STEPS, max(1, math.floor(xr / 2.5)))) if xr > _EPS else 0
    ny = int(min(_SWEEP_STEPS, max(1, math.floor(yr / 2.5)))) if yr > _EPS else 0
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
           float_size: Optional[Tuple[float, float]] = None):
    """``(dx, dy, float_ll)`` for the grid's bottom-left corner, or None when it does not fit.

    With ``float_size = (w, h)`` a floating rectangle (the iso view) must be
    placed too, at least ``view_gap`` clear of every grid rectangle;
    ``float_ll`` is its lower-left corner (None without a floating view).
    """
    inner = _frame_rect(sheet, style).expanded(-CLEARANCE)
    tz = _title_zone(sheet, style).expanded(CLEARANCE)
    obstacles = _obstacles(sheet, style)
    W, H = arr.width, arr.height
    if W > inner.width + _EPS or H > inner.height + _EPS:
        return None
    if float_size is not None and (float_size[0] > inner.width + _EPS or float_size[1] > inner.height + _EPS):
        return None
    rects = list(arr.rects.values())
    gap = style.view_gap

    def free(r: BBox2D) -> bool:
        return _inside(r, inner) and not any(_overlaps(r, o) for o in obstacles)

    spots = None
    if float_size is not None:
        spots = _float_spots(float_size[0], float_size[1], inner, obstacles)
    for dx, dy in _grid_candidates(W, H, inner, tz):
        shifted = [_shift(r, dx, dy) for r in rects]
        if not all(free(s) for s in shifted):
            continue
        if float_size is None:
            return dx, dy, None
        w, h = float_size
        grid = BBox2D(dx, dy, dx + W, dy + H)
        for fx, fy in _free_candidates(w, h, grid, inner, tz, gap):
            fr = BBox2D(fx, fy, fx + w, fy + h)
            if not free(fr):
                continue
            if any(_overlaps(fr.expanded(gap - 1e-6), s) for s in shifted):
                continue
            return dx, dy, (fx, fy)
        hit = _first_clear_spot(spots, w, h, shifted, gap - 1e-6)
        if hit is not None:
            return dx, dy, hit
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
    cells = grid_cells(names, projection, has_section=SECTION_KEY in boxes)
    cells_free = {n: c for n, c in cells.items() if n != "iso"}
    spec_map = {n: ((specs or {}).get(n) or _default_spec(n)) for n in cells}
    extra = extra_mm or {}
    base_sides: Dict[str, Sides] = {}
    for n in cells:
        k = bands.get(n, (0, 0, 0, 0))
        e = extra.get(n, (0.0, 0.0, 0.0, 0.0))
        base_sides[n] = tuple(band_extent(style, int(k[i])) + float(e[i]) for i in range(4))   # type: ignore[assignment]
    sides: Dict[str, Sides] = dict(base_sides)
    info: dict = {"cells": cells, "tried": [], "overflow": False, "sheet": sheet.name, "iso_mode": None}
    if not cells:
        info.update(scale=1.0, origins={}, rects={}, grid=(0.0, 0.0))
        return 1.0, info

    def set_sides(s: float) -> None:
        mins = min_mm_fn(float(s)) if min_mm_fn is not None else {}
        for n, bs in base_sides.items():
            m = mins.get(n)
            sides[n] = bs if m is None else tuple(max(float(bs[i]), float(m[i])) for i in range(4))  # type: ignore[assignment]

    def iso_rel(s: float) -> Tuple[float, float, float, float]:
        # room at the iso view's own scale (it may differ from the sheet scale)
        bb = boxes["iso"]
        bs = base_sides["iso"]
        m = (min_mm_fn(float(s)) if min_mm_fn is not None else {}).get("iso")
        l, b, r, t = bs if m is None else tuple(max(float(bs[i]), float(m[i])) for i in range(4))
        return (s * bb.xmin - l, s * bb.ymin - b, s * bb.xmax + r, s * bb.ymax + t)

    def accept(s, arr, pos, mode, iso_s):
        dx, dy, fll = pos
        origins = {n: (o[0] + dx, o[1] + dy) for n, o in arr.origins.items()}
        rects = {n: _shift(r, dx, dy) for n, r in arr.rects.items()}
        if fll is not None:
            rl = iso_rel(float(iso_s))
            origins["iso"] = (fll[0] - rl[0], fll[1] - rl[1])
            rects["iso"] = BBox2D(fll[0], fll[1], fll[0] + rl[2] - rl[0], fll[1] + rl[3] - rl[1])
        info.update(scale=s, grid=(arr.width, arr.height), origins=origins, rects=rects, iso_mode=mode,
                    cells=cells if mode != "free" else cells_free,
                    iso_scale=(iso_s if "iso" in origins else None))

    for idx, s in enumerate(STANDARD_SCALES):
        set_sides(s)
        arr = _arrange(boxes, cells, spec_map, sides, float(s), style.view_gap)
        pos = _place(arr, sheet, style)
        mode = "cell" if "iso" in cells else None
        iso_s = s
        if pos is None and "iso" in cells:
            # the iso view does not fit its grid cell: float it into free space, at the
            # sheet scale or up to ISO_SCALE_STEPS standard scales smaller (labelled)
            arr = _arrange(boxes, cells_free, spec_map, sides, float(s), style.view_gap)
            mode = "free"
            for iso_s in STANDARD_SCALES[idx: idx + 1 + ISO_SCALE_STEPS]:
                rl = iso_rel(float(iso_s))
                pos = _place(arr, sheet, style, float_size=(rl[2] - rl[0], rl[3] - rl[1]))
                if pos is not None:
                    break
        info["tried"].append((s, pos is not None))
        if pos is not None:
            accept(s, arr, pos, mode, iso_s)
            return s, info
    s = STANDARD_SCALES[-1]
    set_sides(s)
    arr = _arrange(boxes, cells, spec_map, sides, float(s), style.view_gap)
    inner = _frame_rect(sheet, style).expanded(-CLEARANCE)
    accept(s, arr, (inner.xmin, inner.ymax - arr.height, None), "cell" if "iso" in cells else None, s)
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


def strip_items(sheet: SheetSize, style: SheetStyle, scale_label: str, projection: str,
                source_note: str) -> List[SheetItem]:
    """180 x 9 strip above the title block: SCALE | projection symbol | Dimensions in mm | source."""
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
    for (c0, c1), txt in ((cells[0], f"SCALE {scale_label}"), (cells[2], "Dimensions in mm"),
                          (cells[3], source_note)):
        if not txt:
            continue
        t, h = _fit_text(txt, c1 - c0 - 2.0, style.text_height)
        items.append(Text((x0 + 0.5 * (c0 + c1), ym), t, h, anchor="middle", valign="middle"))
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


def _section_items(section: SectionView, origin: Point, scale: float, style: SheetStyle,
                   geom: BBox2D) -> Tuple[List[SheetItem], dict]:
    """Section outlines, hatch per region (outer loop with its holes), body edges, A-A label.

    The section spacing ``hatch_spacing`` of the shorter extent of all loops
    on the sheet is used for every region that it crosses with at least two
    lines (ISO 128-50: one part, one hatching). A region narrower than two
    spacings across the 45 degree lines gets ``region_hatch_spacing``
    (width / 3, at least 0.7 mm); a region whose smallest width (caliper
    width of its outer loop, ``_min_width``, so a thin strip at any angle
    counts, not only an axis-aligned one) is below
    ``HATCH_FILL_BELOW`` (2 mm on the sheet) is filled solid (ISO 128-50
    blackened thin section). QA round 1 issue 29: one spacing from the box of
    all loops left 4 mm regions with 0 or 1 line, so they read as holes.
    """
    items: List[SheetItem] = []
    stats = {"hatch_segments": 0, "hatch_spacing": None, "loops": 0, "hatch_spacings": [],
             "filled_regions": 0}
    off = np.array([float(origin[0]), float(origin[1])])
    loops = [np.asarray(lp, dtype=float).reshape(-1, 2) * scale + off for lp in section.loops if np.size(lp)]
    nested = nest_loops(loops) if loops else []
    ordered: List[np.ndarray] = []
    for outer, holes in nested:
        ordered.append(outer)
        ordered.extend(holes)
    for lp in ordered:
        items.append(Polyline(lp, True, LineStyle.SECTION))
    if ordered:
        bb = BBox2D.from_points(np.vstack(ordered))
        spacing = hatch_spacing(style, min(bb.width, bb.height))
        nrm = np.array([-math.sin(math.pi / 4.0), math.cos(math.pi / 4.0)])
        n_segs = 0
        for outer, holes in nested:
            ob = BBox2D.from_points(outer)
            if min(ob.width, ob.height, _min_width(outer)) < HATCH_FILL_BELOW:
                items.append(Polygon(outer.copy(), LineStyle.SECTION))
                stats["filled_regions"] += 1
                continue
            across = float(np.ptp(outer @ nrm))
            s = spacing if across >= 2.0 * spacing else region_hatch_spacing(spacing, across)
            region = [outer] + list(holes)
            try:
                segs = hatch_lines(region, s, 45.0)
            except Exception as exc:              # pragma: no cover - cap on absurd inputs
                log.warning("layout: hatch failed (%s)", exc)
                segs = np.zeros((0, 2, 2))
            items.append(HatchLines(segs, [lp.copy() for lp in region], LineStyle.HATCH))
            stats["hatch_spacings"].append(float(s))
            n_segs += int(len(segs))
        stats.update(hatch_segments=n_segs, hatch_spacing=spacing, loops=len(ordered))
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


def marker_arrow_at(style: SheetStyle, k: int) -> float:
    """Distance of the section arrow from the outline along the cutting line, given ``k`` bands there."""
    return MARKER_ARROW_AT if k <= 0 else band_extent(style, k) + MARKER_BAND_CLEAR


def _pt2(v) -> Point:
    return (float(v[0]), float(v[1]))


def _marker_items(geo, origin: Point, scale: float, style: SheetStyle, label: str,
                  counts: Tuple[int, int, int, int] = (0, 0, 0, 0)) -> List[SheetItem]:
    _, p2, d2, a2, (t0, s0, t1, s1) = geo
    off = np.array([float(origin[0]), float(origin[1])])
    items: List[SheetItem] = []
    dash_out = style.section_dash_len - MARKER_INSIDE
    for t, side, outward in ((t0, s0, -d2), (t1, s1, d2)):
        e = off + scale * (p2 + t * d2)
        a = e - MARKER_INSIDE * outward
        b = e + dash_out * outward
        items.append(Segment(_pt2(a), _pt2(b), LineStyle.CUTTING))
        at = marker_arrow_at(style, int(counts[SIDES.index(side)]))
        s = e + at * outward
        if at > dash_out + _EPS:          # arrow beyond the dimension bands: narrow chain joins it
            items.append(Segment(_pt2(b), _pt2(s), LineStyle.CENTER))
        tip = s + style.section_arrow_len * a2
        neck = tip - style.section_head_len * a2
        items.append(Segment(_pt2(s), _pt2(neck), LineStyle.CUTTING))
        items.append(arrowhead(tip, a2, style.section_head_len, style.section_head_angle_deg, LineStyle.CUTTING))
        c = s + MARKER_LETTER_AT * outward + 0.5 * style.section_arrow_len * a2
        items.append(Text(_pt2(c), label, style.label_height, anchor="middle", valign="middle"))
    return items


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
    (``allowed``, off ``blocked``); among those the one inside its view's
    placement rect, off every view outline and off other dimension lines,
    with the shortest leader, wins. ``items`` and ``sheet.dimensions`` are
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
        for it in rendered:          # own leader / shelf across another text
            if isinstance(it, Segment) and any(_segment_hits_box(it.a, it.b, o.expanded(-0.2)) for o in others):
                return True
        return False

    outlines = list(geoms.values())

    def callout_score(k: int, rendered: List[SheetItem], t: Text, others: List[BBox2D], rect, e: int, ai: int):
        """None when the placement is not allowed, else (out of rect, on outlines, on lines, e, ai)."""
        if callout_clash(k, rendered, t, others):
            return None
        tb = _box_of(t)
        if not _inside(tb, allowed) or any(_overlaps(tb, b) for b in blocked):
            return None
        boxes_r = [b for b in (_box_of(it) for it in rendered) if b is not None]
        out_rect = 0 if (rect is None or all(_inside(b, rect.expanded(1e-6)) for b in boxes_r)) else 1
        on_outline = sum(1 for o in outlines if _overlaps(tb, o))
        foreign = [items[i] for kk, segs in dim_segments.items() if kk != k for i in segs]
        on_lines = sum(1 for s in foreign if _segment_hits_box(s.a, s.b, tb))
        leader = next((it for it in rendered if isinstance(it, Segment)), None)
        length = math.hypot(leader.b[0] - leader.a[0], leader.b[1] - leader.a[1]) if leader is not None else 0.0
        return (out_rect, on_outline, on_lines, round(length, 3), ai)

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
            if now is not None and now[:3] == (0, 0, 0):
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
                if best is not None and best[0][:3] == (0, 0, 0) and extra > best[0][3]:
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


def text_collisions(items: Sequence[SheetItem], texts_idx: Sequence[int], marker_idx: Sequence[int]) -> List[str]:
    """Descriptions of every pair of overlapping texts and every text crossed by a marker item."""
    out: List[str] = []
    boxes = [(i, _box_of(items[i])) for i in texts_idx]
    boxes = [(i, b) for i, b in boxes if b is not None]
    for a in range(len(boxes)):
        ia, ba = boxes[a]
        for b in range(a + 1, len(boxes)):
            ib, bb = boxes[b]
            if _overlaps(ba, bb):
                out.append(f"{items[ia].text!r} overlaps {items[ib].text!r}")
        for m in marker_idx:
            if m != ia and _item_hits_box(items[m], ba):
                out.append(f"{items[ia].text!r} crossed by the section marker")
                break
    return out


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
    if marker is not None:
        name, _, _, a2, (_, s0, _, s1) = marker
        k = counts.get(name, (0, 0, 0, 0))
        e = extra.get(name, [0.0, 0.0, 0.0, 0.0])
        needs = [(side, MARKER_ALONG + marker_arrow_at(style, k[SIDES.index(side)]) - MARKER_ARROW_AT)
                 for side in (s0, s1)]
        needs.append((_dominant_side(a2), MARKER_PERP))
        for side, need in needs:
            i = SIDES.index(side)
            have = band_extent(style, k[i]) + e[i]
            if have < need:
                add(name, side, e[i] + (need - have))
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


def _short_span_room(plan: DimensionPlan, boxes: Dict[str, BBox2D], style: SheetStyle,
                     labels: Optional[Dict[str, Callable[[float], str]]] = None
                     ) -> Callable[[float], Dict[str, Sides]]:
    """``scale -> {view: (left, bottom, right, top)}`` sheet mm the linear dims reach along their direction.

    Mirrors ``dimensions._render_linear``: a span shorter than
    ``2 * arrow_length + 0.5`` on the sheet gets its line extended by
    ``arrow_length + SHORT_SPAN_EXTRA`` at both ends; a value text wider than
    the span reaches half the difference past each end. The reach is
    measured from the view's geometry box (the bands lie across the
    dimension direction and are reserved separately). ``labels``: view ->
    ``scale -> label text`` of a centred view label (A-A, ISO (1:n)); a label
    wider than the view reaches half the difference past its left and right.
    """
    L = float(style.arrow_length)
    h = float(style.text_height)
    lin = []
    for d in plan.dims:
        if d.kind != "linear" or d.view not in boxes or d.p1 is None or d.p2 is None:
            continue
        a = math.radians(float(d.angle_deg))
        if abs(math.sin(a)) < 1e-6:
            axis = 0
        elif abs(math.cos(a)) < 1e-6:
            axis = 1
        else:
            continue                      # the plan only makes horizontal and vertical dims
        lo = min(float(d.p1[axis]), float(d.p2[axis]))
        hi = max(float(d.p1[axis]), float(d.p2[axis]))
        lin.append((d.view, axis, lo, hi, TEXT_ADVANCE * h * len(d.text or "")))

    def room(scale: float) -> Dict[str, Sides]:
        out: Dict[str, List[float]] = {}
        for view, axis, lo, hi, tw in lin:
            bb = boxes[view]
            span = scale * (hi - lo)
            reach = max(L + SHORT_SPAN_EXTRA if span < 2.0 * L + 0.5 else 0.0, 0.5 * (tw - span))
            if reach <= 0.0:
                continue
            bmin, bmax = (bb.xmin, bb.xmax) if axis == 0 else (bb.ymin, bb.ymax)
            need_lo = reach - scale * (lo - bmin)
            need_hi = reach - scale * (bmax - hi)
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
    counts = band_counts(plan)
    extra = _extra_allowances(plan, boxes, counts, marker, style)
    labels: Dict[str, Callable[[float], str]] = {}
    if has_section:
        sec_label = f"{section.label}-{section.label}"
        labels[SECTION_KEY] = lambda s: sec_label
    if has_iso:
        labels["iso"] = lambda s: iso_label(s)
    short_room = _max_room(_short_span_room(plan, boxes, style, labels), _callout_room(plan, boxes, style))

    def attempt(size: SheetSize):
        return choose_scale(boxes, counts, size, style, options.projection, has_section, has_iso,
                            specs=specs, extra_mm=extra, min_mm_fn=short_room)

    if options.sheet == "auto":
        chosen = None
        for name in ("A4", "A3"):
            s, info = attempt(SHEETS[name])
            if s is not None and s >= 1.0 - 1e-12:
                chosen = (SHEETS[name], s, info)
                break
            chosen = (SHEETS[name], s, info)      # A3 is tried last and kept
    else:
        size = SHEETS[options.sheet]
        s, info = attempt(size)
        chosen = (size, s, info)
    size, s, info = chosen
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
            mk = _marker_items(marker, origins[marker[0]], scale, style, section.label,
                               counts.get(marker[0], (0, 0, 0, 0)))
            marker_idx = list(range(len(items), len(items) + len(mk)))
            items += mk
    label_idx = [i for i, it in enumerate(items) if isinstance(it, Text) and i not in marker_idx]
    dimension_items: List[List[int]] = []
    for d, band in zip(plan.dims, _resolved_bands(plan)):
        if d.view not in origins or d.view == SECTION_KEY:
            log.debug("layout: dimension %r of undrawn view %r skipped", d.text, d.view)
            continue
        ds = _dim_to_sheet(d, band, origins[d.view], view_scale[d.view], geoms[d.view], style)
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
    collisions = text_collisions(items, texts_idx, [i for i in marker_idx if not isinstance(items[i], Text)])
    collision_stats["remaining"] = len(collisions)

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
    items += strip_items(size, style, stext, options.projection, str(meta.get("source_note", "")))

    sheet.meta.update(meta)
    sheet.meta.update({"size": size.name, "orientation": size.orientation, "iso5457": size.iso5457,
                       "scale": scale, "scale_text": stext, "projection": projection,
                       "overflow": overflow, "dimension_items": dimension_items,
                       "title": fields["title"], "owner": fields["owner"],
                       "date": fields["date"], "section_layout": section_stats,
                       "iso_scale": (iso_scale if "iso" in origins else None),
                       "marker_view": marker[0] if marker is not None else None,
                       "text_collisions": collision_stats})
    # the planner's flags (dimension_extent_mismatch, QA issue 2) travel with the layout's, so the
    # pipeline's existing loop over LayoutResult.flags reports them with no extra step
    flags: List[Tuple[str, str, str]] = [tuple(f) for f in (getattr(plan, "flags", None) or [])]
    if not size.iso5457:
        flags.append(("sheet_not_iso5457", "warning",
                      f"{size.name} ({size.orientation}) is not an ISO 5457 sheet (A4 is portrait only)"))
    if overflow:
        flags.append(("views_overflow", "warning",
                      f"the views do not fit {size.name} even at {stext}"))
    if collisions:
        flags.append(("layout_text_collision", "info",
                      f"{len(collisions)} text collision(s) left: " + "; ".join(collisions[:5])))
    placements = {n: ((float(origins[n][0]), float(origins[n][1])), rects[n]) for n in origins}
    log.debug("layout: %s at %s, %d items, %d dims", size.name, stext, len(items), len(sheet.dimensions))
    return LayoutResult(sheet=sheet, scale=scale, scale_text=stext, sheet_size=size,
                        placements=placements, overflow=overflow, flags=flags)


__all__ = ["LayoutResult", "compose", "choose_scale", "grid_cells", "band_counts", "band_extent",
           "frame_items", "title_block_items", "strip_items", "projection_symbol", "marker_arrow_at",
           "SECTION_KEY", "text_collisions"]
