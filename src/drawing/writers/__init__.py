"""Writers that render a ``Sheet`` (sheet mm, origin bottom-left, Y up).

* ``write_svg(sheet, path)`` -> ``{"bytes", "elements", "skipped", "dash_adapted"}``
* ``write_dxf(sheet, path)`` -> ``{"bytes", "entities", "layers", "dimensions",
  "dimensions_skipped", "dimension_items_replaced", "dimension_match",
  "audit_errors", "audit_fixes", "skipped"}`` (ezdxf imported lazily); it
  drops the rendered copies of each written dimension exactly when
  ``sheet.meta["dimension_items"]`` lists them (one list of item indices per
  ``sheet.dimensions`` entry), else by a tight geometric match
* ``write_pdf(sheet, path)`` -> ``{"bytes", "objects", "skipped", "dash_adapted"}``
* ``helvetica_width(text, size)`` - Helvetica advance width in mm.

The helpers below are shared by the three writers and are defined BEFORE
the submodule imports at the bottom on purpose: ``svg.py``/``pdf.py``/
``dxf.py`` import them from this package while it is still initialising.

Shared rules (spec 3.7, writers probe):

* Line widths and dash patterns per ``LineStyle`` come from
  ``style.DEFAULT_STYLE`` (ISO 128 line group 0.5): wide 0.5 for visible,
  section and cutting lines, narrow 0.25 for hidden, centre, hatch,
  dimension and title lines, 0.7 for the frame. Dashes: hidden 3/0.75,
  centre 6/0.75/0.125/0.75, cutting 12/1.5/0.25/1.5 (mm).
* Short hidden elements (QA round 2 issue 2): the pattern starts at phase 0
  on every element, so a hidden element shorter than one dash printed
  solid. ``item_line_props`` fits the hidden pattern to the element length
  (``adapt_dash``: >= 2 dashes on an open element shorter than two
  periods, a whole number >= 4 of periods on a closed curve); the SVG and
  PDF write the fitted pattern, the DXF keeps its one ``HIDDEN_ISO``
  linetype and writes a per-entity ``ltscale`` (``hidden_ltscale``). Each
  writer returns the count as ``dash_adapted``.
* Joins are ROUND in the SVG and PDF (QA round 2 issue 20: the default
  miter join drew spikes at hairpin vertices); caps stay butt.
* U+2300 (diameter sign) is not in WinAnsi/cp1252 (measured: encode
  raises) and is mapped to U+00D8 ``Ø`` in every writer, so the three
  outputs carry the same text.
* Items with a non-finite coordinate or no extent (polyline < 2 points,
  radius <= 0, polygon < 3 points) are skipped and counted under
  ``"skipped"`` with a log warning instead of producing a broken file.
* ``clean_text`` also removes characters that XML 1.0 forbids (``\\x01``
  etc.; TAB/LF/CR become spaces), so a user title cannot break the SVG and
  the three formats carry the same string.
* Text size: ``Text.height`` is the em size (PDF/SVG font size); the DXF
  writes the cap height ``CAP_HEIGHT`` x height so all three show the same
  letters (ISO 3098 defines h as the cap height - open decision, see the
  DXF module).
"""
from __future__ import annotations

import logging
import math
from typing import Optional, Tuple

import numpy as np

from ..model import (Arc, Circle, HatchLines, LineStyle, Polygon, Polyline, Segment, Text)
from ..style import DEFAULT_STYLE, SheetStyle

log = logging.getLogger(__name__)

DIAMETER_SIGN = "⌀"

# Text.height is the font EM size in every writer (PDF Tf size, SVG
# font-size); the Helvetica capital height is CAP_HEIGHT x em. The DXF
# writer gives TEXT / dimension text the cap height (DXF TEXT height is the
# capital height), so the three formats show the same physical letters.
CAP_HEIGHT = 0.718

# Characters outside the XML 1.0 Char production (C0 controls other than
# TAB/LF/CR, surrogates, U+FFFE/U+FFFF) make an SVG unparsable; TAB/LF/CR
# and DEL have no meaning in a one-line drawing text and become spaces.
_DROP_CHARS = {c: None for c in list(range(0x00, 0x09)) + [0x0B, 0x0C]
               + list(range(0x0E, 0x20)) + list(range(0xD800, 0xE000)) + [0xFFFE, 0xFFFF]}
_DROP_CHARS.update({0x09: " ", 0x0A: " ", 0x0D: " ", 0x7F: None})


def clean_text(text) -> str:
    """Writer-safe text: U+2300 -> Ø, control/XML-illegal characters removed
    (TAB/LF/CR -> space); None -> ''."""
    return ("" if text is None else str(text)).replace(DIAMETER_SIGN, "Ø").translate(_DROP_CHARS)


def line_props(style: LineStyle, sheet_style: SheetStyle = DEFAULT_STYLE, *,
               length: Optional[float] = None, closed: bool = False
               ) -> Tuple[float, Optional[Tuple[float, ...]]]:
    """(stroke width mm, dash pattern mm or None) for a ``LineStyle``.

    ``length`` (sheet mm, optional) is the length of the element the pattern
    is drawn on and ``closed`` whether it is a closed curve; a HIDDEN pattern
    is then fitted to it with ``adapt_dash`` (QA round 2 issue 2). Without
    ``length`` the stock pattern is returned.
    """
    s = sheet_style
    st = LineStyle(style)
    if st in (LineStyle.VISIBLE, LineStyle.SECTION):
        return s.wide, None
    if st == LineStyle.CUTTING:
        return s.wide, tuple(s.dash_cutting)
    if st == LineStyle.HIDDEN:
        dash = tuple(s.dash_hidden)
        if length is not None:
            dash = adapt_dash(dash, length, closed)
        return s.narrow, dash
    if st == LineStyle.CENTER:
        return s.narrow, tuple(s.dash_center)
    if st == LineStyle.FRAME:
        return s.frame_width, None
    if st == LineStyle.TEXT:
        return s.text_stroke, None
    return s.narrow, None          # HATCH, DIMENSION, TITLE


# Short hidden elements (QA round 2 issue 2). Every writer starts the dash
# pattern at phase 0 on each element, so with the ISO 3/0.75 pattern an
# element shorter than one dash (blind-hole walls, thin-plate edges, R1 hidden
# circles: 2 pi r < 3.75) prints as a solid stroke. ``adapt_dash`` fits the
# pattern to the element instead: an open element shorter than two periods
# gets exactly two dashes and one gap (scaled 3:0.75 pattern), a closed curve
# a whole number of periods, at least four (so at least four gaps, and no
# merged dash at the closing point). Gaps never shrink below
# MIN_DASH_GAP (or a third of the element), or the narrow 0.25 mm line closes them.
MIN_DASH_GAP = 0.3          # sheet mm
_MIN_CLOSED_PERIODS = 4
# QA round 3 issue 34: ONE tolerance decides 'stock pattern kept' in every writer.
# A fit factor within 1 % of 1 keeps the stock pattern (PDF/SVG used 1e-9 on the
# dash values, the DXF 1 % on the scale: 124 vs 104 adapted on the 22 soup).
FIT_TOL = 0.01
# QA round 3 issue 23: hidden polylines are split at corners (turn above this)
# next to a corner run shorter than two periods but at least _MIN_SPLIT_RUN long
_CORNER_DEG = 30.0
_CORNER_MARGIN_DEG = 0.5    # a corner turns more than 30.5 deg: float noise at 30 deg is no coin toss
_MIN_SPLIT_RUN = 1.0        # sheet mm: shorter runs (mesh zig-zags) do not split
_RING_ROUND_REL = 0.02      # closed ring with vertices within 2 % of r of a circle: dashed whole


def _arc_min_periods(sweep_deg: float) -> int:
    """Whole periods an open hidden arc needs: at least 2 per 180 degrees (>= 1)."""
    return max(1, int(math.ceil(float(sweep_deg) / 90.0 - 1e-6)))


def adapt_dash(dash: Tuple[float, ...], length: float, closed: bool = False, *,
               arc_min_periods: Optional[int] = None) -> Tuple[float, ...]:
    """Dash pattern ``dash`` (dash, gap[, dash, gap ...], mm) fitted to an element.

    Open element (``closed`` False): unchanged when ``length`` >= two periods,
    else two dashes and one gap that fill ``length`` exactly. Closed curve:
    ``n = max(4, round(length / period))`` periods scaled to fill ``length``
    (unchanged when it already holds a whole number of at least 4). Open arc
    (``arc_min_periods`` given, QA round 3 issue 11): ``n = max(arc_min_periods,
    round((length - dash) / period))`` whole periods plus one closing dash,
    so the arc starts and ends on a dash like a circle has no seam (unchanged
    when the stock pattern already fits). Patterns with other than two
    entries are only scaled. Non-finite or non-positive lengths return the
    pattern unchanged.
    """
    dash = tuple(float(v) for v in dash)
    try:
        L = float(length)
    except (TypeError, ValueError):
        return dash
    period = sum(dash)
    if not (math.isfinite(L) and L > 0.0 and period > 0.0) or len(dash) < 2:
        return dash
    if arc_min_periods is not None and not closed:
        n = max(int(arc_min_periods), int(round((L - dash[0]) / period)))
        k = L / (n * period + dash[0])
        if abs(k - 1.0) <= 1e-6:
            return dash
        if len(dash) != 2:
            return tuple(v * k for v in dash)
        gap = min(max(dash[1] * k, MIN_DASH_GAP), L / (2 * n + 1))
        return ((L - n * gap) / (n + 1), gap)
    if closed:
        n = max(_MIN_CLOSED_PERIODS, int(round(L / period)))
        k = L / (n * period)
        if abs(k - 1.0) <= 1e-6:
            return dash
        if len(dash) != 2:
            return tuple(v * k for v in dash)
        p = L / n
        gap = min(max(dash[1] * k, min(MIN_DASH_GAP, p / 2.0)), p / 2.0)
        return (p - gap, gap)
    if L >= 2.0 * period:
        return dash
    if len(dash) != 2:
        k = L / (2.0 * dash[0] + dash[1]) if len(dash) >= 2 else 1.0
        return tuple(v * k for v in dash)
    k = L / (2.0 * dash[0] + dash[1])                    # two dashes + one gap
    gap = max(dash[1] * k, min(dash[1], MIN_DASH_GAP, L / 3.0))
    return (0.5 * (L - gap), gap)


def item_length(item) -> Tuple[Optional[float], bool]:
    """(length in the item's units, closed) of a stroked item; (None, False) otherwise.

    Segment: chord; Polyline: sum of its edges (closing edge when closed; a
    repeated closing point adds nothing); Circle: 2 pi r; Arc: r x sweep (a
    full-turn Arc is closed).
    """
    try:
        if isinstance(item, Segment):
            return float(math.dist(tuple(map(float, item.a)), tuple(map(float, item.b)))), False
        if isinstance(item, Polyline):
            pts = np.asarray(item.points, dtype=np.float64).reshape(-1, 2)
            if len(pts) < 2:
                return None, False
            closed = bool(item.closed)
            ring = np.vstack([pts, pts[:1]]) if closed else pts
            return float(np.linalg.norm(np.diff(ring, axis=0), axis=1).sum()), closed
        if isinstance(item, Circle):
            return 2.0 * math.pi * abs(float(item.radius)), True
        if isinstance(item, Arc):
            span = arc_span(item.start_deg, item.end_deg)
            return abs(float(item.radius)) * math.radians(span), span >= 360.0 - 1e-9
    except (TypeError, ValueError):
        return None, False
    return None, False


def _arc_sweep_of(item) -> Optional[float]:
    """Sweep in degrees of an open Arc (None for anything else or a full turn)."""
    if isinstance(item, Arc):
        try:
            span = arc_span(item.start_deg, item.end_deg)
        except (TypeError, ValueError):
            return None
        return None if span >= 360.0 - 1e-9 else span
    return None


def hidden_fit(item, sheet_style: SheetStyle = DEFAULT_STYLE
               ) -> Optional[Tuple[Tuple[float, ...], float]]:
    """``(dash pattern, scale)`` fitting the hidden pattern to ``item``, or None.

    The one decision the three writers share (QA round 3 issue 34): None
    when ``item`` is not HIDDEN, has no length, or the stock pattern is kept
    - an open element of at least two periods, or a fit factor within
    ``FIT_TOL`` (1 %) of 1. Otherwise the pattern (``adapt_dash``; SVG and
    PDF write it) and the uniform scale of the stock pattern that gives the
    same number of dashes (the DXF entity ``ltscale``). Rules: an open
    element shorter than two periods gets two dashes and one gap; a closed
    curve (Circle, full Arc, closed Polyline) a whole number >= 4 of
    periods; an open Arc (QA round 3 issue 11) whole periods plus a closing
    dash, at least 2 periods per 180 degrees.
    """
    try:
        if LineStyle(item.style) != LineStyle.HIDDEN:
            return None
    except (ValueError, AttributeError):
        return None
    L, closed = item_length(item)
    stock = tuple(float(v) for v in sheet_style.dash_hidden)
    period = sum(stock)
    if L is None or not (L > 0.0 and math.isfinite(L)) or period <= 0.0 or len(stock) < 2:
        return None
    sweep = None if closed else _arc_sweep_of(item)
    if closed:
        n = max(_MIN_CLOSED_PERIODS, int(round(L / period)))
        k = L / (n * period)
        pattern = adapt_dash(stock, L, True)
    elif sweep is not None:
        n_min = _arc_min_periods(sweep)
        n = max(n_min, int(round((L - stock[0]) / period)))
        k = L / (n * period + stock[0])
        pattern = adapt_dash(stock, L, False, arc_min_periods=n_min)
    elif L >= 2.0 * period:
        return None
    else:
        k = L / (2.0 * stock[0] + stock[1])
        pattern = adapt_dash(stock, L, False)
    if abs(k - 1.0) <= FIT_TOL:
        return None
    return tuple(float(v) for v in pattern), float(k)


def item_line_props(item, sheet_style: SheetStyle = DEFAULT_STYLE
                    ) -> Tuple[float, Optional[Tuple[float, ...]], bool]:
    """(width, dash, adapted) for one stroked item: ``line_props`` with the item's
    own length (``hidden_fit``); ``adapted`` True when the pattern was fitted,
    which is exactly when ``hidden_ltscale`` returns a factor."""
    width, stock = line_props(item.style, sheet_style)
    if stock is None or LineStyle(item.style) != LineStyle.HIDDEN:
        return width, stock, False
    fit = hidden_fit(item, sheet_style)
    if fit is None:
        return width, stock, False
    return width, fit[0], True


def hidden_ltscale(item, sheet_style: SheetStyle = DEFAULT_STYLE) -> Optional[float]:
    """Per-entity DXF linetype scale that fits the hidden pattern to ``item``.

    DXF has one pattern per linetype; an entity's ``ltscale`` (group 48)
    scales it. Returns the factor for which the ``HIDDEN_ISO`` pattern
    (``sheet_style.dash_hidden`` at ``$LTSCALE`` 1) shows two dashes on a
    short open element, a whole number (>= 4) of periods on a closed curve,
    or whole periods plus a closing dash on an open arc; None when the stock
    pattern is kept (``hidden_fit``: not HIDDEN, long element, or a factor
    within ``FIT_TOL`` of 1).
    """
    fit = hidden_fit(item, sheet_style)
    return None if fit is None else fit[1]


def _is_round_ring(pts: np.ndarray) -> bool:
    """True when the closed ring ``pts`` (no repeated closing point) is a sampled
    circle: vertices within ``_RING_ROUND_REL`` x r of one circle, one turn."""
    try:
        from ..geometry2d import _ring_circle
        ext = float(np.max(pts.max(axis=0) - pts.min(axis=0)))
        if not ext > 0.0:
            return False
        fit = _ring_circle(pts, _RING_ROUND_REL * 0.5 * ext, chord_check=False)
        return fit is not None and float(np.max(np.abs(
            np.hypot(pts[:, 0] - fit[0][0], pts[:, 1] - fit[0][1]) - fit[1]))) <= _RING_ROUND_REL * fit[1]
    except Exception:
        return False


def hidden_pieces(item, sheet_style: SheetStyle = DEFAULT_STYLE) -> list:
    """The pieces a writer draws for ``item`` (QA round 3 issue 23).

    A HIDDEN Polyline is split at its corner vertices (turn above 30
    degrees) next to a corner run - the length between two corners -
    shorter than two dash periods and at least ``_MIN_SPLIT_RUN`` (1 mm).
    Each piece is an open Polyline (or a Segment for one leg) that the
    writers fit on its own (``hidden_fit``), so every such leg shows at
    least one gap: one pattern over the whole 11.9 mm counterbore U of the
    18 plate covered its 3.18 mm first leg with the first 3 mm dash. Long
    straight runs, gently curved chains and mesh zig-zags with legs under
    1 mm stay one piece. Any other item comes back as ``[item]``.

    Z2 review: a closed ring of 8+ vertices lying on one circle (within 2 %
    of its radius, ``geometry2d._ring_circle`` without the chord check) is a
    sampled circle - a coarse mesh hole rim - and stays whole, so it gets
    the whole-period closed-curve pattern instead of one dash pair per facet;
    a corner turns more than 30.5 degrees (``_CORNER_MARGIN_DEG``), so equal
    vertices of a regular 12-gon are not split by float noise at 30 degrees.
    """
    try:
        if not isinstance(item, Polyline) or LineStyle(item.style) != LineStyle.HIDDEN:
            return [item]
        pts = np.asarray(item.points, dtype=np.float64).reshape(-1, 2)
    except (TypeError, ValueError):
        return [item]
    closed = bool(item.closed)
    if len(pts) < 3 or not np.all(np.isfinite(pts)):
        return [item]
    pts = pts[np.r_[True, np.any(np.diff(pts, axis=0) != 0.0, axis=1)]]   # repeated points
    if closed and len(pts) >= 2 and np.all(pts[0] == pts[-1]):
        pts = pts[:-1]
    n = len(pts)
    if n < 3:
        return [item]
    if closed and n >= 8 and _is_round_ring(pts):
        return [item]
    period = float(sum(float(v) for v in sheet_style.dash_hidden))
    ring = np.vstack([pts, pts[:1]]) if closed else pts
    leg = np.diff(ring, axis=0)
    leg_len = np.hypot(leg[:, 0], leg[:, 1])
    # turn angle at every vertex that has two legs
    if closed:
        a_in, a_out = np.roll(leg, 1, axis=0), leg                    # vertex i: legs i-1, i
        verts = np.arange(n)
    else:
        a_in, a_out = leg[:-1], leg[1:]                                # vertex i+1
        verts = np.arange(1, n - 1)
    cosang = np.einsum("ij,ij->i", a_in, a_out) / (np.hypot(a_in[:, 0], a_in[:, 1])
                                                   * np.hypot(a_out[:, 0], a_out[:, 1]))
    corner = verts[cosang < math.cos(math.radians(_CORNER_DEG + _CORNER_MARGIN_DEG))]
    if len(corner) == 0:
        return [item]
    cum = np.r_[0.0, np.cumsum(leg_len)]                               # arc length at vertex i
    total = float(cum[-1])
    if closed:
        cs = np.sort(corner)
        runs_after = np.r_[np.diff(cum[cs]), total - cum[cs[-1]] + cum[cs[0]]]
        runs_before = np.roll(runs_after, 1)
    else:
        cs = np.sort(corner)
        bounds = np.r_[0.0, cum[cs], total]
        runs = np.diff(bounds)
        runs_before, runs_after = runs[:-1], runs[1:]
    short = np.minimum(runs_before, runs_after)
    split = cs[(short < 2.0 * period) & (short >= _MIN_SPLIT_RUN)]
    if len(split) == 0 or (closed and len(split) < 1):
        return [item]
    if closed:
        order = np.r_[np.arange(split[0], n), np.arange(0, split[0])]
        seq = np.vstack([pts[order], pts[split[0]][None, :]])         # open ring from a split corner
        pos = {int(v): k for k, v in enumerate(order)}
        cuts = sorted(pos[int(v)] for v in split) + [n]
    else:
        seq = pts
        cuts = [0] + sorted(int(v) for v in split) + [n - 1]
    out = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        piece = seq[a:b + 1]
        if len(piece) == 2:
            out.append(Segment((float(piece[0, 0]), float(piece[0, 1])),
                               (float(piece[1, 0]), float(piece[1, 1])), item.style))
        elif len(piece) > 2:
            out.append(Polyline(np.ascontiguousarray(piece), False, item.style))
    return out or [item]


def _finite(arr) -> bool:
    a = np.asarray(arr, dtype=np.float64)
    return a.size > 0 and bool(np.all(np.isfinite(a)))


def valid_item(item) -> bool:
    """True when the item can be written (finite, non-degenerate)."""
    try:
        if isinstance(item, Segment):
            return _finite([item.a, item.b])
        if isinstance(item, Polyline):
            pts = np.asarray(item.points, dtype=np.float64).reshape(-1, 2)
            return pts.shape[0] >= 2 and _finite(pts)
        if isinstance(item, Polygon):
            pts = np.asarray(item.points, dtype=np.float64).reshape(-1, 2)
            return pts.shape[0] >= 3 and _finite(pts)
        if isinstance(item, (Circle, Arc)):
            ok = _finite(item.center) and math.isfinite(float(item.radius)) and float(item.radius) > 0
            if ok and isinstance(item, Arc):
                ok = math.isfinite(float(item.start_deg)) and math.isfinite(float(item.end_deg))
            return ok
        if isinstance(item, HatchLines):
            segs = np.asarray(item.segments, dtype=np.float64)
            if segs.size == 0:
                return False
            return _finite(segs) and all(_finite(lp) for lp in item.loops)
        if isinstance(item, Text):
            return (_finite(item.position) and math.isfinite(float(item.height))
                    and float(item.height) > 0 and math.isfinite(float(item.rotation_deg))
                    and clean_text(item.text) != "")
    except (TypeError, ValueError):
        return False
    return False


def arc_span(start_deg: float, end_deg: float) -> float:
    """CCW sweep in degrees, (0, 360]; equal angles mean a full turn."""
    span = (float(end_deg) - float(start_deg)) % 360.0
    return span if span > 1e-9 else 360.0


def fmt(v: float) -> str:
    """Fixed 3-decimal number without trailing zeros; never '-0'."""
    s = f"{float(v):.3f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "", "-") else s


from .pdf import helvetica_width, write_pdf, PdfWriter  # noqa: E402
from .svg import write_svg  # noqa: E402
from .dxf import write_dxf  # noqa: E402

__all__ = ["write_svg", "write_dxf", "write_pdf", "PdfWriter", "helvetica_width",
           "line_props", "clean_text", "valid_item", "adapt_dash", "item_length",
           "item_line_props", "hidden_ltscale", "hidden_fit", "hidden_pieces", "FIT_TOL"]
