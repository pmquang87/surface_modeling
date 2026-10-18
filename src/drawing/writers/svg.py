"""SVG writer for a ``Sheet`` (plain XML, no dependency).

Layout of the file (pattern measured in the writers probe):

* ``viewBox="0 0 W H"``, ``width="Wmm" height="Hmm"``, a white background
  ``<rect>``;
* all geometry inside ONE ``<g transform="translate(0,H) scale(1,-1)">``
  group, so sheet coordinates (Y up) are written unchanged; arcs are
  ``A r r 0 large 1`` (sweep 1 = CCW in the flipped group), full circles are
  ``<circle>`` (a 360 degree ``A`` degenerates);
* text OUTSIDE that group at ``y' = H - y`` (text inside a flipped group
  renders upside down - confirmed on a PNG), rotated with
  ``rotate(-angle x y)``; vertical centring by a manual baseline shift of
  0.718·size/2 along the text's own down direction, because MuPDF ignores
  ``dominant-baseline`` (measured +1.80 mm error on 5 mm text);
* every drawn element carries its own ``stroke-width`` / ``stroke-dasharray``
  in mm and ``class="<LineStyle value>"`` (no inheritance), which is what the
  tests and the report check - MuPDF ignores ``stroke-dasharray`` in every
  form (measured), so dashes are verified by XML attribute, not by render;
  a hidden element's dasharray is fitted to its own length
  (``item_line_props``, QA round 2 issue 2);
* the group sets ``stroke-linejoin="round"`` (QA round 2 issue 20: miter
  joins drew spikes at hairpin vertices) and butt caps.

Text keeps UTF-8 ``Ø ± °``; U+2300 is mapped to ``Ø`` like in the other
writers. Font size equals ``Text.height`` (same convention as the PDF
writer). The output is deterministic (no timestamps).
"""
from __future__ import annotations

import logging
import math
from typing import List, Optional
from xml.sax.saxutils import escape

import numpy as np

from ..model import (Arc, Circle, HatchLines, Polygon, Polyline, Segment, Sheet, Text)
from ..style import DEFAULT_STYLE, SheetStyle
from . import arc_span, clean_text, fmt, hidden_pieces, item_line_props, line_props, valid_item

log = logging.getLogger(__name__)

HELV_CAP = 0.718
FONT_FAMILY = "Helvetica, Arial, sans-serif"


def _stroke(item, sheet_style: SheetStyle, counter: Optional[list] = None) -> str:
    if isinstance(item, HatchLines):
        width, dash = line_props(item.style, sheet_style)
        adapted = False
    else:
        width, dash, adapted = item_line_props(item, sheet_style)
    if adapted and counter is not None:
        counter[0] += 1
    s = f'class="{item.style.value}" stroke="#000" stroke-width="{fmt(width)}" fill="none"'
    if dash:
        s += ' stroke-dasharray="' + " ".join(fmt(v) for v in dash) + '"'
    return s


def polyline_points(item) -> np.ndarray:
    """(n, 2) points of a Polyline; a closed one drops a repeated closing point.

    ``<polygon>`` closes itself, so writing the first point again at the end
    would add a zero-length closing segment.
    """
    pts = np.asarray(item.points, dtype=np.float64).reshape(-1, 2)
    if bool(item.closed) and pts.shape[0] >= 4 and bool(np.all(np.abs(pts[0] - pts[-1]) <= 1e-9)):
        pts = pts[:-1]
    return pts


def _pts(arr) -> str:
    pts = np.asarray(arr, dtype=np.float64).reshape(-1, 2)
    return " ".join(f"{fmt(x)},{fmt(y)}" for x, y in pts)


def _geom(item, sheet_style: SheetStyle, counter: Optional[list] = None) -> str:
    if isinstance(item, Polygon):
        return (f'<polygon class="{item.style.value}" points="{_pts(item.points)}" '
                f'fill="#000" stroke="none"/>')
    st = _stroke(item, sheet_style, counter)
    if isinstance(item, Segment):
        (x0, y0), (x1, y1) = item.a, item.b
        return f'<line x1="{fmt(x0)}" y1="{fmt(y0)}" x2="{fmt(x1)}" y2="{fmt(y1)}" {st}/>'
    if isinstance(item, Polyline):
        tag = "polygon" if item.closed else "polyline"
        return f'<{tag} points="{_pts(polyline_points(item))}" {st}/>'
    if isinstance(item, Circle):
        cx, cy = item.center
        return f'<circle cx="{fmt(cx)}" cy="{fmt(cy)}" r="{fmt(item.radius)}" {st}/>'
    if isinstance(item, Arc):
        cx, cy = float(item.center[0]), float(item.center[1])
        r = float(item.radius)
        span = arc_span(item.start_deg, item.end_deg)
        if span >= 360.0 - 1e-9:
            return f'<circle cx="{fmt(cx)}" cy="{fmt(cy)}" r="{fmt(r)}" {st}/>'
        t0 = math.radians(float(item.start_deg))
        t1 = math.radians(float(item.start_deg) + span)
        x0, y0 = cx + r * math.cos(t0), cy + r * math.sin(t0)
        x1, y1 = cx + r * math.cos(t1), cy + r * math.sin(t1)
        large = 1 if span > 180.0 else 0
        return (f'<path d="M {fmt(x0)} {fmt(y0)} A {fmt(r)} {fmt(r)} 0 {large} 1 '
                f'{fmt(x1)} {fmt(y1)}" {st}/>')
    if isinstance(item, HatchLines):
        segs = np.asarray(item.segments, dtype=np.float64).reshape(-1, 2, 2)
        d = " ".join(f"M {fmt(a[0])} {fmt(a[1])} L {fmt(b[0])} {fmt(b[1])}" for a, b in segs)
        return f'<path d="{d}" {st}/>'
    raise TypeError(f"svg: unsupported item {type(item).__name__}")


def _text(item: Text, height_sheet: float) -> str:
    size = float(item.height)
    rot = float(item.rotation_deg)
    X, Y = float(item.position[0]), height_sheet - float(item.position[1])
    dy = HELV_CAP * size / 2.0 if item.valign == "middle" else 0.0
    a = math.radians(rot)
    # in y-down screen coordinates the text's 'down' direction after rotate(-rot) is (sin a, cos a)
    X, Y = X + math.sin(a) * dy, Y + math.cos(a) * dy
    anchor = item.anchor if item.anchor in ("start", "middle", "end") else "middle"
    tr = f' transform="rotate({fmt(-rot)} {fmt(X)} {fmt(Y)})"' if abs(rot) > 1e-12 else ""
    return (f'<text class="{item.style.value}" x="{fmt(X)}" y="{fmt(Y)}" '
            f'font-family="{FONT_FAMILY}" font-size="{fmt(size)}" text-anchor="{anchor}" '
            f'fill="#000"{tr}>{escape(clean_text(item.text))}</text>')


def write_svg(sheet: Sheet, path: str, sheet_style: SheetStyle = DEFAULT_STYLE) -> dict:
    """Write ``sheet`` as SVG.

    Returns ``{"bytes": n, "elements": n, "skipped": n, "dash_adapted": n}``
    (``dash_adapted``: hidden elements with a length-fitted dasharray); ``elements`` counts
    the drawn elements (line, polyline, polygon, circle, path, text), not the
    root, the background rect or the group.
    """
    W, H = float(sheet.width), float(sheet.height)
    geom: List[str] = []
    texts: List[str] = []
    skipped = 0
    adapted = [0]
    for item in sheet.items:
        if not valid_item(item):
            skipped += 1
            continue
        if isinstance(item, Text):
            texts.append(_text(item, H))
        else:
            # hidden polylines with short corner legs: one element per piece (QA3 #23)
            for piece in hidden_pieces(item, sheet_style):
                geom.append(_geom(piece, sheet_style, adapted))
    if skipped:
        log.warning("svg: skipped %d degenerate or non-finite items", skipped)
    Ws, Hs = fmt(W), fmt(H)
    out = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{Ws}mm" height="{Hs}mm" '
        f'viewBox="0 0 {Ws} {Hs}">',
        f'<rect x="0" y="0" width="{Ws}" height="{Hs}" fill="#fff" stroke="none"/>',
        f'<g transform="translate(0,{Hs}) scale(1,-1)" stroke-linecap="butt" '
        f'stroke-linejoin="round">',
        *geom,
        "</g>",
        *texts,
        "</svg>",
        "",
    ]
    data = "\n".join(out).encode("utf-8")
    with open(path, "wb") as fh:
        fh.write(data)
    return {"bytes": len(data), "elements": len(geom) + len(texts), "skipped": skipped,
            "dash_adapted": adapted[0]}
