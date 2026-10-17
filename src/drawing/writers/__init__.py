"""Writers that render a ``Sheet`` (sheet mm, origin bottom-left, Y up).

* ``write_svg(sheet, path)`` -> ``{"bytes", "elements", "skipped"}``
* ``write_dxf(sheet, path)`` -> ``{"bytes", "entities", "layers", "dimensions",
  "dimensions_skipped", "dimension_items_replaced", "dimension_match",
  "audit_errors", "audit_fixes", "skipped"}`` (ezdxf imported lazily); it
  drops the rendered copies of each written dimension exactly when
  ``sheet.meta["dimension_items"]`` lists them (one list of item indices per
  ``sheet.dimensions`` entry), else by a tight geometric match
* ``write_pdf(sheet, path)`` -> ``{"bytes", "objects", "skipped"}``
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


def line_props(style: LineStyle, sheet_style: SheetStyle = DEFAULT_STYLE
               ) -> Tuple[float, Optional[Tuple[float, ...]]]:
    """(stroke width mm, dash pattern mm or None) for a ``LineStyle``."""
    s = sheet_style
    st = LineStyle(style)
    if st in (LineStyle.VISIBLE, LineStyle.SECTION):
        return s.wide, None
    if st == LineStyle.CUTTING:
        return s.wide, tuple(s.dash_cutting)
    if st == LineStyle.HIDDEN:
        return s.narrow, tuple(s.dash_hidden)
    if st == LineStyle.CENTER:
        return s.narrow, tuple(s.dash_center)
    if st == LineStyle.FRAME:
        return s.frame_width, None
    if st == LineStyle.TEXT:
        return s.text_stroke, None
    return s.narrow, None          # HATCH, DIMENSION, TITLE


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
           "line_props", "clean_text", "valid_item"]
