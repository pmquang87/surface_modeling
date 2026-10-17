"""Dependency-free vector PDF writer for a ``Sheet`` (port of the probe's minipdf).

One page, one FlateDecode content stream, base-14 Helvetica with
``/WinAnsiEncoding`` (no font embedding). Measured in the writers probe
(PyMuPDF 1.28 / MuPDF 1.29 as the independent reader):

* the file opens with ``is_repaired=False`` and no MuPDF warnings; page size
  reads back exactly (A4 297 x 210, A3 420 x 297 mm);
* widths and dashes read back as written (0.5 mm -> 1.417 pt, 0.25 mm ->
  0.709 pt, 3 mm dash -> 8.504 pt);
* the Helvetica AFM width table below matches MuPDF's ``helv`` for all 100
  characters (ASCII 32..126 plus Ø ° ± × µ), so text centring is exact
  (0.00 mm error on 'SECTION A-A');
* text is cp1252 bytes with ``\\ ( )`` escaped; ``Ø ± °`` encode, U+2300 does
  not and is mapped to ``Ø`` (``clean_text``); other characters outside
  cp1252 become ``?`` (logged);
* arcs are cubic Béziers of <= 90 degrees with k = 4/3·tan(span/4).

Text size convention: the PDF font size equals ``Text.height`` (the probe's
convention, which the layout's width estimate assumes). Vertical centring
(``valign="middle"``) uses the Helvetica cap height 0.718·size.

The output is deterministic: no date or ID is written, so identical sheets
give byte-identical files.
"""
from __future__ import annotations

import logging
import math
import zlib
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..model import (Arc, Circle, HatchLines, LineStyle, Polygon, Polyline, Segment, Sheet, Text)
from ..style import DEFAULT_STYLE, SheetStyle
from . import arc_span, clean_text, fmt, line_props, valid_item

log = logging.getLogger(__name__)

MM = 72.0 / 25.4  # points per mm

# Helvetica AFM advance widths (1/1000 em), printable ASCII 32..126
_HELV_ASCII = [
    278, 278, 355, 556, 556, 889, 667, 191, 333, 333, 389, 584, 278, 333, 278, 278,  # 32-47
    556, 556, 556, 556, 556, 556, 556, 556, 556, 556, 278, 278, 584, 584, 584, 556,  # 48-63
    1015, 667, 667, 722, 722, 667, 611, 778, 722, 278, 500, 667, 556, 833, 722, 778,  # 64-79
    667, 778, 722, 667, 611, 722, 667, 944, 667, 667, 611, 278, 278, 278, 469, 556,  # 80-95
    333, 556, 556, 500, 556, 556, 278, 556, 556, 222, 222, 500, 222, 833, 556, 556,  # 96-111
    556, 556, 333, 500, 278, 556, 500, 722, 500, 500, 500, 334, 260, 334, 584,       # 112-126
]
HELV_W = {chr(32 + i): w for i, w in enumerate(_HELV_ASCII)}
HELV_W.update({"Ø": 778, "°": 400, "±": 584, "×": 584, "µ": 556})
HELV_CAP = 0.718      # cap height / em
_DEFAULT_W = 556      # width used for characters missing from the table


def helvetica_width(text: str, size: float) -> float:
    """Advance width of ``text`` in Helvetica at font size ``size`` (same unit)."""
    return sum(HELV_W.get(c, _DEFAULT_W) for c in clean_text(text)) / 1000.0 * float(size)


def _esc(text: str) -> bytes:
    raw = clean_text(text)
    b = raw.encode("cp1252", errors="replace")
    if b"?" in b and raw.count("?") != b.count(b"?"):
        log.warning("pdf: characters outside cp1252 replaced by '?' in %r", raw)
    return b.replace(b"\\", b"\\\\").replace(b"(", b"\\(").replace(b")", b"\\)")


def arc_beziers(cx: float, cy: float, r: float, a0_deg: float, span_deg: float):
    """CCW arc from ``a0_deg`` over ``span_deg`` as <= 90 degree cubic Béziers.

    Returns ``(start, [(c1, c2, p), ...])``.
    """
    a0 = math.radians(a0_deg)
    total = math.radians(span_deg)
    n = max(1, math.ceil(total / (math.pi / 2) - 1e-9))
    d = total / n
    k = 4.0 / 3.0 * math.tan(d / 4.0)
    start = (cx + r * math.cos(a0), cy + r * math.sin(a0))
    segs = []
    for i in range(n):
        t0, t1 = a0 + i * d, a0 + (i + 1) * d
        p0 = (cx + r * math.cos(t0), cy + r * math.sin(t0))
        p3 = (cx + r * math.cos(t1), cy + r * math.sin(t1))
        c1 = (p0[0] - k * r * math.sin(t0), p0[1] + k * r * math.cos(t0))
        c2 = (p3[0] + k * r * math.sin(t1), p3[1] - k * r * math.cos(t1))
        segs.append((c1, c2, p3))
    return start, segs


class PdfWriter:
    """Accumulates content-stream operators in sheet mm and saves one page."""

    def __init__(self, width_mm: float, height_mm: float, compress: bool = True):
        self.w_mm = float(width_mm)
        self.h_mm = float(height_mm)
        self.compress = compress
        self.ops: List[str] = []
        self._state: Optional[Tuple] = None

    # state -------------------------------------------------------------
    def set_style(self, width_mm: float, dash_mm: Optional[Sequence[float]] = None) -> None:
        key = (round(width_mm, 6), tuple(dash_mm) if dash_mm else None)
        if key == self._state:
            return
        self._state = key
        dash = "[" + " ".join(fmt(v * MM) for v in dash_mm) + "] 0 d" if dash_mm else "[] 0 d"
        self.ops.append(f"{fmt(width_mm * MM)} w {dash}")

    def _p(self, x: float, y: float) -> str:
        return f"{fmt(x * MM)} {fmt(y * MM)}"

    # geometry ----------------------------------------------------------
    def line(self, a, b) -> None:
        self.ops.append(f"{self._p(*a)} m {self._p(*b)} l S")

    def polyline(self, pts, closed: bool = False) -> None:
        s = [f"{self._p(*pts[0])} m"] + [f"{self._p(*p)} l" for p in pts[1:]]
        s.append("h S" if closed else "S")
        self.ops.append(" ".join(s))

    def segments(self, segs) -> None:
        """Many independent segments as ONE stroked path."""
        s = [f"{self._p(*a)} m {self._p(*b)} l" for a, b in segs]
        self.ops.append(" ".join(s) + " S")

    def arc(self, cx, cy, r, a0, span, closed: bool = False) -> None:
        start, segs = arc_beziers(cx, cy, r, a0, span)
        s = [f"{self._p(*start)} m"]
        for c1, c2, p in segs:
            s.append(f"{self._p(*c1)} {self._p(*c2)} {self._p(*p)} c")
        self.ops.append(" ".join(s) + (" h S" if closed else " S"))

    def filled_polygon(self, pts) -> None:
        s = [f"{self._p(*pts[0])} m"] + [f"{self._p(*p)} l" for p in pts[1:]]
        self.ops.append(" ".join(s) + " h f")

    # text --------------------------------------------------------------
    def text(self, x, y, s, size_mm, rot_deg=0.0, anchor="middle", valign="baseline") -> None:
        w = helvetica_width(s, size_mm)
        dx = {"start": 0.0, "end": -w}.get(anchor, -w / 2.0)
        dy = -HELV_CAP * size_mm / 2.0 if valign == "middle" else 0.0
        a = math.radians(rot_deg)
        c, si = math.cos(a), math.sin(a)
        if abs(c) < 1e-12:
            c = 0.0
        if abs(si) < 1e-12:
            si = 0.0
        ox = x + c * dx - si * dy
        oy = y + si * dx + c * dy
        self.ops.append(
            f"BT /F1 {fmt(size_mm * MM)} Tf {fmt(c)} {fmt(si)} {fmt(-si)} {fmt(c)} "
            f"{self._p(ox, oy)} Tm (" + _esc(s).decode("latin-1") + ") Tj ET")

    # output ------------------------------------------------------------
    def to_bytes(self) -> Tuple[bytes, int]:
        content = "\n".join(["0 0 0 RG 0 0 0 rg 0 J 0 j"] + self.ops).encode("latin-1")
        if self.compress:
            content = zlib.compress(content, 6)
            head = f"<< /Length {len(content)} /Filter /FlateDecode >>"
        else:
            head = f"<< /Length {len(content)} >>"
        W, H = fmt(self.w_mm * MM), fmt(self.h_mm * MM)
        objs = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {W} {H}] "
             f"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>").encode("ascii"),
            head.encode("ascii") + b"\nstream\n" + content + b"\nendstream",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
        ]
        out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets = []
        for i, body in enumerate(objs, start=1):
            offsets.append(len(out))
            out += f"{i} 0 obj\n".encode("ascii") + body + b"\nendobj\n"
        xref = len(out)
        out += f"xref\n0 {len(objs) + 1}\n".encode("ascii")
        out += b"0000000000 65535 f \n"
        for off in offsets:
            out += f"{off:010d} 00000 n \n".encode("ascii")
        out += (f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\n"
                f"startxref\n{xref}\n%%EOF\n").encode("ascii")
        return bytes(out), len(objs)


def _draw_item(pw: PdfWriter, item, sheet_style: SheetStyle) -> None:
    if isinstance(item, Text):
        pw.text(float(item.position[0]), float(item.position[1]), item.text, float(item.height),
                float(item.rotation_deg), item.anchor, item.valign)
        return
    width, dash = line_props(item.style, sheet_style)
    if isinstance(item, Polygon):
        pw.set_style(width, None)
        pw.filled_polygon(np.asarray(item.points, dtype=np.float64).reshape(-1, 2).tolist())
        return
    pw.set_style(width, dash)
    if isinstance(item, Segment):
        pw.line(item.a, item.b)
    elif isinstance(item, Polyline):
        pw.polyline(np.asarray(item.points, dtype=np.float64).reshape(-1, 2).tolist(),
                    bool(item.closed))
    elif isinstance(item, Circle):
        pw.arc(float(item.center[0]), float(item.center[1]), float(item.radius), 0.0, 360.0,
               closed=True)
    elif isinstance(item, Arc):
        pw.arc(float(item.center[0]), float(item.center[1]), float(item.radius),
               float(item.start_deg), arc_span(item.start_deg, item.end_deg))
    elif isinstance(item, HatchLines):
        pw.segments(np.asarray(item.segments, dtype=np.float64).reshape(-1, 2, 2).tolist())


def write_pdf(sheet: Sheet, path: str, sheet_style: SheetStyle = DEFAULT_STYLE,
              compress: bool = True) -> dict:
    """Write ``sheet`` as a one-page vector PDF.

    Returns ``{"bytes": n, "objects": n, "skipped": n}``.
    """
    pw = PdfWriter(sheet.width, sheet.height, compress=compress)
    skipped = 0
    for item in sheet.items:
        if not valid_item(item):
            skipped += 1
            continue
        _draw_item(pw, item, sheet_style)
    if skipped:
        log.warning("pdf: skipped %d degenerate or non-finite items", skipped)
    data, n_obj = pw.to_bytes()
    with open(path, "wb") as fh:
        fh.write(data)
    return {"bytes": len(data), "objects": n_obj, "skipped": skipped}
