"""Tests for the drawing writers (SVG, DXF, PDF) - task B4.

One ``Sheet`` fixture holds every item type: segments in every line style,
a circle, a 90 degree arc, a closed polyline, a filled triangle, hatch
lines with boundary loops, texts at the three anchors (with Ø ± °), a
rotated text, the U+2300 diameter sign, and one linear and one diameter
``Dimension`` together with the primitives the layout renders for them.

Independent checks: ``xml.etree`` for SVG, ``ezdxf.readfile`` + ``audit``
for DXF, a hand-written xref/stream parser for PDF and - when the checker
venv exists - PyMuPDF in a subprocess (it is not installed in the main
interpreter).
"""
from __future__ import annotations

import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import collections
import json
import os
import re
import subprocess
import xml.etree.ElementTree as ET
import zlib

import numpy as np
import pytest

from src.drawing.model import (Arc, Circle, Dimension, HatchLines, LineStyle, Polygon,
                               Polyline, Segment, Sheet, Text)
from src.drawing.writers import helvetica_width, write_dxf, write_pdf, write_svg

try:
    import ezdxf  # noqa: F401
    HAS_EZDXF = True
except Exception:  # pragma: no cover - environment dependent
    HAS_EZDXF = False

needs_ezdxf = pytest.mark.skipif(not HAS_EZDXF, reason="ezdxf not installed")

CHECKER_PY = r"C:\Users\pmqua\.venvs\notebooklm\Scripts\python.exe"
needs_checker = pytest.mark.skipif(not os.path.exists(CHECKER_PY),
                                   reason="PyMuPDF checker venv not present")

SVG_NS = "{http://www.w3.org/2000/svg}"
MM = 72.0 / 25.4
LABEL = "SECTION A-A"
SPECIAL = "Ø6.5 ±0.1 45°"


# --------------------------------------------------------------------------- fixture

def _hatch_square():
    outer = np.array([[50.0, 50.0], [110.0, 50.0], [110.0, 110.0], [50.0, 110.0]])
    hole = np.array([[70.0, 70.0], [90.0, 70.0], [90.0, 90.0], [70.0, 90.0]])
    segs = []
    # 45 degree lines x - y = c, clipped to the outer square only (writers do not clip)
    for c in np.arange(-57.0, 60.0, 3.0 * np.sqrt(2.0)):
        x0, x1 = max(50.0, 50.0 + c), min(110.0, 110.0 + c)
        if x1 - x0 > 1e-6:
            segs.append([[x0, x0 - c], [x1, x1 - c]])
    return HatchLines(np.array(segs, dtype=np.float64), [outer, hole])


def make_sheet() -> Sheet:
    items = []
    # frame
    for a, b in [((20, 10), (410, 10)), ((410, 10), (410, 287)),
                 ((410, 287), (20, 287)), ((20, 287), (20, 10))]:
        items.append(Segment(a, b, LineStyle.FRAME))
    y = 250.0
    for st in (LineStyle.VISIBLE, LineStyle.HIDDEN, LineStyle.HIDDEN, LineStyle.SECTION,
               LineStyle.HATCH, LineStyle.CENTER, LineStyle.DIMENSION, LineStyle.CUTTING,
               LineStyle.TITLE):
        items.append(Segment((40.0, y), (120.0, y), st))
        y -= 5.0
    items.append(Circle((200.0, 200.0), 15.0, LineStyle.VISIBLE))
    items.append(Arc((260.0, 200.0), 15.0, 0.0, 90.0, LineStyle.VISIBLE))
    items.append(Polyline(np.array([[300.0, 180.0], [360.0, 180.0], [360.0, 220.0], [300.0, 220.0]]),
                          True, LineStyle.VISIBLE))
    items.append(Polygon(np.array([[150.0, 150.0], [153.5, 151.0], [153.5, 149.0]]),
                         LineStyle.DIMENSION))
    items.append(_hatch_square())
    items.append(Text((150.0, 120.0), SPECIAL, 3.5, anchor="start"))
    items.append(Text((220.0, 120.0), SPECIAL, 3.5, anchor="middle"))
    items.append(Text((300.0, 120.0), SPECIAL, 3.5, anchor="end"))
    items.append(Text((380.0, 150.0), "VERT 25", 3.5, rotation_deg=90.0, valign="middle"))
    items.append(Text((200.0, 260.0), LABEL, 5.0, anchor="middle"))
    items.append(Text((330.0, 60.0), "\u230012 (rev A) path\\x", 3.5, anchor="start"))

    # linear dimension 60 (horizontal) + the primitives the layout renders for it
    lin = Dimension(kind="linear", text="60", value=60.0, p1=(150.0, 60.0), p2=(210.0, 60.0),
                    base=(180.0, 45.0), angle_deg=0.0, view="front")
    items += [Segment((150.0, 59.0), (150.0, 43.0), LineStyle.DIMENSION),
              Segment((210.0, 59.0), (210.0, 43.0), LineStyle.DIMENSION),
              Segment((150.0, 45.0), (210.0, 45.0), LineStyle.DIMENSION),
              Polygon(np.array([[150.0, 45.0], [153.5, 45.46], [153.5, 44.54]])),
              Polygon(np.array([[210.0, 45.0], [206.5, 45.46], [206.5, 44.54]])),
              Text((180.0, 46.0), "60", 3.5, anchor="middle")]
    # diameter dimension, 4 x Ø6.5, leader at 45 degrees
    dia = Dimension(kind="diameter", text="4 x Ø6.5", value=6.5, center=(250.0, 70.0),
                    radius=3.25, leader_angle_deg=45.0, view="top", count=4)
    c45 = np.cos(np.radians(45.0))
    tip = (250.0 + 3.25 * c45, 70.0 + 3.25 * c45)
    end = (250.0 + 15.0 * c45, 70.0 + 15.0 * c45)
    items += [Segment(tip, end, LineStyle.DIMENSION),
              Segment(end, (end[0] + 5.0, end[1]), LineStyle.DIMENSION),
              Polygon(np.array([tip, (tip[0] + 2.2, tip[1] + 2.8), (tip[0] + 2.8, tip[1] + 2.2)])),
              Text((end[0] + 6.0, end[1]), "4 x Ø6.5", 3.5, anchor="start", valign="middle")]
    return Sheet("A3", 420.0, 297.0, items=items, dimensions=[lin, dia],
                 meta={"scale": 1.0, "scale_text": "1:1"})


def _count(sheet, cls, style=None):
    return sum(1 for it in sheet.items
               if isinstance(it, cls) and (style is None or it.style == style))


# --------------------------------------------------------------------------- SVG

@pytest.fixture(scope="module")
def svg_out(tmp_path_factory):
    sheet = make_sheet()
    p = tmp_path_factory.mktemp("svg") / "sheet.svg"
    res = write_svg(sheet, str(p))
    return sheet, str(p), res


def _svg_drawn(root):
    tags = ("line", "polyline", "polygon", "circle", "path", "text")
    return [e for e in root.iter() if e.tag in tuple(SVG_NS + t for t in tags)]


def test_svg_parses_viewbox_and_flip_group(svg_out):
    sheet, path, res = svg_out
    root = ET.parse(path).getroot()
    assert root.get("viewBox") == "0 0 420 297"
    assert root.get("width") == "420mm" and root.get("height") == "297mm"
    flips = [g for g in root.iter(SVG_NS + "g")
             if g.get("transform") == "translate(0,297) scale(1,-1)"]
    assert len(flips) == 1
    assert res["bytes"] == os.path.getsize(path)


def test_svg_text_outside_flipped_group(svg_out):
    sheet, path, res = svg_out
    root = ET.parse(path).getroot()
    flip = next(g for g in root.iter(SVG_NS + "g") if "scale(1,-1)" in (g.get("transform") or ""))
    inside = list(flip.iter(SVG_NS + "text"))
    total = list(root.iter(SVG_NS + "text"))
    n_text = _count(sheet, Text)
    assert n_text >= 6  # precondition: the fixture has texts
    assert inside == []
    assert len(total) == n_text
    # geometry IS inside the flipped group (a writer that put nothing there would fail)
    assert len(list(flip.iter(SVG_NS + "line"))) == _count(sheet, Segment)


def test_svg_dashes_and_widths(svg_out):
    sheet, path, res = svg_out
    root = ET.parse(path).getroot()
    hidden = [e for e in root.iter() if e.get("class") == "hidden"]
    assert len(hidden) == _count(sheet, Segment, LineStyle.HIDDEN) == 2
    for e in hidden:
        assert (e.get("stroke-dasharray") or "").split()[0] == "3"
        assert e.get("stroke-width") == "0.25"
    center = [e for e in root.iter() if e.get("class") == "center"]
    assert center and all(e.get("stroke-dasharray").split()[0] == "6" for e in center)
    cutting = [e for e in root.iter() if e.get("class") == "cutting"]
    assert cutting and all(e.get("stroke-dasharray").split()[0] == "12" for e in cutting)
    visible = [e for e in root.iter() if e.get("class") == "visible"]
    assert visible and all(e.get("stroke-dasharray") is None for e in visible)
    assert all(e.get("stroke-width") == "0.5" for e in visible)
    widths = {e.get("stroke-width") for e in root.iter() if e.get("stroke-width")}
    assert {"0.5", "0.25", "0.7"} <= widths
    frame = [e for e in root.iter() if e.get("class") == "frame"]
    assert len(frame) == 4 and all(e.get("stroke-width") == "0.7" for e in frame)


def test_svg_text_content_and_position(svg_out):
    sheet, path, res = svg_out
    root = ET.parse(path).getroot()
    texts = list(root.iter(SVG_NS + "text"))
    contents = [t.text for t in texts]
    assert contents.count(SPECIAL) == 3
    assert "Ø12 (rev A) path\\x" in contents      # U+2300 mapped to Ø, specials kept
    assert not any("\u2300" in (c or "") for c in contents)
    mid = [t for t in texts if t.text == SPECIAL and t.get("text-anchor") == "middle"]
    assert len(mid) == 1
    assert float(mid[0].get("x")) == pytest.approx(220.0)
    assert float(mid[0].get("y")) == pytest.approx(297.0 - 120.0)
    anchors = sorted(t.get("text-anchor") for t in texts if t.text == SPECIAL)
    assert anchors == ["end", "middle", "start"]
    vert = next(t for t in texts if t.text == "VERT 25")
    x, yv = float(vert.get("x")), float(vert.get("y"))
    # middle valign: baseline shifted by 0.718*h/2 along the rotated text's down direction
    assert x == pytest.approx(380.0 + 0.718 * 3.5 / 2.0, abs=1e-3)
    assert yv == pytest.approx(297.0 - 150.0, abs=1e-3)
    assert vert.get("transform") == f"rotate(-90 {vert.get('x')} {vert.get('y')})"


def test_svg_arc_circle_hatch_and_counts(svg_out):
    sheet, path, res = svg_out
    root = ET.parse(path).getroot()
    circles = list(root.iter(SVG_NS + "circle"))
    assert len(circles) == 1 and float(circles[0].get("r")) == pytest.approx(15.0)
    arcs = [e for e in root.iter(SVG_NS + "path") if " A " in e.get("d")]
    assert len(arcs) == 1
    d = arcs[0].get("d").split()
    # M 275 200 A 15 15 0 0 1 260 215  (CCW quarter in the y-up group)
    assert [float(v) for v in d[1:3]] == pytest.approx([275.0, 200.0])
    assert d[4:9] == ["15", "15", "0", "0", "1"]
    assert [float(v) for v in d[9:11]] == pytest.approx([260.0, 215.0])
    hatch = [e for e in root.iter() if e.get("class") == "hatch" and e.tag == SVG_NS + "path"]
    assert len(hatch) == 1
    n_seg = len(next(it for it in sheet.items if isinstance(it, HatchLines)).segments)
    assert n_seg > 5
    assert hatch[0].get("d").count("M") == n_seg
    fills = [e for e in root.iter(SVG_NS + "polygon") if e.get("fill") not in (None, "none")]
    assert len(fills) == _count(sheet, Polygon)
    closed = [e for e in root.iter(SVG_NS + "polygon") if e.get("fill") == "none"]
    assert len(closed) == 1
    assert res["elements"] == len(_svg_drawn(root))


@needs_checker
def test_svg_pymupdf_page_size(svg_out, tmp_path):
    sheet, path, res = svg_out
    script = tmp_path / "chk_svg.py"
    script.write_text(
        "import sys, json, fitz\n"
        "d = fitz.open(sys.argv[1]); pg = d[0]\n"
        "print(json.dumps({'w': pg.rect.width * 25.4 / 72, 'h': pg.rect.height * 25.4 / 72,"
        " 'text': pg.get_text()}))\n", encoding="utf-8")
    out = subprocess.run([CHECKER_PY, "-I", str(script), path], capture_output=True, text=True,
                         encoding="utf-8", timeout=60)
    assert out.returncode == 0, out.stderr
    data = json.loads(out.stdout.strip().splitlines()[-1])
    assert data["w"] == pytest.approx(420.0, abs=0.05)
    assert data["h"] == pytest.approx(297.0, abs=0.05)
    assert LABEL in data["text"]


# --------------------------------------------------------------------------- DXF

@pytest.fixture(scope="module")
def dxf_out(tmp_path_factory):
    if not HAS_EZDXF:
        pytest.skip("ezdxf not installed")
    sheet = make_sheet()
    p = tmp_path_factory.mktemp("dxf") / "sheet.dxf"
    res = write_dxf(sheet, str(p))
    return sheet, str(p), res


@needs_ezdxf
def test_dxf_readback_audit_clean_and_header(dxf_out):
    import ezdxf
    sheet, path, res = dxf_out
    doc = ezdxf.readfile(path)
    aud = doc.audit()
    assert len(aud.errors) == 0 and len(aud.fixes) == 0
    assert doc.header["$INSUNITS"] == 4
    assert doc.header["$MEASUREMENT"] == 1
    assert doc.header["$LWDISPLAY"] == 1
    assert doc.layers.get("HIDDEN").dxf.linetype.upper() == "HIDDEN_ISO"   # QA2 issue 22
    assert doc.layers.get("HIDDEN").dxf.lineweight == 25
    assert doc.layers.get("VISIBLE").dxf.lineweight == 50
    assert doc.layers.get("CENTER").dxf.linetype.upper() == "CENTER_ISO"
    assert res["bytes"] == os.path.getsize(path)
    assert res["audit_errors"] == 0 and res["audit_fixes"] == 0


@needs_ezdxf
def test_dxf_entity_counts_per_layer(dxf_out):
    import ezdxf
    sheet, path, res = dxf_out
    doc = ezdxf.readfile(path)
    cnt = collections.Counter((e.dxf.layer, e.dxftype()) for e in doc.modelspace())
    n_text = _count(sheet, Text) - 2      # the two dimension texts become DIMENSION entities
    expected = {
        ("FRAME", "LINE"): 4,
        ("VISIBLE", "LINE"): 1, ("VISIBLE", "CIRCLE"): 1, ("VISIBLE", "ARC"): 1,
        ("VISIBLE", "LWPOLYLINE"): 1,
        ("HIDDEN", "LINE"): 2, ("SECTION", "LINE"): 1,
        ("HATCH", "LINE"): 1, ("HATCH", "HATCH"): 1,
        ("CENTER", "LINE"): 1, ("CUTTING", "LINE"): 1, ("TITLE", "LINE"): 1,
        ("DIM", "LINE"): 1, ("DIM", "SOLID"): 1, ("DIM", "DIMENSION"): 2,
        ("TEXT", "TEXT"): n_text,
    }
    assert dict(cnt) == expected
    assert res["entities"] == sum(expected.values())
    assert res["layers"]["HIDDEN"] == 2


@needs_ezdxf
def test_dxf_dimension_entities_and_dedupe(dxf_out):
    import ezdxf
    sheet, path, res = dxf_out
    doc = ezdxf.readfile(path)
    # precondition: the sheet carries the rendered dimension text "60" as a Text item
    assert any(isinstance(it, Text) and it.text == "60" for it in sheet.items)
    msp = doc.modelspace()
    dims = list(msp.query("DIMENSION"))
    assert len(dims) == 2
    texts = {}
    for d in dims:
        blk = doc.blocks.get(d.dxf.geometry)
        assert blk is not None
        texts[d.dimtype & 0x0F] = [e.text for e in blk if e.dxftype() == "MTEXT"]
    dia_txt = " ".join(texts[3])
    assert dia_txt.count("%%c") == 1 and "4 x" in dia_txt and "6.5" in dia_txt
    assert texts[0] == ["60"] or texts.get(1) == ["60"]
    plain = [t.dxf.text for t in msp.query("TEXT")]
    assert "60" not in plain and "4 x Ø6.5" not in plain   # not duplicated as TEXT
    # the standalone DIMENSION-style segment and the triangle are kept
    assert len(msp.query('LINE[layer=="DIM"]')) == 1
    assert len(msp.query('SOLID[layer=="DIM"]')) == 1


@needs_ezdxf
def test_dxf_text_alignment_and_hatch(dxf_out):
    import ezdxf
    from ezdxf.enums import TextEntityAlignment
    sheet, path, res = dxf_out
    doc = ezdxf.readfile(path)
    msp = doc.modelspace()
    special = [t for t in msp.query("TEXT") if t.dxf.text == SPECIAL]
    aligns = sorted(t.get_align_enum().name for t in special)
    assert aligns == sorted([TextEntityAlignment.LEFT.name, TextEntityAlignment.CENTER.name,
                             TextEntityAlignment.RIGHT.name])
    vert = next(t for t in msp.query("TEXT") if t.dxf.text == "VERT 25")
    assert vert.dxf.rotation == pytest.approx(90.0)
    assert vert.get_align_enum() == TextEntityAlignment.MIDDLE_CENTER
    # DXF TEXT height is the cap height: 0.718 x the em size the PDF/SVG use
    assert vert.dxf.height == pytest.approx(0.718 * 3.5, abs=1e-6)
    lab = next(t for t in msp.query("TEXT") if t.dxf.text == LABEL)
    assert lab.dxf.height == pytest.approx(0.718 * 5.0, abs=1e-6)
    dim = msp.query("DIMENSION").first
    mt = [e for e in doc.blocks.get(dim.dxf.geometry) if e.dxftype() == "MTEXT"]
    assert mt and mt[0].dxf.char_height == pytest.approx(0.718 * 3.5, abs=1e-3)
    assert any(t.dxf.text.startswith("Ø12") for t in msp.query("TEXT"))
    h = msp.query("HATCH").first
    assert h.dxf.pattern_name == "ANSI31"
    assert len(h.paths) == 2
    assert h.dxf.hatch_style == 0
    line = h.pattern.lines[0]
    spacing = float(np.hypot(line.offset[0], line.offset[1]))
    assert spacing == pytest.approx(3.0, abs=1e-3)       # fixture spacing 3 mm
    assert line.angle % 180.0 == pytest.approx(45.0, abs=1e-3)
    arc = msp.query("ARC").first
    assert (arc.dxf.start_angle, arc.dxf.end_angle) == pytest.approx((0.0, 90.0))


@needs_ezdxf
def test_dxf_audit_negative_control(dxf_out, tmp_path):
    """An undefined linetype (HIDDEN) must make audit report a fix.

    Proves the 0-fixes assertion above can fail: the same writer output with
    one extra layer that asks for linetype HIDDEN is silently 'fixed'.
    """
    import ezdxf
    sheet, path, res = dxf_out
    doc = ezdxf.readfile(path)
    assert "HIDDEN" not in [lt.dxf.name.upper() for lt in doc.linetypes]   # precondition
    _add_layer_with_linetype(doc, "BADHIDDEN", "HIDDEN")
    bad = tmp_path / "bad.dxf"
    doc.saveas(str(bad))
    aud = ezdxf.readfile(str(bad)).audit()
    assert len(aud.fixes) > 0


def _add_layer_with_linetype(doc, name, linetype):
    doc.layers.add(name, linetype=linetype)


def _rendered_sheet():
    """A Ø20 hole at (100, 100) and a linear 60 dimension, rendered by the REAL
    ``dimensions.render_dimension``, plus a section marker whose cutting line
    runs through the hole and whose default-style (DIMENSION) arrowheads sit
    near the callout but off every dimension axis."""
    from src.drawing.dimensions import render_dimension
    from src.drawing.style import DEFAULT_STYLE
    dia = Dimension(kind="diameter", text="Ø20", value=20.0, center=(100.0, 100.0), radius=10.0,
                    leader_angle_deg=45.0, base=(100.0, 100.0), view="top")
    lin = Dimension(kind="linear", text="60", value=60.0, p1=(70.0, 60.0), p2=(130.0, 60.0),
                    base=(100.0, 50.0), angle_deg=0.0, view="top")
    items, groups = [], []
    for d in (dia, lin):
        r = render_dimension(d, DEFAULT_STYLE)
        groups.append(list(range(len(items), len(items) + len(r))))
        items += r
    marker = [Segment((60.0, 100.0), (72.0, 100.0), LineStyle.CUTTING),
              Segment((128.0, 100.0), (140.0, 100.0), LineStyle.CUTTING),
              Polygon(np.array([[60.0, 125.0], [58.66, 130.0], [61.34, 130.0]])),   # arrow, 30 mm off
              Polygon(np.array([[100.0, 70.0], [98.66, 75.0], [101.34, 75.0]])),    # 30 mm below centre
              Polygon(np.array([[75.0, 63.0], [73.66, 68.0], [76.34, 68.0]])),      # 3 mm above p1 edge
              Text((60.0, 133.0), "A", 5.0)]
    items += marker
    sheet = Sheet("A4", 210.0, 297.0, items=items, dimensions=[dia, lin],
                  meta={"scale": 1.0, "scale_text": "1:1"})
    return sheet, groups, marker


@needs_ezdxf
def test_dxf_keeps_marker_arrows_near_dimensions(tmp_path):
    """Geometric fallback: rendered dimension copies go, nearby filled polygons stay."""
    import ezdxf
    sheet, groups, marker = _rendered_sheet()
    n_rendered = sum(len(g) for g in groups)
    arrows = [m for m in marker if isinstance(m, Polygon)]
    # precondition: every marker arrow lies inside the old square region (centre +- r + 45 /
    # linear box + 10 mm), i.e. the region rule would have dropped it
    for a in arrows:
        p = np.asarray(a.points)
        assert a.style == LineStyle.DIMENSION
        assert (np.all(np.abs(p - [100.0, 100.0]) <= 55.0)
                or np.all((p[:, 0] >= 60) & (p[:, 0] <= 140) & (p[:, 1] >= 40) & (p[:, 1] <= 70)))
    res = write_dxf(sheet, str(tmp_path / "m.dxf"))
    assert res["dimension_match"] == "geometric"
    assert res["dimensions"] == 2
    assert res["dimension_items_replaced"] == n_rendered     # all rendered copies removed
    msp = ezdxf.readfile(str(tmp_path / "m.dxf")).modelspace()
    assert len(msp.query("SOLID")) == len(arrows) == 3       # marker arrows kept
    assert len(msp.query('LINE[layer=="DIM"]')) == 0
    assert len(msp.query('LINE[layer=="CUTTING"]')) == 2
    assert [t.dxf.text for t in msp.query("TEXT")] == ["A"]
    assert len(msp.query("DIMENSION")) == 2


@needs_ezdxf
def test_dxf_exact_dimension_items_from_meta(tmp_path):
    import ezdxf
    sheet, groups, marker = _rendered_sheet()
    # an extra DIMENSION-style segment ON the linear dimension line, not part of any dimension
    extra = Segment((80.0, 50.0), (90.0, 50.0), LineStyle.DIMENSION)
    sheet.items.append(extra)
    n_rendered = sum(len(g) for g in groups)
    # precondition: the geometric rule takes it for a copy
    res_geo = write_dxf(sheet, str(tmp_path / "g.dxf"))
    assert res_geo["dimension_items_replaced"] == n_rendered + 1
    sheet.meta["dimension_items"] = groups
    res = write_dxf(sheet, str(tmp_path / "e.dxf"))
    assert res["dimension_match"] == "exact"
    assert res["dimension_items_replaced"] == n_rendered
    msp = ezdxf.readfile(str(tmp_path / "e.dxf")).modelspace()
    assert len(msp.query('LINE[layer=="DIM"]')) == 1
    assert len(msp.query("SOLID")) == 3
    # a list that does not match the dimensions falls back to the geometric rule
    sheet.meta["dimension_items"] = groups[:1]
    res_bad = write_dxf(sheet, str(tmp_path / "b.dxf"))
    assert res_bad["dimension_match"] == "geometric"
    assert res_bad["dimension_items_replaced"] == n_rendered + 1


@needs_ezdxf
def test_dxf_non_finite_dimension_skipped(tmp_path):
    import ezdxf
    good = Dimension(kind="linear", text="60", value=60.0, p1=(20.0, 20.0), p2=(80.0, 20.0),
                     base=(50.0, 10.0))
    bad = Dimension(kind="linear", text="40", value=40.0, p1=(20.0, 150.0),
                    p2=(float("nan"), 150.0), base=(40.0, 140.0))
    zero = Dimension(kind="diameter", text="Ø0", value=0.0, center=(150.0, 150.0), radius=0.0)
    # the bad dimension's rendered primitives are finite and must stay in the DXF
    kept = [Segment((20.0, 140.0), (60.0, 140.0), LineStyle.DIMENSION),
            Text((40.0, 141.0), "40", 3.5)]
    sheet = Sheet("A4", 210.0, 297.0, items=list(kept), dimensions=[good, bad, zero],
                  meta={"scale": 1.0})
    assert not np.isfinite(bad.p2[0])          # precondition
    res = write_dxf(sheet, str(tmp_path / "n.dxf"))
    assert res["dimensions"] == 1 and res["dimensions_skipped"] == 2
    assert res["skipped"] == 2
    text = (tmp_path / "n.dxf").read_text(encoding="utf-8").splitlines()
    assert not any(v.strip().lower() in ("nan", "-nan", "inf", "-inf") for v in text)
    msp = ezdxf.readfile(str(tmp_path / "n.dxf")).modelspace()
    assert len(msp.query("DIMENSION")) == 1
    assert len(msp.query('LINE[layer=="DIM"]')) == 1
    assert [t.dxf.text for t in msp.query("TEXT")] == ["40"]


@needs_ezdxf
def test_dxf_scale_fallback_from_diameter(tmp_path):
    import ezdxf
    # sheet at 2:1 without meta["scale"] and only a diameter dimension: r 20 sheet = Ø20 model
    dia = Dimension(kind="diameter", text="Ø20", value=20.0, center=(100.0, 100.0), radius=20.0)
    res = write_dxf(Sheet("A4", 210.0, 297.0, dimensions=[dia]), str(tmp_path / "s.dxf"))
    assert res["dimensions"] == 1
    d = ezdxf.readfile(str(tmp_path / "s.dxf")).modelspace().query("DIMENSION").first
    assert d.override().get("dimlfac") == pytest.approx(0.5)


_DXF_VOLATILE = {"$TDCREATE", "$TDUCREATE", "$TDUPDATE", "$TDUUPDATE", "$TDINDWG",
                 "$TDUSRTIMER", "$FINGERPRINTGUID", "$VERSIONGUID"}


def _masked_dxf(path):
    lines = open(path, encoding="utf-8").read().splitlines()
    pairs = [(lines[i].strip(), lines[i + 1]) for i in range(0, len(lines) - 1, 2)]
    out, mask_next = [], False
    for code, val in pairs:
        if mask_next or re.match(r"^\d+\.\d+\.\d+ @ \d{4}-\d\d-\d\dT", val):
            val = "<masked>"          # header dates / GUIDs and ezdxf's '@ timestamp' metadata
        mask_next = code == "9" and val in _DXF_VOLATILE
        out.append((code, val))
    return out


@needs_ezdxf
def test_dxf_deterministic_after_masking(tmp_path):
    sheet = make_sheet()
    write_dxf(sheet, str(tmp_path / "a.dxf"))
    write_dxf(sheet, str(tmp_path / "b.dxf"))
    a, b = _masked_dxf(str(tmp_path / "a.dxf")), _masked_dxf(str(tmp_path / "b.dxf"))
    assert sum(1 for _, v in a if v == "<masked>") >= 4      # the mask is doing something
    assert a == b
    sheet.items.append(Segment((30.0, 30.0), (40.0, 40.0), LineStyle.VISIBLE))
    write_dxf(sheet, str(tmp_path / "c.dxf"))
    assert _masked_dxf(str(tmp_path / "c.dxf")) != a


def test_control_characters_stripped_from_text(tmp_path):
    raw = "AB\x01C\tD\x7f"
    with pytest.raises(ET.ParseError):                       # precondition: XML forbids \x01
        ET.fromstring("<t>\x01</t>")
    sheet = Sheet("A4", 210.0, 297.0, items=[Text((50.0, 50.0), raw, 3.5)])
    write_svg(sheet, str(tmp_path / "c.svg"))
    texts = [t.text for t in ET.parse(str(tmp_path / "c.svg")).getroot().iter(SVG_NS + "text")]
    assert texts == ["ABC D"]
    write_pdf(sheet, str(tmp_path / "c.pdf"))
    data = (tmp_path / "c.pdf").read_bytes()
    assert b"(ABC D) Tj" in _pdf_content(data, _pdf_objects(data))
    if HAS_EZDXF:
        import ezdxf
        write_dxf(sheet, str(tmp_path / "c.dxf"))
        msp = ezdxf.readfile(str(tmp_path / "c.dxf")).modelspace()
        assert [t.dxf.text for t in msp.query("TEXT")] == ["ABC D"]


# --------------------------------------------------------------------------- PDF

@pytest.fixture(scope="module")
def pdf_out(tmp_path_factory):
    sheet = make_sheet()
    p = tmp_path_factory.mktemp("pdf") / "sheet.pdf"
    res = write_pdf(sheet, str(p))
    with open(p, "rb") as fh:
        data = fh.read()
    return sheet, str(p), res, data


def _pdf_objects(data: bytes):
    m = re.search(rb"startxref\s+(\d+)\s+%%EOF\s*$", data)
    assert m, "no startxref"
    xref = int(m.group(1))
    assert data[xref:xref + 4] == b"xref"
    lines = data[xref:].split(b"\n")
    first, n = (int(v) for v in lines[1].split())
    assert first == 0
    entries = lines[2:2 + n]
    assert entries[0].startswith(b"0000000000 65535 f")
    offsets = {}
    for i, e in enumerate(entries[1:], start=1):
        off, gen, kind = e.split()[:3]
        assert kind == b"n"
        offsets[i] = int(off)
    trailer = data[xref:]
    assert re.search(rb"/Size %d\b" % n, trailer)
    return offsets


def _pdf_content(data: bytes, offsets) -> bytes:
    # the content stream is the object with a stream keyword
    for i, off in offsets.items():
        end = data.index(b"endobj", off)
        body = data[off:end]
        if b"stream\n" in body:
            m = re.search(rb"/Length (\d+)", body)
            start = body.index(b"stream\n") + len(b"stream\n")
            raw = body[start:start + int(m.group(1))]
            assert body[start + int(m.group(1)):].lstrip().startswith(b"endstream")
            return zlib.decompress(raw) if b"/FlateDecode" in body else raw
    raise AssertionError("no content stream")


def test_pdf_header_trailer_xref(pdf_out):
    sheet, path, res, data = pdf_out
    assert data.startswith(b"%PDF-1.4")
    assert data.rstrip().endswith(b"%%EOF")
    offsets = _pdf_objects(data)
    assert len(offsets) == res["objects"] >= 5
    for i, off in offsets.items():
        assert data[off:off + len(f"{i} 0 obj")] == f"{i} 0 obj".encode()
    assert res["bytes"] == len(data)
    assert b"/MediaBox [0 0 1190.551 841.89]" in data


def test_pdf_content_ops(pdf_out):
    sheet, path, res, data = pdf_out
    ops = _pdf_content(data, _pdf_objects(data)).decode("latin-1")
    dash_ops = re.findall(r"\[([^\]]*)\] 0 d", ops)
    assert any(d.split()[:1] == ["8.504"] for d in dash_ops)          # hidden 3 mm dash
    assert re.search(r"(^|\s)1\.417 w", ops) and re.search(r"(^|\s)0\.709 w", ops)
    assert re.search(r"(^|\s)1\.984 w", ops)                          # frame 0.7 mm
    assert ops.count(" Tj") == _count(sheet, Text)
    assert len(re.findall(r" h f(\n|$)", ops)) == _count(sheet, Polygon)
    assert ops.count(" c") >= 4 + 1                                  # circle (4) + arc (1)


def test_pdf_text_encoding_and_centering(pdf_out):
    sheet, path, res, data = pdf_out
    ops_b = _pdf_content(data, _pdf_objects(data))
    assert b"(\xd86.5 \xb10.1 45\xb0) Tj" in ops_b                    # cp1252 Ø ± °
    assert b"(\xd812 \\(rev A\\) path\\\\x) Tj" in ops_b              # U+2300 -> Ø, escapes
    ops = ops_b.decode("latin-1")
    w = helvetica_width(LABEL, 5.0)
    m = re.search(r"1 0 -?0 1 (\S+) (\S+) Tm \(SECTION A-A\) Tj", ops)
    assert m, "label not found"
    assert float(m.group(1)) == pytest.approx((200.0 - w / 2.0) * MM, abs=2e-3)
    assert float(m.group(2)) == pytest.approx(260.0 * MM, abs=2e-3)
    m2 = re.search(r"(\S+) (\S+) (\S+) (\S+) (\S+) (\S+) Tm \(VERT 25\) Tj", ops)
    assert [float(v) for v in m2.groups()[:4]] == pytest.approx([0.0, 1.0, -1.0, 0.0])
    # valign="middle" at 90 degrees: the baseline moves +0.718*h/2 in x (text 'down' = +x),
    # the centring moves -w/2 in y; a flipped dy sign shifts e by 0.718*h*MM = 7.1 pt
    wv = helvetica_width("VERT 25", 3.5)
    e, f = float(m2.group(5)), float(m2.group(6))
    assert e == pytest.approx((380.0 + 0.718 * 3.5 / 2.0) * MM, abs=2e-3)
    assert f == pytest.approx((150.0 - wv / 2.0) * MM, abs=2e-3)
    # font size = Text.height (em convention shared with the SVG; the DXF writes the cap height)
    assert re.search(r"/F1 9\.921 Tf 0 1 -1 0 \S+ \S+ Tm \(VERT 25\) Tj", ops)


def test_helvetica_width_table():
    assert helvetica_width("Ø10", 3.5) == pytest.approx((778 + 556 + 556) / 1000.0 * 3.5)
    assert helvetica_width("", 3.5) == 0.0
    assert helvetica_width("\u230010", 3.5) == pytest.approx(helvetica_width("Ø10", 3.5))
    assert helvetica_width("W", 10.0) == pytest.approx(9.44)


@needs_checker
def test_pdf_pymupdf_reads_clean(pdf_out, tmp_path):
    sheet, path, res, data = pdf_out
    script = tmp_path / "chk_pdf.py"
    script.write_text(
        "import sys, json, fitz\n"
        "fitz.TOOLS.reset_mupdf_warnings()\n"
        "d = fitz.open(sys.argv[1]); pg = d[0]\n"
        "dr = pg.get_drawings()\n"
        "spans = [s for b in pg.get_text('dict')['blocks'] for l in b.get('lines', [])"
        " for s in l['spans']]\n"
        "lab = [s['bbox'] for s in spans if s['text'] == 'SECTION A-A']\n"
        "print(json.dumps({'repaired': d.is_repaired, 'pages': d.page_count,"
        " 'w': pg.rect.width * 25.4 / 72, 'h': pg.rect.height * 25.4 / 72,"
        " 'n_draw': len(dr), 'fills': sum(1 for x in dr if x.get('fill') is not None),"
        " 'dashes': sorted({str(x.get('dashes')) for x in dr if x.get('dashes')}),"
        " 'text': pg.get_text(), 'label_bbox': lab[0] if lab else None,"
        " 'warnings': fitz.TOOLS.mupdf_warnings()}))\n", encoding="utf-8")
    out = subprocess.run([CHECKER_PY, "-I", str(script), path], capture_output=True, text=True,
                         encoding="utf-8", timeout=60)
    assert out.returncode == 0, out.stderr
    info = json.loads(out.stdout.strip().splitlines()[-1])
    assert info["repaired"] is False and info["pages"] == 1
    assert info["warnings"] == ""
    assert info["w"] == pytest.approx(420.0, abs=0.01) and info["h"] == pytest.approx(297.0, abs=0.01)
    n_geom = sum(1 for it in sheet.items if not isinstance(it, Text))
    assert info["n_draw"] >= n_geom
    assert info["fills"] == _count(sheet, Polygon)
    assert any(d.startswith("[ 8.504") for d in info["dashes"])
    assert LABEL in info["text"] and SPECIAL in info["text"]
    x0, y0, x1, y1 = info["label_bbox"]
    assert ((x0 + x1) / 2.0) * 25.4 / 72 == pytest.approx(200.0, abs=0.05)   # centred


# --------------------------------------------------------------------------- shared

def test_svg_and_pdf_are_deterministic(tmp_path):
    sheet = make_sheet()
    blobs = {}
    for k in ("a", "b"):
        write_svg(sheet, str(tmp_path / f"{k}.svg"))
        write_pdf(sheet, str(tmp_path / f"{k}.pdf"))
        blobs[k] = ((tmp_path / f"{k}.svg").read_bytes(), (tmp_path / f"{k}.pdf").read_bytes())
    assert blobs["a"] == blobs["b"]
    # and the content depends on the sheet (not a constant writer)
    sheet.items.append(Segment((30.0, 30.0), (40.0, 40.0), LineStyle.VISIBLE))
    write_svg(sheet, str(tmp_path / "c.svg"))
    write_pdf(sheet, str(tmp_path / "c.pdf"))
    assert (tmp_path / "c.svg").read_bytes() != blobs["a"][0]
    assert (tmp_path / "c.pdf").read_bytes() != blobs["a"][1]


def test_empty_sheet_and_bad_items(tmp_path):
    sheet = Sheet("A4", 210.0, 297.0)
    r1 = write_svg(sheet, str(tmp_path / "e.svg"))
    r2 = write_pdf(sheet, str(tmp_path / "e.pdf"))
    root = ET.parse(str(tmp_path / "e.svg")).getroot()
    assert root.get("viewBox") == "0 0 210 297" and r1["elements"] == 0
    data = (tmp_path / "e.pdf").read_bytes()
    _pdf_objects(data)
    assert b"/MediaBox [0 0 595.276 841.89]" in data and r2["objects"] == 5
    # non-finite and degenerate items are skipped, not written, not fatal
    bad = Sheet("A4", 210.0, 297.0, items=[
        Segment((float("nan"), 0.0), (1.0, 1.0)),
        Polyline(np.array([[1.0, 1.0]])),
        Circle((10.0, 10.0), 0.0),
        Segment((5.0, 5.0), (50.0, 5.0)),
    ])
    r3 = write_svg(bad, str(tmp_path / "b.svg"))
    assert r3["elements"] == 1 and r3["skipped"] == 3
    r4 = write_pdf(bad, str(tmp_path / "b.pdf"))
    assert r4["skipped"] == 3
    if HAS_EZDXF:
        import ezdxf
        r5 = write_dxf(bad, str(tmp_path / "b.dxf"))
        assert r5["entities"] == 1 and r5["skipped"] == 3
        r6 = write_dxf(sheet, str(tmp_path / "e.dxf"))
        assert r6["entities"] == 0
        aud = ezdxf.readfile(str(tmp_path / "e.dxf")).audit()
        assert len(aud.errors) == 0 and len(aud.fixes) == 0


# --------------------------------------------------------------------------- QA round 1 (F5)

_DXF_DATE_VARS = {"$TDCREATE", "$TDUCREATE", "$TDUPDATE", "$TDUUPDATE"}
_GUID_RE = re.compile(r"^\{[0-9A-F]{8}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{12}\}$")


def _dxf_pairs(path):
    lines = open(path, encoding="utf-8").read().splitlines()
    return [(lines[i].strip(), lines[i + 1]) for i in range(0, len(lines) - 1, 2)]


def _dxf_header_value(pairs, name):
    for i, (code, val) in enumerate(pairs):
        if code == "9" and val == name:
            return pairs[i + 1][1]
    raise KeyError(name)


def _mask_dates_only(path):
    """Mask ONLY the date: the four header dates and ezdxf's 'version @ timestamp' marker."""
    out, mask_next = [], False
    for code, val in _dxf_pairs(path):
        if mask_next or re.match(r"^\d+\.\d+\.\d+ @ \d{4}-\d\d-\d\dT", val):
            val = "<date>"
        mask_next = code == "9" and val in _DXF_DATE_VARS
        out.append((code, val))
    return out


@needs_ezdxf
def test_dxf_reproducible_masking_only_the_date(tmp_path):
    """Issue 30: GUIDs are derived from the content, dates from the sheet date."""
    sheet = make_sheet()
    write_dxf(sheet, str(tmp_path / "a.dxf"))
    write_dxf(sheet, str(tmp_path / "b.dxf"))
    a, b = _mask_dates_only(str(tmp_path / "a.dxf")), _mask_dates_only(str(tmp_path / "b.dxf"))
    assert sum(1 for _, v in a if v == "<date>") >= 4          # the mask hits the dates
    assert a == b                                               # GUIDs and order are stable
    pa = _dxf_pairs(str(tmp_path / "a.dxf"))
    fp, vg = _dxf_header_value(pa, "$FINGERPRINTGUID"), _dxf_header_value(pa, "$VERSIONGUID")
    assert _GUID_RE.match(fp) and _GUID_RE.match(vg) and fp != vg
    # the GUIDs depend on the drawing content (not a constant)
    sheet2 = make_sheet()
    sheet2.items.append(Segment((30.0, 30.0), (40.0, 40.0), LineStyle.VISIBLE))
    write_dxf(sheet2, str(tmp_path / "c.dxf"))
    pc = _dxf_pairs(str(tmp_path / "c.dxf"))
    assert _dxf_header_value(pc, "$FINGERPRINTGUID") != fp
    # an explicit date gives byte-identical files and the Julian date of that day
    write_dxf(sheet, str(tmp_path / "d1.dxf"), date="2026-01-02")
    write_dxf(sheet, str(tmp_path / "d2.dxf"), date="2026-01-02")
    assert (tmp_path / "d1.dxf").read_bytes() == (tmp_path / "d2.dxf").read_bytes()
    import datetime as _dt
    from ezdxf.tools.juliandate import calendardate   # independent decoder (DXF days start at 00:00)
    pd1 = _dxf_pairs(str(tmp_path / "d1.dxf"))
    for var in ("$TDCREATE", "$TDUCREATE", "$TDUPDATE", "$TDUUPDATE"):
        decoded = calendardate(float(_dxf_header_value(pd1, var)))
        assert decoded == _dt.datetime(2026, 1, 2), (var, decoded)
    # the sheet's own title-block date is used when no date is passed
    sheet.meta["date"] = "2026-01-02"
    write_dxf(sheet, str(tmp_path / "d3.dxf"))
    assert (tmp_path / "d3.dxf").read_bytes() == (tmp_path / "d1.dxf").read_bytes()


@needs_ezdxf
def test_dxf_reproducible_across_processes(tmp_path):
    """Issue 30: interpreters with other hash seeds write the same bytes.

    Seeds 1 and 4 are a measured pair: without the CLASSES sort they wrote
    two different files (LAYOUT / ACDBPLACEHOLDER swapped; ezdxf iterates a
    set of type names).
    """
    sheet = make_sheet()
    sheet.meta["date"] = "2026-03-04"
    write_dxf(sheet, str(tmp_path / "here.dxf"))
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    blobs = []
    for seed in ("1", "4"):
        out = str(tmp_path / f"seed{seed}.dxf")
        code = ("import sys; sys.path.insert(0, %r); sys.path.insert(0, %r)\n"
                "from test_drawing_writers import make_sheet\n"
                "from src.drawing.writers import write_dxf\n"
                "s = make_sheet(); s.meta['date'] = '2026-03-04'\n"
                "write_dxf(s, %r)\n" % (repo, os.path.join(repo, "tests"), out))
        env = dict(os.environ, PYTHONHASHSEED=seed, PYTHONDONTWRITEBYTECODE="1")
        r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, timeout=120)
        assert r.returncode == 0, r.stderr.decode("utf-8", "replace")
        blobs.append(open(out, "rb").read())
    assert blobs[0] == blobs[1] == (tmp_path / "here.dxf").read_bytes()


def test_fonttools_logger_silenced_by_dxf_writer():
    """Issue 20: fontTools' ERROR about a malformed system font never reaches stderr."""
    import logging
    import src.drawing.writers.dxf  # noqa: F401
    assert logging.getLogger("fontTools").getEffectiveLevel() >= logging.CRITICAL
    assert logging.getLogger().getEffectiveLevel() < logging.CRITICAL   # only fontTools


def _poly_sheet():
    t = np.linspace(0.0, np.pi, 40)
    arc_pts = np.column_stack([100.0 + 30.0 * np.cos(t), 100.0 + 30.0 * np.sin(t)])
    square_rep = np.array([[20.0, 20.0], [60.0, 20.0], [60.0, 60.0], [20.0, 60.0], [20.0, 20.0]])
    tri = np.array([[150.0, 20.0], [190.0, 20.0], [170.0, 50.0]])
    two = np.array([[20.0, 150.0], [80.0, 160.0]])
    hatch_loops = _hatch_square()
    hatch_free = HatchLines(np.array([[[200.0, 150.0], [240.0, 150.0]],
                                      [[200.0, 153.0], [240.0, 153.0]],
                                      [[200.0, 156.0], [240.0, 156.0]]]), [])
    hatch_one = HatchLines(np.array([[[300.0, 20.0], [340.0, 60.0]]]),
                           [np.array([[300.0, 20.0], [340.0, 20.0], [340.0, 60.0]])])
    items = [Polyline(arc_pts, False, LineStyle.HIDDEN),        # open, many chords
             Polyline(square_rep, True, LineStyle.VISIBLE),     # closed, end point repeated
             Polyline(tri, True, LineStyle.SECTION),            # closed, no repeat
             Polyline(two, False, LineStyle.VISIBLE),           # open, 2 points
             hatch_loops, hatch_free, hatch_one]
    return Sheet("A3", 420.0, 297.0, items=items, meta={"scale": 1.0})


def test_svg_polylines_open_closed_and_hatches(tmp_path):
    sheet = _poly_sheet()
    res = write_svg(sheet, str(tmp_path / "p.svg"))
    root = ET.parse(str(tmp_path / "p.svg")).getroot()
    plines = list(root.iter(SVG_NS + "polyline"))
    pgons = [e for e in root.iter(SVG_NS + "polygon") if e.get("fill") == "none"]
    assert len(plines) == 2 and len(pgons) == 2
    hidden = [e for e in plines if e.get("stroke-dasharray")]
    assert len(hidden) == 1 and len(hidden[0].get("points").split()) == 40   # ONE element
    npts = sorted(len(e.get("points").split()) for e in pgons)
    assert npts == [3, 4]               # the repeated closing point is not written twice
    paths = list(root.iter(SVG_NS + "path"))
    n_segs = sorted(p.get("d").count("M ") for p in paths)
    assert n_segs == sorted([len(sheet.items[4].segments), 3, 1])
    assert res["elements"] == 7 and res["skipped"] == 0


def test_pdf_polylines_open_closed_and_hatches(tmp_path):
    sheet = _poly_sheet()
    write_pdf(sheet, str(tmp_path / "p.pdf"), compress=False)
    data = (tmp_path / "p.pdf").read_bytes()
    ops = _pdf_content(data, _pdf_objects(data)).decode("latin-1").split("\n")
    closed = [o for o in ops if o.endswith(" h S")]
    assert sorted(o.count(" l") for o in closed) == [2, 3]   # triangle, square without repeat
    long_ = [o for o in ops if o.count(" l") == 39]
    assert len(long_) == 1 and long_[0].endswith(" S") and not long_[0].endswith("h S")
    i = ops.index(long_[0])
    dash = [o for o in ops[:i] if o.endswith(" d")][-1]
    assert dash.split("[")[1].split("]")[0].strip() != ""     # dashed state before the hidden path
    n_hatch_paths = sum(1 for o in ops if o.endswith(" S") and o.count(" m ") >= 1
                        and o.count(" m ") == o.count(" l") and o.count(" m ") in
                        (len(sheet.items[4].segments), 3, 1))
    assert n_hatch_paths >= 3


@needs_ezdxf
def test_dxf_polylines_open_closed_plinegen_and_hatches(tmp_path):
    import ezdxf
    sheet = _poly_sheet()
    res = write_dxf(sheet, str(tmp_path / "p.dxf"))
    doc = ezdxf.readfile(str(tmp_path / "p.dxf"))
    aud = doc.audit()
    assert len(aud.errors) == 0 and len(aud.fixes) == 0
    msp = doc.modelspace()
    lw = list(msp.query("LWPOLYLINE"))
    assert len(lw) == 4
    by_n = {len(e): e for e in lw}
    assert sorted(by_n) == [2, 3, 4, 40]          # square written with 4 vertices, not 5
    assert by_n[4].closed and by_n[3].closed and not by_n[40].closed and not by_n[2].closed
    hidden = by_n[40]
    assert hidden.dxf.layer == "HIDDEN"
    # PLINEGEN (flag 128): the DASHED pattern runs over the whole polyline instead of
    # restarting at every chord vertex (a 40-chord curve would otherwise look solid)
    assert hidden.dxf.flags & 128
    assert len(list(msp.query("HATCH"))) == 2      # hatch with loops + the one-segment hatch
    assert len(list(msp.query("LINE[layer=='HATCH']"))) == 3   # loop-less hatch -> lines
    assert res["skipped"] == 0


@needs_ezdxf
def test_dxf_stabilise_leaves_marker_looking_text_alone(tmp_path):
    """Review F5: only the ezdxf DICTIONARYVAR markers are rewritten, not a TEXT
    entity whose content happens to look like ``x.y.z @ <iso time>``."""
    import ezdxf
    lookalike = "1.2.3 @ 2019-05-06T07:08:09+00:00"
    sheet = make_sheet()
    sheet.items.append(Text((60.0, 30.0), lookalike, 3.5, anchor="start"))
    write_dxf(sheet, str(tmp_path / "t.dxf"), date="2026-01-02")
    doc = ezdxf.readfile(str(tmp_path / "t.dxf"))
    texts = [e.dxf.text for e in doc.modelspace().query("TEXT")]
    assert lookalike in texts                                   # the user text is intact
    # precondition / no-op guard: the real markers were still stamped with the date
    raw = (tmp_path / "t.dxf").read_text(encoding="utf-8")
    markers = re.findall(r"^\d+\.\d+\.\d+\S* @ \d{4}-\d\d-\d\dT\S*$", raw, re.M)
    assert len(markers) == 3                                    # 2 ezdxf markers + the text
    assert sum(1 for m in markers if "@ 2026-01-02T00:00:00+00:00" in m) == 2


# ----------------------------------------------- QA round 2: dashes, joins, DXF linetypes

from src.drawing.style import DEFAULT_STYLE, SheetStyle  # noqa: E402


def _dash_pieces(length, pattern, closed):
    """(dashes, gaps) that a dash ``pattern`` starting at phase 0 shows over ``length``.

    Pieces shorter than 0.05 mm are not counted. An open element's trailing gap is
    invisible (nothing follows it); on a closed curve a final dash runs into the
    first one and the two count once.
    """
    pattern = [float(v) for v in pattern]
    assert sum(pattern) > 0
    pieces, pos, i = [], 0.0, 0
    while pos < length - 1e-9:
        v = pattern[i % len(pattern)]
        seg = min(v, length - pos)
        if seg > 1e-9:
            pieces.append((i % 2 == 0, seg))
        pos += v
        i += 1
    if not closed:
        while pieces and not pieces[-1][0]:
            pieces.pop()
    elif len(pieces) > 1 and pieces[0][0] and pieces[-1][0]:
        pieces.pop()
    dashes = sum(1 for d, ln in pieces if d and ln >= 0.05)
    gaps = sum(1 for d, ln in pieces if not d and ln >= 0.05)
    return dashes, gaps


def _short_hidden_sheet():
    t = np.linspace(0.0, 1.0, 6)
    short_pl = np.column_stack([300.0 + 4.0 * t, 100.0 + 0.5 * np.sin(3 * t)])
    items = [Segment((50.0, 50.0), (52.0, 50.0), LineStyle.HIDDEN),          # 2 mm
             Circle((100.0, 100.0), 1.0, LineStyle.HIDDEN),                   # 6.28 mm closed
             Segment((50.0, 200.0), (100.0, 200.0), LineStyle.HIDDEN),        # 50 mm
             Arc((200.0, 100.0), 1.0, 0.0, 90.0, LineStyle.HIDDEN),           # 1.57 mm
             Polyline(short_pl, False, LineStyle.HIDDEN)]                     # ~4.1 mm open
    lengths = [2.0, 2.0 * np.pi, 50.0, 0.5 * np.pi,
               float(np.linalg.norm(np.diff(short_pl, axis=0), axis=1).sum())]
    closed = [False, True, False, False, False]
    return Sheet("A3", 420.0, 297.0, items=items, meta={"scale": 1.0}), lengths, closed


def _check_short_patterns(patterns, lengths, closed):
    std = tuple(DEFAULT_STYLE.dash_hidden)
    # precondition: the stock pattern shows a solid stroke on the short elements
    assert _dash_pieces(lengths[0], std, False)[0] < 2
    assert _dash_pieces(lengths[1], std, True)[1] < 4
    assert _dash_pieces(lengths[3], std, False)[0] < 2
    for k in (0, 3, 4):
        assert _dash_pieces(lengths[k], patterns[k], False)[0] >= 2, (k, patterns[k])
    assert _dash_pieces(lengths[1], patterns[1], True)[1] >= 4, patterns[1]
    assert tuple(patterns[2]) == pytest.approx(std, abs=1e-3)               # long line unchanged


def test_svg_short_hidden_elements_get_adapted_dashes(tmp_path):
    """QA2 issue 2: every short hidden element shows >= 2 dashes, a closed one >= 4 gaps."""
    sheet, lengths, closed = _short_hidden_sheet()
    res = write_svg(sheet, str(tmp_path / "h.svg"))
    root = ET.parse(str(tmp_path / "h.svg")).getroot()
    g = [e for e in root if e.tag == SVG_NS + "g"][0]
    drawn = list(g)
    assert len(drawn) == 5
    patterns = [[float(v) for v in e.get("stroke-dasharray").split()] for e in drawn]
    _check_short_patterns(patterns, lengths, closed)
    assert drawn[2].get("stroke-dasharray") == "3 0.75"
    assert res["dash_adapted"] == 4


def test_pdf_short_hidden_elements_get_adapted_dashes(tmp_path):
    sheet, lengths, closed = _short_hidden_sheet()
    res = write_pdf(sheet, str(tmp_path / "h.pdf"), compress=False)
    data = (tmp_path / "h.pdf").read_bytes()
    ops = _pdf_content(data, _pdf_objects(data)).decode("latin-1").split("\n")
    patterns, cur = [], None
    for op in ops:
        m = re.search(r"\[([^\]]*)\] 0 d", op)
        if m:
            cur = [float(v) / MM for v in m.group(1).split()]
        if op.endswith(" S"):
            patterns.append(cur)
    assert len(patterns) == 5
    _check_short_patterns(patterns, lengths, closed)
    assert res["dash_adapted"] == 4


@needs_ezdxf
def test_dxf_short_hidden_elements_entity_ltscale(tmp_path):
    """DXF: stock HIDDEN_ISO linetype, short entities scaled by their own ltscale; the
    hidden circle stays ONE CIRCLE on layer HIDDEN."""
    import ezdxf
    sheet, lengths, closed = _short_hidden_sheet()
    res = write_dxf(sheet, str(tmp_path / "h.dxf"))
    assert res["audit_errors"] == 0 and res["audit_fixes"] == 0
    doc = ezdxf.readfile(str(tmp_path / "h.dxf"))
    ents = list(doc.modelspace())
    assert [e.dxftype() for e in ents] == ["LINE", "CIRCLE", "LINE", "ARC", "LWPOLYLINE"]
    assert all(e.dxf.layer == "HIDDEN" for e in ents)
    base = doc.linetypes.get(doc.layers.get("HIDDEN").dxf.linetype).simplified_line_pattern()
    g = float(doc.header.get("$LTSCALE", 1.0))
    patterns = [[v * g * float(e.dxf.get("ltscale", 1.0)) for v in base] for e in ents]
    _check_short_patterns(patterns, lengths, closed)
    assert res["dash_adapted"] == 4


def test_line_props_adapts_hidden_only():
    from src.drawing.writers import line_props
    w, d = line_props(LineStyle.HIDDEN, DEFAULT_STYLE, length=2.0)
    assert _dash_pieces(2.0, d, False)[0] >= 2
    assert line_props(LineStyle.HIDDEN) == (DEFAULT_STYLE.narrow, tuple(DEFAULT_STYLE.dash_hidden))
    assert line_props(LineStyle.VISIBLE, length=2.0) == (DEFAULT_STYLE.wide, None)
    assert line_props(LineStyle.CENTER, length=2.0)[1] == tuple(DEFAULT_STYLE.dash_center)
    # a closed curve whose length is a whole number of periods keeps the pattern
    w, d = line_props(LineStyle.HIDDEN, length=7.5 * 4, closed=True)
    assert d == pytest.approx(DEFAULT_STYLE.dash_hidden)


def _hairpin_sheet():
    a = np.radians(10.0)
    pts = np.array([[100.0 + 40.0 * np.cos(a), 100.0 + 40.0 * np.sin(a)], [100.0, 100.0],
                    [140.0, 100.0]])
    return Sheet("A3", 420.0, 297.0, items=[Polyline(pts, False, LineStyle.VISIBLE)],
                 meta={"scale": 1.0})


def test_svg_round_line_joins(tmp_path):
    """QA2 issue 20: no miter spikes at acute polyline vertices."""
    write_svg(_hairpin_sheet(), str(tmp_path / "j.svg"))
    root = ET.parse(str(tmp_path / "j.svg")).getroot()
    g = [e for e in root if e.tag == SVG_NS + "g"][0]
    assert g.get("stroke-linejoin") == "round"


def test_pdf_round_line_joins(tmp_path):
    write_pdf(_hairpin_sheet(), str(tmp_path / "j.pdf"), compress=False)
    data = (tmp_path / "j.pdf").read_bytes()
    ops = _pdf_content(data, _pdf_objects(data)).decode("latin-1").split("\n")
    first_stroke = next(i for i, o in enumerate(ops) if o.endswith(" S"))
    joins = [m.group(1) for o in ops[:first_stroke] for m in re.finditer(r"(\d) j\b", o)]
    assert joins and joins[-1] == "1"


def _iso_pattern(dash):
    return tuple(float(v) for v in dash)


@needs_ezdxf
@pytest.mark.parametrize("style", [DEFAULT_STYLE,
                                   SheetStyle(dash_hidden=(4.0, 1.0),
                                              dash_center=(8.0, 1.0, 0.5, 1.0),
                                              dash_cutting=(10.0, 2.0, 0.5, 2.0))])
def test_dxf_iso_linetypes_from_sheet_style(tmp_path, style):
    """QA2 issue 22: HIDDEN/CENTER/CUTTING layers use custom ISO linetypes built from
    the SheetStyle dashes at $LTSCALE 1 (stock DASHED/CENTER differ: 1.27/0.635 and
    31.75/6.35/6.35/6.35 at scale 1)."""
    import ezdxf
    res = write_dxf(make_sheet(), str(tmp_path / "lt.dxf"), style)
    assert res["audit_errors"] == 0 and res["audit_fixes"] == 0
    doc = ezdxf.readfile(str(tmp_path / "lt.dxf"))
    assert float(doc.header["$LTSCALE"]) == pytest.approx(1.0)
    for layer, name, dash in (("HIDDEN", "HIDDEN_ISO", style.dash_hidden),
                              ("CENTER", "CENTER_ISO", style.dash_center),
                              ("CUTTING", "CUTTING_ISO", style.dash_cutting)):
        assert doc.layers.get(layer).dxf.linetype.upper() == name
        pat = doc.linetypes.get(name).simplified_line_pattern()
        assert tuple(pat) == pytest.approx(_iso_pattern(dash), abs=1e-6)
    # no-op guard: the stock DASHED pattern is not the ISO hidden pattern
    stock = doc.linetypes.get("DASHED").simplified_line_pattern()
    assert tuple(stock) != pytest.approx(_iso_pattern(style.dash_hidden), abs=1e-6)
