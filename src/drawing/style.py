"""Sheet sizes, scales and every ISO number the layout and writers use.

Sources (standards probe, read from the BIS adoptions of the ISO texts):

* ISO 5457:1999 - A3 420 x 297 landscape, A4 210 x 297 PORTRAIT only
  (4.1: "Only sheets positioned vertically are allowed"); A4 landscape is
  offered as ``A4L`` with ``iso5457=False``. Border 20 mm left (filing
  margin), 10 mm elsewhere, frame 0.7 mm; centring marks 0.7 mm from about
  5 mm outside to 10 mm inside the frame.
* ISO 7200:2004 - title block 180 mm wide; rows of 9 mm measured from the
  Fig. 1/2 vector lines; columns at 0 | 32 | 74.2 | 129 | 180 mm; 4 rows =
  36 mm. Scale and projection symbol go OUTSIDE the block (9 mm strip).
* ISO 128-20 Table 3 / ISO 128-24 line group 0.5 - wide 0.5, narrow 0.25;
  dash patterns in multiples of the line width d: hidden 02.1 = 12d/3d =
  3.0/0.75; centre 04.1 = 24d/3d/0.5d/3d = 6/0.75/0.125/0.75 at d = 0.25;
  cutting 04.2 at d = 0.5 = 12/1.5/0.25/1.5.
* ISO 3098-0 - type B text 3.5 mm, stroke h/10 = 0.35; ISO 128-40 section
  letters sqrt(2) x 3.5 ~ 5 mm, filled 30 degree section arrow head.
* ISO 5455 - scales 50:1 ... 1:100 (``STANDARD_SCALES``).
* ISO 5456-2 Table A.1 - projection symbol parameter h = text height.
* ISO 129 - closed filled arrowhead, 15..90 degree included angle (15 used).

Conventional (DIN 406 teaching practice, NOT ISO; the report must not cite
ISO for these): extension gap 1 mm, overshoot 2 mm, dimension lines 10 mm
then 7 mm pitch, text 1 mm above the line, 20 mm view gap, hatch spacing
2 / 3 / 4 mm by the section's shorter extent (< 20, <= 80, above), minimum
hatch spacing 1.0 mm (= max(0.7, 2 x wide)); thin regions down to 0.7 mm and
solid fill below a mean body thickness of 2.1 mm (``section_fill_below``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class SheetSize:
    name: str
    width: float
    height: float
    iso5457: bool
    orientation: str


SHEETS = {"A3": SheetSize("A3", 420, 297, True, "landscape"),
          "A4": SheetSize("A4", 210, 297, True, "portrait"),
          "A4L": SheetSize("A4L", 297, 210, False, "landscape")}

# sheet mm per model mm, strictly decreasing (ISO 5455)
STANDARD_SCALES = [50, 20, 10, 5, 2, 1, 1 / 2, 1 / 5, 1 / 10, 1 / 20, 1 / 50, 1 / 100]


def _fmt_ratio(v: float) -> str:
    r = round(v)
    if abs(v - r) < 1e-6 * max(1.0, abs(v)):
        return str(int(r))
    return f"{v:.6g}"


def scale_text(scale: float) -> str:
    """ISO 5455 designation: 2 -> "2:1", 1 -> "1:1", 0.5 -> "1:2"."""
    scale = float(scale)
    if not scale > 0:
        raise ValueError(f"scale_text: scale must be positive, got {scale!r}")
    if scale >= 1.0 - 1e-9:
        return f"{_fmt_ratio(scale)}:1"
    return f"1:{_fmt_ratio(1.0 / scale)}"


@dataclass(frozen=True)
class SheetStyle:
    margin_left: float = 20.0
    margin: float = 10.0
    frame_width: float = 0.7
    centring_mark_outside: float = 5.0
    centring_mark_inside: float = 10.0
    wide: float = 0.5
    narrow: float = 0.25
    text_stroke: float = 0.35
    dash_hidden: Tuple[float, ...] = (3.0, 0.75)
    dash_center: Tuple[float, ...] = (6.0, 0.75, 0.125, 0.75)
    dash_cutting: Tuple[float, ...] = (12.0, 1.5, 0.25, 1.5)
    text_height: float = 3.5
    label_height: float = 5.0
    title_label_height: float = 2.5
    arrow_length: float = 3.5
    arrow_angle_deg: float = 15.0
    ext_gap: float = 1.0
    ext_overshoot: float = 2.0
    dim_first_offset: float = 10.0
    dim_pitch: float = 7.0
    dim_text_gap: float = 1.0
    view_gap: float = 20.0
    hatch_spacing_small: float = 2.0
    hatch_spacing_mid: float = 3.0
    hatch_spacing_large: float = 4.0
    title_block_w: float = 180.0
    title_block_h: float = 36.0
    title_row_h: float = 9.0
    title_cols: Tuple[float, ...] = (0.0, 32.0, 74.2, 129.0, 180.0)
    strip_h: float = 9.0
    section_arrow_len: float = 10.0
    section_head_len: float = 5.0
    section_head_angle_deg: float = 30.0
    section_dash_len: float = 12.0
    symbol_h: float = 3.5           # projection symbol size parameter (= text height)


DEFAULT_STYLE = SheetStyle()

_HATCH_MIN = 1.0
_HATCH_SMALL_BELOW = 20.0
_HATCH_MID_UP_TO = 80.0


def hatch_spacing(style: SheetStyle, shorter_extent_sheet_mm: float) -> float:
    """2 mm below 20 mm shorter extent, 3 mm up to 80 mm, 4 mm above; >= 1.0."""
    e = float(shorter_extent_sheet_mm)
    if e < _HATCH_SMALL_BELOW:
        s = style.hatch_spacing_small
    elif e <= _HATCH_MID_UP_TO:
        s = style.hatch_spacing_mid
    else:
        s = style.hatch_spacing_large
    return max(_HATCH_MIN, float(s))


# Thin section regions (QA round 1 issue 29, round 2 issue 3; conventions, not ISO
# numbers): the fill decision is made per BODY from its mean wall thickness on the
# sheet, t = 2 A / P over all its loops (A = outer areas minus hole areas, P = all
# loop perimeters; a long strip of width w gives t ~ w, an annulus gives its radial
# width). A body with t below ``section_fill_below()`` (HATCH_FILL_BELOW, raised to
# 3 x HATCH_MIN_REGION so a hatched region always gets spacing <= t / 3) is filled
# solid (ISO 128-50 allows blackening thin sections); otherwise every region of the
# body is hatched, a region thinner than two section spacings at t / 3, never below
# HATCH_MIN_REGION (``thin_region_spacing``).
# Review R4 (two guards):
# * a region of a hatched body is still filled when its own t is below
#   ``region_fill_below()`` (2 x HATCH_MIN_REGION = 1.4 mm) or its hatch has fewer than
#   HATCH_MIN_SEGMENTS lines: a 0.8 mm island with one hatch line reads as an empty pocket;
# * a body whose 2A/P is thin is NOT filled when more than THICK_CORE_FRACTION of its area
#   lies deeper than ``section_fill_below()`` inside the material (local thickness above
#   about 4 mm): the mean 2A/P of a finned heat sink is pulled below 2.1 mm by the fins,
#   and filling it would blacken its 10 mm base.
HATCH_FILL_BELOW = 2.0
HATCH_MIN_REGION = 0.7
HATCH_MIN_SEGMENTS = 2
THICK_CORE_FRACTION = 0.10

# Lettering minimum (ISO 3098 / ISO 7200 practice: 2.5 mm for notes on the sheet); the strip
# sets a note on two lines rather than shrink it below this height (review R4).
MIN_LETTERING_HEIGHT = 2.5


def section_fill_below() -> float:
    """Mean body thickness (sheet mm) below which a section body is filled solid: 2.1 mm."""
    return max(HATCH_FILL_BELOW, 3.0 * HATCH_MIN_REGION)


def region_fill_below() -> float:
    """Region thickness (2A/P, sheet mm) below which a region of a hatched body is filled: 1.4 mm."""
    return 2.0 * HATCH_MIN_REGION


def thin_region_spacing(section_spacing: float, thickness_sheet_mm: float) -> float:
    """Hatch spacing of one region of mean thickness ``t`` (2A/P, sheet mm) in a hatched body.

    The section spacing when ``t >= 2 * spacing``, else ``t / 3`` floored at
    ``HATCH_MIN_REGION`` (0.7 mm) and capped at the section spacing.
    """
    s = float(section_spacing)
    t = float(thickness_sheet_mm)
    if t >= 2.0 * s:
        return s
    return max(HATCH_MIN_REGION, min(s, t / 3.0))


def region_hatch_spacing(section_spacing: float, across_sheet_mm: float) -> float:
    """Spacing for one section region ``across_sheet_mm`` wide across the hatch lines.

    The section spacing when the region takes at least two lines at it,
    else ``across / 3`` floored at ``HATCH_MIN_REGION`` (0.7 mm).
    """
    s = float(section_spacing)
    w = float(across_sheet_mm)
    if w >= 2.0 * s:
        return s
    return max(HATCH_MIN_REGION, min(s, w / 3.0))
