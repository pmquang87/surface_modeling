"""2D sheet model of the drawing module: line styles, primitives, sheet.

Everything the writers render is a plain dataclass in this module, so the
B-rep route, the mesh route, the dimensioning and the layout all speak one
vocabulary. Conventions (spec 3.1, plan "Shared interfaces"):

* Units: primitives in a ``ProjectedView`` / ``SectionView`` are in view
  coordinates (model millimetres, not shifted). Everything in a ``Sheet`` is
  in sheet millimetres. Origin bottom-left, Y up, everywhere.
* ``transform_primitive`` maps a view primitive onto the sheet
  (``p_sheet = offset + scale * p_view``). Text height is NOT scaled: text
  stays sheet-sized (3.5 mm on the sheet whatever the drawing scale).
* ``BBox2D.of_primitives`` boxes an ``Arc`` by its two end points plus
  the axis quadrant points (0 / 90 / 180 / 270 degrees) that lie inside its
  counter-clockwise sweep (``arc_extreme_points``); only a ``Circle`` (or
  an arc whose start equals its end, the DXF full-circle rule) gets the
  full circle box. The old full-circle rule for every arc inflated the
  view boxes (QA round 1: a screw-head dome R 55 boxed 109.75 mm instead
  of 42.3 mm), and the dimension values and the scale came from them.
* Text is boxed by an ESTIMATE (``text_corners``): advance
  ``TEXT_ADVANCE * height`` per character (ISO 3098 type B is about 0.6 h
  on average, wide glyphs included), and the line spans from
  ``TEXT_DESCENT * height`` below the baseline to ``height`` above it. The
  writers' real fonts differ a little; layout keeps a margin anyway.
* ``LineStyle`` names the ISO 128-24 line types of line group 0.5 (wide 0.5,
  narrow 0.25); widths and dash patterns live in ``style.SheetStyle``
  (sourced in the standards probe: ISO 128-20 Table 3 dash lengths are
  multiples of the line width, hidden 3.0/0.75 at d = 0.25).

No OCP, trimesh or matplotlib import here: only numpy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple, Union

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - type reference only
    from .projection import ViewSpec

Point = Tuple[float, float]


class LineStyle(str, Enum):
    VISIBLE = "visible"      # wide 0.5: visible edges and silhouettes
    HIDDEN = "hidden"        # narrow 0.25 dashed 3/0.75
    SECTION = "section"      # wide 0.5: section loop outlines
    HATCH = "hatch"          # narrow
    CENTER = "center"        # narrow long-dash-dot 6/0.75/0.125/0.75
    DIMENSION = "dimension"  # narrow: extension, dimension, leader lines; arrowheads are Polygon
    CUTTING = "cutting"      # wide long-dash-dot 12/1.5/0.25/1.5 (ISO 04.2)
    FRAME = "frame"          # 0.7
    TITLE = "title"          # narrow: title block inner lines, strip
    TEXT = "text"


@dataclass
class Segment:
    a: Point
    b: Point
    style: LineStyle = LineStyle.VISIBLE


@dataclass
class Polyline:
    points: np.ndarray            # (n, 2) float64
    closed: bool = False
    style: LineStyle = LineStyle.VISIBLE


@dataclass
class Circle:
    center: Point
    radius: float
    style: LineStyle = LineStyle.VISIBLE


@dataclass
class Arc:
    center: Point
    radius: float
    start_deg: float              # counter-clockwise from start to end, degrees
    end_deg: float
    style: LineStyle = LineStyle.VISIBLE


@dataclass
class Polygon:                    # filled (arrowheads, projection symbol fills)
    points: np.ndarray
    style: LineStyle = LineStyle.DIMENSION


@dataclass
class Text:
    position: Point
    text: str
    height: float
    rotation_deg: float = 0.0     # 0 = reads from the bottom, 90 = reads from the right
    anchor: str = "middle"        # "start" | "middle" | "end" (along the baseline)
    valign: str = "baseline"      # "baseline" | "middle"
    style: LineStyle = LineStyle.TEXT


@dataclass
class HatchLines:
    segments: np.ndarray          # (k, 2, 2)
    loops: List[np.ndarray] = field(default_factory=list)   # boundary loops (for the DXF HATCH entity)
    style: LineStyle = LineStyle.HATCH


@dataclass
class Dimension:                  # semantic; coordinates in the space stated by the holder
    kind: str                     # "linear" | "diameter"
    text: str                     # "60", "4 x Ø6.5", "Ø20"
    value: float                  # model mm
    p1: Point = (0.0, 0.0)        # linear: measured points
    p2: Point = (0.0, 0.0)
    base: Point = (0.0, 0.0)      # linear: a point on the dimension line
    angle_deg: float = 0.0        # linear: 0 horizontal, 90 vertical
    center: Point = (0.0, 0.0)    # diameter
    radius: float = 0.0
    leader_angle_deg: float = 45.0  # diameter: leader direction
    view: str = ""                # name of the view it belongs to
    count: int = 1
    # linear: shift of the value text along the dimension line (sheet mm, signed, in the
    # direction p1 -> p2 projected on the line); the layout sets it when the centred text
    # would sit on a section cutting line (dimensions.linear_text_position gives the result)
    text_offset: float = 0.0


Primitive2D = Union[Segment, Polyline, Circle, Arc]
SheetItem = Union[Segment, Polyline, Circle, Arc, Polygon, Text, HatchLines]


def _as_points(pts) -> np.ndarray:
    arr = np.asarray(pts, dtype=np.float64)
    if arr.size == 0:
        return arr.reshape(0, 2)
    return arr.reshape(-1, 2)


def arc_sweep_deg(start_deg: float, end_deg: float) -> float:
    """CCW sweep in degrees; start == end (mod 360) means a full circle (DXF rule)."""
    sweep = float(np.mod(float(end_deg) - float(start_deg), 360.0))
    if sweep < 1e-9 or sweep > 360.0 - 1e-9:
        return 360.0
    return sweep


def arc_extreme_points(center: Point, radius: float, start_deg: float, end_deg: float) -> np.ndarray:
    """(k, 2) points that bound an arc: both end points plus every axis
    quadrant point (0, 90, 180, 270 degrees) inside the CCW sweep. A full
    sweep gives the four quadrant points."""
    cx, cy = float(center[0]), float(center[1])
    r = abs(float(radius))
    sweep = arc_sweep_deg(start_deg, end_deg)
    if sweep >= 360.0:
        angs = [0.0, 90.0, 180.0, 270.0]
    else:
        s0 = float(np.mod(float(start_deg), 360.0))
        angs = [s0, s0 + sweep]
        for q in (0.0, 90.0, 180.0, 270.0):
            if float(np.mod(q - s0, 360.0)) <= sweep + 1e-12:
                angs.append(q)
    a = np.deg2rad(np.asarray(angs, dtype=np.float64))
    return np.column_stack([cx + r * np.cos(a), cy + r * np.sin(a)])


TEXT_ADVANCE = 0.6   # estimated character advance, fraction of the text height
TEXT_DESCENT = 0.25  # estimated descender depth below the baseline, fraction of height


def text_corners(t: "Text") -> np.ndarray:
    """(4, 2) corners of the estimated extent of ``t`` in its own coordinates.

    Width ``TEXT_ADVANCE * height * len(text)``; ``anchor`` places the
    anchor at the start, middle or end of the baseline; ``valign``
    "baseline" spans ``-TEXT_DESCENT * h .. h`` around the anchor, "middle"
    spans ``-h/2 .. h/2``. The rectangle is then rotated by
    ``rotation_deg`` about the anchor. Empty text gives a zero-width box.
    """
    h = float(t.height)
    w = TEXT_ADVANCE * h * len(t.text or "")
    x0 = {"start": 0.0, "end": -w}.get(t.anchor, -w / 2.0)
    if t.valign == "middle":
        y0, y1 = -h / 2.0, h / 2.0
    else:
        y0, y1 = -TEXT_DESCENT * h, h
    local = np.array([[x0, y0], [x0 + w, y0], [x0 + w, y1], [x0, y1]], dtype=np.float64)
    a = np.deg2rad(float(t.rotation_deg))
    c, s = np.cos(a), np.sin(a)
    rot = np.array([[c, -s], [s, c]])
    return local @ rot.T + np.asarray(t.position, dtype=np.float64)


@dataclass
class BBox2D:
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def width(self) -> float:
        return float(self.xmax - self.xmin)

    @property
    def height(self) -> float:
        return float(self.ymax - self.ymin)

    @classmethod
    def from_points(cls, pts: np.ndarray) -> "BBox2D":
        arr = _as_points(pts)
        if arr.shape[0] == 0:
            raise ValueError("BBox2D.from_points needs at least one point")
        lo = arr.min(axis=0)
        hi = arr.max(axis=0)
        return cls(float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1]))

    @classmethod
    def of_primitives(cls, prims: Sequence) -> Optional["BBox2D"]:
        """Box of Segment/Polyline/Circle/Arc/Polygon/HatchLines/Text items.

        A Circle uses its full extent; an Arc its end points plus the
        quadrant points inside its sweep (exact, see the module docstring).
        Text contributes the four corners of its estimated extent (see
        ``text_corners``). Returns None for an empty sequence or one without
        any point.
        """
        chunks: List[np.ndarray] = []
        for p in prims:
            if isinstance(p, Segment):
                chunks.append(_as_points([p.a, p.b]))
            elif isinstance(p, Arc):
                chunks.append(arc_extreme_points(p.center, p.radius, p.start_deg, p.end_deg))
            elif isinstance(p, Circle):
                cx, cy = float(p.center[0]), float(p.center[1])
                r = abs(float(p.radius))
                chunks.append(np.array([[cx - r, cy - r], [cx + r, cy + r]]))
            elif isinstance(p, (Polyline, Polygon)):
                chunks.append(_as_points(p.points))
            elif isinstance(p, HatchLines):
                chunks.append(_as_points(p.segments))
                for loop in p.loops:
                    chunks.append(_as_points(loop))
            elif isinstance(p, Text):
                chunks.append(text_corners(p))
            else:
                raise TypeError(f"BBox2D.of_primitives: unsupported item {type(p).__name__}")
        chunks = [c for c in chunks if c.shape[0]]
        if not chunks:
            return None
        return cls.from_points(np.vstack(chunks))

    def union(self, other: "BBox2D") -> "BBox2D":
        return BBox2D(min(self.xmin, other.xmin), min(self.ymin, other.ymin),
                      max(self.xmax, other.xmax), max(self.ymax, other.ymax))

    def expanded(self, margin: float) -> "BBox2D":
        return BBox2D(self.xmin - margin, self.ymin - margin,
                      self.xmax + margin, self.ymax + margin)

    def contains(self, p: Point) -> bool:
        x, y = float(p[0]), float(p[1])
        return self.xmin <= x <= self.xmax and self.ymin <= y <= self.ymax


@dataclass
class Sheet:
    size_name: str                # "A3" | "A4" | "A4L"
    width: float                  # mm
    height: float
    items: List[SheetItem] = field(default_factory=list)       # ALL geometry, in sheet mm
    dimensions: List[Dimension] = field(default_factory=list)  # sheet mm, for the DXF DIMENSION entities
    meta: dict = field(default_factory=dict)                   # title fields, scale text, projection ...


def _tp(p: Point, offset: Point, scale: float) -> Point:
    return (float(offset[0]) + scale * float(p[0]), float(offset[1]) + scale * float(p[1]))


def _tarr(pts: np.ndarray, offset: Point, scale: float) -> np.ndarray:
    arr = np.asarray(pts, dtype=np.float64)
    off = np.array([float(offset[0]), float(offset[1])])
    return arr * scale + off


def transform_primitive(prim, offset: Point, scale: float):
    """Return a view -> sheet copy: ``p_sheet = offset + scale * p_view``.

    Handles Segment, Polyline, Circle, Arc, Polygon, HatchLines and Text.
    Radii scale; arc angles are kept (scale must be positive, no mirroring).
    Text keeps its height and rotation (text stays sheet-sized).
    """
    scale = float(scale)
    if scale <= 0:
        raise ValueError("transform_primitive: scale must be positive")
    if isinstance(prim, Segment):
        return Segment(_tp(prim.a, offset, scale), _tp(prim.b, offset, scale), prim.style)
    if isinstance(prim, Polyline):
        return Polyline(_tarr(prim.points, offset, scale), prim.closed, prim.style)
    if isinstance(prim, Circle):
        return Circle(_tp(prim.center, offset, scale), float(prim.radius) * scale, prim.style)
    if isinstance(prim, Arc):
        return Arc(_tp(prim.center, offset, scale), float(prim.radius) * scale,
                   prim.start_deg, prim.end_deg, prim.style)
    if isinstance(prim, Polygon):
        return Polygon(_tarr(prim.points, offset, scale), prim.style)
    if isinstance(prim, HatchLines):
        segs = np.asarray(prim.segments, dtype=np.float64)
        new_segs = _tarr(segs, offset, scale) if segs.size else segs.reshape(0, 2, 2).copy()
        return HatchLines(new_segs, [_tarr(lp, offset, scale) for lp in prim.loops], prim.style)
    if isinstance(prim, Text):
        return Text(_tp(prim.position, offset, scale), prim.text, prim.height,
                    prim.rotation_deg, prim.anchor, prim.valign, prim.style)
    raise TypeError(f"transform_primitive: unsupported item {type(prim).__name__}")


# --- shared result objects (both geometry sources produce these) -----------

@dataclass
class ProjectedView:
    spec: "ViewSpec"
    visible: List[Primitive2D]
    hidden: List[Primitive2D]
    bbox: Optional[BBox2D]
    algorithm: str                 # "hlr_exact" | "hlr_poly" | "depth_buffer"
    stats: dict = field(default_factory=dict)


@dataclass
class SectionView:
    spec: "ViewSpec"
    plane_point: np.ndarray        # world
    plane_normal: np.ndarray       # world, points toward the viewer (= spec.n)
    loops: List[np.ndarray]        # closed polygons in view coords, no repeated end point
    body: Optional[ProjectedView]
    label: str = "A"
    marker_view: str = "top"       # principal view that shows the cutting line
    stats: dict = field(default_factory=dict)   # open_wires_dropped, nudged, seconds


@dataclass
class CylFeature:
    center: np.ndarray
    axis: np.ndarray
    radius: float
    length: float
    kind: str                      # "hole" | "boss" | "fillet"
    full_circle: bool
    n_faces: int
