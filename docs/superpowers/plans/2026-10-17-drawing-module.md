# STEP / STL to 2D drawing module — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `python -m src.drawing part.(step|stl) out/` writes `<stem>.pdf/.svg/.dxf` (orthographic views with hidden lines, dimensions, one hatched section, ISO sheet with title block) plus `<stem>.report.json`, routed STEP -> B-rep HLR or STL -> faceted STEP -> B-rep HLR with a trimesh fallback, executed in a sandboxed child process.

**Architecture:** A `src/drawing/` package. Two geometry sources (OCCT B-rep, trimesh mesh) produce the same `ProjectedView` / `SectionView` / `CylFeature` objects; `dimensions.py` and `layout.py` turn them into a `Sheet` of 2D primitives in sheet millimetres; three writers render the `Sheet`; `report.py` assembles the quality report; `sandbox.py` runs the whole child under a Windows job object (POSIX rlimits unverified) and keeps the `RESULT {json}` stdout contract of `src/convert.py`.

**Tech Stack:** Python 3.14, numpy, OCP 7.9.3 (cadquery-ocp), trimesh 5.0 (no shapely), scikit-image (contours), ezdxf 1.4 (DXF), ctypes + psutil (sandbox), pytest. No new heavy dependencies (no matplotlib, no shapely).

Spec: `docs/superpowers/specs/2026-10-17-drawing-module-design.md`. Probe reports with verified, runnable snippets (read the one for your task before coding; copy from the snippets, they ran on this machine):
`C:\Users\pmqua\AppData\Local\Temp\claude\C--Users-pmqua-PycharmProjects-surface-modeling\d068f437-debc-4f61-84c7-e3504e1aa0c0\scratchpad\probes\REPORT-{hlr,brep-analysis,mesh,writers,sandbox,standards}.md` and the scripts next to them.

## Global Constraints

- Python interpreter for everything: `C:\Python314\python.exe` (`python` on PATH). Run tests from the repo root: `python -m pytest tests/test_drawing_<area>.py -q`.
- Every script or test that prints non-ASCII starts with `sys.stdout.reconfigure(encoding="utf-8", errors="replace")` (console is cp1252). Module code must not print to stdout at all: progress goes through `logging` or the `log` callable; stdout belongs to the `RESULT` line.
- OCP imports: flat `from OCP.<Module> import <Name>` only, never `OCP.OCP`; every name must resolve on 7.9.3 (`tests/test_ocp_layout.py` scans `src`). Catch `Exception` around OCCT calls (OCP exception classes are not `Standard_Failure` subclasses). Call `src.io.occt_utils.quiet_occt()` before any STEP read/write.
- Never call `TransferRoots`/`OneShape`/`NbRootsForTransfer` unless `ReadFile(...) == IFSelect_RetDone` (native crash otherwise).
- No shapely, no matplotlib in `src/drawing`. `ezdxf` is imported lazily inside the DXF writer; `psutil` lazily inside `sandbox.py` (optional).
- Units: model millimetres in; sheet millimetres out; origin bottom-left, Y up, everywhere in `model.py`.
- Test discipline (repo rule): every defect/negative test asserts its precondition first and must fail against a no-op or constant implementation; mark OCP tests with `needs_ocp = pytest.mark.skipif(not HAS_OCP, ...)` like `tests/test_convert_cli.py`. Keep each test module under ~60 s.
- Do not touch `src/gui`, `src/nurbs`, `src/subd`. Do not change existing public signatures in `src/io`.
- Implementers do not run `git commit`; the orchestrator commits per stage. Do not create task chips.
- Docstrings: module docstring states purpose + the measured facts it relies on (the repo style, see `src/io/faceted_step.py`).

## File map

```
src/drawing/__init__.py        A1   package docstring, __version__ = "0.1.0", lazy public API (make_drawing)
src/drawing/options.py         A1   DrawingOptions
src/drawing/model.py           A1   LineStyle, primitives, Dimension, BBox2D, Sheet
src/drawing/style.py           A1   SheetSize, SHEETS, SheetStyle, DEFAULT_STYLE, STANDARD_SCALES
src/drawing/sniff.py           A1   sniff(), check_limits()
src/drawing/geometry2d.py      A2   pure 2D algorithms
src/drawing/projection.py      A3   ViewSpec, VIEWS, Orientation, auto_orientation()
src/drawing/brep_load.py       B1   load_step(), analyze_shape(), transform_shape(), faceted_shape_from_mesh()
src/drawing/brep_features.py   B1   cylinder_features()
src/drawing/brep_views.py      B2   project_brep(), prepare_faceted_shell(), edges_to_primitives()
src/drawing/brep_sections.py   B2   choose_section_plane_brep(), section_brep()
src/drawing/mesh_load.py       B3   load_stl_checked(), mesh_census(), classify(), transform_mesh()
src/drawing/mesh_views.py      B3   project_mesh(), raster_span(), edge_visibility(), mask_outline()
src/drawing/mesh_sections.py   B3   choose_section_plane_mesh(), section_mesh()
src/drawing/writers/__init__.py B4
src/drawing/writers/svg.py     B4   write_svg()
src/drawing/writers/dxf.py     B4   write_dxf()
src/drawing/writers/pdf.py     B4   write_pdf(), PdfWriter, helvetica_width()
src/drawing/dimensions.py      B5   plan_dimensions(), render_dimension()
src/drawing/layout.py          B5   compose(), choose_scale(), grid_cells(), title block, symbol, section marker
src/drawing/report.py          C1   Report
src/drawing/pipeline.py        C1   run()
src/drawing/cli.py             C1   main()
src/drawing/__main__.py        C1
src/drawing/sandbox.py         C2   run_sandboxed(), sanitized_env(), WindowsJob
tests/test_drawing_core.py     A1   options/model/style/sniff
tests/test_drawing_geometry2d.py A2
tests/test_drawing_projection.py A3
tests/test_drawing_brep_load.py  B1
tests/test_drawing_brep_views.py B2
tests/test_drawing_mesh.py     B3
tests/test_drawing_writers.py  B4
tests/test_drawing_layout.py   B5
tests/test_drawing_pipeline.py C1   routing + CLI end to end (--no-sandbox and sandboxed)
tests/test_drawing_sandbox.py  C2
docs/drawing_module.md, docs/drawing_service_architecture.md, README.md, requirements.txt   D1
```

## Shared interfaces (copy exactly; later tasks import these names)

```python
# src/drawing/model.py
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Sequence, Tuple, Union
import numpy as np

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
    a: Point; b: Point; style: LineStyle = LineStyle.VISIBLE

@dataclass
class Polyline:
    points: np.ndarray            # (n, 2) float64
    closed: bool = False
    style: LineStyle = LineStyle.VISIBLE

@dataclass
class Circle:
    center: Point; radius: float; style: LineStyle = LineStyle.VISIBLE

@dataclass
class Arc:
    center: Point; radius: float
    start_deg: float; end_deg: float      # counter-clockwise from start to end, degrees
    style: LineStyle = LineStyle.VISIBLE

@dataclass
class Polygon:                            # filled (arrowheads, projection symbol fills)
    points: np.ndarray; style: LineStyle = LineStyle.DIMENSION

@dataclass
class Text:
    position: Point; text: str; height: float
    rotation_deg: float = 0.0             # 0 = reads from the bottom, 90 = reads from the right
    anchor: str = "middle"                # "start" | "middle" | "end" (along the baseline)
    valign: str = "baseline"              # "baseline" | "middle"
    style: LineStyle = LineStyle.TEXT

@dataclass
class HatchLines:
    segments: np.ndarray                  # (k, 2, 2)
    loops: List[np.ndarray] = field(default_factory=list)   # the boundary loops (for the DXF HATCH entity)
    style: LineStyle = LineStyle.HATCH

@dataclass
class Dimension:                          # semantic; coordinates in the space stated by the holder
    kind: str                             # "linear" | "diameter"
    text: str                             # "60", "4 x Ø6.5", "Ø20"
    value: float                          # model mm
    p1: Point = (0.0, 0.0); p2: Point = (0.0, 0.0)     # linear: measured points
    base: Point = (0.0, 0.0)              # linear: a point on the dimension line
    angle_deg: float = 0.0                # linear: 0 horizontal, 90 vertical
    center: Point = (0.0, 0.0); radius: float = 0.0   # diameter
    leader_angle_deg: float = 45.0        # diameter: leader direction
    view: str = ""                        # name of the view it belongs to
    count: int = 1

Primitive2D = Union[Segment, Polyline, Circle, Arc]
SheetItem = Union[Segment, Polyline, Circle, Arc, Polygon, Text, HatchLines]

@dataclass
class BBox2D:
    xmin: float; ymin: float; xmax: float; ymax: float
    @property
    def width(self) -> float: ...
    @property
    def height(self) -> float: ...
    @classmethod
    def from_points(cls, pts: np.ndarray) -> "BBox2D": ...
    @classmethod
    def of_primitives(cls, prims: Sequence) -> Optional["BBox2D"]: ...   # Circle/Arc use their full extent
    def union(self, other: "BBox2D") -> "BBox2D": ...
    def expanded(self, margin: float) -> "BBox2D": ...
    def contains(self, p: Point) -> bool: ...

@dataclass
class Sheet:
    size_name: str                        # "A3" | "A4" | "A4L"
    width: float; height: float           # mm
    items: List[SheetItem] = field(default_factory=list)      # ALL geometry, in sheet mm
    dimensions: List[Dimension] = field(default_factory=list) # sheet mm, for the DXF DIMENSION entities
    meta: dict = field(default_factory=dict)                  # title fields, scale text, projection ...

def transform_primitive(prim, offset: Point, scale: float): ...   # view -> sheet copy (Segment/Polyline/Circle/Arc/Polygon/HatchLines/Text)
```

```python
# src/drawing/options.py
@dataclass
class DrawingOptions:
    formats: Tuple[str, ...] = ("pdf", "svg", "dxf")
    views: Tuple[str, ...] = ("front", "top", "right", "iso")
    section: str = "auto"              # auto | front | top | right | off
    sheet: str = "A3"                  # A3 | A4 | A4L | auto
    projection: str = "first"          # first | third
    orient: str = "auto"               # auto | fixed
    up: str = "z"                      # z | y
    text_height: float = 3.5
    hidden_lines: bool = True
    hole_positions: bool = True
    max_hole_callouts: int = 6
    stl_to_step: str = "auto"          # auto | faceted | nurbs | off
    faceted_max_triangles: int = 30_000
    facet_smooth_angle_deg: float = 20.0
    feature_angle_deg: float = 30.0
    raster_px: int = 1024
    exact_hlr_max_faces: int = 4000
    hlr_time_budget_s: float = 60.0
    classifier_threshold: float = 0.10
    max_input_mb: float = 200.0
    max_triangles: int = 2_000_000
    timeout_s: float = 180.0
    memory_limit_mb: int = 2048
    output_cap_mb: int = 200
    title: Optional[str] = None
    owner: str = ""
    quiet: bool = False
    def validate(self) -> None: ...            # ValueError with the offending field
    def to_json(self) -> str: ...
    @classmethod
    def from_json(cls, text: str) -> "DrawingOptions": ...
    def with_overrides(self, **kw) -> "DrawingOptions": ...
```

```python
# src/drawing/style.py
@dataclass(frozen=True)
class SheetSize:
    name: str; width: float; height: float; iso5457: bool; orientation: str
SHEETS = {"A3": SheetSize("A3", 420, 297, True, "landscape"),
          "A4": SheetSize("A4", 210, 297, True, "portrait"),
          "A4L": SheetSize("A4L", 297, 210, False, "landscape")}
STANDARD_SCALES = [50, 20, 10, 5, 2, 1, 1/2, 1/5, 1/10, 1/20, 1/50, 1/100]   # sheet mm per model mm
def scale_text(scale: float) -> str: ...        # 2 -> "2:1", 0.5 -> "1:2"

@dataclass(frozen=True)
class SheetStyle:
    margin_left: float = 20.0; margin: float = 10.0; frame_width: float = 0.7
    centring_mark_outside: float = 5.0; centring_mark_inside: float = 10.0
    wide: float = 0.5; narrow: float = 0.25; text_stroke: float = 0.35
    dash_hidden: Tuple[float, ...] = (3.0, 0.75)
    dash_center: Tuple[float, ...] = (6.0, 0.75, 0.125, 0.75)
    dash_cutting: Tuple[float, ...] = (12.0, 1.5, 0.25, 1.5)
    text_height: float = 3.5; label_height: float = 5.0; title_label_height: float = 2.5
    arrow_length: float = 3.5; arrow_angle_deg: float = 15.0
    ext_gap: float = 1.0; ext_overshoot: float = 2.0
    dim_first_offset: float = 10.0; dim_pitch: float = 7.0; dim_text_gap: float = 1.0
    view_gap: float = 20.0
    hatch_spacing_small: float = 2.0; hatch_spacing_mid: float = 3.0; hatch_spacing_large: float = 4.0
    title_block_w: float = 180.0; title_block_h: float = 36.0; title_row_h: float = 9.0
    title_cols: Tuple[float, ...] = (0.0, 32.0, 74.2, 129.0, 180.0)
    strip_h: float = 9.0
    section_arrow_len: float = 10.0; section_head_len: float = 5.0; section_head_angle_deg: float = 30.0
    section_dash_len: float = 12.0
    symbol_h: float = 3.5           # projection symbol size parameter (= text height)
DEFAULT_STYLE = SheetStyle()
def hatch_spacing(style: SheetStyle, shorter_extent_sheet_mm: float) -> float: ...   # 2 / 3 / 4 rule, min 1.0
```

```python
# src/drawing/sniff.py
@dataclass
class SniffResult:
    kind: str                      # "step" | "stl_binary" | "stl_ascii" | "unknown"
    size_bytes: int
    triangles: Optional[int]       # declared/scanned for STL, None for STEP
    sha256: str
    extension: str                 # lower-case, e.g. ".stp"
    extension_mismatch: bool
def sniff(path: str, compute_hash: bool = True) -> SniffResult: ...
def check_limits(s: SniffResult, max_input_mb: float, max_triangles: int) -> Optional[str]: ...   # error text or None
```

```python
# src/drawing/projection.py
@dataclass(frozen=True)
class ViewSpec:
    name: str
    n: np.ndarray        # unit, toward the viewer (= OCCT gp_Ax2 main direction)
    x: np.ndarray        # unit, view +x in world (= gp_Ax2 X direction)
    @property
    def y(self) -> np.ndarray: ...        # cross(n, x)
    def to_view(self, pts: np.ndarray) -> np.ndarray: ...   # (n,3) -> (n,3) columns [x, y, depth]; depth = dot(p, n), larger = nearer
    def gp_ax2_args(self) -> Tuple[Tuple[float,float,float], Tuple[float,float,float], Tuple[float,float,float]]: ...  # (origin, n, x)
VIEWS: Dict[str, ViewSpec]    # front back top bottom right left iso, from the spec table
PRINCIPAL = ("front", "back", "top", "bottom", "right", "left")

@dataclass(frozen=True)
class Orientation:
    matrix: np.ndarray    # 3x3 proper rotation (det +1), world' = matrix @ world
    permutation: Tuple[int, int, int]
    flips: Tuple[int, int, int]
    mode: str             # "auto" | "fixed" | "y_up"
    def apply(self, pts: np.ndarray) -> np.ndarray: ...
def auto_orientation(extents: np.ndarray, mode: str = "auto", up: str = "z") -> Orientation: ...
    # auto: longest extent -> X, shortest -> Y (depth), remaining -> Z; choose signs so det = +1
    # fixed: identity (after the optional y_up rotation: y-up file -> z-up: (x, y, z) -> (x, -z, y))
def grid_cell(view: str, projection: str) -> Tuple[int, int]: ...
    # (col, row) relative to front (0,0); first: top (0,-1) bottom (0,1) right (-1,0) left (1,0) back (2,0)
    # third: top (0,1) bottom (0,-1) right (1,0) left (-1,0) back (-2,0); iso -> (1, 1) first / (2, 1) third (adjusted by layout if taken)
```

```python
# shared result objects (defined in src/drawing/model.py too, below the primitives)
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
    plane_point: np.ndarray; plane_normal: np.ndarray      # world, normal points toward the viewer (= spec.n)
    loops: List[np.ndarray]        # closed polygons in view coords, no repeated end point
    body: Optional[ProjectedView]
    label: str = "A"
    marker_view: str = "top"       # principal view that shows the cutting line
    stats: dict = field(default_factory=dict)   # open_wires_dropped, nudged, seconds

@dataclass
class CylFeature:
    center: np.ndarray; axis: np.ndarray; radius: float; length: float
    kind: str                      # "hole" | "boss" | "fillet"
    full_circle: bool; n_faces: int
```

```python
# src/drawing/brep_load.py
@dataclass
class BrepModel:
    shape: object; source_unit: str; roots: int
    census: dict        # solids shells faces edges vertices face_types brepcheck_valid free_edges closed volume_mm3 area_mm2 bbox_min bbox_max inverted
    warnings: List[str]
def load_step(path: str) -> BrepModel                     # ValueError on unreadable/empty; never calls Transfer after a failed read
def analyze_shape(shape, source_unit: str = "mm", roots: int = 1) -> BrepModel
def transform_shape(shape, matrix: np.ndarray)            # gp_Trsf rotation (det +1), BRepBuilderAPI_Transform(copy=True)
def faceted_shape_from_mesh(mesh, smooth_angle_rad: float, warn) -> object   # shell (or solid) via src.io.faceted_step._build_direct/_shell_from_faces/_oriented_solid + EncodeRegularity
def step_unit_sniff(path: str) -> str                      # "mm" | "inch" | "unknown" (text)

# src/drawing/brep_features.py
def cylinder_features(shape, radius_tol: float = 0.01, axis_tol: float = 1e-4) -> List[CylFeature]

# src/drawing/brep_views.py
def project_brep(shape, spec: ViewSpec, *, exact: bool, deflection: float, hidden: bool = True,
                 time_budget_s: Optional[float] = None) -> ProjectedView     # raises TimeoutError when over budget (checked after Update/Hide)
def prepare_faceted_shell(shape, deflection: float)        # BRepMesh_IncrementalMesh(shape, deflection, False, 0.35, True); returns shape
def edges_to_primitives(compound, deflection: float, style: LineStyle) -> List[Primitive2D]

# src/drawing/brep_sections.py
def choose_section_plane_brep(model: BrepModel, features: List[CylFeature], mode: str) -> Tuple[np.ndarray, np.ndarray, str]
    # returns (point, normal(world, = spec.n of the section view), marker_view); mode auto|front|top|right
def section_brep(shape, point, normal, spec: ViewSpec, *, exact: bool, deflection: float, depth_extent: float) -> SectionView
    # nudges once by 1 % of depth_extent when 0 closed loops; stats["nudged"], stats["open_wires_dropped"]

# src/drawing/mesh_load.py
@dataclass
class MeshModel:
    mesh: "trimesh.Trimesh"; census: dict; classifier: dict; warnings: List[str]
def load_stl_checked(path: str) -> "trimesh.Trimesh"      # ValueError: 0 bytes, STEP text, 0 faces
def mesh_census(mesh) -> dict        # triangles vertices watertight winding_consistent components volume_mm3 area_mm2 bbox_min bbox_max euler degenerate_faces duplicate_faces
def classify(mesh, threshold: float = 0.10) -> dict        # verdict planar_big sharp_turn threshold
def transform_mesh(mesh, matrix) -> "trimesh.Trimesh"
def load_mesh_model(path, threshold) -> MeshModel

# src/drawing/mesh_views.py
def project_mesh(mesh, spec: ViewSpec, *, raster_px: int, feature_angle_deg: float, organic: bool,
                 hidden: bool = True) -> ProjectedView
def raster_span(xy: np.ndarray, depth: np.ndarray, faces: np.ndarray, n_px: int, margin_px: int = 4) -> Tuple[np.ndarray, np.ndarray, dict]
    # returns (depth_buffer (H,W) with -inf where empty, mask (H,W) bool, grid dict {x0,y0,px})
def edge_visibility(segments_view: np.ndarray, depth_buffer, grid, tol_px: float = 1.5, nbhd: int = 1) -> Tuple[np.ndarray, np.ndarray]
    # (visible (k,2,2), hidden (m,2,2)) in view coords
def mask_outline(mask: np.ndarray, grid: dict, tol_px: float = 0.75) -> List[np.ndarray]   # closed loops, view coords

# src/drawing/mesh_sections.py
def choose_section_plane_mesh(mesh, mode: str) -> Tuple[np.ndarray, np.ndarray, str]
def section_mesh(mesh, point, normal, spec: ViewSpec, *, raster_px: int, feature_angle_deg: float, organic: bool,
                 depth_extent: float) -> SectionView

# src/drawing/geometry2d.py
def simplify_polyline(points: np.ndarray, tol: float) -> np.ndarray
def is_straight(points: np.ndarray, tol: float) -> bool
def polygon_area(points: np.ndarray) -> float                              # signed
def point_in_polygon(pt, poly: np.ndarray) -> bool                         # half-open crossing rule
def nest_loops(loops: List[np.ndarray]) -> List[Tuple[np.ndarray, List[np.ndarray]]]   # by containment, even-odd depth
def hatch_lines(loops: List[np.ndarray], spacing: float, angle_deg: float = 45.0, phase: float = 0.0) -> np.ndarray   # (k,2,2)
def chain_segments(segments: np.ndarray, tol: float) -> Tuple[List[np.ndarray], int]   # (closed loops w/o repeated end, n_open_dropped)
def merge_collinear(segments: np.ndarray, tol: float) -> np.ndarray
def subtract_covered(hidden: np.ndarray, visible: np.ndarray, tol: float) -> np.ndarray   # remove hidden parts lying on visible segments
def dedupe_primitives(prims: list, tol: float) -> list
def primitives_to_segments(prims: list, deflection: float) -> np.ndarray   # flatten circles/arcs/polylines for hatch/visibility work

# src/drawing/dimensions.py
@dataclass
class DimensionPlan:
    dims: List[Dimension]          # view coords, Dimension.view set; linear dims carry band index in .base via offset computed by layout
    undimensioned: List[dict]
def plan_dimensions(views: Dict[str, ProjectedView], features: List[CylFeature], options: DrawingOptions,
                    style: SheetStyle, model_extents: np.ndarray) -> DimensionPlan
def render_dimension(dim: Dimension, style: SheetStyle) -> List[SheetItem]   # dim in sheet coords -> Segments, Polygons (arrowheads), Text
def format_value(v: float) -> str                                            # "60", "6.5", "12.25"

# src/drawing/layout.py
@dataclass
class LayoutResult:
    sheet: Sheet; scale: float; scale_text: str; sheet_size: SheetSize
    placements: Dict[str, Tuple[Point, BBox2D]]      # view name -> (offset, sheet bbox incl. dimension bands)
    overflow: bool; flags: List[Tuple[str, str, str]]  # (code, severity, detail)
def compose(views: Dict[str, ProjectedView], section: Optional[SectionView], plan: DimensionPlan,
            options: DrawingOptions, style: SheetStyle, meta: dict) -> LayoutResult
def choose_scale(view_boxes: Dict[str, BBox2D], bands: Dict[str, Tuple[int,int,int,int]], sheet: SheetSize,
                 style: SheetStyle, projection: str, has_section: bool, has_iso: bool) -> Tuple[Optional[float], dict]

# src/drawing/writers
def write_svg(sheet: Sheet, path: str) -> dict   # {"bytes": n, "elements": n}
def write_dxf(sheet: Sheet, path: str) -> dict   # {"bytes": n, "entities": n, "layers": {...}}
def write_pdf(sheet: Sheet, path: str) -> dict   # {"bytes": n, "objects": n}

# src/drawing/report.py
class Report:
    def __init__(self, input_path: str, out_dir: str, options: DrawingOptions): ...
    data: dict                                   # the schema of spec 3.8, pre-filled with nulls
    def flag(self, code: str, severity: str, detail: str = "") -> None
    def warn(self, text: str) -> None
    def fallback(self, frm: str, to: str, reason: str) -> None
    def finish(self, ok: bool, error: Optional[str] = None) -> Tuple[int, dict]   # computes level, exit code (0/2/1), timing
    def write(self, path: str) -> None

# src/drawing/pipeline.py
def run(input_path: str, out_dir: str, options: DrawingOptions, log=None) -> Tuple[int, dict]

# src/drawing/sandbox.py
@dataclass
class SandboxResult:
    exit_code: int; result: Optional[dict]; sandbox: dict; stdout_tail: str; stderr_tail: str
def sanitized_env(work_dir: str, repo_root: str) -> dict
def run_sandboxed(argv: List[str], work_dir: str, *, timeout_s: float, memory_limit_mb: int, cpu_limit_s: float,
                  output_cap_mb: int, env: Optional[dict] = None, poll_s: float = 0.1) -> SandboxResult

# src/drawing/cli.py
def main(argv=None) -> int      # parses, sniff+limits, sandbox or direct, prints exactly one "RESULT {json}" line
```

---

## Stage A — foundation (A1, A2, A3 run in parallel; no cross-imports except A1 <- nothing)

### Task A1: options, model, style, sniff

**Files:** Create `src/drawing/__init__.py`, `src/drawing/options.py`, `src/drawing/model.py`, `src/drawing/style.py`, `src/drawing/sniff.py`; Test `tests/test_drawing_core.py`.
**Interfaces:** Produces everything in the shared-interface block for those four modules, including `ProjectedView`, `SectionView`, `CylFeature` (in `model.py`, with `ViewSpec` referenced only as a type under `TYPE_CHECKING`).
**Probe facts to use:** `REPORT-sandbox.md` item 6 (sniff rules: `ISO-10303-21` prefix; binary STL size == 84 + 50·n even when the header says `solid`; ASCII: starts with `solid` and count `facet normal` lines; truncated binary -> unknown), `REPORT-standards.md` (every `SheetStyle` number).

- [ ] Write failing tests in `tests/test_drawing_core.py`:
  - `DrawingOptions().validate()` passes; `sheet="A5"`, `projection="x"`, `formats=("png",)`, `timeout_s=1` (below 15) raise `ValueError` naming the field; `from_json(to_json())` round-trips every field; `with_overrides(sheet="A4")` leaves the original untouched.
  - `BBox2D.of_primitives([Circle((0,0),5), Segment((0,0),(10,0))])` == (-5,-5,10,5); `Arc` extent uses the full circle (document that simplification in the docstring); empty list -> None.
  - `transform_primitive(Segment((1,2),(3,4)), offset=(10,10), scale=2)` -> (12,14)-(16,18); same for Circle radius scaled, Text height NOT scaled (text stays sheet-sized), HatchLines segments scaled.
  - `scale_text(0.5) == "1:2"`, `scale_text(2) == "2:1"`, `scale_text(1) == "1:1"`; `STANDARD_SCALES` strictly decreasing; `hatch_spacing(style, 15) == 2.0`, `(style, 50) == 3.0`, `(style, 100) == 4.0`.
  - sniff: write with `tempfile`: (a) a trimesh box exported binary (`mesh.export(path)`) -> `stl_binary`, triangles 12; (b) the same file with the first 5 header bytes overwritten by `b"solid"` -> still `stl_binary` (precondition: assert the header now starts with `solid`); (c) `mesh.export(path, file_type="stl_ascii")` -> `stl_ascii`, triangles 12; (d) binary file truncated by 10 bytes -> `unknown`; (e) `b"ISO-10303-21;\nHEADER;..."` in a `.stl` file -> kind `step`, `extension_mismatch True`; (f) 0 bytes -> `unknown`; (g) 300 random bytes -> `unknown`; `check_limits` returns a message when `size_bytes > max_input_mb·2**20` or `triangles > max_triangles`, else None; sha256 equals `hashlib.sha256(open(path,'rb').read()).hexdigest()`.
- [ ] Run `python -m pytest tests/test_drawing_core.py -q` -> fails on import.
- [ ] Implement the four modules exactly per the interface block. `sniff` reads at most 4 kB for the text checks, the whole file only for the hash and the ASCII facet count (`for line in fh: if line.lstrip().startswith(b"facet normal")`).
- [ ] Run the tests -> pass. Run `python -m pytest tests/test_ocp_layout.py -q` -> still passes (new files have no OCP imports).

### Task A2: geometry2d

**Files:** Create `src/drawing/geometry2d.py`; Test `tests/test_drawing_geometry2d.py`.
**Interfaces:** Produces the `geometry2d` functions of the interface block. Consumes `model.py` primitives (A1) only in `primitives_to_segments`/`dedupe_primitives`; if A1 is not merged yet, import lazily inside those two functions.
**Probe snippets:** `REPORT-mesh.md` snippets for `hatch` (even-odd, half-open crossing rule, grid phase-locked), `chain_segments` (KD-tree weld at tol, scipy `cKDTree`), `plane_basis`. The HLR report says what `subtract_covered`, `merge_collinear` and `dedupe` must handle (box back edges under visible ones; duplicate fillet arc; chains 0..5 and 5..55 on one line).

- [ ] Write failing tests:
  - `hatch_lines([square 10x10, square hole 4x4 centred], spacing=1.0, 45)`: no segment midpoint lies inside the hole (check with `point_in_polygon`), every midpoint lies inside the outer square, total length within 2 % of (100−16)/1.0·... compute `area/spacing` = 84 and compare `sum(lengths)` to 84 with rel tol 0.03; a hatch line through two hole corners and two outer corners (square (0,0)-(10,10), hole (3,3)-(7,7), diagonal x=y) yields exactly two segments [(0,0)-(3,3)], [(7,7)-(10,10)] (test with `phase` chosen so a line passes through the origin).
  - `chain_segments` on the 12 edges of two separate squares given as unordered segments (shuffled, one square reversed) -> 2 loops of 4 points each, 0 open; with one segment removed -> 1 loop + 1 open dropped; endpoints within `tol` are welded (perturb by 1e-9).
  - `nest_loops` with outer square + inner square + a separate square -> two outers, the first with one inner, regardless of input orientation (feed one loop reversed; precondition: assert `polygon_area` signs differ).
  - `simplify_polyline` of 100 points on a straight line with noise 1e-6 and tol 1e-3 -> 2 points; `is_straight` True; a quarter circle r=10 with tol 1e-3 -> False.
  - `merge_collinear([(0,0)-(5,0), (5,0)-(55,0), (0,0)-(0,10)])` -> 2 segments, one spanning 0..55.
  - `subtract_covered(hidden=[(0,0)-(60,0)], visible=[(0,0)-(20,0),(40,0)-(60,0)])` -> exactly [(20,0)-(40,0)] (within tol); a hidden segment not collinear with any visible one is returned unchanged; fully covered -> empty.
  - `dedupe_primitives([Arc(c,3,270,360), Arc(c,3,270,360), Segment(a,b), Segment(b,a)])` -> 2 items (reversed segment counts as a duplicate).
  - `point_in_polygon` on a vertex and on an edge midpoint behaves consistently (half-open: a point exactly on the left/bottom boundary is inside, on the right/top outside) — document and test both.
- [ ] Run -> fail. Implement. Run -> pass.

### Task A3: projection

**Files:** Create `src/drawing/projection.py`; Test `tests/test_drawing_projection.py`.
**Interfaces:** Produces `ViewSpec`, `VIEWS`, `PRINCIPAL`, `Orientation`, `auto_orientation`, `grid_cell`.
**Probe facts:** `REPORT-hlr.md` view table (N toward viewer, Vx) and the sign facts: front (0,−1,0)/(1,0,0) -> 2D (X, Z); top (0,0,1)/(1,0,0) -> (X, Y); right (1,0,0)/(0,1,0) -> (Y, Z); left (−1,0,0)/(0,−1,0) -> (−Y, Z); back (0,1,0)/(−1,0,0) -> (−X, Z); bottom (0,0,−1)/(1,0,0) -> (X, −Y); iso (1,−1,1)/(1,1,0) with up (−0.408, 0.408, 0.816). `REPORT-standards.md` arrangement grids.

- [ ] Write failing tests: every `VIEWS` entry has orthonormal right-handed (n, x, y) with `y == cross(n, x)`; `to_view` on the point (1,2,3) gives front (1,3,−2)... derive each from the table (front: x = X = 1, y = Z = 3, depth = dot(p, n) = −2); right view of (1,2,3) -> (2, 3, 1); iso `y[2] > 0` (Z projects upward); `auto_orientation(extents=(60, 20, 40))` maps the 60 axis to X and the 20 axis to Y with `det(matrix) == +1`, and `apply` of a bbox gives extents (60, 20, 40) in that order; `mode="fixed"` returns identity; `up="y"` maps (0,1,0) to (0,0,1); `grid_cell("top","first") == (0,-1)`, `grid_cell("top","third") == (0,1)`, `grid_cell("right","first") == (-1,0)`, `grid_cell("right","third") == (1,0)`.
- [ ] Run -> fail. Implement (`gp_ax2_args` returns plain tuples so the module has no OCP import). Run -> pass.

---

## Stage B — sources, writers, layout (B1..B5 in parallel after Stage A is committed)

### Task B1: brep_load + brep_features

**Files:** Create `src/drawing/brep_load.py`, `src/drawing/brep_features.py`; Test `tests/test_drawing_brep_load.py` (own fixture builder `_plate()`: 100x60x15 plate, four Ø6.5 through holes at (8,8) (92,8) (8,52) (92,52), one Ø20 hole at (50,30), a Ø30 x 10 boss at (78,30) on top — the `REPORT-brep-analysis.md` fixture; B2 has its own test file and its own copy of the fixture so the two tasks never edit the same file).
**Consumes:** `model.CylFeature`, `projection.Orientation`.
**Probe snippets:** `REPORT-brep-analysis.md`: safe STEP read (ReadFile gate, `FileUnits`, roots), census with `TopExp.MapShapes_s`, `AddOptimal_s` bbox, `measure_shape(shape, deviation_samples=0)` reuse, cylinder detection with the REVERSED normal flip, coaxial grouping, `step_unit_sniff` text rule, quirks (null shape, open shell, reversed solid, compound). `REPORT-hlr.md` item 5 for `faceted_shape_from_mesh` (`_build_direct` -> `_shell_from_faces` -> `EncodeRegularity_s(shell, angle)`; on a bare compound Encode does nothing).

- [ ] Failing tests (all `needs_ocp`): plate census `face_types == {"Plane": 7, "Cylinder": 6}`, faces 13, edges 30, solids 1, `brepcheck_valid`, `closed`, volume ≈ 90365.21 (rel 1e-6), bbox (0,0,0)-(100,60,25) within 1e-6 (precondition: `BRepBndLib.Add_s` without mesh also gives it; the test for the NURBS over-report: `BRepBuilderAPI_NurbsConvert` of a r=10 cylinder -> `analyze_shape` bbox within 1e-4 of ±10 while `Add_s` is off by > 5); `cylinder_features(plate)` -> 6 features: 5 `hole` (4 with r 3.25, 1 with r 10, all `full_circle`, length 15, centre z 7.5) + 1 `boss` r 15 length 10 centre (78,30,20); after `ShapeUpgrade_ShapeDivideClosed(...).SetNbSplitPoints(1); Perform()` the shape has 19 faces (precondition) and still 6 features each with `n_faces == 2`; negative control: a private helper flag `_flip_reversed=False` (module constant monkeypatched) turns all 5 holes into bosses; `load_step` on a written plate STEP (via `src.io.exporters.export_step`) gives the same census; `load_step` on a file containing `garbage` text raises `ValueError` (and the test process is still alive); an empty `DATA; ENDSEC;` STEP raises `ValueError`; a two-root STEP (two `writer.Transfer` calls) -> `roots == 2`, `census["solids"] == 2`; an inch STEP (`init_step_statics(unit="INCH")`, then reset to MM in `finally`) of a 25.4 mm cube -> `source_unit == "inch"` and bbox 25.4; `step_unit_sniff` agrees; a reversed box -> `census["inverted"] True`, volume reported negative; an open shell -> `solids 0`, `free_edges 4`; `transform_shape(box, R)` with the 90° permutation matrix moves the bbox extents accordingly; `faceted_shape_from_mesh(trimesh.creation.icosphere(2), 0.35, warn)` returns a shell whose edges are coded C1 (count via `BRep_Tool.Continuity_s` over edges with two faces: all 480 ≥ GeomAbs_C1; precondition: before encoding they are C0).
- [ ] Run -> fail. Implement. Run -> pass; run `tests/test_ocp_layout.py` -> pass.

### Task B2: brep_views + brep_sections

**Files:** Create `src/drawing/brep_views.py`, `src/drawing/brep_sections.py`; Test `tests/test_drawing_brep_views.py` (own fixtures; do not depend on B1 being merged — build shapes with BRepPrimAPI in the test; for `faceted_shape_from_mesh` import it from `brep_load` lazily inside the test and skip if it is not there yet).
**Consumes:** `model` primitives/`ProjectedView`/`SectionView`, `projection.ViewSpec/VIEWS`, `geometry2d` (`is_straight`, `dedupe_primitives`, `subtract_covered`, `merge_collinear`, `chain_segments`, `nest_loops`).
**Probe snippets:** `REPORT-hlr.md`: projector from `gp_ax2_args`, compounds to use (V + OutLineV visible, H + OutLineH hidden, Rg1/RgN dropped), per-edge `BuildCurve3d_s` in try/except, `edges_to_primitives` (Line/Circle/Arc with the axis-Z swap, QuasiUniformDeflection polylines collapsed to lines when straight), PolyAlgo sequence (`Load`, `Projector`, `Update`, `HLRBRep_PolyHLRToShape().Update(algo)`, default TolAngular), BRepMesh parameters. `REPORT-brep-analysis.md`: `BRepAlgoAPI_Section(shape, pln, False)` + `ComputePCurveOn1(True)` + `Approximation(True)` + `Build()`, `ConnectEdgesToWires_s`, half-space cut, HLR of the body, explicit in-plane basis = the view's x/y.

- [ ] Failing tests (`needs_ocp`): box 60x40x20 + vertical Ø10 through hole at (30,20): `project_brep(front, exact=True)` -> `visible` has 4 segments spanning x 0..60 / y 0..20 and NO circle; `hidden` contains two vertical segments at x = 25 and x = 35 (the hole walls) and nothing lying on the visible outline (the far mouth and back edges must have been subtracted: assert no hidden segment with both endpoints on y ∈ {0, 20}); `project_brep(top)` -> exactly one `Circle` centre (30,20) r 5 in `visible`; the hidden far mouth circle is deduped away (assert no Circle in `hidden`); filleted box (r3 all edges) front view -> 4 `Arc`s r 3 in visible, no segment at x = 3 or x = 57 (Rg1 dropped), 90° arcs oriented so that their midpoint lies at the corner side (compute the midpoint from `start_deg/end_deg` and check it is outside the inner 54x14 box); hole at 30° -> at least one `Polyline` in visible (ellipse discretized) with max chord deviation < deflection; `hidden=False` -> `hidden == []`; faceted path: `prepare_faceted_shell(faceted_shape_from_mesh(icosphere(2)...))` then `project_brep(..., exact=False)` -> `algorithm == "hlr_poly"`, visible segment count == 24 (the mesh silhouette count from the probe), bbox ≈ ±20 within 1e-6 (precondition: the mesh is icosphere radius 20); two separate boxes compound -> counts equal the sum; `time_budget_s=1e-9` raises `TimeoutError`.
  - Sections on the plate (build it in the test): `choose_section_plane_brep(model, features, "auto")` -> plane through the Ø20 hole axis (point y = 30), normal (0,−1,0), marker view "top"; `section_brep(plate, (50,30,7.5), (0,-1,0), VIEWS["front"], exact=True, deflection=0.02, depth_extent=60)` -> 2 loops with |areas| 600 and 900 (±0.5), `body.visible` non-empty, `stats["open_wires_dropped"] == 0`, `stats["nudged"] False`; a plane missing the shape (y = 500) -> after one nudge still 0 loops -> `loops == []` and `stats["nudged"] True`; plane x = 93 tangent to the boss -> `open_wires_dropped >= 1`.
- [ ] Run -> fail. Implement. Run -> pass; run `tests/test_ocp_layout.py`.

### Task B3: mesh_load + mesh_views + mesh_sections

**Files:** Create the three modules; Test `tests/test_drawing_mesh.py`.
**Consumes:** `model`, `projection`, `geometry2d` (`chain_segments`, `simplify_polyline`, `subtract_covered`, `dedupe_primitives`).
**Probe snippets:** `REPORT-mesh.md`: `load_stl_checked`, census, `classify_edges` (sharp/boundary/silhouette with front = n·d < −1e-9 — translate to the ViewSpec convention: a face is front-facing when `dot(normal, spec.n) > 1e-9`), `raster_span` (scanline, `np.maximum.at`), `edge_visibility` (sample ~1 px, 3x3 **minimum** window, tol 1.5 px), `mask_contours` (skimage `find_contours` on the padded mask + `approximate_polygon`), `mesh_plane` with keyword args + chaining, `plane_basis`, classifier `planar_big` (facets with area > 1 % of total) and `sharp_turn`.

- [ ] Failing tests: `load_stl_checked` raises `ValueError` for 0 bytes, STEP text, and a truncated binary (precondition: `trimesh.load(force="mesh")` returns 0 faces on it); `mesh_census(trimesh.creation.box((60,40,20)))` watertight, volume 48000, components 1, bbox; the repo lattice `tests/test_high_genus_closed_cage._jungle_gym()` gives `volume_mm3 > 0` (abs) — skip if import fails; `classify(box)["verdict"] == "prismatic"` with `planar_big >= 0.2` asserted, icosphere(3) and a Taubin-smoothed noisy icosphere -> `organic` with `planar_big == 0.0` asserted, a 48-segment cylinder -> prismatic; `project_mesh` on two boxes (front box −5..5 x, y −10..−8 (toward viewer), z −3..3; back box 0..30 x, y 0..10, z −4..6; precondition: in the front view the front box occludes part of the back box's vertical edge at x = 1... pick the back box's left vertical edge at x = 0 fully inside the front box's silhouette width so it splits) -> the back box edge appears in both `visible` and `hidden` with split points within 2 px of z = ±3; negative control: the same mesh without the front box -> that edge is a single visible segment and `hidden` is empty; icosphere(4) with `organic=True` -> exactly one closed `Polyline` in visible whose points' radii are within 1.5 px of the radius, `hidden == []`; torus (`trimesh.creation.torus(10, 3)`) with `organic=True` top view -> 2 closed polylines (outer and hole); `raster_span` coverage equals a brute-force per-triangle fill on a 20-triangle mesh (same mask, depths within 1e-9); `section_mesh(torus, plane through axis)` -> 2 loops of radius 3 (fit) at ±10; perpendicular through centre -> 2 concentric loops r 13 and r 7 (within 2 % — polygons); box section coplanar with the top face -> nudged once and then 1 loop (`stats["nudged"] True`); `choose_section_plane_mesh(box, "auto")` -> bbox centre, normal (0,−1,0), marker "top".
- [ ] Run -> fail. Implement. Run -> pass. Warm up `approximate_polygon` once at module import? No — do it lazily and record the time in `stats`.

### Task B4: writers (svg, dxf, pdf)

**Files:** Create `src/drawing/writers/__init__.py`, `svg.py`, `dxf.py`, `pdf.py`; Test `tests/test_drawing_writers.py`.
**Consumes:** `model.Sheet` and every `SheetItem`, `Dimension`, `style.DEFAULT_STYLE` (line widths and dash arrays per `LineStyle`).
**Probe snippets:** `REPORT-writers.md`: `minipdf.py` (port it; keep the Helvetica width table and the WinAnsi/cp1252 encoding; map U+2300 -> Ø), `minisvg.py` pattern (geometry inside `translate(0,H) scale(1,-1)`, text OUTSIDE at y' = H − y with baseline offset 0.718·size/2, `rotate(-angle x y)`), DXF facts (DASHED not HIDDEN; `dimstyle="Standard"` + overrides incl. `dimdsep=46`; `add_diameter_dim` adds `%%c`; `$INSUNITS=4`, `$MEASUREMENT=1`, `$LWDISPLAY=1`; `audit()` must report 0 errors/0 fixes; `TextEntityAlignment` from `ezdxf.enums`).

- [ ] Failing tests: build one `Sheet` fixture with every item type (segments in each `LineStyle`, a circle, a 90° arc, a closed polyline, a filled triangle, hatch lines with loops, texts at anchors start/middle/end with "Ø6.5 ±0.1 45°", a rotated 90° text, one linear and one diameter `Dimension`).
  - SVG: parses with `xml.etree`; root `viewBox == "0 0 420 297"`; every hidden segment has `stroke-dasharray` starting with `3`; `<text>` elements are NOT inside the flipped `<g>` (walk the tree); text content keeps `Ø`; stroke widths 0.5/0.25/0.7 present; count of elements in the result dict matches the tree.
  - DXF: `ezdxf.readfile` ok; `auditor = doc.audit(); assert len(auditor.errors) == 0 and len(auditor.fixes) == 0`; layer `HIDDEN` linetype `DASHED`; entity counts per layer match the fixture (LINE/LWPOLYLINE/CIRCLE/ARC/TEXT/SOLID or HATCH); DIMENSION entities == 2, the diameter one's text contains `%%c` exactly once; `$INSUNITS == 4`; negative control: a fixture that asks for linetype `HIDDEN` (write a tiny private helper to create a layer with `linetype="HIDDEN"`) -> `audit().fixes` non-empty (this proves the audit check is load-bearing).
  - PDF: file starts with `%PDF-1.4`, ends with `%%EOF`; parse the xref table and assert every object offset points at `N 0 obj`; the content stream (inflate) contains ` d` dash operators with `8.504`, `w` widths 1.417/0.709, `Tj` text; non-ASCII `Ø` encoded as byte 0xD8; `helvetica_width("Ø10", 3.5)` equals the sum of the table widths (regression: 778+556+556)/1000·3.5; when `C:\Users\pmqua\.venvs\notebooklm\Scripts\python.exe` exists, run a subprocess that opens the PDF with PyMuPDF and prints `is_repaired`, page size, `len(page.get_drawings())`, whether `"SECTION A-A"` is in `get_text()` and assert on the JSON it prints (skip otherwise with a reason). Add the same optional PyMuPDF render for the SVG size only (dashes cannot be checked by MuPDF).
- [ ] Run -> fail. Implement. Run -> pass. Also render the fixture PDF to PNG (PyMuPDF) into the scratchpad and look at it once (Read tool) to confirm orientation and text are upright.

### Task B5: dimensions + layout

**Files:** Create `src/drawing/dimensions.py`, `src/drawing/layout.py`; Test `tests/test_drawing_layout.py`.
**Consumes:** `model`, `style`, `options`, `projection.grid_cell/VIEWS`, `geometry2d.hatch_lines/nest_loops`, `ProjectedView`/`SectionView`/`CylFeature` (build synthetic ones in tests: a front view of a 100x15 rectangle, a top view of a 100x60 rectangle with five circles, a right view 60x15, an iso view, a section with two loops).
**Probe facts:** `REPORT-standards.md` (all numbers, grids, symbol geometry, `A-A` above the view, cutting line = wide dashes ~12 mm beyond the outline at both ends + 10 mm wide arrows with filled 30° 5 mm heads pointing in the viewing direction, letters 5 mm). Section marker direction: the cutting plane normal `n` points toward the viewer of the section; in the marker view the arrows point along `−n` projected... concretely: for the default (section normal (0,−1,0), marker in the top view where 2D y = world Y) the cutting line is horizontal at the plane's Y and the arrows point toward −Y (down in the top view) because the viewer of the section stands at −Y looking +Y. Implement as: arrow direction in the marker view = `marker_spec.to_view(-n)[:2]` normalized... check the sign with the test below.

- [ ] Failing tests: `format_value(60.0) == "60"`, `(6.5) == "6.5"`, `(12.345) == "12.35"`; `plan_dimensions` on the synthetic views with features [4 holes r 3.25 axis Z, 1 hole r 10 axis Z, 1 boss] -> linear dims: exactly one of value 100 (front), one of 15 (front), one of 60 (top), none duplicated across views; diameter dims: two groups, texts `4 x Ø6.5` and `Ø20` in view "top"; hole position dims present for both groups (4 linear dims) when `hole_positions=True`, absent when False; `max_hole_callouts=1` -> one diameter dim and one `undimensioned` entry with `reason == "cap"`; a hole whose axis is not parallel to any drawn view's n is listed in `undimensioned` with reason `no_axial_view`.
  - `render_dimension` of a horizontal linear dim -> 2 extension `Segment`s (length = gap-corrected: start `ext_gap` from the point, end `ext_overshoot` past the line), 1 dimension `Segment`, 2 `Polygon` arrowheads with 3 points each and length `arrow_length` (±1e-6), 1 `Text` centred on the line midpoint `dim_text_gap` above it, rotation 0; vertical -> rotation 90 and the text to the left of the line; diameter -> a leader `Segment` from the circle at `leader_angle_deg`, one arrowhead touching the circle (distance of the tip from the centre == radius ±1e-6), text at the leader end with anchor "start".
  - `choose_scale` for front 100x15, top 100x60, right 60x15 + iso + section on A3 -> 1:1 (fits: 100 + 60 + gaps < 390); for a 900x600 part -> 1:5 on A3; a 3000x3000 part -> smallest scale and `overflow True`; `sheet="auto"` with the plate -> "A4" (portrait fits at 1:1? width 100 + 60 + 20 + bands... compute; if it does not fit at 1:1 the test asserts the chosen pair (size, scale) is the first in the order A4 then A3 at which it fits — write the expected value from the algorithm, not from a guess).
  - `compose` first angle: the top view's sheet bbox lies BELOW the front view's and shares its x range (±1e-6 after scaling); the right view lies LEFT of the front view and shares the y range; third angle mirrors both (precondition: the synthetic views are asymmetric so a swapped placement fails); every item lies inside the frame; frame = 4 `FRAME` segments at x 20..W−10, y 10..H−10; title block 180x36 at the bottom right with 4 rows and the column lines at 32/74.2/129; the strip above it contains `Text` "SCALE 1:1", "Dimensions in mm", and the projection symbol (2 circles + a 4-point polyline) with circles to the RIGHT of the cone for first angle and LEFT for third; `A-A` label (height 5) exists and sits above the section view bbox; the cutting line marker exists in the top view: 2 `CUTTING` segments, 2 arrow polygons pointing in −Y of the top view (assert the tip y < tail y), 2 `Text` "A"; hatch items exist with spacing matching `hatch_spacing` for the section's shorter extent; `sheet.dimensions` carries the rendered dims in sheet coordinates (count == len(plan.dims)); `meta["scale_text"]`, `meta["projection"]` set; `flags` contains `("sheet_not_iso5457", "warning", ...)` for A4L and nothing for A3.
- [ ] Run -> fail. Implement. Run -> pass.

---

## Stage C — pipeline, CLI, sandbox (C1 and C2 in parallel after Stage B is committed)

### Task C1: report + pipeline + cli + __main__

**Files:** Create `src/drawing/report.py`, `pipeline.py`, `cli.py`, `__main__.py`; Test `tests/test_drawing_pipeline.py`.
**Consumes:** everything from Stages A and B by the interface block; `src.io.faceted_step.export_step_faceted`, `src.io.step_audit.measure_shape`, `src.io.occt_utils.quiet_occt`; for `--stl-to-step nurbs` the same calls as `src/convert.py` (`QuadWrapper(target_face_count=2000).wrap(HalfEdgeMesh.from_trimesh(mesh))`, `SubDToNURBSConverter(continuity="G1").convert(cage, reference_mesh=dense)` -> `result["shape"]`; read `src/convert.py` lines 120-185 for the exact call and the Coons/clamp warnings).
**Pipeline order (spec 3.2–3.4):** sniff -> limits -> load (brep or mesh) -> orientation (`auto_orientation(extents)`; `transform_shape`/`transform_mesh`) -> re-analyze bbox after the transform -> features (brep) -> views for `options.views` (principal + iso) -> section (unless `off`) -> `plan_dimensions` -> `compose` -> writers for `options.formats` -> report. Deflection for HLR discretization = `0.05 / scale_guess` where `scale_guess` comes from `choose_scale` on the bbox-only view boxes before HLR (good enough; recompute nothing). Mesh fallback on any exception/timeout from the B-rep stage when the input was STL; for STEP input a failed view is logged, the view skipped, `hlr_timeout`/`view_failed` flagged.
**CLI:** `python -m src.drawing INPUT OUT_DIR [--formats pdf,svg,dxf] [--views front,top,right,iso] [--section auto|front|top|right|off] [--sheet A3|A4|A4L|auto] [--projection first|third] [--orient auto|fixed] [--up z|y] [--stl-to-step auto|faceted|nurbs|off] [--no-hidden] [--no-hole-positions] [--title T] [--owner O] [--timeout S] [--memory-limit-mb M] [--no-sandbox] [--options-json FILE] [--quiet]`. The `_Parser` class from `src/convert.py` (usage errors -> exit 1). Without `--no-sandbox`: sniff + limits in the parent, then `sandbox.run_sandboxed([sys.executable, "-m", "src.drawing", in, out, "--no-sandbox", "--options-json", f], work_dir, ...)`; merge `sandbox` into the child's result and into `<stem>.report.json`; print one `RESULT` line; exit with the child's code (1 on kill/timeout/no RESULT). With `--no-sandbox`: `pipeline.run` directly; `sandbox: {"enabled": false, ...}`.

- [ ] Failing tests: `Report.finish(ok=True)` with no flags -> exit 0, level "full" when the data says brep + hidden + section + dims; a `warning` flag -> exit 2; `finish(ok=False, error="x")` -> exit 1 and `ok False`; level "reduced" on the mesh route; `run(plate.step, out, DrawingOptions(formats=("svg",)))` (`needs_ocp`, plate written in a fixture) -> exit 0, files `plate.svg` and `plate.report.json` exist, no `.pdf`, report `route.pipeline == "brep"`, `dimensions` contains texts `100`, `60`, `15`, `4 x Ø6.5`, `Ø20`, `section.loops == 2`, `views` has 4 entries with `algorithm == "hlr_exact"`, `geometry.face_types["Cylinder"] == 6`; `run(plate.stl, ...)` (export the plate tessellation with BRepMesh -> trimesh -> STL) -> `route.source == "stl_faceted_step"`, `out/plate.step` exists, `views[0].algorithm == "hlr_poly"`, `route.classifier.verdict == "prismatic"`, `stl_to_step.ok True`; a Taubin-smoothed noisy icosphere STL -> `route.pipeline == "mesh"`, flag `organic_classified`, `stl_to_step.skipped_reason == "organic"`, level "reduced"; `stl_to_step="off"` on the plate STL -> mesh route; monkeypatch `src.drawing.brep_views.project_brep` to raise `RuntimeError("boom")` -> plate STL still produces the SVG with `route.fallbacks[0]["reason"]` containing "boom" and flag `fallback_to_mesh` (exit 2); a 0-face STL -> exit 1, `ok False`; garbage `.step` -> exit 1, no crash; an inch STEP -> flag `unit_not_mm` and exit 2.
  - CLI (subprocess, like `tests/test_convert_cli.py`): `python -m src.drawing plate.step out --no-sandbox --formats svg --quiet` -> exactly one stdout line starting with `RESULT `, exit 0, `sandbox.enabled False`; the sandboxed default run on the same input -> exit 0, one stdout line, `sandbox.enabled True`, `sandbox.isolation in ("job_object", "psutil_tree", "posix_rlimit")`, `peak_rss_mb > 50`, the report file contains the `sandbox` section; missing input -> exit 1 and a RESULT line with `ok False`; `--sheet A5` -> exit 1 with a usage message on stderr and nothing on stdout.
- [ ] Run -> fail. Implement. Run -> pass. Then render `out/plate.pdf` to PNG with the checker venv and LOOK at it (Read tool): views arranged first-angle, hidden hole lines dashed, dims readable, section hatched, title block filled. Fix what looks wrong before reporting.

### Task C2: sandbox

**Files:** Create `src/drawing/sandbox.py`; Test `tests/test_drawing_sandbox.py`.
**Consumes:** nothing from the package (keep it standalone so C1 can import it).
**Probe snippets:** `REPORT-sandbox.md`: `jobobj.py` (ctypes job object: `CreateJobObjectW`, `SetInformationJobObject` class 9 with the 144-byte extended struct, class 7 completion port, `AssignProcessToJobObject(h, int(proc._handle))`, `QueryInformationJobObject` for peaks and times, `TerminateJobObject`, `GetQueuedCompletionStatus` message ids 1 NEW_PROCESS / 6 ACTIVE_PROCESS_ZERO / 7 EXIT_PROCESS / 9 PROCESS_MEMORY_LIMIT / 10 JOB_MEMORY_LIMIT / 8 END_OF_PROCESS_TIME), `capped.py` (threaded reader, 1 MiB head + 64 KiB tail), `worker_proto.py` (the loop: poll 0.1 s, wall timeout, work-dir byte cap, psutil RSS peak, port drain), `sanitized_env` (exact variable list incl. `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `USERPROFILE=TEMP=TMP=work_dir`, `APPDATA` passthrough, `PYTHONSAFEPATH=1`, `PYTHONPATH=repo`), `posix_sandbox_unverified.py`.

- [ ] Failing tests (Windows-only parts marked `pytest.mark.skipif(os.name != "nt")`; POSIX branch has an import/unit test only): child `python -c "import time,subprocess,sys; p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']); print(p.pid, flush=True); time.sleep(60)"` under `timeout_s=3` -> `timed_out True`, `killed True`, `exit_code != 0`, and neither the child PID nor the printed grandchild PID exists afterwards (precondition: `psutil.pid_exists(grandchild)` was True while running — read the PID from the capped stdout before the kill? The result arrives after the kill, so instead have the child write the grandchild PID to a file in `work_dir` immediately; the test reads it and asserts it existed by checking `psutil.Process(pid).create_time()` from a second file the grandchild itself writes... simpler: grandchild = `python -c "import time,os; open('g.pid','w').write(str(os.getpid())); time.sleep(60)"`; the test asserts `g.pid` exists (precondition that it ran) and `not psutil.pid_exists(int(open('g.pid').read()))` after the kill); child allocating `bytearray(1_500_000_000)` under `memory_limit_mb=512` -> `memory_exceeded True`, `exit_code == 1`, `killed False` (the child raised MemoryError), the stderr tail contains `MemoryError`; child printing 50 MB then `RESULT {"ok": true}` -> `stdout_truncated True` and `result == {"ok": True}`; child `print("RESULT " + json.dumps({"ok": True, "x": 1}))` -> `result["x"] == 1`, `exit_code 0`, `wall_s > 0`, `peak_rss_mb > 5`; child printing no RESULT -> `result None`, `sandbox["result_error"] == "no RESULT line"`; `sanitized_env` has `OPENBLAS_NUM_THREADS == "1"`, `PYTHONSAFEPATH == "1"`, `USERPROFILE == work_dir`, no `PYTHONHOME`; a planted `json.py` in the work dir (writes a marker file when imported) is NOT imported by a child running `python -c "import json; print('RESULT {}')"` with the sanitized env (precondition: without `PYTHONSAFEPATH` the marker appears); CPU backstop: a busy loop with `cpu_limit_s=2`, `timeout_s=30` -> `cpu_exceeded True` (exit code & 0xFFFFFFFF == 0xC0000044) within 10 s; return code 3221225477 (0xC0000005) is mapped to `result_error == "native crash (input unreadable by OpenCASCADE?)"` — simulate with a child calling `ctypes.windll.kernel32.ExitProcess(0xC0000005)`.
- [ ] Run -> fail. Implement (`WindowsJob` class, `CappedReader`, `run_sandboxed`, `_posix_run` unverified). Run -> pass (expect ~30 s total).

---

## Stage D — docs, review, polish (after Stage C is committed)

### Task D1: documentation and dependencies

**Files:** Create `docs/drawing_module.md`, `docs/drawing_service_architecture.md`; Modify `README.md` (feature bullet in Features, a "Command-line drawing (STEP/STL → PDF/SVG/DXF)" section after the conversion section, the project-structure tree, dependency tables), `requirements.txt` (add `ezdxf>=1.3` under a `# 2D drawing export` heading and `psutil>=5.9  # optional: sandbox resource reporting`).
**Content:** `drawing_module.md` = usage, every option, routing diagram, report schema (copy from the spec with one real example from the plate run), measured numbers (HLR timings table, classifier table, sandbox facts), ISO-sourced vs conventional rules, known gaps. `drawing_service_architecture.md` = the service architecture in 5–10 bullets (upload -> sniff -> queue -> worker (this module) -> outputs + report; what the container adds: image, non-root user, read-only FS, seccomp, network off, resource limits at the cgroup level duplicating the in-process ones; storage of outputs; API contract), a rough hour breakdown (research/geometry quirks, drawing generation, STL route, sandbox, tests, docs, container+API integration), and the two screening answers written from this repo's history (PROJECT_MISTAKES entries, the memory numbers: SolidWorks face drops, winding-number sign, exact solve, `BRepBndLib.Add_s` over-report, the ReadFile crash, OpenBLAS commit) — honest, specific, no marketing.

### Task D2: full verification and adversarial review (orchestrator-run workflow)

- Full suite: `python -m pytest tests -q -x --ignore=tests/deep_tests` plus `python -m pytest tests/test_gui_fixes.py -q` separately (repo rule); both must be green or the failures explained.
- Visual QA: render the PDFs of (a) the plate STEP, (b) the plate STL (faceted route), (c) the organic blob STL (mesh route), (d) a filleted box, to PNG with PyMuPDF and inspect them.
- Review workflow: three reviewers per area (correctness vs spec, test vacuity, OCCT/trimesh API misuse) -> verified findings -> fixes -> re-run.
- `PROJECT_MISTAKES.md` entries for the traps hit while building.

## Self-review notes

- Spec coverage: 3.1 (A1–C2), 3.2 (C1), 3.3 (A3, B5), 3.4 (B1, B2), 3.5 (B3), 3.6 (A1 style, B5), 3.7 (B4), 3.8 (C1 report), 3.9 (C2), 3.10 (C1), 4 (every task), 5 (D1).
- Type consistency: `ProjectedView.visible/hidden` are `Primitive2D` lists (Segment/Polyline/Circle/Arc) in view coordinates; `SectionView.loops` are `np.ndarray (n,2)` without repeated end point everywhere (geometry2d, brep_sections, mesh_sections, layout); `Dimension` carries view coordinates in `DimensionPlan` and sheet coordinates in `Sheet.dimensions`; `ViewSpec.n` points toward the viewer in both routes and equals the OCCT `gp_Ax2` main direction.
