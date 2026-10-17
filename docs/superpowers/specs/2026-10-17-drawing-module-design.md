# STEP / STL to 2D engineering drawing module — design

Date: 2026-10-17. Branch: `feature/drawing-module` (off `fix/bug-report-2026-10`).
Every number marked *(measured)* comes from the probe scripts under the session
scratchpad `probes/` (reports `REPORT-hlr.md`, `REPORT-brep-analysis.md`,
`REPORT-mesh.md`, `REPORT-writers.md`, `REPORT-sandbox.md`, `REPORT-standards.md`);
the implementation plan copies the relevant snippets.

## 1. Why

An Upwork posting (3D STEP & STL to 2D mechanical engineering drawing module,
fixed price, Danish client, 2026-10-17) asks for a service where a user uploads a
STEP or STL file and receives, with no manual intervention, a basic 2D
engineering drawing: multi-view orthographic projections with dimensions, a
section, a title block, exported as PDF, SVG and DXF, plus a machine-readable
quality report. STEP files go through an OpenCASCADE-based drawing generator;
STL files are first converted to STEP where that produces usable B-rep geometry
and otherwise fall back to mesh-based projection and section views. The worker
must run each job in a sandbox with resource limits. The client's architecture
wants a container; this repo deliberately builds everything *below* the
container boundary (a CLI worker with subprocess isolation) so that it can be
dropped into any container later. No Docker files are part of this work.

This repo already owns the hard half of that pipeline: a validated STL to STEP
converter (`src/convert.py`: smooth NURBS reverse engineering and the exact
faceted B-rep path), a STEP read-back audit, OCCT process-state helpers and a
`RESULT {json}` CLI contract. The drawing module reuses all of it and adds the
2D side. The goal is a working, tested, documented implementation that the
owner can point to in the proposal.

## 2. Scope

In scope:

- `python -m src.drawing input.(step|stp|stl) out_dir/ [options]` producing
  `<stem>.pdf`, `<stem>.svg`, `<stem>.dxf`, `<stem>.report.json`, and for the
  STL route the converted `<stem>.step` when a conversion was made.
- Input sniffing by content (not extension), input limits, SHA-256.
- Routing: STEP -> B-rep pipeline; STL -> prismatic/organic classifier ->
  faceted STEP conversion -> B-rep pipeline, or mesh pipeline when the mesh is
  organic, too large, the conversion fails, the B-rep is unusable, or the user
  says so.
- B-rep pipeline (OCCT): hidden-line-removed orthographic views (visible,
  hidden, outline edges; tangent edges suppressed), isometric view, one section
  view with hatching, geometry census (face types, solids, validity, volume,
  area, exact bbox), cylindrical hole detection for diameter callouts.
- Mesh pipeline (trimesh + numpy, no shapely): feature, boundary and
  silhouette edges, depth-buffer visibility (visible / hidden runs), outline
  from the coverage mask, section loops from plane slicing, hatching, census
  (watertight, components, volume, area, bbox).
- Automatic dimensioning: overall extents without duplication across views,
  hole diameters grouped as `n x Ød`, first-hole position per diameter group;
  capped to avoid clutter; everything not dimensioned is listed in the report.
- Sheet layout: A3 landscape (default, ISO 5457), A4 portrait (ISO) or A4
  landscape (flagged non-ISO), ISO 5455 standard scale, first-angle (default)
  or third-angle arrangement with the projection symbol, ISO 7200 title block
  (180 x 36) plus a 9 mm strip for scale / projection symbol / notes, frame,
  centring marks, view labels, cutting-plane marker A-A.
- Writers: SVG (plain XML), DXF (ezdxf, R2010, layers, true DIMENSION
  entities), PDF (dependency-free vector writer, Helvetica base-14).
- Quality report JSON (schema version 1) with route, geometry, views,
  dimensions, section, outputs, quality flags and level, warnings, timing,
  sandbox section.
- Sandbox worker: child process with wall-clock timeout, memory limit (Windows
  job object + completion port; POSIX rlimits written but unverified), fresh
  working directory, sanitized environment, output caps, whole-tree kill, peak
  RSS, exit-code / RESULT contract.
- Tests (unit, defect-injection, end-to-end CLI), documentation
  (`docs/drawing_module.md`, README section), and an architecture /
  proposal note (`docs/drawing_service_architecture.md`).

Out of scope (stated so nobody expects them): containers, HTTP API, queueing,
GD&T, tolerances, thread or chamfer callouts, auxiliary views, detail views,
multiple sections, BOMs, assembly drawings, ASME-style inch drawings (the sheet
and text defaults are ISO; the arrangement is switchable), GUI integration,
circles/holes on the mesh route.

## 3. Architecture

```
python -m src.drawing in.step out/                 (parent = sandbox worker)
   └─ spawns: python -m src.drawing in.step out/ --no-sandbox --options-json opts.json   (child)
         sniff.py ─► router (pipeline.py) ─► brep_* | mesh_*  ─► ProjectedView / SectionView / CylFeature
         ─► dimensions.py ─► layout.py (Sheet) ─► writers/{svg,dxf,pdf}.py
         ─► report.py ─► stdout: RESULT {json}
   parent: limits, kill, peak RSS, merges the sandbox section into the report file
```

### 3.1 Package `src/drawing/`

| Module | One purpose | Depends on |
|---|---|---|
| `__main__.py` | `python -m src.drawing` entry -> `cli.main` | `cli.py` |
| `cli.py` | argparse into `DrawingOptions`; RESULT contract; exit codes; sandbox-or-direct | `options.py`, `pipeline.py`, `sandbox.py` |
| `options.py` | `DrawingOptions` dataclass, defaults, JSON round trip | – |
| `sniff.py` | detect STEP / binary STL / ASCII STL by content; size and triangle caps; sha256 | – |
| `style.py` | `SheetStyle` + `SHEETS`: every ISO number the layout and writers use | – |
| `model.py` | 2D sheet model: `Sheet`, `View`, primitives, `LineStyle`, `BBox2D` | – |
| `geometry2d.py` | pure 2D maths: polyline simplification, hatch (even-odd, half-open), segment chaining into loops, containment nesting, collinear merge, hidden-under-visible subtraction, dedupe | numpy |
| `projection.py` | `ViewSpec` table (six principal + iso), world -> view transform, auto orientation, gp_Ax2 parameters | numpy |
| `pipeline.py` | the router; runs the right source; dimensions, layout, writers, report | everything below |
| `brep_load.py` | STEP reading (gated `ReadFile`, all roots), unit sniff, census, mass properties, exact bbox, validity, reuse of `step_audit.measure_shape` | OCP, `src.io.occt_utils`, `src.io.step_audit` |
| `brep_features.py` | cylindrical-face grouping into hole / boss / fillet features | OCP |
| `brep_views.py` | HLR projections (exact `HLRBRep_Algo` for B-rep; shell + EncodeRegularity + `HLRBRep_PolyAlgo` for faceted bodies), edge -> 2D primitives | OCP |
| `brep_sections.py` | half-space cut body + section loops | OCP, `brep_views.py` |
| `mesh_load.py` | checked STL loading, census, prismatic/organic classifier | trimesh |
| `mesh_views.py` | per-view edge classification, numpy scanline depth buffer, visible/hidden runs, outline from coverage mask | trimesh, numpy, scikit-image |
| `mesh_sections.py` | plane slicing into closed loops, half-mesh for the section view | trimesh |
| `dimensions.py` | automatic dimension plan and rendering of dimensions into primitives | `model.py`, `style.py` |
| `layout.py` | scale choice, view arrangement, frame, centring marks, title block, strip, projection symbol, section marker, labels | `model.py`, `style.py`, `dimensions.py` |
| `writers/svg.py`, `writers/dxf.py`, `writers/pdf.py` | one writer per format, all consuming a `Sheet` | `model.py` (+ ezdxf) |
| `report.py` | report assembly, quality flags and level, schema version | – |
| `sandbox.py` | the worker: spawn, limits, kill tree, capture, parse RESULT, sandbox section | ctypes, psutil (optional) |

Shared intermediate objects (both sources produce them, so everything from
`dimensions.py` on is source-agnostic):

```python
@dataclass
class ProjectedView:              # one view, in view coordinates (model mm), not shifted
    spec: ViewSpec
    visible: list[Primitive2D]    # Segment / Polyline / Circle / Arc (sharp + outline)
    hidden: list[Primitive2D]
    bbox: BBox2D
    algorithm: str                # "hlr_exact" | "hlr_poly" | "depth_buffer"
    stats: dict                   # counts, seconds

@dataclass
class SectionView:
    spec: ViewSpec                # looking direction of the section
    plane_point: np.ndarray; plane_normal: np.ndarray     # world
    loops: list[np.ndarray]       # closed 2D polygons (view coords), last point != first
    body: ProjectedView | None    # the half body behind the plane
    label: str                    # "A"
    marker_view: str              # the principal view that carries the cutting line

@dataclass
class CylFeature:                 # B-rep route only
    center: np.ndarray; axis: np.ndarray; radius: float; length: float
    kind: str                     # "hole" | "boss" | "fillet"
    full_circle: bool; n_faces: int
```

### 3.2 Routing rules (pipeline.py)

1. `sniff.detect(path)` -> `step`, `stl_binary`, `stl_ascii` or `unknown`.
   STEP: text starts with `ISO-10303-21`. Binary STL: size == 84 + 50·n (the
   header may start with `solid`). ASCII STL: starts with `solid` and the
   facet count comes from a line scan *(measured: 93 MB in 0.25 s)*. Anything
   else -> exit 1, `error`. Size above `max_input_mb` (default 200) or
   triangles above `max_triangles` (default 2,000,000) -> exit 1 before any
   geometry library is imported.
2. STEP -> `brep_load.load_step` -> B-rep pipeline. `ReadFile` must return
   `IFSelect_RetDone` before anything else is called *(measured: calling
   `TransferRoots` after a failed read is a native access violation
   0xC0000005; the sandbox maps that return code to "unreadable input")*.
   No faces -> exit 1. Several roots -> one compound, flag `multiple_bodies`.
3. STL -> `mesh_load.load_stl_checked` (refuses 0 faces; trimesh returns an
   empty mesh for garbage, truncated and STEP-text files *(measured)*) ->
   `classify(mesh)` -> `prismatic | organic` with the features in the report.
   - `--stl-to-step auto` (default): prismatic and triangles <=
     `faceted_max_triangles` (default 30,000) -> faceted STEP
     (`src.io.faceted_step.export_step_faceted(verify=False)` into
     `out_dir/<stem>.step`, in-process `step_audit.measure_shape` on the
     returned shape) -> B-rep pipeline with the faceted HLR path. Otherwise
     mesh pipeline with `stl_to_step.skipped_reason` (`organic` or
     `too_many_triangles`).
   - `--stl-to-step faceted` forces the faceted conversion; `--stl-to-step
     nurbs` runs the smooth converter (`QuadWrapper` + `SubDToNURBSConverter`,
     the same calls as `src.convert.convert`) and draws that B-rep with exact
     HLR; `--stl-to-step off` always uses the mesh pipeline.
   - Any exception, a B-rep with zero faces, or an HLR view that exceeds
     `hlr_time_budget_s` (default 60) falls back to the mesh pipeline. The
     reason is recorded in `route.fallbacks`; the drawing is still produced.
4. Classifier *(measured on 14 meshes)*: `planar_big` = area of coplanar
   facets each larger than 1 % of the total area, divided by the total area.
   CAD-like meshes gave >= 0.20 (worst: a 128-segment cylinder), organic
   meshes (spheres, torus, smoothed blob, genus-28 lattice) gave 0.000.
   Threshold 0.10. Raw `facets_area/area` is useless (1.0 for any quad-strip
   tessellation). Secondary feature reported, not gating: `sharp_turn` =
   share of Σ(edge length · dihedral angle) carried by edges above 30°.

### 3.3 Views and projection (projection.py)

World convention: Z up, the front view shows the -Y face (viewer at -Y
looking toward +Y), X to the right. `--orient auto` (default) permutes the
model axes so that the shortest bbox extent is the depth (Y) and the longest
is the width (X); the permutation is recorded in the report; `--orient fixed`
keeps the file axes; `--up y` rotates a Y-up file first.

OCCT convention *(measured)*: `HLRAlgo_Projector(gp_Ax2(P, N, Vx))` shows the
faces whose normals point toward +N (viewer on the +N side); result 2D
x = Vx·(p−P), y = (N×Vx)·(p−P), Z = 0; coordinates are not shifted (back, left
and bottom views come out negative), so every view is translated by its own
bbox in the layout.

| name | N (toward viewer) | Vx | 2D = |
|---|---|---|---|
| front | (0,−1,0) | (1,0,0) | (X, Z) |
| back | (0,1,0) | (−1,0,0) | (−X, Z) |
| top | (0,0,1) | (1,0,0) | (X, Y) |
| bottom | (0,0,−1) | (1,0,0) | (X, −Y) |
| right | (1,0,0) | (0,1,0) | (Y, Z) |
| left | (−1,0,0) | (0,−1,0) | (−Y, Z) |
| iso | (1,−1,1) | (1,1,0) | up = (−0.408, 0.408, 0.816) |

The mesh route uses the same `ViewSpec` (x axis = Vx, y axis = N×Vx, depth
toward the viewer = N) so both routes agree to the sign.

Arrangement (ISO 5456-2, cells relative to the front view F) *(sourced)*:

```
first angle (ISO, default)          third angle (ASME)
          [ bottom ]                          [  top   ]
[ right ] [   F    ] [ left ]        [ left ] [   F    ] [ right ]
          [  top   ]                          [ bottom ]
```

Default view set: `front,top,right,iso` + section A-A. The iso view goes to
the top-right free cell (above the right-view column in third angle / above
the left-view cell in first angle), the section view goes to the free cell on
the other side of F in the middle row, else to the row below. Views are
aligned: F and top share X; F and right share Y.

### 3.4 B-rep pipeline

- Load: `quiet_occt()`; `STEPControl_Reader.ReadFile` gated on
  `IFSelect_RetDone`; `FileUnits` for the source length unit (cross-checked by
  a Part 21 text sniff that prefers `CONVERSION_BASED_UNIT` over `SI_UNIT`
  and ignores the angle unit); `TransferRoots`; `OneShape` (a compound for
  several roots); `IsNull` check. OCCT converts inch files to mm on read
  *(measured)*. Catch `Exception`, not `Standard_Failure` (OCP exception
  classes derive from `Exception` directly) *(measured)*.
- Census: unique counts via `TopExp.MapShapes_s` (an explorer double-counts
  shared edges), face types by `BRepAdaptor_Surface.GetType()`,
  `BRepCheck_Analyzer.IsValid` (null shape raises -> check first),
  `measure_shape(shape, deviation_samples=0)` for closure / free edges /
  volume, `BRepGProp.SurfaceProperties_s` for area, exact bbox by
  `BRepBndLib.AddOptimal_s(shape, box, False, False)` *(measured: `Add_s`
  over-reports B-splines by the control hull, +10 mm on a r=10 cylinder;
  tessellated nodes under-report by the deflection)*. Flags: `solids == 0`,
  `free_edges > 0`, negative volume (inverted, `ShapeFix_Solid` would heal),
  `brepcheck_invalid`, `multiple_bodies`.
- HLR (exact, for B-rep): `HLRBRep_Algo.Add(shape)`, projector from the
  table, `Update`, `Hide`, `HLRBRep_HLRToShape`. Visible = `VCompound` +
  `OutLineVCompound`; hidden = `HCompound` + `OutLineHCompound`; `Rg1*`,
  `RgN*`, `IsoLine*` dropped (tangent edges are already coded G1 by the
  fillet API, by sewing and by the STEP reader; `EncodeRegularity` is a no-op
  there *(measured)*). Per edge: `BRepLib.BuildCurve3d_s(edge)` in
  try/except (the compound-level call raises on any edge without geometry),
  then `BRepAdaptor_Curve`: Line -> `Segment`; Circle -> `Circle` (full) or
  `Arc` (swap start/end when the circle axis Z < 0, otherwise the complement
  is drawn); Ellipse / BSpline / other -> `GCPnts_QuasiUniformDeflection`
  polyline, collapsed to a `Segment` when every sample lies within tolerance
  of the chord (edge-on circles and straight silhouettes arrive as BSplines).
  Post-processing in `geometry2d`: dedupe, subtract hidden segments that lie
  under visible ones (the back edges of a box and the far mouth of a through
  hole are reported hidden on top of visible edges), merge collinear touching
  segments.
- HLR (faceted bodies from the STL route): build a shell
  (`_shell_from_faces`), `BRepLib.EncodeRegularity_s(shell, facet_smooth_angle)`
  (default 0.35 rad; on a bare compound of faces it silently does nothing),
  `BRepMesh_IncrementalMesh(shape, defl, False, 0.35, True)`, then
  `HLRBRep_PolyAlgo` (`Load`, `Projector`, `Update`;
  `HLRBRep_PolyHLRToShape().Update(algo)`) with the default `TolAngular`
  (raising it shrinks the outline *(measured)*). Exact HLR must not be used on
  encoded faceted shells: it loses a third of the silhouette into `Rg1`
  *(measured)*. Cost *(measured)*: poly 3.8 s at 20k triangles, 27 s at 82k.
- Face-count gate: exact HLR for B-rep up to `exact_hlr_max_faces` (default
  4000); above it the poly path on a `BRepMesh`ed copy with
  `hlr_poly_used` flagged. Exact HLR on a 20k-face body took 9.4 s
  *(measured on planar facets)*.
- Features: per cylindrical face `gp_Cylinder` axis/radius, U span from
  `BRepTools.UVBounds_s`, V range for the length and the centre (the axis
  location can lie outside the face), concavity from the surface normal
  flipped for `TopAbs_REVERSED` faces (without the flip every hole is a boss
  *(measured)*). Coaxial faces with the same radius are grouped (real
  exporters split periodic faces into halves; an OCCT round trip does not).
  `kind`: concave and full circle -> `hole`; concave partial -> `fillet`;
  convex -> `boss`. Cross-check: the text census `cylinder_radii_unique`.
- Section: default plane `auto` = parallel to the front view plane (normal
  Y) through the axis of the largest hole whose axis is perpendicular to Y,
  else through the bbox centre; a plane that produces zero closed loops (miss
  or coplanar with a face) is nudged by 1 % of the depth once, then flagged
  `section_empty`. Loops: `BRepAlgoAPI_Section(shape, pln, False)` with
  `ComputePCurveOn1(True)`, `Approximation(True)`, `Build`; edges connected
  with `ShapeAnalysis_FreeBounds.ConnectEdgesToWires_s`; open wires dropped
  and flagged; discretized into polygons in view coordinates using the view's
  own x/y (the default `gp_Ax2` in-plane basis is arbitrary *(measured)*);
  orientation is arbitrary, nesting is by containment. Body:
  `BRepPrimAPI_MakeHalfSpace(face, point_on_removed_side).Solid()` +
  `BRepAlgoAPI_Cut`, HLR'd with the section's spec *(measured: the cut faces
  coincide with the section loops to 4e-14 mm)*. Hatch: 45°, spacing from
  the style table, even-odd over all loops.

### 3.5 Mesh pipeline

- Load: `trimesh.load(path, force="mesh")` after the sniff; `process=True`
  (vertex merge makes `face_adjacency` work; it does not drop degenerate or
  duplicate faces, those are counted with `nondegenerate_faces` /
  `unique_faces`). Census: watertight, winding consistent, components
  (`graph.connected_components` on the adjacency, 10x faster than `split`),
  `abs(volume)` when watertight (the repo's own lattice fixture is inside
  out), area, bbox, euler number.
- Per view: vertices to view coordinates; candidate edges = sharp
  (`face_adjacency_angles` > `feature_angle`, default 30°) + boundary +
  silhouette (adjacent faces on opposite sides of N); scanline depth buffer
  (`raster_span`, N = `raster_px` default 1024 on the long side *(measured:
  0.05–0.65 s for 12 to 328k triangles)*); each candidate edge sampled about
  one pixel apart, visible when its depth is within 1.5 px of the **3x3
  minimum** of the buffer (a single-pixel test calls outline-coincident edges
  hidden *(measured: 2–54 % wrong)*; with the window 0.00–0.14 % on CAD
  meshes), split into runs. Organic meshes (classifier verdict) take their
  outline from `skimage.measure.find_contours` on the padded coverage mask +
  `approximate_polygon(0.75 px)` (closed outer and inner loops, error <= 1 px)
  instead of silhouette edges, which fragment on noisy surfaces. The first
  `approximate_polygon` call costs 0.4–2.9 s *(measured)* and is counted in
  the time budget.
- Section: `trimesh.intersections.mesh_plane(mesh, plane_normal=, plane_origin=)`
  (keyword arguments; the positional order is a trap) -> segments chained into
  closed loops with a KD-tree weld at 1e-6 · bbox diagonal; a coplanar plane
  gives zero segments like a miss, so the plane is nudged as in 3.4. Body:
  `trimesh.intersections.slice_mesh_plane` keeps the half behind the plane,
  projected with the depth buffer.
- No hole features on this route (flag `holes_not_detected_mesh_route`).

### 3.6 Standards table, dimensioning and layout (style.py, dimensions.py, layout.py)

`SheetStyle` constants *(sourced from ISO 5457, 7200, 128-20, 128-24, 128-40,
3098-0, 5455, 5456-2/-3, 129; the ones marked conv. are DIN 406 teaching
practice, not ISO)*:

- Sheets: A3 landscape 420 x 297 (default); A4 portrait 210 x 297 (ISO 5457
  allows A4 only in portrait); A4 landscape 297 x 210 offered as `A4L` and
  flagged `sheet_not_iso5457`. Frame: 20 mm left (filing margin), 10 mm
  elsewhere, 0.7 mm line; four centring marks 0.7 mm from 5 mm outside the
  frame to 10 mm inside.
- Title block (ISO 7200 Fig. 2): 180 x 36 mm at the bottom-right inside the
  frame, 4 rows of 9 mm, columns at 0 | 32 | 74.2 | 129 | 180 mm. Fields:
  legal owner, identification number (file stem), revision, date of issue,
  sheet 1/1, title, document type ("Auto-generated drawing"), creator
  ("surface_modeling drawing module"), approval (blank, flagged). A separate
  180 x 9 strip directly above it carries SCALE, the projection symbol,
  "Dimensions in mm", and the source note (`STEP`, `STL->faceted STEP`,
  `STL mesh projection`).
- Lines (group 0.5): wide 0.5 (visible, outline, section outline, cutting
  line ends 04.2, section arrows), narrow 0.25 (hidden 02.1, centre, dimension,
  extension, leader, hatch, title block inner lines), frame 0.7, text stroke
  0.35. Dashes at d = 0.25: hidden 3.0 / 0.75; centre 6 / 0.75 / 0.125 / 0.75;
  cutting line (wide, d = 0.5) 12 / 1.5 / 0.25 / 1.5.
- Text: 3.5 mm dimensions and notes; 5 mm section letters, view labels, title
  and identification number; 2.5 mm title-block labels; sans-serif.
- Dimensions (ISO 129 method 1): closed filled arrowhead, 15° included angle,
  3.5 mm long; extension gap 1 mm (conv.), overshoot 2 mm (conv.); first
  dimension line 10 mm from the outline, then 7 mm pitch (conv.); value 1 mm
  above the line, centred, reading from the bottom (horizontal) or the right
  (vertical); `Ø` prefix, `n x Ø6.5`, no units.
- Hatch: narrow 0.25 at 45°; spacing 2 mm when the section's shorter extent
  on the sheet is < 20 mm, 3 mm up to 80 mm, 4 mm above (conv.), never below
  1.0 mm.
- Section marker: cutting line = wide 04.2 dashes ~12 mm beyond the outline
  at both ends; arrows = wide 10 mm shaft with a filled 30° head 5 mm long
  pointing in the viewing direction; letter `A` 5 mm beside each arrow;
  label `A-A` (5 mm) immediately above the section view (ISO).
- Projection symbol at h = 3.5: circles Ø3.5 and Ø7, cone 7 long with ends 3.5
  and 7, gap 1.05, line 0.35; first angle = circles to the RIGHT of the cone,
  third angle = circles to the LEFT.
- Scales (ISO 5455): 1:1, 1:2, 1:5, 1:10, 1:20, 1:50, 1:100; 2:1, 5:1, 10:1,
  20:1, 50:1. The largest scale at which every view bbox plus its dimension
  bands (10 + 7·k mm) plus 20 mm gaps fits the drawing space minus the title
  block area wins; `--sheet auto` tries A4 portrait, then A3 landscape; if
  nothing fits at 1:100 the smallest scale is used and `views_overflow` is
  flagged. Iso view at the true orthographic projection (not the 1.225x
  isometric drawing), labelled with its scale.

Automatic dimension plan (`plan_dimensions`):

1. Overall extents: width (view X) under the front view, height (view Y) left
   of the front view, depth (world Y) as a vertical dimension beside the top
   view. Nothing repeated in other views.
2. Holes: for each drawn principal view, `hole` features whose axis is
   parallel to N (abs(dot) > 0.999); grouped by radius (tolerance 0.01 mm);
   one diameter callout per group on the circle nearest the view's top-right,
   45° leader, text `4 x Ø6.5` or `Ø20`; at most `max_hole_callouts` (default
   6) groups per view by decreasing count then radius; dropped groups go to
   `undimensioned`.
3. Hole positions: the callout circle of each group gets X and Y linear
   dimensions from the view's bbox minimum corner in the next dimension band;
   `--no-hole-positions` turns it off.
4. Everything is rendered into `LineStyle.DIMENSION` primitives and text; the
   DXF writer also receives the semantic `Dimension` objects to emit real
   `DIMENSION` entities.

### 3.7 Writers

All writers take a `Sheet` in sheet millimetres (origin bottom-left, Y up).

- SVG: `viewBox="0 0 W H"`; geometry in `<g transform="translate(0,H)
  scale(1,-1)">`, text OUTSIDE that group at y' = H − y with a manual baseline
  offset of 0.718·size/2 (text inside a flipped group renders upside down;
  `dominant-baseline` is ignored by MuPDF *(measured)*); `stroke-width` and
  `stroke-dasharray` in mm; `text-anchor`; font-family Helvetica, Arial,
  sans-serif; UTF-8 `Ø`; rotated text via `rotate(-90 x y)`; arcs as `A r r 0
  large 1`, full circles as `<circle>`.
- DXF: `ezdxf.new("R2010", setup=True)`; `$INSUNITS=4`, `$MEASUREMENT=1`,
  `$LWDISPLAY=1`; layers VISIBLE (lw 50), HIDDEN (linetype **DASHED** —
  `HIDDEN` does not exist after `setup=True` and is silently replaced by
  CONTINUOUS *(measured)*; lw 25), SECTION (50), HATCH (HATCH entity, ANSI31,
  25), CENTER (CENTER, 25), DIM (DIMENSION entities, `dimstyle="Standard"` +
  overrides `dimtxt 3.5, dimasz 3.5, dimexe 2, dimexo 1, dimdec 1, dimtad 1,
  dimgap 1, dimdsep 46`; the `EZDXF` style has `dimlfac=100` and tick arrows,
  `ISO-25` does not exist; `add_diameter_dim` adds `%%c` itself), TEXT, FRAME
  (lw 50), TITLE (25). Model space at sheet scale. Tests re-read with
  `ezdxf.readfile` and require `audit()` to report 0 errors and 0 fixes.
- PDF: dependency-free writer ported from the probe's `minipdf.py` *(measured:
  PyMuPDF opens it with `is_repaired=False`, no warnings; Helvetica width table
  matches MuPDF exactly for all 100 characters)*: page in points, one
  FlateDecode content stream, Helvetica WinAnsi (cp1252; `Ø ° ±` encode, the
  U+2300 diameter sign does not and is mapped to `Ø`), dashes and widths in
  pt, rotated text via `Tm`, arcs as <= 90° Béziers (k = 4/3·tan(span/4)).
  matplotlib is not used (Type3 fonts, linewidth-relative dashes, 7 kB for 6
  primitives).

Determinism: identical input and options give byte-identical SVG and PDF
except the date field; DXF is compared after masking handles and the date.

### 3.8 Quality report (report.py)

```json
{
  "schema": "surface_modeling.drawing.report/1",
  "ok": true, "exit_code": 0, "error": null,
  "input": {"path": "...", "detected_format": "step|stl_binary|stl_ascii", "extension": ".stp",
            "size_bytes": 0, "sha256": "...", "triangles_declared": null,
            "source_unit": "mm|inch|unknown"},
  "route": {"pipeline": "brep|mesh", "source": "step|stl_faceted_step|stl_nurbs_step|stl_mesh",
            "classifier": {"verdict": "prismatic|organic|n/a", "planar_big": 0.0, "sharp_turn": 0.0,
                           "threshold": 0.10},
            "stl_to_step": {"attempted": false, "mode": "auto|faceted|nurbs|off", "ok": null,
                             "seconds": null, "skipped_reason": null, "error": null, "audit": {}},
            "fallbacks": [{"from": "brep", "to": "mesh", "reason": "..."}],
            "orientation": {"permutation": [0, 1, 2], "flips": [1, 1, 1], "mode": "auto"}},
  "geometry": {"bbox_min": [0,0,0], "bbox_max": [0,0,0], "extents": [0,0,0], "volume_mm3": 0.0,
               "area_mm2": 0.0, "solids": 0, "shells": 0, "faces": 0, "edges": 0,
               "face_types": {"Plane": 0, "Cylinder": 0}, "brepcheck_valid": true, "free_edges": 0,
               "closed": true, "triangles": 0, "watertight": true, "components": 1,
               "features": {"holes": 0, "bosses": 0, "fillets": 0}},
  "views": [{"name": "front", "algorithm": "hlr_exact|hlr_poly|depth_buffer", "visible": 0,
             "hidden": 0, "seconds": 0.0}],
  "dimensions": [{"kind": "linear|diameter", "view": "front", "value_mm": 0.0, "text": "Ø6.5", "count": 1}],
  "undimensioned": [{"kind": "hole_group", "radius": 0.0, "count": 0, "reason": "cap"}],
  "section": {"label": "A", "plane_point": [0,0,0], "plane_normal": [0,1,0], "loops": 0,
              "open_wires_dropped": 0, "hatch_segments": 0, "nudged": false},
  "sheet": {"size": "A3", "orientation": "landscape", "scale": "1:2", "projection": "first_angle",
            "iso5457": true},
  "outputs": {"pdf": "...", "svg": "...", "dxf": "...", "step": null, "report": "..."},
  "quality": {"level": "full|reduced|minimal",
              "flags": [{"code": "...", "severity": "info|warning", "detail": "..."}]},
  "warnings": [], "timing": {"total_s": 0.0, "load_s": 0.0, "views_s": 0.0, "section_s": 0.0,
                             "write_s": 0.0},
  "sandbox": {"enabled": true, "isolation": "job_object|psutil_tree|posix_rlimit|none",
              "timeout_s": 180, "memory_limit_mb": 2048, "cpu_limit_s": 360, "peak_rss_mb": 0.0,
              "peak_commit_mb": 0.0, "wall_s": 0.0, "cpu_s": 0.0, "timed_out": false,
              "memory_exceeded": false, "cpu_exceeded": false, "output_exceeded": false,
              "killed": false, "exit_code": 0, "stdout_truncated": false, "stderr_truncated": false,
              "result_error": null, "stderr_tail": ""},
  "version": {"module": "0.1.0", "ocp": "7.9.3", "trimesh": "5.0.0", "ezdxf": "1.4.4"}
}
```

Quality level: `full` = B-rep route with hidden lines, a section and at least
the overall dimensions; `reduced` = mesh route, or a B-rep route that lost
hidden lines or the section or used the poly HLR; `minimal` = views only.
`warning` flags set exit code 2 (drawing written, look at it):
`mesh_not_watertight`, `multiple_bodies`, `brepcheck_invalid`, `open_shell`,
`inverted_solid`, `fallback_to_mesh`, `hlr_timeout`, `section_empty`,
`section_open_wires`, `views_overflow`, `dimensions_capped`, `unit_not_mm`,
`sheet_not_iso5457`, `approval_blank`. `info` flags:
`holes_not_detected_mesh_route`, `organic_classified`, `too_many_triangles`,
`extension_mismatch`, `facet_edges_suppressed`, `hlr_poly_used`,
`section_nudged`, `degenerate_faces_removed`.

### 3.9 Sandbox worker (sandbox.py)

- The parent is `python -m src.drawing ...` without `--no-sandbox`. Before
  importing any CAD library it runs the sniff and the size / triangle caps,
  then re-launches `sys.executable -m src.drawing <in> <out_dir> --no-sandbox
  --options-json <work_dir>/options.json` with cwd = a fresh
  `tempfile.mkdtemp()` work dir, `stdin=DEVNULL`, `creationflags =
  BELOW_NORMAL_PRIORITY_CLASS | CREATE_NO_WINDOW` on Windows.
- Environment (exactly, *(measured)*): `SYSTEMROOT`, `PATH`, `APPDATA`
  (every CAD package lives in the user site-packages; `-I`/`-s` break all
  imports), `USERPROFILE=TEMP=TMP=work_dir` (ezdxf and cadquery raise at
  import without a home directory), `PYTHONPATH=<repo root>`,
  `PYTHONSAFEPATH=1` (keeps the cwd off `sys.path`; a planted `json.py` is
  otherwise imported), `PYTHONIOENCODING=utf-8`, `PYTHONDONTWRITEBYTECODE=1`,
  `MPLCONFIGDIR=work_dir`, `MPLBACKEND=Agg`, and **`OPENBLAS_NUM_THREADS=1`,
  `OMP_NUM_THREADS=1`** (without it OpenBLAS commits ~628 MB at `import numpy`
  on this 20-thread machine and the child dies under any cap below ~650 MB
  with "OpenBLAS error" before any Python runs).
- Limits: `--timeout` (default 180 s; library imports alone cost 4–5 s, so
  the floor is 15 s), `--memory-limit-mb` (default 2048; OCP + trimesh import
  is 192 MB committed), CPU backstop 2 x timeout, stdout/stderr capped at 1
  MiB head + 64 KiB tail by two reader threads (no deadlock, the RESULT line
  is recovered from the tail), work-dir output cap 200 MB polled every 0.1 s.
- Windows: job object (`JOB_OBJECT_LIMIT_PROCESS_MEMORY | JOB_MEMORY |
  KILL_ON_JOB_CLOSE | DIE_ON_UNHANDLED_EXCEPTION | PROCESS_TIME`) with a
  completion port; nested jobs work (the worker itself already runs inside a
  job) *(measured)*. A memory breach does NOT kill the child: the allocation
  is refused and Python raises `MemoryError`, which the child turns into
  `RESULT ok:false`; `memory_exceeded` therefore comes from the port message
  `PROCESS_MEMORY_LIMIT` (or `MemoryError` / `Unable to allocate` / `OpenBLAS
  error` in stderr), never from the peak counter (a refused allocation never
  shows in it). Timeout or output-cap breach -> `TerminateJobObject` (3 ms,
  kills grandchildren too). CPU breach -> exit 0xC0000044. Fallback when the
  job object cannot be created: psutil tree kill + RSS polling,
  `isolation="psutil_tree"` (it misses re-parented grandchildren). POSIX:
  `start_new_session=True`, rlimits applied by the child itself from an env
  variable, `killpg`; written, marked unverified.
- Contract: the child prints exactly one `RESULT {json}` line on stdout and
  exits 0/2/1 like `src/convert.py`. The parent parses the tail's last line,
  adds `sandbox`, rewrites `out_dir/<stem>.report.json`, prints its own single
  `RESULT` line and exits with the child's code; 1 when the child died, timed
  out, breached a limit or produced no RESULT (`result_error`). Return code
  0xC0000005 is reported as "input unreadable by OpenCASCADE".

### 3.10 Error handling

Nothing in the child raises to the top: `pipeline.run` returns
`(exit_code, report)`; every stage is wrapped, failures become `error` /
`warnings` / `fallbacks` entries, and a partial drawing is still written when
at least one view exists. Usage errors exit 1 with argparse's message on stderr
(the `_Parser` pattern from `src/convert.py`). OCCT's console printer is
removed (`quiet_occt`) so stdout keeps the contract (the STEP parse error
banner otherwise lands on stdout *(measured)*).

## 4. Testing

- Unit (no OCP): `sniff` (STEP text; binary STL whose header says `solid`;
  ASCII STL with exact facet count; truncated binary -> unknown; 0 bytes;
  garbage; caps), `geometry2d` (hatch over a square with a square hole skips
  the hole and totals ≈ area/spacing; a hatch line through two corners gives
  exactly two segments; chaining with a plane through vertices; containment
  nesting; collinear merge; hidden-under-visible subtraction; dedupe),
  `projection` (auto orientation permutation; view axes orthonormal and
  right-handed; the table equals the probe's), `dimensions` (no duplicate
  extents; `4 x Ø6.5` grouping; cap -> `undimensioned`), `layout` (scale pick
  for known extents; A4 portrait -> A3 promotion; first/third-angle cells;
  F/top share X, F/right share Y; title block geometry), writers (SVG parses,
  text outside the flipped group, dasharray attributes present; DXF re-read +
  audit 0/0 + entity counts per layer + DIMENSION entities + `%%c` once; PDF
  header, xref offsets equal the object byte positions, trailer; optional
  PyMuPDF checks in the checker venv when present: `is_repaired False`, page
  size, strings present, dash arrays, text centred).
- B-rep (OCP): plate 100x60x15 with four Ø6.5 holes at the corners, one Ø20
  hole in the middle, a Ø30x10 boss *(the probe's fixture)*: census Plane 7 /
  Cylinder 6, 13 faces, 30 edges; features 5 holes in 2 groups + 1 boss, also
  after `ShapeUpgrade_ShapeDivideClosed` splitting (12 half faces -> 6
  features); negative control: without the REVERSED flip the holes read as
  bosses; front view has hidden hole lines; top view has 5 circles; a filleted
  box shows no tangent lines (`Rg1` dropped) but keeps the corner arcs;
  section y = 30 gives 2 loops with |areas| 600 and 900; hatch segments > 0;
  plan dimensions `100`, `60`, `15`, `4 x Ø6.5`, `Ø20`; an inch STEP reports
  `source_unit: inch` and `unit_not_mm`; garbage STEP -> `ok False` without a
  crash (ReadFile gate); two-root STEP -> `multiple_bodies`.
- Mesh (trimesh): two offset boxes -> the back box's edge splits into visible
  and hidden runs with the split within 2 px of the analytic value; negative
  control without the front box; icosphere outline is one closed contour;
  torus section through the axis gives 2 loops, perpendicular 2 concentric
  loops; classifier: box, plate tessellation -> prismatic (`planar_big` >=
  0.2 asserted as a precondition), icosphere and smoothed blob -> organic
  (`planar_big` == 0 asserted).
- Routing: plate STL -> faceted STEP written -> `route.source ==
  "stl_faceted_step"`, `algorithm == "hlr_poly"`; blob STL -> mesh route with
  `organic_classified`; `--stl-to-step off` forces mesh; a monkeypatched
  exception in the B-rep stage -> `fallbacks` entry and a drawing still
  written; 0-face mesh -> exit 1.
- Sandbox: sleeping child with a grandchild under a 3 s timeout ->
  `timed_out`, `killed`, no surviving PID from the recorded set (precondition:
  the grandchild existed); 1.5 GB allocation under 512 MB -> `memory_exceeded`
  True and the child's RESULT still parsed; 50 MB stdout flood ->
  `stdout_truncated` and RESULT recovered; planted `json.py` in the work dir
  is not imported; env contains `OPENBLAS_NUM_THREADS`.
- CLI end to end: `python -m src.drawing plate.step out/` and `plate.stl`
  produce pdf/svg/dxf/report with the required keys, exactly one stdout line,
  exit 0; `--formats svg` writes only SVG; bad input -> exit 1 and a RESULT
  line with `ok: false`; `--no-sandbox` path identical except the sandbox
  section.
- Discipline (repo rule): every defect test asserts its precondition and
  nothing passes against a no-op implementation; OCP imports stay flat
  (`tests/test_ocp_layout.py` scans `src`); GUI tests untouched.

## 5. Documentation

- `docs/drawing_module.md`: usage, options, routing, report schema, limits,
  measured numbers (HLR timings, classifier table, sandbox facts), known gaps,
  which rules are ISO-sourced and which are conventional.
- `docs/drawing_service_architecture.md`: the service architecture in 5–10
  bullets as it would be proposed (worker, sandbox, routing, outputs, report,
  what the container adds later), a rough hour breakdown, the two screening
  answers backed by this repo's history.
- README: feature bullet, CLI section, project-structure tree, dependency
  table (`ezdxf` required for DXF, `psutil` optional, `scikit-image` already
  present).
- `requirements.txt`: add `ezdxf>=1.3`, `psutil>=5.9` (optional comment).
- `PROJECT_MISTAKES.md`: traps found while building.

## 6. Decisions made without the owner (autonomous run)

The owner asked for the build without a review loop, so these calls were made
here and are easy to flip later: first-angle default (European client); A3
landscape default because ISO 5457 forbids A4 landscape (A4 portrait and a
flagged A4 landscape are options); `front,top,right,iso` + one section by
default; faceted (not NURBS) conversion as the default STL->STEP attempt with
a 30k-triangle cap (exact, fast, HLR-able), NURBS behind a flag;
dependency-free PDF writer (probe-verified); ezdxf as a real dependency for
DXF with DASHED hidden lines; model-space DXF; no containers (owner's
instruction); hole callouts only on true B-rep input.
