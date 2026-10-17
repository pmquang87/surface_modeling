# STEP / STL → 2D engineering drawing (`src/drawing`)

`python -m src.drawing part.step out/` turns a STEP or STL file into a basic
2D engineering drawing — orthographic views with hidden lines, overall and
hole dimensions, one hatched section, an ISO sheet with title block — written
as PDF, SVG and DXF, together with a machine-readable quality report. The job
runs in a sandboxed child process with time and memory limits and reports
through the same `RESULT {json}` stdout contract as `python -m src.convert`.

Design: `docs/superpowers/specs/2026-10-17-drawing-module-design.md`.
Service context and proposal material: `docs/drawing_service_architecture.md`.

## Usage

```bash
python -m src.drawing part.step out/                        # A3, first angle, front/top/right/iso + section A-A, pdf+svg+dxf
python -m src.drawing part.stl  out/                        # STL: classifier -> faceted STEP -> B-rep drawing, or mesh fallback
python -m src.drawing part.stl  out/ --stl-to-step off      # draw from the triangles directly
python -m src.drawing part.step out/ --sheet A4 --projection third --formats pdf
python -m src.drawing part.step out/ --views front,top,right,left,iso --section top
python -m src.drawing part.step out/ --no-sandbox --quiet   # in-process (what the sandboxed parent runs as its child)
```

Outputs in `out/`: `<stem>.pdf`, `<stem>.svg`, `<stem>.dxf` (as selected by
`--formats`), `<stem>.report.json`, and `<stem>.step` when an STL was
converted.

Exit codes: `0` clean; `2` drawing written but the report carries a
`warning`-severity flag (read `quality.flags`); `1` failed (bad input, limits
exceeded, timeout, no drawing). Progress goes to stderr; stdout carries
exactly one line `RESULT {json}` (the report).

### Options

| Option | Default | Meaning |
|---|---|---|
| `--formats` | `pdf,svg,dxf` | any subset |
| `--views` | `front,top,right,iso` | from `front back top bottom left right iso` |
| `--section` | `auto` | `auto` = plane parallel to the front view through the largest hole axis (else bbox centre); `front`/`top`/`right` choose the plane orientation; `off` |
| `--sheet` | `A3` | `A3` (landscape), `A4` (portrait, ISO), `A4L` (landscape, flagged non-ISO), `auto` (A4 then A3) |
| `--projection` | `first` | `first` (ISO, top view below the front view) or `third` (ASME) |
| `--orient` | `auto` | `auto`: longest extent → width, shortest → depth; `fixed`: file axes |
| `--up` | `z` | `y` rotates a Y-up file to Z-up first |
| `--stl-to-step` | `auto` | `auto` (faceted when prismatic and ≤ 30k triangles), `faceted`, `nurbs` (smooth reverse engineering, slow), `off` |
| `--no-hidden` | | omit hidden lines |
| `--hidden-lines-force` | | keep hidden lines on large assemblies (default: omitted above 20 solids or 50 000 raw hidden edges per view, flag `hidden_lines_omitted_assembly`) |
| `--no-hole-positions` | | diameter callouts only |
| `--title`, `--owner` | file stem, empty | title block fields |
| `--timeout` | 180 s | wall clock of the child (floor 15 s: imports alone cost 4–5 s) |
| `--memory-limit-mb` | 2048 | committed-memory cap of the child |
| `--no-sandbox` | | run in-process |
| `--quiet` | | no progress on stderr, except one `INFO src.drawing.stage: stage ...` line per stage boundary (elapsed, time left, RSS) |

Programmatic use: `from src.drawing.pipeline import run; code, report = run(path, out_dir, DrawingOptions(...))`.

## How a file is routed

```
sniff by content (STEP header | binary STL 84+50n | ASCII STL facet count) -> size and triangle caps
  STEP ─────────────────────────────────────────────► B-rep route (exact HLR)
  STL  ─► classifier ─ prismatic & ≤ 30k tri ─► faceted STEP (src.io.faceted_step) ─► B-rep route (polyhedral HLR)
          └─ organic | too large | conversion failed | HLR over budget ─► mesh route (depth buffer)
```

- **B-rep route.** `STEPControl_Reader` (gated on `ReadFile == RetDone`; a
  transfer after a failed read is a native crash), source unit from
  `FileUnits` and a text sniff, all roots as one compound. Census: face types,
  unique solids/shells/faces/edges, `BRepCheck`, closure and free edges
  (`step_audit.measure_shape`), volume, area, exact bbox
  (`BRepBndLib.AddOptimal`). Views: `HLRBRep_Algo`; visible = sharp + outline
  compounds, hidden = their hidden counterparts, tangent (Rg1/RgN) edges
  dropped; lines and circles stay exact, ellipses and splines are discretized;
  hidden segments lying on visible ones are removed. Holes: cylindrical faces
  grouped by axis and radius (exporters split periodic faces), concave + full
  circle = hole. Section: `BRepAlgoAPI_Section` loops (closed wires only) +
  half-space cut body HLR'd, hatched at 45°.
- **Faceted route (STL → STEP).** The mesh becomes an exact planar-face solid
  written to `<stem>.step`; for the drawing the same faces are assembled into
  a shell, edges between near-coplanar facets are coded smooth
  (`EncodeRegularity`, 20° default), and `HLRBRep_PolyAlgo` projects it. No
  circles exist on this route, so there are no hole callouts (flagged).
- **Mesh route.** trimesh load (own guard: trimesh returns an empty mesh for
  garbage, truncated and STEP-text files), census, per view: sharp (> 30°),
  boundary and silhouette edges, a numpy scanline depth buffer (1024 px on
  the long side), visibility by comparing each edge sample with the 3×3
  minimum of the buffer (1.5 px tolerance), outline of organic shapes from the
  coverage-mask contour; sections from `mesh_plane` segments chained into
  loops; the half mesh behind the plane for the section body.

### The classifier

`planar_big` = area of coplanar facets each larger than 1 % of the total
surface, divided by the total surface. Measured on 14 synthetic meshes:

| mesh | planar_big | sharp_turn | verdict |
|---|---|---|---|
| box with hole (coarse / fine) | 0.927 / 0.927 | 0.931 / 0.931 | prismatic |
| plate with 5 holes and boss | 0.780 | 0.782 | prismatic |
| cylinder, 32 / 128 segments | 1.000 / 0.200 | 0.440 | prismatic |
| voxel-remeshed CAD part, Taubin-smoothed | 0.707 | 0.604 | prismatic |
| icosphere (any density) | 0.000 | 0.000 | organic |
| torus | 0.000 | 0.000 | organic |
| noisy smoothed blob | 0.000 | 0.000 | organic |
| genus-28 smoothed lattice (repo fixture) | 0.000 | 0.625 | organic |

Threshold 0.10. The raw planar fraction `facets_area/area` cannot be used
(1.0 for every quad-strip tessellation), and `sharp_turn` alone does not
separate the lattice from the cylinders; both values are reported so a wrong
verdict can be diagnosed. Not yet validated on real scans or fillet-heavy CAD
tessellations.

## Sheet conventions

Line group 0.5 (ISO 128-24): wide 0.5 mm (visible edges, section outline,
cutting line ends, section arrows), narrow 0.25 mm (hidden, centre,
dimension, extension, leader, hatch), frame 0.7. Dashes (ISO 128-20, multiples
of the line width): hidden 3 / 0.75, centre 6 / 0.75 / 0.125 / 0.75, cutting
line 12 / 1.5 / 0.25 / 1.5. Text 3.5 mm (ISO 3098 type B proportions), 5 mm
for section letters and the title. Dimensioning per ISO 129 method 1: closed
filled arrowheads (15°), `Ø` before the value, `4 x Ø6.5` for repeated holes,
no units (a "Dimensions in mm" note sits in the strip). Sheet frame per ISO
5457 (20 mm filing margin, 10 mm elsewhere, centring marks); ISO 5457 permits
A4 only in portrait, so `A4L` is flagged `sheet_not_iso5457`. Title block per
ISO 7200 Fig. 2 (180 × 36, 9 mm rows, columns 32 / 74.2 / 129); scale,
projection symbol and the units note are deliberately outside it (ISO 7200
clause 4) in a 180 × 9 strip. View arrangement per ISO 5456-2: first angle
puts the top view below and the right view to the left of the front view;
third angle mirrors both; projection symbol with the circles to the right of
the cone for first angle, left for third. Section label `A-A` immediately
above the section view (ISO 128-40).

Conventional (DIN 406 teaching practice, not ISO): first dimension line 10 mm
from the outline, 7 mm pitch, 1 mm extension gap, 2 mm overshoot, hatch
spacing 2 / 3 / 4 mm by section size, 20 mm between views. The report never
cites ISO for these.

## Quality report

Schema `surface_modeling.drawing.report/1`; the full field list is in spec
section 3.8. The parts a consumer should read first:

- `quality.level`: `full` (B-rep route with hidden lines, section and
  dimensions), `reduced` (mesh route, or a B-rep route that lost hidden lines,
  the section or used the polyhedral HLR), `minimal` (views only).
- `quality.flags[]`: `{code, severity, detail}`. `warning` flags set exit code
  2: `mesh_not_watertight`, `multiple_bodies`, `brepcheck_invalid`,
  `open_shell`, `inverted_solid`, `fallback_to_mesh`, `hlr_timeout`,
  `section_empty`, `section_open_wires`, `views_overflow`, `dimensions_capped`,
  `unit_not_mm`, `sheet_not_iso5457`, `approval_blank`. `info` flags:
  `holes_not_detected_mesh_route`, `organic_classified`, `too_many_triangles`,
  `extension_mismatch`, `facet_edges_suppressed`, `hlr_poly_used`,
  `section_nudged`, `degenerate_faces_removed`, `hidden_lines_omitted_assembly`.
  `section_skipped_budget` (warning): the run deadline was too close for
  the section.
- `writers.dxf_audited`: the DXF was re-read and audited by ezdxf (False
  above 100 000 sheet items or with less than 10 s left before the run
  deadline; `writers.dxf_audit_skipped_reason` says which).
- `route`: pipeline, source, classifier values, the STL→STEP attempt with its
  audit, every fallback with its reason, the orientation permutation.
- `geometry`: census (B-rep or mesh), exact bbox, volume, area, features.
- `views[]`, `dimensions[]`, `undimensioned[]`, `section`, `sheet`, `outputs`.
- `sandbox`: isolation kind, limits, peak RSS and committed memory, wall and
  CPU seconds, `timed_out`, `memory_exceeded`, `cpu_exceeded`,
  `output_exceeded`, `killed`, raw child exit code, output truncation,
  `result_error`, stderr tail.

## Sandbox

The parent process sniffs and caps the input, then runs
`python -m src.drawing ... --no-sandbox --options-json <file>` as a child with
a fresh working directory, below-normal priority and a minimal environment:
`SYSTEMROOT`, `PATH`, `APPDATA` (the CAD packages live in the user
site-packages; `-I`/`-s` would hide them), `USERPROFILE=TEMP=TMP=<work dir>`
(ezdxf needs a home directory), `PYTHONPATH=<repo>`, `PYTHONSAFEPATH=1`,
`PYTHONIOENCODING=utf-8`, `PYTHONDONTWRITEBYTECODE=1`, `MPLCONFIGDIR`,
`MPLBACKEND=Agg`, `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`.

Windows: a job object (process + job memory limit, kill-on-close,
die-on-unhandled-exception, CPU-time backstop) with a completion port. A
memory breach does not kill the child: the allocation is refused, Python
raises `MemoryError`, the child reports `ok: false`; `memory_exceeded` comes
from the port message. Timeout and output-cap breaches terminate the whole
job (child and grandchildren). Fallback without a job object: psutil tree kill
and RSS polling (`isolation: psutil_tree`; misses re-parented grandchildren).
POSIX: `start_new_session`, rlimits, `killpg` — written, not verified on this
machine. stdout/stderr are captured by threads with a 1 MiB head and a 64 KiB
tail cap; the `RESULT` line is parsed from the tail.

Measured on this machine: OpenCASCADE + trimesh import ≈ 192 MB committed
with `OPENBLAS_NUM_THREADS=1` (≈ 630 MB without it, which dies under any cap
below ~650 MB); library imports ≈ 4–5 s per job; `TerminateJobObject` ≈ 3 ms;
a failed STEP read followed by a transfer exits with 0xC0000005, which the
parent reports as "input unreadable by OpenCASCADE".

## Large assemblies and the run deadline

- **Census above the face gate** (`exact_hlr_max_faces`, 4000 face
  instances): BRepCheck and the adaptive volume run once per unique solid
  (instances grouped by `TShape`, volume x `ScaleFactor^3` per instance,
  sign flipped for a REVERSED instance) and the bbox is
  `brep_load.exact_bbox_bounded`: the per-face loose `Add_s` boxes bound a
  branch and bound, only the faces near the six extremes get `AddOptimal`.
  The box is bit-identical to `AddOptimal_s` of the whole shape (a loose box
  would print 0.43 mm too much into the overall dimensions of the RC buggy).
  `geometry.census_mode` / `bbox_source` / `solid_instances` say which path
  ran. The section's per-body boxes use the same bounded box.
- **Hidden lines**: more than 20 solid instances, or a view with more than
  50 000 raw hidden HLR edges, is drawn without hidden lines
  (`hidden_lines_omitted_assembly`, info; `--hidden-lines-force` keeps them).
  The STEP root count is not used: Inventor and `STEPControl_Writer`
  assemblies arrive as one root.
- **Run deadline**: 0.8 x `timeout_s` from the start of `pipeline.run`. Each
  view gets its share of the time left (minus a 25 % section/sheet reserve,
  at least 2 s, at most `hlr_time_budget_s`); a native HLR step between the
  share and `hlr_time_budget_s` keeps the visible lines and drops the hidden
  ones (`hidden_lines_dropped`). With less than 10 % of the budget left the
  section is skipped (`section_skipped_budget`); the DXF audit needs 10 s.
- **Mesh once**: the poly HLR meshes the drawn shape once per run (a
  `mesh_cache` in the pipeline context); `BRepTools.Triangulation_s` stays
  False when one face cannot be meshed, which made every view re-mesh.

RC buggy front suspension (15 MB Inventor STEP, 211 instances of 91 solids,
10 665 faces, shared machine, other jobs running): before 112 s in-process
(load 43.7, views 30.4, section 26.7, write 4.1), after 97 s sandboxed with
the default 180 s timeout (load 23.1, views 30.4, section 27.1, write 6.9;
peak RSS 695 MB). In-process A/B on the same load: `load_step` 61.0 -> 20.5 s,
one view with hidden lines 12.4 s -> without 5.7 s -> with the reused mesh
5.4 s (HLR 1.97 -> 0.88 s), section 39.7 -> 29.4 s.

## Measured numbers

HLR (front view, this machine):

| body | exact `HLRBRep_Algo` | `HLRBRep_PolyAlgo` (+ meshing) |
|---|---|---|
| box + hole / plate with 6 cylinders | 0.4–190 ms per view | – |
| faceted icosphere 1 280 tri | 0.09 s | 0.02 s (+0.12 s) |
| 5 120 tri | 0.66 s | 0.09 s (+0.60 s) |
| 20 480 tri | 9.4 s | 1.2 s (+2.6 s) |
| 81 920 tri | > 240 s (killed) | 16 s (+11 s mesh, +5 s build) |

Exact HLR on an encoded faceted shell loses a third of the silhouette into
the tangent compound, so faceted bodies always take the polyhedral path.

Mesh route (`raster_span`, N = 1024 / 2048): 12 triangles 48 / 204 ms, 1 092
triangles 66 / 280 ms, 82k 404 / 837 ms, 328k 649 / 1 446 ms. Visibility
against jittered ray casts: 0.00–0.14 % wrong samples on CAD meshes with the
3×3 window (2–54 % with a single pixel), 1.3 % on an oblique torus, 2.9 % on
a noisy blob (fold silhouettes within a pixel of the outline).

B-rep analysis on the test plate: census < 1 ms, cylinder grouping 0.1 ms,
section 7 ms, half-space cut 6 ms, HLR of the cut body 2.4 ms, `measure_shape`
15 ms; reading a 5 120-face faceted STEP (12 MB) 2.6 s.

## Known gaps

- Hole callouts only on true B-rep input (STEP or NURBS conversion); the
  faceted and mesh routes draw holes but do not dimension them.
- One section, one plane orientation; no auxiliary or detail views; no
  tolerances, threads, chamfers, centre lines on holes.
- The classifier was tuned on synthetic meshes; thin sheet metal and
  fillet-heavy tessellations may fall under the threshold (then the mesh route
  still produces a drawing, only without the STEP deliverable).
- POSIX sandbox branch unverified; ASME inch drawings not implemented (text
  heights and sheet sizes are ISO; the arrangement is switchable).
- Third-party STEP exporters were not available for testing; split periodic
  faces were simulated with `ShapeUpgrade_ShapeDivideClosed`.
