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
python -m src.drawing part.stl  out/ --stl-unit in          # the STL is in inch (STL carries no unit)
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
| `--section` | `auto` | `auto`: the holes are tried largest first and the first of the front, top and right planes that contains the hole axis wins (a washer seen face on is cut by the top plane through its bore); without a usable hole a thin part (depth < 0.1 x the smaller other extent) is cut by the top plane through the bbox centre, anything else by the front plane through it (flag `section_fallback`); an empty plane is retried through the centroids of the 3 largest bodies (`section.reason` then ends in `_body_centroid_retry`, `section.retry_mm` gives the offset, and the flag detail says so). Large assemblies get no auto section (below). `front`/`top`/`right` choose the plane orientation (always drawn); `off` |
| `--sheet` | `A3` | `A3` (landscape), `A4` (portrait, ISO), `A4L` (landscape, flagged non-ISO), `auto` (A4 or A3, chosen from the uncapped dimension plan; A3 is preferred over capping the position bands on A4 when A3 shows every dimension at the capped scale or larger) |
| `--projection` | `first` | `first` (ISO, top view below the front view) or `third` (ASME) |
| `--orient` | `auto` | `auto`: longest extent → width, shortest → depth, then the viewing side (below); `fixed`: file axes |
| `--up` | `z` | `y` rotates a Y-up file to Z-up first |
| `--stl-to-step` | `auto` | `auto` (faceted when prismatic and ≤ 30k triangles), `faceted`, `nurbs` (smooth reverse engineering, slow), `off` |
| `--stl-unit` | not given | `mm`, `cm`, `m`, `in`: the unit of an STL file; the mesh is scaled to mm (`input.source_unit` = the unit, `unit_factor`). Not given: read as mm, `source_unit` `assumed_mm`, flag `unit_assumed_mm` |
| `--no-hidden` | | omit hidden lines |
| `--hidden-lines-force` | | keep hidden lines on large assemblies (default: omitted above 20 solids or 50 000 raw hidden HLR edges per view, flag `hidden_lines_omitted_assembly`) |
| `--max-hidden-primitives` | `20000` | drop the hidden lines of a B-rep view with more raw hidden primitives (flag `hidden_lines_dropped`); `none` = no cap. Measured: at most 1393 per view on the NIST parts, 185k–193k on the 10 665-face RC buggy |
| `--no-hole-positions` | | diameter callouts only |
| `--title`, `--owner` | file stem, empty | title block fields |
| `--timeout` | 180 s | wall clock of the child (floor 15 s: imports alone cost 4–5 s) |
| `--memory-limit-mb` | 2048 | committed-memory cap of the child |
| `--no-sandbox` | | run in-process |
| `--options-json` | | `DrawingOptions` as JSON; explicit flags override it |
| `--quiet` | | no progress on stderr, except one `INFO src.drawing.stage: stage ...` line per stage boundary (elapsed, time left, RSS); sandboxed, the parent prints only these lines of the child's stderr tail, after the child ends (the full tail stays in `sandbox.stderr_tail`) |

Programmatic use: `from src.drawing.pipeline import run; code, report = run(path, out_dir, DrawingOptions(...))`.

## How a file is routed

```
sniff by content (STEP header | binary STL 84+50n | ASCII STL facet count) -> size and triangle caps
  STEP ─ B-rep faces ───────────────────────────────► B-rep route (exact HLR; poly HLR above 4000 faces)
       └ triangulations only (AP242 TESSELLATED_*) ──► mesh route (route.source step_tessellated_mesh)
  STL  ─► classifier ─ prismatic & ≤ 30k tri ─► faceted STEP (src.io.faceted_step) ─► B-rep route (polyhedral HLR)
          └─ organic | not_solid | too large | conversion failed | HLR over budget ─► mesh route (depth buffer)
```

- **B-rep route.** `STEPControl_Reader` (gated on `ReadFile == RetDone`; a
  transfer after a failed read is a native crash), source unit from
  `FileUnits` and a text sniff, all roots as one compound. Loose curve sets,
  PMI geometry and supplemental shells next to solids are left out of the
  drawing and the bbox (`loose_curves_ignored`, `supplemental_shells_ignored`;
  counts in `geometry.bodies.dropped`). Census: face types; `solids`,
  `shells`, `faces`, `edges` count instances (TShape + location),
  `solids_unique` counts TShapes; `BRepCheck`, closure and free edges
  (`step_audit.measure_shape`), volume, area, exact bbox
  (`BRepBndLib.AddOptimal`). Views: `HLRBRep_Algo`; visible = sharp + outline
  compounds, hidden = their hidden counterparts, tangent (Rg1/RgN) edges
  dropped; lines and circles stay exact, ellipses and splines are discretized;
  hidden segments lying on visible ones are removed, and hidden curves
  (circles, arcs, discretized ellipses) lying on visible geometry within
  0.05 mm on paper are dropped or trimmed (`views[].hidden_curves_dropped` /
  `hidden_curves_trimmed`); a hidden Arc on a hidden Circle of the same centre
  and radius is dropped. Holes: cylindrical faces grouped by axis and radius
  (exporters split periodic faces), concave + full circle = hole; a hole an
  assembly fastener fills is a `filled_hole`. Section: `BRepAlgoAPI_Section`
  loops (closed wires only) + half-space cut body HLR'd, each body cut on its
  own; fasteners cut lengthwise stay unsectioned (`section.fasteners_unsectioned`).
- **Faceted route (STL → STEP).** The mesh becomes an exact planar-face solid
  written to `<stem>.step`; for the drawing the same faces are assembled into
  a shell, edges between near-coplanar facets are coded smooth
  (`EncodeRegularity`, 20° default), and `HLRBRep_PolyAlgo` projects it. No
  circles exist on this route, so there are no hole callouts (flagged), and
  `undimensioned` carries one entry `{kind: features, reason:
  route_without_brep_features, route}`: the drawn holes, pockets and fillets
  are not dimensioned (the same entry on the mesh and tessellated routes).
- **Tessellated STEP route.** A STEP file whose faces are all AP242
  triangulations (`TESSELLATED_SOLID` / `TRIANGULATED_FACE`, no surfaces)
  cannot run HLR or a B-rep census. The triangulations become one mesh (nodes
  in mm, face locations applied, reversed faces re-wound, duplicate nodes
  merged), flag `step_tessellated_only` (warning), and it is drawn on the
  mesh route with `route.source` `step_tessellated_mesh` and
  `route.stl_to_step.skipped_reason` `tessellated_step`. What the mesh leaves
  out is reported like on the B-rep route: tessellated solids and shells
  outside the drawn part (`supplemental_shells_ignored`) and tessellated PMI
  annotation occurrences (`loose_curves_ignored`), counted in
  `geometry.bodies.dropped` (`tessellated_shells_outside`,
  `annotation_occurrences`; NIST FTC 08: 1 and 60).
- **Mesh route.** trimesh load (own guard: trimesh returns an empty mesh for
  garbage, truncated and STEP-text files), census, per view: sharp (> 30°),
  boundary and silhouette edges (adjacency pairs with a near-degenerate face,
  altitude below 1e-5 x the mesh diagonal, are skipped:
  `views[].n_sliver_pairs`), a numpy scanline depth buffer (1024 px on the
  long side), visibility by comparing each edge sample with the 3×3 minimum of
  the buffer (1.5 px tolerance), outline of organic shapes from the
  coverage-mask contour. The chains are welded (ends within 0.3 mm on paper),
  simplified (0.05 mm) and stripped of hairpins (`views[].welded_ends`,
  `hairpins_removed`); on organic meshes open chains under 3 mm and closed
  loops under 1 mm on paper are dropped (`dropped_chains`,
  `closed_loops_dropped`). Sections from `mesh_plane` segments chained into
  loops (an open chain of an open mesh is closed by a straight chord: flag
  `section_gaps_bridged`); the half mesh behind the plane for the section body.
  No hole features on this route (`undimensioned` kind `features`). An open
  mesh with winding conflicts has no winding-repaired volume; its conflicts
  are disclosed by `mesh_winding_inconsistent` (info) and
  `geometry.winding_conflicts` (with `degenerate_faces` and
  `volume_raw_signed_mm3`).

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
verdict can be diagnosed. A mesh without any closed component (a triangle
soup) is `not_solid`: drawn on the mesh route without section and overall
dimensions (flag `not_a_solid`, level `minimal`). Not yet validated on real
scans or fillet-heavy CAD tessellations.

### The viewing side (depth sign)

`--orient auto` fixes the axis permutation from the extents, then decides
which of the two depth sides the front viewer looks at; the other choice is a
180° turn about Z (a rotation, never a mirror). Recorded in
`route.orientation` (`depth_flipped`, `depth_method`, `depth_scores`,
`depth_reason`).

- **B-rep** (`brep_features.choose_depth_sign_brep`, up to 5000 faces). The
  cavity rule (`brep_cavity`) measures the open-cavity volume seen from each
  side: projected face area x recess below the bbox face on that side. It
  decides when the larger cavity holds at least 2 % of the bbox volume and
  exceeds the other by the 1.25 margin: the part is turned so the viewer
  looks into the cavity (pocket floors dominate, shallow counterbores and
  logos do not). **Open-housing rule:** the cavity rule yields to the face
  rule when the deeper side's mean recess is above 0.4 of the bbox depth — the
  interior of an open housing or tray is drawn from its exterior, as the
  NIST CTC 02 reference does (measured mean recess: CTC 02 0.499, face rule,
  kept; FTC 10 0.313, cavity rule, kept). Otherwise the face rule
  (`brep_faces`): turned when the far side has more than 1.25 x the faces
  facing it and at least 2 more.
- **Mesh** (`mesh_load.choose_depth_sign_mesh`). The same cavity rule per
  triangle (`mesh_cavity`), used only when the projected areas facing the two
  sides agree (a closed, consistently wound surface); otherwise the
  feature-edge length facing each side (`mesh_feature_edges`, margin 1.2).

## Sheet conventions

Line group 0.5 (ISO 128-24): wide 0.5 mm (visible edges, section outline,
cutting line ends, section arrows), narrow 0.25 mm (hidden, centre,
dimension, extension, leader, hatch), frame 0.7. Dashes (ISO 128-20, multiples
of the line width): hidden 3 / 0.75, centre 6 / 0.75 / 0.125 / 0.75, cutting
line 12 / 1.5 / 0.25 / 1.5. A hidden element shorter than two periods gets
two dashes and one centred gap that fill its length; a hidden circle or arc
gets at least 4 whole periods (gap at least 0.3 mm), so short hidden lines
never print solid (`writers.dash_adapted` counts them per format). Hidden
chains of 3 or more points are single polylines, so the dash runs on across
their vertices. SVG and PDF use round line joins (no miter spikes at acute
vertices). DXF: the linetypes `HIDDEN_ISO`, `CENTER_ISO` and `CUTTING_ISO`
are built from the same `SheetStyle.dash_*` patterns at `$LTSCALE` 1, and a
short hidden entity carries its own ltscale. Text 3.5 mm (ISO 3098 type B
proportions), 5 mm for section letters and the title. Dimensioning per ISO
129 method 1: closed filled arrowheads (15°), `Ø` before the value,
`4 x Ø6.5` for repeated holes, no units (a "Dimensions in mm" note sits in
the strip; when the unit is doubtful — `unit_assumed_mm` or `unit_suspect`
raised to a warning — it reads "Dimensions in mm (unit assumed, see report)"
for an STL without `--stl-unit`, "(unit doubtful, see report)" otherwise,
wrapped onto two lines and never below 2.5 mm). Sheet frame per ISO 5457
(20 mm filing margin, 10 mm elsewhere, centring marks); ISO 5457 permits A4
only in portrait, so `A4L` is flagged `sheet_not_iso5457`. Title block per
ISO 7200 Fig. 2 (180 × 36, 9 mm rows, columns 32 / 74.2 / 129); scale,
projection symbol and the units note are deliberately outside it (ISO 7200
clause 4) in a 180 × 9 strip. View arrangement per ISO 5456-2: first angle
puts the top view below and the right view to the left of the front view;
third angle mirrors both; projection symbol with the circles to the right of
the cone for first angle, left for third. Section label `A-A` immediately
above the section view (ISO 128-40).

Hole callouts: coaxial bores whose cylinder faces are separated by a gap (two
walls, or one drilled bore interrupted by a slot or cavity) are counted per
run, for example `2 x Ø100`; a gap between faces of different diameter up to
2 x the radius difference is read as a conical step (chamfer, drill point),
so the bore stays one counterbore / stepped feature. Callouts go on circles
drawn visible in their axial view; holes seen only as hidden circles get no
callout (`undimensioned` reason `hidden_only`, flag `dimension_hidden_only`).
A counterbore on the face away from the viewer (its large circle drawn only
hidden while the bore is visible) gets its leader on the visible bore circle
and the text `4 x Ø6 / Ø10 cbore far side` (kept compact: a longer note
took the free band above the view and pushed other callouts onto long leaders)
(`DimensionPlan.stats["far_side_callouts"]`); near-side and far-side
counterbores of one size are separate callouts.
When the position bands at a scale would overflow the sheet, at most 3
position bands per side are kept and the rest dropped (outermost first,
never an overall dimension; reason `layout_space`, flag `layout_dims_dropped`).
After the resolution pass, the warning `layout_text_collision` counts what
is left: overlapping texts, texts crossed by a line (extension, dimension,
leader, shelf) of another dimension, and section-marker items touching a
dimension line (`sheet.text_collisions`: `remaining`, `text_overlaps`,
`text_line_crossings`, `marker_dimension_crossings`); texts the pass moved
give the info `layout_text_moved`. The pass moves a callout out of its
view's reserved rect into free paper rather than leave its text on another
dimension's line, but never into another view's rect. A value text too wide
for its span goes past the end with no extension line of its own or an
outer band between the span and the text, and not into a corner where the
perpendicular side has bands (when both ends are taken, the end with more
room inside the view wins, as before). The section marker's letter sits the
same way at both ends of the cutting line, and an end whose arrow or letter
would touch an extension or dimension line of its view moves past the
outermost dimension line (as an end whose cutting line lies on an extension
line already did); the reserved room follows.

**Section fill.** The fill is decided per body (the solid each loop comes
from), from its mean wall thickness on the sheet `t = 2A/P` over all its loops
(A = outer areas minus hole areas, P = all perimeters; a strip of width w
gives about w whatever its shape). A body with `t` below 2.1 mm is filled
solid (ISO 128-50 blackened thin section), otherwise every region of the body
is hatched: one part, one treatment. A hatched body has ONE hatch spacing
(ISO 128-50: one part, one hatch): the section spacing (below), unless its
narrow regions (2A/P from 1.4 mm up to two section spacings) hold more than
20 % of the body's area; then the smallest `t / 3` of those regions (at
least 0.7 mm), so a thin ring alone is hatched at about a third of its
width. Guards: a thin body with more than 10 %
of its area deeper than 2.1 mm inside the material (a finned heat sink) is
hatched, not filled; a region of a hatched body under 1.4 mm (2A/P), or with
fewer than 2 hatch lines at the body's spacing, is filled (`fill` `mixed`).
Bodies alternate 45° and
135° hatching. The report gives `section.hatch_spacing_mm` (the drawn
spacing in paper mm, rounded to 4 decimals: the single value, or the smallest
when the bodies differ; null when nothing is hatched, e.g. an all-solid thin
section; before QA round 3 it held the nominal spacing even then),
`hatch_spacings_mm` (per body, null for a filled body, same rounding),
`section.fill` (`hatch`, `solid`, `mixed`),
`filled_regions`, `filled_area_fraction`, `bodies`, `thick_core_bodies`,
`body_thickness_paper_mm` (the per-body `t` above, measured on the sheet in
paper mm, not the part's wall), `body_thickness_model_mm` (the same divided
by the sheet scale: model mm) and `thin_walled_by_design`. The last is True
when every filled body is thinner than 2.1 mm in model mm, so it would be
filled at 1:1 too (sheet metal). More than half of the cut area filled solid
gives the info flag `section_blackened`, except for a thin-walled part: the
flag means the scale blackened walls that 1:1 would hatch (NIST FTC 07: a
4 mm wall at 1:5 is 0.8 mm on paper).

Conventional (DIN 406 teaching practice, not ISO): first dimension line 10 mm
from the outline, 7 mm pitch, 1 mm extension gap, 2 mm overshoot, hatch
spacing 2 / 3 / 4 mm by section size (one spacing per body, `t / 3` when its
narrow regions dominate, see Section fill), 20 mm between views. The report never
cites ISO for these.

## Quality report

Schema `surface_modeling.drawing.report/1`; the full field list is in spec
section 3.8. The parts a consumer should read first:

- `quality.level`: `full` (B-rep route with hidden lines, section and
  dimensions), `reduced` (mesh route, or a B-rep route that lost hidden lines,
  the section or used the polyhedral HLR), `minimal` (views only, or
  `not_a_solid`), `none` (the run failed; no drawing exists).
- `quality.flags[]`: `{code, severity, detail}`. A `warning` flag sets exit
  code 2 when the drawing was written; `info` flags are disclosures. Some info
  codes are raised to warning by the run (column 3). The table below is
  generated from `report.FLAGS` / `ESCALATABLE_FLAGS` / `FLAG_MEANINGS`
  (`report.flag_table_markdown()`); a test keeps it in step with the code.
  `approval_blank` is listed as a warning in the original spec but is `info`:
  the approval field of an auto-generated drawing is always blank, so a
  warning would make every run exit 2.

<!-- flag-table:begin (generated by src.drawing.report.flag_table_markdown) -->
| Code | Severity | Can be raised to warning | Meaning |
|---|---|---|---|
| `mesh_not_watertight` | warning |  | the mesh has boundary or non-manifold edges; volume and section may be approximate |
| `multiple_bodies` | warning |  | more than one solid (STEP) or mesh component (STL); drawn as one part |
| `brepcheck_invalid` | warning |  | BRepCheck_Analyzer reports the B-rep invalid |
| `open_shell` | warning |  | no solid, or free edges: an open surface model (no section, volume null) |
| `inverted_solid` | warning |  | the solid or mesh is inside out (negative signed volume); the absolute volume is reported, the signed raw sum is in the detail |
| `fallback_to_mesh` | warning |  | the B-rep stage of an STL input failed; drawn on the mesh route |
| `hlr_timeout` | warning |  | a view's native HLR exceeded hlr_time_budget_s; that view is missing |
| `section_empty` | warning |  | the section plane cuts no closed loop; no section view |
| `section_open_wires` | warning |  | open section wires were dropped from the section |
| `views_overflow` | warning |  | the views do not fit the sheet even at the smallest standard scale |
| `dimensions_capped` | warning |  | more hole groups than max_hole_callouts; the rest are listed in undimensioned (reason cap) |
| `unit_not_mm` | warning |  | the STEP length unit is not mm; OCCT converted the geometry on read |
| `sheet_not_iso5457` | warning |  | the sheet is not an ISO 5457 sheet (A4 landscape) |
| `view_failed` | warning |  | a view could not be drawn (error, or HLR returned no edges) |
| `section_failed` | warning |  | the section stage raised an error; no section view |
| `writer_failed` | warning |  | an output format could not be written |
| `dimensions_failed` | warning |  | dimension planning failed; the drawing has no dimensions |
| `faceted_audit_mismatch` | warning |  | the drawn faceted body disagrees with the closed mesh (topology or volume) |
| `dimension_extent_mismatch` | warning |  | an overall dimension differs from the model extent |
| `views_tiny` | warning |  | at the largest standard scale the views fill under 10 % of the frame: check the unit |
| `step_tessellated_only` | warning |  | the STEP holds triangulations only; drawn from the triangles (mesh route) |
| `hidden_lines_dropped` | warning |  | a view lost its hidden lines (time budget or max_hidden_primitives) |
| `not_a_solid` | warning |  | no closed component (triangle soup): no section, no overall dimensions, level minimal |
| `mesh_non_manifold` | warning |  | the mesh has edges with more than two faces |
| `layout_text_collision` | warning |  | after the resolution pass dimension texts still overlap, a text is crossed by another dimension's line, or the section marker touches a dimension line (counts in sheet.text_collisions) |
| `section_skipped_budget` | warning |  | the run deadline was too close (or the section's estimated cost too high): no section view |
| `sandbox_timeout` | warning |  | the child exceeded the wall-clock limit (run failed) |
| `sandbox_memory_exceeded` | warning |  | the child exceeded the memory limit (run failed) |
| `sandbox_cpu_exceeded` | warning |  | the child exceeded the CPU-time backstop (run failed) |
| `sandbox_output_exceeded` | warning |  | the child exceeded the output cap (run failed) |
| `sandbox_killed` | warning |  | the child was killed (run failed) |
| `holes_not_detected_mesh_route` | info |  | faceted or mesh route: no hole features, no diameter callouts |
| `organic_classified` | info |  | the STL classifier found an organic shape (mesh route) |
| `too_many_triangles` | info |  | more triangles than faceted_max_triangles: no faceted STEP |
| `extension_mismatch` | info |  | the content format differs from the file extension |
| `facet_edges_suppressed` | info |  | facet edges below facet_smooth_angle_deg are coded smooth and not drawn |
| `hlr_poly_used` | info |  | more faces than exact_hlr_max_faces: polygonal HLR on a meshed copy |
| `section_nudged` | info |  | the section plane gave no loop or lay in a face; moved by 1 % of the depth |
| `degenerate_faces_removed` | info |  | degenerate triangles were skipped in the faceted STEP |
| `approval_blank` | info |  | the approval field is blank (always, for an auto-generated drawing) |
| `loose_curves_ignored` | info |  | loose edges / vertices (sketch, PMI curve sets) or tessellated PMI annotations are not drawn and not in the bbox |
| `supplemental_shells_ignored` | info |  | free shells / faces or tessellated bodies next to the drawn bodies are not drawn |
| `unit_assumed_mm` | info | yes | an STL carries no unit and was read as mm (warning when the size looks implausible) |
| `unit_suspect` | info | yes | the sizes look like another unit; the alternative reading is in the detail |
| `section_fallback` | info |  | no hole axis fits the section mode: bbox-centre plane (moved through a body centroid when it cuts nothing), thin-part plane, or an open B-rep surface model (no section); a mesh that is not a solid reports not_a_solid instead |
| `interior_silhouettes_omitted` | info |  | organic mesh too large for silhouette candidates: interior silhouettes are not drawn |
| `hidden_lines_omitted_assembly` | info |  | large assembly (more than 20 solids, or 50000 raw hidden edges in a view): no hidden lines |
| `layout_text_moved` | info |  | dimension texts were moved off the section marker or other texts |
| `layout_dims_dropped` | info |  | position dimensions were dropped to keep the scale (undimensioned, reason layout_space) |
| `dimensions_omitted_assembly` | info |  | large assembly: no hole callouts and no position dimensions (undimensioned, reason assembly) |
| `section_skipped_assembly` | info |  | large assembly: no automatic section (--section front, top or right draws one) |
| `section_blackened` | info |  | more than half of the cut area is filled solid (walls thin at this scale but hatched at 1:1; not raised for thin-walled parts) |
| `section_gaps_bridged` | info | yes | open section chains of a mesh were closed by straight chords (warning when a chord exceeds 5 % of the section extent) |
| `winding_repaired` | info |  | the face winding is inconsistent; the volume comes from a winding-repaired copy |
| `dimension_hidden_only` | info |  | hole features seen only as hidden circles in their axial view get no callout (undimensioned) |
| `mesh_winding_inconsistent` | info |  | an open mesh has faces wound against their neighbours (winding conflicts, count in the detail and in geometry.winding_conflicts); no repaired volume exists for an open mesh, the views are not affected |
<!-- flag-table:end -->

- `input`: `detected_format` `step`, `stl_binary`, `stl_ascii`,
  `stl_binary_truncated` (fails: the header declares more triangles than the
  file holds) or `unknown`; `source_unit` `mm`, `inch`, `m`, `cm`, ... for
  STEP (`unit_factor` to mm), the `--stl-unit` value or `assumed_mm` for STL;
  `unit_plausibility`.
- `route`: `pipeline` (`brep`, `mesh`), `source` (`step`,
  `stl_faceted_step`, `stl_nurbs_step`, `stl_mesh`, `step_tessellated_mesh`),
  classifier values (`verdict` `prismatic`, `organic`, `not_solid`, `n/a`),
  the STL→STEP attempt with its audit and `skipped_reason` (`organic`,
  `not_solid`, `too_many_triangles`, `off`, `tessellated_step`), every
  fallback with its reason, the orientation (permutation, flips, depth sign).
- `geometry`: census (B-rep or mesh), exact bbox, volume, area, features,
  `bodies` (drawn and dropped geometry; on the tessellated route
  `tessellated_bodies` counts the drawn tessellated bodies and a shell folded
  into `tessellated_shells_outside` is not listed again as `free_shells`;
  `geometry.solids` is `bodies.solids`); mesh routes add
  `winding_conflicts`, `degenerate_faces`, `volume_raw_signed_mm3`.
- `views[]`: algorithm, visible / hidden counts, seconds, time budget, and the
  clean-up counts of the route (`hidden_curves_dropped`,
  `hidden_curves_trimmed`; `welded_ends`, `hairpins_removed`,
  `closed_loops_dropped`, `dropped_chains`, `n_sliver_pairs`).
- `dimensions[]` (what the sheet shows, after the layout's band cap) and
  `undimensioned[]`: kinds `hole_group`, `boss_group`, `fillet_group`,
  `round_group`, `counterbore_depth`, `step_depth`, `body_extent`,
  `body_position`, `overall`, `linear` (a dropped position dim), `features`
  (faceted, mesh and tessellated routes: nothing was detected, so the drawn
  features carry no dimension; `count` null, `route` the source); reasons
  `cap`, `no_axial_view`, `not_supported`, `hidden_only`,
  `filled_by_fastener`, `assembly`, `layout_space`, `not_a_solid`,
  `route_without_brep_features`.
- `section` (plane, loops, fill, `hatch_spacing_mm` / `hatch_spacings_mm`,
  bridged gaps, fasteners left whole; `reason` with the suffix
  `_body_centroid_retry` and `retry_mm` when an empty plane was moved through
  a body centroid),
  `sheet` (size, scale, `unit_note`, `text_collisions`, `scale_candidates`),
  `outputs`.
- `writers.dxf_audited`: the DXF was re-read and audited by ezdxf (False
  above 100 000 sheet items or with less than 10 s left before the run
  deadline; `writers.dxf_audit_skipped_reason` says which);
  `writers.dash_adapted` per format.
- `timing`: `total_s`, `load_s`, `features_s` (B-rep cylinder feature
  detection), `views_s`, `section_s` and `write_s`, the whole sheet stage,
  split into `dimensions_s` + `layout_s` + `writers_s`. Sandboxed, `total_s`
  is the child's wall time, which also holds the interpreter start and the
  imports (the part of `total_s` in no bucket).
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

A model with more than 20 solid instances (`geometry.solids`) is a large
assembly. The STEP root count is not used: Inventor and `STEPControl_Writer`
assemblies arrive as one root.

- **No hidden lines** (`hidden_lines_omitted_assembly`, info;
  `--hidden-lines-force` keeps them). A single view with more than 50 000 raw
  hidden HLR edges is drawn without them too.
- **No hole callouts and no position dimensions**: coaxial bores of different
  solids 0.1 mm apart gave six position dimensions for one hub. Only the
  overall dimensions are drawn; every hole radius group is listed in
  `undimensioned` with reason `assembly`, and every radius group of holes a
  fastener of another solid fills with reason `filled_by_fastener` (flag
  `dimensions_omitted_assembly`, info; its detail counts both).
- **No automatic section**: it cost 30–37 s on the 211-solid RC buggy and came
  out all black. `--section front|top|right` still draws one (flag
  `section_skipped_assembly`, info, otherwise).
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
- **Run deadline**: 0.8 x `timeout_s` from the start of `pipeline.run`. Each
  view gets its share of the time left (minus a 25 % section/sheet reserve,
  at least 2 s, at most `hlr_time_budget_s`); a native HLR step between the
  share and `hlr_time_budget_s` keeps the visible lines and drops the hidden
  ones (`hidden_lines_dropped`). The section is skipped
  (`section_skipped_budget`) with less than 10 % of the budget left, or when
  its estimated cost (1 x the mean view time) exceeds the time left before
  that reserve; the DXF audit needs 10 s.
- **Mesh once**: the poly HLR meshes the drawn shape once per run (a
  `mesh_cache` in the pipeline context); `BRepTools.Triangulation_s` stays
  False when one face cannot be meshed, which made every view re-mesh.

RC buggy front suspension (15 MB Inventor STEP, 211 instances of 91 solids,
10 665 faces), sandboxed with the default 180 s timeout, shared machine:
QA run 2 (hole dimensions and auto section still on) 134–148 s wall (load
37.3, views 40.5, section 37.0, write 16.0 s in the slowest run), past the
144 s run deadline. With the assembly rules the wall time depends strongly
on the machine load: 71.6 s on a quiet machine (child 71.3 s: load 24.5,
views 29.3, section 0, write 5.0 s; peak RSS 717 MB), 131.0 and 148.9 s at
about 48 % CPU load from other jobs (the second past the 144 s run deadline,
write 19–26 s against a 14.4 s write reserve), and a timeout at 180 s under
heavy load (load 89.3 s, feature detection 45.0 s, killed in the views). The
same heavy load with `--timeout 450` took 390.7 s: load 132.9, features
57.9, views 124.7, dimensions 0.5, layout 1.1, writers 64.0 s, 9.7 s in
no bucket (interpreter start and imports). So the time that sat in no
bucket before was mostly the cylinder feature detection (now `features_s`),
which has no deadline check, and the write stage is the writers themselves
(dimension planning and layout cost about 1.5 s); neither has its own budget.
The default 180 s timeout holds 03 only on a lightly loaded machine; use
`--timeout 300` or more on a shared one. Sheet A3 at 1:2 (was 1:10) with the
three overall dimensions only; `undimensioned` lists the 193 holes in 26
radius groups (reason `assembly`) and the 167 holes the fasteners fill in 18
groups (reason `filled_by_fastener`).

Sandbox priority: the sandboxed child runs at below-normal priority. On the
same heavily loaded machine (other agents' jobs at normal priority) three
sandboxed runs, the tiny 15 cube included, got 0.06–0.39 s CPU in 180 s
and timed out without a progress file; the identical runs with the child at
normal priority finished (15 in 27.8 s). A starved child reads exactly like
a hung pipeline: check `sandbox.cpu_s` against `sandbox.wall_s` first. In-process A/B of round 1
on the same load: `load_step` 61.0 -> 20.5 s, one view with hidden lines
12.4 s -> without 5.7 s -> with the reused mesh 5.4 s (HLR 1.97 -> 0.88 s).

## Measured numbers

QA corpus run 3 (30 real-world files: NIST CTC/FTC PMI test parts, Pro/E, NX,
Inventor, SolidWorks, FreeCAD, KiCad, CadQuery and OCCT exports, trimesh models,
scans and a topology-optimised bracket; sandboxed with `--timeout 300`
(`sandbox.timeout_s` 300 in every report), the memory limit and the other
limits at their defaults, on a shared machine: single re-runs were about 40 %
faster, so these wall times include load from other jobs): 28 drawings, 2
rejected inputs (empty and truncated STL, exit 1 in
0.3 s), exit codes 12 × 0 / 16 × 2 / 2 × 1; median wall time 8.3 s over the 28
drawn files, 101 s for the 15 MB / 211-solid Inventor assembly (hidden lines,
section and cross-part dimensions omitted and flagged; 34 % of the 300 s
timeout and 42 % of its 240 s run deadline, and still under the 144 s run
deadline of the default 180 s timeout, at 70 % of it), 33 s for the
554k-triangle wheel; peak RSS median 326 MB, 713 MB on the assembly, 1193 MB on
the wheel (limit 2048 MB). Every DXF re-read with 0 audit errors and 0 fixes,
every PDF opened by PyMuPDF without repair, every SVG parsed. Rendered examples:
`docs/examples/`.

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
  faceted and mesh routes draw holes but do not dimension them. Large
  assemblies get overall dimensions only.
- Not dimensioned (listed in `undimensioned`): boss outer diameters,
  fillet and edge-round radii, the sizes of the bodies of a multi-body model
  and the offsets between them, holes filled by a fastener, counterbore and
  step depths.
- One section, one plane orientation; no auxiliary or detail views; no
  tolerances, threads, chamfers, centre lines on holes.
- The classifier was tuned on synthetic meshes; thin sheet metal and
  fillet-heavy tessellations may fall under the threshold (then the mesh route
  still produces a drawing, only without the STEP deliverable).
- Mesh route on jittered or organic meshes: near-coincident copies of a rim
  (top rim plus a bottom-rim silhouette a few microns apart) still leave a
  curve in several chains.
- POSIX sandbox branch unverified; ASME inch drawings not implemented (text
  heights and sheet sizes are ISO; the arrangement is switchable).
- Third-party STEP exporters were not available for testing; split periodic
  faces were simulated with `ShapeUpgrade_ShapeDivideClosed`.
