# An automated STEP / STL to 2D drawing service — architecture note

This note describes how the drawing module in this repository (`src/drawing`,
see `docs/drawing_module.md`) fits into a production conversion service, what
is already built here, what a service deployment adds on top, and a rough
effort breakdown. It is written as the technical part of a proposal for a
"upload a STEP or STL, get a basic 2D engineering drawing" feature. Everything
below the container boundary exists in this repository and is tested; the
container, the HTTP surface and the queue are described, not built.

## Architecture in nine bullets

1. **One worker CLI, one contract.** `python -m src.drawing in.(step|stl)
   out/` produces `pdf`, `svg`, `dxf` and `report.json`. Progress goes to
   stderr; stdout carries exactly one line `RESULT {json}`; exit codes are
   `0` clean, `2` drawing written but the quality report has warnings, `1`
   failed. The same contract already drives this repo's STL→STEP converter, so
   the service layer never parses human text.
2. **Content sniffing before any geometry library loads.** STEP is recognised
   by the `ISO-10303-21` header, binary STL by the exact size rule
   `84 + 50·n` (the header may lie and start with `solid`), ASCII STL by a
   facet-line count. Size and triangle caps are enforced here, in the parent,
   so a hostile file never reaches OpenCASCADE.
3. **Two geometry routes, same drawing back end.** STEP goes to the B-rep
   route: OpenCASCADE hidden-line removal (`HLRBRep_Algo`), exact circles and
   lines, tangent edges suppressed, hole detection from cylindrical faces, a
   section through the largest hole. STL goes through a prismatic-vs-organic
   classifier; prismatic meshes become an exact faceted STEP solid (this
   repo's `export_step_faceted`) and are drawn with the polyhedral HLR;
   organic meshes, oversized meshes and any conversion failure fall back to
   the mesh route: numpy depth-buffer visibility, silhouette and feature
   edges, mask-contour outlines, plane sections. Both routes emit the same
   `ProjectedView` / `SectionView` objects, so dimensioning, layout and the
   three writers are shared.
4. **Deterministic, standards-based sheet.** ISO 5457 frame (A3 landscape,
   A4 portrait; A4 landscape only as a flagged non-ISO option), ISO 7200 title
   block (180×36) plus a strip for scale, projection symbol and units, line
   group 0.5 (0.5/0.25 mm), ISO 128 dash patterns, ISO 129 dimensioning
   (closed filled arrowheads, Ø prefix, `4 x Ø6.5` grouping), ISO 5455 scales,
   first- or third-angle arrangement. Every number is in one `SheetStyle`
   table and the report marks which rules are ISO-sourced and which are
   conventional practice.
5. **Three writers, independently checked.** SVG is plain XML; DXF is written
   with ezdxf (R2010, layers, real `DIMENSION` and `HATCH` entities, re-read
   and audited in the tests); PDF is a 150-line dependency-free vector writer
   (base-14 Helvetica, exact text metrics, correct xref) verified with
   PyMuPDF. No matplotlib, no shapely.
6. **A quality report, not just files.** Route taken and why (classifier
   features, fallbacks with reasons), geometry census (validity, closed
   shells, free edges, inverted solids, units), per-view algorithm and edge
   counts, every dimension placed and every one dropped, section statistics,
   sheet and scale, resource usage, and a quality level `full / reduced /
   minimal` with coded flags. A client UI can show "dimensions not available
   for mesh input" instead of a silent gap.
7. **Sandbox inside the process boundary.** The worker re-launches itself as a
   child with a fresh working directory, a sanitized environment, a wall-clock
   timeout, a memory cap (Windows job object + completion port; POSIX rlimits),
   a CPU backstop, output-size caps and a whole-tree kill. Measured facts that
   shaped it: a memory breach raises `MemoryError` in the child rather than
   killing it, OpenBLAS commits ~630 MB at `import numpy` unless thread count
   is pinned, and a failed STEP read followed by a transfer is a native access
   violation that only a subprocess can contain.
8. **What the container adds.** A fixed image (Python, cadquery-ocp, trimesh,
   ezdxf), non-root user, read-only root filesystem, no network, cgroup CPU
   and memory limits that duplicate the in-process ones, a scratch volume per
   job, and a thin adapter that pulls a job from the queue, runs the CLI, and
   uploads the four output files plus the report. The worker already runs
   unchanged under such an adapter because it needs nothing but a Python
   interpreter and a writable temp dir.
9. **Testing discipline.** Unit tests for the 2D algorithms and writers,
   OpenCASCADE tests on synthetic parts (plate with holes and boss, filleted
   box, split periodic faces, inch files, two-root files, garbage files),
   mesh tests with analytic truth (depth-buffer visibility against ray casts),
   sandbox tests (timeout with a grandchild, memory breach, stdout flood,
   planted module in the work dir), and end-to-end CLI runs whose PDFs are
   rendered and inspected.

## Rough hour breakdown

| Work package | Hours | Status in this repo |
|---|---|---|
| Geometry spike: OCCT HLR conventions, section API, cylinder detection, trimesh quirks, sandbox facts | 6 | done (probe reports) |
| B-rep route: STEP reading, census, HLR views, primitives, sections | 8 | done |
| STL route: classifier, faceted STEP, polyhedral HLR, mesh depth-buffer fallback, mesh sections | 10 | done |
| Dimensioning and sheet layout (ISO tables, scale choice, arrangement, title block, markers) | 8 | done |
| Writers SVG / DXF / PDF with independent verification | 6 | done |
| Sandboxed worker (job object, env, caps, tree kill, resource report) | 6 | done |
| Quality report, tests, visual QA on rendered PDFs | 8 | done |
| Container image, queue adapter, API contract, deployment docs, handover | 8 | not started (service side) |
| **Total** | **60** | |

The first seven rows exist here and can be demonstrated; the last row is what
a service integration adds.

## Screening question 1: a CAD geometry project and the problems hit

The project behind this repository converts topology-optimisation STL meshes
into STEP solids that SolidWorks feature recognition can edit, plus the exact
faceted path, a shell/thicken operation and now the drawing module. Problems
that cost real time, and what fixed them:

- **SolidWorks silently dropped faces** from STEP files that OpenCASCADE,
  FreeCAD and our own audit called valid (sliver edges around 0.15 mm, patch
  interiors spanning more than ~1.3 quad diagonals). Fixed by driving
  SolidWorks over COM to count imported faces and bisecting face subsets to
  localise the drops, then preventing them at the cage level (collapsing short
  decimated edges, flipping needle triangles, forcing convex quads) and
  adding a control-net fairing term to the fit. The release gate is the
  SolidWorks face count, not the OCC validity flag.
- **An ill-conditioned least-squares fit** (condition number 1e4–1e5)
  solved with an iterative solver stopped early and produced results that
  changed with rigid offsets; the apparent "continuity weight dependence" was
  an artefact of the early stop. Fixed with a direct sparse solve and two
  refinement steps; measured, then documented honestly that the fitter does
  not enforce tangent continuity.
- **Decimation debris:** the quadric decimator emits non-manifold edges and
  micro components from watertight input; a repair that re-derived its
  thresholds per round ate half the mesh. Fixed by freezing thresholds on the
  input mesh and making every keep-N heuristic traversal-direction aware.
- **Signed distance sign ambiguity** inside hollow and self-overlapping
  bodies: fixed by signing whole flood-fill components with the generalized
  winding number, trusting it only near integers, sampled across the
  component.
- **OpenCASCADE facts that bite:** `Interface_Static.SetCVal` is a silent
  no-op before the STEP controller is initialised; `Bnd_Box` over-reports
  B-spline bodies by the control-hull margin (+10 mm on a r=10 cylinder), so
  overall dimensions must come from `BRepBndLib.AddOptimal`; the default
  volume integrator under-integrates many-span curved faces; a STEP round
  trip heals inverted solids so orientation must be checked in memory;
  surface normals from `BRepLProp` ignore face orientation, so every hole
  reads as a boss until `TopAbs_REVERSED` is honoured; calling
  `TransferRoots` after a failed `ReadFile` is a native crash.
- **Drawing-side traps found while building this module:** ezdxf has no
  `HIDDEN` linetype after `setup=True` and silently substitutes `CONTINUOUS`
  (caught by auditing the re-read file); a single-pixel depth test calls
  outline-coincident edges hidden (fixed with a 3×3 minimum window);
  OpenBLAS commits ~630 MB on import on a 20-thread machine, which kills a
  memory-capped child before any Python runs.

## Screening question 2: STL files that cannot become usable B-rep STEP

Decide, do not pretend. Three things make an STL unusable as a B-rep: it is
organic (no planar or analytic regions to recover), it is too large for an
exact faceted solid within the time budget, or it is damaged (holes,
non-manifold edges, inverted components). The module therefore:

1. **Classifies first.** A measured feature — the share of area in coplanar
   facets larger than 1 % of the surface — separates CAD-like tessellations
   (≥ 0.20 on every test part) from organic meshes (0.00), so the decision is
   explainable and appears in the report with its value and threshold.
2. **Converts only when the result is honest geometry.** Prismatic meshes
   become an exact faceted B-rep solid (one planar face per triangle,
   coplanar faces merged, closed by construction, audited by re-reading the
   file). Smooth NURBS reverse engineering is available behind a flag for
   organic parts that need an editable STEP, with its measured accuracy
   (mean 0.07 mm on a 1.2 m part) stated, but it is slow and its patch
   boundaries would clutter a drawing, so it is not the default.
3. **Falls back to the mesh itself** for organic, oversized or failed cases:
   the drawing is produced from the triangles directly (depth-buffer hidden
   lines, mask-contour outlines, plane sections with hatching). Hole callouts
   are not possible on this route and the report says so (`quality.level:
   reduced`, flag `holes_not_detected_mesh_route`) instead of inventing
   circles from polygons.
4. **Caps the attempt.** The faceted conversion is bounded by a triangle count
   and the whole job by a wall-clock and memory limit in a sandboxed child, so
   a pathological scan cannot stall the queue; a fallback is recorded with its
   reason rather than retried blindly.
5. **Repairs only what can be measured.** Non-watertight input is flagged, not
   patched silently; a repair that moves shared vertices can open the shell it
   was meant to close (measured on this project: a wire fix moved 93 vertices
   by up to 12 mm).

## Portfolio pointers

- This repository: `src/convert.py` (STL→STEP CLI with the audit and the
  `RESULT` contract), `src/io/faceted_step.py`, `src/io/step_audit.py`,
  `src/drawing/` (this module), `docs/lessons_from_stl2step.md` (a review of a
  comparable C++/OCCT converter with measured findings), the test suite
  (defect-injection tests with preconditions).
- Measured numbers worth quoting live in `docs/drawing_module.md`.
