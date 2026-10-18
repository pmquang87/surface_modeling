# Proposal: automated STEP & STL → 2D engineering drawing service

Job: "3D (STEP & STL) to 2D mechanical engineering drawing module" (Upwork, fixed price, posted 2026-10-17).
Draft for Minh Quang Pham. Fill in the price before sending; everything else is ready.

---

## Cover letter (paste into the proposal field)

Hello,

I am a mechanical engineer (FEM and CAD automation, Dresden) and I build Python tools on OpenCASCADE and trimesh. Before writing this I built a working version of the module you describe and ran it on 30 real STEP and STL files from the wild: the NIST MBE test parts (CTC-01/02, FTC-06/07/08/10/11), exports from Pro/E, NX, Inventor, SolidWorks, FreeCAD and KiCad, scanned and topology-optimised STLs, and a few deliberately broken files. The code, the tests and the drawings are public in my repository, so you can judge the work instead of the words.

What it does today: `python -m src.drawing part.step out/` returns a PDF, an SVG, a DXF and a JSON quality report. The drawing has first- or third-angle orthographic views with hidden lines, an isometric view, overall and hole dimensions (`4 x Ø6.5`), one hatched section A-A with its cutting-plane marker, an ISO 5457 sheet with an ISO 7200 title block and projection symbol. STEP files go through OpenCASCADE hidden-line removal with exact circles and lines. STL files are classified: prismatic meshes become an exact faceted STEP solid and are drawn from that; organic, oversized or damaged meshes fall back to a trimesh depth-buffer projection with silhouette outlines and plane sections. Every job runs in a sandboxed child process with a wall-clock timeout, a memory cap, output caps and a whole-tree kill; the report says which route was taken and why, what was dimensioned and what was not, and every quality flag with its severity.

What I learned on the 30 files is in the proposal below, because it is the part a container cannot fix for you: construction curves and PMI geometry that a STEP carries next to the solid, inch files that declare one unit and mean another, hidden-line cleanup that goes quadratic on a 15 MB assembly, a memory cap that does not kill a Python child but raises MemoryError, and a 630 MB OpenBLAS allocation at `import numpy` that kills any capped worker before your code runs.

The module is built as a CLI below the container boundary on purpose. Your worker image, queue adapter and API are the remaining work, and I have listed them with hours. I would start with a short call to agree on the drawing conventions your users expect (ISO or ASME, sheet size, which dimensions matter most), then deliver in two milestones.

Best regards,
Minh Quang Pham

Repository: https://github.com/pmquang87/python-surfacing (branch `feature/drawing-module`, folder `src/drawing`; docs in `docs/drawing_module.md`).
Example drawings: see the attached PNGs (NIST CTC-01 plate from STEP, the same SolidWorks bracket from STEP and from STL, a topology-optimised bracket from a 300k-triangle STL).

---

## Proposed architecture (9 bullets)

1. **Worker CLI with a machine contract.** One command per job; stdout carries exactly one line `RESULT {json}`, progress goes to stderr, exit codes 0 / 2 (drawing written, warnings) / 1 (failed). The service layer never parses human text.
2. **Content sniffing and caps before any CAD library loads**: STEP by the `ISO-10303-21` header, binary STL by the exact `84 + 50·n` size rule (the header may say `solid`), ASCII STL by facet count; size and triangle caps enforced in the parent.
3. **Two geometry routes, one drawing back end.** STEP → OpenCASCADE B-rep route (HLR with exact circles and lines, tangent edges suppressed, cylindrical-face hole detection, section by boolean cut). STL → classifier → faceted STEP solid + polyhedral HLR, or mesh route (numpy depth buffer, silhouettes, mask-contour outline, plane sections). Both routes emit the same view/section/feature objects, so dimensioning, layout and writers are shared.
4. **Standards-based sheet in one table**: ISO 5457 frames, ISO 7200 title block, ISO 128 line groups and dash patterns, ISO 129 dimensioning, ISO 5455 scales, ISO 5456-2 arrangements and projection symbol; ASME variant is a table change, not a rewrite.
5. **Three writers, independently verified**: plain-XML SVG; DXF via ezdxf with real DIMENSION and HATCH entities (re-read and audited in tests); a 150-line dependency-free vector PDF writer (base-14 Helvetica, verified with PyMuPDF).
6. **A quality report, not just files**: route and classifier values, fallbacks with reasons, geometry census (validity, closed shells, free edges, inverted solids, units), per-view algorithm and counts, every dimension placed and dropped, section statistics, sheet and scale, resource usage, and a quality level `full / reduced / minimal` with coded flags your UI can show.
7. **Sandbox inside the process boundary**: fresh work dir, sanitized environment, wall-clock timeout, memory cap (Windows job object + completion port; POSIX rlimits), CPU backstop, output caps, whole-tree kill, peak memory in the report; a progress file so a killed job still reports its last stage.
8. **Container and service layer (the remaining work)**: fixed image with cadquery-ocp, trimesh, ezdxf; non-root, read-only root FS, no network; cgroup limits duplicating the in-process ones; a queue adapter that runs the CLI and uploads the four files plus the report; a small HTTP API for upload and status.
9. **Test discipline**: defect-injection unit tests with preconditions, OpenCASCADE fixtures (split periodic faces, inch files, two-root files, garbage files), mesh fixtures with ray-cast truth, sandbox tests (grandchild kill, memory breach, stdout flood, planted module), and a 30-file real-world QA corpus with rendered-PDF inspection.

## Rough hour breakdown

| Work package | Hours | Status |
|---|---|---|
| Geometry spike (OCCT HLR conventions, sections, hole detection, trimesh quirks, sandbox facts) | 6 | done, measured |
| B-rep route: STEP reading, census, HLR views, primitives, sections | 8 | done |
| STL route: classifier, faceted STEP, polyhedral HLR, mesh depth-buffer fallback, sections | 10 | done |
| Dimensioning and sheet layout (ISO tables, scale, arrangement, title block, markers) | 8 | done |
| Writers SVG / DXF / PDF with independent verification | 6 | done |
| Sandboxed worker | 6 | done |
| Quality report, tests, real-world QA corpus and fixes | 12 | done (round 1 of QA fixed; see numbers below) |
| Your conventions (ISO/ASME, sheet, dimension priorities), container image, queue adapter, API, deployment docs, handover | 10–14 | to do with you |
| **Total** | **≈ 66–70** | about 56 already invested |

Milestones I propose: (1) the module running in your container on your sample files with the report wired into your UI; (2) the conventions and dimension rules tuned to your users' parts, plus the API and documentation.

## Screening question 1 — a CAD geometry project and the problems I hit

The project behind the repository converts topology-optimisation STL meshes into STEP solids that SolidWorks feature recognition can edit, and now adds the drawing module. Problems that cost real time, and how they were solved:

- **SolidWorks silently dropped faces** from STEP files that OpenCASCADE, FreeCAD and my own audit called valid (sliver edges ≈ 0.15 mm, patch interiors spanning > 1.3 quad diagonals). I drove SolidWorks over COM to count imported faces, bisected face subsets to localise the drops, then prevented them at the mesh-cage level and with a fairing term in the fit. The release gate became the SolidWorks face count, not the OCC validity flag.
- **An ill-conditioned least-squares surface fit** (condition number 1e4–1e5) solved iteratively stopped early and gave results that moved with rigid offsets. A direct sparse solve with refinement fixed it; the honest conclusion (no true tangent continuity) went into the docs.
- **Decimation debris**: the quadric decimator emits non-manifold edges and micro components from watertight input, and a repair that re-derived its thresholds per round ate half the mesh. Thresholds are now frozen on the input mesh and every keep-N rule is traversal-direction aware.
- **Signed-distance sign ambiguity** inside hollow and overlapping bodies: solved by signing whole flood-fill components with the generalized winding number, trusted only near integer values.
- **OpenCASCADE facts that bite** (all measured): `Interface_Static.SetCVal` is a silent no-op before the STEP controller is initialised; `Bnd_Box` over-reports B-spline bodies by the control-hull margin (+10 mm on a r=10 cylinder), so dimensions must come from `BRepBndLib.AddOptimal`; the default volume integrator under-integrates many-span curved faces; a STEP round trip heals inverted solids, so orientation must be checked in memory; surface normals from `BRepLProp` ignore face orientation, so every hole reads as a boss until `TopAbs_REVERSED` is honoured; calling `TransferRoots` after a failed `ReadFile` is a native crash, which only a subprocess can contain.
- **Found on the 30-file corpus for this job**: STEP files carry sketch curves, construction geometry and PMI shells next to the solid (NIST CTC-01's overall size read 1170 × 650 instead of 800 × 450 until loose edges were filtered); a Pro/E assembly declares inch with millimetre-sized numbers; hidden-line cleanup written as pairwise comparisons went quadratic on 192k primitives from a 15 MB Inventor assembly (215 s) while the OCCT HLR itself took 2.6 s; ezdxf has no `HIDDEN` linetype after setup and silently substitutes `CONTINUOUS`; a Windows memory cap raises `MemoryError` in the child instead of killing it; OpenBLAS commits ≈ 630 MB at `import numpy` on a 20-thread machine, so `OPENBLAS_NUM_THREADS=1` is mandatory in any capped worker.

## Screening question 2 — STL files that cannot become usable B-rep STEP

Decide, do not pretend. An STL is unusable as a B-rep when it is organic (nothing analytic to recover), too large for an exact faceted solid within the budget, or damaged (open edges, non-manifold edges, inverted winding, triangle soup). The module:

1. **Classifies first** with a measured feature: the share of area in coplanar facets larger than 1 % of the surface. CAD-like tessellations scored ≥ 0.20 on every test part, organic meshes 0.00; the value and threshold go into the report so a wrong verdict is explainable.
2. **Converts only into honest geometry**: prismatic meshes become an exact faceted B-rep solid (planar faces, coplanar merge, closed by construction, audited by re-reading the file). Smooth NURBS reverse engineering exists behind a flag for organic parts that need an editable STEP (measured mean deviation 0.07 mm on a 1.2 m part), but it is slow and would clutter a drawing with patch boundaries, so it is not the default.
3. **Falls back to the mesh itself** for organic, oversized or failed cases: depth-buffer hidden lines, mask-contour outlines, plane sections with hatching. Hole callouts are impossible on that route and the report says so (`quality.level: reduced`, flag `holes_not_detected_mesh_route`) instead of inventing circles from polygons.
4. **Caps the attempt**: triangle count for the conversion, wall-clock and memory for the whole job, in a sandboxed child; a fallback is recorded with its reason, never retried blindly.
5. **Repairs only what it can measure**: non-watertight, non-manifold or inverted input is flagged with counts; a triangle soup is reported as "not a solid" and drawn without dimensions; a repair that moves shared vertices can open the shell it was meant to close (measured on my project: a wire fix moved 93 vertices by up to 12 mm), so such repairs are not applied silently.

## Portfolio (CAD / geometry)

- `src/convert.py`: STL → STEP CLI (smooth NURBS reverse engineering and exact faceted path) with a read-back audit and the `RESULT` contract; validated against SolidWorks imports.
- `src/io/faceted_step.py`, `src/io/step_audit.py`, `src/io/step_census.py`: faceted B-rep builder, independent STEP audit (closure, free edges, signed volume, bbox, deviation, OpenCASCADE-free Part 21 census).
- `src/drawing/`: this module (≈ 9 500 lines, ≈ 400 tests across ten suites) with `docs/drawing_module.md` (usage, routing, report schema, measured numbers, known gaps).
- `docs/lessons_from_stl2step.md`: a measured review of a comparable C++/OpenCASCADE converter.
- Background: FEM simulation (OpenRadioss, LS-DYNA), CNC and machine-tool engineering; Python tooling for solver automation and post-processing.

## Numbers from the real-world QA corpus (30 files, final run after three QA/fix rounds)

| | |
|---|---|
| Files | 30: 16 STEP (Pro/E, NX, Inventor, ST-Developer, NIST AP203/AP242, SolidWorks, FreeCAD, KiCad, CadQuery, OCCT), 14 STL (binary, ASCII, two-body, soup, empty, truncated, 554k and 300k triangles) |
| Drawings produced | 28; the 2 rejections are the empty STL ("empty STL (0 facets)") and the truncated binary STL ("header declares 3476 triangles, file holds 3475.8"), both exit 1 with a `RESULT` line |
| Exit codes | 12 clean (0), 16 with warning flags (2: inch units, multiple bodies, non-watertight or inverted meshes, unit suspicion, assembly drawn without hidden lines and without a section), 2 failed (1) |
| Routes | 15 STEP B-rep, 1 tessellated-only STEP routed to the mesh path, 5 STL → faceted STEP → B-rep, 7 STL mesh route |
| Wall time per file (sandboxed, imports included, quiet machine) | median 5.9 s; 63 s for the 15 MB / 211-solid Inventor assembly (killed at 300 s before the performance work); 26 s for the 554k-triangle wheel |
| Peak memory | median 327 MB; 1.2 GB on the 554k-triangle wheel; 716 MB on the assembly |
| Output checks | all 28 DXF files re-read with 0 audit errors and 0 fixes; all PDFs open in PyMuPDF without repair; all SVGs parse with the sheet viewBox |
| NIST reference check | CTC-01 overall 800 × 450 × 150 and callouts 4 × Ø35, 4 × Ø25, 2 × Ø20 match the NIST drawing; CTC-02 2 × Ø100 / 2 × Ø52 and FTC-07 8 × Ø6.35 (= 8 × Ø.250) match; inch files converted and flagged |

Known limitations, all disclosed in the report: no hole callouts on the faceted and mesh routes; one section per drawing; bosses, radii and inter-body positions are not dimensioned (listed under `undimensioned`); hidden lines are omitted on assemblies with more than 20 solids unless forced; STL units are assumed mm unless `--stl-unit` is given (a plausibility check flags inch-sized and metre-sized meshes); the POSIX sandbox branch is written but unverified on this Windows machine.
