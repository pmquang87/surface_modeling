# Lessons from stl2step

A read-only review of BlinkingSun/stl2step (MIT, C++ on OpenCASCADE) against this
repo's STL to STEP pipeline. Every number was measured, by stl2step's findings or
by reviewers probing our OCP 7.9.3 install.

## A closed, watertight shell can enclose the wrong volume, and a gate over an empty list says yes

stl2step shipped a closed, watertight 25-face shell its file census called valid:
44536.7 mm3 against a mesh volume of 15868.9, plus 181 percent, from UV-trimmed
cylinder sheets running past their wires and inverted on hole walls. A proximity
test between STEP tessellation and mesh cannot see an inverted or doubled patch;
volume can. And `all([])` is `True`, so our own audit used to pass a STEP that
re-read as zero shells and zero solids. Tolerance splits by mode: a faceted
rebuild must match the mesh to machine precision, a smooth NURBS fit legitimately
differs by 0.08 to 0.10 percent here.

**Where it lands:** `step_audit.verdict` requires `shells > 0` and `solids >= 1`,
always reports `volume_delta_pct` but gates it only under `--volume-tol`; the
faceted exporter gates at 1e-4 percent (1e-6 relative): merging coplanar
triangles into one face shifts OCCT's integrated volume by up to that much on
identical geometry (5.4e-7 measured on the 63,676-triangle foxcore void body).

*stl2step: FINDINGS-VOLUMEFIX.md:28-43; src/stl2step.cpp:649-674*

## Pass Eps to BRepGProp, or the integrator invents volume

The default overload already cost us a +1.9 percent phantom error on a valid
1200 mm part; stl2step adds the mechanism and the magnitude. Without `Eps`,
`VolumeProperties` uses a fixed Gauss scheme that under-integrates a curved face
bounded by a many-span polyline: 62410.25 mm3 against a true 62445.89, a 0.0598
percent artifact of the integrator, not the geometry. `Precision::Confusion()`
switches OCCT to adaptive 2D Gauss and leaves 0.00272 percent, the real chord
defect. Many-span wires on curved faces are what a patch-fitted B-Rep produces.

**Where it lands:** `src/io/step_audit.py::brep_volume`.

*stl2step: src/stl2step.cpp:191-204; CHANGELOG.md:45*

## The witness must not share the defendant's code

stl2step keeps two censuses of the written file and lets neither link the engine's
translation units; its header says a witness sharing the defendant's code is
worthless. One build's in-memory BRepCheck said invalid while the written file's
census said valid. Our audit re-reads with the same OCP that wrote the file, so a
defect OCCT emits and heals on read stays invisible. Know the limit too: a text
census sees face, edge and surface-type counts only, so it does **not** predict
the faces SolidWorks silently drops.

**Where it lands:** `src/io/step_census.py` (vendored), read back as
`census_faces`.

*stl2step: tests/gates/census/step_census.cpp:7-9; tests/tools/step_census.py:1-11; FINDINGS-W3.md:98*

## Census the mesh first; sewing is a repair of last resort

stl2step sews only where its mesh-side census found open edges, non-manifold edges
or winding conflicts; clean components go straight to `MakeShell`, and the census
is plain arrays built before any OCCT object exists. Only the sewn side is
superlinear: measured here on OCP 7.9.3, 0.16 s direct against 1.9 s sewn at
5,120 triangles and 0.60 s against 12.4 s at 20,480, so 4x the triangles cost the
direct path 3.8x and the sewn one 6.5x. Both gave one closed shell, so speed is
only half of it: a weld at a tolerance also decides which surfaces touch, and can
close a thin web. OCCT 7.9 already
ships `BRepBuilderAPI_MakeShapeOnMesh`, so what transfers is the surrounding
discipline: re-stamp `Closed()` from a measured `BRep_Tool::IsClosed`, orient by
`BRepClass3d_SolidClassifier::PerformInfinitePoint` and not by a repair's side
effects, run `UnifySameDomain` once (`ConcatBSplines = False`, 0.001 degree) and
check the face count after, and raise vertex and edge tolerances to the measured
planar deviation (`fitPlanarTolerances`) rather than run ShapeFix. That last one
proved necessary on a rotated float32 1200 mm box.

**Where it lands:** `src/io/faceted_step.py::export_step_faceted`;
`src/nurbs/converter.py:164` still sews unconditionally at 1e-4.

*stl2step: src/stl2step.cpp:219-241, :497-504, :734-760, :899-903*

## A repair that moves shared vertices opens the shell it was called to close

`ShapeFix_Wire` closes gaps by moving vertices shared with neighbouring faces, so
the repair propagates: 93 of 1601 shared vertices moved by up to 12.45 mm,
tolerances raised to 13.71 mm, at a sewing tolerance of 0.0018. stl2step disables
almost all of `ShapeFix_Face`, keeps only `FixOrientationMode`, and refuses
`ShapeFix_Shape` on analytic runs. Both alternatives were measured and banned, the
in-place one taking one body's free edges from 49 to 82. A vertex moved past the
sliver threshold is how the faces SolidWorks drops get manufactured.

**Where it lands:** `src/nurbs/converter.py:218`, `ShapeFix_Solid` with default
modes.

*stl2step: src/refit_build.cpp:10691-10709; tests/diag/body11/KNOWN-GAP.md:71-76*

## OCCT's STEP statics do not exist until the controller has run

stl2step sets `write.step.schema`, `write.step.unit` and `write.step.product.name`
together, because `Interface_Static` is process-global mutable state. OCCT
registers those statics in `STEPControl_Controller::Init()`, which runs from the
reader or writer constructor; before that, `SetCVal_s` returns `False` and does
nothing. Our old schema line was therefore a silent no-op that worked only
because it asked for the default. Assert each return: a setter that cannot fail is
a check that cannot fail.

**Where it lands:** `src/io/occt_utils.py::init_step_statics`.

*stl2step: src/stl2step.cpp:1042-1044*

## Give every ratio gate an absolute floor, and never gate on triangle count

Eigenvalues of an area-weighted covariance carry units of area, so a ratio built
from them means nothing below a noise floor: a vertex displaced by the linear
tolerance eps on a body of size L tilts a unit normal by eps/L, so the floor is
A(eps/L)^2. A 1e-6-relative radial jitter, 300x below eps, pushed a two-to-three
band cylinder claim over its 0.05 threshold and shattered it into planes. Use
`max(relative * statistic, absolute floor)`, the floor from something physical.
stl2step shows the wrong way to scope one: much of its recogniser runs only inside
a triangle-count window drawn around specific fixtures, one bound held by a
`static_assert` naming a 15300-triangle file. The healthy counterpart is nearby:
both bands defined once in one header under a written `DO NOT WIDEN` rationale,
and an adjudication that kept six proposals and dropped seven, each with a reason.

**Where it lands:** `src/reverse_engineering/mesh_tools.py` —
`collapse_short_edges` (0.15 of the median edge, line 521) and
`flip_needle_triangles` (0.05, line 609) divide by a statistic that is itself
noise near the file's quantization.

*stl2step: src/refit_grow.cpp:318-343; src/refit_internal.hpp:110-124; src/refit_math.cpp:18-22; FINDINGS-INT.md:19-24, :32-38*

## Near-parallel directions: take the angle from atan2(|cross|, dot)

Two unit directions that are bitwise (anti)parallel still dot to one plus or minus
an ulp, and `sqrt(1 - dot^2)` turns that ulp into a spurious tilt near 1e-8 rad;
over 2000 random placements of one cone the sqrt form demoted 669 exact rims and
reported fake deviations up to 1.55e-7 mm. Mind the scale, though: measured here,
`arccos` at theta = 1e-3 rad is off by 7.8e-15 rad and the noise floor only
dominates below roughly 1.4e-8 rad, so the crease gate below is **not** broken.
Prefer `atan2(norm(cross), dot)` anyway: accurate over the whole range, no clip.

**Where it lands:** `src/core/halfedge_mesh.py:345-347`
(`expand_selection_by_angle`), currently `np.arccos(np.clip(dot, -1, 1))`.

*stl2step: src/refit_cone_math.cpp:235-253; docs/1.3.0-linkage-detectors.md:260-271*

## A measured number belongs to one build, so pin it there

Byte-reproducibility comes from habits at the point of production: collections
sorted by a key ending in an id, float reductions in ascending local id, iterative
steps hard-capped by a named constant (64 Jacobi sweeps, 20 Pratt Newton
iterations), warnings sorted after one part emitted the same 34 strings in six
orders. The environment moves too: one generator produced a 54-triangle STL on
macOS (MD5 `71b455f8`) and a 64-triangle one on Linux (`43541a9b`) from the same
OCCT family, because `BRepMesh` depends on libm and FMA targets; at fixed input
one platform recognised 419 cylinders where another
recognised 583. So commit the tessellated bytes, refuse to overwrite a pinned
fixture, and set any cross-platform ceiling to the **maximum** measured. The same
argument runs against opportunistic JIT: our `@njit(fastmath=True)` kernels
produced control points up to 0.35 mm from the pure-Python path through the
ill-conditioned `lsqr`, and were slower.

**Where it lands:** `src/nurbs/g3_fitter.py`, from which numba has now been
removed and where `tests/test_determinism.py` fails if it comes back; golden
numbers from `tessellated_nodes` must carry their OCP/OCCT build.

*stl2step: src/refit_internal.hpp:28-31; src/refit_math.cpp:36-37; src/stl2step.cpp:1162-1167; FINDINGS-H.md:30, :54-56; tests/corpus/PLATFORM-DIVERGENCE.md:1-24*

## A green suite proves nothing about what git actually contains

A new header was created but never staged, since `git add -u` stages only tracked
files, so every working-tree preflight passed while every clean checkout failed.
The remedy is to clone the repo from itself over `file://`, test there, and refuse
a green marker on a dirty tree. Python is worse than C++ here: a module that
exists only in the working tree imports fine forever.

**Where it lands:** the whole `tests/` tree; this repo has no CI and reaches
`src/` by `sys.path` insertion.

*stl2step: scripts/ci-local-gate.sh:4-8, :187; scripts/ci-windows-preflight.sh:267-272*

## What we deliberately did not port and why

**TrueForm analytic recognition.** It recovers analytic surfaces from a CAD
tessellation. TOSCA meshes are not CAD tessellations, and the design-space bodies
here already arrive as analytic STEP.

**DXF export.** trimesh already emits `ARC` and `CIRCLE` entities.

**Never-clobber the derived output path.** stl2step derives a default output beside
the input; we do not. `output` is a required positional argument in
`src/convert.py`, and the GUI's save dialog prompts on overwrite. Partial-write
cleanup was kept; the guard has nothing to protect.

**Hardened boolean subtract** (fuzzy value from the smallest feature, `GlueShift`,
plain retry). Probed against OCP 7.9.3 it makes correct cuts wrong: a corner notch
sharing three exactly coincident faces already cuts to the exact analytic answer
with plain `BRepAlgoAPI_Cut` at fuzzy 0. On the Foxcore pair the fix was geometric
clearance. This repo carries no boolean code, so neither probe is reproducible
from the tree.

## Provenance

[BlinkingSun/stl2step](https://github.com/BlinkingSun/stl2step), MIT, "Copyright
(c) 2026 stl2step contributors". Reviewed at commit `162631f`, release v1.3.0,
2026-09-05, author Joshua Roberts. Read-only checkout: `C:\tmp_stl2step\stl2step`.

**Vendored** (MIT notice kept in the file header): `src/io/step_census.py`, from
`tests/tools/step_census.py`. **Ported as patterns** (idea and measurements, code
written fresh): the statics order in `occt_utils.py`; the volume `Eps` and
non-empty verdict in `step_audit.py`; the mesh census, shell-versus-sew routing,
measured closure, classifier orientation, unify settings and `fitPlanarTolerances`
in `faceted_step.py`. Each names stl2step in its docstring.
