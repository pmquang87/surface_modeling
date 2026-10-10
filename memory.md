# Surface Modeling - AI Agent Memory

## Development State
- **Core Architecture**: The `HalfEdgeMesh` topology system is stable and robust.
- **Reverse Engineering**: `QuadWrapper` and `ShrinkWrapper` algorithms have been implemented. A major bug regarding concave quads rendering as holes was recently fixed by enforcing convexity checks during the `_tri_to_quad` phase.
- **Math**: NURBS conversion fits one degree-5 Bezier patch per cage quad. Patches share their boundary curves (G0); `--continuity G1`/`G2`/`G3` all add the same third-derivative smoothing rows, and tangent/curvature continuity is NOT enforced (measured 2026-10: hard tangent rows cost 24-37 % fidelity on the real part). Reference fits add a control-net fairing term (weight 0.01).
- **Testing**: A new deep testing suite has been initiated in `tests/deep_tests/`. 
  - The Property-Based Testing suite (via `hypothesis`) is complete.
  - Computational Geometry and Core Fuzzing test suites are currently in progress by subagents.

## Deployment Details
- **GitHub Repository**: Upload all code to `master` branch of `https://github.com/pmquang87/surface_modeling`

## User Preferences
- **Performance**: High speed algorithms are preferred (e.g. fast 1k-faces decimation).
- **Topology Integrity**: Perfect watertightness (no boundary edges on solid parts) is strictly required. Always verify with local HalfEdge inspection.
- **Simplifier**: Should be an opt-in (default off) option while saving or exporting.
