# Surface Modeling - AI Agent Memory

## Development State
- **Core Architecture**: The `HalfEdgeMesh` topology system is stable and robust.
- **Reverse Engineering**: `QuadWrapper` and `ShrinkWrapper` algorithms have been implemented. A major bug regarding concave quads rendering as holes was recently fixed by enforcing convexity checks during the `_tri_to_quad` phase.
- **Math**: B-Spline and NURBS geometry generation is supported up to G3 continuity.
- **Testing**: A new deep testing suite has been initiated in `tests/deep_tests/`. 
  - The Property-Based Testing suite (via `hypothesis`) is complete.
  - Computational Geometry and Core Fuzzing test suites are currently in progress by subagents.

## Deployment Details
- **GitHub Repository**: Upload all code to `master` branch of `https://github.com/pmquang87/surface_modeling`

## User Preferences
- **Performance**: High speed algorithms are preferred (e.g. fast 1k-faces decimation).
- **Topology Integrity**: Perfect watertightness (no boundary edges on solid parts) is strictly required. Always verify with local HalfEdge inspection.
- **Simplifier**: Should be an opt-in (default off) option while saving or exporting.
