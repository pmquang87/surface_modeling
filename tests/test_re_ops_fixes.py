"""Regression tests for reverse-engineering / mesh-operation bug fixes.

Each test class pins down one reported defect that was reproduced with a probe
before it was fixed:

A quad_wrap  target_face_count was a decimation budget, not a quad count
B quad_wrap  frozen_face_ids / feature_angle silently ignored
C quad_wrap  curvature + cross-field computed then discarded
D mesh_tools decimate_mesh(frozen_vertices=...) dropped frozen vertices and
             collapsed antipodal (non-adjacent) vertex pairs
E mesh_tools fill_holes only closed <=4-edge holes and dropped the result flag
F mesh_tools compute_mesh_quality faked non_manifold_edges, had no angle keys
G mesh_tools quad cages were silently triangulated
H mesh_tools tolerance -> power-of-ten digits rounded the wrong way
I shrink_wrap per-vertex hit masks (O(V^2)) + one closest_point call per miss
J shrink_wrap subdivision_levels stored but never applied
K shell_thicken voxel size mixed coordinate axes
L shell_thicken assumed signed_distance is positive OUTSIDE (it is not)
M shell_thicken thicken_surface ignored direction and doubled the wall
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
import trimesh

from src.core.halfedge_mesh import HalfEdgeMesh
from src.operations import shell_thicken
from src.operations.shell_thicken import _compute_sdf, shell_solid, thicken_surface
from src.reverse_engineering import shrink_wrap as shrink_wrap_mod
from src.reverse_engineering.mesh_tools import (
    compute_mesh_quality,
    decimate_mesh,
    fill_holes,
    remove_duplicate_vertices,
)
from src.reverse_engineering.quad_wrap import QUADS_PER_DECIMATED_TRIANGLE, QuadWrapper
from src.reverse_engineering.shrink_wrap import ShrinkWrapper


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _icosphere(subdivisions=3):
    return trimesh.creation.icosphere(subdivisions=subdivisions)


def _sphere_with_hole(rings=1):
    """Icosphere with the faces around vertex 0 removed (bigger `rings` ->
    longer boundary loop)."""
    tm = _icosphere(3)
    faces = tm.faces
    touched = np.any(faces == 0, axis=1)
    for _ in range(rings - 1):
        inner = set(faces[np.where(touched)[0]].ravel().tolist())
        touched = np.array([bool(set(f.tolist()) & inner) for f in faces])
    holed = trimesh.Trimesh(vertices=tm.vertices.copy(), faces=faces[~touched],
                            process=False)
    holed.remove_unreferenced_vertices()
    return holed


def _flat_sheet(n=11, size=10.0):
    """Open square sheet in the z=0 plane, face normals pointing +z."""
    xs = np.linspace(-size / 2, size / 2, n)
    X, Y = np.meshgrid(xs, xs, indexing='ij')
    verts = np.column_stack([X.ravel(), Y.ravel(), np.zeros(n * n)])
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            a = i * n + j
            b = (i + 1) * n + j
            c = (i + 1) * n + j + 1
            d = i * n + j + 1
            faces.append([a, b, c])
            faces.append([a, c, d])
    return HalfEdgeMesh.from_arrays(verts, faces)


def _quad_sheet():
    """Two quads sharing an edge (a quad cage, not a triangle mesh)."""
    verts = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 0, 0], [2, 1, 0],
    ], dtype=float)
    return HalfEdgeMesh.from_arrays(verts, [[0, 1, 2, 3], [1, 4, 5, 2]])


def _non_manifold_edges(tm):
    edges = np.sort(tm.faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return int((counts > 2).sum())


# --------------------------------------------------------------------------
# A. quad_wrap: target_face_count means final quads
# --------------------------------------------------------------------------

class TestQuadWrapTargetCount:
    @pytest.mark.parametrize("target", [60, 200, 500])
    def test_final_quad_count_tracks_target(self, target):
        dense = _icosphere(4)  # 5120 triangles
        w = QuadWrapper(target_face_count=target)
        param_V, param_F, field = w._miq_parametrization(dense)
        _, quad_F = w._extract_pure_quads(param_V, param_F, field)
        # before the fix this was ~4.5x the request
        assert 0.75 * target <= len(quad_F) <= 1.3 * target, (
            f"target {target} produced {len(quad_F)} quads "
            f"({len(quad_F) / target:.2f}x)"
        )

    def test_decimation_budget_has_a_floor_of_four(self):
        dense = _icosphere(2)
        w = QuadWrapper(target_face_count=1)
        _, param_F, _ = w._miq_parametrization(dense)
        assert len(param_F) >= 4

    def test_budget_is_target_over_the_documented_ratio(self):
        w = QuadWrapper(target_face_count=1000)
        expected = int(round(1000 / QUADS_PER_DECIMATED_TRIANGLE))
        dense = _icosphere(4)
        _, param_F, _ = w._miq_parametrization(dense)
        # decimation lands close to but not exactly on the request
        assert abs(len(param_F) - expected) <= 0.25 * expected

    def test_full_wrap_returns_about_the_requested_number_of_quads(self):
        dense = HalfEdgeMesh.from_trimesh(_icosphere(3))
        cage = QuadWrapper(target_face_count=120, smoothing_weight=0.1).wrap(dense)
        assert 0.7 * 120 <= len(cage.faces) <= 1.4 * 120
        assert all(len(cage.get_face_vertices(f)) == 4 for f in cage.faces)


# --------------------------------------------------------------------------
# B. quad_wrap: honest about the unimplemented parameters
# --------------------------------------------------------------------------

class TestQuadWrapUnimplementedParams:
    def test_frozen_face_ids_warns(self):
        with pytest.warns(UserWarning, match="frozen_face_ids"):
            QuadWrapper(target_face_count=50, frozen_face_ids=[0, 1])

    def test_feature_angle_warns(self):
        with pytest.warns(UserWarning, match="feature_angle"):
            QuadWrapper(target_face_count=50, feature_angle=45.0)

    def test_defaults_do_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            QuadWrapper(target_face_count=50, smoothing_weight=0.3)

    def test_wrap_level_frozen_faces_warn(self):
        mesh = HalfEdgeMesh.from_trimesh(_icosphere(1))
        w = QuadWrapper(target_face_count=40, smoothing_weight=0.1)
        with pytest.warns(UserWarning, match="frozen_face_ids"):
            w.wrap(mesh, frozen_face_ids=[0])

    def test_warning_is_emitted_once_per_instance(self):
        w = QuadWrapper(target_face_count=40)
        with pytest.warns(UserWarning, match="frozen_face_ids"):
            w._warn_unimplemented(frozen=True)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            w._warn_unimplemented(frozen=True)  # second time: silent

    def test_docstring_admits_the_gap(self):
        doc = QuadWrapper.__doc__
        assert "NOT IMPLEMENTED" in doc
        assert "feature_angle" in doc and "frozen_face_ids" in doc


# --------------------------------------------------------------------------
# C. quad_wrap: the discarded curvature / cross-field work is skipped
# --------------------------------------------------------------------------

class TestQuadWrapSkipsCrossField:
    def test_wrap_never_calls_the_curvature_stage(self, monkeypatch):
        calls = []

        def boom(self, *a, **kw):
            calls.append(1)
            raise AssertionError("curvature stage must not run")

        monkeypatch.setattr(QuadWrapper, "_compute_curvatures", boom)
        monkeypatch.setattr(QuadWrapper, "_propagate_cross_field", boom)

        mesh = HalfEdgeMesh.from_trimesh(_icosphere(2))
        cage = QuadWrapper(target_face_count=60, smoothing_weight=0.1).wrap(mesh)
        assert calls == []
        assert len(cage.faces) > 0

    def test_miq_parametrization_accepts_no_cross_field(self):
        w = QuadWrapper(target_face_count=60)
        V, F, field = w._miq_parametrization(_icosphere(2))
        assert len(F) > 0 and len(V) > 0
        assert field.shape[0] == len(V)


# --------------------------------------------------------------------------
# D. mesh_tools: frozen-vertex decimation
# --------------------------------------------------------------------------

class TestDecimateWithFrozenVertices:
    @staticmethod
    def _run(n_frozen=50, target=400):
        tm = _icosphere(3)  # 642 verts, 1280 faces
        mesh = HalfEdgeMesh.from_trimesh(tm)
        rng = np.random.default_rng(0)
        frozen = sorted(rng.choice(len(mesh.vertices), n_frozen, replace=False).tolist())
        frozen_pos = np.array([mesh.vertices[i].position for i in frozen])
        out = decimate_mesh(mesh, target_faces=target, frozen_vertices=frozen)
        return tm, mesh, frozen_pos, out

    def test_every_frozen_vertex_survives_exactly(self):
        _, _, frozen_pos, out = self._run()
        out_pos = np.array([v.position for v in out.vertices])
        # exact match: a frozen vertex must never be moved either
        kept = 0
        for p in frozen_pos:
            if np.isclose(np.abs(out_pos - p).sum(axis=1), 0.0, atol=1e-12).any():
                kept += 1
        assert kept == len(frozen_pos), f"lost {len(frozen_pos) - kept} frozen vertices"

    def test_it_actually_decimates(self):
        _, mesh, _, out = self._run()
        assert len(out.faces) < len(mesh.faces)

    def test_no_antipodal_collapses(self):
        """The old alias-chain merged vertices that were never adjacent; on a
        unit sphere that showed up as edges spanning the whole diameter."""
        tm, _, _, out = self._run()
        out_tm = out.to_trimesh()
        longest = float(out_tm.edges_unique_length.max())
        assert longest < 8 * float(tm.edges_unique_length.max()), (
            f"edge of length {longest:.3f} on a unit sphere -> non-adjacent collapse"
        )

    def test_result_stays_manifold_and_closed(self):
        _, _, _, out = self._run()
        out_tm = out.to_trimesh()
        assert _non_manifold_edges(out_tm) == 0
        assert out_tm.is_watertight
        assert out_tm.is_winding_consistent

    def test_volume_is_roughly_preserved(self):
        tm, _, _, out = self._run()
        assert out.to_trimesh().volume == pytest.approx(tm.volume, rel=0.15)

    def test_freezing_everything_is_a_no_op(self):
        mesh = HalfEdgeMesh.from_trimesh(_icosphere(2))
        allv = list(range(len(mesh.vertices)))
        out = decimate_mesh(mesh, target_faces=10, frozen_vertices=allv)
        assert len(out.faces) == len(mesh.faces)


# --------------------------------------------------------------------------
# E. mesh_tools: fill_holes
# --------------------------------------------------------------------------

class TestFillHoles:
    @pytest.mark.parametrize("rings", [1, 2, 3])
    def test_holes_larger_than_four_edges_are_closed(self, rings):
        holed = _sphere_with_hole(rings)
        assert not holed.is_watertight
        out = fill_holes(HalfEdgeMesh.from_trimesh(holed), max_hole_edges=40)
        out_tm = out.to_trimesh()
        assert out_tm.is_watertight, f"{rings}-ring hole left open"
        assert out_tm.is_winding_consistent, "patch faces wound the wrong way"

    def test_filled_patch_keeps_the_volume(self):
        holed = _sphere_with_hole(2)
        out = fill_holes(HalfEdgeMesh.from_trimesh(holed), max_hole_edges=40)
        assert out.to_trimesh().volume == pytest.approx(_icosphere(3).volume, rel=0.02)

    def test_max_hole_edges_is_honoured_and_reported(self):
        holed = _sphere_with_hole(2)
        mesh = HalfEdgeMesh.from_trimesh(holed)
        with pytest.warns(UserWarning, match="left open"):
            out = fill_holes(mesh, max_hole_edges=4)
        assert len(out.faces) == len(holed.faces), "hole was filled despite the limit"
        assert not out.to_trimesh().is_watertight

    def test_closed_mesh_is_unchanged(self):
        tm = _icosphere(2)
        out = fill_holes(HalfEdgeMesh.from_trimesh(tm))
        assert len(out.faces) == len(tm.faces)

    def test_flat_open_sheet_boundary_is_closed(self):
        sheet = _flat_sheet(n=5, size=4.0)
        out = fill_holes(sheet, max_hole_edges=40)
        assert len(out.faces) > len(sheet.faces)

    def test_does_not_recreate_an_existing_chord(self):
        """Square sheet already split along the A-C diagonal: closing its
        boundary must use B-D, otherwise A-C gets a third face."""
        verts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
        sheet = HalfEdgeMesh.from_arrays(verts, [[0, 1, 2], [0, 2, 3]])
        out = fill_holes(sheet, max_hole_edges=10)
        out_tm = out.to_trimesh()
        assert len(out.faces) == 4
        assert _non_manifold_edges(out_tm) == 0, "an existing chord got a third face"
        assert out_tm.is_watertight

    def test_ear_clip_reports_failure_instead_of_guessing(self):
        from src.reverse_engineering.mesh_tools import _ear_clip
        square = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
        assert _ear_clip(square) is not None
        # both diagonals blocked -> no legal triangulation
        assert _ear_clip(square, forbidden={(0, 2), (1, 3)}) is None

    def test_large_hole_falls_back_to_a_centroid_fan(self, monkeypatch):
        """When ear clipping cannot find a legal ear the hole is still closed
        (with an extra centre vertex) rather than left open."""
        from src.reverse_engineering import mesh_tools
        monkeypatch.setattr(mesh_tools, "_ear_clip", lambda *a, **kw: None)
        holed = _sphere_with_hole(2)
        out = mesh_tools.fill_holes(HalfEdgeMesh.from_trimesh(holed), max_hole_edges=40)
        out_tm = out.to_trimesh()
        assert out_tm.is_watertight
        assert len(out.vertices) == len(holed.vertices) + 1


# --------------------------------------------------------------------------
# F. mesh_tools: real quality metrics
# --------------------------------------------------------------------------

class TestMeshQuality:
    def test_angle_keys_exist_and_are_degrees(self):
        stats = compute_mesh_quality(HalfEdgeMesh.from_trimesh(_icosphere(2)))
        for key in ('min_angle', 'max_angle', 'avg_angle'):
            assert key in stats, f"documented key {key} missing"
        assert 0 < stats['min_angle'] <= stats['avg_angle'] <= stats['max_angle'] < 180
        # equilateral-ish triangles on an icosphere
        assert stats['avg_angle'] == pytest.approx(60.0, abs=1.0)

    def test_non_manifold_edges_are_counted(self):
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, -1, 0]],
                         dtype=float)
        mesh = HalfEdgeMesh.from_arrays(verts, [[0, 1, 2], [0, 1, 3], [0, 1, 4]])
        assert compute_mesh_quality(mesh)['non_manifold_edges'] == 1

    def test_clean_mesh_reports_zero_non_manifold(self):
        stats = compute_mesh_quality(HalfEdgeMesh.from_trimesh(_icosphere(2)))
        assert stats['non_manifold_edges'] == 0
        assert stats['boundary_edges'] == 0
        assert stats['watertight'] is True

    def test_open_mesh_reports_boundary_edges(self):
        stats = compute_mesh_quality(HalfEdgeMesh.from_trimesh(_sphere_with_hole(1)))
        assert stats['boundary_edges'] > 0
        assert stats['non_manifold_edges'] == 0


# --------------------------------------------------------------------------
# G. mesh_tools: quad cages come back triangulated -- say so
# --------------------------------------------------------------------------

class TestQuadCageTriangulationWarning:
    def test_decimate_mesh_warns_on_quads(self):
        with pytest.warns(UserWarning, match="non-triangle"):
            out = decimate_mesh(_quad_sheet(), target_faces=4)
        assert all(len(out.get_face_vertices(f)) == 3 for f in out.faces)

    def test_remove_duplicate_vertices_warns_on_quads(self):
        with pytest.warns(UserWarning, match="non-triangle"):
            remove_duplicate_vertices(_quad_sheet())

    def test_triangle_input_does_not_warn(self):
        mesh = HalfEdgeMesh.from_trimesh(_icosphere(2))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            decimate_mesh(mesh, target_faces=40)
            remove_duplicate_vertices(mesh)


# --------------------------------------------------------------------------
# H. mesh_tools: conservative tolerance -> digits rounding
# --------------------------------------------------------------------------

class TestMergeTolerance:
    @staticmethod
    def _pair_mesh(x0, x1):
        verts = np.array([
            [x0, 0, 0], [1, 0, 0], [0, 1, 0],
            [x1, 0, 0], [1, 0, 0], [0, 1, 0],
        ], dtype=float)
        return HalfEdgeMesh.from_arrays(verts, [[0, 1, 2], [3, 4, 5]])

    def test_vertices_further_apart_than_tolerance_are_not_merged(self):
        # 7e-5 apart, but both land in the same 1e-4 grid cell: the old
        # round() mapping produced digits=4 for tol 5e-5 and fused them.
        mesh = self._pair_mesh(0.00006, 0.00013)
        out = remove_duplicate_vertices(mesh, tolerance=5e-5)
        assert len(out.vertices) == 4, "merged a pair further apart than tolerance"

    def test_a_coarser_tolerance_still_merges(self):
        mesh = self._pair_mesh(0.00006, 0.00013)
        out = remove_duplicate_vertices(mesh, tolerance=1e-4)
        assert len(out.vertices) == 3

    def test_exact_duplicates_always_merge(self):
        mesh = self._pair_mesh(0.0, 0.0)
        out = remove_duplicate_vertices(mesh, tolerance=1e-6)
        assert len(out.vertices) == 3

    def test_effective_grid_never_exceeds_the_requested_tolerance(self):
        for tol in (5e-5, 1e-6, 2e-3, 3e-2, 1e-3):
            digits = max(0, int(np.ceil(-np.log10(max(tol, 1e-12)))))
            assert 10.0 ** -digits <= tol * (1 + 1e-12)


# --------------------------------------------------------------------------
# I. shrink_wrap: vectorized ray-cast projection
# --------------------------------------------------------------------------

def _reference_ray_cast(wrapper, vertices, normals, ref_tm):
    """The pre-fix per-vertex implementation, kept as the numeric oracle."""
    intersector = trimesh.ray.ray_triangle.RayMeshIntersector(ref_tm)
    origins = np.vstack((vertices, vertices))
    directions = np.vstack((normals, -normals))
    locations, index_ray, _ = intersector.intersects_location(
        ray_origins=origins, ray_directions=directions)
    out = []
    for i, v in enumerate(vertices):
        mask = (index_ray == i) | (index_ray == i + len(vertices))
        hits = locations[mask]
        if len(hits):
            out.append(hits[int(np.argmin(np.linalg.norm(hits - v, axis=1)))])
        else:
            out.append(wrapper._project_to_surface(np.array([v]), ref_tm)[0])
    return np.array(out)


class TestRayCastProjection:
    def test_matches_the_per_vertex_reference_exactly(self):
        ref = _icosphere(3)
        cage = _icosphere(2)
        verts = np.array(cage.vertices) * 1.35
        norms = np.array(cage.vertex_normals)
        w = ShrinkWrapper(iterations=1, subdivision_levels=0, smooth_weight=0.0)
        got = w._ray_cast_projection(verts, norms, ref)
        want = _reference_ray_cast(w, verts, norms, ref)
        assert np.array_equal(got, want)

    def test_projects_onto_the_reference_surface(self):
        ref = _icosphere(3)
        cage = _icosphere(2)
        verts = np.array(cage.vertices) * 1.35
        norms = np.array(cage.vertex_normals)
        w = ShrinkWrapper(iterations=1, subdivision_levels=0, smooth_weight=0.0)
        got = w._ray_cast_projection(verts, norms, ref)
        radii = np.linalg.norm(got, axis=1)
        assert np.abs(radii - 1.0).max() < 0.01

    def test_rays_that_miss_fall_back_to_the_closest_point(self):
        """A cage far off to the side: every ray misses, so the whole batch
        goes through the single batched closest_point call."""
        ref = _icosphere(2)
        verts = np.array([[10.0, 0.0, 0.0], [0.0, 12.0, 0.0], [0.0, 0.0, -9.0]])
        norms = np.tile([0.0, 0.0, 1.0], (3, 1))  # parallel to the surface / missing
        w = ShrinkWrapper(iterations=1, subdivision_levels=0, smooth_weight=0.0)
        got = w._ray_cast_projection(verts, norms, ref)
        want = _reference_ray_cast(w, verts, norms, ref)
        assert np.allclose(got, want)
        assert np.abs(np.linalg.norm(got, axis=1) - 1.0).max() < 0.05

    def test_all_misses_go_through_a_single_batched_query(self, monkeypatch):
        """The old code issued one closest_point call per missed vertex, which
        dominated the runtime on cages that mostly miss."""
        ref = _icosphere(2)
        verts = np.array([[10.0, 0.0, z] for z in np.linspace(-3, 3, 25)])
        norms = np.tile([0.0, 0.0, 1.0], (len(verts), 1))
        w = ShrinkWrapper(iterations=1, subdivision_levels=0, smooth_weight=0.0)

        calls = []
        real = ShrinkWrapper._project_to_surface

        def counting(self, vertices, ref_tm):
            calls.append(len(vertices))
            return real(self, vertices, ref_tm)

        monkeypatch.setattr(ShrinkWrapper, "_project_to_surface", counting)
        w._ray_cast_projection(verts, norms, ref)
        assert len(calls) <= 1, f"{len(calls)} closest_point calls for {len(verts)} misses"
        assert calls and calls[0] == len(verts)

    def test_empty_hit_set_is_handled(self):
        ref = _icosphere(1)
        w = ShrinkWrapper(iterations=1, subdivision_levels=0, smooth_weight=0.0)
        verts = np.array([[5.0, 5.0, 5.0]])
        norms = np.array([[1.0, 0.0, 0.0]])
        got = w._ray_cast_projection(verts, norms, ref)
        assert got.shape == (1, 3)
        assert np.isfinite(got).all()


# --------------------------------------------------------------------------
# J. shrink_wrap: subdivision_levels
# --------------------------------------------------------------------------

class TestShrinkWrapSubdivision:
    def test_default_is_zero(self):
        assert ShrinkWrapper().subdivision_levels == 0

    def test_zero_leaves_the_cage_density_alone(self):
        ref = HalfEdgeMesh.from_trimesh(_icosphere(3))
        cage = HalfEdgeMesh.from_trimesh(_icosphere(1))
        out = ShrinkWrapper(iterations=1, subdivision_levels=0,
                            smooth_weight=0.0).wrap(cage, ref)
        assert len(out.faces) == len(cage.faces)

    def test_one_level_splits_every_face(self):
        """Catmull-Clark splits an n-gon into n quads, so a triangle cage
        triples and a quad cage quadruples."""
        ref = HalfEdgeMesh.from_trimesh(_icosphere(3))
        tri_cage = HalfEdgeMesh.from_trimesh(_icosphere(1))
        out = ShrinkWrapper(iterations=1, subdivision_levels=1,
                            smooth_weight=0.0).wrap(tri_cage, ref)
        assert len(out.faces) == 3 * len(tri_cage.faces)

        quad_cage = _quad_sheet()
        out_q = ShrinkWrapper(iterations=1, subdivision_levels=1,
                              smooth_weight=0.0).wrap(quad_cage, ref)
        assert len(out_q.faces) == 4 * len(quad_cage.faces)

    def test_two_levels_compound(self):
        ref = HalfEdgeMesh.from_trimesh(_icosphere(3))
        cage = HalfEdgeMesh.from_trimesh(_icosphere(1))
        out = ShrinkWrapper(iterations=1, subdivision_levels=2,
                            smooth_weight=0.0).wrap(cage, ref)
        assert len(out.faces) == 3 * 4 * len(cage.faces)

    def test_subdivided_cage_still_lands_on_the_reference(self):
        ref = HalfEdgeMesh.from_trimesh(_icosphere(3))
        cage = HalfEdgeMesh.from_trimesh(_icosphere(1))
        out = ShrinkWrapper(iterations=2, subdivision_levels=1,
                            smooth_weight=0.2).wrap(cage, ref)
        radii = np.linalg.norm(np.array([v.position for v in out.vertices]), axis=1)
        assert np.abs(radii - 1.0).max() < 0.05

    def test_quad_wrap_relax_passes_zero_explicitly(self, monkeypatch):
        """quad_wrap must not start subdividing its cage because the default
        changed -- the pipeline behaviour has to stay exactly the same."""
        seen = {}
        real_init = ShrinkWrapper.__init__

        def spy(self, *args, **kwargs):
            seen.update(kwargs)
            return real_init(self, *args, **kwargs)

        monkeypatch.setattr(shrink_wrap_mod.ShrinkWrapper, "__init__", spy)
        mesh = HalfEdgeMesh.from_trimesh(_icosphere(2))
        QuadWrapper(target_face_count=60, smoothing_weight=0.1).wrap(mesh)
        assert seen.get("subdivision_levels") == 0


# --------------------------------------------------------------------------
# K. shell_thicken: per-axis voxel grid
# --------------------------------------------------------------------------

class TestSdfGrid:
    @staticmethod
    def _box(translate=(0.0, 0.0, 0.0), extents=(20.0, 20.0, 20.0)):
        b = trimesh.creation.box(extents=extents)
        b.apply_translation(np.asarray(translate, dtype=float))
        return b

    def test_grid_does_not_collapse_for_a_far_from_origin_mesh(self):
        """The real STLs sit at x ~= 1135..1209; the old cross-axis voxel size
        turned the grid into 2x2x2."""
        b = self._box(translate=(1150.0, 0.0, 0.0))
        grid, origin, voxel = _compute_sdf(np.array(b.vertices), np.array(b.faces), 32)
        assert min(grid.shape) >= 30, f"grid collapsed to {grid.shape}"
        assert voxel == pytest.approx(24.0 / 32, rel=0.05)

    def test_translated_and_centred_grids_agree(self):
        a = self._box()
        b = self._box(translate=(1150.0, -700.0, 300.0))
        ga, _, va = _compute_sdf(np.array(a.vertices), np.array(a.faces), 24)
        gb, ob, vb = _compute_sdf(np.array(b.vertices), np.array(b.faces), 24)
        assert ga.shape == gb.shape
        assert va == pytest.approx(vb)
        assert np.allclose(ga, gb, atol=1e-6)
        assert ob == pytest.approx([1150.0 - 12.0, -700.0 - 12.0, 300.0 - 12.0])

    def test_non_cubic_box_keeps_every_axis_resolved(self):
        b = self._box(extents=(40.0, 4.0, 12.0))
        grid, _, _ = _compute_sdf(np.array(b.vertices), np.array(b.faces), 32)
        assert grid.shape[0] >= 30
        assert min(grid.shape) >= 4


# --------------------------------------------------------------------------
# L. shell_thicken: SDF sign convention
# --------------------------------------------------------------------------

class TestSdfSign:
    def test_trimesh_signed_distance_is_positive_inside(self):
        """The premise of the fix -- pinned so a trimesh upgrade can't flip it
        silently underneath us."""
        cube = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
        sd = trimesh.proximity.signed_distance(cube, np.array([[0., 0., 0.],
                                                               [5., 0., 0.]]))
        assert sd[0] > 0 and sd[1] < 0

    def test_module_sdf_is_positive_outside(self):
        cube = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
        grid, origin, voxel = _compute_sdf(np.array(cube.vertices),
                                           np.array(cube.faces), 16)
        centre_idx = tuple(s // 2 for s in grid.shape)
        assert grid[centre_idx] < 0, "centre of the cube must be negative"
        assert grid[0, 0, 0] > 0, "grid corner is outside -> positive"

    @staticmethod
    def _extents_of(result):
        tm = result.to_trimesh()
        assert len(tm.faces) > 0
        return tm.vertices.max(axis=0) - tm.vertices.min(axis=0)

    def test_inward_shrinks_the_body(self):
        box = trimesh.creation.box(extents=(20.0, 20.0, 20.0))
        mesh = HalfEdgeMesh.from_arrays(np.array(box.vertices), box.faces.tolist())
        out = shell_solid(mesh, thickness=2.0, direction='inward',
                          resolution=48, smooth_iterations=0)
        assert self._extents_of(out) == pytest.approx([16.0] * 3, abs=0.6)

    def test_outward_grows_the_body(self):
        box = trimesh.creation.box(extents=(20.0, 20.0, 20.0))
        mesh = HalfEdgeMesh.from_arrays(np.array(box.vertices), box.faces.tolist())
        out = shell_solid(mesh, thickness=2.0, direction='outward',
                          resolution=48, smooth_iterations=0)
        assert self._extents_of(out) == pytest.approx([24.0] * 3, abs=0.6)

    def test_far_from_origin_body_still_shells_correctly(self):
        box = trimesh.creation.box(extents=(20.0, 20.0, 20.0))
        box.apply_translation([1150.0, 17.0, 78.0])
        mesh = HalfEdgeMesh.from_arrays(np.array(box.vertices), box.faces.tolist())
        out = shell_solid(mesh, thickness=2.0, direction='inward',
                          resolution=48, smooth_iterations=0)
        tm = out.to_trimesh()
        assert self._extents_of(out) == pytest.approx([16.0] * 3, abs=0.8)
        centre = (tm.vertices.max(axis=0) + tm.vertices.min(axis=0)) / 2
        assert centre == pytest.approx([1150.0, 17.0, 78.0], abs=0.8)


# --------------------------------------------------------------------------
# M. shell_thicken: thicken_surface direction and wall thickness
# --------------------------------------------------------------------------

class TestThickenSurface:
    RES = 56
    T = 1.0

    @staticmethod
    def _z_span(result, footprint=None, offset=(0.0, 0.0)):
        """z extent of the result, optionally restricted to vertices well
        inside the sheet's footprint.

        The rim of an open sheet is a genuine sharp corner of the offset
        region, and marching cubes rounds it over about half a voxel, so the
        overall bounding box is only accurate to that. Measured over the
        interior the wall position is exact.
        """
        tm = result.to_trimesh()
        assert len(tm.faces) > 0
        v = tm.vertices
        if footprint is not None:
            keep = ((np.abs(v[:, 0] - offset[0]) < footprint)
                    & (np.abs(v[:, 1] - offset[1]) < footprint))
            v = v[keep]
            assert len(v) > 0
        return float(v[:, 2].min()), float(v[:, 2].max())

    def test_both_is_symmetric_and_one_thickness_thick(self):
        out = thicken_surface(_flat_sheet(), thickness=self.T, direction='both',
                              resolution=self.RES, smooth_iterations=0)
        lo, hi = self._z_span(out, footprint=4.0)
        assert hi - lo == pytest.approx(self.T, abs=0.05)
        assert lo == pytest.approx(-self.T / 2, abs=0.03)
        assert hi == pytest.approx(self.T / 2, abs=0.03)

    def test_outward_puts_all_material_on_the_normal_side(self):
        # _flat_sheet has +z face normals
        out = thicken_surface(_flat_sheet(), thickness=self.T, direction='outward',
                              resolution=self.RES, smooth_iterations=0)
        lo, hi = self._z_span(out, footprint=4.0)
        assert hi - lo == pytest.approx(self.T, abs=0.05), "wall is not `thickness` thick"
        assert lo == pytest.approx(0.0, abs=0.03), "material leaked below the surface"
        assert hi == pytest.approx(self.T, abs=0.03)

    def test_inward_puts_all_material_on_the_far_side(self):
        out = thicken_surface(_flat_sheet(), thickness=self.T, direction='inward',
                              resolution=self.RES, smooth_iterations=0)
        lo, hi = self._z_span(out, footprint=4.0)
        assert hi - lo == pytest.approx(self.T, abs=0.05)
        assert lo == pytest.approx(-self.T, abs=0.03)
        assert hi == pytest.approx(0.0, abs=0.03)

    def test_the_old_double_thickness_bug_is_gone(self):
        """'outward' used to extract |sd| == thickness, giving a symmetric
        slab of 2 * thickness."""
        out = thicken_surface(_flat_sheet(), thickness=self.T, direction='outward',
                              resolution=self.RES, smooth_iterations=0)
        lo, hi = self._z_span(out, footprint=4.0)
        assert hi - lo < 1.5 * self.T

    def test_directions_are_mirror_images(self):
        up = thicken_surface(_flat_sheet(), thickness=self.T, direction='outward',
                             resolution=self.RES, smooth_iterations=0)
        down = thicken_surface(_flat_sheet(), thickness=self.T, direction='inward',
                               resolution=self.RES, smooth_iterations=0)
        lo_u, hi_u = self._z_span(up, footprint=4.0)
        lo_d, hi_d = self._z_span(down, footprint=4.0)
        assert lo_u == pytest.approx(-hi_d, abs=0.05)
        assert hi_u == pytest.approx(-lo_d, abs=0.05)

    def test_thickness_scales(self):
        thin = thicken_surface(_flat_sheet(), thickness=0.5, direction='both',
                               resolution=self.RES, smooth_iterations=0)
        lo, hi = self._z_span(thin, footprint=4.0)
        assert hi - lo == pytest.approx(0.5, abs=0.05)

    def test_far_from_origin_sheet_is_thickened_correctly(self):
        sheet = _flat_sheet()
        for v in sheet.vertices:
            v.position = v.position + np.array([1150.0, 0.0, 0.0])
        out = thicken_surface(sheet, thickness=self.T, direction='both',
                              resolution=self.RES, smooth_iterations=0)
        lo, hi = self._z_span(out, footprint=4.0, offset=(1150.0, 0.0))
        assert hi - lo == pytest.approx(self.T, abs=0.05)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
