"""Regression tests for the STL -> quad cage -> NURBS -> STEP pipeline.

These pin down bugs found while converting real topology-optimization STLs:
- trimesh 5.0 renamed simplify_quadric_decimation args (percent, face_count),
  so the old positional call raised and decimation was silently skipped.
- G3Fitter treated quad corners as a zigzag tensor grid while the converter
  passes them in cyclic winding order, producing self-intersecting (bowtie)
  patches whose boundary curves never coincide with neighbor patches.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest

from src.core.halfedge_mesh import HalfEdgeMesh
from src.nurbs.converter import SubDToNURBSConverter
from src.reverse_engineering.quad_wrap import QuadWrapper


def _two_adjacent_quads():
    """Two unit quads in the z=0 plane sharing the edge x=1, cyclic winding."""
    verts = np.array([
        [0, 0, 0], [1, 0, 0], [2, 0, 0],
        [0, 1, 0], [1, 1, 0], [2, 1, 0],
    ], dtype=float)
    faces = [[0, 1, 4, 3], [1, 2, 5, 4]]
    return HalfEdgeMesh.from_arrays(verts, faces)


def _unique_edge_lengths(t_mesh):
    """Lengths of the unique undirected edges of a trimesh."""
    edges = np.unique(
        np.sort(t_mesh.faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1), axis=0)
    return np.linalg.norm(
        t_mesh.vertices[edges[:, 0]] - t_mesh.vertices[edges[:, 1]], axis=1)


def _min_over_median_edge(t_mesh):
    lens = _unique_edge_lengths(t_mesh)
    return float(lens.min() / np.median(lens))


class TestQuadWrapDecimation:
    def test_miq_parametrization_actually_decimates(self):
        import trimesh
        dense = trimesh.creation.icosphere(subdivisions=4)  # 5120 tris
        wrapper = QuadWrapper(target_face_count=50)
        param_V, param_F, _ = wrapper._miq_parametrization(dense, None)
        # target = 50 / 2.2 ~ 23 triangles; anything near the input count
        # means decimation silently failed. The LOWER bound matters just as
        # much: _repair_decimated's debris filter can delete the whole body,
        # and an upper-bound-only assert calls that a success.
        assert 12 <= len(param_F) <= 3 * 105, (
            f"decimation produced {len(param_F)} faces from "
            f"{len(dense.faces)} (expected roughly 22)"
        )
        assert len(param_V) >= 8, f"only {len(param_V)} vertices survived"

        # ...and the decimated body must still cover the original sphere.
        param_mesh = trimesh.Trimesh(vertices=param_V, faces=param_F, process=False)
        scale = float(np.abs(dense.extents).max())
        assert np.abs(param_mesh.bounds - dense.bounds).max() < 0.15 * scale, (
            f"decimated body no longer covers the input: "
            f"{param_mesh.bounds.tolist()} vs {dense.bounds.tolist()}"
        )


class TestPatchCornerOrder:
    def test_patch_boundary_has_no_diagonals(self):
        mesh = _two_adjacent_quads()
        conv = SubDToNURBSConverter()
        patches = conv.generate_patches(mesh)
        assert len(patches) == 2
        for p in patches:
            cycle = [p[0, 0], p[5, 0], p[5, 5], p[0, 5]]
            for a in range(4):
                seg = np.linalg.norm(cycle[(a + 1) % 4] - cycle[a])
                # each patch is a unit quad: boundary corner-to-corner
                # segments must be edges (len 1), never diagonals (len ~1.414)
                assert seg == pytest.approx(1.0, abs=0.05), (
                    f"patch boundary contains a diagonal segment (len {seg:.3f})"
                )

    def test_adjacent_patches_share_boundary_curve(self):
        mesh = _two_adjacent_quads()
        conv = SubDToNURBSConverter()
        patches = conv.generate_patches(mesh)

        shared = {(1.0, 0.0, 0.0), (1.0, 1.0, 0.0)}

        def boundary_rows(p):
            return [p[0, :], p[5, :], p[:, 0], p[:, 5]]

        curves = []
        for p in patches:
            for row in boundary_rows(p):
                e0 = tuple(np.round(row[0], 6))
                e1 = tuple(np.round(row[5], 6))
                if {e0, e1} == shared:
                    curves.append(np.array(row))

        assert len(curves) == 2, (
            "each patch must have exactly one boundary curve on the shared edge"
        )
        a, b = curves
        same = np.allclose(a, b, atol=1e-9)
        reversed_match = np.allclose(a, b[::-1], atol=1e-9)
        assert same or reversed_match, (
            "shared boundary curves differ -> sewing cannot join the patches"
        )

    def test_patch_corners_interpolate_face_corners(self):
        """All four ctrl-grid corners must sit on the quad's (limit) corners,
        in the same cyclic order the face is wound in.

        On this flat sheet the Catmull-Clark limit positions coincide with the
        input positions (verified: max deviation 0.0), so the face's vertex
        positions ARE the expected patch corners.
        """
        mesh = _two_adjacent_quads()
        conv = SubDToNURBSConverter()
        patches = conv.generate_patches(mesh)
        assert len(patches) == 2
        for face, p in zip(mesh.faces, patches):
            face_verts = mesh.get_face_vertices(face)
            face_pts = {tuple(np.round(v.position, 6)) for v in face_verts}
            ctrl_corner_pts = {
                tuple(np.round(p[i, j], 6)) for i in (0, 5) for j in (0, 5)
            }
            assert ctrl_corner_pts == face_pts, (
                f"patch corners {sorted(ctrl_corner_pts)} are not the face's "
                f"corners {sorted(face_pts)}"
            )
            # Set equality alone still tolerates the original bowtie bug
            # (c0,c1,c2,c3 laid onto (0,0),(5,0),(0,5),(5,5)); pin the winding.
            for k, (i, j) in enumerate([(0, 0), (5, 0), (5, 5), (0, 5)]):
                assert np.allclose(p[i, j], face_verts[k].position, atol=1e-9), (
                    f"ctrl corner ({i},{j}) = {p[i, j]} does not match face "
                    f"vertex {k} at {face_verts[k].position} -> bowtie patch"
                )


class TestReferenceModePatches:
    def test_shared_boundary_cur_curve_with_reference(self):
        """Reference-fitted patches (tangent-plane boundary curves) must still
        produce identical shared boundary curves so sewing can close."""
        mesh = _two_adjacent_quads()
        conv = SubDToNURBSConverter()
        patches = conv.generate_patches(mesh, reference_mesh=mesh)

        shared = {(1.0, 0.0, 0.0), (1.0, 1.0, 0.0)}
        curves = []
        for p in patches:
            for row in [p[0, :], p[5, :], p[:, 0], p[:, 5]]:
                e0 = tuple(np.round(row[0], 6))
                e1 = tuple(np.round(row[5], 6))
                if {e0, e1} == shared:
                    curves.append(np.array(row))
        assert len(curves) == 2
        a, b = curves
        assert np.allclose(a, b, atol=1e-9) or np.allclose(a, b[::-1], atol=1e-9)

    def test_reference_mode_corners_stay_on_surface(self):
        """Reference mode must pin the patch corners to the CAGE vertices, not
        to the Catmull-Clark limit positions (which are millimetres off a
        shrink-wrapped cage).

        The flat two-quad sheet cannot test this at all: there limit == cage,
        so reference and non-reference mode produce identical output. Use a
        twice-subdivided box, where the two differ by a measurable 0.0195.
        """
        import trimesh
        from src.subd.primitives import create_box
        from src.subd.catmull_clark import subdivide, evaluate_limit_surface

        mesh = subdivide(create_box(), 2)
        cage = np.array([v.position for v in mesh.vertices])
        limit, _ = evaluate_limit_surface(mesh)
        gap = np.abs(limit - cage).max()
        # precondition: the fixture must be able to tell the two modes apart
        assert gap > 1e-3, f"fixture cannot distinguish limit from cage (gap {gap})"

        conv = SubDToNURBSConverter()
        patches = conv.generate_patches(mesh, reference_mesh=mesh)
        quad_faces = [f for f in mesh.faces if len(mesh.get_face_vertices(f)) == 4]
        assert len(patches) == len(quad_faces) > 0

        dev_cage = 0.0
        dev_limit = 0.0
        for face, p in zip(quad_faces, patches):
            fv = mesh.get_face_vertices(face)
            for k, (i, j) in enumerate([(0, 0), (5, 0), (5, 5), (0, 5)]):
                dev_cage = max(dev_cage, np.abs(p[i, j] - fv[k].position).max())
                dev_limit = max(dev_limit, np.abs(p[i, j] - limit[fv[k].index]).max())
        assert dev_cage < 1e-9, (
            f"reference-mode corners drifted off the cage by {dev_cage:.3e}"
        )
        assert dev_limit > 1e-3, (
            "reference-mode corners are the Catmull-Clark limit positions -- "
            "reference mode is not in effect"
        )

        # every control point (not just the corners) must hug the reference
        # surface: a broken boundary-tangent construction throws the six-point
        # edge curves far off the part while leaving the corners intact.
        all_ctrl = np.vstack([p.reshape(-1, 3) for p in patches])
        ref_tm = mesh.to_trimesh()
        _, dist, _ = trimesh.proximity.closest_point(ref_tm, all_ctrl)
        quad_diag = np.median([
            np.linalg.norm(
                np.array([p[0, 0], p[5, 0], p[5, 5], p[0, 5]]).max(axis=0)
                - np.array([p[0, 0], p[5, 0], p[5, 5], p[0, 5]]).min(axis=0))
            for p in patches
        ])
        assert dist.max() < 0.25 * quad_diag, (
            f"control points sit up to {dist.max():.4f} off the reference "
            f"surface (quad size {quad_diag:.4f})"
        )


class TestMeshToolsTrimesh5:
    def test_decimate_mesh_reduces_faces(self):
        import trimesh
        from src.reverse_engineering.mesh_tools import decimate_mesh
        dense = HalfEdgeMesh.from_trimesh(trimesh.creation.icosphere(subdivisions=3))
        out = decimate_mesh(dense, target_faces=100)
        # Bracket from BOTH sides: an upper-bound-only assert is satisfied by a
        # 20-face blob and by an entirely EMPTY mesh. Real code returns exactly
        # 100 faces for target_faces=100.
        assert 50 <= len(out.faces) <= 300, (
            f"decimation silently failed: {len(dense.faces)} -> {len(out.faces)}"
        )
        # ...and the shape must survive the reduction.
        tin, tout = dense.to_trimesh(), out.to_trimesh()
        scale = float(np.abs(tin.extents).max())
        assert np.abs(tout.bounds - tin.bounds).max() < 0.1 * scale, (
            f"decimated mesh no longer covers the input: "
            f"{tout.bounds.tolist()} vs {tin.bounds.tolist()}"
        )
        assert abs(tout.volume - tin.volume) / abs(tin.volume) < 0.15, (
            f"decimation changed the volume too much: "
            f"{tin.volume:.4f} -> {tout.volume:.4f}"
        )

    def test_remove_duplicate_vertices_merges(self):
        from src.reverse_engineering.mesh_tools import remove_duplicate_vertices
        # two triangles sharing an edge, but with duplicated corner vertices
        verts = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0],
            [1, 0, 0], [1, 1, 0], [0, 1, 0],
        ], dtype=float)
        faces = [[0, 1, 2], [3, 4, 5]]
        mesh = HalfEdgeMesh.from_arrays(verts, faces)
        out = remove_duplicate_vertices(mesh, tolerance=1e-6)
        assert len(out.vertices) == 4, (
            f"expected 4 vertices after merge, got {len(out.vertices)}"
        )


class TestDecimationRepair:
    def test_repair_drops_debris_and_nonmanifold(self):
        import trimesh
        base = trimesh.creation.icosphere(subdivisions=1)  # 80 faces, watertight
        # non-manifold: an extra triangle glued onto an existing edge
        e0, e1 = base.faces[0][0], base.faces[0][1]
        apex_idx = len(base.vertices)
        verts = np.vstack([base.vertices, base.vertices[e0] + [0.5, 0.5, 0.5]])
        extra = np.array([[e0, e1, apex_idx]])
        # debris: a lone far-away triangle (its own component)
        debris_off = len(verts)
        verts = np.vstack([verts, [[10, 0, 0], [11, 0, 0], [10, 1, 0]]])
        debris = np.array([[debris_off, debris_off + 1, debris_off + 2]])
        dirty = trimesh.Trimesh(
            vertices=verts,
            faces=np.vstack([base.faces, extra, debris]),
            process=False,
        )
        # One precondition PER injected defect. `not is_watertight` is
        # satisfied by either defect alone, so if one injection silently
        # stopped working its post-condition below would go vacuous while the
        # other assert kept the test green -- the exact failure mode this file
        # exists to prevent.
        assert not dirty.is_watertight
        assert len(dirty.split(only_watertight=False)) == 3, (
            "debris component not created"
        )
        dirty_edges = np.sort(
            dirty.faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
        _, dirty_counts = np.unique(dirty_edges, axis=0, return_counts=True)
        assert (dirty_counts > 2).sum() == 1, "non-manifold edge not created"

        repaired = QuadWrapper(target_face_count=50)._repair_decimated(dirty)
        comps = repaired.split(only_watertight=False)
        assert len(comps) == 1, f"debris not dropped: {len(comps)} components"
        edges = np.sort(repaired.faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
        _, counts = np.unique(edges, axis=0, return_counts=True)
        assert (counts <= 2).all(), "non-manifold edges remain"
        assert repaired.is_watertight


class TestRemoveSliverEdges:
    def _mesh_with_short_edge(self):
        import trimesh
        base = trimesh.creation.icosphere(subdivisions=2)
        verts = np.asarray(base.vertices).copy()
        # shrink an ACTUAL mesh edge to 2% of its length
        a, b = base.edges_unique[0]
        verts[b] = verts[a] + 0.02 * (verts[b] - verts[a])
        dirty = trimesh.Trimesh(vertices=verts, faces=base.faces, process=False)
        # precondition: the defect must exist, or every assertion below is
        # vacuous (a repair no-op would pass) — this is exactly how the first
        # version of this test silently tested nothing
        assert _min_over_median_edge(dirty) < 0.15, "defect construction failed"
        return dirty

    def test_collapse_short_edges(self):
        from src.reverse_engineering.mesh_tools import collapse_short_edges
        dirty = self._mesh_with_short_edge()
        out = collapse_short_edges(dirty, rel_threshold=0.15)
        assert len(out.faces) < len(dirty.faces), "collapse did not happen"
        assert _min_over_median_edge(out) >= 0.15
        assert out.is_watertight

    def test_flip_needle_triangles(self):
        import trimesh
        from src.reverse_engineering.mesh_tools import flip_needle_triangles

        def worst_height(m):
            pts = m.vertices[m.faces]
            e = np.roll(pts, -1, axis=1) - pts
            areas = 0.5 * np.linalg.norm(np.cross(e[:, 0], -e[:, 2]), axis=1)
            longest = np.linalg.norm(e, axis=2).max(axis=1)
            return (2.0 * areas / longest).min()

        base = trimesh.creation.icosphere(subdivisions=2)
        verts = np.asarray(base.vertices).copy()
        # squash an ACTUAL face: its apex moves almost onto the opposite
        # edge's midpoint -> needle triangle with long edges, tiny height
        p, q, r = base.faces[0]
        verts[r] = 0.998 * (verts[p] + verts[q]) / 2.0 + 0.002 * verts[r]
        dirty = trimesh.Trimesh(vertices=verts, faces=base.faces, process=False)
        # precondition: the needle must actually be below the repair threshold
        med = np.median(np.linalg.norm(
            dirty.vertices[dirty.edges_unique[:, 0]] -
            dirty.vertices[dirty.edges_unique[:, 1]], axis=1))
        assert worst_height(dirty) < 0.05 * med, "defect construction failed"

        out = flip_needle_triangles(dirty, rel_height=0.05)
        assert worst_height(out) > worst_height(dirty)
        # "some improvement" is not enough: a stub that nudges the apex by 1e-6
        # satisfies the strict inequality above while the needle survives.
        # Demand the needle is actually gone, i.e. back above the threshold the
        # repair itself works to. Measured on real code: 0.0616 vs 0.0156.
        med_out = np.median(np.linalg.norm(
            out.vertices[out.edges_unique[:, 0]] -
            out.vertices[out.edges_unique[:, 1]], axis=1))
        assert worst_height(out) >= 0.05 * med_out, (
            f"needle not repaired: height {worst_height(out):.6f} still below "
            f"{0.05 * med_out:.6f}"
        )
        assert out.is_watertight

    def test_remove_sliver_edges_halfedge_roundtrip(self):
        from src.reverse_engineering.mesh_tools import remove_sliver_edges
        dirty = HalfEdgeMesh.from_trimesh(self._mesh_with_short_edge())
        # Re-assert the precondition on the mesh ACTUALLY handed to the repair:
        # _mesh_with_short_edge checks the trimesh, not the HalfEdgeMesh.
        in_tm = dirty.to_trimesh()
        assert _min_over_median_edge(in_tm) < 0.15, (
            "the sliver did not survive the HalfEdgeMesh round-trip"
        )
        # the input is already closed, so the boundary-edge assert below is
        # about PRESERVING that, not about the repair
        assert sum(1 for e in dirty.edges if dirty.is_boundary_edge(e)) == 0

        out = remove_sliver_edges(dirty)
        assert len(out.faces) < len(dirty.faces)
        # the missing post-condition: the sliver must be GONE. Collapsing the
        # wrong (longest) edge also drops faces and keeps the mesh closed.
        # Measured: real code reaches 0.879, the wrong-edge variant stays at
        # 0.0176.
        out_tm = out.to_trimesh()
        assert _min_over_median_edge(out_tm) >= 0.15, (
            f"sliver edge survived the repair: min/median = "
            f"{_min_over_median_edge(out_tm):.4f}"
        )
        assert sum(1 for e in out.edges if out.is_boundary_edge(e)) == 0


class TestForceConvexFallback:
    def test_irreparably_concave_quad_gets_forced_convex(self):
        """A quad that was concave BEFORE relaxation cannot be fixed by the
        revert strategy; the parallelogram fallback must still fix it
        (SolidWorks rejects a whole body over one folded patch)."""
        verts = np.array([
            [0, 0, 0], [2, 0, 0], [0.4, 0.4, 0], [0, 2, 0],  # reflex at idx 2
        ], dtype=float)
        mesh = HalfEdgeMesh.from_arrays(verts, [[0, 1, 2, 3]])
        wrapper = QuadWrapper()
        assert wrapper._concave_quad_ids(mesh) == [0]

        fallback = np.array([v.position.copy() for v in mesh.vertices])
        wrapper._repair_concave_quads(mesh, mesh, fallback)
        assert wrapper._concave_quad_ids(mesh) == []


class TestExpectedBodyCount:
    def test_debris_above_size_threshold_still_dropped(self):
        """A decimation fragment can be big enough to pass the size filter;
        the input body count must cap the surviving components regardless."""
        import trimesh
        main = trimesh.creation.icosphere(subdivisions=2)  # 320 faces
        blob = trimesh.creation.icosphere(subdivisions=1)  # 80 faces > 1% size
        blob.apply_translation([10, 0, 0])
        dirty = trimesh.util.concatenate([main, blob])

        repaired = QuadWrapper(target_face_count=100)._repair_decimated(
            dirty, expected_bodies=1)
        comps = repaired.split(only_watertight=False)
        assert len(comps) == 1
        assert len(comps[0].faces) == 320  # the largest body survived

    def test_wrap_single_body_input_yields_single_component_cage(self):
        import trimesh
        dense = trimesh.creation.icosphere(subdivisions=4)  # 5120 tris
        cage = QuadWrapper(target_face_count=300).wrap(
            HalfEdgeMesh.from_trimesh(dense))

        # `len(comps) == 1` is already true of the dense INPUT, and wrap()
        # returns exactly that (reference_mesh.copy()) from its except branch.
        # So first assert a real quad cage was produced. Measured on real
        # code: 292 faces, all quads, 1 component.
        assert len(cage.faces) < len(dense.faces), (
            f"wrap returned an undecimated mesh: {len(dense.faces)} -> "
            f"{len(cage.faces)} faces (the except fallback fired)"
        )
        assert all(len(cage.get_face_vertices(f)) == 4 for f in cage.faces), (
            "wrap did not produce a pure-quad cage"
        )
        assert 150 <= len(cage.faces) <= 600, (
            f"cage size {len(cage.faces)} far from the requested 300 quads"
        )

        comps = cage.to_trimesh().split(only_watertight=False)
        assert len(comps) == 1


class TestLinearSubdivide:
    def test_subdivide_smooth_kwarg_accepted(self):
        from src.subd.primitives import create_box
        from src.subd.catmull_clark import subdivide
        box = create_box()
        out = subdivide(box, 1, smooth=False)
        assert len(out.faces) == 4 * len(box.faces)

        # The face count is identical for smooth=True, so a wrapper that
        # silently discards the kwarg passes a shape-only assert. Make it
        # behavioural: the two modes must produce different geometry.
        smooth_out = subdivide(box, 1, smooth=True)
        assert len(smooth_out.faces) == len(out.faces)
        lin_pts = np.array(sorted(
            tuple(np.round(v.position, 9)) for v in out.vertices))
        smooth_pts = np.array(sorted(
            tuple(np.round(v.position, 9)) for v in smooth_out.vertices))
        assert lin_pts.shape == smooth_pts.shape
        assert not np.allclose(lin_pts, smooth_pts), (
            "smooth=False produced the same geometry as smooth=True -> the "
            "kwarg is being ignored"
        )

    def test_linear_subdivide_preserves_positions(self):
        from src.subd.primitives import create_box
        from src.subd.catmull_clark import subdivide
        box = create_box()
        orig = {tuple(np.round(v.position, 9)) for v in box.vertices}
        out = subdivide(box, 1, smooth=False)
        new = {tuple(np.round(v.position, 9)) for v in out.vertices}
        assert orig.issubset(new), "linear subdivision moved existing vertices"


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
