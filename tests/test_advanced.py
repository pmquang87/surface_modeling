import pytest
import numpy as np

# 1. SoA Mesh Imports
from src.core.mesh_soa import MeshSOA

# 2. GPU Compute Imports
try:
    import pyopencl as cl
    # NOTE: this used to import a class named `GPUCatmullClark`, which does not
    # exist in src/subd/gpu_compute.py -- the ImportError was swallowed below,
    # so TestGPUCompute was permanently skipped and never ran anywhere.
    from src.subd.gpu_compute import OpenCLSubdivider
    HAS_OPENCL = True
except (ImportError, Exception):
    HAS_OPENCL = False

# 3. T-Splines Imports
from src.core.tsplines import TMesh

# 4. G3 NURBS Imports
from src.nurbs.g3_fitter import G3Fitter

class TestSoAMesh:
    def test_mesh_soa_creation(self):
        # A simple quad made of 2 triangles
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0]
        ], dtype=np.float64)
        
        faces = np.array([
            [0, 1, 2],
            [0, 2, 3]
        ], dtype=np.int64)
        
        mesh = MeshSOA(vertices, faces)
        
        assert mesh.V == 4
        assert mesh.F == 2
        assert mesh.E == 5  # 4 boundary edges + 1 diagonal
        
    def test_face_normals(self):
        # Three independent triangles, deliberately chosen so that
        #  - the RAW cross product is NOT unit length (so a missing
        #    normalisation is detectable),
        #  - one face is not axis aligned (so a hard-coded [0,0,1] fails),
        #  - one face is the reverse winding of another (pins the sign).
        vertices = np.array([
            [0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, 0.0],   # raw cross (0,0,6)
            [0.0, 0.0, 0.0], [0.0, 3.0, 0.0], [2.0, 0.0, 0.0],   # reversed -> (0,0,-6)
            [0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 2.0],   # raw cross (0,-4,4)
        ])
        faces = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])

        mesh = MeshSOA(vertices, faces)
        normals = mesh.compute_face_normals()

        assert normals.shape == (3, 3)
        # normalisation
        assert np.allclose(np.linalg.norm(normals, axis=1), 1.0), (
            f"face normals are not unit length: {np.linalg.norm(normals, axis=1)}"
        )
        # direction + sign convention
        r2 = 1.0 / np.sqrt(2.0)
        assert np.allclose(normals[0], [0.0, 0.0, 1.0])
        assert np.allclose(normals[1], [0.0, 0.0, -1.0])
        assert np.allclose(normals[2], [0.0, -r2, r2])

    def test_vertex_normals(self):
        # Two triangles meeting at a 90-degree fold along the shared edge A-B.
        # The two shared vertices must get the AREA-WEIGHTED blend of both face
        # normals, while each apex keeps its own face normal -- so every vertex
        # normal is different and copying one face normal everywhere fails.
        vertices = np.array([
            [0.0, 0.0, 0.0],   # A - on the fold
            [1.0, 0.0, 0.0],   # B - on the fold
            [0.5, 1.0, 0.0],   # C - apex of the z=0 triangle
            [0.5, 0.0, 2.0],   # D - apex of the y=0 triangle (twice the area)
        ])
        faces = np.array([
            [0, 1, 2],   # face normal (0, 0, 1), raw cross magnitude 1
            [1, 0, 3],   # face normal (0, 1, 0), raw cross magnitude 2
        ])

        mesh = MeshSOA(vertices, faces)
        f_normals = mesh.compute_face_normals()
        assert np.allclose(f_normals[0], [0.0, 0.0, 1.0])
        assert np.allclose(f_normals[1], [0.0, 1.0, 0.0])

        v_normals = mesh.compute_vertex_normals()
        assert v_normals.shape == (4, 3)
        assert np.allclose(np.linalg.norm(v_normals, axis=1), 1.0), (
            f"vertex normals are not unit length: {np.linalg.norm(v_normals, axis=1)}"
        )

        # area-weighted blend: (0,0,1)*1 + (0,1,0)*2 -> (0,2,1)/sqrt(5).
        # An UNWEIGHTED average of the unit face normals would give
        # (0, 1/sqrt2, 1/sqrt2) instead, so this also pins the weighting.
        fold = np.array([0.0, 2.0, 1.0]) / np.sqrt(5.0)
        assert np.allclose(v_normals[0], fold)
        assert np.allclose(v_normals[1], fold)
        assert np.allclose(v_normals[2], [0.0, 0.0, 1.0])
        assert np.allclose(v_normals[3], [0.0, 1.0, 0.0])


@pytest.mark.skipif(not HAS_OPENCL, reason="PyOpenCL is not available or no context can be created.")
class TestGPUCompute:
    def test_gpu_subdivision(self):
        # Single quad (needs to be flattened)
        # However, OpenCLSubdivider.subdivide expects
        # either a 1D array of face indices or 2D array of uniform faces.
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0]
        ], dtype=np.float32)

        face_vertices = np.array([
            [0, 1, 2, 3]
        ], dtype=np.int32)

        try:
            subdivider = OpenCLSubdivider()
        except cl.Error:
            pytest.skip("No OpenCL devices found for GPU test.")

        new_vertices, new_face_vertices, new_face_offsets = subdivider.subdivide(vertices, face_vertices)

        # A single quad subdivides into 4 quads, so 4 faces * 4 vertices = 16 new face vertex indices
        assert len(new_face_vertices) == 16
        # The number of new faces should be 4 (offsets length is 5)
        assert len(new_face_offsets) == 5

        # Original vertices (4) + Edges (4) + Faces (1) = 9 new vertices
        assert new_vertices.shape == (9, 3)

        # Sizes alone say nothing: a kernel returning right-sized buffers of
        # zeros/NaN would pass. For this unit square the Catmull-Clark answer
        # is fully determined, so pin every coordinate.
        assert np.all(np.isfinite(new_vertices)), "GPU kernel produced NaN/inf"

        # Layout is vstack([vertex_points(4), edge_points(4), face_points(1)]).
        # All 4 edges are boundary edges -> edge points are edge midpoints, and
        # each corner is a boundary vertex with 2 boundary edges, so the
        # boundary vertex rule gives (avg(2 adjacent edge midpoints) + P) / 2.
        expected = np.array([
            [0.125, 0.125, 0.0],   # corner v0 = (0,0,0)
            [0.875, 0.125, 0.0],   # corner v1 = (1,0,0)
            [0.875, 0.875, 0.0],   # corner v2 = (1,1,0)
            [0.125, 0.875, 0.0],   # corner v3 = (0,1,0)
            [0.5, 0.0, 0.0],       # edge (v0,v1) midpoint
            [1.0, 0.5, 0.0],       # edge (v1,v2) midpoint
            [0.5, 1.0, 0.0],       # edge (v2,v3) midpoint
            [0.0, 0.5, 0.0],       # edge (v3,v0) midpoint
            [0.5, 0.5, 0.0],       # face point = centroid
        ], dtype=np.float64)
        assert np.allclose(new_vertices, expected, atol=1e-5), (
            f"GPU subdivision positions wrong:\n{new_vertices}"
        )

        # Each new quad is (corner, next edge point, face point, prev edge point).
        expected_faces = np.array([
            0, 4, 8, 7,
            1, 5, 8, 4,
            2, 6, 8, 5,
            3, 7, 8, 6,
        ], dtype=np.int32)
        assert np.array_equal(np.asarray(new_face_vertices), expected_faces)
        assert np.array_equal(np.asarray(new_face_offsets),
                              np.array([0, 4, 8, 12, 16], dtype=np.int32))

        # Cross-check against the independent CPU implementation on the same
        # input (verified offline: it yields exactly the position set above),
        # which additionally guards against GPU/CPU divergence.
        from src.core.halfedge_mesh import HalfEdgeMesh
        from src.subd.catmull_clark import subdivide as cpu_subdivide
        cpu_mesh = HalfEdgeMesh()
        for p in vertices:
            cpu_mesh.add_vertex(np.asarray(p, dtype=np.float64).tolist())
        cpu_mesh.add_face([0, 1, 2, 3])
        cpu_out = cpu_subdivide(cpu_mesh, 1)
        cpu_pts = np.array(sorted(
            tuple(np.round(np.asarray(v.position, dtype=np.float64), 6))
            for v in cpu_out.vertices))
        gpu_pts = np.array(sorted(
            tuple(x) for x in np.round(np.asarray(new_vertices, dtype=np.float64), 6)))
        assert cpu_pts.shape == gpu_pts.shape
        assert np.allclose(cpu_pts, gpu_pts, atol=1e-5), (
            f"GPU subdivision diverges from the CPU implementation:\n"
            f"{gpu_pts}\nvs\n{cpu_pts}"
        )


class TestTSplines:
    def test_tmesh_creation(self):
        mesh = TMesh()
        
        v1 = mesh.add_vertex(0, 0, 0)
        v2 = mesh.add_vertex(1, 0, 0)
        v3 = mesh.add_vertex(1, 1, 0)
        
        assert len(mesh.vertices) == 3
        
        e1 = mesh.add_edge(v1.id, v2.id, 'right', 'left')
        assert len(mesh.edges) == 1
        assert e1.knot_interval == 1.0
        
    @staticmethod
    def _chain(intervals):
        """A horizontal chain of 5 vertices joined by the given knot intervals."""
        mesh = TMesh()
        vs = [mesh.add_vertex(float(i), 0.0, 0.0) for i in range(5)]
        for i, iv in enumerate(intervals):
            mesh.add_edge(vs[i].id, vs[i + 1].id, 'right', 'left', iv)
        return mesh, vs

    def test_local_knot_vector(self):
        # `len(kv) == degree + 2` is structurally guaranteed by the
        # implementation, so it tests nothing. Pin the VALUES instead: the
        # knots are the signed cumulative knot intervals walked outwards from
        # the centre vertex, with the centre knot at 0.
        mesh, vs = self._chain([1.0, 1.0, 1.0, 1.0])

        kv = mesh.extract_local_knot_vector(vs[2].id, direction='s', degree=3)
        assert len(kv) == 5
        assert kv == [-2.0, -1.0, 0.0, 1.0, 2.0], kv

    def test_local_knot_vector_non_uniform(self):
        # Non-uniform intervals: a stub returning constants cannot survive.
        mesh, vs = self._chain([1.0, 2.0, 0.5, 3.0])

        # walking out from v2: left 2.0 then 1.0, right 0.5 then 3.0
        assert mesh.extract_local_knot_vector(vs[2].id, direction='s', degree=3) == \
            [-3.0, -2.0, 0.0, 0.5, 3.5]
        # walking out from v1: left 1.0 then nothing, right 2.0 then 0.5
        assert mesh.extract_local_knot_vector(vs[1].id, direction='s', degree=3) == \
            [-1.0, -1.0, 0.0, 2.0, 2.5]

    def test_local_knot_vector_boundary_padding(self):
        # At a chain end the missing side must pad with the documented 0.0
        # interval, i.e. repeated knots, not run off the mesh.
        mesh, vs = self._chain([1.0, 1.0, 1.0, 1.0])

        assert mesh.extract_local_knot_vector(vs[0].id, direction='s', degree=3) == \
            [0.0, 0.0, 0.0, 1.0, 2.0]
        assert mesh.extract_local_knot_vector(vs[4].id, direction='s', degree=3) == \
            [-2.0, -1.0, 0.0, 0.0, 0.0]
        # the chain has no up/down edges at all, so the 't' direction is
        # entirely padding
        assert mesh.extract_local_knot_vector(vs[2].id, direction='t', degree=3) == \
            [0.0, 0.0, 0.0, 0.0, 0.0]


    def test_edge_split(self):
        mesh = TMesh()
        v1 = mesh.add_vertex(0, 0, 0)
        v2 = mesh.add_vertex(2, 0, 0)
        e = mesh.add_edge(v1.id, v2.id, 'right', 'left', 2.0)
        
        new_v = mesh.split_edge(e.id, alpha=0.5)
        
        assert len(mesh.vertices) == 3
        assert len(mesh.edges) == 2
        
        assert new_v.x == 1.0
        # Check knot intervals
        for edge in mesh.edges.values():
            assert edge.knot_interval == 1.0


def _bilinear_grid(c0, c1, c2, c3, n=6):
    """The analytic answer for a Coons patch over 4 corners with chord tangents
    and zero second derivatives: every control point is the bilinear blend

        p(u, v) = (1-u)(1-v) c0 + u(1-v) c1 + uv c2 + (1-u)v c3

    sampled at u = i/(n-1), v = j/(n-1). Both boundary curves (uniform spacing
    along each chord, because d1 == the chord itself) and the Coons interior
    blend collapse to this, so it pins all 36 control points, not just the 4
    corners.
    """
    t = np.linspace(0.0, 1.0, n)
    uu, vv = np.meshgrid(t, t, indexing='ij')
    return (((1 - uu) * (1 - vv))[..., None] * np.asarray(c0, dtype=float)
            + (uu * (1 - vv))[..., None] * np.asarray(c1, dtype=float)
            + (uu * vv)[..., None] * np.asarray(c2, dtype=float)
            + ((1 - uu) * vv)[..., None] * np.asarray(c3, dtype=float))


def _bernstein(n, i, t):
    from math import comb
    return comb(n, i) * (t ** i) * ((1 - t) ** (n - i))


def _eval_bezier_patch(ctrl, u, v):
    """Evaluate the degree-5 Bezier patch defined by a 6x6x3 control grid."""
    out = np.zeros(3)
    for i in range(6):
        for j in range(6):
            out += _bernstein(5, i, u) * _bernstein(5, j, v) * ctrl[i, j]
    return out


class TestG3Fitter:
    # corners in cyclic winding order around the quad (the convention
    # used by HalfEdgeMesh.get_face_vertices / SubDToNURBSConverter)
    UNIT_QUAD = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    SKEW_QUAD = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.5],
        [2.5, 3.0, 1.0],
        [-0.5, 2.0, -1.0],
    ])

    def test_generate_patch(self):
        fitter = G3Fitter()

        patch = fitter.generate_patch(self.UNIT_QUAD)

        assert patch.shape == (6, 6, 3)

        # Check corners G0 positional matching (cyclic -> tensor grid)
        assert np.allclose(patch[0, 0], self.UNIT_QUAD[0])
        assert np.allclose(patch[5, 0], self.UNIT_QUAD[1])
        assert np.allclose(patch[5, 5], self.UNIT_QUAD[2])
        assert np.allclose(patch[0, 5], self.UNIT_QUAD[3])

        # ...and ALL 36 control points, not just the 4 corners: the two
        # derivative-based boundary curves and the Coons interior blend were
        # completely untested when only the corners were checked.
        assert np.allclose(patch, _bilinear_grid(*self.UNIT_QUAD), atol=1e-12), (
            f"control grid is not the expected Coons/bilinear grid:\n{patch}"
        )

    def test_generate_patch_non_planar_quad(self):
        # Same analytic law on a skewed, non-planar quad, so a patch builder
        # that special-cases the unit square or drops the z axis fails.
        fitter = G3Fitter()
        patch = fitter.generate_patch(self.SKEW_QUAD)

        assert patch.shape == (6, 6, 3)
        assert np.allclose(patch, _bilinear_grid(*self.SKEW_QUAD), atol=1e-12)
        # boundary rows run along the actual quad edges (uniform spacing)
        assert np.allclose(patch[:, 0],
                           [self.SKEW_QUAD[0] + (i / 5.0) * (self.SKEW_QUAD[1] - self.SKEW_QUAD[0])
                            for i in range(6)], atol=1e-12)

    def _two_quad_fixture(self, bump=0.0):
        """Two adjacent unit quads sharing the edge (1,0,0)-(1,1,0), each with
        a 6x6 grid of dense samples taken off a KNOWN analytic surface.

        Without 'dense_points' and 'neighbors' the whole LSPIA sparse solve --
        the documented point of fit_surface -- is unreachable: no equation is
        ever added and the solver block is skipped entirely.
        """
        c_a = np.array([[0., 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
        c_b = np.array([[1., 0, 0], [2, 0, 0], [2, 1, 0], [1, 1, 0]])
        t = np.linspace(0.0, 1.0, 6)
        uu, vv = np.meshgrid(t, t, indexing='ij')
        # z = bump * u(1-u) v(1-v): zero on the whole quad boundary, so it is
        # exactly representable by a degree-5 patch whose boundary control
        # rows are fixed at z = 0.
        z = bump * uu * (1 - uu) * vv * (1 - vv)
        d_a = _bilinear_grid(*c_a)
        d_a[..., 2] = z
        d_b = _bilinear_grid(*c_b)
        d_b[..., 2] = z
        quad_mesh = [
            {'corners': c_a, 'dense_points': d_a, 'neighbors': [1, -1, -1, -1]},
            {'corners': c_b, 'dense_points': d_b, 'neighbors': [0, -1, -1, -1]},
        ]
        return quad_mesh, t

    def test_fit_surface_reproduces_dense_samples(self):
        # continuity_weight=0 disables the soft smoothness equations, so the
        # least-squares solution must reproduce the sampled surface exactly.
        fitter = G3Fitter(continuity_weight=0.0)
        quad_mesh, t = self._two_quad_fixture(bump=0.3)

        coons = [fitter.generate_patch(q['corners']) for q in quad_mesh]
        patches = fitter.fit_surface(quad_mesh)

        assert len(patches) == 2
        for p in patches:
            assert p.shape == (6, 6, 3)
            assert np.all(np.isfinite(p))

        for k, (p, q) in enumerate(zip(patches, quad_mesh)):
            # corners still interpolate the quad (cyclic -> tensor grid)
            assert np.allclose(p[0, 0], q['corners'][0], atol=1e-12)
            assert np.allclose(p[5, 0], q['corners'][1], atol=1e-12)
            assert np.allclose(p[5, 5], q['corners'][2], atol=1e-12)
            assert np.allclose(p[0, 5], q['corners'][3], atol=1e-12)

            # the fitted patch reproduces the dense samples
            for r in range(6):
                for c in range(6):
                    got = _eval_bezier_patch(p, t[r], t[c])
                    assert np.allclose(got, q['dense_points'][r, c], atol=1e-6), (
                        f"patch {k} misses sample ({r},{c}): {got} != "
                        f"{q['dense_points'][r, c]}"
                    )

            # the SOLVER actually moved the interior away from the Coons
            # initialization -- otherwise fit_surface would just be
            # generate_patch under a different name.
            moved = np.abs(p[1:5, 1:5] - coons[k][1:5, 1:5]).max()
            assert moved > 5e-3, (
                f"patch {k}: LSPIA solve left the Coons interior untouched "
                f"(max move {moved:.2e})"
            )

    def test_fit_surface_planar_and_shared_boundary(self):
        # Default continuity weight, exactly planar dense samples: the fitted
        # patches must stay in the plane and the two patches must agree on the
        # shared boundary curve (otherwise sewing cannot join them).
        fitter = G3Fitter()
        quad_mesh, _ = self._two_quad_fixture(bump=0.0)

        patches = fitter.fit_surface(quad_mesh)
        assert len(patches) == 2

        for k, (p, q) in enumerate(zip(patches, quad_mesh)):
            assert p.shape == (6, 6, 3)
            assert np.all(np.isfinite(p))
            assert np.abs(p[..., 2]).max() < 1e-12, (
                f"patch {k} left the z=0 plane despite planar input data"
            )
            assert np.allclose(p[0, 0], q['corners'][0], atol=1e-12)
            assert np.allclose(p[5, 0], q['corners'][1], atol=1e-12)
            assert np.allclose(p[5, 5], q['corners'][2], atol=1e-12)
            assert np.allclose(p[0, 5], q['corners'][3], atol=1e-12)

        shared = {(1.0, 0.0, 0.0), (1.0, 1.0, 0.0)}
        curves = []
        for p in patches:
            for row in (p[0, :], p[5, :], p[:, 0], p[:, 5]):
                ends = {tuple(np.round(row[0], 6)), tuple(np.round(row[5], 6))}
                if ends == shared:
                    curves.append(np.array(row))
        assert len(curves) == 2, (
            f"expected one shared boundary curve per patch, got {len(curves)}"
        )
        a, b = curves
        assert np.allclose(a, b, atol=1e-9) or np.allclose(a, b[::-1], atol=1e-9), (
            "shared boundary curves differ -> sewing cannot join the patches"
        )
