import pytest
import numpy as np

# 1. SoA Mesh Imports
from src.core.mesh_soa import MeshSOA

# 2. GPU Compute Imports
try:
    import pyopencl as cl
    from src.subd.gpu_compute import GPUCatmullClark
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
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ])
        faces = np.array([[0, 1, 2]])
        
        mesh = MeshSOA(vertices, faces)
        normals = mesh.compute_face_normals()
        
        assert normals.shape == (1, 3)
        assert np.allclose(normals[0], [0.0, 0.0, 1.0])
        
    def test_vertex_normals(self):
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ])
        faces = np.array([[0, 1, 2]])
        
        mesh = MeshSOA(vertices, faces)
        v_normals = mesh.compute_vertex_normals()
        
        assert v_normals.shape == (3, 3)
        for i in range(3):
            assert np.allclose(v_normals[i], [0.0, 0.0, 1.0])


@pytest.mark.skipif(not HAS_OPENCL, reason="PyOpenCL is not available or no context can be created.")
class TestGPUCompute:
    def test_gpu_subdivision(self):
        # Single quad (needs to be flattened)
        # However, GPUCatmullClark subdivide method expects 
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
            subdivider = GPUCatmullClark()
        except cl.Error:
            pytest.skip("No OpenCL devices found for GPU test.")
            
        new_vertices, new_face_vertices, new_face_offsets = subdivider.subdivide(vertices, face_vertices)
        
        # A single quad subdivides into 4 quads, so 4 faces * 4 vertices = 16 new face vertex indices
        assert len(new_face_vertices) == 16
        # The number of new faces should be 4 (offsets length is 5)
        assert len(new_face_offsets) == 5
        
        # Original vertices (4) + Edges (4) + Faces (1) = 9 new vertices
        assert new_vertices.shape == (9, 3)


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
        
    def test_local_knot_vector(self):
        mesh = TMesh()
        
        # Create a line of 5 vertices
        v0 = mesh.add_vertex(0, 0, 0)
        v1 = mesh.add_vertex(1, 0, 0)
        v2 = mesh.add_vertex(2, 0, 0)
        v3 = mesh.add_vertex(3, 0, 0)
        v4 = mesh.add_vertex(4, 0, 0)
        
        mesh.add_edge(v0.id, v1.id, 'right', 'left', 1.0)
        mesh.add_edge(v1.id, v2.id, 'right', 'left', 1.0)
        mesh.add_edge(v2.id, v3.id, 'right', 'left', 1.0)
        mesh.add_edge(v3.id, v4.id, 'right', 'left', 1.0)
        
        kv = mesh.extract_local_knot_vector(v2.id, direction='s', degree=3)
        assert len(kv) == 5
        
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


class TestG3Fitter:
    def test_generate_patch(self):
        fitter = G3Fitter()
        
        corners = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0]
        ])
        
        patch = fitter.generate_patch(corners)
        
        assert patch.shape == (6, 6, 3)
        
        # Check corners G0 positional matching
        assert np.allclose(patch[0, 0], corners[0])
        assert np.allclose(patch[0, 5], corners[1])
        assert np.allclose(patch[5, 0], corners[2])
        assert np.allclose(patch[5, 5], corners[3])

    def test_fit_surface(self):
        fitter = G3Fitter()
        quad_mesh = [
            {'corners': np.array([
                [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]
            ])},
            {'corners': np.array([
                [1, 0, 0], [2, 0, 0], [1, 1, 0], [2, 1, 0]
            ])}
        ]
        
        patches = fitter.fit_surface(quad_mesh)
        assert len(patches) == 2
        for p in patches:
            assert p.shape == (6, 6, 3)
