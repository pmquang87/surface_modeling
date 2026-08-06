
import os
import sys

# Ensure src is in the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.halfedge_mesh import HalfEdgeMesh
from src.subd.primitives import create_box, create_cylinder, create_sphere
from src.reverse_engineering.quad_wrap import QuadWrapper
from src.reverse_engineering.mesh_tools import decimate_mesh, compute_mesh_quality
from src.subd import catmull_clark

def assert_mesh_is_watertight(mesh: HalfEdgeMesh):
    """Helper to assert a mesh has no holes and is manifold."""
    stats = compute_mesh_quality(mesh)
    assert stats['boundary_edges'] == 0, f"Mesh has {stats['boundary_edges']} boundary edges (holes)!"
    assert stats['non_manifold_edges'] == 0, "Mesh has non-manifold edges!"
    
    # Also verify that every halfedge has a twin
    for he in mesh.half_edges:
        assert he.twin is not None, f"HalfEdge {he.index} has no twin, mesh is open!"

def test_quad_wrap_preserves_watertightness():
    """Test that quad wrap produces a closed mesh when given a closed mesh."""
    # Start with a simple closed sphere
    sphere = create_sphere(radius=5.0, rings=8, segments=8)
    assert_mesh_is_watertight(sphere)
    
    wrapper = QuadWrapper(target_face_count=20, smoothing_weight=0.1)
    quad_mesh = wrapper.wrap(sphere)
    
    # The resulting quad mesh should still be watertight
    assert len(quad_mesh.faces) > 0
    assert_mesh_is_watertight(quad_mesh)

def test_decimation_preserves_watertightness():
    """Test that decimation preserves the closed volume."""
    sphere = create_sphere(radius=5.0, rings=16, segments=16)
    assert_mesh_is_watertight(sphere)
    
    decimated = decimate_mesh(sphere, target_faces=50)
    print(f"Decimated faces: {len(decimated.faces)}, original: {len(sphere.faces)}")
    assert len(decimated.faces) > 0
    assert_mesh_is_watertight(decimated)

def test_subdivision_preserves_watertightness():
    """Test that Catmull-Clark subdivision preserves watertightness."""
    cube = create_box(width=2.0)
    assert_mesh_is_watertight(cube)
    
    subdivided = catmull_clark.subdivide(cube, levels=2)
    
    assert len(subdivided.faces) == 6 * 16 # 6 faces * 4^2
    assert_mesh_is_watertight(subdivided)

if __name__ == "__main__":
    print("Testing quad wrap...")
    test_quad_wrap_preserves_watertightness()
    print("Testing decimation...")
    test_decimation_preserves_watertightness()
    print("Testing subdivision...")
    test_subdivision_preserves_watertightness()
    print("All topological invariants passed!")
