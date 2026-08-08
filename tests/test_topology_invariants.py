
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

def _face_size_histogram(mesh: HalfEdgeMesh) -> dict:
    hist = {}
    for f in mesh.faces:
        n = len(mesh.get_face_vertices(f))
        hist[n] = hist.get(n, 0) + 1
    return hist

def test_quad_wrap_preserves_watertightness():
    """Test that quad wrap produces a closed QUAD cage when given a closed mesh.

    Watertightness alone is vacuous here: the input sphere is already
    watertight, so a wrap() that returned its input unchanged (which is exactly
    what the `except` branch in QuadWrapper.wrap does) would satisfy it. So
    assert the transformation happened FIRST, then the invariant.
    """
    # Start with a simple closed sphere
    sphere = create_sphere(radius=5.0, rings=8, segments=8)
    assert_mesh_is_watertight(sphere)

    # Precondition: the input is a MIXED tri/quad mesh (16 pole triangles),
    # so "every output face is a quad" below cannot be satisfied by an
    # identity wrap.
    in_hist = _face_size_histogram(sphere)
    assert in_hist == {3: 16, 4: 48}, f"fixture changed: {in_hist}"

    wrapper = QuadWrapper(target_face_count=20, smoothing_weight=0.1)
    quad_mesh = wrapper.wrap(sphere)

    # Effects: a NEW, decimated, pure-quad cage.
    assert quad_mesh is not sphere
    assert len(quad_mesh.faces) > 0
    out_hist = _face_size_histogram(quad_mesh)
    assert set(out_hist) == {4}, (
        f"quad wrap did not produce a pure-quad cage: face sizes {out_hist}"
    )
    assert len(quad_mesh.faces) < len(sphere.faces), (
        f"quad wrap did not decimate: {len(sphere.faces)} -> {len(quad_mesh.faces)}"
    )
    # target_face_count=20 -> 16 quads measured; bracket from BOTH sides so a
    # cage that collapsed to a few faces also fails.
    assert 8 <= len(quad_mesh.faces) <= 40, (
        f"cage size {len(quad_mesh.faces)} far from the requested 20 quads"
    )

    # The resulting quad mesh should still be watertight
    assert_mesh_is_watertight(quad_mesh)

def test_decimation_preserves_watertightness():
    """Test that decimation reduces the face count AND preserves the closed volume."""
    sphere = create_sphere(radius=5.0, rings=16, segments=16)
    assert_mesh_is_watertight(sphere)

    decimated = decimate_mesh(sphere, target_faces=50)

    # decimate_mesh has three `return mesh.copy()` fallback paths; without
    # these asserts a silently skipped decimation is indistinguishable from a
    # successful one (the sphere is already watertight).
    assert decimated is not sphere
    assert len(decimated.faces) > 0
    assert len(decimated.faces) < len(sphere.faces), (
        f"decimation silently fell back: {len(sphere.faces)} -> "
        f"{len(decimated.faces)} faces"
    )
    # target_faces=50 -> exactly 50 measured; bracket both sides.
    assert 25 <= len(decimated.faces) <= 75, (
        f"decimation missed the requested 50-face budget: {len(decimated.faces)}"
    )

    # The body must survive: an over-aggressive collapse or a degenerate
    # result would keep the mesh "closed" while destroying the shape.
    vol_in = sphere.to_trimesh().volume
    vol_out = decimated.to_trimesh().volume
    assert abs(vol_out - vol_in) / abs(vol_in) < 0.15, (
        f"decimation changed the volume too much: {vol_in:.3f} -> {vol_out:.3f}"
    )

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
