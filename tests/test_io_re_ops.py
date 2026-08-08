import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tempfile
import traceback
import time

import numpy as np
import pytest
import trimesh

from src.io.importers import import_stl, import_obj, import_step
from src.io.exporters import export_stl, export_obj, export_step
from src.reverse_engineering.mesh_tools import smooth_mesh, fill_holes, offset_mesh, decimate_mesh, compute_mesh_quality
from src.reverse_engineering.quad_wrap import QuadWrapper
from src.reverse_engineering.shrink_wrap import ShrinkWrapper
from src.operations.shell_thicken import shell_solid, thicken_surface
from src.subd.primitives import create_box, create_plane
from src.nurbs.converter import SubDToNURBSConverter
from src.core.halfedge_mesh import HalfEdgeMesh

STL_FILE = None
DENSE_STL_FILE = None
STEP_FILE = None

# Known properties of the fixtures, so the tests can pin values instead of
# asserting "not None". A stub that ignores its input cannot reproduce these.
BOX_VERTS, BOX_FACES, BOX_VOLUME = 8, 12, 1.0
SPHERE_VERTS, SPHERE_FACES = 642, 1280
SPHERE_VOLUME = 4.1527407490072425   # trimesh.creation.icosphere(subdivisions=3)


def setup_test_files():
    global STL_FILE, DENSE_STL_FILE, STEP_FILE
    mesh = trimesh.creation.box()
    f_stl = tempfile.NamedTemporaryFile(suffix='.stl', delete=False)
    f_stl.close()
    STL_FILE = f_stl.name
    mesh.export(STL_FILE)

    # A dense fixture, so the reverse-engineering operations have something to
    # actually operate on (a 12-triangle box cannot be decimated or wrapped).
    sphere = trimesh.creation.icosphere(subdivisions=3)
    f_dense = tempfile.NamedTemporaryFile(suffix='.stl', delete=False)
    f_dense.close()
    DENSE_STL_FILE = f_dense.name
    sphere.export(DENSE_STL_FILE)

    # A deliberately degenerate STEP: header only, EMPTY data section. The
    # tests below pin the documented behaviour for that file; real B-Rep
    # geometry is covered by test_importer_step_real_geometry, which
    # round-trips through the project's own export_step.
    f_stp = tempfile.NamedTemporaryFile(suffix='.stp', delete=False)
    f_stp.write(b"ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\nENDSEC;\nEND-ISO-10303-21;\n")
    f_stp.close()
    STEP_FILE = f_stp.name

setup_test_files()

results = {"PASS": 0, "FAIL": 0}

def run_test(name, func):
    print(f"--- Running Test: {name} ---")
    try:
        start = time.time()
        func()
        end = time.time()
        print(f"PASS: {name} ({end - start:.2f}s)\n")
        results["PASS"] += 1
    except Exception as e:
        print(f"FAIL: {name}")
        traceback.print_exc()
        print()
        results["FAIL"] += 1


def _positions(mesh):
    return np.array([v.position for v in mesh.vertices], dtype=float)


def _sorted_positions(mesh):
    """Vertex coordinates in a comparison-friendly, order-independent form."""
    p = _positions(mesh)
    return p[np.lexsort((p[:, 2], p[:, 1], p[:, 0]))]


# --------------------------------------------------------------------------
# importers
# --------------------------------------------------------------------------

def test_importer_stl_real():
    mesh = import_stl(STL_FILE)
    assert mesh is not None, "Imported STL is None"
    # the fixture is a trimesh unit box -- pin it, so an importer that ignores
    # the path and returns some other mesh cannot pass
    assert len(mesh.vertices) == BOX_VERTS
    assert len(mesh.faces) == BOX_FACES
    tm = mesh.to_trimesh()
    assert tm.is_watertight
    assert tm.volume == pytest.approx(BOX_VOLUME, rel=1e-9)
    assert np.allclose(tm.bounds, [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], atol=1e-9)


def test_importer_stl_nonexistent():
    try:
        import_stl("nonexistent_file_12345.stl")
        assert False, "Should have raised an error for nonexistent file"
    except Exception as e:
        assert "nonexistent_file_12345.stl" in str(e) or isinstance(e, (FileNotFoundError, ValueError, SystemError)), f"Expected specific error, got: {e}"


def test_importer_obj():
    with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as f:
        f.write(b"v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
        temp_obj = f.name

    mesh = import_obj(temp_obj)
    os.remove(temp_obj)
    assert mesh is not None
    assert len(mesh.vertices) == 3
    assert len(mesh.faces) == 1

    want = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    assert np.allclose(_sorted_positions(mesh), want[np.lexsort(
        (want[:, 2], want[:, 1], want[:, 0]))], atol=1e-12), "coordinates were not parsed"

    corner_idx = [v.index for v in mesh.get_face_vertices(mesh.faces[0])]
    assert sorted(corner_idx) == [0, 1, 2]
    corners = np.array([mesh.vertices[i].position for i in corner_idx])
    assert corners.min() == pytest.approx(0.0, abs=1e-12)
    assert corners.max() == pytest.approx(1.0, abs=1e-12)
    assert mesh.to_trimesh().area == pytest.approx(0.5, rel=1e-9)


def test_importer_obj_face_indices_are_one_based():
    """OBJ 'f' indices are 1-BASED; an off-by-one is the classic importer bug.

    A single triangle cannot catch it -- any rotation of {0, 1, 2} names the
    same three vertices. Two disjoint triangles 10 units apart can: a shifted
    index stitches corners from both islands together and the area explodes.
    """
    with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as f:
        f.write(b"v 0 0 0\nv 1 0 0\nv 0 1 0\n"
                b"v 10 0 0\nv 11 0 0\nv 10 1 0\n"
                b"f 1 2 3\nf 4 5 6\n")
        temp_obj = f.name

    mesh = import_obj(temp_obj)
    os.remove(temp_obj)
    assert len(mesh.vertices) == 6
    assert len(mesh.faces) == 2
    assert mesh.to_trimesh().area == pytest.approx(1.0, rel=1e-9), (
        "faces do not consist of the vertices the 1-based indices name")
    for face in mesh.faces:
        pts = np.array([v.position for v in mesh.get_face_vertices(face)])
        span = pts.max(axis=0) - pts.min(axis=0)
        assert np.allclose(span, [1.0, 1.0, 0.0], atol=1e-9), (
            f"a face spans both islands: {pts.tolist()}")


def test_importer_obj_ngon_and_negative_coordinates():
    """A 4-sided face and negative coordinates -- the ngon path plus a sign."""
    with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as f:
        f.write(b"v -1 -1 0\nv 1 -1 0\nv 1 1 0\nv -1 1 0\nf 1 2 3 4\n")
        temp_obj = f.name

    mesh = import_obj(temp_obj)
    os.remove(temp_obj)
    assert len(mesh.vertices) == 4
    assert len(mesh.faces) == 2, "the quad must come back fanned into 2 triangles"
    want = np.array([[-1., -1., 0.], [1., -1., 0.], [1., 1., 0.], [-1., 1., 0.]])
    assert np.allclose(_sorted_positions(mesh),
                       want[np.lexsort((want[:, 2], want[:, 1], want[:, 0]))],
                       atol=1e-12)
    assert mesh.to_trimesh().area == pytest.approx(4.0, rel=1e-9)


def test_importer_step():
    """The fixture STEP has an EMPTY data section.

    There is no geometry to import, so pin the documented degenerate result
    rather than accepting any object at all.
    """
    res = import_step(STEP_FILE)
    assert isinstance(res, dict)
    assert set(res) >= {'shape', 'mesh', 'vertices', 'faces'}
    assert res['shape'] is None, "an empty DATA section cannot yield a shape"
    assert res['mesh'] is None
    assert list(res['faces']) == []
    assert len(np.asarray(res['vertices'])) == 0


def test_importer_step_real_geometry():
    """Round-trip real B-Rep geometry so the importer is exercised on a body."""
    pytest.importorskip("OCP")
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox

    shape = BRepPrimAPI_MakeBox(10.0, 20.0, 30.0).Shape()
    with tempfile.NamedTemporaryFile(suffix='.stp', delete=False) as f:
        temp_step = f.name
    try:
        export_step(shape, temp_step)
        assert os.path.getsize(temp_step) > 0
        res = import_step(temp_step)
    finally:
        if os.path.exists(temp_step):
            os.remove(temp_step)

    assert res['shape'] is not None
    assert res['mesh'] is not None
    verts = np.asarray(res['vertices'])
    assert verts.shape == (8, 3), "a box tessellates to 8 welded corners"
    assert len(res['faces']) == 12
    assert np.allclose(verts.min(axis=0), [0.0, 0.0, 0.0], atol=1e-6)
    assert np.allclose(verts.max(axis=0), [10.0, 20.0, 30.0], atol=1e-6)
    tm = res['mesh'].to_trimesh()
    assert tm.is_watertight
    assert tm.volume == pytest.approx(10.0 * 20.0 * 30.0, rel=1e-6)


# --------------------------------------------------------------------------
# exporters
# --------------------------------------------------------------------------

def test_exporter_stl():
    """Round-trip geometry, not cardinality.

    Asserting only the vertex count passes for any writer that keeps the count
    while mangling the coordinates, winding or connectivity.
    """
    mesh = import_stl(STL_FILE)
    src_pos = _sorted_positions(mesh)
    src_tm = mesh.to_trimesh()

    for binary in (True, False):
        with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as f:
            temp_stl = f.name
        try:
            export_stl(mesh, temp_stl, binary=binary)
            assert os.path.getsize(temp_stl) > 0, f"empty file (binary={binary})"
            mesh2 = import_stl(temp_stl)
        finally:
            os.remove(temp_stl)

        assert len(mesh2.vertices) == len(mesh.vertices), f"binary={binary}"
        assert len(mesh2.faces) == len(mesh.faces), f"binary={binary}"
        assert np.allclose(_sorted_positions(mesh2), src_pos, atol=1e-6), (
            f"coordinates changed on the round trip (binary={binary})")
        tm2 = mesh2.to_trimesh()
        assert tm2.is_watertight, f"binary={binary}"
        assert tm2.volume == pytest.approx(src_tm.volume, rel=1e-6)
        assert np.allclose(tm2.bounds, src_tm.bounds, atol=1e-6)


def test_exporter_obj():
    mesh = import_stl(STL_FILE)
    with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as f:
        temp_obj = f.name
    try:
        export_obj(mesh, temp_obj)
        with open(temp_obj, 'r') as f:
            lines = f.read().splitlines()
        mesh2 = import_obj(temp_obj)
    finally:
        os.remove(temp_obj)

    # count the lines instead of grepping for a substring: 'v ' in content is
    # satisfied by a single literal vertex
    assert sum(1 for l in lines if l.startswith('v ')) == len(mesh.vertices)
    assert sum(1 for l in lines if l.startswith('f ')) == len(mesh.faces)

    assert len(mesh2.vertices) == len(mesh.vertices)
    assert len(mesh2.faces) == len(mesh.faces)
    assert np.allclose(_sorted_positions(mesh2), _sorted_positions(mesh), atol=1e-9)
    assert mesh2.to_trimesh().volume == pytest.approx(mesh.to_trimesh().volume,
                                                      rel=1e-9)


def test_exporter_stl_empty():
    """An empty mesh is REJECTED, not silently written as a 0-face file."""
    mesh = HalfEdgeMesh()
    temp_stl = os.path.join(tempfile.gettempdir(),
                            f"empty_export_{os.getpid()}.stl")
    if os.path.exists(temp_stl):
        os.remove(temp_stl)
    try:
        with pytest.raises(ValueError, match="empty"):
            export_stl(mesh, temp_stl)
        assert not os.path.exists(temp_stl), (
            "a rejected export must not leave a file behind")
    finally:
        if os.path.exists(temp_stl):
            os.remove(temp_stl)


# --------------------------------------------------------------------------
# reverse engineering -- one operation per test, each with a real assertion
# --------------------------------------------------------------------------

def test_re_decimate_reduces_the_face_count():
    mesh = import_stl(DENSE_STL_FILE)
    assert len(mesh.faces) == SPHERE_FACES, "dense fixture changed"
    out = decimate_mesh(mesh, target_faces=200)
    assert len(out.faces) == 200
    assert len(out.vertices) < len(mesh.vertices)
    assert out.to_trimesh().volume == pytest.approx(SPHERE_VOLUME, rel=0.05)


def test_re_smooth_mesh_moves_vertices_and_keeps_the_volume():
    mesh = import_stl(DENSE_STL_FILE)
    before = _positions(mesh)
    out = smooth_mesh(mesh, iterations=1)
    after = _positions(out)
    assert after.shape == before.shape
    step = np.linalg.norm(after - before, axis=1)
    assert step.max() > 1e-4, "smoothing did not move any vertex"
    assert step.max() < 0.05, "smoothing displaced the surface (real max 0.0030)"
    # Taubin is the volume-preserving variant: it stays within 0.1% here,
    # while three plain Laplacian passes shrink the sphere by ~5%
    assert out.to_trimesh().volume == pytest.approx(SPHERE_VOLUME, rel=0.01)
    lap = smooth_mesh(mesh, iterations=3, method='laplacian')
    assert lap.to_trimesh().volume < 0.99 * SPHERE_VOLUME, (
        "the Laplacian control did not shrink -- smoothing is not running")


def test_re_offset_mesh_moves_every_vertex_along_its_normal():
    mesh = import_stl(DENSE_STL_FILE)
    mesh.compute_vertex_normals()
    before = _positions(mesh)
    normals = np.array([v.normal for v in mesh.vertices], dtype=float)

    out = offset_mesh(mesh, distance=0.1)
    after = _positions(out)
    step = after - before
    assert np.allclose(np.linalg.norm(step, axis=1), 0.1, atol=1e-9), (
        "not every vertex moved by exactly `distance`")
    assert np.allclose(np.einsum('ij,ij->i', step, normals), 0.1, atol=1e-9), (
        "the offset was not along the vertex normal")
    # unit sphere offset by 0.1 -> radius 1.1
    assert np.allclose(np.linalg.norm(after, axis=1), 1.1, atol=1e-4)


def test_re_fill_holes_closes_an_injected_hole():
    """Run the repair on a mesh that actually has a defect.

    The old pipeline ran fill_holes over a watertight box: there was no hole,
    so an identity implementation was indistinguishable from a working one.
    """
    mesh = import_stl(STL_FILE)
    arrays = mesh.to_arrays()
    verts = np.asarray(arrays['vertices'])
    faces = [list(f) for f in arrays['faces']]
    broken = HalfEdgeMesh.from_arrays(verts, faces[:-1])   # drop one triangle

    before = compute_mesh_quality(broken)
    assert before['boundary_edges'] == 3, "the defect was not injected"
    assert before['watertight'] is False
    assert before['face_count'] == len(faces) - 1

    out = fill_holes(broken, max_hole_edges=20)
    after = compute_mesh_quality(out)
    assert after['boundary_edges'] == 0, "the hole was left open"
    assert after['watertight'] is True
    assert after['face_count'] == len(faces)
    assert out.to_trimesh().volume == pytest.approx(BOX_VOLUME, rel=1e-9)


def test_re_compute_mesh_quality_reports_real_numbers():
    mesh = import_stl(DENSE_STL_FILE)
    q = compute_mesh_quality(mesh)
    assert isinstance(q, dict), "compute_mesh_quality didn't return a dict"
    # an empty dict used to satisfy this test -- pin the documented contents
    assert q['face_count'] == SPHERE_FACES
    assert q['vertex_count'] == SPHERE_VERTS
    assert q['watertight'] is True
    assert q['boundary_edges'] == 0
    assert q['non_manifold_edges'] == 0
    assert 0 < q['min_angle'] <= q['avg_angle'] <= q['max_angle'] < 180
    assert q['avg_angle'] == pytest.approx(60.0, abs=1.0)   # near-equilateral
    assert 0 < q['min_area'] <= q['max_area']


def test_re_quad_wrap_produces_a_quad_cage_on_the_reference():
    mesh = import_stl(DENSE_STL_FILE)
    ref_tm = mesh.to_trimesh()
    cage = QuadWrapper(target_face_count=200, smoothing_weight=0.1).wrap(mesh)

    assert len(cage.faces) > 0
    assert all(len(cage.get_face_vertices(f)) == 4 for f in cage.faces), (
        "the result is not a pure quad cage")
    assert 0.7 * 200 <= len(cage.faces) <= 1.4 * 200, (
        f"{len(cage.faces)} quads for a target of 200")
    assert len(cage.vertices) < len(mesh.vertices), "no simplification happened"

    pts = _positions(cage)
    _, dist, _ = trimesh.proximity.closest_point(ref_tm, pts)
    assert dist.max() < 0.02, f"cage sits {dist.max():.4f} off the reference"


def test_re_shrink_wrap_pulls_the_cage_onto_the_reference():
    mesh = import_stl(DENSE_STL_FILE)          # unit sphere
    cage = create_box(width=3, height=3, depth=3)
    start = _positions(cage)
    assert np.linalg.norm(start, axis=1).min() > 2.0, (
        "the cage has to start well off the reference surface")

    out = ShrinkWrapper(iterations=2).wrap(cage, mesh)
    assert len(out.faces) == len(cage.faces)
    pts = _positions(out)
    assert not np.allclose(pts, start), "the cage was not moved at all"
    _, dist, _ = trimesh.proximity.closest_point(mesh.to_trimesh(), pts)
    assert dist.max() < 1e-4, "cage vertices did not land on the reference"
    assert np.abs(np.linalg.norm(pts, axis=1) - 1.0).max() < 0.02


# --------------------------------------------------------------------------
# operations
# --------------------------------------------------------------------------

def test_operations_shell():
    box = create_box(width=1, height=1, depth=1)
    shelled = shell_solid(box, thickness=0.1)
    assert shelled is not None
    tm = shelled.to_trimesh()
    # shell_solid returns the OFFSET SURFACE and defaults to direction='inward',
    # so a 1x1x1 box comes back as a closed 0.8 box, marching-cubes tessellated.
    assert len(shelled.faces) > len(box.faces), "no isosurface was extracted"
    assert tm.is_watertight
    assert np.allclose(tm.extents, [0.8, 0.8, 0.8], atol=0.02), (
        f"body was not offset inward: extents {tm.extents}")
    assert tm.volume == pytest.approx(0.8 ** 3, rel=0.02)


def test_operations_thicken():
    plane = create_plane(width=1, height=1)
    assert not plane.to_trimesh().is_watertight, "the input must be an open sheet"
    thickened = thicken_surface(plane, thickness=0.1)
    assert thickened is not None
    tm = thickened.to_trimesh()
    # an open sheet must come back as a closed slab: returning the input
    # unchanged cannot be watertight
    assert tm.is_watertight
    assert len(thickened.faces) > len(plane.faces)
    # create_plane lies in the XZ plane, so the wall grows along y
    assert tm.extents[1] == pytest.approx(0.1, abs=0.01)
    assert tm.volume == pytest.approx(1.0 * 1.0 * 0.1, rel=0.2)


def test_operations_zero_thickness():
    """A zero shell is a zero offset: the body comes back at its original size
    rather than raising or collapsing."""
    box = create_box(width=1, height=1, depth=1)
    shelled = shell_solid(box, thickness=0.0)
    tm = shelled.to_trimesh()
    assert len(shelled.faces) > 0
    assert tm.is_watertight
    assert np.allclose(tm.extents, [1.0, 1.0, 1.0], atol=0.02)
    assert tm.volume == pytest.approx(1.0, rel=0.02)


def test_operations_empty():
    """An empty mesh is passed straight through, not crashed on."""
    mesh = HalfEdgeMesh()
    out = shell_solid(mesh, thickness=0.1)
    assert isinstance(out, HalfEdgeMesh)
    assert len(out.vertices) == 0
    assert len(out.faces) == 0


# --------------------------------------------------------------------------
# NURBS conversion
# --------------------------------------------------------------------------

def test_nurbs_converter():
    box = create_box(subdivisions=1)
    quad_faces = [f for f in box.faces if len(box.get_face_vertices(f)) == 4]
    assert len(quad_faces) == 24, "one Catmull-Clark level on a cube gives 24 quads"

    converter = SubDToNURBSConverter()
    res = converter.convert(box)
    assert isinstance(res, dict)
    assert set(res) >= {'patches', 'shape', 'mesh'}
    assert res['mesh'] is box

    # an empty stub with the right key names used to pass -- assert the contents
    assert len(res['patches']) == len(quad_faces), "one 6x6 patch per quad face"
    for patch in res['patches']:
        arr = np.asarray(patch)
        assert arr.shape == (6, 6, 3), f"unexpected patch shape {arr.shape}"
        assert np.isfinite(arr).all()

    # and assert they sit ON the cage, so a converter emitting well-shaped
    # patches in the wrong place is caught
    ctrl = np.concatenate([np.asarray(p).reshape(-1, 3) for p in res['patches']])
    cage = _positions(box)
    assert np.allclose(ctrl.min(axis=0), cage.min(axis=0), atol=0.15)
    assert np.allclose(ctrl.max(axis=0), cage.max(axis=0), atol=0.15)


def test_nurbs_converter_builds_a_brep():
    pytest.importorskip("OCP")
    box = create_box(subdivisions=1)
    res = SubDToNURBSConverter().convert(box)
    assert res['shape'] is not None, "OCP is installed, so a B-Rep must be built"
    assert not res['shape'].IsNull()


if __name__ == '__main__':
    tests = [
        ("importer_stl_real", test_importer_stl_real),
        ("importer_stl_nonexistent", test_importer_stl_nonexistent),
        ("importer_obj", test_importer_obj),
        ("importer_obj_one_based", test_importer_obj_face_indices_are_one_based),
        ("importer_obj_ngon", test_importer_obj_ngon_and_negative_coordinates),
        ("importer_step", test_importer_step),
        ("importer_step_real_geometry", test_importer_step_real_geometry),
        ("exporter_stl", test_exporter_stl),
        ("exporter_obj", test_exporter_obj),
        ("exporter_stl_empty", test_exporter_stl_empty),
        ("re_decimate", test_re_decimate_reduces_the_face_count),
        ("re_smooth", test_re_smooth_mesh_moves_vertices_and_keeps_the_volume),
        ("re_offset", test_re_offset_mesh_moves_every_vertex_along_its_normal),
        ("re_fill_holes", test_re_fill_holes_closes_an_injected_hole),
        ("re_mesh_quality", test_re_compute_mesh_quality_reports_real_numbers),
        ("re_quad_wrap", test_re_quad_wrap_produces_a_quad_cage_on_the_reference),
        ("re_shrink_wrap", test_re_shrink_wrap_pulls_the_cage_onto_the_reference),
        ("operations_shell", test_operations_shell),
        ("operations_thicken", test_operations_thicken),
        ("operations_zero_thickness", test_operations_zero_thickness),
        ("operations_empty", test_operations_empty),
        ("nurbs_converter", test_nurbs_converter),
        ("nurbs_converter_brep", test_nurbs_converter_builds_a_brep),
    ]

    for name, func in tests:
        run_test(name, func)

    print("=== SUMMARY ===")
    print(f"PASS: {results['PASS']}")
    print(f"FAIL: {results['FAIL']}")
