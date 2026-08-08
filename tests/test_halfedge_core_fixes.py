
import os
import sys

import numpy as np
import pytest

# Ensure src is in the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.halfedge_mesh import HalfEdgeMesh


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def assert_half_edge_invariants(mesh: HalfEdgeMesh):
    """The invariants that must hold for *any* input, manifold or not."""
    # Without this the whole helper is a loop over an empty list: an add_face
    # that silently builds nothing would satisfy every caller vacuously.
    assert mesh.half_edges, "invariant check ran on an empty mesh"
    for he in mesh.half_edges:
        assert he.twin is not he, f"HalfEdge {he.index} is its own twin"
        assert he.twin is None or he.twin.twin is he, (
            f"Broken twin involution at half-edge {he.index}: "
            f"twin={he.twin.index}, twin.twin="
            f"{he.twin.twin.index if he.twin.twin else None}"
        )
        assert he.edge is not None, f"HalfEdge {he.index} has no edge record"
        assert he.edge.half_edge in (he, he.twin), (
            f"Edge {he.edge.index} does not point back at half-edge {he.index}"
        )
        if he.twin is not None:
            assert he.edge is he.twin.edge, (
                f"Twinned half-edges {he.index}/{he.twin.index} own different edges"
            )


def make_quad_grid(nx: int, ny: int) -> HalfEdgeMesh:
    """(nx x ny) quads in the XY plane, wound CCW so every normal is +Z."""
    mesh = HalfEdgeMesh()
    for j in range(ny + 1):
        for i in range(nx + 1):
            mesh.add_vertex([float(i), float(j), 0.0])

    def idx(i, j):
        return j * (nx + 1) + i

    for j in range(ny):
        for i in range(nx):
            mesh.add_face([idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)])
    return mesh


def make_quad_tube(segments: int = 6) -> HalfEdgeMesh:
    """Closed band of quads (open at both ends) - the vertical edges form a
    closed edge loop."""
    mesh = HalfEdgeMesh()
    for z in (0.0, 1.0):
        for i in range(segments):
            a = 2.0 * np.pi * i / segments
            mesh.add_vertex([np.cos(a), np.sin(a), z])
    for i in range(segments):
        j = (i + 1) % segments
        mesh.add_face([i, j, j + segments, i + segments])
    return mesh


def edge_endpoints(edge):
    he = edge.half_edge
    return frozenset((he.vertex.index, he.prev.vertex.index))


def find_edge(mesh: HalfEdgeMesh, a: int, b: int):
    target = frozenset((a, b))
    for e in mesh.edges:
        if edge_endpoints(e) == target:
            return e
    return None


def edge_face_indices(mesh: HalfEdgeMesh, edge):
    return {f.index for f in mesh.get_edge_faces(edge) if f is not None}


# ---------------------------------------------------------------------------
# Bug 1 - add_face must never steal an already paired twin
# ---------------------------------------------------------------------------

def test_repeated_directed_edge_keeps_twin_involution():
    """A third face reusing the directed edge (1,0) must not re-pair the
    half-edge that already twins with it."""
    mesh = HalfEdgeMesh()
    for p in [(0, 0, 0), (1, 0, 0), (0.5, 1, 0), (0.5, -1, 0), (0.5, -1, 1)]:
        mesh.add_vertex(p)

    f0 = mesh.add_face([0, 1, 2])
    f1 = mesh.add_face([1, 0, 3])   # legitimately twins (1,0) against (0,1)
    f2 = mesh.add_face([1, 0, 4])   # repeats the directed edge (1,0)

    assert f0 is not None and f1 is not None and f2 is not None
    assert_half_edge_invariants(mesh)

    # The first pairing wins: (0,1) stays twinned with the (1,0) of face 1.
    he_01 = mesh._he_dict[(0, 1)]
    assert he_01.twin is not None
    assert he_01.twin.face is f1

    # The duplicate directed edge of face 2 gets its own (boundary) edge.
    dup = [he for he in mesh.half_edges if he.face is f2 and he.vertex.index == 0]
    assert len(dup) == 1
    assert dup[0].twin is None
    assert dup[0].edge is not None and dup[0].edge.half_edge is dup[0]


def test_duplicate_face_does_not_corrupt_topology():
    """process=True style inputs can hand us the very same face twice."""
    mesh = HalfEdgeMesh()
    for p in [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)]:
        mesh.add_vertex(p)
    mesh.add_face([0, 1, 2])
    mesh.add_face([1, 3, 2])

    # Precondition: the two legitimate faces really do share one twinned edge,
    # so the duplicates below have something they could corrupt.
    he_12 = mesh._he_dict[(1, 2)]
    he_21 = mesh._he_dict[(2, 1)]
    assert he_12.twin is he_21 and he_21.twin is he_12
    shared_edge = he_12.edge
    assert he_21.edge is shared_edge
    n_he, n_edges, n_faces = len(mesh.half_edges), len(mesh.edges), len(mesh.faces)
    assert (n_he, n_edges, n_faces) == (6, 5, 2)

    mesh.add_face([0, 1, 2])   # exact duplicate
    mesh.add_face([2, 0, 1])   # rotated duplicate, same directed edges

    assert_half_edge_invariants(mesh)

    # The duplicates must get their own (boundary) half-edges and edges instead
    # of stealing the pairing the first two faces established.
    assert he_12.twin is he_21 and he_21.twin is he_12
    assert he_12.edge is shared_edge and he_21.edge is shared_edge
    assert len(mesh.faces) == n_faces + 2
    assert len(mesh.half_edges) == n_he + 6
    assert len(mesh.edges) == n_edges + 6
    # Still exactly one twin pair - every duplicate half-edge stays unpaired.
    assert sum(1 for he in mesh.half_edges if he.twin is not None) == 2


def test_bowtie_fan_keeps_twin_involution():
    """Four triangles all hanging off the same edge {0, 1}.

    The winding alternates on purpose: with all four faces wound the same way
    the directed edge (1,0) is never produced, nothing can ever pair, and every
    twin assertion below would be vacuously true.
    """
    mesh = HalfEdgeMesh()
    mesh.add_vertex([0, 0, 0])
    mesh.add_vertex([1, 0, 0])
    for k in range(4):
        mesh.add_vertex([0.5, np.cos(k), np.sin(k)])
    for k in range(4):
        mesh.add_face([0, 1, 2 + k] if k % 2 == 0 else [1, 0, 2 + k])

    # Preconditions: the fan was really built, and it really does contain the
    # twin pairing whose involution the invariants below check.
    assert len(mesh.faces) == 4
    assert len(mesh.half_edges) == 12
    twinned = [he for he in mesh.half_edges if he.twin is not None]
    assert len(twinned) >= 2, "fan built no twin pairs - nothing to test"

    assert_half_edge_invariants(mesh)

    # The first pairing of the shared edge wins; the two later faces reusing the
    # same directed edges get boundary half-edges of their own.
    he_01 = mesh._he_dict[(0, 1)]
    he_10 = mesh._he_dict[(1, 0)]
    assert he_01.twin is he_10 and he_10.twin is he_01
    assert he_01.edge is he_10.edge
    assert len(mesh.edges) == 11


# ---------------------------------------------------------------------------
# Bug 2 - two-sided fan walk at boundary vertices
# ---------------------------------------------------------------------------

def test_boundary_vertex_fan_walks_both_directions():
    """vertex.half_edge sits in the middle of an open fan, so a one-sided walk
    silently drops the faces on the other side."""
    mesh = HalfEdgeMesh()
    mesh.add_vertex([0.0, 0.0, 0.0])     # 0 - fan centre, on the boundary
    mesh.add_vertex([1.0, 0.0, 0.0])     # 1
    mesh.add_vertex([0.5, 1.0, 0.0])     # 2
    mesh.add_vertex([-0.5, 1.0, 0.0])    # 3
    mesh.add_vertex([-1.0, 0.0, 0.0])    # 4

    # The middle face is added first, so vertex 0 stores the middle spoke.
    f_mid = mesh.add_face([0, 2, 3])
    f_right = mesh.add_face([0, 1, 2])
    f_left = mesh.add_face([0, 3, 4])

    v0 = mesh.vertices[0]
    assert mesh.is_boundary_vertex(v0)
    assert v0.half_edge.face is f_mid

    faces = mesh.get_vertex_faces(v0)
    assert {f.index for f in faces} == {f_mid.index, f_right.index, f_left.index}
    assert len(faces) == 3

    # The fan must come back in contiguous ring order.
    ordered = [f.index for f in faces]
    assert ordered in ([f_left.index, f_mid.index, f_right.index],
                       [f_right.index, f_mid.index, f_left.index])

    neighbours = {n.index for n in mesh.get_vertex_neighbors(v0)}
    # Vertex 4 is reachable only through the half-edge 4->0; this structure has
    # no boundary half-edges, so vertex 0 owns no outgoing spoke towards it and
    # an outgoing-fan walk cannot report it.
    assert neighbours == {1, 2, 3}
    # Asserting the number, not `len(get_vertex_neighbors(...))` - vertex_valence
    # *is* that len(), so comparing the two can never fail.
    assert mesh.vertex_valence(v0) == 3
    # The two-sided walk must not hand the same spoke back twice.
    fan = mesh.get_vertex_fan(v0)
    assert len({he.index for he in fan}) == len(fan) == 3


def test_boundary_vertex_normal_uses_whole_fan():
    """compute_vertex_normals averages get_vertex_faces, so a truncated fan
    biases the normal of every boundary vertex."""
    mesh = HalfEdgeMesh()
    mesh.add_vertex([0.0, 0.0, 0.0])
    mesh.add_vertex([1.0, 0.0, 0.0])
    mesh.add_vertex([0.5, 1.0, 0.0])
    mesh.add_vertex([-0.5, 1.0, 0.0])
    mesh.add_vertex([-1.0, 0.0, 1.0])   # tilts the left face out of the plane
    mesh.add_face([0, 2, 3])
    mesh.add_face([0, 1, 2])
    mesh.add_face([0, 3, 4])

    mesh.compute_vertex_normals()
    # Every face of this mesh touches vertex 0, so its normal is the average of
    # all three face normals - a truncated fan gives a visibly different answer.
    expected = np.sum([f.normal for f in mesh.faces], axis=0)
    expected = expected / np.linalg.norm(expected)
    assert np.allclose(mesh.vertices[0].normal, expected, atol=1e-9)


def test_interior_vertex_fan_unchanged():
    """Interior vertices must behave exactly as before."""
    mesh = make_quad_grid(2, 2)
    centre = mesh.vertices[4]
    assert not mesh.is_boundary_vertex(centre)
    assert len(mesh.get_vertex_faces(centre)) == 4
    assert len(mesh.get_vertex_neighbors(centre)) == 4
    assert mesh.vertex_valence(centre) == 4
    assert {n.index for n in mesh.get_vertex_neighbors(centre)} == {1, 3, 5, 7}


# ---------------------------------------------------------------------------
# Bug 3 - Newell's method for face normals
# ---------------------------------------------------------------------------

def test_face_normal_of_non_convex_quad_with_reflex_first_vertex():
    """A CCW dart whose reflex corner is vertex 0: the v0/v1/v[-1] triangle
    points the wrong way, Newell's method does not."""
    mesh = HalfEdgeMesh()
    mesh.add_vertex([0.0, 1.0, 0.0])    # 0 - reflex notch
    mesh.add_vertex([2.0, 0.0, 0.0])    # 1 - right wing
    mesh.add_vertex([0.0, 3.0, 0.0])    # 2 - tip
    mesh.add_vertex([-2.0, 0.0, 0.0])   # 3 - left wing
    face = mesh.add_face([0, 1, 2, 3])

    mesh.compute_face_normals()
    assert np.allclose(face.normal, [0.0, 0.0, 1.0], atol=1e-9), (
        f"Non-convex quad normal flipped: {face.normal}"
    )


def test_face_normals_of_convex_polygons_unchanged():
    mesh = HalfEdgeMesh()
    mesh.add_vertex([0.0, 0.0, 0.0])
    mesh.add_vertex([1.0, 0.0, 0.0])
    mesh.add_vertex([1.0, 1.0, 0.0])
    mesh.add_vertex([0.0, 1.0, 0.0])
    tri = mesh.add_face([0, 1, 2])
    quad = mesh.add_face([0, 2, 3])
    mesh.compute_face_normals()
    assert np.allclose(tri.normal, [0.0, 0.0, 1.0], atol=1e-12)
    assert np.allclose(quad.normal, [0.0, 0.0, 1.0], atol=1e-12)


def test_face_normal_is_translation_invariant():
    """Newell must not lose the sign when the polygon sits far from the origin."""
    offset = np.array([1000.0, -500.0, 250.0])
    normals = []
    for shift in (np.zeros(3), offset):
        mesh = HalfEdgeMesh()
        for p in ([0.0, 1.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [-2.0, 0.0, 0.0]):
            mesh.add_vertex(np.array(p) + shift)
        f = mesh.add_face([0, 1, 2, 3])
        mesh.compute_face_normals()
        normals.append(f.normal.copy())

    # The absolute answer, at BOTH offsets. Comparing the two against each other
    # alone proves nothing: a no-op (both stay zero) and the pre-fix corner
    # triangle (both come out as [0, 0, -1]) are translation invariant too.
    for shift, n in zip(("origin", "far away"), normals):
        assert np.linalg.norm(n) == pytest.approx(1.0, abs=1e-9), f"{shift}: {n}"
        assert np.allclose(n, [0.0, 0.0, 1.0], atol=1e-9), f"{shift}: {n}"
    assert np.allclose(normals[0], normals[1], atol=1e-9)


# ---------------------------------------------------------------------------
# Bug 4 - faces with repeated consecutive vertex indices
# ---------------------------------------------------------------------------

def test_add_face_rejects_degenerate_vertex_list():
    mesh = HalfEdgeMesh()
    for p in [(0, 0, 0), (1, 0, 0), (0, 1, 0)]:
        mesh.add_vertex(p)

    assert mesh.add_face([0, 0, 1]) is None
    assert mesh.add_face([0, 0, 0]) is None
    assert mesh.add_face([0, 1]) is None
    assert mesh.faces == []
    assert mesh.half_edges == []
    assert mesh.edges == []

    # Positive control: rejecting everything (an add_face that silently drops
    # every face) must not pass this test.
    face = mesh.add_face([0, 1, 2])
    assert face is not None
    assert len(mesh.faces) == 1
    assert [v.index for v in mesh.get_face_vertices(face)] == [0, 1, 2]
    assert len(mesh.half_edges) == 3
    assert len(mesh.edges) == 3


def test_add_face_drops_repeated_consecutive_indices():
    mesh = HalfEdgeMesh()
    for p in [(0, 0, 0), (1, 0, 0), (0, 1, 0)]:
        mesh.add_vertex(p)

    face = mesh.add_face([0, 1, 1, 2, 2])
    assert face is not None
    assert [v.index for v in mesh.get_face_vertices(face)] == [0, 1, 2]
    assert len(mesh.half_edges) == 3
    assert_half_edge_invariants(mesh)


def test_add_face_drops_wrapped_duplicate_index():
    mesh = HalfEdgeMesh()
    for p in [(0, 0, 0), (1, 0, 0), (0, 1, 0)]:
        mesh.add_vertex(p)

    face = mesh.add_face([0, 1, 2, 0])
    assert face is not None
    assert [v.index for v in mesh.get_face_vertices(face)] == [0, 1, 2]
    assert len(mesh.half_edges) == 3
    assert_half_edge_invariants(mesh)


def test_from_arrays_survives_degenerate_faces():
    verts = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
    faces = [[0, 1, 2], [1, 1, 3], [1, 3, 2]]
    mesh = HalfEdgeMesh.from_arrays(verts, faces)
    assert len(mesh.faces) == 2
    assert [f.index for f in mesh.faces] == [0, 1]
    assert_half_edge_invariants(mesh)
    mesh.compute_face_normals()
    mesh.compute_vertex_normals()


# ---------------------------------------------------------------------------
# Bug 5 - get_edge_loop on a closed quad ring
# ---------------------------------------------------------------------------

def test_edge_loop_on_closed_quad_ring_has_no_duplicates():
    segments = 6
    mesh = make_quad_tube(segments)
    start = find_edge(mesh, 0, segments)
    assert start is not None

    loop = mesh.get_edge_loop(start)

    assert len(loop) == segments, f"expected {segments} edges, got {len(loop)}"
    assert len({e.index for e in loop}) == segments, "edge loop contains duplicates"

    expected = {frozenset((i, i + segments)) for i in range(segments)}
    assert {edge_endpoints(e) for e in loop} == expected

    # ring order: consecutive entries (cyclically) share a quad
    for k in range(segments):
        a = loop[k]
        b = loop[(k + 1) % segments]
        assert edge_face_indices(mesh, a) & edge_face_indices(mesh, b), (
            f"loop entries {k} and {(k + 1) % segments} are not adjacent"
        )


def test_edge_loop_on_open_quad_strip():
    mesh = make_quad_grid(4, 1)
    # vertical edge between (1,0) and (1,1) -> grid indices 1 and 6
    start = find_edge(mesh, 1, 6)
    assert start is not None

    loop = mesh.get_edge_loop(start)
    assert len({e.index for e in loop}) == len(loop)
    expected = {frozenset((i, i + 5)) for i in range(5)}
    assert {edge_endpoints(e) for e in loop} == expected
    for k in range(len(loop) - 1):
        assert edge_face_indices(mesh, loop[k]) & edge_face_indices(mesh, loop[k + 1])


# ---------------------------------------------------------------------------
# Bug 6 - expand_selection_by_angle must not reuse stale normals
# ---------------------------------------------------------------------------

def test_expand_selection_by_angle_recomputes_normals_after_vertex_move():
    mesh = make_quad_grid(2, 1)
    # faces: 0 -> [0,1,4,3], 1 -> [1,2,5,4]
    assert len(mesh.faces) == 2

    first = set(mesh.expand_selection_by_angle([0], 30.0))
    assert first == {0, 1}, "coplanar neighbours should be picked up"

    # Fold the second quad steeply upwards (~84 degrees).
    mesh.vertices[2].position = np.array([2.0, 0.0, 10.0])
    mesh.vertices[5].position = np.array([2.0, 1.0, 10.0])

    second = set(mesh.expand_selection_by_angle([0], 30.0))
    assert second == {0}, f"stale normals reused, got {sorted(second)}"

    wide = set(mesh.expand_selection_by_angle([0], 89.0))
    assert wide == {0, 1}


# ---------------------------------------------------------------------------
# Bug 7 - to_pyvista on face-less meshes
# ---------------------------------------------------------------------------

def test_to_pyvista_point_cloud_without_faces():
    pv = pytest.importorskip("pyvista")
    mesh = HalfEdgeMesh()
    for p in [(0, 0, 0), (1, 0, 0), (0, 1, 0)]:
        mesh.add_vertex(p)

    poly = mesh.to_pyvista()
    assert isinstance(poly, pv.PolyData)
    assert poly.n_points == 3
    assert len(np.asarray(poly.faces)) == 0
    assert np.allclose(poly.points[1], [1.0, 0.0, 0.0])


def test_to_pyvista_completely_empty_mesh():
    pv = pytest.importorskip("pyvista")
    poly = HalfEdgeMesh().to_pyvista()
    assert isinstance(poly, pv.PolyData)
    assert poly.n_points == 0
    assert len(np.asarray(poly.faces)) == 0


def test_to_pyvista_with_faces_still_works():
    pv = pytest.importorskip("pyvista")
    mesh = make_quad_grid(2, 2)
    poly = mesh.to_pyvista()
    assert poly.n_points == 9
    assert poly.n_cells == 4

    # Counts alone cannot tell a correct export from four degenerate cells, so
    # check the connectivity pyvista actually received. Faces are packed as
    # [n, i0, .., in-1, n, ...].
    packed = [int(i) for i in np.asarray(poly.faces)]
    cells = []
    pos = 0
    while pos < len(packed):
        n = packed[pos]
        cells.append(packed[pos + 1:pos + 1 + n])
        pos += n + 1
    assert cells == [[v.index for v in mesh.get_face_vertices(f)] for f in mesh.faces]
    assert cells == [[0, 1, 4, 3], [1, 2, 5, 4], [3, 4, 7, 6], [4, 5, 8, 7]]
    # ...and that the point array carries coordinates, not just a length.
    assert np.allclose(poly.points[8], [2.0, 2.0, 0.0])
    assert np.allclose(poly.points[0], [0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# mesh elements must be weak-referenceable
# ---------------------------------------------------------------------------

def test_mesh_elements_are_weak_referenceable():
    """Vertex/Face/Edge/HalfEdge use __slots__, which suppresses __weakref__
    unless it is listed explicitly. Without it a leak audit cannot observe an
    element at all: any handle it keeps is a strong one that prevents the very
    collection it is trying to measure."""
    import weakref

    mesh = make_quad_grid(3, 3)
    refs = {
        "vertex": weakref.ref(mesh.vertices[0]),
        "face": weakref.ref(mesh.faces[0]),
        "edge": weakref.ref(mesh.edges[0]),
        "half_edge": weakref.ref(mesh.half_edges[0]),
    }
    for name, ref in refs.items():
        assert ref() is not None, f"{name} weakref is already dead"
    assert refs["vertex"]() is mesh.vertices[0]
    assert refs["face"]() is mesh.faces[0]
    assert refs["edge"]() is mesh.edges[0]
    assert refs["half_edge"]() is mesh.half_edges[0]


def test_mesh_elements_die_with_the_mesh():
    """The point of the weakrefs: they must actually go None once the mesh is
    dropped, otherwise the audit reports a leak that is not there (or misses
    one that is)."""
    import gc
    import weakref

    mesh = make_quad_grid(3, 3)
    refs = [weakref.ref(mesh.vertices[0]), weakref.ref(mesh.faces[0]),
            weakref.ref(mesh.edges[0]), weakref.ref(mesh.half_edges[0])]

    del mesh
    gc.collect()

    assert [r() for r in refs] == [None, None, None, None], (
        "mesh elements outlived the mesh -- the half-edge graph is a reference "
        "cycle, so this needs a gc pass, not just refcounting")
