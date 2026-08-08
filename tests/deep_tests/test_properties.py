import contextlib
import io
from collections import defaultdict

import numpy as np
import pytest
import trimesh
import trimesh.creation as creation
from hypothesis import given, settings, strategies as st
from hypothesis.extra.numpy import arrays

from src.core.halfedge_mesh import HalfEdgeMesh
from src.reverse_engineering.quad_wrap import QuadWrapper


# --- helpers -----------------------------------------------------------------

def face_edge_map(mesh):
    """Undirected vertex pair -> indices of the faces using it.

    Deliberately does NOT call is_boundary_edge / get_edge_faces, so it can be
    used as the independent ground truth those predicates are checked against.
    """
    incidence = defaultdict(list)
    for face_index, face in enumerate(mesh.to_arrays()['faces']):
        n = len(face)
        for i in range(n):
            incidence[tuple(sorted((face[i], face[(i + 1) % n])))].append(face_index)
    return dict(incidence)


def edge_key(edge):
    """The undirected vertex-index pair an Edge record spans."""
    return tuple(sorted((edge.half_edge.prev.vertex.index, edge.half_edge.vertex.index)))


def pointer_signature(mesh):
    """Index-level fingerprint of the whole half-edge graph.

    Two meshes with equal signatures have isomorphic connectivity, which is what
    ``copy()`` must reproduce.
    """
    def idx(obj):
        return None if obj is None else obj.index

    return (
        [tuple(v.position) for v in mesh.vertices],
        [idx(v.half_edge) for v in mesh.vertices],
        [idx(f.half_edge) for f in mesh.faces],
        [idx(e.half_edge) for e in mesh.edges],
        [(idx(he.vertex), idx(he.face), idx(he.edge),
          idx(he.next), idx(he.prev), idx(he.twin)) for he in mesh.half_edges],
    )


def polygon_area(positions):
    """Newell area of a polygon, computed here so it is independent of the
    mesh's own normal code."""
    P = np.asarray(positions, dtype=np.float64)
    Q = np.roll(P, -1, axis=0)
    d, s = P - Q, P + Q
    n = np.array([np.dot(d[:, 1], s[:, 2]),
                  np.dot(d[:, 2], s[:, 0]),
                  np.dot(d[:, 0], s[:, 1])])
    return float(np.linalg.norm(n)) / 2.0


# Strategy for generating random degenerate or valid meshes (vertices and triangle faces)
@st.composite
def random_trimesh_data(draw, max_vertices=50, max_faces=100):
    num_vertices = draw(st.integers(min_value=3, max_value=max_vertices))
    # Allow finite floats
    vertices = draw(arrays(
        dtype=float,
        shape=(num_vertices, 3),
        elements=st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False)
    ))

    # Each face gets three DISTINCT vertex indices. Drawing the indices freely
    # (the previous strategy) made hypothesis shrink almost every face to
    # [0, 0, 0], which add_face rejects outright: 81% of the generated examples
    # ended up with ZERO faces, so every downstream assertion looped over an
    # empty list. The topology is still random and freely non-manifold
    # (duplicated faces, >2 faces per edge, unreferenced and coincident
    # vertices) -- only the trivially-empty corpus is gone.
    faces = draw(st.lists(
        st.lists(st.integers(min_value=0, max_value=num_vertices - 1),
                 min_size=3, max_size=3, unique=True),
        min_size=1, max_size=max_faces,
    ))

    return vertices, np.array(faces, dtype=int)

# Strategy for valid closed manifold meshes
@st.composite
def valid_manifold_meshes(draw):
    mesh_type = draw(st.sampled_from(['box', 'icosahedron', 'icosphere']))
    if mesh_type == 'box':
        mesh = creation.box()
    elif mesh_type == 'icosahedron':
        mesh = creation.icosahedron()
    elif mesh_type == 'icosphere':
        subdivisions = draw(st.integers(min_value=1, max_value=3))
        mesh = creation.icosphere(subdivisions=subdivisions)
    return mesh


@st.composite
def open_boundary_meshes(draw):
    """A triangulated rectangular patch: a topological disk with a real rim.

    The closed meshes above can never make ``is_boundary_edge`` /
    ``is_boundary_vertex`` return True, so a constant-False implementation of
    both passes every assertion about them. These meshes exercise the other
    branch.
    """
    nx = draw(st.integers(min_value=1, max_value=4))
    ny = draw(st.integers(min_value=1, max_value=4))
    idx = {}
    vertices = []
    for i in range(nx + 1):
        for j in range(ny + 1):
            idx[(i, j)] = len(vertices)
            vertices.append([float(i), float(j), 0.0])
    faces = []
    for i in range(nx):
        for j in range(ny):
            a, b, c, d = idx[(i, j)], idx[(i + 1, j)], idx[(i + 1, j + 1)], idx[(i, j + 1)]
            faces.append([a, b, c])
            faces.append([a, c, d])
    return np.array(vertices, dtype=float), faces


@settings(deadline=None)
@given(random_trimesh_data())
def test_halfedge_mesh_random_topology_no_crash(mesh_data):
    vertices, faces = mesh_data

    # Check that from_arrays does not crash with degenerate meshes (e.g. out of bounds, non-manifold)
    mesh = HalfEdgeMesh.from_arrays(vertices, faces.tolist())
    assert isinstance(mesh, HalfEdgeMesh)

    # every face of the strategy has three distinct indices, so none may be
    # dropped as degenerate -- this pins the corpus down to non-empty meshes
    assert len(mesh.vertices) == len(vertices)
    assert len(mesh.faces) == len(faces) > 0
    assert len(mesh.half_edges) == 3 * len(faces)

    # Check that basic functions don't crash *and* produce usable normals
    mesh.compute_face_normals()
    mesh.compute_vertex_normals()
    for f in mesh.faces:
        n = f.normal
        assert np.isfinite(n).all(), "non-finite face normal"
        area = polygon_area([v.position for v in mesh.get_face_vertices(f)])
        if area > 1e-6:
            assert abs(np.linalg.norm(n) - 1.0) < 1e-9, \
                f"face with area {area} has non-unit normal {n}"
        else:
            # a collapsed face legitimately has no normal
            assert np.linalg.norm(n) < 1e-9 or abs(np.linalg.norm(n) - 1.0) < 1e-9
    for v in mesh.vertices:
        assert np.isfinite(v.normal).all(), "non-finite vertex normal"
        length = np.linalg.norm(v.normal)
        assert length < 1e-9 or abs(length - 1.0) < 1e-9, \
            f"vertex normal is neither zero nor unit: {v.normal}"

    mesh_copy = mesh.copy()
    assert len(mesh_copy.vertices) == len(mesh.vertices)
    assert len(mesh_copy.faces) == len(mesh.faces)
    assert len(mesh_copy.edges) == len(mesh.edges)
    assert len(mesh_copy.half_edges) == len(mesh.half_edges)

    # copy() must be a DEEP copy: equal element counts are also what an
    # aliasing `return self` produces.
    assert mesh_copy is not mesh
    assert mesh_copy.vertices[0] is not mesh.vertices[0]
    assert mesh_copy.faces[0] is not mesh.faces[0]
    assert mesh_copy.half_edges[0] is not mesh.half_edges[0]
    if mesh.edges:
        assert mesh_copy.edges[0] is not mesh.edges[0]

    original_position = mesh.vertices[0].position.copy()
    mesh_copy.vertices[0].position = original_position + 12345.0
    assert np.array_equal(mesh.vertices[0].position, original_position), \
        "mutating the copy changed the original"
    mesh_copy.vertices[0].position = original_position

    # ... and the copied half-edge graph must be isomorphic to the original
    assert pointer_signature(mesh_copy) == pointer_signature(mesh), \
        "copy() did not reproduce the half-edge connectivity"

# The full quad-wrap pipeline (decimation + shrink wrap + repair) runs on every
# example now that the corpus really contains faces, so the example budget is
# capped to keep the test in the seconds range.
@settings(deadline=None, max_examples=50)
@given(random_trimesh_data())
def test_quad_wrapper_random_topology_no_crash(mesh_data):
    vertices, faces = mesh_data
    mesh = HalfEdgeMesh.from_arrays(vertices, faces.tolist())
    assert len(mesh.faces) > 0

    wrapper = QuadWrapper(target_face_count=10)
    # wrap() swallows every exception internally and returns the untouched
    # dense triangle mesh, so "it returned a HalfEdgeMesh" is also what a total
    # pipeline failure looks like. Capture the give-up message to tell them
    # apart.
    log = io.StringIO()
    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        result = wrapper.wrap(mesh)
    output = log.getvalue()

    assert isinstance(result, HalfEdgeMesh)
    assert 'quad wrap failed' not in output, \
        f"quad wrap fell back to the unmodified input mesh:\n{output}"
    assert result is not mesh

    # whatever came back must be a pure quad cage (an empty cage is allowed:
    # decimation can legitimately erase a soup of unconnected triangles)
    for f in result.faces:
        assert len(result.get_face_vertices(f)) == 4, \
            "wrap() returned non-quad faces -- the triangle input was handed back"
    if result.faces:
        assert len(result.vertices) > 0
        assert len(result.edges) > 0

@settings(deadline=None)
@given(valid_manifold_meshes())
def test_topological_invariants_manifold(trimesh_obj):
    he_mesh = HalfEdgeMesh.from_trimesh(trimesh_obj)

    V = len(he_mesh.vertices)
    E = len(he_mesh.edges)
    F = len(he_mesh.faces)

    assert V > 0 and E > 0 and F > 0

    # For a closed sphere-like manifold (genus 0), Euler characteristic V - E + F == 2
    assert V - E + F == 2

    # Independent ground truth from the face list alone
    incidence = face_edge_map(he_mesh)
    assert len(incidence) == E, "Edge records disagree with the face list"
    assert all(len(faces) == 2 for faces in incidence.values()), \
        "input mesh is not a closed manifold"

    # Check that every edge has exactly two adjacent faces -- and that they are
    # the two faces the face list says, so a get_edge_faces stubbed out to a
    # constant non-None pair cannot satisfy this.
    for edge in he_mesh.edges:
        f1, f2 = he_mesh.get_edge_faces(edge)
        assert f1 is not None
        assert f2 is not None
        assert f1 is not f2, "an edge reported the same face twice"
        assert {f1.index, f2.index} == set(incidence[edge_key(edge)]), \
            "get_edge_faces returned faces that do not border this edge"
        assert he_mesh.faces[f1.index] is f1 and he_mesh.faces[f2.index] is f2, \
            "get_edge_faces returned faces that are not part of the mesh"

    # Check that boundary detection returns False for all edges and vertices
    for vertex in he_mesh.vertices:
        assert not he_mesh.is_boundary_vertex(vertex)
    for edge in he_mesh.edges:
        assert not he_mesh.is_boundary_edge(edge)


@settings(deadline=None)
@given(open_boundary_meshes())
def test_boundary_detection_open_mesh(mesh_data):
    """The complementary branch: on a mesh WITH a rim the boundary predicates
    must return True for exactly the rim, so a constant implementation of
    is_boundary_edge / is_boundary_vertex / get_edge_faces cannot pass."""
    vertices, faces = mesh_data
    he_mesh = HalfEdgeMesh.from_arrays(vertices, faces)

    V, E, F = len(he_mesh.vertices), len(he_mesh.edges), len(he_mesh.faces)
    assert F == len(faces) > 0
    # a triangulated rectangle is a topological disk
    assert V - E + F == 1, f"expected a disk (chi = 1), got {V}-{E}+{F}"

    incidence = face_edge_map(he_mesh)
    assert len(incidence) == E
    expected_boundary_edges = {k for k, f in incidence.items() if len(f) == 1}
    expected_interior_edges = {k for k, f in incidence.items() if len(f) == 2}
    assert expected_boundary_edges, "fixture has no rim"
    assert expected_interior_edges, "fixture has no interior edge"
    expected_boundary_vertices = {i for pair in expected_boundary_edges for i in pair}

    got_boundary_edges = {edge_key(e) for e in he_mesh.edges if he_mesh.is_boundary_edge(e)}
    assert got_boundary_edges == expected_boundary_edges, \
        "is_boundary_edge disagrees with the independently counted rim"

    got_boundary_vertices = {v.index for v in he_mesh.vertices if he_mesh.is_boundary_vertex(v)}
    assert got_boundary_vertices == expected_boundary_vertices, \
        "is_boundary_vertex disagrees with the independently counted rim"

    for e in he_mesh.edges:
        f1, f2 = he_mesh.get_edge_faces(e)
        expected = set(incidence[edge_key(e)])
        if edge_key(e) in expected_boundary_edges:
            assert f1 is not None and f2 is None, \
                "a rim edge must report exactly one incident face"
            assert {f1.index} == expected
        else:
            assert f1 is not None and f2 is not None and f1 is not f2, \
                "an interior edge must report two distinct faces"
            assert {f1.index, f2.index} == expected, \
                "get_edge_faces returned faces that do not border this edge"

@settings(deadline=None)
@given(valid_manifold_meshes())
def test_halfedge_mesh_pointers_manifold(trimesh_obj):
    he_mesh = HalfEdgeMesh.from_trimesh(trimesh_obj)

    # Non-emptiness and element counts first: every loop below iterates over
    # these lists, so an add_face that builds nothing would otherwise satisfy
    # the whole test vacuously.
    assert len(he_mesh.faces) == len(trimesh_obj.faces)
    face_sizes = [len(he_mesh.get_face_vertices(f)) for f in he_mesh.faces]
    assert all(n == 3 for n in face_sizes), "trimesh fixtures are triangle meshes"
    assert len(he_mesh.half_edges) == sum(face_sizes)
    # closed manifold: every half-edge is twinned, so half as many Edge records
    assert len(he_mesh.edges) == len(he_mesh.half_edges) // 2

    # Check halfedge next/prev circularity
    for he in he_mesh.half_edges:
        assert he.next is not None
        assert he.prev is not None
        assert he.next.prev == he
        assert he.prev.next == he

        # Twin consistency. On a CLOSED manifold every half-edge must have a
        # twin -- branching on `if he.twin is not None` would let a total loss
        # of twin pairing through the else-branch untouched.
        assert he.twin is not None, "closed manifold has an untwinned half-edge"
        assert he.twin.twin == he
        assert he.edge == he.twin.edge
        assert he.twin is not he
        # the twin runs the opposite way along the same edge
        assert he.twin.vertex == he.prev.vertex
        assert he.twin.prev.vertex == he.vertex

        # Check vertex, face, edge back-pointers
        assert he.vertex is not None
        assert he.face is not None
        assert he.face.half_edge is not None

    # Every face cycle closes after exactly len(face) steps
    for f, n in zip(he_mesh.faces, face_sizes):
        assert f.half_edge is not None
        assert f.half_edge.face == f
        curr = f.half_edge
        for _ in range(n):
            assert curr.face == f
            curr = curr.next
        assert curr is f.half_edge, "face cycle does not close"

    for v in he_mesh.vertices:
        # every vertex of a closed manifold is used by a face
        assert v.half_edge is not None
        # he.vertex stores the target vertex.
        # The source vertex is he.prev.vertex
        assert v.half_edge.prev.vertex == v

    for e in he_mesh.edges:
        assert e.half_edge is not None
        assert e.half_edge.edge == e
        assert e.half_edge.twin is not None
        assert e.half_edge.twin.edge == e
