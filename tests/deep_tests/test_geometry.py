from collections import defaultdict

import pytest
import numpy as np
import trimesh
from src.core.halfedge_mesh import HalfEdgeMesh
from src.reverse_engineering.quad_wrap import QuadWrapper
from src.nurbs.converter import SubDToNURBSConverter
from src.subd.catmull_clark import evaluate_limit_surface

def is_quad_convex(vertices):
    v0, v1, v2, v3 = vertices
    normal = np.cross(v2 - v0, v3 - v1)
    if np.linalg.norm(normal) < 1e-10:
        return False
    normal = normal / np.linalg.norm(normal)
    edges = [v1 - v0, v2 - v1, v3 - v2, v0 - v3]
    signs = []
    for i in range(4):
        cross = np.cross(edges[i], edges[(i+1)%4])
        dot = np.dot(cross, normal)
        if abs(dot) > 1e-8:
            signs.append(np.sign(dot))
    if len(signs) < 3: # If too many collinear edges, it's degenerate/non-convex
        return False
    return len(set(signs)) == 1


def face_edge_incidence(mesh):
    """Undirected vertex pair -> number of incident faces.

    Derived from the exported face list only, so it never consults
    ``is_boundary_edge``/``get_edge_faces``: a boundary predicate stubbed out to
    a constant cannot make this agree with itself.
    """
    incidence = defaultdict(int)
    for face in mesh.to_arrays()['faces']:
        n = len(face)
        for i in range(n):
            incidence[tuple(sorted((face[i], face[(i + 1) % n])))] += 1
    return dict(incidence)


def he_from_trimesh(mesh):
    he_mesh = HalfEdgeMesh()
    vertex_map = {}
    for i, v in enumerate(mesh.vertices):
        vert = he_mesh.add_vertex(v.tolist())
        vertex_map[i] = vert.index
    for f in mesh.faces:
        he_mesh.add_face([vertex_map[v] for v in f])
    return he_mesh


def generate_complex_mesh():
    # Create a high-genus mesh by using a single torus (genus 1)
    mesh = trimesh.creation.torus(major_radius=10, minor_radius=2, major_sections=32, minor_sections=16)
    # Ensure it's watertight
    assert mesh.is_watertight
    # Convert to HalfEdgeMesh
    return he_from_trimesh(mesh)


def test_is_quad_convex_positive_control():
    """The convexity predicate used below must be able to say 'no'.

    Without this, ``test_quadwrapper_convexity`` would still pass if
    ``is_quad_convex`` were replaced by ``return True``.
    """
    convex = [np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]),
              np.array([1.0, 1.0, 0.0]), np.array([0.0, 1.0, 0.0])]
    assert is_quad_convex(convex)

    # third corner pulled inside the triangle of the other three -> reflex
    concave = [np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]),
               np.array([0.2, 0.2, 0.0]), np.array([0.0, 1.0, 0.0])]
    assert not is_quad_convex(concave)

    # fully collapsed quad: no well-defined diagonal normal
    degenerate = [np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)]
    assert not is_quad_convex(degenerate)

    # all four corners on one line
    collinear = [np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]),
                 np.array([2.0, 0.0, 0.0]), np.array([3.0, 0.0, 0.0])]
    assert not is_quad_convex(collinear)


def test_quadwrapper_convexity():
    he_mesh = generate_complex_mesh()
    assert len(he_mesh.faces) == 1024, "torus fixture changed"

    wrapper = QuadWrapper(target_face_count=200, smoothing_weight=0.5)
    quad_mesh = wrapper.wrap(he_mesh)

    # --- the wrap must actually have produced a cage -------------------------
    # wrap() has two silent give-up paths (empty-mesh early return and the
    # except-branch that hands back reference_mesh.copy()). Both leave a cage
    # with no quads at all, which would make the convexity loop below trivially
    # true, so pin the cage down before measuring convexity.
    assert quad_mesh is not he_mesh
    faces = quad_mesh.faces
    assert len(faces) > 0, "quad wrap returned an empty cage"
    quads = [f for f in faces if len(quad_mesh.get_face_vertices(f)) == 4]
    assert len(quads) == len(faces), (
        f"cage must be PURE quad, got {len(faces) - len(quads)} non-quad face(s) "
        f"out of {len(faces)} -- the dense triangle input was handed back")
    # target_face_count is honoured to roughly +-15% (measured cage: 194 quads)
    assert 140 <= len(quads) <= 260, f"target_face_count ignored: {len(quads)} quads"
    assert len(faces) < len(he_mesh.faces), "cage is not coarser than the input"

    non_convex_count = 0
    for face in quad_mesh.faces:
        verts = quad_mesh.get_face_vertices(face)
        if len(verts) == 4:
            positions = [np.array(v.position) for v in verts]
            if not is_quad_convex(positions):
                non_convex_count += 1

    assert non_convex_count == 0, f"Found {non_convex_count} non-convex quads!"

def test_quadwrapper_high_genus_watertight():
    he_mesh = generate_complex_mesh()
    wrapper = QuadWrapper(target_face_count=300, smoothing_weight=0.5)
    quad_mesh = wrapper.wrap(he_mesh)

    # --- the cage exists and is a real quad cage -----------------------------
    # An empty cage, or the untouched triangle input, is watertight too; both
    # of wrap()'s give-up paths have to be excluded before "no boundary edge"
    # means anything.
    assert quad_mesh is not he_mesh
    faces = quad_mesh.faces
    assert len(faces) > 0, "quad wrap returned an empty cage"
    assert all(len(quad_mesh.get_face_vertices(f)) == 4 for f in faces), \
        "cage must be PURE quad"
    assert 210 <= len(faces) <= 390, f"target_face_count ignored: {len(faces)} quads"
    assert len(faces) < len(he_mesh.faces)

    # --- watertight, counted independently of the mesh's own predicate -------
    incidence = face_edge_incidence(quad_mesh)
    assert incidence, "cage has no edges at all"
    assert len(incidence) == len(quad_mesh.edges), \
        "half-edge Edge records disagree with the face list"
    open_edges = {k: v for k, v in incidence.items() if v != 2}
    assert not open_edges, \
        f"Mesh is not watertight, {len(open_edges)} edge(s) with != 2 faces"

    # --- the topology the test name claims: genus 1 --------------------------
    V, E, F = len(quad_mesh.vertices), len(quad_mesh.edges), len(faces)
    assert V - E + F == 0, (
        f"torus cage must keep Euler characteristic 0 (genus 1), got "
        f"V-E+F = {V}-{E}+{F} = {V - E + F}")

    # the shipped predicate must agree with the independent count
    boundary_edges = [e for e in quad_mesh.edges if quad_mesh.is_boundary_edge(e)]
    assert len(boundary_edges) == 0, f"Mesh is not watertight, found {len(boundary_edges)} boundary edges!"


def _boundary_rows(patch):
    """The four boundary control rows of a 6x6 patch."""
    return [patch[0, :, :], patch[5, :, :], patch[:, 0, :], patch[:, 5, :]]


def _shared_boundary_row(p1, p2, atol=1e-9):
    """Return a boundary row that both patches carry identically, else None.

    Two Bezier patches are G0 (sewable) along a shared cage edge exactly when
    one boundary control row of each coincides, up to reversal. A collapsed
    (zero-length) row does not count -- otherwise two patches degenerated onto
    the same point would 'match'.
    """
    for r1 in _boundary_rows(p1):
        if np.linalg.norm(r1[-1] - r1[0]) <= atol:
            continue
        for r2 in _boundary_rows(p2):
            if np.allclose(r1, r2, atol=atol) or np.allclose(r1, r2[::-1], atol=atol):
                return r1
    return None


def test_nurbs_continuity():
    # Generate a simple cube and convert to NURBS
    mesh = trimesh.creation.box()
    he_mesh = he_from_trimesh(mesh)

    # Convert to quads first!
    wrapper = QuadWrapper(target_face_count=20, smoothing_weight=0.1)
    quad_mesh = wrapper.wrap(he_mesh)

    quad_faces = [f for f in quad_mesh.faces if len(quad_mesh.get_face_vertices(f)) == 4]
    assert len(quad_mesh.faces) > 0, "quad wrap returned an empty cage"
    assert len(quad_faces) == len(quad_mesh.faces), \
        "the cage handed to the converter must be pure quad"

    converter = SubDToNURBSConverter(continuity='G2', tolerance=1e-4)
    # NOTE: generate_patches has no subdivision_levels parameter -- it used to
    # accept one and never read it; the cage is fitted at the density it
    # arrives with.
    patches = converter.generate_patches(quad_mesh)

    # one patch per cage quad, each a degree-5 (6x6) control grid
    assert len(patches) == len(quad_faces) > 0
    for patch in patches:
        assert patch.shape == (6, 6, 3), "Patch control points should be a 6x6 grid for G3!"
        assert np.isfinite(patch).all(), "patch contains NaN/Inf control points"

    # --- 1. each patch is anchored on ITS OWN cage quad ----------------------
    # The corners of patch k are the limit-surface positions of cage quad k's
    # corners; a fitter that returns zeros (or any patch unrelated to the cage)
    # fails here even though the 6x6 shape is structurally guaranteed.
    limit_positions, _ = evaluate_limit_surface(quad_mesh)
    for k, (patch, face) in enumerate(zip(patches, quad_faces)):
        expected = [limit_positions[v.index] for v in quad_mesh.get_face_vertices(face)]
        actual = [patch[0, 0], patch[5, 0], patch[5, 5], patch[0, 5]]
        for e, a in zip(expected, actual):
            assert np.allclose(e, a, atol=1e-9), (
                f"patch {k} corner {a} is not on its cage quad corner {e}")

    # --- 2. the control net stays on the cube and actually spans it ----------
    ctrl = np.asarray(patches).reshape(-1, 3)
    lo, hi = mesh.bounds
    margin = 0.1 * (hi - lo)
    assert (ctrl >= lo - margin).all() and (ctrl <= hi + margin).all(), \
        "control points escaped the cube's bounding box"
    span = ctrl.max(axis=0) - ctrl.min(axis=0)
    assert (span > 0.5 * (hi - lo)).all(), \
        f"patch cloud collapsed, spans only {span} of {(hi - lo)}"

    # --- 3. the continuity the test name claims: G0 across every cage edge ---
    face_to_patch = {f.index: k for k, f in enumerate(quad_faces)}
    checked = 0
    for edge in quad_mesh.edges:
        f1, f2 = quad_mesh.get_edge_faces(edge)
        if f1 is None or f2 is None:
            continue
        k1, k2 = face_to_patch[f1.index], face_to_patch[f2.index]
        if k1 == k2:
            continue
        assert _shared_boundary_row(patches[k1], patches[k2]) is not None, (
            f"patches {k1}/{k2} meet along a cage edge but share no boundary "
            f"curve -- the sewn shell would have a crack there")
        checked += 1
    assert checked == len(quad_mesh.edges), (
        f"only {checked} of {len(quad_mesh.edges)} cage edges were G0-checked")

    # --- 4. the sewn B-Rep ---------------------------------------------------
    pytest.importorskip("OCP", reason="cadquery-ocp not installed")
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_FACE, TopAbs_SOLID

    def count(shape, kind):
        exp = TopExp_Explorer(shape, kind)
        n = 0
        while exp.More():
            n += 1
            exp.Next()
        return n

    shape = converter.build_shape(patches, simplify=False)
    assert shape is not None, "cadquery-ocp is installed, build_shape must not return None"
    assert count(shape, TopAbs_FACE) == len(patches), \
        "every patch must survive as a B-Rep face"
    assert count(shape, TopAbs_SOLID) == 1, \
        "the cube's patches must sew into exactly one closed solid"
