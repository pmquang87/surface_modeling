"""
Regression tests for reported defects in src/subd/{catmull_clark,primitives,editing}.py.

Each test names the defect it pins down. Geometric claims are checked against an
independent authority wherever one exists (trimesh for solid integrity, the exact
bicubic B-spline stencil for the Catmull-Clark limit) rather than against the
implementation's own report of itself.
"""

import gc
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest

from src.core.halfedge_mesh import HalfEdgeMesh
from src.subd.catmull_clark import subdivide, evaluate_limit_surface, identify_regular_regions
from src.subd.primitives import (create_box, create_cylinder, create_torus,
                                 create_cone, create_plane, create_sphere)
from src.subd.editing import (extrude_faces, extrude_edges, insert_edge_loop,
                              mirror_mesh, bridge_faces, knife_cut, inset_faces)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def solid_report(mesh: HalfEdgeMesh):
    """Independent integrity check via trimesh."""
    tm = mesh.to_trimesh()
    return {
        'watertight': bool(tm.is_watertight),
        'winding_consistent': bool(tm.is_winding_consistent),
        'volume': float(tm.volume),
    }


def dangling_half_edges(mesh: HalfEdgeMesh) -> int:
    return sum(1 for he in mesh.half_edges if he.twin is None)


def directed_edges(face_indices):
    n = len(face_indices)
    return [(face_indices[i], face_indices[(i + 1) % n]) for i in range(n)]


def assert_no_repeated_directed_edge(mesh: HalfEdgeMesh):
    """A consistently wound surface uses every directed edge at most once."""
    seen = set()
    for f in mesh.faces:
        fv = [v.index for v in mesh.get_face_vertices(f)]
        for de in directed_edges(fv):
            assert de not in seen, f"directed edge {de} used twice -> inconsistent winding"
            seen.add(de)


def reference_vertex_normals(mesh: HalfEdgeMesh, positions: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals of `mesh`'s topology at `positions`.

    Written out independently of src (sum of cross(P_i, P_i+1) around each face,
    which is twice the projected area times the unit normal) so it can be used as
    an authority on *which* positions the module's normals were built from.
    """
    normals = np.zeros((len(mesh.vertices), 3))
    for f in mesh.faces:
        fv = [v.index for v in mesh.get_face_vertices(f)]
        if len(fv) < 3:
            continue
        P = positions[fv]
        fn = np.zeros(3)
        for k in range(len(fv)):
            fn = fn + np.cross(P[k], P[(k + 1) % len(fv)])
        for idx in fv:
            normals[idx] += fn
    lengths = np.linalg.norm(normals, axis=1)
    good = lengths > 1e-12
    normals[good] /= lengths[good][:, None]
    return normals


CLOSED_PRIMITIVES = {
    'box': create_box,
    'cylinder': create_cylinder,
    'torus': create_torus,
    'cone': create_cone,
    'sphere': create_sphere,
}


# --------------------------------------------------------------------------
# primitives: findings 4 (cylinder caps), 5 (sphere), 6 (cone)
# --------------------------------------------------------------------------

@pytest.mark.parametrize('name', sorted(CLOSED_PRIMITIVES))
def test_closed_primitives_are_watertight_consistent_and_positive(name):
    mesh = CLOSED_PRIMITIVES[name]()
    rep = solid_report(mesh)
    assert rep['watertight'], f"{name} is not watertight: {rep}"
    assert rep['winding_consistent'], f"{name} has inconsistent winding: {rep}"
    assert rep['volume'] > 0, f"{name} is inside-out (volume {rep['volume']}): {rep}"


@pytest.mark.parametrize('name', sorted(CLOSED_PRIMITIVES))
def test_closed_primitives_have_no_dangling_half_edges(name):
    """Finding 4: the cylinder caps were wound like the walls, so cap half-edges
    never found a twin and the half-edge mesh itself was open."""
    mesh = CLOSED_PRIMITIVES[name]()
    assert dangling_half_edges(mesh) == 0, \
        f"{name}: {dangling_half_edges(mesh)} half-edges without a twin"
    assert_no_repeated_directed_edge(mesh)


def test_plane_is_open_but_consistently_wound():
    mesh = create_plane()
    rep = solid_report(mesh)
    assert not rep['watertight'], "an open plane should not be watertight"
    assert rep['winding_consistent']
    assert_no_repeated_directed_edge(mesh)


# Two Catmull-Clark levels turn every n-gon into n quads and then each quad into
# four, so the face count is pinned by the base cage alone. Vertex counts and
# volumes are measured once against the working implementation; a subdivision
# that silently does nothing cannot reproduce them.
SUBDIVIDED_TWICE = {          # name: (vertices, faces, volume)
    'box':      (98,   96,   0.35011720591938866),
    'cone':     (130,  128,  0.09908914521553071),
    'cylinder': (194,  192,  0.36082545313278970),
    'sphere':   (706,  704,  0.37630248746478734),
    'torus':    (2048, 2048, 1.38338436748291900),
}


@pytest.mark.parametrize('name', sorted(CLOSED_PRIMITIVES))
def test_closed_primitives_survive_subdivision(name):
    base = CLOSED_PRIMITIVES[name]()
    mesh = CLOSED_PRIMITIVES[name](subdivisions=2)
    exp_verts, exp_faces, exp_volume = SUBDIVIDED_TWICE[name]

    # Precondition: the refinement actually ran. Without this the assertions
    # below are satisfied by the un-subdivided cage, which is already a valid
    # solid, so a no-op `subdivide` would look correct.
    corners = sum(len(base.get_face_vertices(f)) for f in base.faces)
    assert len(mesh.faces) == 4 * corners == exp_faces, (
        f"{name}: 2 levels over {len(base.faces)} faces ({corners} corners) gave "
        f"{len(mesh.faces)} faces, expected {4 * corners}")
    assert len(mesh.vertices) == exp_verts, \
        f"{name}: {len(mesh.vertices)} vertices after 2 levels, expected {exp_verts}"

    rep = solid_report(mesh)
    assert rep['watertight'], f"subdivided {name} not watertight: {rep}"
    assert rep['winding_consistent'], f"subdivided {name} winding inconsistent: {rep}"
    assert rep['volume'] > 0, f"subdivided {name} inside-out: {rep}"
    # Catmull-Clark pulls the cage in towards the limit surface.
    base_volume = solid_report(base)['volume']
    assert rep['volume'] < base_volume, \
        f"subdivided {name} did not shrink: {rep['volume']} vs cage {base_volume}"
    assert rep['volume'] == pytest.approx(exp_volume, rel=1e-9)


def test_cylinder_volume_matches_analytic_prism():
    """Sanity anchor: caps wound correctly give the regular-prism volume.

    The volume alone does not see the cap winding - a cylinder built with the
    finding-4 defect has the same volume to the last bit - so the winding claim
    is carried by the integrity checks, not by the number.
    """
    segs = 16
    r, h = 0.5, 1.0
    mesh = create_cylinder(radius=r, height=h, segments=segs)
    expected = 0.5 * segs * r * r * np.sin(2 * np.pi / segs) * h
    rep = solid_report(mesh)
    assert rep['volume'] == pytest.approx(expected, rel=1e-9)
    assert rep['winding_consistent'], f"cap winding is inconsistent: {rep}"
    assert dangling_half_edges(mesh) == 0, \
        f"{dangling_half_edges(mesh)} cap half-edges never found a twin"


def test_cone_volume_matches_analytic_pyramid():
    """Same caveat as the cylinder: a flipped base ring leaves the volume alone."""
    segs = 16
    r, h = 0.5, 1.0
    mesh = create_cone(radius=r, height=h, segments=segs)
    base_area = 0.5 * segs * r * r * np.sin(2 * np.pi / segs)
    rep = solid_report(mesh)
    assert rep['volume'] == pytest.approx(base_area * h / 3.0, rel=1e-9)
    assert rep['winding_consistent'], f"base winding is inconsistent: {rep}"
    assert dangling_half_edges(mesh) == 0, \
        f"{dangling_half_edges(mesh)} base half-edges never found a twin"


def test_sphere_volume_is_positive_and_near_analytic():
    mesh = create_sphere(radius=1.0, segments=32, rings=24)
    vol = solid_report(mesh)['volume']
    assert vol > 0
    assert vol == pytest.approx(4.0 / 3.0 * np.pi, rel=0.02)


# --------------------------------------------------------------------------
# catmull_clark finding 1: subdivision must not be O(V * E)
# --------------------------------------------------------------------------

def _best_subdivide_time(mesh, repeats=2):
    """Fastest of `repeats` passes.

    A single sample is worth little here: a GC pause or the scheduler can inflate
    one run several-fold when the whole suite is running. The minimum is the
    sample least contaminated by interference.
    """
    best = float('inf')
    for _ in range(repeats):
        gc.collect()
        t0 = time.perf_counter()
        out = subdivide(mesh, 1)
        best = min(best, time.perf_counter() - t0)
        del out
    return best


def test_subdivision_scales_linearly_not_quadratically():
    """Finding 1: incident edges were found by rescanning every edge for every
    vertex. Each level quadruples V and E, so a quadratic pass costs ~16x per
    level where a linear one costs ~4x."""
    small = subdivide(create_box(), 5)          # ~6k vertices
    large = subdivide(small, 1)                 # ~25k vertices
    assert len(small.vertices) > 6000
    assert len(large.vertices) > 3 * len(small.vertices)

    t_small = _best_subdivide_time(small)
    t_large = _best_subdivide_time(large)
    growth = t_large / t_small

    # Linear ~4x, quadratic ~16x (measured ~15x before the fix). 8x separates them.
    assert growth < 8.0, (
        f"cost grew {growth:.1f}x for {len(large.vertices) / len(small.vertices):.1f}x "
        f"the mesh ({t_small:.3f}s -> {t_large:.3f}s): still super-linear")


def test_subdivision_throughput_is_flat():
    """Cost per vertex must not climb with mesh size."""
    small = subdivide(create_box(), 4)
    large = subdivide(small, 2)
    assert len(large.vertices) > 10 * len(small.vertices)

    us_small = _best_subdivide_time(small) / len(small.vertices) * 1e6
    us_large = _best_subdivide_time(large) / len(large.vertices) * 1e6

    assert us_large < 4.0 * us_small, (
        f"per-vertex cost rose from {us_small:.0f} to {us_large:.0f} us "
        f"between {len(small.vertices)} and {len(large.vertices)} vertices")


def test_subdivision_absolute_budget():
    """A ~25k-vertex pass must not take minutes."""
    base = subdivide(create_box(), 5)
    assert len(base.vertices) > 6000
    t0 = time.perf_counter()
    out = subdivide(base, 1)
    dt = time.perf_counter() - t0
    assert len(out.vertices) > 4 * len(base.vertices) - 10
    assert dt < 10.0, f"one subdivision pass over {len(base.vertices)} vertices took {dt:.1f}s"


def test_subdivide_matches_reference_vertex_rule():
    """The speed-up must not change the geometry: check the vertex/edge/face
    rules against a straightforward independent implementation."""
    mesh = create_torus(major_segments=8, minor_segments=6)

    face_pt = {}
    for f in mesh.faces:
        face_pt[f.index] = np.mean([v.position for v in mesh.get_face_vertices(f)], axis=0)
    edge_mid = {}
    for e in mesh.edges:
        edge_mid[e.index] = (e.half_edge.vertex.position
                             + e.half_edge.prev.vertex.position) / 2.0

    expected = {}
    for v in mesh.vertices:
        inc = [e for e in mesh.edges
               if e.half_edge.vertex is v or e.half_edge.prev.vertex is v]
        faces = mesh.get_vertex_faces(v)
        n = len(inc)
        F = np.mean([face_pt[f.index] for f in faces], axis=0)
        R = np.mean([edge_mid[e.index] for e in inc], axis=0)
        expected[v.index] = (F + 2 * R + (n - 3) * v.position) / n

    V, E, F_count = len(mesh.vertices), len(mesh.edges), len(mesh.faces)
    out = subdivide(mesh, 1)
    # The three blocks are laid down in order: originals, then one point per
    # edge, then one per face. Checking only the first block leaves the edge rule
    # structurally unreachable - the original vertices are built from edge
    # MIDpoints, never from edge points.
    assert len(out.vertices) == V + E + F_count, \
        f"expected {V}+{E}+{F_count} vertices, got {len(out.vertices)}"

    for v in mesh.vertices:
        assert np.allclose(out.vertices[v.index].position, expected[v.index], atol=1e-12), \
            f"vertex {v.index} moved to the wrong place"

    for e in mesh.edges:
        v_src = e.half_edge.prev.vertex.position
        v_tgt = e.half_edge.vertex.position
        f1, f2 = mesh.get_edge_faces(e)
        if f2 is None:                       # boundary edge -> plain midpoint
            ref = (v_src + v_tgt) / 2.0
        else:
            ref = (v_src + v_tgt + face_pt[f1.index] + face_pt[f2.index]) / 4.0
        assert np.allclose(out.vertices[V + e.index].position, ref, atol=1e-12), \
            f"edge point {e.index} is at the wrong place"

    for f in mesh.faces:
        assert np.allclose(out.vertices[V + E + f.index].position,
                           face_pt[f.index], atol=1e-12), \
            f"face point {f.index} is not the face centroid"


def test_subdivide_keyword_contract_still_works():
    """The GUI calls subdivide(mesh, levels, smooth=...)."""
    box = create_box()
    linear = subdivide(box, levels=1, smooth=False)
    smooth = subdivide(box, levels=1, smooth=True)
    assert len(linear.faces) == len(smooth.faces) == 24
    # linear subdivision leaves the original corners exactly in place
    for v in box.vertices:
        assert np.allclose(linear.vertices[v.index].position, v.position)
    # smooth subdivision pulls them inward
    assert not np.allclose(smooth.vertices[0].position, box.vertices[0].position)
    assert len(subdivide(box, 0).faces) == len(box.faces)


# --------------------------------------------------------------------------
# catmull_clark finding 2: the limit stencil
# --------------------------------------------------------------------------

def bicubic_limit_for_regular_vertex(mesh, v):
    """Exact limit of a valence-4 vertex whose 1-ring is four quads.

    There the Catmull-Clark limit surface *is* a uniform bicubic B-spline, whose
    control-point limit is the tensor product of the cubic basis (1/6, 4/6, 1/6):
        (16*P + 4*sum(edge neighbours) + sum(diagonals)) / 36
    """
    faces = mesh.get_vertex_faces(v)
    if len(faces) != 4:
        return None
    neighbours = mesh.get_vertex_neighbors(v)
    if len(neighbours) != 4:
        return None
    diagonals = []
    for f in faces:
        fv = [x.index for x in mesh.get_face_vertices(f)]
        if len(fv) != 4 or v.index not in fv:
            return None
        k = fv.index(v.index)
        diagonals.append(mesh.vertices[fv[(k + 2) % 4]].position)
    return (16.0 * v.position
            + 4.0 * np.sum([nb.position for nb in neighbours], axis=0)
            + np.sum(diagonals, axis=0)) / 36.0


def test_limit_surface_matches_exact_bicubic_stencil_on_regular_mesh():
    """Finding 2: on an all-quad valence-4 mesh the limit position is known in
    closed form, so the stencil can be checked exactly rather than by eyeball."""
    mesh = create_torus(major_segments=12, minor_segments=8)
    limit, _ = evaluate_limit_surface(mesh)

    checked = 0
    for v in mesh.vertices:
        ref = bicubic_limit_for_regular_vertex(mesh, v)
        if ref is None:
            continue
        checked += 1
        assert np.allclose(limit[v.index], ref, atol=1e-12), (
            f"vertex {v.index}: limit {limit[v.index]} != exact bicubic {ref}")
    assert checked > 50, f"expected many regular vertices, checked only {checked}"


def test_limit_surface_is_invariant_under_subdivision():
    """The limit point of a vertex does not depend on how far the cage has
    already been refined. Original vertex i keeps index i through a pass, so
    L(M)[i] must equal L(subdivide(M))[i] exactly."""
    for mesh in (subdivide(create_box(), 1),
                 create_torus(major_segments=8, minor_segments=6)):
        nv = len(mesh.vertices)
        fine = subdivide(mesh, 1)
        # Precondition: the cage really was refined. Comparing a mesh with
        # itself gives drift == 0 and asserts nothing.
        assert len(fine.vertices) > 3 * nv, \
            f"cage was not refined: {nv} -> {len(fine.vertices)} vertices"

        here, _ = evaluate_limit_surface(mesh)
        finer, _ = evaluate_limit_surface(fine)
        drift = np.linalg.norm(here[:nv] - finer[:nv], axis=1).max()
        assert drift < 1e-12, f"limit position drifted by {drift:.3e} after one refinement"


def test_limit_surface_converges_to_deep_subdivision():
    """The cage vertex must actually travel to the reported limit point: the
    residual has to shrink geometrically, not settle on a constant offset."""
    base = subdivide(create_box(), 1)
    nv = len(base.vertices)
    limit, _ = evaluate_limit_surface(base)

    errors = []
    deep = base
    for _ in range(6):
        deep = subdivide(deep, 1)
        truth = np.array([deep.vertices[i].position for i in range(nv)])
        errors.append(float(np.linalg.norm(limit[:nv] - truth, axis=1).max()))

    for a, b in zip(errors, errors[1:]):
        assert b < a, f"residual stopped shrinking: {errors}"
    assert errors[-1] < 0.02 * errors[0], \
        f"residual plateaued instead of vanishing: {errors}"


def _bent_grid():
    """A 4x4 grid with a boundary, bent out of its own plane.

    A *flat* uniform grid is already its own limit surface (the stencil is an
    affine combination of symmetric coplanar points, so every interior vertex
    maps to itself). Bending it is what makes 'interior vertices move' a claim
    that can fail.
    """
    mesh = create_plane(subdivisions_x=3, subdivisions_y=3)
    for v in mesh.vertices:
        x, z = float(v.position[0]), float(v.position[2])
        v.position = np.array([x, 0.4 * np.cos(2.5 * x) * np.cos(1.7 * z), z])
    return mesh


def test_limit_surface_preserves_boundary_vertices():
    """Boundary vertices stay put - and only boundary vertices do."""
    plane = create_plane(subdivisions_x=3, subdivisions_y=3)
    boundary = [v for v in plane.vertices if plane.is_boundary_vertex(v)]
    interior = [v for v in plane.vertices if not plane.is_boundary_vertex(v)]
    assert len(plane.vertices) == 16
    assert len(boundary) == 12 and len(interior) == 4, \
        f"fixture changed: {len(boundary)} boundary / {len(interior)} interior"

    limit, _ = evaluate_limit_surface(plane)
    for v in boundary:
        assert np.allclose(limit[v.index], v.position), \
            f"boundary vertex {v.index} was moved off the cage"

    # The complementary half: on a cage that is not already the limit surface,
    # the interior must move. Without this the whole test is satisfied by an
    # implementation that copies its input through.
    bent = _bent_grid()
    limit_b, _ = evaluate_limit_surface(bent)
    moved = 0
    for v in bent.vertices:
        if bent.is_boundary_vertex(v):
            assert np.allclose(limit_b[v.index], v.position), \
                f"boundary vertex {v.index} was moved off the bent cage"
        else:
            assert not np.allclose(limit_b[v.index], v.position), \
                f"interior vertex {v.index} did not travel to the limit surface"
            moved += 1
    assert moved == 4, f"expected 4 interior vertices, checked {moved}"


def test_limit_surface_is_affine_invariant():
    """Stencil weights must sum to one, or the surface drifts under translation."""
    mesh = create_torus(major_segments=8, minor_segments=6)
    limit_a, _ = evaluate_limit_surface(mesh)

    # Precondition: the identity map is trivially affine invariant, so the
    # comparison below only means something once the stencil has moved something.
    cage = np.array([v.position for v in mesh.vertices])
    assert not np.allclose(limit_a, cage), \
        "limit surface is identical to the cage - no stencil was applied"

    shift = np.array([3.0, -1.5, 7.25])
    moved = mesh.copy()
    for v in moved.vertices:
        v.position = v.position + shift
    limit_b, _ = evaluate_limit_surface(moved)

    assert np.allclose(limit_b, limit_a + shift, atol=1e-12)


# --------------------------------------------------------------------------
# catmull_clark finding 3: honesty of evaluate_limit_surface
# --------------------------------------------------------------------------

def test_evaluate_limit_surface_does_not_mutate_the_callers_mesh():
    """Finding 3: the function called mesh.compute_vertex_normals(), silently
    overwriting normals the caller owned."""
    mesh = subdivide(create_box(), 1)
    for v in mesh.vertices:
        v.normal = np.array([1.0, 0.0, 0.0])
    for f in mesh.faces:
        f.normal = np.array([0.0, 1.0, 0.0])
    positions_before = np.array([v.position.copy() for v in mesh.vertices])

    limit, normals = evaluate_limit_surface(mesh)

    # Precondition: the call did some work. A function that touches nothing
    # trivially mutates nothing, so the assertions below need a live result -
    # and it has to be a plausible one, or returning zeros would also qualify.
    assert limit.shape == (len(mesh.vertices), 3)
    assert normals.shape == (len(mesh.vertices), 3)
    assert np.all(np.isfinite(limit)) and np.all(np.isfinite(normals))
    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-9), \
        "limit normals are not unit length"
    travel = np.linalg.norm(limit - positions_before, axis=1).max()
    assert 1e-6 < travel < 0.2, (
        f"limit positions moved by {travel:.4g}: expected a small but non-zero "
        "shrink towards the limit surface")

    assert all(np.allclose(v.normal, [1.0, 0.0, 0.0]) for v in mesh.vertices), \
        "vertex normals were overwritten"
    assert all(np.allclose(f.normal, [0.0, 1.0, 0.0]) for f in mesh.faces), \
        "face normals were overwritten"
    assert np.allclose(np.array([v.position for v in mesh.vertices]), positions_before)


def test_evaluate_limit_surface_returns_usable_unit_normals():
    """Whatever approximation is documented, the normals must at least be unit
    length and point outward on a convex closed shape."""
    mesh = create_sphere(radius=1.0, segments=16, rings=12)
    limit, normals = evaluate_limit_surface(mesh)
    assert normals.shape == (len(mesh.vertices), 3)

    lengths = np.linalg.norm(normals, axis=1)
    assert np.allclose(lengths, 1.0, atol=1e-9), "limit normals are not unit length"

    radial = limit / np.linalg.norm(limit, axis=1, keepdims=True)
    assert np.all(np.einsum('ij,ij->i', radial, normals) > 0.9), \
        "normals do not point outward on a sphere"


def test_limit_normals_are_built_from_the_limit_positions_not_the_cage():
    """The documented contract: area-weighted normals of the mesh formed by the
    LIMIT positions.

    On a sphere the limit map is a radial rescale, so cage and limit normals
    agree and the outward test above cannot tell them apart. A torus separates
    them by ~5e-2, which is far outside any rounding.
    """
    mesh = create_torus(major_segments=8, minor_segments=6)
    cage = np.array([v.position for v in mesh.vertices])
    limit, normals = evaluate_limit_surface(mesh)

    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-12), \
        "limit normals are not unit length"

    from_limit = reference_vertex_normals(mesh, limit)
    from_cage = reference_vertex_normals(mesh, cage)

    # Precondition: on this fixture the two really do differ.
    separation = float(np.abs(from_limit - from_cage).max())
    assert separation > 1e-3, \
        f"fixture no longer separates cage from limit normals ({separation:.2e})"

    assert np.allclose(normals, from_limit, atol=1e-12), \
        "limit normals do not match the area-weighted normals of the limit mesh"
    assert not np.allclose(normals, from_cage, atol=1e-6), \
        "limit normals were computed from the cage positions"


def test_identify_regular_regions_partitions_every_vertex():
    mesh = create_box()
    regular, irregular = identify_regular_regions(mesh)
    assert len(regular) + len(irregular) == len(mesh.vertices)
    # every cube corner is valence 3 -> irregular
    assert len(regular) == 0

    torus = create_torus(major_segments=8, minor_segments=6)
    regular, irregular = identify_regular_regions(torus)
    assert len(regular) + len(irregular) == len(torus.vertices)
    assert len(irregular) == 0, "a quad torus is regular everywhere"


def test_identify_regular_regions_needs_quad_surroundings_not_just_valence():
    """Valence 4 is not enough: the 1-ring must be quads.

    Both all-quad fixtures above are blind to this - a rule that only counted
    neighbours would score them identically.
    """
    sphere = create_sphere()          # 8 segments, 6 rings
    regular, irregular = identify_regular_regions(sphere)
    assert len(regular) + len(irregular) == len(sphere.vertices) == 42

    val4_on_triangles = [v.index for v in sphere.vertices
                         if len(sphere.get_vertex_neighbors(v)) == 4
                         and any(len(sphere.get_face_vertices(f)) != 4
                                 for f in sphere.get_vertex_faces(v))]
    # the two rings flanking the poles: valence 4, but sitting on the pole fans
    assert len(val4_on_triangles) == 16, \
        f"fixture changed: {len(val4_on_triangles)} valence-4 vertices touch a triangle"
    reg_idx = {v.index for v in regular}
    for idx in val4_on_triangles:
        assert idx not in reg_idx, \
            f"vertex {idx} has valence 4 but triangles around it - not regular"
    assert len(regular) == 24 and len(irregular) == 18

    cone = create_cone()              # nothing but triangles and one n-gon
    regular, irregular = identify_regular_regions(cone)
    assert len(regular) == 0, "no vertex of a triangulated cone is regular"
    assert len(irregular) == len(cone.vertices) == 9


def test_identify_regular_regions_uses_the_boundary_rule():
    """On an open patch the rule is valence <= 3, not valence == 4."""
    plane = create_plane(subdivisions_x=3, subdivisions_y=3)
    regular, irregular = identify_regular_regions(plane)
    assert len(regular) + len(irregular) == len(plane.vertices) == 16

    boundary = [v for v in plane.vertices if plane.is_boundary_vertex(v)]
    assert len(boundary) == 12
    assert all(len(plane.get_vertex_neighbors(v)) <= 3 for v in boundary)

    # 12 boundary vertices of valence 2 or 3 plus 4 interior of valence 4:
    # a plain valence == 4 count would report only 4.
    assert len(regular) == 16, \
        f"boundary rule ignored: {len(regular)} regular, {len(irregular)} irregular"
    assert len(irregular) == 0


# --------------------------------------------------------------------------
# editing finding 7: extrude_edges winding + unnormalised direction
# --------------------------------------------------------------------------

def test_extrude_edges_winds_the_new_strip_opposite_to_its_source_face():
    """Finding 7: the strip reused the source face's directed edge, so the two
    faces could never be twinned and the result was never watertight-compatible."""
    plane = create_plane()
    boundary = [e.index for e in plane.edges if plane.is_boundary_edge(e)]
    out = extrude_edges(plane, boundary[:1], distance=0.5,
                        direction=np.array([0.0, 1.0, 0.0]))
    assert_no_repeated_directed_edge(out)

    # the shared edge must now be traversed in opposite directions
    faces = [[v.index for v in out.get_face_vertices(f)] for f in out.faces]
    a, b = set(directed_edges(faces[0])), set(directed_edges(faces[-1]))
    assert not (a & b), "source face and new strip share a same-direction edge"
    assert a & {(y, x) for (x, y) in b}, "source face and new strip are not joined"


def test_extrude_edges_closes_a_box_when_all_boundary_edges_are_extruded():
    """Extruding every boundary edge of an open plane must give a manifold band."""
    plane = create_plane()
    boundary = [e.index for e in plane.edges if plane.is_boundary_edge(e)]
    assert len(boundary) == 4
    out = extrude_edges(plane, boundary, distance=0.5,
                        direction=np.array([0.0, 1.0, 0.0]))

    # The real code `continue`s past any edge it cannot handle, so "extruded
    # nothing" is a live failure mode and the plane satisfies every winding
    # check on its own. Pin the band that must have been built: one strip quad
    # per boundary edge and two new vertices each.
    assert len(out.faces) == len(plane.faces) + 4 == 5, \
        f"expected the source face plus 4 strip quads, got {len(out.faces)}"
    assert len(out.vertices) == len(plane.vertices) + 8 == 12, \
        f"expected 8 new vertices, got {len(out.vertices) - len(plane.vertices)}"

    assert_no_repeated_directed_edge(out)
    # Every strip quad twins with the source face and with its two neighbours;
    # only the free outer rim (4 quads x 3 open sides) stays dangling.
    assert dangling_half_edges(out) == 12, \
        f"strip is not a manifold band: {dangling_half_edges(out)} dangling half-edges"


def test_extrude_edges_normalises_the_direction_vector():
    """Finding 9: `distance` was scaled by |direction|."""
    plane = create_plane()
    e0 = [e.index for e in plane.edges if plane.is_boundary_edge(e)][:1]
    edge = plane.edges[e0[0]]
    src_v1 = edge.half_edge.prev.vertex.index
    src_v2 = edge.half_edge.vertex.index

    unit = extrude_edges(plane, e0, distance=1.0, direction=np.array([0.0, 1.0, 0.0]))
    long = extrude_edges(plane, e0, distance=1.0, direction=np.array([0.0, 5.0, 0.0]))

    # Anchor the magnitude to `distance`, not just the agreement of two results:
    # a version that ignored `distance` altogether would also agree with itself.
    assert len(unit.vertices) == len(plane.vertices) + 2, \
        f"nothing was extruded ({len(unit.vertices)} vertices)"
    assert np.linalg.norm(unit.vertices[-2].position
                          - plane.vertices[src_v1].position) == pytest.approx(1.0, abs=1e-12)
    assert np.linalg.norm(unit.vertices[-1].position
                          - plane.vertices[src_v2].position) == pytest.approx(1.0, abs=1e-12)

    assert np.allclose(unit.vertices[-1].position, long.vertices[-1].position), \
        "extrusion distance depends on the length of `direction`"


def test_extrude_faces_normalises_the_direction_vector():
    box = create_box()
    src = [v.index for v in box.get_face_vertices(box.faces[0])]
    unit = extrude_faces(box, [0], distance=1.0, direction=np.array([0.0, 0.0, 1.0]))
    long = extrude_faces(box, [0], distance=1.0, direction=np.array([0.0, 0.0, 9.0]))

    # Without this, "extruded nothing" makes both results the untouched box and
    # the comparison below is trivially true.
    assert len(unit.vertices) == len(box.vertices) + 4, \
        f"nothing was extruded ({len(unit.vertices)} vertices)"
    for k, v_idx in enumerate(src):
        offset = unit.vertices[len(box.vertices) + k].position - box.vertices[v_idx].position
        assert np.linalg.norm(offset) == pytest.approx(1.0, abs=1e-12), \
            f"new vertex {k} moved by {np.linalg.norm(offset)}, expected the distance 1.0"

    assert np.allclose(np.array([v.position for v in unit.vertices]),
                       np.array([v.position for v in long.vertices]))


def test_extrude_faces_survives_a_degenerate_face():
    """Finding 9: a zero-area face has a zero normal, and the unguarded divide
    turned every new vertex into NaN. The documented behaviour is to skip it."""
    mesh = HalfEdgeMesh.from_arrays(
        [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], [[0, 1, 2, 3]])
    out = extrude_faces(mesh, [0], distance=0.5)
    positions = np.array([v.position for v in out.vertices])
    assert not np.isnan(positions).any(), "degenerate face produced NaN vertices"
    assert not np.isinf(positions).any()

    # Skipped, not extruded to some other finite garbage.
    assert len(out.vertices) == 4 and len(out.faces) == 1, \
        f"degenerate face was not skipped: {len(out.vertices)} verts, {len(out.faces)} faces"
    assert np.allclose(positions, np.array([v.position for v in mesh.vertices]))

    # Positive control in the same test: the skip must be about *this* face, not
    # a globally inert extrude_faces hiding behind a no-NaN assertion.
    box = create_box()
    good = extrude_faces(box, [0], distance=0.5)
    assert len(good.vertices) == len(box.vertices) + 4
    assert len(good.faces) == len(box.faces) + 4


def test_extrude_faces_still_produces_a_solid():
    box = create_box()
    out = extrude_faces(box, [0], distance=0.3)
    rep = solid_report(out)
    assert rep['watertight'] and rep['winding_consistent']
    assert rep['volume'] == pytest.approx(1.3, rel=1e-9)


# --------------------------------------------------------------------------
# editing finding 8: insert_edge_loop must not zigzag
# --------------------------------------------------------------------------

def test_insert_edge_loop_places_vertices_consistently_around_the_ring():
    """Finding 8: each new point was interpolated along its edge's arbitrary
    half-edge direction, so the loop zigzagged between the two ends."""
    cyl = create_cylinder(radius=1.0, height=2.0, segments=8)
    target = None
    for e in cyl.edges:
        a = e.half_edge.prev.vertex.position
        b = e.half_edge.vertex.position
        if abs(a[1] - b[1]) > 1e-6:
            target = e.index
            break
    assert target is not None

    before = len(cyl.vertices)
    out = insert_edge_loop(cyl, target, position=0.25)
    new_ys = [float(v.position[1]) for v in out.vertices[before:]]
    assert len(new_ys) == 8, f"expected 8 new vertices, got {len(new_ys)}"
    spread = max(new_ys) - min(new_ys)
    assert spread < 1e-9, f"edge loop zigzags: new Y values {sorted(new_ys)}"
    # and it should sit a quarter of the way up, not half
    assert abs(abs(new_ys[0]) - 0.5) < 1e-9, f"loop at Y={new_ys[0]}, expected +/-0.5"


def test_insert_edge_loop_keeps_the_mesh_closed():
    cyl = create_cylinder(radius=1.0, height=2.0, segments=8)
    target = None
    for e in cyl.edges:
        a = e.half_edge.prev.vertex.position
        b = e.half_edge.vertex.position
        if abs(a[1] - b[1]) > 1e-6:
            target = e.index
            break
    assert target is not None
    out = insert_edge_loop(cyl, target, position=0.5)

    # An unmodified cylinder is already closed, so the closure checks only mean
    # something once the loop is known to be there: 8 new vertices around the
    # ring and each of the 8 wall quads split in two.
    assert len(out.vertices) == len(cyl.vertices) + 8, \
        f"expected 8 new vertices, got {len(out.vertices) - len(cyl.vertices)}"
    assert len(out.faces) == len(cyl.faces) + 8, \
        f"expected 8 extra faces, got {len(out.faces) - len(cyl.faces)}"

    assert dangling_half_edges(out) == 0
    assert solid_report(out)['watertight']
    # catches a split quad that came out mis-wound, which the pair above cannot
    assert_no_repeated_directed_edge(out)


def test_insert_edge_loop_is_symmetric_about_the_midpoint():
    """position=p and position=1-p must give mirrored loops, which only holds
    if the loop is oriented consistently.

    Compared per loop, not as a multiset: for an edge (a, b) the point at p is
    0.75*ya + 0.25*yb and at 1-p it is the swap, so flipping one edge moves a
    value from one list to the other and the multiset mirror survives every
    zigzag pattern. A single Y per loop is what 'consistently oriented' means.
    """
    def loop_y(pos):
        cyl = create_cylinder(radius=1.0, height=2.0, segments=8)
        target = next(e.index for e in cyl.edges
                      if abs(e.half_edge.prev.vertex.position[1]
                             - e.half_edge.vertex.position[1]) > 1e-6)
        before = len(cyl.vertices)
        out = insert_edge_loop(cyl, target, position=pos)
        return sorted(float(v.position[1]) for v in out.vertices[before:])

    lo, hi = loop_y(0.25), loop_y(0.75)
    assert len(lo) == len(hi) == 8, f"no loop was inserted: {len(lo)} / {len(hi)} points"
    # a 2.0-high cylinder runs from Y=-1 to Y=+1, so a quarter-way loop is flat
    # at -0.5 and its mirror flat at +0.5
    assert set(np.round(lo, 12)) == {-0.5}, f"loop at 0.25 zigzags: {lo}"
    assert set(np.round(hi, 12)) == {0.5}, f"loop at 0.75 zigzags: {hi}"
    assert np.allclose(lo, -np.array(hi)[::-1]), f"{lo} vs {hi}"


# --------------------------------------------------------------------------
# editing finding 9: mirror_mesh
# --------------------------------------------------------------------------

def test_mirror_mesh_drops_faces_lying_on_the_mirror_plane():
    """Finding 9: a face in the mirror plane got duplicated onto itself,
    producing a zero-thickness double wall."""
    plane = create_plane(width=2.0, height=2.0)   # lies exactly in y = 0
    out = mirror_mesh(plane, axis='y')
    assert len(out.faces) == 1, \
        f"face on the mirror plane was duplicated ({len(out.faces)} faces)"
    assert_no_repeated_directed_edge(out)

    # A one-face fixture cannot tell "dropped the on-plane face" from "mirrored
    # nothing at all". Mix the two cases: one quad in y = 0, one at y = 1.
    mixed = HalfEdgeMesh.from_arrays(
        [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1],
         [0, 1, 0], [1, 1, 0], [1, 1, 1], [0, 1, 1]],
        [[0, 1, 2, 3],        # lies in the mirror plane -> reflection dropped
         [4, 5, 6, 7]])       # off the plane -> must be reflected to y = -1
    out = mirror_mesh(mixed, axis='y')
    assert len(out.faces) == 3, \
        f"expected 2 originals + 1 reflection, got {len(out.faces)}"
    assert len(out.vertices) == 12, \
        f"expected 8 originals + 4 reflected, got {len(out.vertices)}"
    ys = sorted({round(float(v.position[1]), 12) for v in out.vertices})
    assert ys == [-1.0, 0.0, 1.0], f"reflection is missing or misplaced: {ys}"
    assert_no_repeated_directed_edge(out)


def test_mirror_mesh_of_a_half_solid_is_a_closed_solid():
    """Half a box, mirrored, must close up into a watertight box."""
    verts = [[0, -1, -1], [1, -1, -1], [1, 1, -1], [0, 1, -1],
             [0, -1, 1], [1, -1, 1], [1, 1, 1], [0, 1, 1]]
    faces = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4],
             [2, 3, 7, 6], [1, 2, 6, 5]]          # open at x = 0
    half = HalfEdgeMesh.from_arrays(verts, faces)
    out = mirror_mesh(half, axis='x')
    rep = solid_report(out)
    assert rep['watertight'], f"mirrored half-box is not closed: {rep}"
    assert rep['winding_consistent'], rep
    assert rep['volume'] == pytest.approx(8.0, rel=1e-9)


def test_mirror_mesh_keeps_geometry_when_nothing_is_on_the_plane():
    box = create_box()
    for v in box.vertices:
        v.position = v.position + np.array([5.0, 0.0, 0.0])
    out = mirror_mesh(box, axis='x')
    assert len(out.vertices) == 16
    assert len(out.faces) == 12


def test_mirror_mesh_is_not_quadratic():
    """Finding 9: the merge rebuilt an (N,3) array for every single vertex."""
    mesh = subdivide(create_box(), 4)
    assert len(mesh.vertices) > 1000
    t0 = time.perf_counter()
    out = mirror_mesh(mesh, axis='x')
    dt = time.perf_counter() - t0
    # `>= len(mesh.faces)` accepts a mirror that produced nothing as long as it
    # was fast. No face of a subdivided box lies wholly in x = 0, so every face
    # must have been reflected; the box is symmetric about x = 0, so every
    # mirrored vertex merges back onto an existing one.
    assert len(out.faces) == 2 * len(mesh.faces), \
        f"mirroring gave {len(out.faces)} faces, expected {2 * len(mesh.faces)}"
    assert len(out.vertices) == len(mesh.vertices), \
        f"merge failed: {len(out.vertices)} vertices, expected {len(mesh.vertices)}"
    assert dt < 5.0, f"mirroring {len(mesh.vertices)} vertices took {dt:.1f}s"


# --------------------------------------------------------------------------
# editing finding 9: bridge_faces must not destroy the input it cannot bridge
# --------------------------------------------------------------------------

def test_bridge_faces_leaves_the_mesh_alone_when_loops_do_not_match():
    """Finding 9: both face groups were deleted before the loop lengths were
    compared, so a mismatch silently punched two holes in the model."""
    cyl = create_cylinder(segments=8)
    wall, cap = 0, len(cyl.faces) - 1
    before_faces = len(cyl.faces)
    before_verts = len(cyl.vertices)
    before_positions = np.array([v.position for v in cyl.vertices])

    # Precondition: this request really is unbridgeable. If create_cylinder ever
    # changed so the two loops matched, the test would silently start measuring
    # the other branch and still pass.
    assert len(cyl.get_face_vertices(cyl.faces[wall])) == 4
    assert len(cyl.get_face_vertices(cyl.faces[cap])) == 8
    assert (len(cyl.get_face_vertices(cyl.faces[wall]))
            != len(cyl.get_face_vertices(cyl.faces[cap])))

    out = bridge_faces(cyl, [wall], [cap])

    assert len(out.faces) == before_faces, (
        f"unbridgeable request destroyed the input: {before_faces} faces in, "
        f"{len(out.faces)} out")
    assert len(out.vertices) == before_verts
    assert np.allclose(np.array([v.position for v in out.vertices]), before_positions)
    assert solid_report(out)['watertight'], "input was left with holes"

    # Positive control: refusing everything is also "not destroying the input",
    # so the refusal has to be shown to be selective.
    two_quads = HalfEdgeMesh.from_arrays(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
         [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
        [[0, 1, 2, 3], [7, 6, 5, 4]])
    bridged = bridge_faces(two_quads, [0], [1])
    assert len(bridged.faces) == 4, (
        "two matching 4-vertex loops must be bridged: expected the 2 caps "
        f"replaced by 4 side quads, got {len(bridged.faces)} faces")
    assert len(bridged.vertices) == 8


def test_bridge_faces_still_bridges_matching_loops():
    """The real use: join two separate solids into one closed shell.

    (Bridging two faces of the *same* box is degenerate - the bridge quads land
    on top of the existing side walls - so it is not a valid case to assert on.)
    """
    verts, faces = [], []
    for shift in ([0.0, 0.0, 0.0], [3.0, 0.0, 0.0]):
        box = create_box()
        offset = len(verts)
        verts.extend(v.position + np.array(shift) for v in box.vertices)
        faces.extend([v.index + offset for v in box.get_face_vertices(f)]
                     for f in box.faces)
    mesh = HalfEdgeMesh.from_arrays(verts, faces)
    assert len(mesh.faces) == 12

    out = bridge_faces(mesh, [0], [6])
    assert len(out.faces) == 14, "expected 12 - 2 deleted + 4 bridge quads"
    assert dangling_half_edges(out) == 0, "bridged shell is not closed"


# --------------------------------------------------------------------------
# editing finding 9: knife_cut must use the line it is given
# --------------------------------------------------------------------------

def test_knife_cut_depends_on_the_cut_line():
    """Finding 9: p1/p2 were ignored and the face was always split corner to
    corner, so every cut produced identical output."""
    box = create_box(width=2.0, height=2.0, depth=2.0)
    face = box.faces[0]
    fv = [v.index for v in box.get_face_vertices(face)]
    corners = np.array([box.vertices[i].position for i in fv])
    z = corners[0][2]

    horizontal = knife_cut(box, 0, np.array([-5.0, 0.0, z]), np.array([5.0, 0.0, z]))
    vertical = knife_cut(box, 0, np.array([0.0, -5.0, z]), np.array([0.0, 5.0, z]))

    h = [[v.index for v in horizontal.get_face_vertices(f)] for f in horizontal.faces]
    v = [[v.index for v in vertical.get_face_vertices(f)] for f in vertical.faces]
    assert h != v, "knife_cut ignores the cut line"


def test_knife_cut_splits_the_face_along_the_line():
    box = create_box(width=2.0, height=2.0, depth=2.0)
    face = box.faces[0]
    fv = [v.index for v in box.get_face_vertices(face)]
    z = box.vertices[fv[0]].position[2]

    before = len(box.faces)
    out = knife_cut(box, 0, np.array([-5.0, 0.0, z]), np.array([5.0, 0.0, z]))

    assert len(out.faces) == before + 1, "the cut face should become two faces"
    # the two new points must lie on the cut line (y = 0) and on the face plane
    added = [v.position for v in out.vertices[len(box.vertices):]]
    assert len(added) == 2, f"expected 2 new points on the cut, got {len(added)}"
    for p in added:
        assert abs(p[1]) < 1e-12, f"cut point {p} is not on the line y = 0"
        assert abs(p[2] - z) < 1e-12, f"cut point {p} left the face plane"


def test_knife_cut_keeps_the_mesh_watertight():
    """Splitting one face must not leave a T-junction against its neighbours."""
    box = create_box(width=2.0, height=2.0, depth=2.0)
    fv = [v.index for v in box.get_face_vertices(box.faces[0])]
    z = box.vertices[fv[0]].position[2]
    out = knife_cut(box, 0, np.array([-5.0, 0.0, z]), np.array([5.0, 0.0, z]))

    # The uncut box is already watertight with volume 8, and no cut can change
    # the volume, so every assertion below holds on the *input*. Pin the cut
    # itself first: one face becomes two, and two points are inserted.
    assert len(out.faces) == len(box.faces) + 1, \
        f"the cut face should become two faces ({len(out.faces)} vs {len(box.faces)})"
    assert len(out.vertices) == len(box.vertices) + 2, \
        f"expected 2 new points on the cut, got {len(out.vertices) - len(box.vertices)}"

    # trimesh's is_watertight only counts edge multiplicity, so the dangling
    # half-edge count is what actually rules out a T-junction here.
    assert dangling_half_edges(out) == 0, "knife cut opened the mesh"
    rep = solid_report(out)
    assert rep['watertight'], rep
    assert rep['volume'] == pytest.approx(8.0, rel=1e-9)


def test_knife_cut_that_misses_the_face_is_a_no_op():
    box = create_box()
    out = knife_cut(box, 0, np.array([10.0, 10.0, 10.0]), np.array([10.0, 11.0, 10.0]))
    assert len(out.faces) == len(box.faces)
    assert len(out.vertices) == len(box.vertices)


def test_knife_cut_out_of_range_face_is_a_no_op():
    box = create_box()
    out = knife_cut(box, 999, np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))
    assert len(out.faces) == len(box.faces)


# --------------------------------------------------------------------------
# untouched neighbours: make sure the edits did not break adjacent operations
# --------------------------------------------------------------------------

def test_inset_faces_still_works():
    box = create_box()
    out = inset_faces(box, [0], inset_amount=0.1)
    assert len(out.faces) == len(box.faces) + 4
    assert dangling_half_edges(out) == 0
