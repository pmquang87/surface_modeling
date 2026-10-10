"""Same-pattern follow-ups to the bug report ``bug-report.md`` (automated bug
hunt on commit 4e2241347de43175401df20c6722f38fa70820b4), found by the
integration review of the merged fix branch.

Each test names its item and the original bug it mirrors:

* R1 (mirrors M15, cand 7/8 - "small but valid faces get bad normals") -
  ``SubDToNURBSConverter._evaluate_dense_grid`` took the bump normal from
  ``cross(c1 - c0, c3 - c0)`` with the absolute test ``n_len > 1e-6`` and fell
  back to (0, 0, 1). A valid quad with edges below ~1e-3 (a mm-size cage
  modelled in metres) was bumped in-plane along z instead of along its normal,
  and a truly degenerate quad got an arbitrary +z bump.
* R2 (mirrors M14, cand 17 - "bridge_faces punches holes") - ``bridge_faces``
  still punched holes when the two groups shared a face (also through index
  aliases such as [4] and [-2] on a 6-face box), and for opposite faces of a
  box, whose new side quads coincide with the existing walls; and (found in
  the review of that fix) when the groups touch at a single corner (a
  bow-tie vertex with back-to-back side triangles) or share a chain of
  boundary edges (its inner vertices orphaned, the side faces folded or of
  zero area).

Test discipline (the repo's vacuous-test rules): each test first asserts that
the defect / precondition really is in its input, every checker gets an input
it must accept and one it must reject, and counts are bracketed from both
sides (what must be refused AND what must still be done).
"""
import os
import sys
from collections import Counter

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.halfedge_mesh import HalfEdgeMesh  # noqa: E402
from src.nurbs.converter import SubDToNURBSConverter  # noqa: E402
from src.subd.editing import bridge_faces  # noqa: E402
from src.subd.primitives import (  # noqa: E402
    create_box, create_cylinder, create_plane, create_torus)


# --------------------------------------------------------------------------
# R1 (mirrors M15): the dense-grid bump normal of a small quad
# --------------------------------------------------------------------------

def _xz_quad(side, origin=(0.0, 0.0, 0.0)):
    """A square in the x-z plane; its true normal is -y (c1-c0 = +x, c3-c0 = +z)."""
    p = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]], dtype=np.float64)
    p = p * side + np.asarray(origin, dtype=np.float64)
    m = HalfEdgeMesh.from_arrays(p, [[0, 1, 2, 3]])
    return m, p


def _grid(mesh, positions, grid_size=5):
    return SubDToNURBSConverter()._evaluate_dense_grid(
        mesh, mesh.faces[0], positions, grid_size=grid_size)


def _bilinear(corners, grid_size):
    c0, c1, c2, c3 = corners
    u = np.linspace(0.0, 1.0, grid_size)
    g = np.zeros((grid_size, grid_size, 3))
    for i, a in enumerate(u):
        for j, b in enumerate(u):
            g[i, j] = (1 - a) * (1 - b) * c0 + a * (1 - b) * c1 + a * b * c2 + (1 - a) * b * c3
    return g


def _old_dense_grid(mesh, face, limit_positions, grid_size):
    """Verbatim copy of the aec2331 algorithm, the 'before' of the R1 fix."""
    corners = [limit_positions[v.index] for v in mesh.get_face_vertices(face)]
    c0, c1, c2, c3 = corners
    normal = np.cross(c1 - c0, c3 - c0)
    n_len = np.linalg.norm(normal)
    if n_len > 1e-6:
        normal = normal / n_len
    else:
        normal = np.array([0, 0, 1])
    grid = np.zeros((grid_size, grid_size, 3))
    vals = np.linspace(0, 1, grid_size)
    for i, u in enumerate(vals):
        for j, v in enumerate(vals):
            p = (1-u)*(1-v)*c0 + u*(1-v)*c1 + u*v*c2 + (1-u)*v*c3
            bump = u * (1 - u) * v * (1 - v) * 4.0
            side_len = np.linalg.norm(c1 - c0)
            grid[i, j] = p + normal * (side_len * 0.1) * bump
    return grid


def test_r1_old_reference_reproduces_the_defect():
    """R1 (mirrors M15): the checker below must be able to say 'no'. The
    aec2331 algorithm bumps the 1e-4 quad along +z, in its own plane."""
    m, p = _xz_quad(1e-4)
    centre = _old_dense_grid(m, m.faces[0], p, 5)[2, 2] - p.mean(axis=0)
    np.testing.assert_allclose(centre / 1e-4, [0.0, 0.0, 0.025], atol=1e-12)


@pytest.mark.parametrize("side", [1e-6, 1e-4, 1e-2, 1e3])
def test_r1_dense_grid_is_scale_invariant(side):
    """R1 (mirrors M15): the bump of a valid quad goes along its normal at
    every scale. Expected: grid / side is the same for side 1 and `side`, and
    the centre is displaced by 0.025 * side along the face normal (-y)."""
    m1, p1 = _xz_quad(1.0)
    g1 = _grid(m1, p1)
    # precondition: at side 1 the bump exists and is along the true normal
    np.testing.assert_allclose(g1[2, 2] - p1.mean(axis=0), [0.0, -0.025, 0.0], atol=1e-15)

    m, p = _xz_quad(side)
    g = _grid(m, p)
    centre = (g[2, 2] - p.mean(axis=0)) / side
    np.testing.assert_allclose(centre, [0.0, -0.025, 0.0], atol=1e-12,
                               err_msg=f"side {side}: bump not along the face normal")
    np.testing.assert_allclose(g / side, g1, atol=1e-12)


def test_r1_degenerate_quad_gets_no_bump():
    """R1 (mirrors M15): a zero-area quad has no normal. Expected: zero
    displacement (the plain bilinear grid), not an arbitrary +z bump, at
    unit size and for a small exactly collinear quad far from the origin
    (dyadic steps, so P0 + t*d is exact, as in the M15 core test)."""
    lines = [
        np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=np.float64) * 0.5,
        np.array([1150.3, -20.7, 37.1])
        + np.array([t * np.array([3.0, -5.0, 7.0]) * 2.0 ** -17 for t in range(4)]),
    ]
    for p in lines:
        side = np.linalg.norm(p[1] - p[0])
        # precondition: exactly zero area
        assert np.array_equal(np.cross(p[1] - p[0], p[3] - p[0]), np.zeros(3))
        m = HalfEdgeMesh.from_arrays(p, [[0, 1, 2, 3]])
        # precondition: the aec2331 code does bump it, along +z
        old = _old_dense_grid(m, m.faces[0], p, 5)
        # (the bump peaks at 0.25 * 0.1 * side in the centre)
        assert old[2, 2, 2] - _bilinear(p, 5)[2, 2, 2] > 0.02 * side
        g = _grid(m, p)
        assert np.all(np.isfinite(g))
        np.testing.assert_array_equal(g, _bilinear(p, 5))

    # positive control: a valid quad is not mistaken for a degenerate one
    m, p = _xz_quad(1e-6, origin=(0.0, 0.0, 0.0))
    assert np.abs(_grid(m, p) - _bilinear(p, 5)).max() > 0.02 * 1e-6


def test_r1_ordinary_scale_grids_are_unchanged():
    """R1 (mirrors M15): at ordinary scale the GUI 'Convert to NURBS' output
    must not change. Expected: bit-identical to the aec2331 algorithm for
    every face of a limit-evaluated box and for random non-degenerate quads."""
    from src.subd.catmull_clark import evaluate_limit_surface
    box = create_box()
    lim, _ = evaluate_limit_surface(box)
    for f in box.faces:
        for gs in (5, 6, 17):
            new = SubDToNURBSConverter()._evaluate_dense_grid(box, f, lim, grid_size=gs)
            np.testing.assert_array_equal(new, _old_dense_grid(box, f, lim, gs))

    rng = np.random.default_rng(1234)
    for _ in range(50):
        p = rng.uniform(-2.0, 2.0, size=(4, 3))
        m = HalfEdgeMesh.from_arrays(p, [[0, 1, 2, 3]])
        if np.linalg.norm(np.cross(p[1] - p[0], p[3] - p[0])) < 1e-3:
            continue
        np.testing.assert_array_equal(_grid(m, p, 6), _old_dense_grid(m, m.faces[0], p, 6))


# --------------------------------------------------------------------------
# R2 (mirrors M14): bridge_faces on overlapping groups / coinciding walls
# --------------------------------------------------------------------------

def _faces(mesh):
    return [[v.index for v in mesh.get_face_vertices(f)] for f in mesh.faces]


def _directed(mesh):
    d = Counter()
    for fv in _faces(mesh):
        for i in range(len(fv)):
            d[(fv[i], fv[(i + 1) % len(fv)])] += 1
    return d


def _boundary(mesh):
    """Directed edges with no opposite partner: the open boundary."""
    d = _directed(mesh)
    return {k for k in d if (k[1], k[0]) not in d}


def _same(a, b):
    if len(a.vertices) != len(b.vertices) or _faces(a) != _faces(b):
        return False
    pa = np.array([v.position for v in a.vertices])
    pb = np.array([v.position for v in b.vertices])
    return np.array_equal(pa, pb)


def _link_components(mesh):
    """For each vertex, the number of connected pieces of its link: nodes are
    its neighbours, and each face corner at the vertex joins the corner's two
    neighbours. A manifold vertex has one piece (a path or a cycle), a pinched
    bow-tie vertex two or more, an unused vertex none."""
    links = {v: [] for v in range(len(mesh.vertices))}
    for fv in _faces(mesh):
        for k, v in enumerate(fv):
            links[v].append((fv[k - 1], fv[(k + 1) % len(fv)]))
    pieces = {}
    for v, link in links.items():
        adj = {}
        for p, q in link:
            adj.setdefault(p, set()).add(q)
            adj.setdefault(q, set()).add(p)
        seen, n = set(), 0
        for start in adj:
            if start in seen:
                continue
            n += 1
            stack = [start]
            while stack:
                u = stack.pop()
                if u not in seen:
                    seen.add(u)
                    stack.extend(adj[u] - seen)
        pieces[v] = n
    return pieces


def _sound_bridge(before, after) -> bool:
    """Every directed edge used at most once, the open boundary unchanged
    (none for a closed input), every face has all-distinct vertices, and every
    vertex is manifold: one fan of faces, and no vertex that had faces before
    is left without one (a single-face group has no inner vertex to drop)."""
    if max(_directed(after).values(), default=0) > 1:
        return False
    if any(len(set(fv)) != len(fv) for fv in _faces(after)):
        return False
    if _boundary(after) != _boundary(before):
        return False
    was_used = _link_components(before)
    pieces = _link_components(after)
    if any(c > 1 for c in pieces.values()):
        return False
    return all(pieces.get(v, 0) == 1 for v, c in was_used.items() if c > 0)


# Verbatim outputs of bridge_faces at 45087dd (the first R2 fix), kept as inputs
# the checker must reject.
_PLANE3_BOWTIE = [[1, 5, 6, 2], [2, 6, 7, 3], [4, 8, 9, 5], [6, 10, 11, 7], [8, 12, 13, 9],
                  [9, 13, 14, 10], [10, 14, 15, 11], [0, 4, 9, 10], [4, 5, 9], [5, 1, 6],
                  [1, 0, 10, 6]]       # plane 3x3, bridge_faces(m, [0], [4])
_BOX_ORPHAN = [[2, 3, 7, 6], [1, 2, 6, 5], [3, 2, 7], [2, 1, 6, 7],
               [1, 5, 6]]              # box, bridge_faces(m, [0, 2], [1, 4])


def test_r2_checkers_reject_broken_meshes():
    """The checkers must be able to say 'no'."""
    box = create_box()
    assert _sound_bridge(box, box) and _same(box, box.copy())
    holed = HalfEdgeMesh.from_arrays([v.position for v in box.vertices], _faces(box)[1:])
    assert not _sound_bridge(box, holed) and not _same(box, holed)
    doubled = HalfEdgeMesh.from_arrays([v.position for v in box.vertices],
                                       _faces(box) + [_faces(box)[0]])
    assert not _sound_bridge(box, doubled)
    moved = box.copy()
    moved.vertices[0].position = moved.vertices[0].position + 1e-9
    assert not _same(box, moved)

    # bow-tie: every directed edge once and the boundary unchanged, so only
    # the fan check can see that vertex 5 has two separate fans
    plane = create_plane(subdivisions_x=3, subdivisions_y=3)
    bowtie = HalfEdgeMesh.from_arrays([v.position for v in plane.vertices], _PLANE3_BOWTIE)
    assert _faces(bowtie) == _PLANE3_BOWTIE
    assert max(_directed(bowtie).values()) == 1 and _boundary(bowtie) == _boundary(plane)
    assert _link_components(bowtie)[5] == 2
    assert not _sound_bridge(plane, bowtie)

    # closed and every directed edge once, but vertices 0 and 4 lost all faces
    orphan = HalfEdgeMesh.from_arrays([v.position for v in box.vertices], _BOX_ORPHAN)
    assert max(_directed(orphan).values()) == 1 and not _boundary(orphan)
    assert max(_link_components(orphan).values()) == 1
    assert [v for v, c in _link_components(orphan).items() if c == 0] == [0, 4]
    assert not _sound_bridge(box, orphan)

    # a face that passes the same vertex twice (not consecutively); otherwise
    # sound, so only the all-distinct check rejects it
    pts = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
    pinched = HalfEdgeMesh.from_arrays(pts, [[0, 1, 2, 1, 3]])
    assert _faces(pinched) == [[0, 1, 2, 1, 3]]
    assert max(_directed(pinched).values()) == 1
    assert max(_link_components(pinched).values()) == 1
    assert not _sound_bridge(pinched, pinched)


@pytest.mark.parametrize("a,b", [([f], [f]) for f in range(6)] + [([4], [-2]), ([-6], [0]),
                                                                  ([0, 2], [2, 0])])
def test_r2_overlapping_groups_return_the_untouched_mesh(a, b):
    """R2 (mirrors M14): groups that share a face (after negative-index
    normalisation) cannot be bridged. The old code deleted the face and built
    side quads from it twice: 6 -> 9 faces, 6 open half-edges. Expected: the
    untouched mesh, as the docstring promises."""
    box = create_box()
    assert {i % 6 for i in a} & {i % 6 for i in b}, "precondition: groups overlap"
    out = bridge_faces(box, a, b)
    assert _same(out, box), (f"{len(box.faces)} faces in, {len(out.faces)} out, "
                             f"{sum(1 for he in out.half_edges if he.twin is None)} open half-edges")
    assert out is not box


def test_r2_every_ordered_pair_of_box_faces():
    """R2 (mirrors M14): for opposite faces of a box the four new side quads
    are the four existing walls, so the old code gave 6 -> 8 faces with 8 open
    half-edges. Expected for every ordered pair: a sound closed mesh or exactly
    the untouched box; and bracketed both ways, the 6 same-face and 6
    opposite-face pairs are refused while the 24 adjacent pairs still bridge."""
    box = create_box()
    fl = _faces(box)
    refused, bridged = set(), set()
    for a in range(6):
        for b in range(6):
            out = bridge_faces(box, [a], [b])
            if _same(out, box):
                refused.add((a, b))
            else:
                assert _sound_bridge(box, out), (a, b, _faces(out))
                bridged.add((a, b))
    opposite = {(a, b) for a in range(6) for b in range(6)
                if a != b and not set(fl[a]) & set(fl[b])}
    # precondition: opposite faces really span all 8 corners, so every wall's
    # vertex set is an edge of a joined to an edge of b
    assert len(opposite) == 6
    assert all(len(set(fl[a]) | set(fl[b])) == 8 for a, b in opposite)
    same = {(a, a) for a in range(6)}
    assert refused == same | opposite
    assert len(bridged) == 24


_SCAN_MESHES = {
    "plane3x3": lambda: create_plane(subdivisions_x=3, subdivisions_y=3),
    "plane4x2": lambda: create_plane(subdivisions_x=4, subdivisions_y=2),
    "plane3x1": lambda: create_plane(subdivisions_x=3, subdivisions_y=1),
    "plane1x3": lambda: create_plane(subdivisions_x=1, subdivisions_y=3),
    "torus8x4": lambda: create_torus(major_segments=8, minor_segments=4),
    "cylinder8": lambda: create_cylinder(segments=8),
}


@pytest.mark.parametrize("name", list(_SCAN_MESHES))
def test_r2_every_ordered_pair_of_single_faces(name):
    """R2 (mirrors M14) on open grids and closed meshes: every ordered pair of
    faces gives a sound bridge (boundary unchanged, every directed edge used at
    most once, every vertex one fan and none orphaned) or exactly the
    untouched mesh."""
    m = _SCAN_MESHES[name]()
    n = len(m.faces)
    fl = _faces(m)
    if name.startswith("plane"):
        nx, ny = (int(k) for k in name[5:].split("x"))
        assert n == nx * ny and len(_boundary(m)) == 2 * (nx + ny)
    else:
        assert not _boundary(m) and _sound_bridge(m, m)
    n_bridged = 0
    for a in range(n):
        for b in range(n):
            out = bridge_faces(m, [a], [b])
            if _same(out, m):
                continue
            assert _sound_bridge(m, out), (a, b, _faces(out))
            n_bridged += 1
    # both sides of the bracket: same-face pairs are always refused, and the
    # operation is not refusing everything (pairs of same-size faces that
    # share an edge still bridge; a cylinder cap and a side quad cannot)
    assert n_bridged <= n * n - n
    edge_sharing = sum(1 for a in range(n) for b in range(n)
                       if len(set(fl[a]) & set(fl[b])) == 2 and len(fl[a]) == len(fl[b]))
    assert n_bridged >= edge_sharing > 0


@pytest.mark.parametrize("make,a,b,edge_pair", [
    (lambda: create_plane(subdivisions_x=3, subdivisions_y=3), 0, 4, (0, 1)),
    (lambda: create_plane(subdivisions_x=3, subdivisions_y=3), 1, 3, (1, 4)),
    (lambda: create_torus(major_segments=8, minor_segments=4), 0, 5, (0, 1)),
    (lambda: create_box(subdivisions=1), 0, 2, (0, 1)),
], ids=["plane3x3-0-4", "plane3x3-1-3", "torus8x4-0-5", "box_sub1-0-2"])
def test_r2_corner_touching_faces_return_the_untouched_mesh(make, a, b, edge_pair):
    """R2 (mirrors M14): two faces that share exactly one corner. The side
    quads at that corner collapse to triangles lying back-to-back on kept
    faces, and the corner keeps two separate face fans (a bow-tie). Every
    directed edge is still used once, so the directed-edge check let it
    through (45087dd gave plane 3x3 (0, 4) a pinched vertex 5). Expected: the
    untouched mesh, both ways round; bracket: on the same mesh, two faces
    that share a full edge still bridge soundly."""
    m = make()
    fl = _faces(m)
    assert len(set(fl[a]) & set(fl[b])) == 1, "precondition: exactly one shared vertex"
    for x, y in ((a, b), (b, a)):
        out = bridge_faces(m, [x], [y])
        assert _same(out, m), (x, y, _faces(out), _link_components(out))
        assert out is not m
    ea, eb = edge_pair
    assert len(set(fl[ea]) & set(fl[eb])) == 2
    out = bridge_faces(m, [ea], [eb])
    assert not _same(out, m) and _sound_bridge(m, out)


@pytest.mark.parametrize("make,a,b,orphans", [
    (create_box, [0, 2], [1, 4], [0, 4]),
    (lambda: create_plane(subdivisions_x=3, subdivisions_y=3), [0, 1], [3, 4], [5]),
], ids=["box", "plane3x3"])
def test_r2_groups_sharing_an_edge_chain_return_the_untouched_mesh(make, a, b, orphans):
    """R2 (mirrors M14): two 2-face groups whose boundary loops share a chain
    of edges. The side quads along the chain collapse, the chain's inner
    vertices lose every face, and what is built is degenerate (box: triangle
    [3, 2, 7] folded onto [2, 3, 7, 6]; plane: the collinear triangle
    [0, 4, 8]). Expected: the untouched mesh."""
    m = make()
    fl = _faces(m)
    for v in orphans:
        # precondition: every face at v is in one of the groups, and both
        # groups touch v, so v is on both loops (not inside one group)
        at_v = {i for i, f in enumerate(fl) if v in f}
        assert at_v <= set(a + b) and at_v & set(a) and at_v & set(b)
    out = bridge_faces(m, a, b)
    assert _same(out, m), (_faces(out), _link_components(out))


def test_r2_inner_vertex_of_a_group_may_still_be_dropped():
    """R2 positive control for the orphan rule: two 2x2 patches facing each
    other on parallel 4x4 grids. Each patch's centre vertex is inside its group
    (on neither loop), so dropping it is the normal result of a bridge.
    Expected: 32 - 8 + 8 faces, the two grid rims as the only boundary, and
    every vertex one fan except the two dropped centres."""
    verts, faces = [], []
    for z, flip in ((0.0, False), (1.0, True)):
        off = len(verts)
        verts.extend([x, y, z] for y in range(5) for x in range(5))
        for j in range(4):
            for i in range(4):
                q = [off + 5 * j + i, off + 5 * j + i + 1,
                     off + 5 * (j + 1) + i + 1, off + 5 * (j + 1) + i]
                faces.append(q[::-1] if flip else q)
    m = HalfEdgeMesh.from_arrays(verts, faces)
    patch_a = [5, 6, 9, 10]
    patch_b = [16 + f for f in patch_a]
    centres = [12, 25 + 12]
    fl = _faces(m)
    # precondition: each centre is surrounded by its own patch only
    assert {i for i, f in enumerate(fl) if 12 in f} == set(patch_a)
    assert {i for i, f in enumerate(fl) if 37 in f} == set(patch_b)
    out = bridge_faces(m, patch_a, patch_b)
    assert not _same(out, m)
    assert len(out.faces) == 32 - 8 + 8 and len(out.vertices) == 50
    assert max(_directed(out).values()) == 1 and _boundary(out) == _boundary(m)
    pieces = _link_components(out)
    assert sorted(v for v, c in pieces.items() if c == 0) == centres
    assert all(c == 1 for v, c in pieces.items() if v not in centres)


def test_r2_facing_quads_on_parallel_grids_still_bridge():
    """R2 (mirrors M14) positive control: the centre cells of two parallel
    3x3 grids facing each other are genuinely bridgeable. Expected: 18 - 2 + 4
    faces, no new vertices, a sound bridge whose boundary is the two outer
    grid rims."""
    verts, faces = [], []
    for z, flip in ((0.0, False), (1.0, True)):
        off = len(verts)
        verts.extend([x, y, z] for y in range(4) for x in range(4))
        for j in range(3):
            for i in range(3):
                q = [off + 4 * j + i, off + 4 * j + i + 1,
                     off + 4 * (j + 1) + i + 1, off + 4 * (j + 1) + i]
                faces.append(q[::-1] if flip else q)
    m = HalfEdgeMesh.from_arrays(verts, faces)
    assert len(m.faces) == 18 and len(_boundary(m)) == 24

    out = bridge_faces(m, [4], [13])
    assert not _same(out, m)
    assert len(out.faces) == 20 and len(out.vertices) == 32
    assert _sound_bridge(m, out)


def test_r2_side_quad_on_an_existing_reversed_wall_is_refused():
    """R2 (mirrors M14): two facing quads plus a loose wall between them. The
    side quad over edge 0-1 is [0, 1, 5, 4]; the wall [4, 5, 1, 0] has the
    same corners wound the other way, so that quad reuses none of the wall's
    directed edges (its neighbour [1, 2, 6, 5] reuses 5 -> 1), and the old code
    glued a second face onto the wall. Expected: the untouched mesh; without
    the wall, the same request still bridges."""
    pts = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
           [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]
    two = HalfEdgeMesh.from_arrays(pts, [[0, 1, 2, 3], [7, 6, 5, 4]])
    bridged = bridge_faces(two, [0], [1])
    # precondition: the bridge builds exactly the side quad the wall reverses
    assert [0, 1, 5, 4] in _faces(bridged) and [1, 2, 6, 5] in _faces(bridged)
    assert _sound_bridge(two, bridged)

    walled = HalfEdgeMesh.from_arrays(pts, [[0, 1, 2, 3], [7, 6, 5, 4], [4, 5, 1, 0]])
    out = bridge_faces(walled, [0], [1])
    assert _same(out, walled), _faces(out)
