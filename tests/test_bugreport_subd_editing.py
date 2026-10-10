"""
Regression tests for the SubD editing ops in src/subd/editing.py.

They pin bugs from the automated bug hunt on commit
4e2241347de43175401df20c6722f38fa70820b4 (bug-report.md):

  H3  (cand 16)            insert_edge_loop leaves a T-junction in the non-quad
                           face that ends the ring (cylinder cap, sphere pole).
  M13 (cand 14, cand 15)   extrude_faces / inset_faces punch holes for a
                           negative or repeated face index; the same index
                           handling is pinned for extrude_edges and bridge_faces.
  M14 (cand 17)            bridge_faces bridges a group whose boundary is not
                           one simple closed loop, punching holes.
  S:mirror_axis            mirror_mesh silently mirrored across X for an axis
                           it did not recognise, such as 'Y'.

Topology is judged by checkers written here, independent of the module: every
undirected edge must be used exactly once in each direction, and the Euler
characteristic is counted from the face lists. Each checker is first shown a
broken mesh it has to reject.
"""

import os
import sys
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest

from src.core.halfedge_mesh import HalfEdgeMesh
from src.subd.primitives import (create_box, create_cylinder, create_plane,
                                 create_sphere, create_torus)
from src.subd.editing import (extrude_faces, extrude_edges, inset_faces,
                              insert_edge_loop, bridge_faces, mirror_mesh)


# --------------------------------------------------------------------------
# independent checkers
# --------------------------------------------------------------------------

def face_lists(mesh):
    return [[v.index for v in mesh.get_face_vertices(f)] for f in mesh.faces]


def directed_counts(mesh):
    d = Counter()
    for fv in face_lists(mesh):
        n = len(fv)
        for i in range(n):
            d[(fv[i], fv[(i + 1) % n])] += 1
    return d


def bad_edges(mesh) -> int:
    """Undirected edges NOT used exactly once in each direction.

    0 means closed and consistently wound; an open edge, a T-junction or a
    doubled face all count.
    """
    d = directed_counts(mesh)
    und = {tuple(sorted(k)) for k in d}
    return sum(1 for a, b in und if not (d[(a, b)] == 1 and d[(b, a)] == 1))


def open_half_edges(mesh) -> int:
    return sum(1 for he in mesh.half_edges if he.twin is None)


def euler(mesh) -> int:
    d = directed_counts(mesh)
    n_edges = len({tuple(sorted(k)) for k in d})
    used = {i for fv in face_lists(mesh) for i in fv}
    return len(used) - n_edges + len(mesh.faces)


def isolated_vertices(mesh) -> int:
    used = {i for fv in face_lists(mesh) for i in fv}
    return len(mesh.vertices) - len(used)


def same_mesh(a, b) -> bool:
    if len(a.vertices) != len(b.vertices) or face_lists(a) != face_lists(b):
        return False
    pa = np.array([v.position for v in a.vertices])
    pb = np.array([v.position for v in b.vertices])
    return np.allclose(pa, pb, atol=1e-12)


def test_checkers_reject_broken_meshes():
    """Every checker must be able to say 'no'."""
    box = create_box()
    assert bad_edges(box) == 0 and open_half_edges(box) == 0 and euler(box) == 2
    holed = HalfEdgeMesh.from_arrays([v.position for v in box.vertices], face_lists(box)[1:])
    assert bad_edges(holed) == 4
    assert open_half_edges(holed) == 4
    assert euler(holed) == 1
    # T-junction: a vertex inserted on edge (0,1) in only one of its two faces
    verts = [v.position for v in box.vertices] + [(box.vertices[0].position
                                                   + box.vertices[1].position) / 2]
    fl = face_lists(box)
    k = next(i for i, fv in enumerate(fl) if fv == [0, 1, 5, 4])
    fl[k] = [0, 8, 1, 5, 4]
    t = HalfEdgeMesh.from_arrays(verts, fl)
    assert bad_edges(t) == 3          # (0,8), (8,1) one-sided, (0,1) one-sided
    assert open_half_edges(t) == 3
    assert isolated_vertices(HalfEdgeMesh.from_arrays(verts, face_lists(box))) == 1
    assert not same_mesh(box, holed)
    assert same_mesh(box, create_box())


# --------------------------------------------------------------------------
# H3 (cand 16): insert_edge_loop at a non-quad face
# --------------------------------------------------------------------------

def _edge_between(mesh, pred):
    for e in mesh.edges:
        a = e.half_edge.prev.vertex.position
        b = e.half_edge.vertex.position
        if pred(a, b):
            return e
    raise AssertionError("no such edge")


def _ring(mesh, e):
    return list({x.index: x for x in mesh.get_edge_loop(e)}.values())


def _midpoints(mesh, ring):
    return sorted(tuple(np.round((x.half_edge.prev.vertex.position
                                  + x.half_edge.vertex.position) / 2, 12)) for x in ring)


def test_insert_edge_loop_through_cylinder_cap_stays_closed():
    """H3 (cand 16): a ring that ends at the cylinder's n-gon caps must put its
    new vertex into the caps too. Expected: 0 open half-edges, consistent
    winding, Euler 2, +2 vertices at the rim-edge midpoints, +1 face, and both
    8-gon caps become 9-gons."""
    cyl = create_cylinder(radius=1.0, height=2.0, segments=8)
    rim = _edge_between(cyl, lambda a, b: abs(a[1] + 1) < 1e-9 and abs(b[1] + 1) < 1e-9)
    ring = _ring(cyl, rim)
    # precondition: the ring is the bottom rim edge and the top rim edge above
    # it, ending in the two caps
    assert len(ring) == 2
    assert sorted(len(fv) for fv in face_lists(cyl)).count(8) == 2
    assert bad_edges(cyl) == 0 and euler(cyl) == 2

    out = insert_edge_loop(cyl, rim.index)

    assert len(out.vertices) == len(cyl.vertices) + 2
    assert len(out.faces) == len(cyl.faces) + 1
    new_pts = sorted(tuple(np.round(v.position, 12)) for v in out.vertices[len(cyl.vertices):])
    assert new_pts == _midpoints(cyl, ring)
    assert open_half_edges(out) == 0
    assert bad_edges(out) == 0
    assert euler(out) == 2
    assert isolated_vertices(out) == 0
    sizes = Counter(len(fv) for fv in face_lists(out))
    assert sizes == Counter({4: 9, 9: 2}), sizes


def test_insert_edge_loop_through_sphere_poles_stays_closed():
    """H3 (cand 16): a meridian ring of a UV sphere ends at two pole triangles,
    which must each gain the new vertex (becoming quads). Expected: +5 vertices
    at the latitude-edge midpoints, +4 faces (4 quads split), 14 triangles left,
    0 open half-edges, consistent winding, Euler 2."""
    sph = create_sphere(radius=1.0, segments=8, rings=6)
    pole_y = 1.0
    lat = _edge_between(sph, lambda a, b: abs(a[1] - b[1]) < 1e-9
                        and abs(abs(a[1]) - pole_y) > 1e-6)
    ring = _ring(sph, lat)
    assert len(ring) == 5
    assert Counter(len(fv) for fv in face_lists(sph)) == Counter({3: 16, 4: 32})
    assert bad_edges(sph) == 0 and euler(sph) == 2

    out = insert_edge_loop(sph, lat.index)

    assert len(out.vertices) == len(sph.vertices) + 5
    assert len(out.faces) == len(sph.faces) + 4
    new_pts = sorted(tuple(np.round(v.position, 12)) for v in out.vertices[len(sph.vertices):])
    assert new_pts == _midpoints(sph, ring)
    assert open_half_edges(out) == 0
    assert bad_edges(out) == 0
    assert euler(out) == 2
    assert Counter(len(fv) for fv in face_lists(out)) == Counter({3: 14, 4: 38})


@pytest.mark.parametrize("name, make", [
    ("box", create_box),
    ("cylinder", create_cylinder),
    ("sphere", create_sphere),
    ("torus", lambda: create_torus(major_segments=8, minor_segments=4)),
])
def test_insert_edge_loop_on_every_edge_of_the_primitives(name, make):
    """H3 (cand 16): for every edge of every closed primitive the result must be
    closed and consistently wound with Euler preserved and no stray vertex;
    one new vertex per ring edge, and one extra face per quad the ring crosses
    (so exactly the ring size for a closed ring of quads)."""
    m = make()
    chi = euler(m)
    assert bad_edges(m) == 0
    failures = []
    for e in m.edges:
        ring = _ring(m, e)
        ring_idx = {x.index for x in ring}
        touched = {f.index for x in ring for f in m.get_edge_faces(x) if f is not None}
        quads = sum(1 for fi in touched if len(m.get_face_vertices(m.faces[fi])) == 4)
        all_quads = quads == len(touched)
        out = insert_edge_loop(m, e.index)
        got = (len(out.vertices) - len(m.vertices), len(out.faces) - len(m.faces),
               bad_edges(out), open_half_edges(out), euler(out), isolated_vertices(out))
        want = (len(ring_idx), len(ring_idx) if all_quads else quads, 0, 0, chi, 0)
        if got != want:
            failures.append((e.index, got, want))
    assert not failures, f"{name}: {len(failures)} bad edges, first {failures[:3]}"


def test_insert_edge_loop_into_an_ngon_with_two_split_edges():
    """H3 (cand 16): a horseshoe of quads whose two ends both run into the same
    hexagon puts TWO new vertices into that hexagon, each between its own
    edge's endpoints. Expected: hexagon -> octagon, boundary unchanged, no
    interior crack, Euler preserved."""
    k = 6                     # radial edges r0..r5, quads between r_i and r_i+1
    verts, faces = [], []
    for i in range(k):
        th = np.pi * 1.5 * i / (k - 1)
        verts.append([np.cos(th), np.sin(th), 0.0])            # inner  2i
        verts.append([2 * np.cos(th), 2 * np.sin(th), 0.0])    # outer  2i+1
    for i in range(k - 1):
        faces.append([2 * i, 2 * i + 1, 2 * i + 3, 2 * i + 2])
    th = np.pi * 1.75
    verts.append([np.cos(th), np.sin(th), 0.0])                # m_in   2k
    verts.append([2 * np.cos(th), 2 * np.sin(th), 0.0])        # m_out  2k+1
    last_in, last_out = 2 * (k - 1), 2 * (k - 1) + 1
    hexagon = [last_in, last_out, 2 * k + 1, 1, 0, 2 * k]
    faces.append(hexagon)
    m = HalfEdgeMesh.from_arrays(verts, faces)

    # precondition: the hexagon twins with both end quads, and the ring from a
    # middle radial edge covers all k radial edges, ending in the hexagon twice
    d = directed_counts(m)
    interior = [k_ for k_ in d if (k_[1], k_[0]) in d]
    assert len(interior) == 2 * k
    mid = next(e for e in m.edges
               if {e.half_edge.prev.vertex.index, e.half_edge.vertex.index} == {4, 5})
    ring = _ring(m, mid)
    assert len(ring) == k
    hex_face = m.faces[len(faces) - 1]
    assert sum(1 for x in m.get_face_edges(hex_face) if x.index in {r.index for r in ring}) == 2
    boundary_before = open_half_edges(m)
    chi = euler(m)

    out = insert_edge_loop(m, mid.index)

    assert len(out.vertices) == len(m.vertices) + k
    assert len(out.faces) == len(m.faces) + (k - 1)
    assert open_half_edges(out) == boundary_before
    d2 = directed_counts(out)
    assert all(c == 1 for c in d2.values())
    # interior edges: the old 2k plus k-1 new ones across the split quads, plus
    # the halves of the k split radial edges (each split edge becomes two)
    assert sum(1 for k_ in d2 if (k_[1], k_[0]) in d2) == 2 * (2 * k + (k - 1))
    assert euler(out) == chi
    octagon = [fv for fv in face_lists(out) if len(fv) == 8]
    assert len(octagon) == 1
    # each new vertex sits between the two endpoints of the edge it split
    oc = octagon[0]
    for a, b in [(last_in, last_out), (1, 0)]:
        ia = oc.index(a)
        between = oc[(ia + 1) % 8]
        assert oc[(ia + 2) % 8] == b
        p = out.vertices[between].position
        assert np.allclose(p, (np.array(verts[a]) + np.array(verts[b])) / 2, atol=1e-12)
