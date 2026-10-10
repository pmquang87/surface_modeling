"""R4: the no-reference NURBS conversion samples the TRUE limit surface.

``SubDToNURBSConverter.generate_patches`` without a reference mesh (the GUI
"Convert to NURBS" path for SubD modelling) took its corners from
``evaluate_limit_surface`` (crease-aware), but the dense data grid from
``_evaluate_dense_grid``: bilinear interpolation of the corner limit positions
plus a bump of 0.025 * |c1 - c0| along the face normal at the centre. The bump
ignores creases and is only a crude stand-in for the smooth limit surface too.
A box whose edges are all fully creased has the limit cube [-0.5, 0.5]^3
(volume 1); it converted to volume 1.0667.

Fix: subdivide the cage 2 levels with the repo's own ``_subdivide_once``,
trace each quad's 5 x 5 sub-grid (``subdivide_with_quad_lineage``), take the
limit positions there (exact limit points at the dyadic parameters
(i, j) / 4 of the face) and resample them to the fitter's 6 x 6 grid by
tensor quartic Lagrange interpolation (exact where the limit is a bicubic
polynomial on the face, i.e. away from extraordinary vertices, and on planar
faces).

The true limit surface used as the reference here is the cage subdivided 5
levels (4 for the sphere, torus and dart cages, to keep the tests fast) with
every vertex moved to its limit position; dense samples at non-dyadic
parameters are compared with its triangulation, so their distance includes
the triangulation's chord sag (measured below 2e-4 of the bounding-box
diagonal at those levels).

G0 is used for every exactness claim: it has no continuity equations, so the
patch interiors depend on the dense grid alone. G1/G2 checks use loose
tolerances (what those levels enforce is being changed elsewhere).

Test discipline (the repo's vacuous-test rules): each test first asserts that
the defect / precondition is really in its input, every checker gets an input
it must reject, and counts are bracketed from both sides.
"""
import functools
import os
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.halfedge_mesh import HalfEdgeMesh  # noqa: E402
from src.nurbs import g3_fitter  # noqa: E402
from src.nurbs.converter import SubDToNURBSConverter  # noqa: E402
from src.subd.catmull_clark import subdivide, evaluate_limit_surface  # noqa: E402
from src.subd.primitives import (  # noqa: E402
    create_box, create_cylinder, create_sphere, create_torus)

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BASE = "1f93473"   # the commit before this fix


# --------------------------------------------------------------------------
# cages
# --------------------------------------------------------------------------

def creased_box():
    m = create_box()
    for e in m.edges:
        e.crease_weight = 1.0
    return m


def loop_box():
    """The four edges round the top face creased: every top vertex has two
    crease edges (a crease curve) and one smooth one."""
    m = create_box()
    top = {2, 3, 7, 6}
    for e in m.edges:
        if {e.half_edge.prev.vertex.index, e.half_edge.vertex.index} <= top:
            e.crease_weight = 1.0
    return m


def dart_box():
    """One creased edge, from a face-centre vertex to an edge-point vertex of
    a once-subdivided box: both its ends are darts (the crease fades out
    inside a smooth surface)."""
    m = create_box(subdivisions=1)
    for e in m.edges:
        lo, hi = sorted((e.half_edge.prev.vertex.index, e.half_edge.vertex.index))
        if 8 <= lo < 20 and hi >= 20:     # edge point (8..19) - face point (20..25)
            e.crease_weight = 1.0
            return m
    raise AssertionError("no edge-point to face-point edge")


CASES = {
    "smooth_box": (create_box, 5),
    "sphere": (create_sphere, 4),
    "torus": (lambda: create_torus(major_segments=8, minor_segments=4), 4),
    "creased_box": (creased_box, 5),
    "loop_box": (loop_box, 5),
    "dart_box": (dart_box, 4),
}


def _crease_counts(mesh):
    """Number of creased edges at each vertex."""
    n = {v.index: 0 for v in mesh.vertices}
    for e in mesh.edges:
        if e.crease_weight > 0:
            n[e.half_edge.prev.vertex.index] += 1
            n[e.half_edge.vertex.index] += 1
    return n


@functools.lru_cache(maxsize=None)
def _reference(name):
    """(limit triangulation, limit vertex positions, bbox diagonal)."""
    import trimesh
    make, levels = CASES[name]
    fine = subdivide(make(), levels)
    lim, _ = evaluate_limit_surface(fine)
    tris = []
    for f in fine.faces:
        q = [v.index for v in fine.get_face_vertices(f)]
        assert len(q) == 4
        tris += [[q[0], q[1], q[2]], [q[0], q[2], q[3]]]
    ref = trimesh.Trimesh(vertices=lim, faces=np.array(tris), process=False)
    diag = float(np.linalg.norm(lim.max(axis=0) - lim.min(axis=0)))
    return ref, lim, diag


def _distance(ref, pts):
    import trimesh
    _, d, _ = trimesh.proximity.closest_point(ref, np.asarray(pts).reshape(-1, 3))
    return np.asarray(d)


def _fitter_input(monkeypatch, mesh, continuity="G0", **kw):
    """generate_patches(mesh), returning (patches, the quads it fitted)."""
    seen = {}
    orig = g3_fitter.G3Fitter.fit_surface

    def spy(self, quad_mesh):
        seen["quads"] = quad_mesh
        return orig(self, quad_mesh)

    monkeypatch.setattr(g3_fitter.G3Fitter, "fit_surface", spy)
    patches = SubDToNURBSConverter(continuity=continuity).generate_patches(mesh, **kw)
    monkeypatch.undo()
    return patches, seen["quads"]


def _heuristic_grids(mesh, grid_size=6):
    """The 1f93473 dense grids (the bump heuristic, still reachable as the
    fallback _evaluate_dense_grid)."""
    lim, _ = evaluate_limit_surface(mesh)
    conv = SubDToNURBSConverter()
    return [conv._evaluate_dense_grid(mesh, f, lim, grid_size=grid_size)
            for f in mesh.faces if len(mesh.get_face_vertices(f)) == 4]


def _volume_and_bbox(patches):
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    shape = SubDToNURBSConverter().build_shape(patches)
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, props)
    box = Bnd_Box()
    BRepBndLib.AddOptimal_s(shape, box, False, False)
    return props.Mass(), np.array(box.Get()), shape


# --------------------------------------------------------------------------
# the fully creased box is the unit cube
# --------------------------------------------------------------------------

def test_fully_creased_box_converts_to_the_unit_cube(monkeypatch):
    """R4: every edge of the box creased (weight 1) makes the limit surface
    the cube [-0.5, 0.5]^3. Expected (G0): volume 1 and bbox +-0.5. On
    1f93473 the corners were already right, but the bump bulged all six
    faces: volume 1.0667, bbox +-0.523.

    Tolerance 1e-9 on the volume and 1e-7 on the bbox: the new dense samples
    lie in the face planes to rounding (checked below), so every patch is
    planar and the only error left is OCC's Gauss integration of six
    planar Bezier faces and its bbox gap, both far below 1e-9 / 1e-7 and
    all four to six orders of magnitude below the 0.0667 / 0.023 defect."""
    mesh = creased_box()
    lim, _ = evaluate_limit_surface(mesh)
    # precondition: the corners are right, so only the dense grid can be wrong
    np.testing.assert_allclose(np.abs(lim), 0.5, atol=1e-15)
    # precondition: the heuristic grid really bulges off the face planes
    heur = np.array(_heuristic_grids(mesh))
    assert np.abs(heur).max() > 0.52

    patches, quads = _fitter_input(monkeypatch, mesh, "G0")
    vol, bbox, _ = _volume_and_bbox(patches)
    assert abs(vol - 1.0) < 1e-9, f"volume {vol} (1f93473: 1.0667)"
    np.testing.assert_allclose(bbox, [-0.5, -0.5, -0.5, 0.5, 0.5, 0.5], atol=1e-7)

    dense = np.array([q["dense_points"] for q in quads])
    assert dense.shape == (6, 6, 6, 3)
    # every sample on the cube's surface: max |coord| = 0.5 at each point,
    # and so every control point of the fitted patches
    np.testing.assert_allclose(np.abs(dense).max(axis=-1), 0.5, atol=1e-15)
    np.testing.assert_allclose(np.abs(np.array(patches)).max(axis=-1), 0.5, atol=1e-15)


def test_fully_creased_box_with_continuity_rows_stays_close(monkeypatch):
    """R4, default continuity (loose: what G1/G2 enforce is changed
    elsewhere). Expected: volume within 1 % of 1, i.e. well below the
    6.7 % of 1f93473."""
    patches, _ = _fitter_input(monkeypatch, creased_box(), "G2")
    vol, bbox, _ = _volume_and_bbox(patches)
    assert abs(vol - 1.0) < 1e-2, vol
    assert np.abs(bbox).max() < 0.505, bbox


# --------------------------------------------------------------------------
# dense grids lie on the limit surface
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", list(CASES))
def test_dense_grid_lies_on_the_limit_surface(monkeypatch, name):
    """R4: the samples generate_patches hands to the fitter are within
    1e-3 of the bbox diagonal of the true limit surface (5-level reference;
    4 for sphere, torus, dart). Measured at 1f93473 (the bump): 2.3e-2 (creased
    box) to 1.5e-1 (smooth and loop box) absolute; now 1.7e-4 (sphere) to
    4.3e-4 (torus), i.e. 1.1e-4 to 2.1e-4 of the diagonal: reference chord
    sag (all of it on the regular torus) plus the quartic resampling near
    extraordinary vertices. On the creased box the samples are exact.
    Fails on 1f93473 for every case (the smooth ones too)."""
    mesh = CASES[name][0]()
    ref, _, diag = _reference(name)
    bound = 1e-3 * diag

    if name == "loop_box":
        # precondition: a crease curve (2-crease vertices), not corners
        assert sorted(set(_crease_counts(mesh).values())) == [0, 2]
    if name == "dart_box":
        assert sorted(_crease_counts(mesh).values())[-2:] == [1, 1]
        assert sum(_crease_counts(mesh).values()) == 2

    n_quads = sum(1 for f in mesh.faces if len(mesh.get_face_vertices(f)) == 4)
    # precondition: the checker can say no - the 1f93473 grid is far off
    heur = np.array(_heuristic_grids(mesh))
    assert _distance(ref, heur).max() > 5 * bound

    _, quads = _fitter_input(monkeypatch, mesh, "G0")
    dense = np.array([q["dense_points"] for q in quads])
    # bracket: one 6 x 6 grid per quad, no more and no less
    assert dense.shape == (n_quads, 6, 6, 3)
    d = _distance(ref, dense)
    assert d.max() < bound, f"{name}: max {d.max():.3e}, bound {bound:.3e}"
    if name == "creased_box":
        assert d.max() < 1e-12


@pytest.mark.parametrize("name", ["smooth_box", "creased_box", "loop_box", "dart_box"])
def test_dyadic_samples_are_exact_limit_points(name):
    """R4: without resampling (grid_size = 2**levels + 1) every sample IS a
    limit point: it coincides with a vertex of the 4/5-level limit mesh to
    rounding (the dyadic vertex survives all further levels and its limit
    position does not depend on the level it is evaluated at). Control:
    the resampled 6 x 6 grid of the smooth box is not on vertices."""
    from scipy.spatial import cKDTree
    mesh = CASES[name][0]()
    _, lim, diag = _reference(name)
    tree = cKDTree(lim)
    conv = SubDToNURBSConverter()
    for levels in (1, 2):
        n = 2 ** levels + 1
        grids = conv._limit_dense_grids(mesh, grid_size=n, levels=levels)
        pts = np.array(list(grids.values())).reshape(-1, 3)
        assert pts.shape[0] == len(mesh.faces) * n * n
        d, _ = tree.query(pts)
        assert d.max() < 1e-12 * diag, (levels, d.max())
    if name == "smooth_box":
        six = np.array(list(conv._limit_dense_grids(mesh, grid_size=6).values()))
        d, _ = tree.query(six.reshape(-1, 3))
        assert d.max() > 1e-3 * diag


# --------------------------------------------------------------------------
# quad lineage
# --------------------------------------------------------------------------

def _cycle_key(idx):
    """A face's vertex cycle, rotated to start at its smallest index."""
    k = idx.index(min(idx))
    return tuple(idx[k:] + idx[:k])


def _lineage_ok(coarse, fine, lineage, levels, coarse_limit=None, fine_limit=None):
    """True iff `lineage` maps every quad of `coarse` onto its own sub-grid of
    `fine`: one grid per quad and none for other faces, shape 2**k + 1,
    corners [0,0], [n,0], [n,n], [0,n] = the quad's c0..c3, every grid cell a
    face of `fine` with the cell's cyclic order (so the winding and the u/v
    axes are right), no fine face claimed twice, and - when limit positions
    are given - the corner samples at the quad's own corner limit positions."""
    n = 2 ** levels + 1
    quads = {f.index: [v.index for v in coarse.get_face_vertices(f)]
             for f in coarse.faces if len(coarse.get_face_vertices(f)) == 4}
    if set(lineage) != set(quads):
        return False
    fine_faces = {_cycle_key([v.index for v in fine.get_face_vertices(f)]) for f in fine.faces}
    claimed = set()
    for fidx, g in lineage.items():
        g = np.asarray(g)
        if g.shape != (n, n) or len(set(g.ravel().tolist())) != n * n:
            return False
        c0, c1, c2, c3 = quads[fidx]
        if [g[0, 0], g[-1, 0], g[-1, -1], g[0, -1]] != [c0, c1, c2, c3]:
            return False
        for i in range(n - 1):
            for j in range(n - 1):
                key = _cycle_key([int(g[i, j]), int(g[i + 1, j]),
                                  int(g[i + 1, j + 1]), int(g[i, j + 1])])
                if key not in fine_faces or key in claimed:
                    return False
                claimed.add(key)
        if coarse_limit is not None:
            corners = fine_limit[[g[0, 0], g[-1, 0], g[-1, -1], g[0, -1]]]
            scale = np.abs(coarse_limit).max() + 1.0
            if np.abs(corners - coarse_limit[[c0, c1, c2, c3]]).max() > 1e-13 * scale:
                return False
    return True


LINEAGE_MESHES = {
    "box": create_box,
    "creased_box": creased_box,
    "sphere": create_sphere,        # triangle fans at both poles
    "cylinder": create_cylinder,    # two octagon caps
    "torus": lambda: create_torus(major_segments=8, minor_segments=4),
}


@pytest.mark.parametrize("name", list(LINEAGE_MESHES))
@pytest.mark.parametrize("levels", [0, 1, 2, 3])
def test_quad_lineage_maps_every_quad_onto_its_own_sub_grid(name, levels):
    """R4: subdivide_with_quad_lineage traces each quad's (2**k+1)^2 vertex
    grid through k levels, also next to triangles and n-gons (sphere poles,
    cylinder caps). Expected: the checker accepts it, the corner samples sit
    at the quad's own corner limit positions, and - bracketed both ways -
    the grids claim exactly 4**k fine faces per quad while the fine faces
    left over are exactly the sub-faces of the non-quads (4**(k-1) per side)."""
    from src.subd.catmull_clark import subdivide_with_quad_lineage
    mesh = LINEAGE_MESHES[name]()
    fine, lineage = subdivide_with_quad_lineage(mesh, levels)
    # the returned mesh is the ordinary subdivision
    ref = subdivide(mesh, levels)
    assert [[v.index for v in fine.get_face_vertices(f)] for f in fine.faces] == \
        [[v.index for v in ref.get_face_vertices(f)] for f in ref.faces]
    assert np.array_equal(np.array([v.position for v in fine.vertices]),
                          np.array([v.position for v in ref.vertices]))

    coarse_lim, _ = evaluate_limit_surface(mesh)
    fine_lim, _ = evaluate_limit_surface(fine)
    assert _lineage_ok(mesh, fine, lineage, levels, coarse_lim, fine_lim)

    sides = [len(mesh.get_face_vertices(f)) for f in mesh.faces]
    n_quads = sides.count(4)
    other_sides = sum(s for s in sides if s != 4)
    if name in ("sphere", "cylinder"):
        assert n_quads < len(sides) and other_sides > 0   # precondition: non-quads
    assert len(lineage) == n_quads
    if levels > 0:
        assert len(fine.faces) - 4 ** levels * n_quads == 4 ** (levels - 1) * other_sides


def test_quad_lineage_checker_rejects_wrong_mappings():
    """The checker must be able to say no: a transposed grid (u and v
    swapped, winding reversed), two interior samples swapped, two quads'
    grids exchanged, a quad missing, and correct indices whose corner
    positions are compared with another quad's corners."""
    from src.subd.catmull_clark import subdivide_with_quad_lineage
    mesh = create_box()
    fine, lineage = subdivide_with_quad_lineage(mesh, 2)
    lim0, _ = evaluate_limit_surface(mesh)
    lim2, _ = evaluate_limit_surface(fine)
    assert _lineage_ok(mesh, fine, lineage, 2, lim0, lim2)

    def changed(edits):
        out = {k: v.copy() for k, v in lineage.items()}
        out.update(edits)
        return out

    assert not _lineage_ok(mesh, fine, changed({0: lineage[0].T.copy()}), 2)
    swapped = lineage[1].copy()
    swapped[1, 1], swapped[1, 2] = swapped[1, 2], swapped[1, 1]
    assert not _lineage_ok(mesh, fine, changed({1: swapped}), 2)
    assert not _lineage_ok(mesh, fine, changed({2: lineage[3], 3: lineage[2]}), 2)
    missing = changed({})
    del missing[5]
    assert not _lineage_ok(mesh, fine, missing, 2)
    # a rotated grid keeps every cell a face of the right winding but moves
    # the corners off c0..c3
    assert not _lineage_ok(mesh, fine, changed({4: np.rot90(lineage[4]).copy()}), 2)
    # right indices, wrong corner positions
    assert not _lineage_ok(mesh, fine, lineage, 2, lim0[::-1].copy(), lim2)


def test_quad_lineage_refuses_an_unrecognised_numbering(monkeypatch, capsys):
    """The helper checks every cell against the mesh it refines, so if
    _subdivide_once ever numbers its output differently it raises instead
    of scrambling the grids, and generate_patches falls back to the
    heuristic grid with a message."""
    from src.subd import catmull_clark as cc
    orig = cc._subdivide_once

    def reversed_vertices(mesh, smooth=True):
        out = orig(mesh, smooth)
        n = len(out.vertices)
        return HalfEdgeMesh.from_arrays(
            [out.vertices[n - 1 - i].position for i in range(n)],
            [[n - 1 - v.index for v in out.get_face_vertices(f)] for f in out.faces])

    monkeypatch.setattr(cc, "_subdivide_once", reversed_vertices)
    with pytest.raises(ValueError, match="quad lineage"):
        cc.subdivide_with_quad_lineage(create_box(), 1)
    mesh = creased_box()
    # (the renumbering patch is still active during this conversion;
    # _fitter_input's undo() removes it afterwards)
    patches, quads = _fitter_input(monkeypatch, mesh, "G0")
    assert cc._subdivide_once is orig
    assert "limit-surface sampling failed" in capsys.readouterr().out
    heur = _heuristic_grids(mesh)
    assert all(np.array_equal(q["dense_points"], h) for q, h in zip(quads, heur))


# --------------------------------------------------------------------------
# the reference-fit path is untouched
# --------------------------------------------------------------------------

def _base_converter_class():
    try:
        src = subprocess.run(["git", "-C", REPO, "show", f"{BASE}:src/nurbs/converter.py"],
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as e:
        pytest.skip(f"cannot read {BASE}'s converter.py: {e}")
    ns = {"__name__": "converter_at_base"}
    exec(compile(src, f"converter.py@{BASE}", "exec"), ns)
    return ns["SubDToNURBSConverter"]


@pytest.mark.parametrize("continuity", ["G0", "G2"])
def test_reference_path_is_bit_identical_to_base(monkeypatch, continuity):
    """R4: with a reference mesh the dense samples come from projecting onto
    it, as before. Expected: patches bit-identical to 1f93473's converter
    (its source, run against the same fitter), and the limit sampling is
    never called. Control: without the reference mesh the two converters do
    differ, so the comparison can fail."""
    Base = _base_converter_class()
    cage = create_box(subdivisions=1)
    for e in cage.edges[:6]:
        e.crease_weight = 1.0
    target = subdivide(create_box(), 3)

    calls = []
    orig = SubDToNURBSConverter._limit_dense_grids
    monkeypatch.setattr(SubDToNURBSConverter, "_limit_dense_grids",
                        lambda self, *a, **k: calls.append(1) or orig(self, *a, **k))
    new = SubDToNURBSConverter(continuity=continuity).generate_patches(
        cage.copy(), reference_mesh=target)
    old = Base(continuity=continuity).generate_patches(cage.copy(), reference_mesh=target)
    assert calls == []
    assert len(new) == len(old) == len(cage.faces)
    for a, b in zip(new, old):
        assert np.array_equal(a, b)

    new_free = SubDToNURBSConverter(continuity=continuity).generate_patches(cage.copy())
    old_free = Base(continuity=continuity).generate_patches(cage.copy())
    assert calls == [1]
    assert max(np.abs(a - b).max() for a, b in zip(new_free, old_free)) > 1e-3


# --------------------------------------------------------------------------
# R1 properties for the new grid: scale invariance, no spurious bump
# --------------------------------------------------------------------------

def _xz_quad(side, origin=(0.0, 0.0, 0.0)):
    p = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]], dtype=np.float64)
    p = p * side + np.asarray(origin, dtype=np.float64)
    return HalfEdgeMesh.from_arrays(p, [[0, 1, 2, 3]]), p


@pytest.mark.parametrize("side", [1e-6, 1e-4, 1e-2, 1e3])
def test_limit_grid_is_scale_invariant(monkeypatch, side):
    """R1 kept (mirrors M15): the dense grid of a scaled and moved cage is
    the scaled and moved grid, from mm cages in metres up to 1e3, for a
    single open quad and for a closed creased cage."""
    for make in (lambda: _xz_quad(1.0)[0], loop_box):
        unit = make()
        _, q1 = _fitter_input(monkeypatch, unit, "G0")
        g1 = np.array([q["dense_points"] for q in q1])
        shift = np.array([3.0, -7.0, 11.0]) * side
        moved = unit.copy()
        for v in moved.vertices:
            v.position = np.asarray(v.position, dtype=np.float64) * side + shift
        _, qs = _fitter_input(monkeypatch, moved, "G0")
        gs = np.array([q["dense_points"] for q in qs])
        np.testing.assert_allclose((gs - shift) / side, g1, atol=1e-12)


@pytest.mark.parametrize("side", [1e-6, 1e-4, 1.0, 1e3])
def test_flat_quad_gets_no_bump(monkeypatch, side):
    """R4 / R1: the limit surface of a planar cage is planar, so the samples
    of a single x-z quad keep y == 0 exactly at every scale (1f93473 bumped
    its centre by 0.025 * side along -y), and a zero-area (collinear) quad
    stays on its line."""
    m, p = _xz_quad(side)
    _, quads = _fitter_input(monkeypatch, m, "G0")
    g = quads[0]["dense_points"]
    assert g.shape == (6, 6, 3)
    # precondition: the samples are spread over the quad, not collapsed
    assert np.ptp(g[..., 0]) > 0.5 * side and np.ptp(g[..., 2]) > 0.5 * side
    assert np.array_equal(g[..., 1], np.zeros((6, 6))), np.abs(g[..., 1]).max() / side

    line = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=np.float64) * side
    m = HalfEdgeMesh.from_arrays(line, [[0, 1, 2, 3]])
    patches, quads = _fitter_input(monkeypatch, m, "G0")
    g = quads[0]["dense_points"]
    assert np.all(np.isfinite(g)) and np.all(np.isfinite(patches[0]))
    assert np.array_equal(g[..., 1:], np.zeros((6, 6, 2)))
