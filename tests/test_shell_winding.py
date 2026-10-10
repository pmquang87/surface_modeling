"""Enclosed SDF components are signed by the generalized winding number.

``_compute_sdf`` (src/operations/shell_thicken.py) signs every flood-fill
component that does not touch the grid border as a whole. Commit 267d66b did
that by a cosine-weighted vote of the component's near-surface cells' exact
nearest-triangle sides. The vote is LOCAL: material between internal,
same-orientation bodies (plates inside a box, not booleaned) only sees the
plates' outward faces and votes "air": box + 5 plates at resolution 48, the
vote signs 5,492 far-field material cells (all four gaps) as air.

The fix signs each enclosed component by the generalized winding number
w (Jacobson, Kavan & Sorkine-Hornung 2013), inside iff w > 0.5, so
overlapping bodies (w = 2) are inside and voids (w = 0) are air. w is sampled
at deep cells SPREAD over the component and trusted only near an integer:
  * all samples near an integer, one side of 0.5 -> that side;
  * near-integer samples on both sides (a hole wider than the wall seal joins
    a void and the material into one component) -> each cell by w, logged;
  * w fractional (holes, flipped triangles) -> the old vote, logged.
The wall cells next to the surface keep the local exact side, so shell_solid
of unbooleaned overlapping bodies is NOT improved by any of this.

Ground truth here is analytic (box membership / box and sphere distances,
solid angles of a cube seen from its centre) or trimesh's ray containment,
never the module's own output.
"""
import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
import trimesh

from src.operations import shell_thicken
from src.operations.shell_thicken import (_GWN_INTEGER_TOL, _compute_sdf,
                                          _sample_cells, _winding_mode,
                                          _winding_numbers)

LOGGER = shell_thicken.__name__


def _cube(half=1.0):
    return trimesh.creation.box(extents=(2 * half,) * 3)


def _grid_points(grid, origin, voxel):
    axes = [origin[k] + np.arange(grid.shape[k]) * voxel for k in range(3)]
    return np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)


def _index_of(point, origin, voxel):
    idx = np.rint((np.asarray(point) - origin) / voxel).astype(int)
    assert np.allclose(origin + idx * voxel, point, atol=1e-9), (
        "test setup: the probe point must be a grid point")
    return tuple(idx)


# --------------------------------------------------------------------------
# the winding-number helper
# --------------------------------------------------------------------------

class TestWindingNumberHelper:
    def test_closed_cube_is_one_inside_and_zero_outside(self):
        tm = _cube()
        assert tm.is_watertight and tm.volume > 0
        inside = np.array([[0.0, 0.0, 0.0], [0.9, -0.9, 0.5], [0.0, 0.0, 0.999]])
        outside = np.array([[3.0, 0.0, 0.0], [1.001, 0.0, 0.0], [-5.0, 7.0, 2.0]])
        w_in = _winding_numbers(tm.triangles, inside)
        w_out = _winding_numbers(tm.triangles, outside)
        assert w_in == pytest.approx(1.0, abs=1e-9)
        assert w_out == pytest.approx(0.0, abs=1e-9)
        # the inside check rejects outside points and vice versa
        assert np.all(np.abs(w_out - 1.0) > 0.99)
        assert np.all(np.abs(w_in) > 0.99)

    def test_reversed_cube_is_minus_one_inside(self):
        tm = _cube()
        tm.invert()
        assert tm.volume < 0  # the defect: normals point inward
        w = _winding_numbers(tm.triangles, [[0.0, 0.0, 0.0], [0.5, 0.2, -0.7]])
        assert w == pytest.approx(-1.0, abs=1e-9)
        # the same check rejects the correctly wound cube
        w_ok = _winding_numbers(_cube().triangles, [[0.0, 0.0, 0.0]])
        assert abs(w_ok[0] + 1.0) > 1.99

    def test_cube_with_a_missing_face_is_five_sixths_at_the_centre(self):
        """Each face of a cube subtends 4 pi / 6 from its centre: drop one
        and w(centre) = 5/6. Off centre, w is no longer 5/6 (a hole makes w
        vary continuously), but stays strictly between 0 and 1."""
        tm = _cube()
        keep = tm.face_normals[:, 2] < 0.5          # drop the +z face
        assert (~keep).sum() == 2
        holed = trimesh.Trimesh(tm.vertices, tm.faces[keep], process=False)
        assert not holed.is_watertight
        w = _winding_numbers(holed.triangles,
                             [[0.0, 0.0, 0.0], [0.0, 0.0, 0.8], [0.0, 0.0, -0.8]])
        assert w[0] == pytest.approx(5.0 / 6.0, abs=1e-9)
        assert 0.0 < w[1] < 5.0 / 6.0 < w[2] < 1.0     # near the hole: smaller
        # the 5/6 check rejects the closed cube's centre value (1) ...
        w_closed = _winding_numbers(tm.triangles, [[0.0, 0.0, 0.0]])[0]
        assert abs(w_closed - 5.0 / 6.0) > 0.1
        # ... and the off-centre values
        assert abs(w[1] - 5.0 / 6.0) > 0.05 and abs(w[2] - 5.0 / 6.0) > 0.05

    def test_matches_ray_containment_on_a_non_convex_solid(self):
        """Independent checker: trimesh's ray containment. A closed annulus
        (non-convex) at 400 random points; w rounds to 0/1 exactly as the
        rays say, and is integral to round-off. Chunking (tiny budget) gives
        the same numbers."""
        tm = trimesh.creation.annulus(r_min=0.4, r_max=1.0, height=0.5)
        assert tm.is_watertight and tm.volume > 0
        rng = np.random.default_rng(1)
        pts = rng.uniform(-1.2, 1.2, size=(400, 3)) * [1.0, 1.0, 0.5]
        # keep points off the surface (round-off and ray ties)
        d = np.abs(trimesh.proximity.signed_distance(tm, pts))
        pts = pts[d > 1e-3]
        assert len(pts) > 350
        contains = tm.contains(pts)
        assert 50 < contains.sum() < len(pts) - 50  # both classes present
        w = _winding_numbers(tm.triangles, pts)
        assert np.abs(w - np.rint(w)).max() < 1e-9
        assert np.array_equal(np.rint(w).astype(int), contains.astype(int))
        w_chunked = _winding_numbers(tm.triangles, pts, budget=7)
        assert np.allclose(w_chunked, w, atol=1e-12)
        # the comparison rejects a wrong answer: an inverted copy
        inv = tm.copy()
        inv.invert()
        w_inv = _winding_numbers(inv.triangles, pts)
        assert not np.array_equal(np.rint(w_inv).astype(int), contains.astype(int))

    def test_overlapping_bodies_add_up(self):
        """Two same-orientation cubes overlapping in [-0.3, 0.3] x R^2: w = 2
        in the overlap -- the union rule (w > 0.5) puts it inside."""
        a = trimesh.creation.box(extents=(1.3, 2, 2),
                                 transform=trimesh.transformations.translation_matrix((-0.35, 0, 0)))
        b = trimesh.creation.box(extents=(1.3, 2, 2),
                                 transform=trimesh.transformations.translation_matrix((0.35, 0, 0)))
        soup = trimesh.util.concatenate([a, b])
        w = _winding_numbers(soup.triangles,
                             [[0.0, 0, 0], [-0.8, 0, 0], [0.8, 0, 0], [2.0, 0, 0]])
        assert w == pytest.approx([2.0, 1.0, 1.0, 0.0], abs=1e-9)

    @pytest.mark.parametrize("tri_shape, pts_shape", [
        ((4, 3), (2, 3)),        # flat face list, not corner coordinates
        ((4, 3, 2), (2, 3)),     # 2D corners
        ((4, 3, 3), (3,)),       # a single point without the batch axis
        ((4, 3, 3), (2, 2)),     # 2D points
    ])
    def test_rejects_malformed_input(self, tri_shape, pts_shape):
        with pytest.raises(ValueError):
            _winding_numbers(np.zeros(tri_shape), np.zeros(pts_shape))


# --------------------------------------------------------------------------
# nested internal plates: the vote's known failure
# --------------------------------------------------------------------------

# Box [-1, 1]^3 plus 5 internal plates 1.71 x 1.71 x 0.06, same orientation,
# not booleaned. 1.71 (not the reviewer's 1.7) and the z offsets keep every
# plate face plane off the grid coordinates (origin -1.2, voxel 0.05 at
# resolution 48): see PROJECT_MISTAKES.md "SDF / shell tests".
PLATE_Z = (-0.61, -0.31, -0.01, 0.29, 0.59)
PLATE_HALF = (0.855, 0.855, 0.03)


def _box_sdf(points, centre, half):
    q = np.abs(points - np.asarray(centre)) - np.asarray(half)
    return (np.linalg.norm(np.maximum(q, 0.0), axis=1)
            + np.minimum(q.max(axis=1), 0.0))


def _nested_plates():
    parts = [_cube()]
    for z in PLATE_Z:
        parts.append(trimesh.creation.box(
            extents=2 * np.array(PLATE_HALF),
            transform=trimesh.transformations.translation_matrix((0, 0, z))))
    return trimesh.util.concatenate(parts)


class TestNestedPlates:
    RES = 48

    def test_far_field_material_between_plates_is_inside(self):
        """Every far-field cell (> 2 voxels from every surface) inside the box
        is material -- the plates lie fully inside it -- and must be signed
        inside. Precondition: the near-surface vote signs the components
        between the plates as air (they only see the plates' outward faces).
        Fails on 1f93473 (no diagnostics; the vote signed 5,492 of these
        cells as air)."""
        tm = _nested_plates()
        assert len(tm.faces) == 6 * 12
        # premise: union = the box (independent: ray containment)
        between = np.array([[0.0, 0.0, z + 0.15] for z in PLATE_Z[:-1]])
        assert tm.contains(between).all()

        diag = {}
        grid, origin, voxel = _compute_sdf(np.asarray(tm.vertices, float),
                                           np.asarray(tm.faces), self.RES,
                                           diagnostics=diag)
        assert voxel == pytest.approx(0.05)
        pts = _grid_points(grid, origin, voxel)
        dist = np.abs(_box_sdf(pts, (0, 0, 0), (1, 1, 1)))
        for z in PLATE_Z:
            dist = np.minimum(dist, np.abs(_box_sdf(pts, (0, 0, z), PLATE_HALF)))
        material = np.all(np.abs(pts) < 1.0, axis=1)
        far = dist > 2 * voxel
        far_material = far & material

        # the components holding the far-field cells between the plates
        labels = diag['labels'].ravel()
        gap = (far_material & (pts[:, 2] > PLATE_Z[0]) & (pts[:, 2] < PLATE_Z[-1])
               & np.all(np.abs(pts[:, :2]) < PLATE_HALF[0], axis=1))
        gap_labels = np.unique(labels[gap])
        assert len(gap_labels) == 4, gap_labels      # one per gap
        assert np.isin(gap_labels, diag['enclosed']).all()   # all enclosed
        at = np.searchsorted(diag['enclosed'], gap_labels)
        # precondition: the vote alone gets every gap wrong ...
        assert np.all(diag['vote_sign'][at] == +1.0)
        n_vote_wrong = int(np.isin(labels[far_material], gap_labels).sum())
        assert 4000 < n_vote_wrong < 7000
        # ... while the winding number says inside (w = 1 exactly)
        assert diag['winding'][at] == pytest.approx(1.0, abs=1e-9)
        assert not diag['fallback'].any()
        assert np.all(diag['mode'] == 'winding')

        g = grid.ravel()
        assert far_material.sum() > 15_000 and (far & ~material).sum() > 20_000
        wrong_in = int((g[far_material] >= 0).sum())
        wrong_out = int((g[far & ~material] <= 0).sum())
        assert wrong_in == 0, f"{wrong_in} far-field material cells signed air"
        assert wrong_out == 0, f"{wrong_out} far-field air cells signed inside"
        assert np.all(diag['sign'][at] == -1.0)       # the winding decided

    @staticmethod
    def _holed_plates(n=16, r_hole=0.07):
        """The same plates, each top face an n x n quad grid with the cells
        whose centre lies within `r_hole` of 6 hole centres removed (a soup:
        T-junctions at the rim, as the hole is all that matters). Each hole
        is 1-2 cells, at most 0.21 x 0.11 -- sealed by the wall cells
        (2 x 1.62 voxels = 0.16 per side of the gap)."""
        parts = [_cube()]
        removed = 0
        holes = [(x, y) for x in (-0.5, 0.0, 0.5) for y in (-0.3, 0.3)]
        u = np.linspace(-PLATE_HALF[0], PLATE_HALF[0], n + 1)
        mid = (u[:-1] + u[1:]) / 2
        I, J = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
        cut = np.zeros((n, n), dtype=bool)
        for hx, hy in holes:
            cut |= np.hypot(mid[I] - hx, mid[J] - hy) < r_hole
        a = I * (n + 1) + J
        quads = np.stack([a, a + n + 1, a + n + 2, a + 1], axis=-1)[~cut]
        top = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
        X, Y = np.meshgrid(u, u, indexing='ij')
        for z in PLATE_Z:
            box = trimesh.creation.box(
                extents=2 * np.array(PLATE_HALF),
                transform=trimesh.transformations.translation_matrix((0, 0, z)))
            sides = box.face_normals[:, 2] < 0.5
            parts.append(trimesh.Trimesh(box.vertices, box.faces[sides], process=False))
            V = np.column_stack([X.ravel(), Y.ravel(),
                                 np.full(X.size, z + PLATE_HALF[2])])
            parts.append(trimesh.Trimesh(V, top, process=False))
            removed += 2 * int(cut.sum())
        return trimesh.util.concatenate(parts), removed

    def test_small_sealed_holes_keep_the_winding_decision(self):
        """Six small holes in each plate's top face: the holes are sealed by
        the wall cells (the components stay the gaps), but samples next to a
        hole read w = 0.95-1.13. Requiring EVERY sample within 0.05 of an
        integer (f502d0a) sent the gaps to the vote, which signs them air:
        5,492 far-field material cells. At least 75 % near an integer, all on
        one side of 0.5, keeps them 'winding'."""
        tm, removed = self._holed_plates()
        assert removed == 5 * 2 * 8
        up = tm.face_normals[:, 2] > 0.5
        assert up.sum() == 2 + 5 * (2 * 16 * 16 - 16)   # top faces wound up
        assert not tm.is_watertight

        diag = {}
        grid, origin, voxel = _compute_sdf(np.asarray(tm.vertices, float),
                                           np.asarray(tm.faces), self.RES,
                                           exact_band=0.05, diagnostics=diag)
        assert voxel == pytest.approx(0.05)
        pts = _grid_points(grid, origin, voxel)
        dist = np.abs(_box_sdf(pts, (0, 0, 0), (1, 1, 1)))
        for z in PLATE_Z:
            dist = np.minimum(dist, np.abs(_box_sdf(pts, (0, 0, z), PLATE_HALF)))
        material = np.all(np.abs(pts) < 1.0, axis=1)
        far = dist > 2 * voxel
        far_material = far & material

        labels = diag['labels'].ravel()
        gap = (far_material & (pts[:, 2] > PLATE_Z[0]) & (pts[:, 2] < PLATE_Z[-1])
               & np.all(np.abs(pts[:, :2]) < PLATE_HALF[0], axis=1))
        gap_labels = np.unique(labels[gap])
        assert len(gap_labels) == 4, gap_labels      # still sealed: one per gap
        at = np.searchsorted(diag['enclosed'], gap_labels)
        assert np.array_equal(diag['enclosed'][at], gap_labels)
        # precondition: the holes push some samples off the integer (so an
        # every-sample rule refuses), and the vote would sign the gaps air
        assert np.all(diag['near_fraction'][at] < 1.0)
        assert np.all(diag['w_max'][at] > 1.0 + _GWN_INTEGER_TOL)
        assert np.all(diag['vote_sign'][at] == +1.0)
        assert np.all(diag['w_min'][at] > 0.5)       # all samples inside

        g = grid.ravel()
        assert 15_000 < far_material.sum() < 18_000
        wrong_in = int((g[far_material] >= 0).sum())
        wrong_out = int((g[far & ~material] <= 0).sum())
        assert wrong_in == 0, f"{wrong_in} far-field material cells signed air"
        assert wrong_out == 0, f"{wrong_out} far-field air cells signed inside"
        assert np.all(diag['mode'][at] == 'winding')
        assert np.all(diag['near_fraction'][at] >= 0.75)



# --------------------------------------------------------------------------
# the sampler: deep AND spread
# --------------------------------------------------------------------------

class TestSampleCells:
    @staticmethod
    def _field():
        """Label 1: a 12^3 block whose depth peaks at (7.5, 7.5, 7.5) (so its
        5 deepest cells all lie in ONE of the 3^3 blocks); label 2: a 3-cell
        rod; label 3: not requested. Depth is distinct everywhere."""
        shape = (16, 16, 16)
        labels = np.zeros(shape, dtype=np.int32)
        labels[2:14, 2:14, 2:14] = 1
        labels[0, 0, 0:3] = 2
        labels[15, 15, 15] = 3
        ijk = np.stack(np.meshgrid(*[np.arange(n) for n in shape],
                                   indexing='ij'), axis=-1).reshape(-1, 3)
        depth = 100.0 - np.linalg.norm(ijk - 7.5, axis=1) + 1e-6 * np.arange(len(ijk))
        return shape, labels.ravel(), depth, ijk

    def test_takes_the_k_deepest_and_the_deepest_of_every_block(self):
        shape, labels, depth, ijk = self._field()
        cells, lab = _sample_cells(labels, depth, np.array([1, 2]), shape)
        assert len(np.unique(cells)) == len(cells)
        assert np.all(np.diff(lab) >= 0)                  # grouped, ascending
        assert set(lab.tolist()) == {1, 2}                # label 3 not asked
        assert np.array_equal(labels[cells], lab)

        one = np.flatnonzero(labels == 1)
        top5 = one[np.argsort(-depth[one])[:5]]
        picked = set(cells[lab == 1].tolist())
        assert set(top5.tolist()) <= picked, "the 5 deepest cells are missing"
        # independent block deepest: bbox 2..13 cut into 3 blocks of 4 cells
        blk = (ijk[one] - 2) // 4
        key = blk[:, 0] * 9 + blk[:, 1] * 3 + blk[:, 2]
        block_deepest = {int(one[key == b][np.argmax(depth[one][key == b])])
                         for b in range(27)}
        assert len(block_deepest) == 27
        assert block_deepest <= picked, "a block's deepest cell is missing"
        # the 5 deepest share one block, so exactly 27 + 4 samples
        assert len(picked) == 27 + 4
        # the rod has fewer cells than samples: all of them
        assert sorted(cells[lab == 2].tolist()) == np.flatnonzero(labels == 2).tolist()

    def test_the_sample_is_the_deep_end(self):
        """An input the depth check must reject: the shallowest cells. The
        sample's depths dominate every non-sampled cell of its block."""
        shape, labels, depth, ijk = self._field()
        cells, lab = _sample_cells(labels, depth, np.array([1]), shape)
        one = np.flatnonzero(labels == 1)
        shallow = one[np.argsort(depth[one])[:5]]
        assert not set(shallow.tolist()) & set(cells.tolist())
        assert np.median(depth[cells]) > np.percentile(depth[one], 75)


# --------------------------------------------------------------------------
# P1: a hole wider than the wall seal joins the void and the material
# --------------------------------------------------------------------------

R_VOID = 0.6
CAP_DEG = 20.0


def _holed_hollow_sphere(cap_deg=CAP_DEG):
    """Outward sphere r = 1 plus an inside-out sphere r = 0.6 (the void's
    faces point into it) with every triangle within cap_deg of +z removed:
    at 20 degrees a hole about 0.4 across, much wider than the wall seal (2 x 1.62 voxels
    = 0.16 at resolution 48), so the flood fill joins void and material."""
    outer = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
    void = trimesh.creation.icosphere(subdivisions=4, radius=R_VOID)
    void.invert()
    c = void.triangles_center
    polar = np.degrees(np.arccos(c[:, 2] / np.linalg.norm(c, axis=1)))
    void = trimesh.Trimesh(void.vertices, void.faces[polar > cap_deg],
                           process=False)
    return trimesh.util.concatenate([outer, void])


class TestHoleJoinsVoidAndMaterial:
    RES = 48   # voxel 0.05

    def test_far_field_cells_are_signed_per_cell_by_w(self, caplog):
        """Far-field material (0.6 < r < 1) inside, far-field void (r < 0.6,
        away from the hole's throat) outside. Fails on d3e9416: one component
        sampled at its 5 deepest cells -- all at the void centre, w = 0.03 --
        signed all 12,861 far-field material cells air. A median over spread
        samples (w = 1 in most of them) fails the other way: the void inside."""
        tm = _holed_hollow_sphere()
        assert 100 < 5120 - (len(tm.faces) - 5120) < 200    # cap removed
        assert not tm.is_watertight
        diag = {}
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            grid, origin, voxel = _compute_sdf(np.asarray(tm.vertices, float),
                                               np.asarray(tm.faces), self.RES,
                                               diagnostics=diag)
        assert voxel == pytest.approx(0.05)
        assert 2 * 1.62 * voxel < 0.3      # the hole is far wider than the seal
        pts = _grid_points(grid, origin, voxel)
        r = np.linalg.norm(pts, axis=1)
        polar = np.degrees(np.arccos(np.clip(pts[:, 2] / np.maximum(r, 1e-12), -1, 1)))
        sd = np.maximum(r - 1.0, R_VOID - r)                 # + in air
        far = np.abs(sd) > 2 * voxel
        far_material = far & (sd < 0)
        far_void = far & (r < R_VOID) & ((polar > CAP_DEG + 20) | (r < 0.3))

        # precondition: ONE enclosed component holds both regions ...
        labels = diag['labels'].ravel()
        assert len(diag['enclosed']) == 1
        L = diag['enclosed'][0]
        assert np.all(labels[far_material] == L) and np.all(labels[far_void] == L)
        # ... and its deepest cells see only the void (w ~ 0): sampling only
        # them would sign the material air
        comp = np.flatnonzero(labels == L)
        deep_cells = comp[np.argsort(-np.abs(grid.ravel()[comp]))[:5]]
        assert np.all(r[deep_cells] < 0.2)
        w_deep = _winding_numbers(tm.triangles, pts[deep_cells])
        assert np.all(w_deep < _GWN_INTEGER_TOL)

        g = grid.ravel()
        assert 10_000 < far_material.sum() < 16_000 and 3_000 < far_void.sum() < 5_000
        wrong_material = int((g[far_material] >= 0).sum())
        wrong_void = int((g[far_void] <= 0).sum())
        assert wrong_material == 0, f"{wrong_material} far material cells signed air"
        assert wrong_void == 0, f"{wrong_void} far void cells signed inside"
        # the throat itself (no surface there) is a transition; it holds
        # only a handful of cells signed against the radial truth
        wrong_all = int(((g < 0) != (sd < 0))[far].sum())
        assert wrong_all < 20, wrong_all

        # the spread samples straddle 0.5 at both integers: the per-cell path
        assert diag['w_min'][0] < _GWN_INTEGER_TOL
        assert abs(diag['w_max'][0] - 1.0) < _GWN_INTEGER_TOL
        assert diag['mode'][0] == 'split'
        logs = [r_ for r_ in caplog.records
                if r_.name == LOGGER and 'span both sides' in r_.getMessage()]
        assert len(logs) == 1

    def test_split_needs_only_half_the_samples_near_an_integer(self):
        """A 30-degree cap: more samples sit in the wider throat, only ~71 %
        are within 0.05 of an integer -- above _GWN_SPLIT_FRACTION (0.5),
        below 0.75. The per-cell path must still be taken: the vote (what a
        stricter split threshold such as 0.9 falls back to) signs the whole
        component inside, 3,586 far-field void cells."""
        cap = 30.0
        tm = _holed_hollow_sphere(cap)
        removed = 2 * 5120 - len(tm.faces)
        assert 300 < removed < 450
        diag = {}
        grid, origin, voxel = _compute_sdf(np.asarray(tm.vertices, float),
                                           np.asarray(tm.faces), self.RES,
                                           diagnostics=diag)
        pts = _grid_points(grid, origin, voxel)
        r = np.linalg.norm(pts, axis=1)
        polar = np.degrees(np.arccos(np.clip(pts[:, 2] / np.maximum(r, 1e-12), -1, 1)))
        sd = np.maximum(r - 1.0, R_VOID - r)
        far = np.abs(sd) > 2 * voxel
        far_material = far & (sd < 0)
        far_void = far & (r < R_VOID) & ((polar > cap + 20) | (r < 0.3))

        assert len(diag['enclosed']) == 1
        # precondition: the near-integer fraction lies between the split
        # threshold and the winding one, and the vote says "inside"
        assert 0.5 <= diag['near_fraction'][0] < 0.75
        assert diag['vote_sign'][0] == -1.0

        g = grid.ravel()
        assert 10_000 < far_material.sum() < 16_000 and 3_000 < far_void.sum() < 5_000
        wrong_material = int((g[far_material] >= 0).sum())
        wrong_void = int((g[far_void] <= 0).sum())
        assert wrong_material == 0, f"{wrong_material} far material cells signed air"
        assert wrong_void == 0, f"{wrong_void} far void cells signed inside"
        assert diag['mode'][0] == 'split'


# --------------------------------------------------------------------------
# P2 and the fallback: w fractional -> the vote
# --------------------------------------------------------------------------

class TestFractionalWindingFallsBackToVote:
    RES = 32   # voxel 0.075, origin -1.2: the faces at +-1 are off the grid

    @staticmethod
    def _cube_two_faces_reversed():
        """Cube [-1, 1]^3 with its +z and -z faces wound inward. Each face
        subtends 1/6 of the sphere from the centre, so w(centre) =
        4/6 - 2/6 = 1/3: fractional and BELOW 0.5, so the winding number
        alone would call the interior air."""
        tm = _cube()
        faces = np.array(tm.faces)
        z_faces = np.abs(tm.face_normals[:, 2]) > 0.5
        assert z_faces.sum() == 4
        faces[z_faces] = faces[z_faces, ::-1]
        return trimesh.Trimesh(tm.vertices, faces, process=False)

    @staticmethod
    def _run(tm, res, caplog):
        diag = {}
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            grid, origin, voxel = _compute_sdf(np.asarray(tm.vertices, float),
                                               np.asarray(tm.faces), res,
                                               diagnostics=diag)
        logs = [r for r in caplog.records
                if r.name == LOGGER and 'fractional winding' in r.getMessage()]
        return grid, origin, voxel, diag, logs

    def test_fractional_interior_is_signed_by_the_vote_and_logged(self, caplog):
        tm = self._cube_two_faces_reversed()
        assert not tm.is_winding_consistent       # the injected defect
        w_centre = _winding_numbers(tm.triangles, [[0.0, 0.0, 0.0]])[0]
        assert w_centre == pytest.approx(1.0 / 3.0, abs=1e-9)

        grid, origin, voxel, diag, logs = self._run(tm, self.RES, caplog)
        assert len(diag['enclosed']) == 1             # the interior
        assert diag['w_max'][0] < 0.5                 # w alone says air
        assert diag['w_min'][0] > _GWN_INTEGER_TOL    # ... but fractional
        assert diag['mode'][0] == 'vote' and bool(diag['fallback'][0])
        assert diag['vote_sign'][0] == -1.0           # 4 good faces outvote 2
        assert diag['sign'][0] == -1.0                # the vote decided
        assert len(logs) == 1

        assert grid[_index_of([0.0, 0.0, 0.0], origin, voxel)] == pytest.approx(
            -1.0, abs=0.01)
        pts = _grid_points(grid, origin, voxel)
        deep = np.max(np.abs(pts), axis=1) < 1.0 - 2 * voxel
        assert deep.sum() > 10_000
        assert np.all(grid.ravel()[deep] < 0)

    def test_sphere_with_40_percent_flipped_triangles(self, caplog):
        """P2: 40 % of a sphere's triangles flipped at random gives w ~ 0.2
        everywhere inside -- below 0.5 and outside the old ambiguity band
        |w - 0.5| < 0.25, so d3e9416 signed all far-field material air. w is
        far from an integer: the vote decides (it gets the interior right)."""
        s = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
        faces = np.array(s.faces)
        flip = np.random.default_rng(0).random(len(faces)) < 0.4
        assert 0.38 < flip.mean() < 0.42
        faces[flip] = faces[flip, ::-1]
        tm = trimesh.Trimesh(s.vertices, faces, process=False)
        assert not tm.is_winding_consistent

        grid, origin, voxel, diag, logs = self._run(tm, self.RES, caplog)
        assert len(diag['enclosed']) == 1
        assert 0.1 < diag['w_min'][0] and diag['w_max'][0] < 0.3   # precondition
        assert diag['mode'][0] == 'vote' and diag['sign'][0] == -1.0
        assert len(logs) == 1

        pts = _grid_points(grid, origin, voxel)
        sd = np.linalg.norm(pts, axis=1) - 1.0
        far = np.abs(sd) > 2 * voxel
        g = grid.ravel()
        assert (far & (sd < 0)).sum() > 5_000
        assert int((g[far & (sd < 0)] >= 0).sum()) == 0
        assert int((g[far & (sd > 0)] <= 0).sum()) == 0

    def test_consistent_cube_uses_no_fallback(self, caplog):
        """Control (an input the fallback check must reject): the same cube
        wound consistently has w = 1 in its interior, no fallback, no log."""
        tm = _cube()
        grid, origin, voxel, diag, logs = self._run(tm, self.RES, caplog)
        assert len(diag['enclosed']) == 1
        assert diag['w_min'][0] == pytest.approx(1.0, abs=1e-9)
        assert diag['w_max'][0] == pytest.approx(1.0, abs=1e-9)
        assert diag['mode'][0] == 'winding' and not diag['fallback'].any()
        assert not [r for r in caplog.records if r.name == LOGGER]
        assert grid[_index_of([0.0, 0.0, 0.0], origin, voxel)] == pytest.approx(
            -1.0, abs=0.01)


# --------------------------------------------------------------------------
# the decision rule, at its boundaries
# --------------------------------------------------------------------------

def _w(*runs):
    """Concatenate (count, value) runs into one sample array."""
    return np.concatenate([np.full(n, v, dtype=float) for n, v in runs])


class TestWindingMode:
    @pytest.mark.parametrize("w, expected", [
        (_w((32, 1.0)), ('winding', True)),
        (_w((32, 0.0)), ('winding', False)),
        (_w((32, 2.0)), ('winding', True)),        # overlap: union inside
        (_w((32, -1.0)), ('winding', False)),      # inside-out body: air
        (_w((31, 1.0), (1, 1.0 + 1e-9)), ('winding', True)),
    ])
    def test_clean_components(self, w, expected):
        assert _winding_mode(w) == expected

    def test_winding_needs_three_quarters_near_an_integer(self):
        """24 of 32 (75 %) near 1, the rest at 1.13 (next to a small sealed
        hole) -> 'winding'; 23 of 32 -> no longer trusted -> 'vote'."""
        assert _winding_mode(_w((24, 1.0), (8, 1.13))) == ('winding', True)
        assert _winding_mode(_w((23, 1.0), (9, 1.13))) == ('vote', None)
        assert _winding_mode(_w((24, 0.0), (8, 0.12))) == ('winding', False)

    def test_winding_needs_every_sample_on_one_side(self):
        """31 samples at exactly 1 and ONE at 0.45 -> not 'winding'; the
        near-integer samples are all on one side, so not 'split' either."""
        assert _winding_mode(_w((31, 1.0), (1, 0.45))) == ('vote', None)
        assert _winding_mode(_w((31, 0.0), (1, 0.55))) == ('vote', None)
        assert _winding_mode(_w((31, 1.0), (1, 0.55))) == ('winding', True)

    def test_split_needs_half_near_an_integer_on_both_sides(self):
        """16 of 32 near an integer (8 at 0, 8 at 1), 16 in a throat at 0.3
        -> 'split'; 15 of 32 -> 'vote'; near samples on one side only ->
        'vote'."""
        assert _winding_mode(_w((8, 0.0), (8, 1.0), (16, 0.3))) == ('split', None)
        assert _winding_mode(_w((7, 0.0), (8, 1.0), (17, 0.3))) == ('vote', None)
        assert _winding_mode(_w((16, 1.0), (16, 0.3))) == ('vote', None)
        # many near-integer samples on both sides -> split, not winding
        assert _winding_mode(_w((16, 0.0), (16, 1.0))) == ('split', None)

    def test_fractional_everywhere_goes_to_the_vote(self):
        """40-42 % flipped triangles: w ~ 0.16-0.2 everywhere. A wider
        tolerance (0.2) would call this 'winding', air, silently."""
        rng = np.random.default_rng(0)
        assert _winding_mode(0.16 + 0.03 * rng.random(32)) == ('vote', None)
        assert _winding_mode(0.7 + 0.03 * rng.random(32)) == ('vote', None)
        assert _GWN_INTEGER_TOL < 0.16
