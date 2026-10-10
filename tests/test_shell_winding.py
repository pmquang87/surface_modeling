"""Enclosed SDF components are signed by the generalized winding number.

``_compute_sdf`` (src/operations/shell_thicken.py) signs every flood-fill
component that does not touch the grid border as a whole. Commit 267d66b did
that by a cosine-weighted vote of the component's near-surface cells' exact
nearest-triangle sides. The vote is LOCAL: material between internal,
same-orientation bodies (plates inside a box, not booleaned) only sees the
plates' outward faces and votes "air". A reviewer measured 10,972 such
far-field material cells signed as air (box + 5 plates, resolution 48).

The fix signs each enclosed component by the generalized winding number
w (Jacobson, Kavan & Sorkine-Hornung 2013) at its deepest cells: inside iff
w > 0.5, so overlapping bodies (w = 2) are inside, voids (w = 0) are air. The
vote survives only as the fallback when w is ambiguous
(|w - 0.5| < _GWN_AMBIGUOUS), which is logged.

Ground truth here is analytic (box membership / box distances, solid angles
of a cube seen from its centre) or trimesh's ray containment, never the
module's own output.
"""
import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
import trimesh

from src.operations import shell_thicken
from src.operations.shell_thicken import (_GWN_AMBIGUOUS, _compute_sdf,
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

        g = grid.ravel()
        assert far_material.sum() > 15_000 and (far & ~material).sum() > 20_000
        wrong_in = int((g[far_material] >= 0).sum())
        wrong_out = int((g[far & ~material] <= 0).sum())
        assert wrong_in == 0, f"{wrong_in} far-field material cells signed air"
        assert wrong_out == 0, f"{wrong_out} far-field air cells signed inside"
        assert np.all(diag['sign'][at] == -1.0)       # the winding decided


# --------------------------------------------------------------------------
# the ambiguous band: fallback to the vote
# --------------------------------------------------------------------------

class TestAmbiguousWindingFallsBackToVote:
    RES = 32   # voxel 0.075, origin -1.2: the faces at +-1 are off the grid

    @staticmethod
    def _cube_two_faces_reversed():
        """Cube [-1, 1]^3 with its +z and -z faces wound inward. Each face
        subtends 1/6 of the sphere from the centre, so w(centre) =
        4/6 - 2/6 = 1/3: in the ambiguous band and BELOW 0.5, so the winding
        number alone would call the interior air."""
        tm = _cube()
        faces = np.array(tm.faces)
        z_faces = np.abs(tm.face_normals[:, 2]) > 0.5
        assert z_faces.sum() == 4
        faces[z_faces] = faces[z_faces, ::-1]
        return trimesh.Trimesh(tm.vertices, faces, process=False)

    def test_ambiguous_interior_is_signed_by_the_vote_and_logged(self, caplog):
        tm = self._cube_two_faces_reversed()
        assert not tm.is_winding_consistent       # the injected defect
        w_centre = _winding_numbers(tm.triangles, [[0.0, 0.0, 0.0]])[0]
        assert w_centre == pytest.approx(1.0 / 3.0, abs=1e-9)
        assert abs(w_centre - 0.5) < _GWN_AMBIGUOUS < 0.5   # inside the band

        diag = {}
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            grid, origin, voxel = _compute_sdf(np.asarray(tm.vertices, float),
                                               np.asarray(tm.faces), self.RES,
                                               diagnostics=diag)
        assert len(diag['enclosed']) == 1             # the interior
        w = diag['winding'][0]
        assert abs(w - 0.5) < _GWN_AMBIGUOUS and w < 0.5
        assert bool(diag['fallback'][0])
        assert diag['vote_sign'][0] == -1.0           # 4 good faces outvote 2
        assert diag['sign'][0] == -1.0                # the vote decided
        fallback_logs = [r for r in caplog.records
                         if r.name == LOGGER and 'ambiguous winding' in r.getMessage()]
        assert len(fallback_logs) == 1

        assert grid[_index_of([0.0, 0.0, 0.0], origin, voxel)] == pytest.approx(
            -1.0, abs=0.01)
        pts = _grid_points(grid, origin, voxel)
        deep = np.max(np.abs(pts), axis=1) < 1.0 - 2 * voxel
        assert deep.sum() > 10_000
        assert np.all(grid.ravel()[deep] < 0)

    def test_consistent_cube_uses_no_fallback(self, caplog):
        """Control (an input the fallback check must reject): the same cube
        wound consistently has w = 1 in its interior, no fallback, no log."""
        tm = _cube()
        diag = {}
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            grid, origin, voxel = _compute_sdf(np.asarray(tm.vertices, float),
                                               np.asarray(tm.faces), self.RES,
                                               diagnostics=diag)
        assert len(diag['enclosed']) == 1
        assert diag['winding'][0] == pytest.approx(1.0, abs=1e-9)
        assert not diag['fallback'].any()
        assert not [r for r in caplog.records if r.name == LOGGER]
        assert grid[_index_of([0.0, 0.0, 0.0], origin, voxel)] == pytest.approx(
            -1.0, abs=0.01)
