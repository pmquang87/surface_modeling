"""Regression tests for the shell / thicken SDF sign of ENCLOSED regions.

Source: the automated bug hunt report on commit
4e2241347de43175401df20c6722f38fa70820b4, entries

M6 (cand 22) the void of a hollow solid was signed "inside": shell_solid grew
             spurious surfaces inside the void.
M7 (cand 33) a cavity that opens through a vent narrower than the flood fill's
             wall cells was signed "inside": SDF -0.8 at the centre of the air
             cavity and spurious surfaces in the air.

Root cause (both): ``_compute_sdf`` signed every flood-fill component that the
padded grid corner cannot reach as inside. Enclosed is not inside. Each
enclosed component is now signed by a vote of its cells' exact
nearest-triangle sides.

Ground truth is analytic (radial distance for the spheres, set membership for
the boxes) or trimesh's own containment-signed distance, never the module's
own output.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
import trimesh

from src.core.halfedge_mesh import HalfEdgeMesh
from src.operations.shell_thicken import _compute_sdf, shell_solid

# The flood fill in _compute_sdf treats a cell as a wall (surface-crossing)
# cell when it lies within (0.87 + 0.75) voxels of the surface: half a voxel
# diagonal plus the sampling error of the coarse distance.
WALL_CELLS_IN_VOXELS = 0.87 + 0.75


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------

R_OUT, R_VOID, R_ISLAND = 1.0, 0.6, 0.3


def _hollow_sphere(island=False, flipped_every=0):
    """Ball of radius R_OUT with a spherical void of radius R_VOID.

    The void's normals point INTO the void: it is air, outside the material.
    `island` adds a solid ball of radius R_ISLAND floating in the void.
    `flipped_every` > 0 flips every n-th triangle (inconsistent normals).
    """
    outer = trimesh.creation.icosphere(subdivisions=3, radius=R_OUT)
    void = trimesh.creation.icosphere(subdivisions=3, radius=R_VOID)
    void.invert()
    parts = [outer, void]
    if island:
        parts.append(trimesh.creation.icosphere(subdivisions=3, radius=R_ISLAND))
    tm = trimesh.util.concatenate(parts)
    if flipped_every:
        faces = np.array(tm.faces)
        faces[::flipped_every] = faces[::flipped_every, ::-1]
        tm = trimesh.Trimesh(vertices=tm.vertices, faces=faces, process=False)
    return tm


def _sphere_sd(points, island=False):
    """Analytic signed distance (positive OUTSIDE the material)."""
    r = np.linalg.norm(points, axis=1)
    air_void = R_VOID - r
    if island:
        air_void = np.minimum(air_void, r - R_ISLAND)
    return np.maximum(r - R_OUT, air_void)


def _vented_box(cavity_half=0.8, vent_half=0.05):
    """Box [-1, 1]^3 with an air cavity [-c, c]^3 that opens to the top face
    through a square vent [-h, h]^2 x [c, 1]. One closed, consistently wound
    surface; the cavity and vent faces point into the air."""
    a, c, h = 1.0, float(cavity_half), float(vent_half)
    verts = []

    def corners(s):
        idx = {}
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    idx[(sx, sy, sz)] = len(verts)
                    verts.append((sx * s, sy * s, sz * s))
        return idx

    def hole(z):
        ring = []
        for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            ring.append(len(verts))
            verts.append((sx * h, sy * h, z))
        return ring

    outer, cavity = corners(a), corners(c)
    hole_top, hole_cav = hole(a), hole(c)
    faces = []

    def quad(p, q, r, s):
        faces.extend([[p, q, r], [p, r, s]])

    def box_with_holed_top(B, ring_in):
        """Outward-wound box faces; the +z face is a ring around `ring_in`."""
        quad(B[(-1, -1, -1)], B[(-1, 1, -1)], B[(1, 1, -1)], B[(1, -1, -1)])
        quad(B[(-1, -1, -1)], B[(1, -1, -1)], B[(1, -1, 1)], B[(-1, -1, 1)])
        quad(B[(1, -1, -1)], B[(1, 1, -1)], B[(1, 1, 1)], B[(1, -1, 1)])
        quad(B[(1, 1, -1)], B[(-1, 1, -1)], B[(-1, 1, 1)], B[(1, 1, 1)])
        quad(B[(-1, 1, -1)], B[(-1, -1, -1)], B[(-1, -1, 1)], B[(-1, 1, 1)])
        ring_out = [B[(-1, -1, 1)], B[(1, -1, 1)], B[(1, 1, 1)], B[(-1, 1, 1)]]
        for k in range(4):
            quad(ring_out[k], ring_out[(k + 1) % 4],
                 ring_in[(k + 1) % 4], ring_in[k])

    box_with_holed_top(outer, hole_top)
    n_outer = len(faces)
    box_with_holed_top(cavity, hole_cav)
    faces[n_outer:] = [f[::-1] for f in faces[n_outer:]]  # cavity is air
    for k in range(4):
        quad(hole_cav[k], hole_top[k], hole_top[(k + 1) % 4],
             hole_cav[(k + 1) % 4])
    return trimesh.Trimesh(vertices=np.array(verts, dtype=float),
                           faces=np.array(faces), process=False)


def _in_vented_box_material(points, cavity_half=0.8, vent_half=0.05):
    p = np.asarray(points)
    in_box = np.all(np.abs(p) < 1.0, axis=1)
    in_cavity = np.all(np.abs(p) < cavity_half, axis=1)
    in_vent = ((np.abs(p[:, 0]) < vent_half) & (np.abs(p[:, 1]) < vent_half)
               & (p[:, 2] > 0.0))
    return in_box & ~in_cavity & ~in_vent


def _grid_points(grid, origin, voxel):
    axes = [origin[k] + np.arange(grid.shape[k]) * voxel for k in range(3)]
    return np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)


def _index_of(point, origin, voxel):
    idx = np.rint((np.asarray(point) - origin) / voxel).astype(int)
    assert np.allclose(origin + idx * voxel, point, atol=1e-9), (
        "test setup: the probe point must be a grid point")
    return tuple(idx)


def _sdf(tm, resolution, padding=0.1):
    return _compute_sdf(np.asarray(tm.vertices, dtype=float),
                        np.asarray(tm.faces), resolution, padding=padding)


def _shell(tm, thickness, resolution):
    mesh = HalfEdgeMesh.from_arrays(np.asarray(tm.vertices, dtype=float),
                                    np.asarray(tm.faces).tolist())
    out = shell_solid(mesh, thickness=thickness, direction='inward',
                      resolution=resolution, smooth_iterations=0)
    return out.to_trimesh()


# --------------------------------------------------------------------------
# M6 (cand 22): the void of a hollow solid
# --------------------------------------------------------------------------

class TestHollowSolidVoid:
    RES = 32  # voxel 0.075

    @staticmethod
    def _assert_hollow_sphere_premise(tm, island=False):
        """The input really is a hollow ball whose void is AIR."""
        assert tm.is_watertight and tm.is_winding_consistent
        void_faces = np.abs(np.linalg.norm(tm.triangles_center, axis=1)
                            - R_VOID) < 0.02
        assert void_faces.sum() == 1280
        # void normals point at the centre, i.e. into the void
        assert np.all(np.einsum('ij,ij->i', tm.face_normals[void_faces],
                                tm.triangles_center[void_faces]) < 0)
        # independent ray-based containment: void is air, wall is material
        probes = np.array([[0.0, 0.0, 0.45], [0.0, 0.8, 0.0], [1.1, 0.0, 0.0]])
        assert list(tm.contains(probes)) == [False, True, False]
        if island:
            assert bool(tm.contains([[0.0, 0.0, 0.0]])[0])
        else:
            assert not bool(tm.contains([[0.0, 0.0, 0.0]])[0])

    def test_void_of_a_hollow_sphere_is_signed_outside(self):
        """M6 (cand 22): the void (r < 0.6) of a hollow ball is air, so the SDF
        there is POSITIVE: +0.6 at the centre (the inner icosphere's
        inradius, 0.597), and equal to the analytic distance everywhere. The
        buggy flood fill signed the whole void as inside (-0.597)."""
        tm = _hollow_sphere()
        self._assert_hollow_sphere_premise(tm)
        grid, origin, voxel = _sdf(tm, self.RES)
        assert voxel == pytest.approx(0.075)

        centre = grid[_index_of([0.0, 0.0, 0.0], origin, voxel)]
        inradius = float(trimesh.proximity.closest_point(
            tm, [[0.0, 0.0, 0.0]])[1][0])
        assert inradius == pytest.approx(0.597, abs=0.002)
        assert centre == pytest.approx(inradius, abs=0.01)

        pts = _grid_points(grid, origin, voxel)
        truth = _sphere_sd(pts)
        err = np.abs(grid.ravel() - truth)
        void_cells = (truth > 0.1) & (np.linalg.norm(pts, axis=1) < R_VOID)
        assert void_cells.sum() > 1000, "the void must contain many cells"
        assert np.all(grid.ravel()[void_cells] > 0), (
            f"{int((grid.ravel()[void_cells] <= 0).sum())} of "
            f"{int(void_cells.sum())} void cells are signed inside")
        assert err.max() < 0.03, f"max |sdf - analytic| = {err.max():.3f}"

    def test_island_in_the_void_is_inside_again(self):
        """M6 (cand 22): every enclosed component is signed on its own. A solid
        ball (r < 0.3) floating in the void is material again (-0.3 at the
        centre) while the void around it (0.3 < r < 0.6) is air -- neither
        "all enclosed components are inside" (the bug) nor "every enclosed
        component is air" passes this."""
        tm = _hollow_sphere(island=True)
        self._assert_hollow_sphere_premise(tm, island=True)
        grid, origin, voxel = _sdf(tm, self.RES)

        assert grid[_index_of([0.0, 0.0, 0.0], origin, voxel)] == pytest.approx(
            -R_ISLAND, abs=0.01)
        assert grid[_index_of([0.0, 0.0, 0.45], origin, voxel)] == pytest.approx(
            +0.15, abs=0.01)
        assert grid[_index_of([0.0, 0.0, 0.825], origin, voxel)] == pytest.approx(
            -0.175, abs=0.01)
        truth = _sphere_sd(_grid_points(grid, origin, voxel), island=True)
        err = np.abs(grid.ravel() - truth)
        assert err.max() < 0.03, f"max |sdf - analytic| = {err.max():.3f}"

    def test_a_few_flipped_triangles_do_not_flip_a_component(self):
        """M6 (cand 22): the component sign is a MAJORITY vote. With every 7th
        triangle flipped (normals inconsistent), the void still comes out air
        and the material between the spheres still comes out inside -- a
        single contrary vote must not decide a component."""
        tm = _hollow_sphere(flipped_every=7)
        assert not tm.is_winding_consistent  # the injected defect is there
        grid, origin, voxel = _sdf(tm, self.RES)
        pts = _grid_points(grid, origin, voxel)
        truth = _sphere_sd(pts)
        # beyond the wall cells (1.62 voxels), so signed by the component vote
        far = np.abs(truth) > 2 * voxel
        void_far = far & (truth > 0) & (np.linalg.norm(pts, axis=1) < R_VOID)
        wall_far = far & (truth < 0)
        assert void_far.sum() > 500 and wall_far.sum() > 1000
        assert np.all(grid.ravel()[void_far] > 0), "void signed inside"
        assert np.all(grid.ravel()[wall_far] < 0), "material signed outside"

    def test_component_touching_the_grid_border_stays_outside(self):
        """Guard for the M6 (cand 22) fix: the vote must never override the
        border. An inside-out box puts every exterior cell on the -normal
        side, so the exterior component votes "inside"; it touches the padded
        grid border and must still be signed outside (+), far from the box
        exactly its analytic distance."""
        box = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
        box.invert()
        assert box.volume < 0  # the injected defect: normals point inward
        grid, origin, voxel = _sdf(box, 32, padding=0.5)  # grid [-2, 2]^3
        assert voxel == pytest.approx(0.125)
        pts = _grid_points(grid, origin, voxel)
        outside = np.max(np.abs(pts), axis=1) > 1.0 + 3 * voxel
        assert outside.sum() > 10_000
        g = grid.ravel()
        assert np.all(g[outside] > 0), (
            f"{int((g[outside] <= 0).sum())} exterior cells signed inside")
        # grid corner (-2, -2, -2): sqrt(3) from the box corner
        assert grid[_index_of([-2.0, -2.0, -2.0], origin, voxel)] == pytest.approx(
            np.sqrt(3), abs=0.05)

    def test_shell_of_a_hollow_sphere_has_only_the_two_walls(self):
        """M6 (cand 22), the report's case at resolution 48 instead of 64 (the
        cost is the exact refinement; the defect does not depend on it):
        shell_solid(thickness=0.1, direction='inward') of the hollow ball gives
        exactly the two walls -- 0.9 <= r <= 1.0 and 0.6 <= r <= 0.7, four
        closed surfaces, volume 4/3 pi (1 - 0.9^3 + 0.7^3 - 0.6^3) = 1.667.
        The bug added two spurious surfaces inside the void (the report
        counted 7,420 vertices at resolution 64)."""
        tm = _hollow_sphere()
        out = _shell(tm, thickness=0.1, resolution=48)
        r = np.linalg.norm(out.vertices, axis=1)
        assert len(r) > 5000, "no wall was extracted"
        in_void = int((r < R_VOID - 0.02).sum())
        assert in_void == 0, f"{in_void} vertices of spurious surface in the void"

        comps = out.split(only_watertight=False)
        assert len(comps) == 4, f"expected 4 wall surfaces, got {len(comps)}"
        radii = sorted(float(np.linalg.norm(c.vertices, axis=1).mean())
                       for c in comps)
        assert radii == pytest.approx([0.6, 0.7, 0.9, 1.0], abs=0.01)
        ideal = 4.0 / 3.0 * np.pi * (1.0 - 0.9 ** 3 + 0.7 ** 3 - 0.6 ** 3)
        assert out.volume == pytest.approx(ideal, rel=0.03)
        assert out.is_watertight


# --------------------------------------------------------------------------
# M7 (cand 33): a cavity behind a vent narrower than the wall cells
# --------------------------------------------------------------------------

class TestVentedCavity:
    RES = 32  # voxel 0.075

    @staticmethod
    def _assert_vented_box_premise(tm, cavity_half, vent_half):
        """One closed surface; the cavity is air and reaches the outside."""
        assert tm.is_watertight and tm.is_winding_consistent
        ideal = 8.0 - (2 * cavity_half) ** 3 - (2 * vent_half) ** 2 * (1 - cavity_half)
        assert tm.volume == pytest.approx(ideal, abs=1e-9)
        assert not bool(tm.contains([[0.0, 0.0, 0.0]])[0])
        assert bool(tm.contains([[0.0, 0.0, -(1 + cavity_half) / 2]])[0])
        # a ray from the cavity centre straight up leaves through the vent
        assert not tm.ray.intersects_any([[0.0, 0.0, 0.0]], [[0.0, 0.0, 1.0]])[0]

    @pytest.mark.parametrize("vent_half, sealed", [(0.05, True), (0.25, False)],
                             ids=["narrow-vent", "wide-vent-control"])
    def test_cavity_air_is_signed_outside(self, vent_half, sealed):
        """M7 (cand 33): the report's box [-1,1]^3 with an air cavity
        [-0.8,0.8]^3 and a 0.1-wide vent. The cavity is air, so the SDF at its
        centre is +0.8, and every grid cell's sign matches set membership. The
        narrow vent is closed off by wall cells, so the flood fill cannot reach
        the cavity; the bug then signed it inside (-0.8). The wide vent is the
        control: the flood fill reaches the cavity and it was right before."""
        tm = _vented_box(cavity_half=0.8, vent_half=vent_half)
        self._assert_vented_box_premise(tm, 0.8, vent_half)
        grid, origin, voxel = _sdf(tm, self.RES)
        assert voxel == pytest.approx(0.075)
        # narrow: the vent lies inside the wall cells -> sealed for the fill
        assert (vent_half < WALL_CELLS_IN_VOXELS * voxel) == sealed

        centre = grid[_index_of([0.0, 0.0, 0.0], origin, voxel)]
        assert centre == pytest.approx(0.8, abs=0.005)

        pts = _grid_points(grid, origin, voxel)
        material = _in_vented_box_material(pts, 0.8, vent_half)
        wrong = (grid.ravel() < 0) != material
        assert material.sum() > 2000 and (~material).sum() > 10_000
        assert int(wrong.sum()) == 0, (
            f"{int(wrong.sum())} cells signed against membership, "
            f"{int((wrong & ~material).sum())} of them in the air")

    def test_shell_puts_no_wall_in_the_vented_cavity(self):
        """M7 (cand 33) through shell_solid(thickness=0.1, direction='inward',
        resolution=56). The cavity is [-0.55,0.55]^3: a 0.45 wall keeps the
        inner offset off the slab's mid-surface (the report's 0.2 slab is
        exactly two thicknesses), and 0.55 is no grid coordinate, so no grid
        point sits on a cavity face. The vent is the report's 0.1, still
        inside the wall cells. Expected: every wall vertex lies on the input
        surface (true sd = 0) or on its inner offset (sd = -0.1), two closed
        surfaces, and the wall volume is the material 6.6645 minus the core
        deeper than 0.1, 3.6468 (Steiner formula for the rounded cavity
        offset) = 3.0177. The bug put spurious surfaces in the cavity air."""
        c, h, t = 0.55, 0.05, 0.1
        tm = _vented_box(cavity_half=c, vent_half=h)
        self._assert_vented_box_premise(tm, c, h)
        out = _shell(tm, thickness=t, resolution=56)
        assert len(out.vertices) > 10_000, "no wall was extracted"

        voxel = 2.4 / 56
        assert h < WALL_CELLS_IN_VOXELS * voxel  # the fill cannot pass the vent
        sd = -trimesh.proximity.signed_distance(tm, out.vertices)  # + in air
        in_air = int((sd > 0.5 * voxel).sum())
        assert in_air == 0, f"{in_air} wall vertices lie in the air"
        on_surface = np.abs(sd) < 0.3 * voxel
        on_offset = np.abs(sd + t) < 0.3 * voxel
        assert on_surface.sum() > 5000 and on_offset.sum() > 5000
        stray = int((~on_surface & ~on_offset).sum())
        assert stray == 0, f"{stray} vertices on neither wall surface"

        comps = out.split(only_watertight=False)
        assert len(comps) == 2, f"expected 2 wall surfaces, got {len(comps)}"
        a = 2 * c  # cavity edge length
        steiner = a ** 3 + 6 * a ** 2 * t + 3 * np.pi * a * t ** 2             + 4.0 / 3.0 * np.pi * t ** 3
        vent_core = ((2 * h) ** 2 + 4 * (2 * h) * t + np.pi * t ** 2)             * ((1 - t) - (c + t))
        material = 8.0 - a ** 3 - (2 * h) ** 2 * (1 - c)
        ideal = material - ((2 - 2 * t) ** 3 - steiner - vent_core)
        assert ideal == pytest.approx(3.0177, abs=1e-4)
        assert out.volume == pytest.approx(ideal, rel=0.02)
        assert out.is_watertight
