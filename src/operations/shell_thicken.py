import logging

import numpy as np
from typing import List, Optional, Tuple, Any
import trimesh

from src.core.halfedge_mesh import HalfEdgeMesh

logger = logging.getLogger(__name__)

# An enclosed flood-fill component is signed by the generalized winding number
# w (inside iff w > 0.5), sampled at cells SPREAD over the component. For a
# closed, consistently wound surface w is an integer off the surface (0 air,
# 1 inside, 2 where two same-orientation bodies overlap), exact to round-off,
# and constant on a flood-fill component (which never crosses the surface).
# It drifts from an integer only through defects: a hole of solid angle Omega
# seen from the point shifts w by Omega / 4 pi, a flipped triangle by twice
# that. So w is trusted only where it is near an integer:
#   * every sample within _GWN_INTEGER_TOL of an integer, all on one side of
#     0.5 -> the component is that side ("winding");
#   * at least half the samples near an integer, and those on BOTH sides ->
#     a hole joins two regions into one component (samples in the hole's
#     throat are fractional by construction): each cell is signed by w on its
#     own ("split");
#   * otherwise (w fractional: holes or flipped triangles all around) -> the
#     local vote of the near-surface cells' exact sides ("vote").
# _GWN_INTEGER_TOL = 0.05: round-off is ~1e-12, and a 5 % defect still leaves
# the nearest integer unambiguous (a sphere with 40 of 1280 triangles removed
# gives 0.968 at the centre); a sphere with 40 % of its triangles flipped
# gives ~0.2, a 20-degree hole in a void ~0.03 at the void centre but up to
# 0.3 in its throat.
_GWN_INTEGER_TOL = 0.05
_GWN_SAMPLES = 5          # deepest cells per component ...
_GWN_STRATA = 3           # ... plus the deepest cell of each of 3^3 blocks
# A split component evaluates w on a lattice subsample of its cells, at most
# _GWN_SPLIT_PAIRS point x triangle pairs (~2 s), 64..8192 points.
_GWN_SPLIT_PAIRS = 20_000_000


def _winding_numbers(triangles: np.ndarray, points: np.ndarray,
                     budget: int = 32_768) -> np.ndarray:
    """Generalized winding number of a triangle soup at `points`.

    w(p) = 1/(4 pi) * sum of the signed solid angles of the triangles seen
    from p (Jacobson, Kavan & Sorkine-Hornung 2013), each by Van Oosterom &
    Strackee: Omega = 2 atan2(a . (b x c),
    |a||b||c| + (a.b)|c| + (a.c)|b| + (b.c)|a|), with a, b, c the corners
    relative to p. A triangle counts positive when p lies behind it (on its
    -normal side), so w = +1 inside an outward-wound closed body, 0 outside,
    -1 inside an inside-out one.

    Args:
        triangles: (T, 3, 3) corner coordinates.
        points: (P, 3) query points.
        budget: point x triangle pairs evaluated at once. Small enough that
            the ~40 (budget,) float64 temporaries stay in cache: 32k pairs
            run ~3x faster per pair than 1M (measured, 64k triangles).
    Returns:
        (P,) winding numbers.
    """
    tri = np.asarray(triangles, dtype=np.float64)
    pts = np.asarray(points, dtype=np.float64)
    if tri.ndim != 3 or tri.shape[1:] != (3, 3):
        raise ValueError(f"triangles must be (T, 3, 3), got {tri.shape}")
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must be (P, 3), got {pts.shape}")
    w = np.zeros(len(pts))
    if len(tri) == 0 or len(pts) == 0:
        return w
    corners = np.ascontiguousarray(tri.transpose(1, 2, 0))  # (corner, xyz, T)
    p_chunk = max(1, min(len(pts), 64, budget))
    t_chunk = max(1, budget // p_chunk)
    for ps in range(0, len(pts), p_chunk):
        p = pts[ps:ps + p_chunk]
        px, py, pz = p[:, 0:1], p[:, 1:2], p[:, 2:3]
        acc = np.zeros(len(p))
        for ts in range(0, len(tri), t_chunk):
            k = corners[:, :, ts:ts + t_chunk]
            ax, ay, az = k[0, 0] - px, k[0, 1] - py, k[0, 2] - pz
            bx, by, bz = k[1, 0] - px, k[1, 1] - py, k[1, 2] - pz
            cx, cy, cz = k[2, 0] - px, k[2, 1] - py, k[2, 2] - pz
            la = np.sqrt(ax * ax + ay * ay + az * az)
            lb = np.sqrt(bx * bx + by * by + bz * bz)
            lc = np.sqrt(cx * cx + cy * cy + cz * cz)
            det = (ax * (by * cz - bz * cy) + ay * (bz * cx - bx * cz)
                   + az * (bx * cy - by * cx))
            den = (la * lb * lc + (ax * bx + ay * by + az * bz) * lc
                   + (ax * cx + ay * cy + az * cz) * lb
                   + (bx * cx + by * cy + bz * cz) * la)
            acc += np.arctan2(det, den).sum(axis=1)
        w[ps:ps + p_chunk] = acc / (2.0 * np.pi)   # 2 atan2 / (4 pi)
    return w


def _sample_cells(labels_flat: np.ndarray, depth: np.ndarray,
                  comp_ids: np.ndarray, shape: Tuple[int, int, int],
                  k_deep: int = _GWN_SAMPLES,
                  strata: int = _GWN_STRATA) -> Tuple[np.ndarray, np.ndarray]:
    """Spread, deep sample cells of each label in `comp_ids`.

    For every component: its `k_deep` cells of largest `depth`, plus the
    deepest cell of each occupied block when the component's bounding box is
    cut into `strata` blocks per axis. Deep keeps w far from the surface;
    spread sees every region of a component that a hole joined together (the
    deepest cells alone all sit in the widest region).

    Returns (cell indices into the flat grid, their label), grouped by
    ascending label, no duplicates.
    """
    member = np.zeros(int(labels_flat.max()) + 1, dtype=bool)
    member[comp_ids] = True
    cells = np.flatnonzero(member[labels_flat])
    if len(cells) == 0:
        return cells, cells.copy()
    lab = labels_flat[cells]
    order = np.argsort(lab, kind='stable')
    cells, lab = cells[order], lab[order]
    starts = np.flatnonzero(np.r_[True, lab[1:] != lab[:-1]])
    group = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(lab)]))
    neg_depth = -depth[cells]

    def first_per(key):
        o = np.lexsort((neg_depth, key))        # by key, deepest first
        ko = key[o]
        return o, np.flatnonzero(np.r_[True, ko[1:] != ko[:-1]])

    ijk = np.column_stack(np.unravel_index(cells, shape))
    lo = np.minimum.reduceat(ijk, starts, axis=0)
    hi = np.maximum.reduceat(ijk, starts, axis=0)
    size = -(-(hi - lo + 1) // strata)          # block edge, ceil
    blk = (ijk - lo[group]) // size[group]      # each in [0, strata)
    key = (group * strata + blk[:, 0]) * strata ** 2 + blk[:, 1] * strata + blk[:, 2]
    o, first = first_per(key)
    per_block = o[first]

    o, first = first_per(group)
    rank = np.arange(len(o)) - np.repeat(first, np.diff(np.r_[first, len(o)]))
    deepest = o[rank < k_deep]

    pick = np.unique(np.r_[per_block, deepest])  # positions; cells sorted by label
    return cells[pick], lab[pick]


def _sign_split_component(comp_cells: np.ndarray, shape: Tuple[int, int, int],
                          points: np.ndarray, triangles: np.ndarray,
                          seed_cells: np.ndarray, seed_w: np.ndarray
                          ) -> Tuple[np.ndarray, int]:
    """Per-cell sign (+1 outside) of one component that spans both sides.

    w is evaluated on a lattice subsample of the component (stride grown
    until at most clip(_GWN_SPLIT_PAIRS / T, 64, 8192) points) plus the
    already evaluated `seed_cells`; every other cell takes the side of the
    nearest evaluated cell ALONG the component (6-connected growth inside
    it), so a side never leaks through the wall cells into a region of the
    component across a surface. Returns (signs aligned with `comp_cells`,
    number of evaluated cells).
    """
    from scipy import ndimage

    n_cap = int(np.clip(_GWN_SPLIT_PAIRS // max(len(triangles), 1), 64, 8192))
    ijk = np.column_stack(np.unravel_index(comp_cells, shape))
    stride = 1
    while True:
        on_lattice = np.all(ijk % stride == 0, axis=1)
        if on_lattice.sum() <= n_cap:
            break
        stride += 1
    lattice = comp_cells[on_lattice]
    w_lat = _winding_numbers(triangles, points[lattice])
    marked = np.r_[lattice, seed_cells]
    inside = np.r_[w_lat, seed_w] > 0.5

    lo = ijk.min(axis=0)
    sub_shape = tuple(ijk.max(axis=0) - lo + 1)
    def local(flat):
        return tuple((np.column_stack(np.unravel_index(flat, shape)) - lo).T)

    mask = np.zeros(sub_shape, dtype=bool)
    mask[local(comp_cells)] = True
    state = np.zeros(sub_shape, dtype=np.int8)   # 0 unknown, 1 inside, 2 outside
    state[local(marked)] = np.where(inside, 1, 2)
    while True:
        free = mask & (state == 0)
        if not free.any():
            break
        grow_in = ndimage.binary_dilation(state == 1) & free
        grow_out = ndimage.binary_dilation(state == 2) & free & ~grow_in
        if not (grow_in.any() or grow_out.any()):
            break                                # unreachable (cannot happen)
        state[grow_in] = 1
        state[grow_out] = 2
    signs = np.where(state[local(comp_cells)] == 1, -1.0, 1.0)
    return signs, len(marked)


def _oriented_distance(tmesh: 'trimesh.Trimesh', points: np.ndarray) -> np.ndarray:
    """Signed distance to an OPEN, oriented surface.

    ``trimesh.proximity.signed_distance`` needs a watertight mesh (it signs by
    containment), so for a sheet the sign is meaningless. Sign by the normal of
    the nearest triangle instead: positive on the +normal ("outward") side.
    """
    closest, dist, tri_id = trimesh.proximity.closest_point(tmesh, points)
    normals = tmesh.face_normals[tri_id]
    side = np.einsum('ij,ij->i', points - closest, normals)
    sign = np.where(side < 0.0, -1.0, 1.0)
    return dist * sign


def _compute_sdf(vertices: np.ndarray, faces: np.ndarray,
                 grid_resolution: int, padding: float = 0.1,
                 open_surface: bool = False,
                 exact_band: Optional[float] = None,
                 diagnostics: Optional[dict] = None) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute a signed distance field on a voxel grid.

    Sign convention: POSITIVE OUTSIDE / on the +normal side, negative inside.

    Scalability: ``trimesh.proximity.signed_distance`` on the full grid
    broadcasts points x triangles (a real 63k-triangle part at resolution 192
    attempted a 244 GiB allocation). Instead:
      1. coarse unsigned distance for EVERY grid point from a KDTree over a
         dense surface sampling (~voxel/2 spacing),
      2. exact unsigned distance (chunked closest-point queries) only inside
         ``exact_band`` -- the region whose values marching cubes will read,
      3. sign from a voxel flood fill: cells the surface crosses are walls,
         every component touching the grid border is outside, and each
         enclosed component is signed by the generalized winding number at
         spread, deep sample cells (inside iff w > 0.5): as a whole when all
         samples agree near an integer, per cell when a hole joined two
         regions, and by a vote of its cells' exact nearest-triangle sides
         when w is fractional -- enclosed is not the same as inside (the void
         of a hollow solid, a cavity behind a vent narrower than the wall
         cells). Near-surface
         cells sign by the exact nearest triangle's normal. This stays
         correct at thin features where pure normal-based signing is
         ambiguous.

    Args:
        open_surface: sign by the nearest triangle's normal instead of by
            enclosure -- required for sheets, which have no inside.
        exact_band: absolute distance (same units as the mesh) within which
            values are refined to exact surface distance. None -> 4 voxels
            (enough to extract the level-0 surface).
        diagnostics: if a dict is given, it is filled with the per-component
            sign decision of the flood fill (closed surfaces only): arrays
            ``enclosed`` (label ids), ``size``, ``n_samples``, ``winding`` /
            ``w_min`` / ``w_max`` (median / range of w over the samples),
            ``mode`` ('winding', 'split' or 'vote'), ``vote`` / ``vote_sign``
            (the near-surface vote), ``sign`` (final, +1 outside, 0 = per
            cell), ``fallback`` (vote used), plus ``labels`` (the label grid)
            and ``n_components``.

    Returns:
        sdf_grid: 3D numpy array of signed distances
        grid_origin: (3,) origin of the grid
        voxel_size: float, size of each voxel
    """
    from scipy import ndimage
    from scipy.spatial import cKDTree

    if len(faces) == 0 or len(vertices) == 0:
        raise ValueError("Empty mesh provided.")

    tmesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    extents = bbox_max - bbox_min
    max_extent = extents.max() if extents.max() > 0 else 1.0

    pad_val = max_extent * padding
    bbox_min = bbox_min - pad_val
    bbox_max = bbox_max + pad_val

    # Voxel size from the PADDED extent of the largest axis. The old
    # (bbox_max.max() - bbox_min.min()) mixed coordinate axes: for a part
    # sitting at x ~= 1150 it returned ~1170 instead of ~24 and the grid
    # collapsed to 2x2x2.
    padded_extents = bbox_max - bbox_min
    voxel_size = float(padded_extents.max() / max(1, grid_resolution))
    if not np.isfinite(voxel_size) or voxel_size <= 0:
        voxel_size = 1.0

    nx = int(np.ceil(padded_extents[0] / voxel_size)) + 1
    ny = int(np.ceil(padded_extents[1] / voxel_size)) + 1
    nz = int(np.ceil(padded_extents[2] / voxel_size)) + 1
    nx = max(nx, 2); ny = max(ny, 2); nz = max(nz, 2)

    x = np.linspace(bbox_min[0], bbox_min[0] + (nx-1)*voxel_size, nx)
    y = np.linspace(bbox_min[1], bbox_min[1] + (ny-1)*voxel_size, ny)
    z = np.linspace(bbox_min[2], bbox_min[2] + (nz-1)*voxel_size, nz)

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    # 1. Coarse unsigned distance from a dense surface sampling.
    target_spacing = voxel_size / 2.0
    n_samples = int(np.clip(4.0 * tmesh.area / max(target_spacing ** 2, 1e-12),
                            20_000, 3_000_000))
    samples, sample_faces = trimesh.sample.sample_surface(tmesh, n_samples, seed=0)
    tree = cKDTree(samples)
    d_est, nearest_idx = tree.query(points, workers=-1)
    sampling_err = 1.5 * target_spacing

    # 2. Exact refinement inside the band marching cubes will actually read.
    # Keep the exact closest points / triangles: signing near-surface cells
    # with a nearest SAMPLE's normal fails at creases (a near-tie can pick a
    # sample on the adjacent face, flipping the sign and bulging the level-0
    # surface outward at box edges); the exact closest point has no such tie.
    band = (4.0 * voxel_size) if exact_band is None else float(exact_band)
    refine_mask = d_est < band + 2.0 * voxel_size + sampling_err
    refine_pts = points[refine_mask]
    exact_closest = np.empty((len(refine_pts), 3))
    exact_tri = np.empty(len(refine_pts), dtype=np.int64)
    if len(refine_pts):
        exact = np.empty(len(refine_pts))
        chunk = 200_000
        for s in range(0, len(refine_pts), chunk):
            closest, dist, tri = tmesh.nearest.on_surface(refine_pts[s:s + chunk])
            exact[s:s + chunk] = dist
            exact_closest[s:s + chunk] = closest
            exact_tri[s:s + chunk] = tri
        d_est[refine_mask] = exact

    def _exact_side():
        """(p - closest) . normal of the exact nearest triangle, per refined
        point: positive on its +normal side."""
        normals = tmesh.face_normals[exact_tri]
        return np.einsum('ij,ij->i', refine_pts - exact_closest, normals)

    def _exact_normal_sign():
        """Sign for the refined points from the exact nearest triangle."""
        return np.where(_exact_side() < 0.0, -1.0, 1.0)

    if open_surface:
        # Sheets have no enclosure: sign by the nearest facet's normal —
        # exact within the refined band, sample-based far away (where the
        # sign never reaches an extraction level anyway).
        normals = tmesh.face_normals[sample_faces[nearest_idx]]
        side = np.einsum('ij,ij->i', points - samples[nearest_idx], normals)
        sign = np.where(side < 0.0, -1.0, 1.0)
        if len(refine_pts):
            sign[refine_mask] = _exact_normal_sign()
        sdf_grid = (sign * d_est).reshape((nx, ny, nz))
        grid_origin = np.array([x[0], y[0], z[0]])
        return sdf_grid, grid_origin, voxel_size

    # 3. Sign by voxel flood fill.
    d_grid = d_est.reshape((nx, ny, nz))
    wall = d_grid < (voxel_size * 0.87 + sampling_err)  # cells the surface crosses
    labels, n_labels = ndimage.label(~wall)
    outside_label = labels[0, 0, 0]  # padded corner is always outside
    if outside_label == 0:
        raise RuntimeError(
            "SDF sign classification failed: the padded grid corner lies in a "
            "surface-crossing cell; increase resolution or padding.")

    # Enclosed is not inside: the void of a hollow solid, or a cavity behind a
    # vent narrower than the wall cells, is sealed off by wall cells yet is
    # air. The surface never passes between two face-adjacent non-wall cells
    # (a crossing would put one of them within half a voxel of it), so for a
    # closed surface each component lies on ONE side of it. Signing:
    #   * touching the grid border -> outside;
    #   * else by the generalized winding number at spread, deep sample cells
    #     (see _GWN_INTEGER_TOL for the three modes): a global test -- it sees
    #     every body, so overlapping same-orientation bodies (w = 2) stay
    #     inside, where a local test sees only the nearest one (material
    #     between nested internal plates is next to the plates' outward faces
    #     and looks like air). A hole wider than the wall seal joins two
    #     regions into one component: then each cell is signed by w;
    #   * where w is fractional (holes, flipped triangles), by a vote of the
    #     component's refined cells' exact nearest-triangle sides -- the
    #     crease-robust test the wall cells use below -- each weighted by the
    #     cosine between (p - closest) and the normal, so near-ties at edges
    #     and vertices count for little. A component without votes stays
    #     inside.
    labels_flat = labels.ravel()
    votes = np.zeros(n_labels + 1)
    if len(refine_pts):
        side = _exact_side()
        cosine = side / np.maximum(exact, 1e-300)
        votes = np.bincount(labels_flat[refine_mask], weights=cosine,
                            minlength=n_labels + 1)
    vote_sign = np.where(votes > 0.0, 1.0, -1.0)
    comp_sign = vote_sign.copy()

    border = np.zeros(n_labels + 1, dtype=bool)
    for face in (labels[[0, -1]], labels[:, [0, -1]], labels[:, :, [0, -1]]):
        border[face.ravel()] = True
    comp_sign[border] = 1.0
    enclosed = np.flatnonzero(~border)
    enclosed = enclosed[enclosed > 0]            # label 0 = wall cells
    n_enc = len(enclosed)
    winding = np.zeros(n_enc)
    w_min, w_max = np.zeros(n_enc), np.zeros(n_enc)
    n_samples = np.zeros(n_enc, dtype=np.int64)
    mode = np.full(n_enc, 'winding', dtype='<U7')
    split_cells, split_signs, n_split_evals = [], [], 0
    if n_enc:
        triangles = tmesh.triangles
        cells, cell_lab = _sample_cells(labels_flat, d_est, enclosed,
                                        (nx, ny, nz))
        w_cells = _winding_numbers(triangles, points[cells])
        # _sample_cells groups by ascending label, as `enclosed` is sorted
        starts = np.flatnonzero(np.r_[True, cell_lab[1:] != cell_lab[:-1]])
        groups = np.split(np.arange(len(cells)), starts[1:])
        for i, g in enumerate(groups):
            w = w_cells[g]
            near = np.abs(w - np.rint(w)) < _GWN_INTEGER_TOL
            inside = w > 0.5
            winding[i], w_min[i], w_max[i] = np.median(w), w.min(), w.max()
            n_samples[i] = len(g)
            label = enclosed[i]
            if near.all() and (inside.all() or not inside.any()):
                comp_sign[label] = -1.0 if inside[0] else 1.0
            elif near.mean() >= 0.5 and inside[near].any() and not inside[near].all():
                mode[i] = 'split'
                comp = np.flatnonzero(labels_flat == label)
                signs, n_eval = _sign_split_component(
                    comp, (nx, ny, nz), points, triangles, cells[g], w)
                split_cells.append(comp)
                split_signs.append(signs)
                n_split_evals += n_eval
            else:
                mode[i] = 'vote'
                comp_sign[label] = vote_sign[label]
        if (mode == 'split').any():
            sel = mode == 'split'
            logger.warning(
                "SDF sign: %d enclosed component(s) span both sides of the "
                "surface (w from %s to %s; a hole?) and are signed per cell "
                "from %d winding evaluations",
                int(sel.sum()), f"{w_min[sel].min():.2f}",
                f"{w_max[sel].max():.2f}", n_split_evals)
        if (mode == 'vote').any():
            sel = mode == 'vote'
            logger.warning(
                "SDF sign: %d of %d enclosed component(s) have a fractional "
                "winding number (median w = %s; holes or flipped triangles?) "
                "and are signed by the near-surface vote instead",
                int(sel.sum()), n_enc,
                ", ".join(f"{v:.2f}" for v in winding[sel][:5])
                + (", ..." if sel.sum() > 5 else ""))
    if diagnostics is not None:
        diagnostics.update(
            n_components=int(n_labels),
            enclosed=enclosed,
            size=np.bincount(labels_flat, minlength=n_labels + 1)[enclosed],
            winding=winding, w_min=w_min, w_max=w_max, n_samples=n_samples,
            mode=mode,
            vote=votes[enclosed],
            vote_sign=vote_sign[enclosed],
            sign=np.where(mode == 'split', 0.0, comp_sign[enclosed]),
            fallback=mode == 'vote',
            labels=labels)
    sign_flat = comp_sign[labels_flat]
    for comp, signs in zip(split_cells, split_signs):
        sign_flat[comp] = signs

    # Near-surface (wall) cells: enclosure says nothing — they ARE the wall.
    # Sign them from the exact nearest triangle; every wall cell lies inside
    # the refine band (wall threshold < refine threshold for any band >= 0).
    if len(refine_pts):
        wall_flat = wall.ravel()
        exact_sign = np.where(side < 0.0, -1.0, 1.0)
        wall_in_refine = wall_flat[refine_mask]
        refined_sign_full = sign_flat[refine_mask]
        refined_sign_full[wall_in_refine] = exact_sign[wall_in_refine]
        sign_flat[refine_mask] = refined_sign_full

    sdf_grid = (sign_flat * d_est).reshape((nx, ny, nz))
    grid_origin = np.array([x[0], y[0], z[0]])
    return sdf_grid, grid_origin, voxel_size


def _marching_cubes(sdf_grid: np.ndarray, level: float,
                    grid_origin: np.ndarray, voxel_size: float) -> Tuple[np.ndarray, np.ndarray]:
    """Extract isosurface from SDF using marching cubes.

    Returns (vertices, faces) arrays.
    """
    try:
        from skimage.measure import marching_cubes
        verts, faces, _, _ = marching_cubes(sdf_grid, level=level, spacing=(voxel_size, voxel_size, voxel_size))
        verts += grid_origin
        return verts, faces
    except ImportError:
        # Fallback using scipy ndimage if possible or just return empty
        # Scipy doesn't have a direct marching cubes, so we return empty to trigger normal offset fallback.
        print("Warning: skimage.measure.marching_cubes not found. Isosurface extraction unavailable.")
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)


def _offset_along_normals(mesh: HalfEdgeMesh, distance: float) -> HalfEdgeMesh:
    """Simple normal-based offset (used as fallback).

    Move each vertex along its normal by the given distance.
    This is the naive approach that fails on complex geometry.
    """
    offset_mesh = mesh.copy()
    offset_mesh.compute_vertex_normals()
    for v in offset_mesh.vertices:
        v.position += v.normal * distance
    return offset_mesh


def _fix_self_intersections(mesh: HalfEdgeMesh) -> HalfEdgeMesh:
    """Detect and repair self-intersecting faces.

    Uses spatial hashing to find intersecting face pairs,
    then resolves by local remeshing.
    """
    # Placeholder for actual mesh repair logic.
    # The marching cubes approach inherently avoids self-intersections.
    return mesh


def _smooth_mesh(mesh: HalfEdgeMesh, iterations: int) -> HalfEdgeMesh:
    """Apply simple Laplacian smoothing."""
    for _ in range(iterations):
        new_positions = []
        for v in mesh.vertices:
            neighbors = mesh.get_vertex_neighbors(v)
            if neighbors:
                avg_pos = np.mean([n.position for n in neighbors], axis=0)
                new_positions.append(avg_pos)
            else:
                new_positions.append(v.position)
        for v, p in zip(mesh.vertices, new_positions):
            v.position = p
    return mesh


def _padding_for(vertices: np.ndarray, offset: float) -> float:
    """Bounding-box padding (as a fraction of the largest axis) that still
    contains an offset of `offset` plus a couple of voxels of slack.

    The old expression used ``verts.max() - verts.min()`` -- the spread across
    ALL coordinates -- which for a part sitting at x ~= 1150 is ~1200 and made
    the padding vanish.
    """
    extents = vertices.max(axis=0) - vertices.min(axis=0)
    max_extent = float(extents.max()) if float(extents.max()) > 0 else 1.0
    return float(max(0.1, 2.0 * abs(offset) / max_extent))


def _band_for(thickness: float, direction: str) -> Tuple[float, float]:
    """(centre, half) of the wall band for `direction`.

    The wall is always the slab ``centre - half <= sd <= centre + half``, i.e.
    exactly `thickness` thick; `direction` only decides where that slab sits
    relative to the input surface (which is sd == 0):

        'inward'  -> [-thickness, 0]   input surface is the OUTER wall
        'outward' -> [0, +thickness]   input surface is the INNER wall
        'both'    -> [-thickness/2, +thickness/2]
    """
    half = abs(thickness) / 2.0
    if direction == 'outward':
        return half, half
    if direction == 'inward':
        return -half, half
    return 0.0, half


def _extract_wall_band(vertices: np.ndarray, faces: np.ndarray,
                       centre: float, half: float,
                       resolution: int, smooth_iterations: int,
                       open_surface: bool,
                       fallback_mesh: HalfEdgeMesh) -> HalfEdgeMesh:
    """Closed surface bounding the slab ``centre - half <= sd <= centre + half``.

    The boundary of that slab is the level set ``|sd - centre| == half``, so a
    SINGLE marching-cubes pass extracts BOTH walls at once (plus, for an open
    input sheet, the caps that join them) as one watertight mesh. That band
    trick is what performs the "subtract the offset body from the original"
    step -- no boolean operation is involved.

    Args:
        open_surface: sign the distance field by the nearest triangle's normal
            (needed for sheets) instead of by containment (closed solids).
        fallback_mesh: kept for signature compatibility; extraction failure
            RAISES instead of silently returning a normal offset (a plain
            offset is not a wall, and callers could not tell the difference).
    """
    padding = _padding_for(vertices, abs(centre) + half)
    sdf_grid, grid_origin, voxel_size = _compute_sdf(
        vertices, faces, resolution, padding=padding, open_surface=open_surface,
        exact_band=abs(centre) + half,
    )

    try:
        field = np.abs(sdf_grid - centre)
        wall_verts, wall_faces = _marching_cubes(field, half, grid_origin, voxel_size)
    except Exception as e:
        raise RuntimeError(
            f"SDF wall extraction failed ({e}). The wall band "
            f"[{centre - half:.3f}, {centre + half:.3f}] may be thinner than "
            f"the voxel size {voxel_size:.3f} — raise `resolution`."
        ) from e
    if len(wall_verts) == 0:
        raise RuntimeError(
            f"SDF wall extraction produced no surface: the wall band "
            f"[{centre - half:.3f}, {centre + half:.3f}] is not resolved at "
            f"voxel size {voxel_size:.3f} — raise `resolution`.")
    result_mesh = HalfEdgeMesh.from_arrays(wall_verts, wall_faces.tolist())
    _smooth_mesh(result_mesh, smooth_iterations)
    return _fix_self_intersections(result_mesh)


def shell_solid(mesh: HalfEdgeMesh, thickness: float = 1.0,
                direction: str = 'inward',
                excluded_faces: Optional[List[int]] = None,
                resolution: int = 64,
                smooth_iterations: int = 3) -> HalfEdgeMesh:
    """Hollow out a solid body, leaving a wall `thickness` thick.

    The result is the WALL itself, not a resized copy of the body: a closed,
    watertight mesh with TWO surfaces -- one where the input surface was, one
    offset from it by `thickness` -- bounding the material in between. A 1x1x1
    box shelled inward by 0.1 therefore keeps its 1x1x1 bounding box and has a
    volume of 1 - 0.8^3 = 0.488; it does NOT come back as a solid 0.8 box.

    Unlike traditional CAD offset, which fails on high-curvature geometry, this
    uses a voxel/SDF approach that cannot self-intersect.

    Algorithm:
    1. Compute a signed distance field (SDF) over the padded bounding box
       (positive OUTSIDE -- see ``_compute_sdf``).
    2. The wall is the slab ``centre - t/2 <= sd <= centre + t/2``, with
       ``centre`` chosen by `direction`. Its boundary is the level set
       ``|sd - centre| == t/2``, so ONE marching-cubes pass produces the
       original-side and the offset-side surface together -- that single pass
       is what makes the result hollow. Shared with ``thicken_surface`` in
       ``_extract_wall_band``.
    3. Smooth the result to remove voxelization artifacts.
    4. Handle excluded faces (leave open holes for those faces).

    Accuracy: the walls sit where asked to within about half a voxel, so the
    enclosed volume is good to a few percent at the default resolution; raise
    `resolution` for a thin wall on a large part (the wall must span several
    voxels or marching cubes cannot resolve it at all).

    Args:
        mesh: Input solid mesh
        thickness: Wall thickness; must be strictly positive
        direction: where the wall sits relative to the input surface --
            'inward'  = material inside it (input surface = OUTER wall)
            'outward' = material outside it (input surface = INNER wall)
            'both'    = straddles it, half the wall on either side
        excluded_faces: Face indices to exclude (create openings)
        resolution: Voxel grid resolution (higher = more precise but slower)
        smooth_iterations: Post-processing smooth passes

    Returns: HalfEdgeMesh of the hollow shell (the wall)

    Raises:
        ValueError: if `thickness` is not strictly positive. A wall of zero
            thickness is not a shell; it used to be silently accepted as
            "offset by 0", i.e. a multi-second no-op returning the input body.
    """
    if not np.isfinite(thickness) or thickness <= 0:
        raise ValueError(
            f"shell_solid: thickness must be > 0, got {thickness!r}; "
            f"a shell with no wall is not a shell")

    if len(mesh.vertices) == 0:
        return mesh.copy()

    # If faces are excluded, we effectively have an open surface, so we delegate to thicken
    if excluded_faces and len(excluded_faces) > 0:
        # Create a new mesh without the excluded faces
        open_mesh = HalfEdgeMesh()
        arrays = mesh.to_arrays()
        verts = arrays['vertices']
        faces = arrays['faces']
        valid_faces = [f for i, f in enumerate(faces) if i not in excluded_faces]
        open_mesh = HalfEdgeMesh.from_arrays(verts, valid_faces)

        # Thicken the open mesh
        return thicken_surface(open_mesh, thickness, direction, resolution, smooth_iterations)

    tm = mesh.to_trimesh()
    verts = np.asarray(tm.vertices, dtype=np.float64)
    tri_faces = np.asarray(tm.faces)

    centre, half = _band_for(thickness, direction)
    return _extract_wall_band(verts, tri_faces, centre, half,
                              resolution, smooth_iterations,
                              open_surface=False, fallback_mesh=mesh)


def thicken_surface(mesh: HalfEdgeMesh, thickness: float = 1.0,
                    direction: str = 'both',
                    resolution: int = 64,
                    smooth_iterations: int = 3) -> HalfEdgeMesh:
    """Thicken a surface body to create a solid with uniform wall thickness.

    For open surface meshes: creates a solid by offsetting in the specified
    direction and closing the boundary.

    Algorithm:
    1. Compute an oriented distance field from the input surface (positive on
       the +normal / 'outward' side)
    2. The wall is the slab {centre - thickness/2 <= sd <= centre + thickness/2},
       where centre is 0 for 'both', +thickness/2 for 'outward' and
       -thickness/2 for 'inward'. Its boundary is the level set
       |sd - centre| == thickness/2, so ONE marching-cubes pass produces the
       whole closed solid including the caps.
    3. Smooth result

    ``thickness`` is the TOTAL wall thickness in every mode. The previous
    implementation extracted |sd| == thickness for the one-sided modes, which
    both ignored `direction` (the result was symmetric) and produced a wall of
    2 * thickness.

    Accuracy: away from the open boundary the wall sits exactly where asked.
    At the rim the one-sided slab has a sharp corner (and the nearest-triangle
    sign flips discontinuously across the surface plane there), so marching
    cubes rounds the cap over roughly half a voxel; raise `resolution` if that
    matters.

    Args:
        mesh: Input surface mesh (may be open)
        thickness: Total wall thickness
        direction: 'outward' (all material on the +normal side), 'inward'
            (all material on the -normal side), or 'both' (symmetric)
        resolution: Voxel grid resolution
        smooth_iterations: Post-processing smooth passes

    Returns: HalfEdgeMesh of the thickened solid
    """
    if len(mesh.vertices) == 0:
        return mesh.copy()

    tm = mesh.to_trimesh()
    verts = np.asarray(tm.vertices, dtype=np.float64)
    tri_faces = np.asarray(tm.faces)

    centre, half = _band_for(thickness, direction)
    return _extract_wall_band(verts, tri_faces, centre, half,
                              resolution, smooth_iterations,
                              open_surface=True, fallback_mesh=mesh)
