import numpy as np
from typing import List, Optional, Tuple, Any
import trimesh

from src.core.halfedge_mesh import HalfEdgeMesh


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
                 open_surface: bool = False) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute a signed distance field on a voxel grid.

    Sign convention: POSITIVE OUTSIDE / on the +normal side, negative inside.
    Note ``trimesh.proximity.signed_distance`` uses the OPPOSITE convention
    (positive inside); the sign is flipped here so the whole module can use one
    convention and marching-cubes levels read as plain offsets.

    Args:
        open_surface: sign by the nearest triangle's normal instead of by
            containment -- required for sheets, which have no inside.

    Returns:
        sdf_grid: 3D numpy array of signed distances
        grid_origin: (3,) origin of the grid
        voxel_size: float, size of each voxel
    """
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

    # Create grid
    nx = int(np.ceil(padded_extents[0] / voxel_size)) + 1
    ny = int(np.ceil(padded_extents[1] / voxel_size)) + 1
    nz = int(np.ceil(padded_extents[2] / voxel_size)) + 1

    # Ensure at least 2x2x2
    nx = max(nx, 2); ny = max(ny, 2); nz = max(nz, 2)

    x = np.linspace(bbox_min[0], bbox_min[0] + (nx-1)*voxel_size, nx)
    y = np.linspace(bbox_min[1], bbox_min[1] + (ny-1)*voxel_size, ny)
    z = np.linspace(bbox_min[2], bbox_min[2] + (nz-1)*voxel_size, nz)

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    try:
        if open_surface:
            sd = _oriented_distance(tmesh, points)
        else:
            from trimesh.proximity import signed_distance
            # trimesh: POSITIVE INSIDE -> negate for positive-outside.
            sd = -np.asarray(signed_distance(tmesh, points))
        sdf_grid = sd.reshape((nx, ny, nz))
    except Exception as e:
        # Fallback to UNSIGNED distance using scipy KDTree. This breaks the
        # sign convention above (everything reads as outside), so any offset
        # extracted from it is one-sided -- say so instead of failing quietly.
        print(f"Warning: signed distance unavailable ({e}); falling back to an "
              f"UNSIGNED point distance field -- inward offsets will be wrong.")
        from scipy.spatial import cKDTree
        tree = cKDTree(vertices)
        distances, _ = tree.query(points)
        sdf_grid = distances.reshape((nx, ny, nz))

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


def shell_solid(mesh: HalfEdgeMesh, thickness: float = 1.0,
                direction: str = 'inward',
                excluded_faces: Optional[List[int]] = None,
                resolution: int = 64,
                smooth_iterations: int = 3) -> HalfEdgeMesh:
    """Create a thin-walled shell from a solid mesh body.
    
    Unlike traditional CAD offset which fails on high-curvature geometry,
    this uses a point-cloud/voxel-based approach to handle self-intersections.
    
    Algorithm:
    1. Compute signed distance field (SDF) from the input mesh
       - Voxelize the bounding box at the given resolution
       - For each voxel, compute signed distance to the mesh surface
    2. Create offset surface by extracting the isosurface at distance=thickness
       - Use marching cubes on the SDF
    3. Boolean subtract offset from original (or just return offset shell)
    4. Smooth the result to remove voxelization artifacts
    5. Handle excluded faces (leave open holes for those faces)
    
    Args:
        mesh: Input solid mesh
        thickness: Wall thickness
        direction: which way the offset surface moves --
            'inward'  = shrink the body by `thickness`
            'outward' = grow the body by `thickness`
            'both'    = grow by `thickness / 2` (the outer face of a wall
                        centred on the input surface)
        excluded_faces: Face indices to exclude (create openings)
        resolution: Voxel grid resolution (higher = more precise but slower)
        smooth_iterations: Post-processing smooth passes

    Returns: HalfEdgeMesh of the offset (shell) surface
    """
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

    # Signed offset along the OUTWARD normal direction.
    if direction == 'inward':
        offset = -abs(thickness)
    elif direction == 'outward':
        offset = abs(thickness)
    else:  # both
        offset = abs(thickness) / 2.0

    padding = _padding_for(verts, offset)
    sdf_grid, grid_origin, voxel_size = _compute_sdf(verts, tri_faces, resolution, padding=padding)

    # _compute_sdf is positive OUTSIDE, so the surface offset outward by d is
    # exactly the level set sdf == d (and d < 0 shrinks the body).
    level = offset

    try:
        shell_verts, shell_faces = _marching_cubes(sdf_grid, level, grid_origin, voxel_size)
        if len(shell_verts) > 0:
            result_mesh = HalfEdgeMesh.from_arrays(shell_verts, shell_faces.tolist())
            _smooth_mesh(result_mesh, smooth_iterations)
            return _fix_self_intersections(result_mesh)
        else:
            return _offset_along_normals(mesh, offset)
    except Exception as e:
        print(f"SDF shelling failed, falling back to normal offset: {e}")
        return _offset_along_normals(mesh, offset)


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

    half = abs(thickness) / 2.0
    if direction == 'outward':
        centre = half
    elif direction == 'inward':
        centre = -half
    else:  # both
        centre = 0.0

    # Room for the far side of the wall plus slack.
    padding = _padding_for(verts, abs(centre) + half)
    sdf_grid, grid_origin, voxel_size = _compute_sdf(
        verts, tri_faces, resolution, padding=padding, open_surface=True
    )

    try:
        # |sd - centre| == half is exactly the boundary of the requested slab.
        field = np.abs(sdf_grid - centre)
        thick_verts, thick_faces = _marching_cubes(field, half, grid_origin, voxel_size)

        if len(thick_verts) > 0:
            result_mesh = HalfEdgeMesh.from_arrays(thick_verts, thick_faces.tolist())
            _smooth_mesh(result_mesh, smooth_iterations)
            return _fix_self_intersections(result_mesh)
        else:
            return _offset_along_normals(mesh, centre)
    except Exception as e:
        print(f"SDF thickening failed, falling back to normal offset: {e}")
        return _offset_along_normals(mesh, centre)
