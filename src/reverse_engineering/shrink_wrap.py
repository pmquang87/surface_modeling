import numpy as np
import trimesh
from typing import Optional, List, Tuple
from src.core.halfedge_mesh import HalfEdgeMesh

class ShrinkWrapper:
    """Projects a Sub-D control cage onto a reference mesh surface.
    
    Like a digital vacuum seal, snaps each vertex to the closest point
    on the reference mesh.
    """
    
    def __init__(self, iterations: int = 5,
                 subdivision_levels: int = 0,
                 smooth_weight: float = 0.5,
                 projection_mode: str = 'ray_cast',
                 frozen_vertices: Optional[List[int]] = None):
        """
        Args:
            iterations: number of project-then-smooth cycles
            subdivision_levels: linearly subdivide the control cage this many
                times before projecting (0 = leave the cage as given). Linear
                (non-smoothing) subdivision is used so the cage shape is
                unchanged, only denser. Defaults to 0 because the projection
                loop is the expensive part and callers that want a denser cage
                usually subdivide it themselves.
            smooth_weight: Tangential Laplacian smoothing weight (0 = no smooth, 1 = full smooth)
            projection_mode: 'closest_point' or 'ray_cast' (project along normals)
            frozen_vertices: list of vertex indices to keep frozen during projection and smoothing

        Note: with subdivision_levels > 0 the returned mesh has MORE vertices
        than the input cage, so ``frozen_vertices`` indices still refer to the
        original cage vertices (linear subdivision keeps them first in order).
        """
        self.iterations = iterations
        self.subdivision_levels = subdivision_levels
        self.smooth_weight = smooth_weight
        self.projection_mode = projection_mode
        self.frozen_vertices = frozen_vertices or []
        
    def wrap(self, cage_mesh: HalfEdgeMesh, reference_mesh: HalfEdgeMesh, frozen_vertices: Optional[List[int]] = None) -> HalfEdgeMesh:
        """Project cage mesh onto reference surface.
        
        Pipeline:
        1. Optionally subdivide the cage mesh to target density
        2. For each iteration:
           a. Apply Tangential Laplacian smoothing to prevent volumetric shrinkage
           b. Project each cage vertex via Ray-Casting (or closest point)
           c. Apply feature-snapping
        3. Recompute normals
        
        Returns: HalfEdgeMesh with vertices projected onto reference
        """
        if len(cage_mesh.vertices) == 0 or len(reference_mesh.faces) == 0:
            return cage_mesh.copy()

        if self.subdivision_levels and self.subdivision_levels > 0:
            # Linear (midpoint) subdivision: denser cage, identical shape, and
            # existing vertex indices are preserved so frozen ids stay valid.
            from src.subd.catmull_clark import subdivide
            result_mesh = subdivide(cage_mesh, self.subdivision_levels, smooth=False)
        else:
            result_mesh = cage_mesh.copy()

        # Convert reference to trimesh for fast spatial queries
        ref_trimesh = reference_mesh.to_trimesh()
        
        active_frozen = frozen_vertices if frozen_vertices is not None else self.frozen_vertices
        frozen_set = set(active_frozen)

        for it in range(self.iterations):
            # a. Apply Tangential Laplacian smoothing to prevent volumetric shrinkage
            if self.smooth_weight > 0:
                self._tangential_laplacian_smooth(result_mesh, self.smooth_weight, frozen_set=frozen_set)
                
            # b. Project vertices
            vertices = np.array([v.position for v in result_mesh.vertices])
            
            if self.projection_mode == 'ray_cast':
                result_mesh.compute_vertex_normals()
                normals = np.array([v.normal for v in result_mesh.vertices])
                if len(normals) == 0:
                    normals = np.zeros_like(vertices)
                projected_pts = self._ray_cast_projection(vertices, normals, ref_trimesh)
            else:
                projected_pts = self._project_to_surface(vertices, ref_trimesh)
                
            # c. Update positions and apply feature-snapping
            for i, v in enumerate(result_mesh.vertices):
                if i not in frozen_set:
                    v.position = self._snap_to_features(projected_pts[i], ref_trimesh)
                
        result_mesh.compute_vertex_normals()
        return result_mesh
        
    def _project_to_surface(self, vertices: np.ndarray, reference_trimesh: trimesh.Trimesh) -> np.ndarray:
        """Find closest points on reference surface for each vertex."""
        try:
            closest_points, distances, triangle_id = trimesh.proximity.closest_point(
                reference_trimesh, vertices
            )
            return closest_points
        except Exception as e:
            print(f"Error projecting to surface: {e}")
            return vertices

    def _ray_cast_projection(self, vertices: np.ndarray, normals: np.ndarray, reference_trimesh: trimesh.Trimesh) -> np.ndarray:
        """Project vertices along their normal direction via ray casting.

        Both the +normal and -normal rays are cast for every vertex, and the
        hit closest to the vertex wins. Hits are grouped by ray id with a single
        lexsort instead of one boolean mask per vertex (which was O(V * hits),
        i.e. quadratic), and every ray-miss is resolved with ONE batched
        ``closest_point`` call instead of one call per missed vertex.
        """
        try:
            intersector = trimesh.ray.ray_triangle.RayMeshIntersector(reference_trimesh)

            n = len(vertices)
            origins = np.vstack((vertices, vertices))
            directions = np.vstack((normals, -normals))

            locations, index_ray, index_tri = intersector.intersects_location(
                ray_origins=origins,
                ray_directions=directions
            )

            projected = np.array(vertices, dtype=np.float64, copy=True)
            hit_ids = np.zeros(0, dtype=np.int64)

            if len(index_ray):
                # ray i and ray i+n both belong to vertex i
                vid = np.asarray(index_ray, dtype=np.int64) % n
                dist = np.linalg.norm(locations - vertices[vid], axis=1)
                # primary key vertex id, then distance, then original order so
                # ties resolve exactly like the previous argmin did
                order = np.lexsort((np.arange(len(vid)), dist, vid))
                vid_s = vid[order]
                loc_s = locations[order]
                first = np.empty(len(vid_s), dtype=bool)
                first[0] = True
                if len(vid_s) > 1:
                    first[1:] = vid_s[1:] != vid_s[:-1]
                hit_ids = vid_s[first]
                projected[hit_ids] = loc_s[first]

            missed = np.setdiff1d(np.arange(n, dtype=np.int64), hit_ids, assume_unique=False)
            if len(missed):
                projected[missed] = self._project_to_surface(vertices[missed], reference_trimesh)

            return projected
        except Exception as e:
            print(f"Error in ray casting: {e}")
            return self._project_to_surface(vertices, reference_trimesh)


    def _tangential_laplacian_smooth(self, mesh: HalfEdgeMesh, weight: float, boundary_fixed: bool = True, frozen_set: set = None):
        """Apply Tangential Laplacian smoothing."""
        frozen_set = frozen_set or set()
        new_positions = []
        mesh.compute_vertex_normals()
        
        for i, v in enumerate(mesh.vertices):
            if i in frozen_set or (boundary_fixed and mesh.is_boundary_vertex(v)):
                new_positions.append(v.position.copy())
                continue
                
            neighbors = mesh.get_vertex_neighbors(v)
            if not neighbors:
                new_positions.append(v.position.copy())
                continue
                
            avg_pos = np.mean([n.position for n in neighbors], axis=0)
            update_vec = avg_pos - v.position
            
            # Tangential component
            normal = v.normal if v.normal is not None else np.array([0., 1., 0.])
            normal_comp = np.dot(update_vec, normal) * normal
            tangential_update = update_vec - normal_comp
            
            smoothed = v.position + weight * tangential_update
            new_positions.append(smoothed)
            
        for i, v in enumerate(mesh.vertices):
            v.position = new_positions[i]
            
    def _snap_to_features(self, point: np.ndarray, reference_trimesh: trimesh.Trimesh) -> np.ndarray:
        """Snap points to sharp features (edges/corners) of the reference mesh."""
        # Real implementation would find feature edges. 
        # Here we just pass through.
        return point
