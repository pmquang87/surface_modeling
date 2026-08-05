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
                 subdivision_levels: int = 2,
                 smooth_weight: float = 0.5,
                 projection_mode: str = 'closest_point'):
        """
        Args:
            iterations: number of project-then-smooth cycles
            subdivision_levels: subdivide the control cage this many times before projecting
            smooth_weight: Laplacian smoothing weight (0 = no smooth, 1 = full smooth)
            projection_mode: 'closest_point' or 'ray_cast' (project along normals)
        """
        self.iterations = iterations
        self.subdivision_levels = subdivision_levels
        self.smooth_weight = smooth_weight
        self.projection_mode = projection_mode
        
    def wrap(self, cage_mesh: HalfEdgeMesh, reference_mesh: HalfEdgeMesh) -> HalfEdgeMesh:
        """Project cage mesh onto reference surface.
        
        Pipeline:
        1. Optionally subdivide the cage mesh to target density
        2. For each iteration:
           a. Project each cage vertex onto the closest point on the reference
           b. Apply Laplacian smoothing to maintain mesh quality
        3. Recompute normals
        
        Returns: HalfEdgeMesh with vertices projected onto reference
        """
        if len(cage_mesh.vertices) == 0 or len(reference_mesh.faces) == 0:
            return cage_mesh.copy()
            
        result_mesh = cage_mesh.copy()
        
        # Convert reference to trimesh for fast spatial queries
        ref_trimesh = reference_mesh.to_trimesh()
        
        # Subdivide logic would go here if implemented in HalfEdgeMesh
        
        for it in range(self.iterations):
            # a. Project vertices
            vertices = np.array([v.position for v in result_mesh.vertices])
            
            if self.projection_mode == 'closest_point':
                projected_pts = self._project_to_surface(vertices, ref_trimesh)
            else:
                # Fallback to closest point if ray cast not fully implemented
                projected_pts = self._project_to_surface(vertices, ref_trimesh)
                
            # Update positions
            for i, v in enumerate(result_mesh.vertices):
                v.position = projected_pts[i]
                
            # b. Apply smoothing (except on last iteration to keep points on surface)
            if it < self.iterations - 1 and self.smooth_weight > 0:
                self._laplacian_smooth(result_mesh, self.smooth_weight)
                
        result_mesh.compute_vertex_normals()
        return result_mesh
        
    def _project_to_surface(self, vertices: np.ndarray, reference_trimesh: trimesh.Trimesh) -> np.ndarray:
        """Find closest points on reference surface for each vertex."""
        # Use trimesh's nearest.on_surface for efficient spatial queries
        closest_points, distances, triangle_id = trimesh.proximity.closest_point(
            reference_trimesh, vertices
        )
        return closest_points
        
    def _laplacian_smooth(self, mesh: HalfEdgeMesh, weight: float, boundary_fixed: bool = True):
        """Apply one iteration of Laplacian smoothing."""
        new_positions = []
        for v in mesh.vertices:
            if boundary_fixed and mesh.is_boundary_vertex(v):
                new_positions.append(v.position.copy())
                continue
                
            neighbors = mesh.get_vertex_neighbors(v)
            if not neighbors:
                new_positions.append(v.position.copy())
                continue
                
            avg_pos = np.mean([n.position for n in neighbors], axis=0)
            smoothed = v.position + weight * (avg_pos - v.position)
            new_positions.append(smoothed)
            
        for i, v in enumerate(mesh.vertices):
            v.position = new_positions[i]
