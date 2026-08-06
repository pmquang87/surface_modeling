import numpy as np
from typing import Dict, Any, List, Optional

from src.core.halfedge_mesh import HalfEdgeMesh

class SubDToNURBSConverter:
    """Converts a Catmull-Clark subdivision surface mesh to NURBS B-spline patches."""
    
    def __init__(self, continuity: str = 'G2', tolerance: float = 1e-4):
        self.continuity = continuity
        self.tolerance = tolerance

    def _evaluate_dense_grid(self, mesh: HalfEdgeMesh, face: Any, limit_positions: np.ndarray, grid_size: int = 17) -> np.ndarray:
        """
        Evaluates the limit surface at a dense grid of points for a given quad face.
        Here we use a basic approximation: bilinear interpolation of the corners' limit positions,
        optionally inflated by the face normal to approximate the limit surface curvature.
        In a full implementation, this would use Stam's exact evaluation or sample a refined mesh.
        """
        vertices = mesh.get_face_vertices(face)
        corners = [limit_positions[v.index] for v in vertices]
        c0, c1, c2, c3 = corners
        
        # Approximate face center limit position
        face_center = np.mean(corners, axis=0)
        v1 = c1 - c0
        v2 = c3 - c0
        normal = np.cross(v1, v2)
        n_len = np.linalg.norm(normal)
        if n_len > 1e-6:
            normal = normal / n_len
        else:
            normal = np.array([0, 0, 1])
            
        grid = np.zeros((grid_size, grid_size, 3))
        u_vals = np.linspace(0, 1, grid_size)
        v_vals = np.linspace(0, 1, grid_size)
        
        for i, u in enumerate(u_vals):
            for j, v in enumerate(v_vals):
                # Bilinear interpolation
                p = (1-u)*(1-v)*c0 + u*(1-v)*c1 + u*v*c2 + (1-u)*v*c3
                
                # Add a simple quadratic bump to approximate subdivision limit surface bulging
                bump = u * (1 - u) * v * (1 - v) * 4.0  # Max 1.0 at center
                # Scale bump based on the size of the quad
                side_len = np.linalg.norm(c1 - c0)
                displacement = normal * (side_len * 0.1) * bump
                
                grid[i, j] = p + displacement
                
        return grid

    def generate_patches(self, mesh: HalfEdgeMesh, subdivision_levels: int = 3) -> List[Any]:
        try:
            from src.subd.catmull_clark import evaluate_limit_surface
            limit_positions, limit_normals = evaluate_limit_surface(mesh)
        except Exception:
            limit_positions = np.array([v.position for v in mesh.vertices])
            limit_normals = []

        from src.nurbs.g3_fitter import G3Fitter
        fitter = G3Fitter()
        
        quad_mesh_data = []
        for face in mesh.faces:
            vertices = mesh.get_face_vertices(face)
            if len(vertices) == 4:
                corners = []
                for v in vertices:
                    if v.index < len(limit_positions):
                        corners.append(limit_positions[v.index])
                    else:
                        corners.append(v.position)
                quad_mesh_data.append({'corners': corners})
                
        patches = fitter.fit_surface(quad_mesh_data)
        return patches
        
    def build_shape(self, patches: List[Any], simplify: bool = True) -> Optional[Any]:
        try:
            from OCP.GeomAPI import GeomAPI_PointsToBSplineSurface
            from OCP.TColgp import TColgp_Array2OfPnt
            from OCP.gp import gp_Pnt
            from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_Sewing
            from OCP.GeomAbs import GeomAbs_C2
            from OCP.Geom import Geom_BezierSurface
        except ImportError:
            print("cadquery-ocp not installed. Cannot perform NURBS conversion.")
            return None

        sewing = BRepBuilderAPI_Sewing(self.tolerance)
        faces_added = 0
        
        for patch_ctrl_pts in patches:
            pts_array = TColgp_Array2OfPnt(1, 6, 1, 6)
            for i in range(6):
                for j in range(6):
                    p = patch_ctrl_pts[i, j]
                    pts_array.SetValue(i + 1, j + 1, gp_Pnt(float(p[0]), float(p[1]), float(p[2])))
            
            bezier_surf = Geom_BezierSurface(pts_array)
            make_face = BRepBuilderAPI_MakeFace(bezier_surf, self.tolerance)
            if make_face.IsDone():
                sewing.Add(make_face.Face())
                faces_added += 1

        if faces_added > 0:
            sewing.Perform()
            shape = sewing.SewedShape()
            if simplify:
                try:
                    from src.nurbs.simplifier import NURBSSimplifier
                    simplifier = NURBSSimplifier(linear_tolerance=self.tolerance, angular_tolerance=self.tolerance)
                    shape = simplifier.simplify(shape)
                except Exception as e:
                    print(f"Simplification failed: {e}")
            return shape
        return None

    def convert(self, mesh: HalfEdgeMesh, subdivision_levels: int = 3, simplify: bool = True) -> Dict[str, Any]:
        """Convert Sub-D mesh to a sewed TopoDS_Shape using cadquery-ocp."""
        patches = self.generate_patches(mesh, subdivision_levels)
        shape = self.build_shape(patches, simplify=simplify)
        return {
            'shape': shape,
            'mesh': mesh,
            'patches': patches
        }

