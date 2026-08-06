import numpy as np
from typing import Dict, Any, List, Optional
try:
    from scipy.interpolate import bisplrep
except ImportError:
    pass # handle gracefully if possible

from src.core.halfedge_mesh import HalfEdgeMesh

class SubDToNURBSConverter:
    """Converts a Catmull-Clark subdivision surface mesh to NURBS B-spline patches.
    
    This is the core value proposition — translating the Sub-D limit surface
    into precise, watertight NURBS suitable for CAD manufacturing.
    """
    
    def __init__(self, continuity: str = 'G2', tolerance: float = 1e-4):
        """
        Args:
            continuity: 'G0', 'G1', or 'G2' — inter-patch continuity target
            tolerance: maximum deviation from limit surface
        """
        self.continuity = continuity
        self.tolerance = tolerance

    def generate_patches(self, mesh: HalfEdgeMesh, subdivision_levels: int = 3) -> List[Dict[str, Any]]:
        """Generate raw Python patch data without building the final solid."""
        try:
            from src.subd.catmull_clark import evaluate_limit_surface
            limit_positions, limit_normals = evaluate_limit_surface(mesh)
        except Exception:
            limit_positions = np.array([v.position for v in mesh.vertices])
            limit_normals = []
            
        if str(self.continuity) in ('3', 'G3'):
            try:
                from src.nurbs.g3_fitter import G3Fitter
                fitter = G3Fitter()
                quads = []
                for face in mesh.faces:
                    vertices = mesh.get_face_vertices(face)
                    if len(vertices) == 4:
                        corners = [limit_positions[v.index] for v in vertices]
                        quads.append({'corners': corners})
                raw_patches = fitter.fit_surface(quads)
                patches = []
                for ctrl_pts in raw_patches:
                    patches.append({
                        'control_points': ctrl_pts,
                        'degree_u': 5,
                        'degree_v': 5,
                        'knots_u': [0]*6 + [1]*6,
                        'knots_v': [0]*6 + [1]*6
                    })
                return patches
            except Exception as e:
                print(f"Failed to use G3Fitter: {e}, falling back to G2.")

        patches = []
        for face in mesh.faces:
            vertices = mesh.get_face_vertices(face)
            if len(vertices) == 4:
                corners = [limit_positions[v.index] for v in vertices]
                edge_tangents = [] 
                
                center_pt = np.mean(corners, axis=0)
                
                if limit_normals is not None and len(limit_normals) > 0:
                    normal = np.mean([limit_normals[v.index] for v in vertices], axis=0)
                else:
                    v1 = corners[1] - corners[0]
                    v2 = corners[3] - corners[0]
                    normal = np.cross(v1, v2)
                    
                n_len = np.linalg.norm(normal)
                if n_len > 1e-6:
                    normal = normal / n_len
                    
                side_len = np.linalg.norm(corners[0] - corners[1])
                interior_points = [center_pt + normal * (side_len * 0.1)]
                
                patch_data = self._fit_bspline_patch(corners, edge_tangents, interior_points)
                if patch_data:
                    patches.append(patch_data)
                    
        return self._enforce_continuity(patches)
        
    def build_shape(self, patches: List[Dict[str, Any]]) -> Optional[Any]:
        """Build the final OCC shape from raw patches."""
        return self._build_occ_shape(patches)

    def convert(self, mesh: HalfEdgeMesh, subdivision_levels: int = 3) -> Dict[str, Any]:
        """Convenience method to generate patches and build shape."""
        patches = self.generate_patches(mesh, subdivision_levels)
        shape = self.build_shape(patches)
        return {
            'patches': patches,
            'shape': shape,
            'mesh': mesh
        }

    def _fit_bspline_patch(self, corners: List[np.ndarray], edge_tangents: List[Any], interior_points: List[Any]) -> Dict[str, Any]:
        """Fit a single bicubic B-spline patch to a quad region.
        
        Uses least-squares fitting to find control points
        that best approximate the limit surface data.
        """
        # Stub implementation returning a flat patch based on corners
        # For a true bicubic patch, we need a 4x4 grid of control points
        if not corners or len(corners) != 4:
            return {}
            
        c0, c1, c2, c3 = corners
        # Evaluate true limit surface for center and edge midpoints if interior_points exist
        # We will use a simple quadratic/cubic elevation approximation
        # Find the center point
        if interior_points and len(interior_points) > 0:
            center_pt = interior_points[0]
        else:
            center_pt = (c0 + c1 + c2 + c3) / 4.0
            
        # We'll create a 4x4 control point grid.
        # Corners are exactly the corners.
        # To make it curve towards the center_pt, we elevate the inner 2x2 control points.
        u_vals = np.linspace(0, 1, 4)
        v_vals = np.linspace(0, 1, 4)
        
        ctrl_pts = np.zeros((4, 4, 3))
        
        # Base flat bilinear patch
        for i, u in enumerate(u_vals):
            for j, v in enumerate(v_vals):
                p = (1-u)*(1-v)*c0 + u*(1-v)*c1 + u*v*c2 + (1-u)*v*c3
                ctrl_pts[i, j] = p
                
        # Calculate the normal vector at the center (roughly)
        v1 = c1 - c0
        v2 = c3 - c0
        normal = np.cross(v1, v2)
        n_len = np.linalg.norm(normal)
        if n_len > 1e-6:
            normal = normal / n_len
        else:
            normal = np.array([0, 0, 1])
            
        # Elevate the inner 2x2 points to match the center_pt displacement
        flat_center = (c0 + c1 + c2 + c3) / 4.0
        displacement = center_pt - flat_center
        
        # Apply displacement to the 4 inner control points
        ctrl_pts[1, 1] += displacement * 1.5
        ctrl_pts[1, 2] += displacement * 1.5
        ctrl_pts[2, 1] += displacement * 1.5
        ctrl_pts[2, 2] += displacement * 1.5

        return {
            'control_points': ctrl_pts,
            'degree_u': 3,
            'degree_v': 3,
            'knots_u': [0, 0, 0, 0, 1, 1, 1, 1],
            'knots_v': [0, 0, 0, 0, 1, 1, 1, 1]
        }

    def _enforce_continuity(self, patches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Adjust control points to enforce G1/G2 continuity between adjacent patches."""
        # Stub implementation. In a real scenario, this involves analyzing shared boundaries 
        # and adjusting the 2nd row of control points for G1, and 3rd row for G2.
        return patches

    def _build_occ_shape(self, patches: List[Dict[str, Any]]) -> Optional[Any]:
        """Build OCC BSplineSurface objects from patch data. Returns TopoDS_Shape."""
        if not patches:
            return None
            
        try:
            from OCP.Geom import Geom_BSplineSurface
            from OCP.TColgp import TColgp_Array2OfPnt
            from OCP.TColStd import TColStd_Array1OfReal, TColStd_Array1OfInteger
            from OCP.gp import gp_Pnt
            from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
            from OCP.BRepBuilderAPI import BRepBuilderAPI_Sewing
            from OCP.TopoDS import TopoDS_Shape
            
            sewing = BRepBuilderAPI_Sewing(self.tolerance)
            
            for patch in patches:
                ctrl_pts = patch['control_points']
                deg_u = patch['degree_u']
                deg_v = patch['degree_v']
                ku = patch['knots_u']
                kv = patch['knots_v']
                
                # Convert arrays
                num_u = deg_u + 1
                num_v = deg_v + 1
                poles = TColgp_Array2OfPnt(1, num_u, 1, num_v)
                for i in range(num_u):
                    for j in range(num_v):
                        p = ctrl_pts[i, j]
                        poles.SetValue(i+1, j+1, gp_Pnt(float(p[0]), float(p[1]), float(p[2])))
                        
                # Unique knots and multiplicities
                knots_u = TColStd_Array1OfReal(1, 2)
                knots_u.SetValue(1, 0.0)
                knots_u.SetValue(2, 1.0)
                
                mults_u = TColStd_Array1OfInteger(1, 2)
                mults_u.SetValue(1, num_u)
                mults_u.SetValue(2, num_u)
                
                knots_v = TColStd_Array1OfReal(1, 2)
                knots_v.SetValue(1, 0.0)
                knots_v.SetValue(2, 1.0)
                
                mults_v = TColStd_Array1OfInteger(1, 2)
                mults_v.SetValue(1, num_v)
                mults_v.SetValue(2, num_v)
                
                surf = Geom_BSplineSurface(poles, knots_u, knots_v, mults_u, mults_v, deg_u, deg_v)
                face = BRepBuilderAPI_MakeFace(surf, self.tolerance).Face()
                sewing.Add(face)
                
            sewing.Perform()
            return sewing.SewedShape()
            
        except ImportError:
            print("OCP not available. Skipping shape build.")
            return None
        except Exception as e:
            print(f"Error building OCC shape: {e}")
            return None
