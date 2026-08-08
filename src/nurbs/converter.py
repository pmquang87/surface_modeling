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

    def generate_patches(self, mesh: HalfEdgeMesh, subdivision_levels: int = 3,
                         reference_mesh: Optional[HalfEdgeMesh] = None) -> List[Any]:
        ref_tm = None
        if reference_mesh is not None and len(reference_mesh.faces) > 0:
            ref_tm = reference_mesh.to_trimesh()

        if ref_tm is None:
            try:
                from src.subd.catmull_clark import evaluate_limit_surface
                limit_positions, limit_normals = evaluate_limit_surface(mesh)
            except Exception:
                limit_positions = np.array([v.position for v in mesh.vertices])
        else:
            # Reverse engineering: the cage vertices already lie ON the target
            # surface (shrink wrap projects them there). Catmull-Clark limit
            # positions would pull low-valence vertices millimetres off it.
            limit_positions = np.array([v.position for v in mesh.vertices])
            mesh.compute_vertex_normals()

        from src.nurbs.g3_fitter import G3Fitter
        # Map the requested continuity level onto the fitter's soft-constraint
        # weight; data equations have weight 1, so values far above ~50 make
        # the solver satisfy smoothness identities while ignoring the surface.
        continuity_weights = {'G0': 0.0, 'G1': 5.0, 'G2': 20.0, 'G3': 50.0}
        fitter = G3Fitter(continuity_weight=continuity_weights.get(self.continuity, 20.0))

        quad_faces = [f for f in mesh.faces if len(mesh.get_face_vertices(f)) == 4]
        face_to_idx = {f.index: idx for idx, f in enumerate(quad_faces)}

        grid_size = 6
        quad_mesh_data = []
        for face in quad_faces:
            vertices = mesh.get_face_vertices(face)
            corners = []
            for v in vertices:
                if v.index < len(limit_positions):
                    corners.append(limit_positions[v.index])
                else:
                    corners.append(v.position)

            neighbors = []
            he = face.half_edge
            curr = he
            for _ in range(4):
                if curr.twin and curr.twin.face and curr.twin.face.index in face_to_idx:
                    neighbors.append(face_to_idx[curr.twin.face.index])
                else:
                    neighbors.append(-1)
                curr = curr.next

            if ref_tm is None:
                dense_points = self._evaluate_dense_grid(mesh, face, limit_positions, grid_size=grid_size)
                corner_normals = None
            else:
                dense_points = None  # filled below by batched projection
                # per-vertex normals give both patches at a shared edge the
                # same tangent-plane boundary curve (sewable, but curved)
                corner_normals = [v.normal.copy() for v in vertices]
            quad_mesh_data.append({
                'corners': corners,
                'dense_points': dense_points,
                'corner_normals': corner_normals,
                'neighbors': neighbors
            })

        if ref_tm is not None and quad_mesh_data:
            # Sample each quad bilinearly and project the samples onto the
            # reference surface in ONE batched query, so the fitter fits the
            # actual scanned geometry instead of a synthetic bump heuristic.
            import trimesh as _trimesh
            u = np.linspace(0.0, 1.0, grid_size)
            uu, vv = np.meshgrid(u, u, indexing='ij')
            w0 = ((1 - uu) * (1 - vv))[..., None]
            w1 = (uu * (1 - vv))[..., None]
            w2 = (uu * vv)[..., None]
            w3 = ((1 - uu) * vv)[..., None]
            grids = []
            for qd in quad_mesh_data:
                c0, c1, c2, c3 = qd['corners']
                g = w0 * c0 + w1 * c1 + w2 * c2 + w3 * c3
                grids.append(g.reshape(-1, 3))
            all_pts = np.vstack(grids)
            projected, _, _ = _trimesh.proximity.closest_point(ref_tm, all_pts)
            projected = np.asarray(projected).reshape(len(quad_mesh_data), grid_size, grid_size, 3)
            for qd, g in zip(quad_mesh_data, projected):
                qd['dense_points'] = g

        patches = fitter.fit_surface(quad_mesh_data)
        return patches
        
    def build_shape(self, patches: List[Any], simplify: bool = False) -> Optional[Any]:
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
            shape = self._promote_closed_shells(shape)
            if simplify:
                try:
                    from src.nurbs.simplifier import NURBSSimplifier
                    # angular tolerance is in radians, not mm — keep the
                    # simplifier's intended ~0.1 rad merge threshold
                    simplifier = NURBSSimplifier(linear_tolerance=self.tolerance, angular_tolerance=0.1)
                    shape = simplifier.simplify(shape)
                except Exception as e:
                    print(f"Simplification failed: {e}")
            return shape
        return None

    @staticmethod
    def _promote_closed_shells(shape: Any) -> Any:
        """Turn closed shells into solids so CAD importers see solid bodies."""
        try:
            from OCP.TopExp import TopExp_Explorer, TopExp
            from OCP.TopAbs import TopAbs_SHELL, TopAbs_FACE
            from OCP.TopoDS import TopoDS, TopoDS_Compound
            from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeSolid
            from OCP.ShapeFix import ShapeFix_Solid
            from OCP.BRep import BRep_Builder
            from OCP.TopTools import TopTools_IndexedMapOfShape

            children = []
            faces_in_shells = TopTools_IndexedMapOfShape()
            exp = TopExp_Explorer(shape, TopAbs_SHELL)
            while exp.More():
                shell = TopoDS.Shell_s(exp.Current())
                TopExp.MapShapes_s(shell, TopAbs_FACE, faces_in_shells)
                child = shell
                if shell.Closed():
                    mk = BRepBuilderAPI_MakeSolid(shell)
                    if mk.IsDone():
                        fixer = ShapeFix_Solid(mk.Solid())
                        fixer.Perform()
                        child = fixer.Solid()
                children.append(child)
                exp.Next()

            # keep faces that sewing could not attach to any shell —
            # dropping them would silently delete surface area
            exp = TopExp_Explorer(shape, TopAbs_FACE)
            loose = 0
            while exp.More():
                face = exp.Current()
                if not faces_in_shells.Contains(face):
                    children.append(face)
                    loose += 1
                exp.Next()
            if loose:
                print(f"NURBS conversion: {loose} face(s) could not be sewn into "
                      f"a shell and are kept as free faces")

            if not children:
                return shape
            if len(children) == 1:
                return children[0]
            comp = TopoDS_Compound()
            builder = BRep_Builder()
            builder.MakeCompound(comp)
            for c in children:
                builder.Add(comp, c)
            return comp
        except Exception as e:
            print(f"Shell-to-solid promotion failed: {e}")
            return shape

    def convert(self, mesh: HalfEdgeMesh, subdivision_levels: int = 3, simplify: bool = False,
                reference_mesh: Optional[HalfEdgeMesh] = None) -> Dict[str, Any]:
        """Convert Sub-D mesh to a sewed TopoDS_Shape using cadquery-ocp.

        reference_mesh: optional dense mesh (e.g. the original STL) — when
        given, patches are fitted to that surface instead of a synthetic
        approximation of the Catmull-Clark limit surface.
        """
        patches = self.generate_patches(mesh, subdivision_levels, reference_mesh=reference_mesh)
        shape = self.build_shape(patches, simplify=simplify)
        return {
            'shape': shape,
            'mesh': mesh,
            'patches': patches
        }

