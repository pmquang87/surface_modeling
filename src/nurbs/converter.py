import numpy as np
from typing import Dict, Any, List, Optional

from src.core.halfedge_mesh import HalfEdgeMesh

class SubDToNURBSConverter:
    """Converts a Catmull-Clark subdivision surface mesh to NURBS B-spline patches."""
    
    def __init__(self, continuity: str = 'G2', tolerance: float = 1e-4):
        self.continuity = continuity
        self.tolerance = tolerance

    # Subdivision levels behind the no-reference dense grids: 2**2 + 1 = 5
    # exact limit samples per side, resampled to the fitter's 6 (see
    # _limit_dense_grids for why not 3 levels / 9 samples).
    dense_grid_levels = 2

    def _limit_dense_grids(self, mesh: HalfEdgeMesh, grid_size: int = 6,
                           levels: Optional[int] = None) -> Dict[int, np.ndarray]:
        """Dense samples of the TRUE Catmull-Clark limit surface, per quad.

        Returns ``{face.index: (grid_size, grid_size, 3)}`` for every quad of
        `mesh`, laid out like the fitter's tensor grid: entry ``[i, j]`` at
        ``(u, v) = (i, j) / (grid_size - 1)``, u along c0 -> c1, v along
        c0 -> c3 of ``mesh.get_face_vertices(face)``.

        The mesh is subdivided `levels` times with the repo's own
        `_subdivide_once` (creases, corners and darts included), and each
        quad's (2**levels + 1)^2 sub-grid of vertices is found with
        `subdivide_with_quad_lineage`. The limit positions of those vertices
        are exact limit points at the dyadic parameters (i, j) / 2**levels of
        the face; the corner samples are the limit positions of the quad's
        own cage vertices, so on a closed cage they agree with the patch
        corners to rounding. (On an
        open cage `evaluate_limit_surface` keeps boundary vertices where
        the subdivided mesh has them, which is not where the coarse cage
        has them; the boundary samples do not enter the fit, whose
        boundary rows are fixed.)

        When grid_size is not 2**levels + 1 the samples are resampled by
        tensor Lagrange interpolation of degree 2**levels in u and v. At the
        default 2 levels that is quartic through 5 x 5 nodes, which is exact
        wherever the limit is a bicubic polynomial on the face (every face
        whose 1-ring is regular, i.e. a uniform bicubic B-spline piece, and
        every planar face) and reproduces the nodes, so the corners stay
        exact. Faces at an extraordinary vertex are not polynomial there:
        measured against 3 levels (exact samples at the eighths), the
        resampling is off by at most 1.2e-3 of the bounding-box diagonal on
        a unit-cube cage (every face touches four valence-3 vertices, the
        worst case), 7e-5 on create_sphere, and 0 (to rounding) on
        create_torus and the fully creased box; the old bump was off by up
        to 0.1 of the diagonal. 3 levels would avoid the resampling but
        costs ~5x the subdivision and limit evaluation (5000-quad torus
        cage: ~21 s vs ~104 s, on a conversion of ~50-120 s) and, at
        9 x 9, 2.25x the fitter's data equations, while the patch error of
        this path is dominated by its straight boundary chords anyway.

        Raises (ValueError / anything from the subdivision) when the quad
        lineage cannot be traced; the caller falls back to the heuristic.
        """
        from src.subd.catmull_clark import (subdivide_with_quad_lineage,
                                            evaluate_limit_surface)
        if levels is None:
            levels = self.dense_grid_levels
        fine, lineage = subdivide_with_quad_lineage(mesh, levels)
        if not lineage:
            return {}
        fine_limit, _ = evaluate_limit_surface(fine)
        n = 2 ** levels + 1
        nodes = np.linspace(0.0, 1.0, n)
        targets = np.linspace(0.0, 1.0, grid_size)
        if grid_size == n:
            L = None
        else:
            # L[t, a] = a-th Lagrange basis polynomial on `nodes` at target t
            L = np.ones((grid_size, n))
            for a in range(n):
                for b in range(n):
                    if b != a:
                        L[:, a] *= (targets - nodes[b]) / (nodes[a] - nodes[b])
        grids = {}
        for fidx, idx in lineage.items():
            g = fine_limit[idx]
            if L is not None:
                g = np.einsum('ta,abk,sb->tsk', L, g, L)
            grids[fidx] = g
        return grids

    def _evaluate_dense_grid(self, mesh: HalfEdgeMesh, face: Any, limit_positions: np.ndarray, grid_size: int = 17) -> np.ndarray:
        """
        HEURISTIC dense grid for one quad, no longer used by generate_patches
        (which samples the true limit surface, `_limit_dense_grids`); it is
        its fallback when the quad lineage cannot be traced.
        Bilinear interpolation of the corners' limit positions, inflated by
        the face normal by a fixed bump of 0.025 * |c1 - c0| at the centre.
        The bump ignores creases and is only a crude stand-in for the limit
        surface: a fully creased unit box came out with volume 1.0667.
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
        # Scale-relative degeneracy test, the M15 rule of
        # HalfEdgeMesh.compute_face_normals: |n| is an area, so compare it with
        # the squared edge lengths. The old absolute 1e-6 bumped every valid
        # quad with sub-1e-3 edges in-plane along z. A zero-area quad has no
        # normal and gets no bump at all.
        d = np.asarray(corners, dtype=np.float64)
        d = d - np.roll(d, -1, axis=0)
        if n_len > 1e-12 * np.einsum('ij,ij->', d, d):
            normal = normal / n_len
        else:
            normal = np.zeros(3)
            
        grid = np.zeros((grid_size, grid_size, 3))
        u_vals = np.linspace(0, 1, grid_size)
        v_vals = np.linspace(0, 1, grid_size)
        
        for i, u in enumerate(u_vals):
            for j, v in enumerate(v_vals):
                # Bilinear interpolation
                p = (1-u)*(1-v)*c0 + u*(1-v)*c1 + u*v*c2 + (1-u)*v*c3
                
                # Add a simple quadratic bump to approximate subdivision limit surface bulging
                bump = u * (1 - u) * v * (1 - v) * 4.0  # Max 0.25 at center
                # Scale bump based on the size of the quad
                side_len = np.linalg.norm(c1 - c0)
                displacement = normal * (side_len * 0.1) * bump
                
                grid[i, j] = p + displacement
                
        return grid

    def generate_patches(self, mesh: HalfEdgeMesh,
                         reference_mesh: Optional[HalfEdgeMesh] = None) -> List[Any]:
        """Fit one 6x6 Bezier patch per quad face of `mesh`.

        One patch per cage quad: the cage is fitted at the density it arrives
        with. (Without a reference mesh the cage is subdivided internally,
        but only to sample the limit surface for the fit,
        `_limit_dense_grids`.) (An unused ``subdivision_levels`` argument used to be
        accepted here and in ``convert`` -- it was never read; use
        ``src.subd.catmull_clark.subdivide`` on the cage beforehand if you want
        a denser one.)
        """
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
        limit_grids = {}
        if ref_tm is None and quad_faces:
            try:
                limit_grids = self._limit_dense_grids(mesh, grid_size=grid_size)
            except Exception as e:
                print(f"NURBS conversion: limit-surface sampling failed ({e}); "
                      f"using the approximate dense grid")
                limit_grids = {}
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
                dense_points = limit_grids.get(face.index)
                if dense_points is None:
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
                    # use the simplifier's conservative defaults (0.001 deg
                    # merge angle, Confusion-level linear tolerance); it keeps
                    # the original shape when the merge moves the volume
                    simplifier = NURBSSimplifier()
                    shape = simplifier.simplify(shape)
                    if simplifier.last_report and not simplifier.last_report.get("accepted", True):
                        print(f"Simplification rejected: {simplifier.last_report.get('reason')}")
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

    def convert(self, mesh: HalfEdgeMesh, simplify: bool = False,
                reference_mesh: Optional[HalfEdgeMesh] = None) -> Dict[str, Any]:
        """Convert Sub-D mesh to a sewed TopoDS_Shape using cadquery-ocp.

        reference_mesh: optional dense mesh (e.g. the original STL) — when
        given, patches are fitted to that surface instead of a synthetic
        approximation of the Catmull-Clark limit surface.
        """
        patches = self.generate_patches(mesh, reference_mesh=reference_mesh)
        shape = self.build_shape(patches, simplify=simplify)
        return {
            'shape': shape,
            'mesh': mesh,
            'patches': patches
        }

