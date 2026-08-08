import warnings
import numpy as np
import trimesh
import scipy.sparse as sp
from typing import Optional, List, Tuple
from src.core.halfedge_mesh import HalfEdgeMesh

# Empirically measured on icospheres and on the real topology-optimisation
# STLs: after pair-merging adjacent triangles into quads and then splitting
# every quad into 4 / every leftover triangle into 3, one decimated triangle
# becomes ~2.15 final quads. Divide the requested quad count by this to get the
# decimation budget.
QUADS_PER_DECIMATED_TRIANGLE = 2.2


class QuadWrapper:
    """Generates a quad-dominant control cage wrapped around a dense reference mesh.

    Pipeline:
    1. Decimate the reference mesh to a triangle budget derived from
       ``target_face_count`` (see below)
    2. Merge adjacent triangle pairs into convex quads
    3. Split every quad into 4 and every leftover triangle into 3, giving a
       pure-quad cage
    4. Relax the cage onto the reference surface and untangle concave quads

    ``target_face_count`` is the approximate number of quads in the RETURNED
    cage. Because step 3 multiplies the count, the decimation budget in step 1
    is ``target_face_count / QUADS_PER_DECIMATED_TRIANGLE`` triangles (floor of
    4). It is an approximation: the merge ratio depends on how planar the input
    is, so expect the final count within roughly +-15% of the request.

    NOT IMPLEMENTED: ``feature_angle`` and ``frozen_face_ids`` are accepted for
    API compatibility but have no effect -- there is no feature-edge detection
    and no face freezing in this implementation. Setting either raises a
    warning.

    The curvature / cross-field stage (``_compute_curvatures`` and
    ``_propagate_cross_field``) is NOT part of the pipeline: the MIQ stage is
    simulated by quadric decimation and ignores any field it is given, so
    computing one just burned seconds on large meshes. Both methods are kept
    for reference and for a future real MIQ solver.
    """

    def __init__(self, target_face_count: int = 500,
                 smoothing_weight: float = 0.6,
                 feature_angle: float = 30.0,
                 frozen_face_ids: Optional[List[int]] = None):
        self.target_face_count = target_face_count
        self.smoothing_weight = smoothing_weight
        self.feature_angle = feature_angle
        self.frozen_face_ids = frozen_face_ids or []
        self._warned_unimplemented = False
        if frozen_face_ids or feature_angle != 30.0:
            self._warn_unimplemented(
                frozen=bool(frozen_face_ids),
                feature=(feature_angle != 30.0),
            )

    def _warn_unimplemented(self, frozen: bool = False, feature: bool = False) -> None:
        """One warning per wrapper instance about the inert parameters."""
        if self._warned_unimplemented:
            return
        names = []
        if frozen:
            names.append("frozen_face_ids")
        if feature:
            names.append("feature_angle")
        if not names:
            return
        self._warned_unimplemented = True
        warnings.warn(
            f"QuadWrapper: {' and '.join(names)} is NOT implemented and will be "
            f"ignored -- there is no feature-edge detection or face freezing in "
            f"this quad wrap. The cage is built purely from quadric decimation.",
            UserWarning,
            stacklevel=3,
        )

    def wrap(self, reference_mesh: HalfEdgeMesh, frozen_face_ids: Optional[List[int]] = None) -> HalfEdgeMesh:
        if frozen_face_ids:
            self._warn_unimplemented(frozen=True)

        if len(reference_mesh.faces) == 0:
            return HalfEdgeMesh()

        tri_mesh = reference_mesh.to_trimesh()

        try:
            # 1.-3. Parametrization-Based Quad Meshing (MIQ solver stand-in).
            # No cross field is computed: _miq_parametrization ignores it.
            param_V, param_F, param_field = self._miq_parametrization(tri_mesh, None)

            # 4. Extract pure quads from the parametrization
            quad_V, quad_F = self._extract_pure_quads(param_V, param_F, param_field)

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"WARNING: quad wrap failed ({e}); returning the ORIGINAL dense "
                  f"triangle mesh unchanged — downstream NURBS conversion will "
                  f"find no quads and produce no patches.")
            return reference_mesh.copy()
        
        # Build the initial pure quad mesh
        he_mesh = HalfEdgeMesh()
        vertex_map = {}
        for i, v in enumerate(quad_V):
            vert = he_mesh.add_vertex(v.tolist())
            vertex_map[i] = vert.index
            
        for q in quad_F:
            he_mesh.add_face([vertex_map[v] for v in q])
            
        # 5. Relax the final quad mesh (Shrinkwrap/Laplacian)
        pre_relax = np.array([v.position.copy() for v in he_mesh.vertices])
        he_mesh = self._relax_mesh(he_mesh, reference_mesh)

        # Smoothing + projection can fold quads concave; the renderer treats
        # concave quads as holes, so untangle them (the pre-relax cage from
        # _extract_pure_quads is convex-clean and serves as the safe state).
        self._repair_concave_quads(he_mesh, reference_mesh, pre_relax)

        return he_mesh
        
    def _compute_curvatures(self, mesh: trimesh.Trimesh) -> Tuple[np.ndarray, np.ndarray]:
        """Compute approximate principal curvature directions at vertices.

        NOT called by ``wrap()``: ``_miq_parametrization`` is a decimation
        stand-in that ignores any cross field, so running this only cost time
        (seconds on a 60k-triangle STL). Kept for a future real MIQ solver.
        """
        normals = mesh.vertex_normals
        b1 = np.zeros_like(normals)
        b1[:, 0] = 1.0
        dot_n_b1 = np.abs(np.sum(normals * b1, axis=1))
        mask = dot_n_b1 > 0.9
        b1[mask] = [0.0, 1.0, 0.0]
        
        b1 = b1 - normals * np.sum(normals * b1, axis=1)[:, None]
        norms_b1 = np.linalg.norm(b1, axis=1, keepdims=True)
        b1 = np.divide(b1, norms_b1, out=np.zeros_like(b1), where=norms_b1>1e-10)
        
        b2 = np.cross(normals, b1)
        return b1, b2
        
    def _propagate_cross_field(self, mesh: trimesh.Trimesh, U: np.ndarray, V: np.ndarray) -> np.ndarray:
        """Diffuse the cross field across the mesh.

        NOT called by ``wrap()`` -- see ``_compute_curvatures``.
        """
        edges = mesh.edges_unique
        row, col = edges[:, 0], edges[:, 1]
        data = np.ones(len(edges))
        
        n_v = len(mesh.vertices)
        adj = sp.coo_matrix((data, (row, col)), shape=(n_v, n_v))
        adj = adj + adj.T
        
        field = U.copy()
        normals = mesh.vertex_normals
        
        for _ in range(5):
            field = adj.dot(field)
            norms = np.linalg.norm(field, axis=1, keepdims=True)
            field = np.divide(field, norms, out=np.zeros_like(field), where=norms>1e-10)
            
            field = field - normals * np.sum(normals * field, axis=1)[:, None]
            norms = np.linalg.norm(field, axis=1, keepdims=True)
            field = np.divide(field, norms, out=np.zeros_like(field), where=norms>1e-10)
            
        return field
        
    def _miq_parametrization(self, mesh: trimesh.Trimesh, cross_field: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simulates a Mixed-Integer Quadrangulation (MIQ) solver parametrization.

        ``cross_field`` is ignored (see the class docstring); the stand-in is a
        quadric decimation to the triangle budget that ``_extract_pure_quads``
        will blow up into roughly ``target_face_count`` quads.
        """
        target_triangles = max(4, int(round(self.target_face_count / QUADS_PER_DECIMATED_TRIANGLE)))
        try:
            # trimesh >= 5 renamed the args to (percent, face_count, aggression);
            # passing the count positionally lands in `percent` and raises.
            decimated = mesh.simplify_quadric_decimation(face_count=target_triangles)
        except TypeError:
            # older trimesh: face_count was the first positional argument
            decimated = mesh.simplify_quadric_decimation(target_triangles)
        except Exception as e:
            print(f"Warning: quadric decimation failed ({e}); continuing with "
                  f"undecimated mesh of {len(mesh.faces)} faces")
            decimated = mesh

        decimated = self._repair_decimated(decimated)
            
        return np.array(decimated.vertices), np.array(decimated.faces), np.zeros((len(decimated.vertices), 3))

    def _repair_decimated(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Clean up quadric-decimation artifacts.

        fast_simplification routinely emits duplicated vertices, degenerate
        slivers, tiny debris components and a few non-manifold edges even for
        perfectly watertight input. Any of these corrupts the half-edge cage
        and later splits the sewn B-Rep into multiple shells.
        """
        from collections import Counter

        mesh = mesh.copy()
        mesh.merge_vertices()
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.update_faces(mesh.unique_faces())
        mesh.remove_unreferenced_vertices()

        # Drop debris components: a closed surface needs >= 4 faces, and
        # decimation scraps are tiny compared to any real body.
        components = mesh.split(only_watertight=False)
        if len(components) > 1:
            min_faces = max(8, int(0.01 * len(mesh.faces)))
            keep = [c for c in components if len(c.faces) >= min_faces]
            if keep and len(keep) < len(components):
                dropped = len(components) - len(keep)
                print(f"Quad wrap: dropped {dropped} debris component(s) "
                      f"(< {min_faces} faces) after decimation")
                mesh = trimesh.util.concatenate(keep)

        # Remove extra faces on non-manifold edges (incidence > 2), keeping
        # the two largest faces, then close any holes this opened.
        for _ in range(3):
            edges = np.sort(mesh.faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
            edge_faces = {}
            for fi, e in zip(np.repeat(np.arange(len(mesh.faces)), 3), map(tuple, edges)):
                edge_faces.setdefault(e, []).append(fi)
            bad = {e: fs for e, fs in edge_faces.items() if len(fs) > 2}
            if not bad:
                break
            areas = mesh.area_faces
            remove = set()
            for e, fs in bad.items():
                fs_sorted = sorted(fs, key=lambda fi: areas[fi], reverse=True)
                remove.update(fs_sorted[2:])
            mask = np.ones(len(mesh.faces), dtype=bool)
            mask[list(remove)] = False
            mesh.update_faces(mask)
            trimesh.repair.fill_holes(mesh)
            mesh.merge_vertices()
            mesh.remove_unreferenced_vertices()

        return mesh
        
    def _extract_pure_quads(self, V: np.ndarray, F: np.ndarray, field: np.ndarray) -> Tuple[np.ndarray, List[List[int]]]:
        """Extracts pure quads from the cross-field and ensures watertightness."""
        from collections import defaultdict
        edge_to_faces = defaultdict(list)
        for i, face in enumerate(F):
            for j in range(3):
                edge = tuple(sorted((face[j], face[(j+1)%3])))
                edge_to_faces[edge].append(i)
                
        adj_faces = []
        for edge, faces in edge_to_faces.items():
            if len(faces) == 2:
                adj_faces.append((faces[0], faces[1], edge))
                
        weights = []
        face_normals = []
        for face in F:
            v0, v1, v2 = V[face[0]], V[face[1]], V[face[2]]
            n = np.cross(v1 - v0, v2 - v0)
            norm = np.linalg.norm(n)
            face_normals.append(n / norm if norm > 1e-10 else np.array([0., 0., 1.]))
        face_normals = np.array(face_normals)
        
        for f1, f2, edge in adj_faces:
            n1 = face_normals[f1]
            n2 = face_normals[f2]
            weights.append(np.dot(n1, n2))
            
        adj_faces = [x for _, x in sorted(zip(weights, adj_faces), key=lambda pair: pair[0], reverse=True)]
        
        merged = set()
        quads = []
        
        def is_convex(q_verts):
            v0, v1, v2, v3 = q_verts
            normal = np.cross(v2 - v0, v3 - v1)
            if np.linalg.norm(normal) < 1e-10:
                return False
            normal = normal / np.linalg.norm(normal)
            edges = [v1 - v0, v2 - v1, v3 - v2, v0 - v3]
            signs = []
            for i in range(4):
                cross = np.cross(edges[i], edges[(i+1)%4])
                dot = np.dot(cross, normal)
                if abs(dot) > 1e-8:
                    signs.append(np.sign(dot))
            if len(signs) < 3:
                return False
            return len(set(signs)) == 1

        for f1, f2, edge in adj_faces:
            if f1 not in merged and f2 not in merged:
                if f1 == f2:
                    continue
                face1_verts = list(F[f1])
                face2_verts = list(F[f2])
                opp1 = [v for v in face1_verts if v not in edge]
                opp2 = [v for v in face2_verts if v not in edge]
                if len(opp1) != 1 or len(opp2) != 1:
                    continue  # degenerate facet (repeated vertex index)
                v_f1_opp = opp1[0]
                v_f2_opp = opp2[0]
                
                idx_e0 = face1_verts.index(edge[0])
                if face1_verts[(idx_e0 + 1) % 3] == edge[1]:
                    prop_quad = [v_f1_opp, edge[0], v_f2_opp, edge[1]]
                else:
                    prop_quad = [v_f1_opp, edge[1], v_f2_opp, edge[0]]
                    
                q_positions = [V[vi] for vi in prop_quad]
                if is_convex(q_positions):
                    merged.add(f1)
                    merged.add(f2)
                    quads.append(prop_quad)
                    
        tris = []
        for i, face in enumerate(F):
            if i not in merged:
                tris.append(list(face))
                
        # Subdivide mixed mesh to guarantee pure quads
        new_V = list(V)
        new_quads = []
        edge_midpoints = {}
        
        def get_edge_midpoint(v1, v2):
            e = tuple(sorted((v1, v2)))
            if e not in edge_midpoints:
                idx = len(new_V)
                new_V.append((V[v1] + V[v2]) / 2.0)
                edge_midpoints[e] = idx
            return edge_midpoints[e]
            
        for q in quads:
            center_idx = len(new_V)
            new_V.append(np.mean([V[vi] for vi in q], axis=0))
            e0 = get_edge_midpoint(q[0], q[1])
            e1 = get_edge_midpoint(q[1], q[2])
            e2 = get_edge_midpoint(q[2], q[3])
            e3 = get_edge_midpoint(q[3], q[0])
            new_quads.extend([
                [q[0], e0, center_idx, e3],
                [q[1], e1, center_idx, e0],
                [q[2], e2, center_idx, e1],
                [q[3], e3, center_idx, e2]
            ])
            
        for t in tris:
            center_idx = len(new_V)
            new_V.append(np.mean([V[vi] for vi in t], axis=0))
            e0 = get_edge_midpoint(t[0], t[1])
            e1 = get_edge_midpoint(t[1], t[2])
            e2 = get_edge_midpoint(t[2], t[0])
            new_quads.extend([
                [t[0], e0, center_idx, e2],
                [t[1], e1, center_idx, e0],
                [t[2], e2, center_idx, e1]
            ])
            
        return np.array(new_V), new_quads
        
    def _relax_mesh(self, mesh: HalfEdgeMesh, reference: HalfEdgeMesh) -> HalfEdgeMesh:
        """Laplacian smoothing & Shrinkwrap onto reference mesh."""
        from src.reverse_engineering.shrink_wrap import ShrinkWrapper
        # subdivision_levels=0 explicitly: the cage produced by
        # _extract_pure_quads is already at the requested density, subdividing
        # it here would multiply the quad count again.
        wrapper = ShrinkWrapper(iterations=3, subdivision_levels=0,
                                smooth_weight=self.smoothing_weight,
                                projection_mode='ray_cast')
        return wrapper.wrap(mesh, reference)

    @staticmethod
    def _concave_quad_ids(mesh: HalfEdgeMesh) -> List[int]:
        """Quads that are concave OR degenerate (matches the renderer's needs:
        a valid quad has a well-defined diagonal normal and at least three
        corner turns of a single consistent sign)."""
        bad = []
        for f in mesh.faces:
            vs = [v.position for v in mesh.get_face_vertices(f)]
            if len(vs) != 4:
                continue
            v0, v1, v2, v3 = vs
            normal = np.cross(v2 - v0, v3 - v1)
            nl = np.linalg.norm(normal)
            if nl < 1e-10:
                bad.append(f.index)
                continue
            normal /= nl
            edges = [v1 - v0, v2 - v1, v3 - v2, v0 - v3]
            signs = []
            for i in range(4):
                d = np.dot(np.cross(edges[i], edges[(i + 1) % 4]), normal)
                if abs(d) > 1e-8:
                    signs.append(np.sign(d))
            if len(signs) < 3 or len(set(signs)) > 1:
                bad.append(f.index)
        return bad

    def _repair_concave_quads(self, mesh: HalfEdgeMesh, reference: HalfEdgeMesh,
                              fallback_positions: np.ndarray,
                              max_iterations: int = 12) -> None:
        """Untangle concave/degenerate quads produced by relaxation.

        Relaxation itself is smooth-then-project, so repeating that operator
        cannot untangle its own folds. Instead, blend offending vertices back
        towards their pre-relaxation positions (which formed convex quads),
        widening the blend each round; the final rounds revert exactly, which
        is guaranteed to restore convexity once every vertex of a bad quad is
        reverted.
        """
        ref_tm = reference.to_trimesh()
        for it in range(max_iterations):
            bad = self._concave_quad_ids(mesh)
            if not bad:
                return
            alpha = min(1.0, 0.4 + 0.2 * it)
            project = alpha < 1.0  # final rounds: exact revert, no re-projection
            vert_ids = sorted({
                v.index
                for fi in bad
                for v in mesh.get_face_vertices(mesh.faces[fi])
            })
            targets = []
            for vi in vert_ids:
                cur = mesh.vertices[vi].position
                targets.append(cur + alpha * (fallback_positions[vi] - cur))
            targets = np.array(targets)
            if project:
                targets, _, _ = trimesh.proximity.closest_point(ref_tm, targets)
            for vi, p in zip(vert_ids, targets):
                mesh.vertices[vi].position = np.asarray(p, dtype=np.float64)
        left = len(self._concave_quad_ids(mesh))
        if left:
            print(f"Quad wrap: {left} concave quad(s) could not be untangled")

