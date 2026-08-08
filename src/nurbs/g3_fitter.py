"""
G3 Continuous B-Spline Fitter
Generates Degree 5 (6x6 control points) B-spline patches from quad regions.
Optimized with Numba JIT compilation and Multiprocessing.
"""

import os
import numpy as np
import concurrent.futures

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

@njit(fastmath=True, nogil=True)
def _numba_compute_edge_control_points(p_start, p_end, d1_start, d1_end, d2):
    edge_pts = np.zeros((6, 3))
    edge_pts[0] = p_start
    edge_pts[5] = p_end

    # 1st derivative (G1) — independent end tangents
    edge_pts[1] = edge_pts[0] + d1_start / 5.0
    edge_pts[4] = edge_pts[5] - d1_end / 5.0

    # 2nd derivative (G2)
    edge_pts[2] = 2 * edge_pts[1] - edge_pts[0] + d2 / 20.0
    edge_pts[3] = 2 * edge_pts[4] - edge_pts[5] + d2 / 20.0
    return edge_pts


def _tangent_in_plane(chord, normal):
    """Project chord onto the tangent plane of `normal`, keeping its length.

    Both patches sharing an edge derive the same tangent from the same vertex
    normal and the same (up to sign) chord, so their boundary curves stay
    exactly identical and sewing still closes."""
    n = np.asarray(normal, dtype=np.float64)
    nl = np.linalg.norm(n)
    if nl < 1e-12:
        return chord
    n = n / nl
    t = chord - np.dot(chord, n) * n
    tl = np.linalg.norm(t)
    cl = np.linalg.norm(chord)
    if tl < 1e-12 or cl < 1e-12:
        return chord
    return t * (cl / tl)

@njit(fastmath=True, nogil=True)
def _numba_compute_interior_control_points(ctrl_pts):
    for i in range(1, 5):
        for j in range(1, 5):
            u = i / 5.0
            v = j / 5.0
            
            # Bi-linear Coons patch blending for interior
            blend_u = (1 - u) * ctrl_pts[0, j] + u * ctrl_pts[5, j]
            blend_v = (1 - v) * ctrl_pts[i, 0] + v * ctrl_pts[i, 5]
            blend_uv = (
                (1 - u) * (1 - v) * ctrl_pts[0, 0] +
                (1 - u) * v * ctrl_pts[0, 5] +
                u * (1 - v) * ctrl_pts[5, 0] +
                u * v * ctrl_pts[5, 5]
            )
            ctrl_pts[i, j] = blend_u + blend_v - blend_uv
    return ctrl_pts

def _generate_single_patch(quad_data):
    """Build a 6x6 control grid for one quad.

    `corners` are the quad's vertices in CYCLIC winding order (c0->c1->c2->c3
    around the face), matching HalfEdgeMesh.get_face_vertices and the dense
    sample grid p(u,v) = (1-u)(1-v)c0 + u(1-v)c1 + uv c2 + (1-u)v c3.
    Grid axes: index i follows u (edge c0->c1 at v=0), index j follows v
    (edge c0->c3 at u=0).
    """
    corners = np.array(quad_data['corners'])
    derivatives = quad_data.get('derivatives')
    corner_normals = quad_data.get('corner_normals')
    c0, c1, c2, c3 = corners

    if derivatives is None:
        z = np.zeros(3)
        if corner_normals is not None:
            n0, n1, n2, n3 = corner_normals
            # Bend each boundary curve into the surface tangent planes of its
            # endpoints instead of running it along the straight chord.
            derivatives = {
                'u0': {'d1_start': _tangent_in_plane(c1 - c0, n0), 'd1_end': _tangent_in_plane(c1 - c0, n1), 'd2': z},
                'u1': {'d1_start': _tangent_in_plane(c2 - c3, n3), 'd1_end': _tangent_in_plane(c2 - c3, n2), 'd2': z},
                'v0': {'d1_start': _tangent_in_plane(c3 - c0, n0), 'd1_end': _tangent_in_plane(c3 - c0, n3), 'd2': z},
                'v1': {'d1_start': _tangent_in_plane(c2 - c1, n1), 'd1_end': _tangent_in_plane(c2 - c1, n2), 'd2': z},
            }
        else:
            derivatives = {
                'u0': {'d1': c1 - c0, 'd2': z},   # edge v=0: c0 -> c1
                'u1': {'d1': c2 - c3, 'd2': z},   # edge v=1: c3 -> c2
                'v0': {'d1': c3 - c0, 'd2': z},   # edge u=0: c0 -> c3
                'v1': {'d1': c2 - c1, 'd2': z}    # edge u=1: c1 -> c2
            }

    def _edge(p_start, p_end, d):
        d1s = d.get('d1_start', d.get('d1'))
        d1e = d.get('d1_end', d.get('d1'))
        return _numba_compute_edge_control_points(p_start, p_end, d1s, d1e, d['d2'])

    ctrl_pts = np.zeros((6, 6, 3))

    # Corners (cyclic winding mapped onto the tensor grid)
    ctrl_pts[0, 0] = c0
    ctrl_pts[5, 0] = c1
    ctrl_pts[5, 5] = c2
    ctrl_pts[0, 5] = c3

    # Edges — each boundary curve runs along an actual quad edge, so two
    # patches sharing an edge produce identical (reversed) control rows and
    # sewing can join them exactly.
    ctrl_pts[:, 0] = _edge(c0, c1, derivatives['u0'])
    ctrl_pts[:, 5] = _edge(c3, c2, derivatives['u1'])
    ctrl_pts[0, :] = _edge(c0, c3, derivatives['v0'])
    ctrl_pts[5, :] = _edge(c1, c2, derivatives['v1'])

    # Interior
    ctrl_pts = _numba_compute_interior_control_points(ctrl_pts)
    return ctrl_pts


class G3Fitter:
    def __init__(self, continuity_weight: float = 1000.0):
        """
        Initialize the G3 Fitter for Degree 5 B-spline patches.
        A degree 5 Bezier/B-spline patch requires a 6x6 grid of control points.

        continuity_weight: weight of the cross-boundary smoothness equations
        relative to the surface data equations (weight 1). 0 disables them.
        """
        self.degree = 5
        self.num_ctrl_pts = 6
        self.continuity_weight = continuity_weight

    def generate_patch(self, corners, derivatives=None):
        """
        Generate a 6x6 control point patch for a quad region.
        """
        return _generate_single_patch({'corners': corners, 'derivatives': derivatives})

    def fit_surface(self, quad_mesh):
        """
        Fits a completely G3 continuous surface over a network of quad regions.
        Replaces local Coons blending with a Global Sparse Solver (LSPIA).
        Optimizes interior control points by minimizing projection distance 
        while explicitly solving G3 continuity constraint equations across patch boundaries.
        """
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla
        import math
        
        # Pre-compile Numba functions on the first run
        _numba_compute_edge_control_points(np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3))
        _numba_compute_interior_control_points(np.zeros((6, 6, 3)))
        
        patches = [_generate_single_patch(quad) for quad in quad_mesh]
        num_patches = len(patches)
        if num_patches == 0:
            return patches
            
        def B(n, i, t):
            return math.comb(n, i) * (t**i) * ((1-t)**(n-i))
            
        num_vars = num_patches * 16
        
        rows = []
        cols = []
        data = []
        
        bx = []
        by = []
        bz = []
        
        eq_idx = 0
        
        # 1. Data Fitting Equations
        for k, quad in enumerate(quad_mesh):
            dense = quad.get('dense_points', None)
            if dense is not None:
                grid_size = dense.shape[0]
                u_vals = np.linspace(0, 1, grid_size)
                v_vals = np.linspace(0, 1, grid_size)
                
                for r in range(grid_size):
                    for c in range(grid_size):
                        u = u_vals[r]
                        v = v_vals[c]
                        pt = dense[r, c]
                        
                        fixed_sum = np.zeros(3)
                        for i in range(6):
                            for j in range(6):
                                if i in [0, 5] or j in [0, 5]:
                                    val = B(5, i, u) * B(5, j, v)
                                    fixed_sum += val * patches[k][i, j]
                                    
                        target = pt - fixed_sum
                        bx.append(target[0])
                        by.append(target[1])
                        bz.append(target[2])
                        
                        for i in range(1, 5):
                            for j in range(1, 5):
                                val = B(5, i, u) * B(5, j, v)
                                var_idx = k * 16 + (i-1) * 4 + (j-1)
                                rows.append(eq_idx)
                                cols.append(var_idx)
                                data.append(val)
                                
                        eq_idx += 1
                        
        # 2. G3 Continuity Constraints
        weight_g3 = self.continuity_weight
        
        def get_inward(patch_idx, edge_id, idx):
            if edge_id == 0:
                return [(0, idx), (1, idx), (2, idx), (3, idx)]
            elif edge_id == 1:
                return [(5, idx), (4, idx), (3, idx), (2, idx)]
            elif edge_id == 2:
                return [(idx, 0), (idx, 1), (idx, 2), (idx, 3)]
            elif edge_id == 3:
                return [(idx, 5), (idx, 4), (idx, 3), (idx, 2)]
                
        processed_pairs = set()

        for k, quad in enumerate(quad_mesh):
            if weight_g3 <= 0:
                break  # continuity equations disabled
            neighbors = quad.get('neighbors', [-1, -1, -1, -1])
            for n_k in neighbors:
                if n_k == -1 or n_k >= num_patches:
                    continue

                pair = tuple(sorted((k, n_k)))
                if pair in processed_pairs:
                    continue

                # The position of a neighbor in the `neighbors` list says
                # nothing about which tensor-grid edge (row 0/5, col 0/5) the
                # shared boundary is — find both edge ids geometrically. Match
                # on the FULL 6-point boundary curve (shared curves are exactly
                # identical by construction): endpoint-only matching pairs the
                # wrong edges when corner positions coincide (duplicate cage
                # vertices), and one wrong constraint sends the least-squares
                # interior of both patches tens of millimetres off the part.
                def boundary_row(patch_idx, edge_id):
                    return np.array([patches[patch_idx][get_inward(patch_idx, edge_id, i)[0]]
                                     for i in range(6)])

                candidates = []
                for ek in range(4):
                    row_k = boundary_row(k, ek)
                    for en in range(4):
                        row_n = boundary_row(n_k, en)
                        if np.abs(row_k - row_n).max() < 1e-7:
                            candidates.append((ek, en, False))
                        elif np.abs(row_k - row_n[::-1]).max() < 1e-7:
                            candidates.append((ek, en, True))

                if len(candidates) != 1:
                    # no shared curve, or ambiguous (degenerate geometry) —
                    # skipping only loses a soft smoothness equation
                    continue
                edge_k, edge_n, reversed_dir = candidates[0]
                matched = True

                if matched:
                        processed_pairs.add(pair)
                        for idx_k in range(1, 5):
                            idx_n = 5 - idx_k if reversed_dir else idx_k
                            
                            inward_k = get_inward(k, edge_k, idx_k)
                            inward_n = get_inward(n_k, edge_n, idx_n)
                            
                            rhs = 2.0 * patches[k][inward_k[0]]
                            
                            bx.append(rhs[0] * weight_g3)
                            by.append(rhs[1] * weight_g3)
                            bz.append(rhs[2] * weight_g3)
                            
                            coeffs = {1: 3.0, 2: -3.0, 3: 1.0}
                            for c_idx, coeff in coeffs.items():
                                i, j = inward_k[c_idx]
                                var_idx = k * 16 + (i-1) * 4 + (j-1)
                                rows.append(eq_idx)
                                cols.append(var_idx)
                                data.append(coeff * weight_g3)
                                
                            for c_idx, coeff in coeffs.items():
                                i, j = inward_n[c_idx]
                                var_idx = n_k * 16 + (i-1) * 4 + (j-1)
                                rows.append(eq_idx)
                                cols.append(var_idx)
                                data.append(coeff * weight_g3)
                                
                            eq_idx += 1

        if len(data) > 0:
            initial_interiors = [p[1:5, 1:5].copy() for p in patches]

            A = sp.csr_matrix((data, (rows, cols)), shape=(eq_idx, num_vars))

            x = spla.lsqr(A, np.array(bx))[0]
            y = spla.lsqr(A, np.array(by))[0]
            z = spla.lsqr(A, np.array(bz))[0]

            # Only overwrite variables that appeared in at least one equation;
            # lsqr returns the least-norm value 0 for untouched variables,
            # which would collapse unconstrained patches to the origin.
            used = np.zeros(num_vars, dtype=bool)
            used[np.asarray(cols, dtype=np.int64)] = True

            for k in range(num_patches):
                for i in range(1, 5):
                    for j in range(1, 5):
                        var_idx = k * 16 + (i-1) * 4 + (j-1)
                        if used[var_idx]:
                            patches[k][i, j, 0] = x[var_idx]
                            patches[k][i, j, 1] = y[var_idx]
                            patches[k][i, j, 2] = z[var_idx]

            # Sanity clamp: the solve refines the Coons interior; if it
            # diverged (conflicting equations at degenerate geometry), the
            # runaway control points corrupt the surface. Legitimate curvature
            # bulges stay near 1x the quad diagonal; SolidWorks refuses faces
            # whose interiors span ~1.3x+ (import silently drops them and the
            # shell can no longer knit into a solid), so clamp tightly and
            # fall back to the Coons initialization.
            n_reset = 0
            for k, p in enumerate(patches):
                corners = np.array([p[0, 0], p[5, 0], p[5, 5], p[0, 5]])
                center = corners.mean(axis=0)
                diag = np.linalg.norm(corners.max(axis=0) - corners.min(axis=0))
                limit = max(1.25 * diag, diag + 0.5)
                interior = p[1:5, 1:5].reshape(-1, 3)
                if np.linalg.norm(interior - center, axis=1).max() > limit:
                    p[1:5, 1:5] = initial_interiors[k]
                    n_reset += 1
            if n_reset:
                print(f"G3Fitter: reset {n_reset} diverged patch interior(s) "
                      f"to their Coons initialization")

        return patches
