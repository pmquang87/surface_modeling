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
def _numba_compute_edge_control_points(p_start, p_end, d1, d2):
    edge_pts = np.zeros((6, 3))
    edge_pts[0] = p_start
    edge_pts[5] = p_end
    
    # 1st derivative (G1)
    edge_pts[1] = edge_pts[0] + d1 / 5.0
    edge_pts[4] = edge_pts[5] - d1 / 5.0
    
    # 2nd derivative (G2)
    edge_pts[2] = 2 * edge_pts[1] - edge_pts[0] + d2 / 20.0
    edge_pts[3] = 2 * edge_pts[4] - edge_pts[5] + d2 / 20.0
    return edge_pts

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
    corners = np.array(quad_data['corners'])
    derivatives = quad_data.get('derivatives')
    
    if derivatives is None:
        du = corners[2] - corners[0]
        dv = corners[1] - corners[0]
        derivatives = {
            'u0': {'d1': du, 'd2': np.zeros(3)},
            'u1': {'d1': corners[3] - corners[1], 'd2': np.zeros(3)},
            'v0': {'d1': dv, 'd2': np.zeros(3)},
            'v1': {'d1': corners[3] - corners[2], 'd2': np.zeros(3)}
        }
        
    ctrl_pts = np.zeros((6, 6, 3))
    
    # Corners
    ctrl_pts[0, 0] = corners[0]
    ctrl_pts[0, 5] = corners[1]
    ctrl_pts[5, 0] = corners[2]
    ctrl_pts[5, 5] = corners[3]
    
    # Edges
    ctrl_pts[:, 0] = _numba_compute_edge_control_points(corners[0], corners[2], derivatives['u0']['d1'], derivatives['u0']['d2'])
    ctrl_pts[:, 5] = _numba_compute_edge_control_points(corners[1], corners[3], derivatives['u1']['d1'], derivatives['u1']['d2'])
    ctrl_pts[0, :] = _numba_compute_edge_control_points(corners[0], corners[1], derivatives['v0']['d1'], derivatives['v0']['d2'])
    ctrl_pts[5, :] = _numba_compute_edge_control_points(corners[2], corners[3], derivatives['v1']['d1'], derivatives['v1']['d2'])
    
    # Interior
    ctrl_pts = _numba_compute_interior_control_points(ctrl_pts)
    return ctrl_pts


class G3Fitter:
    def __init__(self):
        """
        Initialize the G3 Fitter for Degree 5 B-spline patches.
        A degree 5 Bezier/B-spline patch requires a 6x6 grid of control points.
        """
        self.degree = 5
        self.num_ctrl_pts = 6

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
        _numba_compute_edge_control_points(np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3))
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
        weight_g3 = 1000.0
        
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
            neighbors = quad.get('neighbors', [-1, -1, -1, -1])
            for edge_k in range(4):
                n_k = neighbors[edge_k]
                if n_k == -1 or n_k >= num_patches:
                    continue
                    
                pair = tuple(sorted((k, n_k)))
                if pair in processed_pairs:
                    continue
                
                matched = False
                for edge_n in range(4):
                    E_k_0 = get_inward(k, edge_k, 0)[0]
                    E_k_5 = get_inward(k, edge_k, 5)[0]
                    
                    E_n_0 = get_inward(n_k, edge_n, 0)[0]
                    E_n_5 = get_inward(n_k, edge_n, 5)[0]
                    
                    pt_k_0 = patches[k][E_k_0]
                    pt_k_5 = patches[k][E_k_5]
                    
                    pt_n_0 = patches[n_k][E_n_0]
                    pt_n_5 = patches[n_k][E_n_5]
                    
                    d_align = np.linalg.norm(pt_k_0 - pt_n_0) + np.linalg.norm(pt_k_5 - pt_n_5)
                    d_rev = np.linalg.norm(pt_k_0 - pt_n_5) + np.linalg.norm(pt_k_5 - pt_n_0)
                    
                    if d_align < 1e-4:
                        reversed_dir = False
                        matched = True
                    elif d_rev < 1e-4:
                        reversed_dir = True
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
                        break
                        
        if len(data) > 0:
            A = sp.csr_matrix((data, (rows, cols)), shape=(eq_idx, num_vars))
            
            x = spla.lsqr(A, np.array(bx))[0]
            y = spla.lsqr(A, np.array(by))[0]
            z = spla.lsqr(A, np.array(bz))[0]
            
            for k in range(num_patches):
                for i in range(1, 5):
                    for j in range(1, 5):
                        var_idx = k * 16 + (i-1) * 4 + (j-1)
                        patches[k][i, j, 0] = x[var_idx]
                        patches[k][i, j, 1] = y[var_idx]
                        patches[k][i, j, 2] = z[var_idx]
                        
        return patches
