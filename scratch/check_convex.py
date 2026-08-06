import sys
import os
import trimesh
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.core.halfedge_mesh import HalfEdgeMesh
from src.reverse_engineering.quad_wrap import QuadWrapper

stl_path = r"E:\foxcore_data\_MITEB\Miteb_Flaechenrueckfuehrung\7_LLzugdruck_maxstress_smooth_iso0p3_inv.STL"
mesh = trimesh.load(stl_path)

wrapper = QuadWrapper(target_face_count=5000, smoothing_weight=0.1)
dec_V, dec_F, dec_field = wrapper._anisotropic_decimate(mesh, None, 5000)

quads, tris = wrapper._tri_to_quad(dec_V, dec_F, dec_field)

def is_convex(quad, V):
    v0 = V[quad[0]]
    v1 = V[quad[1]]
    v2 = V[quad[2]]
    v3 = V[quad[3]]
    
    # Check angles between adjacent edges
    # We project the quad to its local 2D plane
    normal = np.cross(v2 - v0, v3 - v1)
    if np.linalg.norm(normal) < 1e-8:
        return False # Degenerate
    normal = normal / np.linalg.norm(normal)
    
    # Calculate signs of cross products of adjacent edges
    edges = [v1 - v0, v2 - v1, v3 - v2, v0 - v3]
    signs = []
    for i in range(4):
        cross = np.cross(edges[i], edges[(i+1)%4])
        dot = np.dot(cross, normal)
        signs.append(np.sign(dot))
        
    return len(set([s for s in signs if s != 0])) <= 1

non_convex_count = 0
for q in quads:
    if not is_convex(q, dec_V):
        non_convex_count += 1

print(f"Total quads: {len(quads)}")
print(f"Non-convex quads: {non_convex_count}")
