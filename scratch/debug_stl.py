import sys
import os
import trimesh

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.core.halfedge_mesh import HalfEdgeMesh
from src.reverse_engineering.quad_wrap import QuadWrapper
from src.reverse_engineering.mesh_tools import compute_mesh_quality

stl_path = r"E:\foxcore_data\_MITEB\Miteb_Flaechenrueckfuehrung\7_LLzugdruck_maxstress_smooth_iso0p3_inv.STL"
print(f"Loading {stl_path}...")
mesh = trimesh.load(stl_path)
print(f"Original STL faces: {len(mesh.faces)}, watertight: {mesh.is_watertight}")

# Convert to HalfEdgeMesh
he_mesh = HalfEdgeMesh.from_trimesh(mesh)
stats = compute_mesh_quality(he_mesh)
print(f"Original HalfEdge stats: {stats}")

wrapper = QuadWrapper(target_face_count=5000, smoothing_weight=0.1)

# Step 1: Anisotropic decimation
dec_V, dec_F, dec_field = wrapper._anisotropic_decimate(mesh, None, 5000)
print(f"Decimated mesh faces: {len(dec_F)}")

he_dec = HalfEdgeMesh()
for v in dec_V:
    he_dec.add_vertex(v)
for f in dec_F:
    he_dec.add_face(f)

stats_dec = compute_mesh_quality(he_dec)
print(f"Decimated HalfEdge stats: {stats_dec}")

# Step 2: Tri-to-quad
dec_mesh = trimesh.Trimesh(vertices=dec_V, faces=dec_F, process=True)
dec_mesh.fill_holes()
dec_V = dec_mesh.vertices
dec_F = dec_mesh.faces
import numpy as np
field = np.zeros((len(dec_V), 3))
quads, tris = wrapper._tri_to_quad(dec_V, dec_F, field)
print(f"Tri-to-quad produced {len(quads)} quads and {len(tris)} tris")

he_quad = HalfEdgeMesh()
for v in dec_mesh.vertices:
    he_quad.add_vertex(v)
for q in quads:
    he_quad.add_face(q)
for t in tris:
    he_quad.add_face(t)

stats_quad = compute_mesh_quality(he_quad)
print(f"Quad HalfEdge stats: {stats_quad}")

# find the boundary edge
for edge in he_quad.edges:
    if he_quad.is_boundary_edge(edge):
        print(f"Boundary Edge: {edge.index}")
        he = edge.half_edge
        print(f"  v1: {he.prev.vertex.position}")
        print(f"  v2: {he.vertex.position}")
        print(f"  Face: {[v.index for v in he_quad.get_face_vertices(he.face)]}")
        if he.twin:
            print(f"  Twin Face: {he.twin.face}")
        else:
            print("  No twin")

