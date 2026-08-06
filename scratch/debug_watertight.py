import trimesh
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.subd.primitives import create_sphere
from src.reverse_engineering.quad_wrap import QuadWrapper

sphere = create_sphere(radius=5.0, rings=8, segments=8)
wrapper = QuadWrapper(target_face_count=20, smoothing_weight=0.1)
quad_mesh = wrapper.wrap(sphere)

t_mesh = quad_mesh.to_trimesh()
print(f"Trimesh watertight: {t_mesh.is_watertight}")
print(f"Trimesh winding consistent: {t_mesh.is_winding_consistent}")
print(f"Trimesh volume: {t_mesh.volume}")
print(f"Trimesh edges count: {len(t_mesh.edges)}")
print(f"Trimesh edges_unique count: {len(t_mesh.edges_unique)}")
print(f"Faces sharing edges_unique: {t_mesh.edges_unique_length}")

for f in quad_mesh.faces:
    fv = [v.index for v in quad_mesh.get_face_vertices(f)]
    print(fv)
    assert len(fv) == len(set(fv)), f"Duplicate vertices in face! {fv}"
