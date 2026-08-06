import trimesh
# Create a sphere with 1000 faces
mesh = trimesh.creation.icosphere(subdivisions=3)
print(f"Original faces: {len(mesh.faces)}")

# Try with 0.9
m1 = mesh.simplify_quadric_decimation(0.9)
print(f"Decimated 0.9 faces: {len(m1.faces) if hasattr(m1, 'faces') else 'Error'}")

# Try with 100
m2 = mesh.simplify_quadric_decimation(100)
print(f"Decimated 100 faces: {len(m2.faces) if hasattr(m2, 'faces') else 'Error'}")
